import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from contextlib import contextmanager
from io import BytesIO, StringIO
import json
import os
from pathlib import Path
from queue import Empty, Queue
import selectors
import shlex
from subprocess import Popen, PIPE, STDOUT, run as sp_run
from tempfile import TemporaryDirectory
import threading
import time
import traceback
import logging
from typing import IO, Callable, ContextManager, Dict, Generic, Iterator, List, Literal, Optional, Tuple, TypeVar, TypedDict, cast
from e2b import Sandbox
from logging import Logger
from datasets import load_dataset
from fastapi import FastAPI
from fastapi.templating import Jinja2Templates
import requests
from commands import OpenInterpreterCommand
from constants import LOCAL
from runners import docker_daemon, get_free_port, talk_to_oi_server
from swe_bench import SWEBenchTask
from coordinators import TaskLifecycle, WebSocketsManager
from commands import commands

root_logger = logging.getLogger()
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.INFO)
console_handler.setFormatter(formatter)
root_logger.addHandler(console_handler)


module_logger = root_logger.getChild(__name__)
module_logger.setLevel(logging.DEBUG)


class SWEBenchPrediction(TypedDict):
    instance_id: str
    model_patch: str
    model_name_or_path: str


class ArgumentsNamespace(argparse.Namespace):
    command: str
    ntasks: Optional[int]
    nworkers: Optional[int]
    split: str


def dexec_cmd(container_id: str, exec_args: List[str], cmd: List[str]) -> List[str]:
    """
    Executes the given `cmd` in the container at `container_id`, passing `exec_args` in as command-line
    arguments to the `exec` call, and piping stdout and stderr to the docker desktop logger.
    
    Assumes anything else being logged to Docker Desktop is trash.

    Args:
        - container_id: the container identifier as printed to stdout when `docker run -d ...` was originally run.
        - exec_args: a list of arguments to pass into the call to `docker exec...`.
        - cmd: the command to be run on the container as a list of tokens.
    
    Returns:
        The full command to run on the host machine to invoke `cmd` in the container.  Does not run in the container's shell.
    """
    return ["docker", "exec", "-t", *exec_args, container_id, *cmd]


logs_q: Queue[Tuple[logging.Logger, Popen[bytes]]] = Queue()
def _process_background_logs():
    """
    This function manages the thread that enables non-blocking subprocess logging.  It's very silly and is the reason this
    script doesn't work as described on Windows.
    """

    # the data being registered in each instance is the tuple (Logger, Popen).
    sel = selectors.DefaultSelector()

    while True:
        # Either:
        # a) add another selector based off the queue, or
        # b) read from selectors.
        try:
            lp = logs_q.get_nowait()
        except Empty:
            lp = None

        if lp is not None:
            logger, p = lp
            if p.stdout is not None:
                sel.register(p.stdout, selectors.EVENT_READ, (logger, p))
            if p.stderr is not None:
                sel.register(p.stderr, selectors.EVENT_READ, (logger, p))
        else:
            events = sel.select(timeout=0)
            for key, _ in events:
                logger, p = cast(Tuple[logging.Logger, Popen[bytes]], key.data)
                stream = cast(IO[bytes], key.fileobj)
                logger.debug(stream.readline().decode().rstrip())

                if p.poll() is not None:
                    if p.stdout is not None:
                        sel.unregister(p.stdout)
                    if p.stderr is not None:
                        sel.unregister(p.stderr)
                    logger.debug(stream.read().decode().rstrip())


def generate_lines_out(p: Popen[bytes]) -> Iterator[bytes]:
    """
    Takes in a Popen whose stdout spits out bytes.

    Args:
        - p: a Popen that spits out bytes.
    
    Returns:
        An iterator over the lines as they are written to `p`'s stdout.
    """

    if p.stdout is None:
        raise ValueError("Expected Popen to have a non-None standard out!")
    while p.poll() is None:
        yield p.stdout.readline().rstrip()
    yield p.stdout.read().rstrip()


logging_thread = threading.Thread(target=_process_background_logs)
logging_thread.daemon = True  # so the thread terminates on exit.
logging_thread.start()
def log_nonblocking(logger: Logger, p: Popen):
    logger.debug(f"NON-BLOCKING COMMAND: {p.args}")
    logs_q.put((logger, p))


def log_nonblocking_thread(logger: Logger, p: Popen[bytes]):
    def log(message: str):
        if len(message.strip()) > 0:
            logger.debug(f"NONBLOCKING THREAD: {message}")

    def run_logging_thread():
        time.sleep(0.1)  # booooooo why aren't the readlines blocking anymore??
        while p.poll() is not None:
            if p.stdout is not None:
                log(p.stdout.readline().decode())
            if p.stderr is not None:
                log(p.stderr.readline().decode())
        if p.stdout is not None:
            log(p.stdout.read().decode())
        if p.stderr is not None:
            log(p.stderr.read().decode())

    t = threading.Thread(target=run_logging_thread)
    t.daemon = True
    t.start()


def log_blocking(logger: Logger, p: Popen[bytes]):
    logger.debug(f"BLOCKING COMMAND: {p.args}")
    assert p.stdout is not None
    while p.poll() is None:
        logger.debug(p.stdout.readline().decode().strip())
    logger.debug(p.stdout.read().decode().strip())


def make_task_loggers(run_dir: Path, parent_logger: logging.Logger, task_id: str) -> Tuple[logging.Logger, logging.Logger]:
    """
    Creates folders to put logging stuff if they don't exist.

    Args:
        - run_dir: the path to the directory storing the current run's logs.
        - task_name: the name of the current task.

    Returns the tuple:
        0: Logger for container (DEBUG+).
        1: Logger for plaintext oi response (DEBUG+).
        Both are children of the module_logger.
    """
    base = run_dir / task_id
    base.mkdir(parents=True, exist_ok=True)
    common_logger = parent_logger

    # container_logger = common_logger.getChild("container")
    container_logger = common_logger.getChild("container")
    container_handler = logging.FileHandler(base / "container.log")
    container_handler.setLevel(logging.DEBUG)
    container_handler.setFormatter(formatter)
    container_logger.addHandler(container_handler)

    plain_logger = common_logger.getChild("plain")
    plain_file_handler = logging.FileHandler(base / "plain.log")
    plain_file_handler.setLevel(logging.DEBUG)
    plain_file_handler.terminator = ""
    plain_logger.addHandler(plain_file_handler)

    return container_logger, plain_logger


def make_container_loggers(run_dir: Path, parent_logger: logging.Logger, task_id: str) -> Tuple[logging.Logger, logging.Logger]:
    """
    Creates folders to put logging stuff if they don't exist.

    Args:
        - run_dir: the path to the directory storing the current run's logs.
        - task_name: the name of the current task.

    Returns the tuple:
        0: Logger for container (DEBUG+).
        1: Logger for plaintext oi response (DEBUG+).
        Both are children of the module_logger.
    """
    base = run_dir / task_id
    base.mkdir(parents=True, exist_ok=True)
    # common_logger = parent_logger
    common_logger = parent_logger.getChild(task_id)

    # container_logger = common_logger.getChild("container")
    # container_logger = common_logger.getChild("container")
    container_logger = common_logger.getChild("container")
    container_handler = logging.FileHandler(base / "container.log")
    container_handler.setLevel(logging.DEBUG)
    container_handler.setFormatter(formatter)
    container_logger.addHandler(container_handler)

    plain_logger = common_logger.getChild("plain")
    plain_file_handler = logging.FileHandler(base / "plain.log")
    plain_file_handler.setLevel(logging.DEBUG)
    plain_file_handler.terminator = ""
    plain_logger.addHandler(plain_file_handler)

    return container_logger, plain_logger


SWEBenchRunner = Callable[
    # The task
    # The command (OI object configuration)
    # The container-wide logger
    # The AI response logger
    [SWEBenchTask, OpenInterpreterCommand, logging.Logger, logging.Logger],
    SWEBenchPrediction | None
]


@contextmanager
def docker_process(image_name: str, container_name: str, docker_args: List[str], run_args: List[str] | None = None) -> Iterator[Popen[bytes]]:
    rargs = [] if run_args is None else run_args
    cmd = ["docker", "run", "--name", container_name, *docker_args, image_name, *rargs]
    p = Popen(cmd, stdout=PIPE, stderr=STDOUT)
    try:
        yield p
    finally:
        sp_run(["docker", "kill", container_name])


def run_swe_bench_task_on_oi_server(normal_address: str, websocket_address: str, task: SWEBenchTask, task_logger: Logger, ai_logger: Logger) -> str:
    """
    Returns the diff.
    """
    task_logger.debug(f"NORMAL ADDRESS: {normal_address}")
    task_logger.debug(f"WEBSOCKET ADDRESS: {websocket_address}")

    while True:
        try:
            hb_response = requests.get(f"{normal_address}/heartbeat")
            if hb_response.ok:
                task_logger.debug("connected!")
                task_logger.debug(f"HB RESPONSE: {hb_response.json()}")
                break
        except requests.exceptions.ConnectionError:
            task_logger.debug("waiting to connect...")
            time.sleep(1)

    repo = task["repo"].replace("/", "__")
    remote = f"https://github.com/swe-bench/{repo}.git"

    clone_response = requests.post(f"{normal_address}/run", json={"language": "shell", "code": f"git clone {remote} repo"})
    task_logger.debug(f"THE CLONE RESPONSE: {clone_response.json()}")

    cd_response = requests.post(f"{normal_address}/run", json={"language": "shell", "code": "cd repo"})
    task_logger.debug(f"THE CD RESPONSE: {cd_response.json()}")

    checkout_response = requests.post(f"{normal_address}/run", json={"language": "shell", "code": f"git checkout {task['base_commit']}"})
    task_logger.debug(f"THE CHECKOUT RESPONSE: {checkout_response.json()}")

    prompt = "\n".join([
        "Edit the repo in the cwd until you solve the following problem.  Actually edit the files you think will solve the problem.",
        f"PROBLEM: {task['problem_statement']}",
    ])
    talk_to_oi_server(websocket_address, prompt, lambda bs: ai_logger.debug(bs.decode()))

    diff_response = requests.post(f"{normal_address}/run", json={"language": "shell", "code": "git add --all && git diff --cached"})
    diff = cast(str, diff_response.json()["output"][0]["content"]).rstrip()
    task_logger.debug(f"THE DIFF RESPONSE: {diff}")
    return diff


def swe_bench_docker_runner_with_run_endpoint(task: SWEBenchTask, command: OpenInterpreterCommand, task_logger: Logger, ai_logger: Logger) -> SWEBenchPrediction | None:
    worker_name = "server-worker"
    container_name = f"{task['instance_id']}_{time.time()}"
    cwd = "/"
    port = get_free_port()
    address = f"127.0.0.1:{port}"

    command_json_str = json.dumps(command)
    with docker_process(
        worker_name, container_name,
        ["-t", "-p", f"{address}:8000"],
        ["python", "-m", "worker.run", command_json_str, cwd]
    ) as server_p:
        for line in generate_lines_out(server_p):
            task_logger.debug(line.decode())
            if "Uvicorn running" in line.decode():
                break
        log_nonblocking(task_logger, server_p)

        diff = run_swe_bench_task_on_oi_server(f"http://{address}", f"ws://{address}", task, task_logger, ai_logger)

    return {
        "instance_id": task["instance_id"],
        "model_patch": diff,
        "model_name_or_path": command["model"] if "model" in command else "<N/A>"
    }


def swe_bench_e2b_runner_with_run_endpoint(task: SWEBenchTask, command: OpenInterpreterCommand, task_logger: Logger, ai_logger: Logger) -> SWEBenchPrediction | None:
    command_json_str = json.dumps(command)
    cwd = Path("/")

    with Sandbox(template="server-worker") as sandbox:
        task_logger.debug(f"started sandbox: {sandbox.sandbox_id}")
        task_logger.debug(f"switching current working directory to: {cwd}")
        sandbox.commands.run(f"cd {cwd}")

        sandbox.files.make_dir("main")
        address = sandbox.get_host(port=8000)

        p = sandbox.commands.run(
            f"sudo python -m worker.run {shlex.quote(command_json_str)} {cwd / 'main'}",
            background=True,
            on_stdout=task_logger.debug,
            on_stderr=task_logger.debug)
        
        diff = run_swe_bench_task_on_oi_server(f"https://{address}", f"wss://{address}", task, task_logger, ai_logger)
        p.kill()
        sandbox.kill()
        
    return {
        "instance_id": task["instance_id"],
        "model_patch": diff,
        "model_name_or_path": command["model"] if "model" in command else "<N/A>"
    }


RunnerType = Literal["docker", "e2b"]
def swe_bench_runner_with_endpoint(rt: RunnerType) -> SWEBenchRunner:
    if rt == "docker":
        return swe_bench_docker_runner_with_run_endpoint
    elif rt == "e2b":
        return swe_bench_e2b_runner_with_run_endpoint


Task = TypeVar("Task")
class GeneralizedTask(TypedDict, Generic[Task]):
    id: str
    task: Task


if __name__ == "__main__":
    default_command_id = "gpt35turbo"
    default_split = "dev"
    default_runner_type: RunnerType = "e2b"

    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--command", action="store", type=str, default=default_command_id)
    parser.add_argument("-s", "--split", action="store", type=str, default=default_split)
    parser.add_argument("-nt", "--ntasks", action="store", type=int)
    parser.add_argument("-nw", "--nworkers", action="store", type=int)
    args = parser.parse_args(namespace=ArgumentsNamespace())

    if args.command not in commands:
        print(f"'{args.command}' not recognized as a command configuration id")
        exit(1)
    
    print("command configuration:", args.command)

    command_id = args.command
    command = commands[command_id]
    run_id = f"{time.time()}_swebench_{command_id}"
    max_total: int | None = args.ntasks
    dataset_path = "princeton-nlp/SWE-bench_Lite"
    split = args.split
    ds = cast(List[SWEBenchTask], list(load_dataset(dataset_path, split=split)))[:max_total]
    tasks = ds[:max_total or len(ds)]
    total = len(tasks)
    # just a list of patch strings.
    patches: List[str] = []

    # instance_id: { "id": ..., task: ... }
    tasks_map = {
        task["instance_id"]: {
            "id": task["instance_id"],
            "task": task,
        }
        for task in tasks
    }

    LOGS = LOCAL / "logs"
    CURRENT_RUN = LOGS / run_id
    CURRENT_RUN.mkdir(parents=True, exist_ok=True)

    run_file_handler = logging.FileHandler(CURRENT_RUN / "run.log")
    run_file_handler.setFormatter(formatter)
    module_logger.addHandler(run_file_handler)

    runner = swe_bench_runner_with_endpoint(default_runner_type)
    def run_task(task: SWEBenchTask, command: OpenInterpreterCommand) -> SWEBenchPrediction | None:
        task_logger, ai_logger = make_container_loggers(CURRENT_RUN, module_logger, task["instance_id"])
        try:
            result = runner(task, command, task_logger, ai_logger)
            return result
        except BaseException as e:
            task_logger.exception(traceback.format_exc())
            return None

    with ThreadPoolExecutor(max_workers=args.nworkers) as pool:
        n_finished_lock = threading.Lock()
        n_finished = 0
        def make_fs(id: str):
            def start():
                module_logger.info(f"task {id}: RUNNING...")
            def log(s: str):
                module_logger.info(f"task {id} log: {s}")
            def done(r: SWEBenchPrediction | None):
                global n_finished, n_finished_lock
                with n_finished_lock:
                    n_finished += 1
                module_logger.info(f"task {id}: DONE! ({n_finished}/{total})")
            return start, log, done

        run_task_args = [(task, command, make_fs(task["instance_id"])) for task in tasks]
        apps = []
        for args in run_task_args:
            tlc = TaskLifecycle[SWEBenchPrediction | None]()
            start, log, done = make_fs(args[0]["instance_id"])
            tlc.add_start_fn(start)
            tlc.add_done_fn(done)
            apps.append((tlc.wrap(run_task), (*args[:-1],)))
        
        module_logger.info(f"Running {total} tasks across {pool._max_workers} workers...")

        futures = [pool.submit(fn, *args) for fn, args in apps]
        
        for f in as_completed(futures):
            p = f.result()
            if p is not None:
                patches.append(p)
        
        module_logger.info("Finished!")

        save_path = CURRENT_RUN / "predictions.json"
        module_logger.info(f"Saving SWE-Bench prediction to {save_path}.")

        to_run = "\n  ".join([
            "python -m swebench.harness.run_evaluation \\",
                f"--dataset_name {dataset_path} \\",
                f"--predictions_path {save_path.absolute()} \\",
                # following from https://github.com/princeton-nlp/SWE-bench?tab=readme-ov-file#-usage
                f"--max_workers {min(24, int(0.75 * (os.cpu_count() or 2)))} \\",
                f"--run_id {run_id} \\",
                f"--split {split}",
        ])
        # using print here because the formatting gets in the way of copy-pasting.
        print()
        print("A few things...")
        print(f"1) Logs saved to: {CURRENT_RUN.absolute()}")
        print("2) Run the following within SWE-Bench's root directory to evaluate the generated patches.")
        print("=====COPY=====")
        print(to_run)
        print("===END COPY===")
        
        with open(save_path, "a") as f:
            json.dump(patches, f)
