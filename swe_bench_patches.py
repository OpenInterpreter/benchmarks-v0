import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from io import BytesIO, StringIO
import json
import logging
import os
from queue import Empty, Queue
import selectors
import shlex
from subprocess import Popen, PIPE, STDOUT, run
from tempfile import TemporaryDirectory
import threading
import time
from typing import IO, Dict, Iterator, List, Literal, Optional, Tuple, TypedDict, cast
from datasets import load_dataset
from commands import OpenInterpreterCommand
from constants import LOCAL
from runners import docker_daemon, get_free_port, talk_to_oi_server
from swe_bench import SWEBenchTask
from coordinators import TaskLifecycle
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
    """
    return ["docker", "exec", "-t", *exec_args, container_id, *cmd]
    # return ["docker", "exec", "-t", *exec_args, container_id, "sh", "-c", f"{' '.join(cmd)} > /proc/1/fd/1 2>/proc/1/fd/2"]


logs_q: Queue[Tuple[logging.Logger, Popen]] = Queue()
def process_background_logs():
    # the data being registered in each instance is the associated logger.
    sel = selectors.DefaultSelector()
    # key_to_lp: Dict[selectors.SelectorKey, Tuple[logging.Logger, subprocess.Popen]] = {}

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
                sel.register(p.stdout, selectors.EVENT_WRITE, (logger, p))
            if p.stderr is not None:
                sel.register(p.stderr, selectors.EVENT_WRITE, (logger, p))
        else:
            events = sel.select(timeout=0)
            for key, _ in events:
                logger, p = cast(Tuple[logging.Logger, Popen], key.data)
                stream = cast(IO, key.fileobj)
                logger.debug(stream.readline().decode())

                if p.poll() is not None:
                    if p.stdout is not None:
                        sel.unregister(p.stdout)
                    if p.stderr is not None:
                        sel.unregister(p.stderr)
                    logger.debug(stream.read().decode())


def generate_lines_out(p: Popen[bytes]) -> Iterator[bytes]:
    if p.stdout is None:
        raise ValueError("Expected Popen to have a non-None standard out!")
    while p.poll() is None:
        yield p.stdout.readline()
    yield p.stdout.read()


logging_thread = threading.Thread(target=process_background_logs)
logging_thread.daemon = True  # so the thread terminates on exit.
logging_thread.start()
def log_nonblocking(logger: logging.Logger, p: Popen):
    logger.debug(f"NON-BLOCKING COMMAND: {p.args}")
    logs_q.put((logger, p))


def log_blocking(logger: logging.Logger, p: Popen[bytes]):
    logger.debug(f"BLOCKING COMMAND: {p.args}")
    assert p.stdout is not None
    while p.poll() is None:
        logger.debug(p.stdout.readline().decode())
    logger.debug(p.stdout.read().decode())


if __name__ == "__main__":
    default_command_id = "gpt35turbo"

    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--command", action="store", type=str, default=default_command_id)
    parser.add_argument("-s", "--split", action="store", type=str, default="dev")
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

    LOGS = LOCAL / "logs"
    CURRENT_RUN = LOGS / run_id
    CURRENT_RUN.mkdir(parents=True, exist_ok=True)

    run_file_handler = logging.FileHandler(CURRENT_RUN / "run.log")
    run_file_handler.setFormatter(formatter)
    module_logger.addHandler(run_file_handler)

    def make_loggers(name: str) -> Tuple[logging.Logger, logging.Logger]:
        """
        Creates folders to put logging stuff if they don't exist.

        Returns at index
            0: Logger for entire container (DEBUG+),
            1: Logger for plaintext response (DEBUG+),
            both are children of the module_logger.
        """
        base = CURRENT_RUN / name
        base.mkdir(parents=True, exist_ok=True)

        container_logger = module_logger.getChild(name)
        container_handler = logging.FileHandler(base / "container.log")
        container_handler.setLevel(logging.DEBUG)
        container_handler.setFormatter(formatter)
        container_logger.addHandler(container_handler)

        plain_logger = module_logger.getChild("plain")
        plain_file_handler = logging.FileHandler(base / "plain.log")
        plain_file_handler.setLevel(logging.DEBUG)
        plain_file_handler.terminator = ""
        plain_logger.addHandler(plain_file_handler)

        return container_logger, plain_logger

    with TemporaryDirectory() as repos_dir:
        repos = set(t["repo"] for t in tasks)
        for r in repos:
            name = r.replace("/", "__")
            remote = f"https://github.com/swe-bench/{name}.git"
            log_blocking(module_logger, Popen(["git", "clone", remote], cwd=repos_dir, stdout=PIPE, stderr=STDOUT))

        def run_task(task: SWEBenchTask, command: OpenInterpreterCommand) -> SWEBenchPrediction:
            container_name = f"{task['instance_id']}_{time.time()}"
            container_log, response_log = make_loggers(container_name)
            worker_name = "server-worker"

            port = get_free_port()
            with (
                TemporaryDirectory() as td,
                docker_daemon(worker_name, ["--name", container_name, "-v", f"{td}:/main", "-p", f"{port}:8000"]) as container_id,
            ):
                # download repository.
                repo = task["repo"].replace("/", "__")
                log_blocking(container_log, Popen(["cp", "-R", f"{repos_dir}/{repo}", f"{td}/repo"], stdout=PIPE, stderr=STDOUT))

                log_blocking(
                    container_log,
                    Popen(
                        dexec_cmd(container_id, ["--workdir", "/main/repo"], ["sh", "-c", f"git checkout {task['base_commit']}"]),
                        stdout=PIPE,
                        stderr=STDOUT))

                command_json_str = json.dumps(command)
                p = Popen(dexec_cmd(container_id, [], ["python", "-m", "worker.run", command_json_str, "/main/repo"]), stdout=PIPE, stderr=STDOUT)
                for line in generate_lines_out(p):
                    if "Uvicorn running" in line.decode():
                        break
                log_nonblocking(container_log, p)
                
                prompt = "\n".join([
                    "Edit the repo in the cwd until you solve the following problem.  Actually edit the files you think will solve the problem.",
                    f"PROBLEM: {task['problem_statement']}",
                ])
                talk_to_oi_server(f"localhost:{port}", prompt, lambda bs: response_log.debug(bs.decode()))

                # Generate the diff of staged changes
                log_blocking(
                    container_log,
                    Popen(
                        # ["docker", "exec", "--workdir", "/main/repo", "-t", container_id, "sh", "-c", "git add --all && git diff --cached > ../diff.patch"],
                        dexec_cmd(container_id, ["--workdir", "/main/repo"], ["sh", "-c", "git add --all && git diff --cached > ../diff.patch"]),
                        stdout=PIPE,
                        stderr=STDOUT))

                with open(f"{td}/diff.patch", "r") as f:
                    container_log.debug("DIFF:")
                    diff = f.read()

            return {
                "instance_id": task["instance_id"],
                "model_patch": diff,
                "model_name_or_path": command["model"] if "model" in command else "<N/A>"
            }

        with ThreadPoolExecutor(max_workers=args.nworkers) as pool:
            n_finished_lock = threading.Lock()
            n_finished = 0
            def make_fs(id: str):
                def start():
                    module_logger.info(f"task {id}: RUNNING...")
                def log(s: str):
                    module_logger.info(f"task {id} log: {s}")
                def done(r: SWEBenchPrediction):
                    global n_finished, n_finished_lock
                    with n_finished_lock:
                        n_finished += 1
                    module_logger.info(f"task {id}: DONE! ({n_finished}/{total})")
                return start, log, done

            run_task_args = [(task, command, make_fs(task["instance_id"])) for task in tasks]
            apps = []
            for args in run_task_args:
                tlc = TaskLifecycle[SWEBenchPrediction]()
                start, log, done = make_fs(args[0]["instance_id"])
                tlc.add_start_fn(start)
                tlc.add_done_fn(done)
                apps.append((tlc.wrap(run_task), (*args[:-1],)))
            
            module_logger.info(f"Running {total} tasks across {pool._max_workers} workers...")

            futures = [pool.submit(fn, *args) for fn, args in apps]
            
            for f in as_completed(futures):
                patches.append(f.result())
            
            module_logger.info("Finished!")

            save_path = CURRENT_RUN / "predictions.json"
            module_logger.info(f"Saving SWE-Bench prediction to {save_path}.")

            to_run = "\n  ".join([
                "python -m swebench.harness.run_evaluation \\",
                    f"--dataset_name {dataset_path} \\",
                    f"--predictions_path {save_path.absolute()} \\",
                    # I don't remember where I got the following formula from but I think it's a good idea.
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
