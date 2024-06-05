import asyncio
import json
import logging
import os
import subprocess
import sys
import tempfile
import threading
import traceback
import uuid
import shlex
import time
import uvicorn
from contextlib import contextmanager
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from queue import Queue
from threading import Thread
from abc import ABC, abstractmethod
from datetime import datetime
from typing import Any, Callable, Dict, Generic, List, Literal, NotRequired, Optional, ParamSpec, Set, Tuple, TypeVar, TypedDict, cast
from fastapi import FastAPI, Request, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from fsspec import AbstractFileSystem
from interpreter import OpenInterpreter
from rich.spinner import Spinner
from rich.live import Live
from rich.console import Console, Group, RenderableType
from rich.spinner import Spinner
from rich.text import Text
from rich.padding import Padding

from utils import LocalBasedFS, wrapping_offset
import worker


logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.DEBUG)
logger.addHandler(console_handler)

Task = TypeVar("Task")
LMC = Dict[str, str]
ResultStatus = Literal["correct", "incorrect", "unknown", "error"]


class ZeroShotTask(TypedDict):
    id: str
    prompt: str


class OpenInterpreterCommand(TypedDict):
    auto_run: NotRequired[bool]
    os_mode: NotRequired[bool]
    model: NotRequired[str]
    context_window: NotRequired[int]
    api_base: NotRequired[str]
    api_key: NotRequired[str]
    custom_instructions: NotRequired[str]
    supports_functions: NotRequired[bool]


class TaskResult(TypedDict):
    task_id: str
    command: OpenInterpreterCommand
    prompt: str
    start: Optional[datetime]
    end: Optional[datetime]
    messages: List[LMC]
    status: ResultStatus


class LoadedTask(Generic[Task]):
    @abstractmethod
    def setup_input_dir(self, fs: AbstractFileSystem):
        ...
    
    @abstractmethod
    def to_zero_shot(self) -> ZeroShotTask:
        raise NotImplementedError()
    
    @abstractmethod
    def to_result_status(self, messages: List[LMC]) -> ResultStatus:
        raise NotImplementedError()
   

class TaskSetModifier(ABC, Generic[Task]):
    @abstractmethod
    def modify(self, task_set: List[Task]) -> List[Task]:
        ...


class IdModifier(Generic[Task], TaskSetModifier[Task]):
    def modify(self, task_set: List[Task]) -> List[Task]:
        return task_set


@dataclass
class SizeOffsetModifier(Generic[Task], TaskSetModifier[Task]):
    offset: int
    ntasks: Optional[int]

    def modify(self, task_set: List[Task]) -> List[Task]:
        return wrapping_offset([t for t in task_set], self.offset, self.ntasks or len(task_set))
    

@dataclass
class ModifierPipe(Generic[Task], TaskSetModifier[Task]):
    mods: List[TaskSetModifier[Task]]

    def modify(self, task_set: List[Task]) -> List[Task]:
        current = task_set
        for mod in self.mods:
            if len(current) == 0:
                break
            current = mod.modify(current)
        return current


class TasksStore(Generic[Task]):
    @abstractmethod
    def get_tasks(self) -> List[Task]:
        raise NotImplementedError()

    @abstractmethod
    def load_task(self, task: Task) -> LoadedTask[Task]:
        raise NotImplementedError()


class BenchmarkRunner(ABC):
    @abstractmethod
    def run(self, lt: LoadedTask[Task], command: OpenInterpreterCommand, prompt: str, write: Callable[[bytes], None] = lambda _: None) -> List[LMC]:
        ...


class DefaultBenchmarkRunner(BenchmarkRunner):
    def run(self, lt: LoadedTask, command: OpenInterpreterCommand, prompt: str, write) -> List[LMC]:
        with tempfile.TemporaryDirectory() as worker_dir:
            input_dir = Path(worker_dir) / Path("input")
            output_dir = Path(worker_dir) / Path("output")
            input_dir.mkdir(parents=True, exist_ok=True)
            lt.setup_input_dir(LocalBasedFS(str(input_dir)))

            command_json_str = json.dumps(command)
            p = subprocess.Popen([
                "python", "-m", "worker.run",
                command_json_str, f"{shlex.quote(prompt)}", worker_dir, output_dir
            ], stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
            while p.poll() is None and p.stdout is not None:
                write(p.stdout.read())

            messages_path = worker_dir / worker.OUTPUT_PATH
            with open(messages_path, "r") as f:
                messages = json.load(f)
                return messages


class DockerBenchmarkRunner(BenchmarkRunner):
    WORKER_NAME = "worker"

    def run(self, lt: LoadedTask[Task], command: OpenInterpreterCommand, prompt: str, write) -> List[LMC]:
        with tempfile.TemporaryDirectory() as temp_dir:
            output_dir = Path(temp_dir) / Path("output")
            input_dir = Path(temp_dir) / Path("input")
            command_json_str = json.dumps(command)
            input_dir.mkdir(parents=True, exist_ok=True)
            output_dir.mkdir(parents=True, exist_ok=True)
            lt.setup_input_dir(LocalBasedFS(str(input_dir)))
            container_name = f"{lt.to_zero_shot()['id']}_{time.time()}"
            dcmd = [
                "docker", "run", "-t",
                "-v", f"{input_dir}:/input", "-v", f"{output_dir}:/output",
                "--name", container_name,
                DockerBenchmarkRunner.WORKER_NAME,
                command_json_str, f"{shlex.quote(prompt)}", "/", "/output"
            ]

            # subprocess.run(dcmd, stdout=subprocess.DEVNULL)
            p = subprocess.Popen(dcmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
            while p.poll() is None and p.stdout is not None:
                write(p.stdout.read())

            messages_path = Path(temp_dir) / worker.OUTPUT_PATH
            if not messages_path.exists():
                # this will happen if (for example) the container is stopped before it finishes.
                logger.debug(f"couldn't find {messages_path}!")
                return []
            with open(messages_path) as f:
                messages = json.load(f)
                return messages
    

def run_benchmark(benchmark: TasksStore, mod: TaskSetModifier, command: OpenInterpreterCommand) -> List[TaskResult]:
    all_tasks = mod.modify(benchmark.get_tasks())
    runner: BenchmarkRunner = DefaultBenchmarkRunner()
    results: List[TaskResult] = []

    logger.debug(f"Running {len(all_tasks)} task(s)...")

    for task in all_tasks:
        lt = benchmark.load_task(task)
        zstask = lt.to_zero_shot()

        logger.debug(f"  Running task {zstask['id']}...")
        start = datetime.now()
        messages  = runner.run(lt, command, zstask["prompt"], lambda _: None)
        end = datetime.now()

        status = lt.to_result_status(messages)
        result: TaskResult = {
            "task_id": zstask["id"],
            "command": command,
            "prompt": zstask["prompt"],
            "start": start,
            "end": end,
            "status": status,
            "messages": messages,
        }

        results.append(result)

    logger.debug("done!")

    return results


def run_task(lt: LoadedTask[Task], command: OpenInterpreterCommand, runner: BenchmarkRunner) -> TaskResult:
    zstask = lt.to_zero_shot()
    start = datetime.now()
    try:
        messages = runner.run(lt, command, zstask["prompt"])
        status = lt.to_result_status(messages)
    except Exception as e:
        logger.debug(f"  task {zstask['id']}: EXCEPTION!")
        logger.debug(traceback.print_exc(file=sys.stdout))
        status = "error"
        messages = []
    finally:
        end = datetime.now()
        return {
            "task_id": zstask["id"],
            "command": command,
            "prompt": zstask["prompt"],
            "start": start,
            "end": end,
            "messages": messages,
            "status": status
        }


Result = TypeVar("Result")
_P = ParamSpec("_P")


class TaskDisplay(Generic[Result]):
    def __init__(
            self,
            max_cap: int,
            to_start_str: Callable[[str], RenderableType],
            to_stop_str: Callable[[str, Result], RenderableType]
        ):
        self._max_cap = max_cap
        self._to_start_str = to_start_str
        self._to_stop_str = to_stop_str
        self._lock = threading.Lock()
        self._started_ids: List[Tuple[int, str]] = []
        self._results: Dict[int, Result] = {}

    def wrap(self, fn: Callable[_P, Result], ext_str: str) -> Callable[_P, Result]:
        def wrapped_fn(*args, **kwargs):
            ident = id(wrapped_fn)
            self._started(ident, ext_str)
            result = fn(*args, **kwargs)
            self._stopped(ident, result)
            return result

        return wrapped_fn  # type: ignore

    def _started(self, ident: int, ext_str: str):
        with self._lock:
            self._started_ids.append((ident, ext_str))
    
    def _stopped(self, ident: int, result):
        with self._lock:
            self._results[ident] = result
    
    def _render(self, ident: int, ext_str: str) -> RenderableType:
        if ident not in self._results:  # not done yet!
            return self._to_start_str(ext_str)
        else:  # done!
            return self._to_stop_str(ext_str, self._results[ident])

    def display_until_done(self):
        g = Group()
        console = Console()
        with Live(g, console=console) as live:
            while True:
                with self._lock:
                    renderables = [Padding(self._render(ident, ext), (0, 0, 0, 2)) for ident, ext in self._started_ids]
                    live.update(Group(*renderables))
                    if len(self._results) == self._max_cap:
                        break
                time.sleep(1)


def status_style(status: ResultStatus) -> str:
    if status == 'correct':
        return 'green'
    elif status == 'unknown':
        return 'yellow'
    elif status == 'incorrect':
        return 'red'
    elif status == 'error':
        return 'red blink'


def status_character(status: ResultStatus) -> str:
    if status == 'correct':
        return '✅'
    elif status == 'unknown':
        return '🤷'
    elif status == 'incorrect':
        return '❌'
    elif status == 'error':
        return '❗'
    

def run_benchmark_worker_pool(benchmark: TasksStore[Task], mod: TaskSetModifier[Task], command: OpenInterpreterCommand, runner: BenchmarkRunner, n_workers: Optional[int] = None) -> List[TaskResult]:
    all_tasks = mod.modify(benchmark.get_tasks())
    task_results: List[TaskResult] = []

    actual_n_workers = n_workers or os.cpu_count()
    with ThreadPoolExecutor(max_workers=actual_n_workers) as pool:
        logger.debug(f"Running {len(all_tasks)} tasks across {actual_n_workers} threads...")
        d = TaskDisplay[TaskResult](
            len(all_tasks),
            lambda ext: Text(f"task {ext}: ..."),
            lambda ext, r: Text(f"task {ext}: {status_character(r['status'])}")
        )

        run_task_args = [(benchmark.load_task(t), command, runner) for t in all_tasks]
        apps = [(d.wrap(run_task, args[0].to_zero_shot()['id']), args) for args in run_task_args]
        futures = [pool.submit(fn, *args) for fn, args in apps]
        
        d.display_until_done()
        for f in as_completed(futures):
            task_results.append(f.result())

        logger.debug(f"Done!")
    
    return task_results


def run_benchmark_threaded(benchmark: TasksStore[Task], mod: TaskSetModifier, command: OpenInterpreterCommand, n_threads: int = 2) -> List[TaskResult]:
    all_tasks = mod.modify(benchmark.get_tasks())
    runner: BenchmarkRunner = DefaultBenchmarkRunner()
    results: Queue[TaskResult] = Queue()
    threads: List[Tuple[Queue, Thread]] = []

    def run_task(task_queue: Queue):
        thread_id = uuid.uuid4()
        # THERE IS A RACE CONDITION -- check if empty, then get will NOT work.  Should be atomic op.
        # actually jk this isn't a problem because tasks are assigned before any threads are started,
        # and we aren't assigning anything after thread creation.
        # YES I am cheating but it's fine.
        while not task_queue.empty():
            task = task_queue.get()
            lt = benchmark.load_task(task)
            zstask = lt.to_zero_shot()
            logger.debug(f"  task {zstask['id']} on thread {thread_id}: RUNNING...")
            start = datetime.now()
            messages = runner.run(lt, command, zstask["prompt"], lambda _: None)
            end = datetime.now()
            status = lt.to_result_status(task)
            logger.debug(f"  task {zstask['id']} on thread {thread_id}: DONE!")
            results.put({
                "task_id": zstask["id"],
                "command": command,
                "prompt": zstask["prompt"],
                "start": start,
                "end": end,
                "messages": messages,
                "status": status
            })

    logger.debug(f"Setting up {n_threads} thread(s)...")

    # setting up threads.
    for _ in range(0, n_threads):
        q = Queue()
        threads.append((q, Thread(target=run_task, args=(q,))))
    
    logger.debug("done!")
    logger.debug(f"Assigning {len(all_tasks)} tasks to {n_threads} threads...")
    
    # assigning tasks to threads in a round-robin manner.
    th_index = 0
    for task in all_tasks:
        q, _ = threads[th_index]
        q.put(task)
        th_index = (th_index + 1) % n_threads
    
    logger.debug("done!")
    logger.debug(f"Starting {n_threads} thread(s)...")
    
    # starting threads.
    for q, th in threads:
        th.start()
        logger.debug(f"  Started thread with {q.qsize()} tasks.")
    
    logger.debug("done!")
    logger.debug(f"Running {len(all_tasks)} tasks across {n_threads} thread(s)...")

    # joining threads.
    for _, th in threads:
        th.join()
    
    logger.debug("done!")

    return list(results.queue)


def judge_result(initial_prompt: str, last_msg: str, expected: str) -> ResultStatus:
    judge = OpenInterpreter()
    judge.llm.model = "gpt-4"
    judge.llm.context_window = 128000  # type: ignore

    judge.system_message = "You are a grading AI. Answer with the single word 'correct' or 'incorrect', and do NOT answer in markdown."
    q = f"""
    
# QUESTION:
{initial_prompt}
# CORRECT ANSWER:
{expected}
---
# STUDENT'S ANSWER:
{last_msg}
---

Did the student get the answer correct?

    """.strip()
    
    try:
        judge_msgs = cast(List[LMC], judge.chat(q, display=False))
        assert len(judge_msgs) > 0, "the judge is speechless!"

        judge_result = judge_msgs[0]["content"].strip().lower()
        assert judge_result in {"correct", "incorrect", "unknown", "error"}, f"the judge's response was unexpected! response: {judge_result}"
    finally:
        judge.computer.terminate()

    return judge_result  # type: ignore


class TaskSession:
    def __init__(self):
        self._history = bytearray()
        self._websockets: Set[WebSocket] = set()
        self._lock = threading.Lock()

    def is_connected(self, ws: WebSocket):
        with self._lock:
            return ws in self._websockets
    
    async def _send_bytes_to(self, ws: WebSocket, b: bytes) -> bool:
        """
        Returns True if the websocket has been disconnected from.
        Returns False otherwise.

        Assumes this thread has access to self._lock.
        """
        try:
            await ws.send_text(b.decode("utf-8"))
            # await ws.send_bytes(b)
            return False
        except (WebSocketDisconnect, RuntimeError):
            return True
    
    async def _broadcast(self, bs: bytes):
        # Assumes this thread has access to self._lock.
        to_remove = set()
        for ws in self._websockets:
            should_remove = await self._send_bytes_to(ws, bs)
            if should_remove:
                to_remove.add(ws)
        for ws in to_remove:
            self.remove_websocket(ws)


    async def add_websocket(self, ws: WebSocket):
        with self._lock:
            # does the following line need access to the lock?
            self._websockets.add(ws)
            await self._send_bytes_to(ws, self._history)

    def remove_websocket(self, ws: WebSocket):
        self._websockets.remove(ws)

    async def write(self, bs: bytes):
        with self._lock:
            for b in bs:
                self._history.append(b)
            await self._broadcast(bs)


# yoinked from https://stackoverflow.com/questions/61577643/python-how-to-use-fastapi-and-uvicorn-run-without-blocking-the-thread.
class Server(uvicorn.Server):
    def install_signal_handlers(self):
        pass

    @contextmanager
    def run_in_thread(self):
        thread = threading.Thread(target=self.run)
        thread.start()
        try:
            while not self.started:
                time.sleep(1e-3)
            yield
        finally:
            self.should_exit = True
            thread.join()


def run_benchmark_worker_pool_with_server(
        tasks: TasksStore[Task],
        mod: TaskSetModifier[Task],
        cmd: OpenInterpreterCommand,
        rnnr: BenchmarkRunner,
        nworkers: int | None = None
    ) -> List[TaskResult]:

    app = FastAPI()
    templates = Jinja2Templates("templates")

    @app.get("/view/{task_id}", response_class=HTMLResponse)
    async def view(request: Request, task_id: str):
        return templates.TemplateResponse(
            request,
            name="logs.html.j2",
            context={"task_id": task_id})
    
    all_tasks = [tasks.load_task(t) for t in mod.modify(tasks.get_tasks())]
    zs_tasks = [(t, t.to_zero_shot()) for t in all_tasks]
    tasks_map = {zst["id"]: TaskSession() for _, zst in zs_tasks}

    results_lock = threading.Lock()
    task_results: List[TaskResult] = []

    @app.websocket("/logs/{task_id}")
    async def logs(websocket: WebSocket, task_id: str):
        await websocket.accept()
        session = tasks_map[task_id]
        await session.add_websocket(websocket)
        while session.is_connected(websocket):
            with results_lock:
                if len(task_results) == len(all_tasks):
                    break
            await asyncio.sleep(1)

    def run_task(lt: LoadedTask[Task], zs: ZeroShotTask, session: TaskSession) -> TaskResult:
        def write(b: bytes):
            asyncio.run(session.write(b))

        start = datetime.now()
        try:
            messages = rnnr.run(lt, cmd, zs["prompt"], write)
            status = lt.to_result_status(messages)
        except Exception as e:
            logger.debug(f"  task {zs['id']}: EXCEPTION!")
            logger.debug(traceback.print_exc(file=sys.stdout))
            status = "error"
            messages = []
        finally:
            end = datetime.now()
            return {
                "task_id": zs["id"],
                "command": cmd,
                "prompt": zs["prompt"],
                "start": start,
                "end": end,
                "messages": messages,
                "status": status
            }
        
    config = uvicorn.Config(app, log_level="warning")
    host, port = config.host, config.port
    server = Server(config=config)

    td = TaskDisplay[TaskResult](
        len(all_tasks),
        lambda ext:
            Spinner("dots", style="yellow", text=
                Text(f"task ")
                    # .append(ext, style=f"link http://{host}:{port}/view/{ext}")
                    .append(f"http://{host}:{port}/view/{ext}")
                    .append(": ...")),
        lambda ext, r:
            Text(f"🏁 task ")
                # .append(ext, style=f"link http://{host}:{port}/view/{ext}")
                .append(f"http://{host}:{port}/view/{ext}")
                .append(f": {status_character(r['status'])}")
    )

    with server.run_in_thread(), ThreadPoolExecutor(max_workers=nworkers) as pool:
        args_list = [(lt, zs, tasks_map[zs["id"]]) for lt, zs in zs_tasks]
        futures = [pool.submit(td.wrap(run_task, args[1]["id"]), *args) for args in args_list]

        td.display_until_done()

        with results_lock:
            # just do it all in one go!
            for f in as_completed(futures):
                task_results.append(f.result())
        
        logger.debug("Hold CTRL+C to close the server.")
        while True:
            try:
                time.sleep(1)
            except KeyboardInterrupt:
                break

    return task_results


@dataclass
class OIBenchmarks:
    tasks: TasksStore
    command: OpenInterpreterCommand
    runner: BenchmarkRunner = field(default_factory=DockerBenchmarkRunner)
    modifier: TaskSetModifier = field(default_factory=IdModifier)
    nworkers: Optional[int] = None

    def run(self) -> List[TaskResult]:
        # results = run_benchmark_worker_pool(self.tasks, self.modifier, self.command, self.runner, self.nworkers)
        results = run_benchmark_worker_pool_with_server(self.tasks, self.modifier, self.command, self.runner, self.nworkers)
        return results
