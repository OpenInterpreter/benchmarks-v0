import asyncio
import asyncio.subprocess
import json
import logging
import os
import sys
import threading
import traceback
import uuid
import time
import uvicorn
from hypercorn.config import Config
from hypercorn.asyncio import serve
from contextlib import contextmanager
from contextlib import contextmanager
from concurrent.futures import Future, ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from queue import Queue
from threading import Thread
from datetime import datetime
from typing import Any, Callable, Any, Dict, Generic, List, Literal, Optional, ParamSpec, Set, Tuple, TypeVar, TypedDict, Union, cast
from fastapi import FastAPI, Request, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from interpreter import OpenInterpreter
from commands import OpenInterpreterCommand
from rich.spinner import Spinner
from rich.live import Live
from rich.console import Console, Group, RenderableType
from rich.spinner import Spinner
from rich.text import Text
from rich.padding import Padding

from modifiers import IdModifier, TaskSetModifier
from runners import BenchmarkRunner, DockerBenchmarkRunner
from task import LMC, LoadedTask, ResultStatus, TaskResult, TasksStore, ZeroShotTask


logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.DEBUG)
logger.addHandler(console_handler)

Value = TypeVar("Value")
class LockedValue(Generic[Value]):
    """
    Only works with primitives.
    """
    def __init__(self, value: Value):
        self._lock = threading.Lock()
        self._value = value
    
    def get(self):
        return self._value

    def set(self, value: Value):
        with self._lock:
            self._value = value


class LockedObject(Generic[Value]):
    def __init__(self, obj: Value):
        self._lock = threading.Lock()
        self._obj = obj
    
    @contextmanager
    def use(self):
        with self._lock:
            yield self._obj


DO_NOTHING = lambda _: None


def run_benchmark(benchmark: TasksStore, mod: TaskSetModifier, command: OpenInterpreterCommand, runner: BenchmarkRunner) -> List[TaskResult]:
    all_tasks = mod.modify(benchmark.get_tasks())
    results: List[TaskResult] = []

    logger.debug(f"Running {len(all_tasks)} task(s)...")

    for task in all_tasks:
        lt = benchmark.load_task(task)
        zstask = lt.to_zero_shot()

        logger.debug(f"  Running task {zstask['id']}...")
        start = datetime.now()
        messages  = runner.run(lt, command, DO_NOTHING, lambda: False, DO_NOTHING)
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


Result = TypeVar("Result")
_P = ParamSpec("_P")


class TaskLifecycle(Generic[Result]):
    def __init__(self):
        self._start_fns: List[Callable[[], None]] = []
        self._done_fns: List[Callable[[Result], None]] = []
    
    def add_start_fn(self, fn: Callable[[], None]):
        self._start_fns.append(fn)
    
    def add_done_fn(self, fn: Callable[[Result], None]):
        self._done_fns.append(fn)
    
    def wrap(self, fn: Callable[_P, Result]) -> Callable[_P, Result]:
        def wrapped_fn(*args, **kwargs) -> Result:
            for sfn in self._start_fns: sfn()
            result = fn(*args, **kwargs)
            for dfn in self._done_fns: dfn(result)
            return result
        return wrapped_fn  # type: ignore


def run_task(lt: LoadedTask, command: OpenInterpreterCommand, runner: BenchmarkRunner, log) -> TaskResult:
    zstask = lt.to_zero_shot()
    start = datetime.now()
    try:
        messages = runner.run(lt, command, DO_NOTHING, lambda: False, log)
        status = lt.to_result_status(messages)
    except Exception as e:
        log(traceback.print_exc(file=sys.stdout))
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


def run_benchmark_worker_pool(benchmark: TasksStore, mod: TaskSetModifier, command: OpenInterpreterCommand, runner: BenchmarkRunner, n_workers: Optional[int] = None) -> List[TaskResult]:
    all_tasks = [benchmark.load_task(t) for t in mod.modify(benchmark.get_tasks())]
    task_results: List[TaskResult] = []

    actual_n_workers = n_workers or os.cpu_count()
    with ThreadPoolExecutor(max_workers=actual_n_workers) as pool:
        logger.debug(f"Running {len(all_tasks)} tasks across {actual_n_workers} threads...")

        zero_shots = [(lt, lt.to_zero_shot()) for lt in all_tasks]

        def make_fs(id: str):
            def start():
                logger.debug(f"  task {id}: RUNNING...")
            def log(s: str):
                logger.debug(f"  task {id} log: {s}")
            def done(r: TaskResult):
                logger.debug(f"  task {r['task_id']}: {r['status']}!")
            return start, log, done

        run_task_args = [(lt, command, runner, make_fs(zs["id"])) for lt, zs in zero_shots]
        apps = []
        for args in run_task_args:
            tlc = TaskLifecycle[TaskResult]()
            start, log, done = make_fs(args[0].to_zero_shot()['id'])
            tlc.add_start_fn(start)
            tlc.add_done_fn(done)
            apps.append((tlc.wrap(run_task), (*args[:-1], log)))
        futures = [pool.submit(fn, *args) for fn, args in apps]
        
        for f in as_completed(futures):
            task_results.append(f.result())

        logger.debug(f"Done!")
    
    return task_results


def run_benchmark_threaded(benchmark: TasksStore, mod: TaskSetModifier, command: OpenInterpreterCommand, runner: BenchmarkRunner, n_threads: int = 2) -> List[TaskResult]:
    all_tasks = mod.modify(benchmark.get_tasks())
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
            messages = runner.run(lt, command, DO_NOTHING, lambda: False, DO_NOTHING)
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


class WebSocketsManager:
    def __init__(self):
        self._lock = threading.Lock()

        # read and modified by multiple threads.
        self._history: List[bytes] = []

        # read and added to by multiple coroutines, threads.
        self._websockets: Set[WebSocket] = set()
        # read and modified by multiple coroutines, threads.
        self._disconnect_events: Dict[WebSocket, asyncio.Event] = {}
        # read and modified by multiple threads.
        self._is_closed = False

    def _is_connected(self, ws: WebSocket):
        return ws in self._websockets

    async def add(self, ws: WebSocket):
        with self._lock:
            self._websockets.add(ws)
            self._disconnect_events[ws] = asyncio.Event()
            for bs in self._history:
                await self._send_bytes_to(ws, bs)

    def _remove(self, ws: WebSocket):
        """
        Assumes we're inside self._ws_lock's critical section.
        """
        self._websockets.remove(ws)
        self._disconnect_events[ws].set()
        del self._disconnect_events[ws]
    
    def is_closed(self):
        return self._is_closed
    
    def close(self):
        with self._lock:
            for ws in self._websockets:
                self._disconnect_events[ws].set()
            self._is_closed = True
            self._websockets.clear()
    
    async def wait_until_disconnect(self, websocket: WebSocket):
        await self._disconnect_events[websocket].wait()
    
    async def write(self, bs: bytes):
        with self._lock:
            self._history.append(bs)
        
            if len(self._websockets) == 0:
                return

            # the actual broadcast portion.
            to_remove = set()
            for ws in self._websockets:
                should_remove = await self._send_bytes_to(ws, bs)
                if should_remove:
                    to_remove.add(ws)
            for ws in to_remove:
                self._remove(ws)
    
    async def write_json(self, j: Any):
        await self.write(json.dumps(j).encode("utf-8"))

    async def _send_bytes_to(self, ws: WebSocket, b: bytes) -> bool:
        """
        Returns True if the websocket has been disconnected from.
        Returns False otherwise.

        Assumes this thread has access to self._lock.
        """
        try:
            await ws.send_bytes(b)
            return False
        except (WebSocketDisconnect, RuntimeError):
            return True


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
            # self.should_exit = True
            self.force_exit = True
            logger.debug("about to shutdown server!")
            asyncio.run(self.shutdown())
            logger.debug("about to join server thread with main thread")
            thread.join(timeout=10)


@contextmanager
def run_background_server(app: FastAPI):
    loop = asyncio.new_event_loop()
    shutdown_event = asyncio.Event()

    def _start_server():
        c = Config()
        coroutine = serve(app, c, shutdown_trigger=shutdown_event.wait)  # type: ignore
        loop.run_until_complete(coroutine)  # type: ignore

    th = threading.Thread(target=_start_server)
    th.start()
    try:
        yield
    finally:
        loop.call_soon_threadsafe(shutdown_event.set)
        logger.debug("about to join threads")
        th.join()
        logger.debug("joined!")


class TaskStartedPayload(TypedDict):
    tag: Literal["started"]


class TaskDonePayload(TypedDict):
    tag: Literal["done"]
    status: ResultStatus


class TaskLogPayload(TypedDict):
    tag: Literal["log"]
    message: str


TaskUpdatePayload = Union[
    TaskStartedPayload,
    TaskDonePayload,
    TaskLogPayload
]


class TaskUpdate(TypedDict):
    task_id: str
    payload: TaskUpdatePayload


def run_benchmark_worker_pool_with_server(
        tasks: TasksStore,
        mod: TaskSetModifier,
        cmd: OpenInterpreterCommand,
        rnnr: BenchmarkRunner,
        nworkers: int | None = None
    ) -> List[TaskResult]:

    app = FastAPI()
    templates = Jinja2Templates("templates")

    all_tasks = [tasks.load_task(t) for t in mod.modify(tasks.get_tasks())]
    results_lock = threading.Lock()
    task_results: List[TaskResult] = []
    zs_tasks = [(t, t.to_zero_shot()) for t in all_tasks]
    zs_map = {zst["id"]: zst for _, zst in zs_tasks}
    task_managers = {zst["id"]: WebSocketsManager() for _, zst in zs_tasks}
    updates_manager = WebSocketsManager()

    @app.get("/view/{task_id}", response_class=HTMLResponse)
    async def view(request: Request, task_id: str):
        prompt = zs_map[task_id]["prompt"]
        return templates.TemplateResponse(
            request,
            name="logs.html.j2",
            context={"task_id": task_id, "prompt": prompt, "command": json.dumps(cmd, indent=2)})

    @app.websocket("/logs/{task_id}")
    async def logs(websocket: WebSocket, task_id: str):
        await websocket.accept()
        await task_managers[task_id].add(websocket)
        await task_managers[task_id].wait_until_disconnect(websocket)
        await websocket.close()
    
    @app.post("/stop/{task_id}")
    async def stop(task_id: str) -> bool:
        task_managers[task_id].close()
        return True

    @app.get("/", response_class=HTMLResponse)
    async def full(request: Request):
        return templates.TemplateResponse(
            request,
            name="full.html.j2",
            context={"tasks": [zs["id"] for _, zs in zs_tasks]}
        )

    @app.websocket("/updates")
    async def updates(websocket: WebSocket):
        await websocket.accept()
        await updates_manager.add(websocket)
        await updates_manager.wait_until_disconnect(websocket)
        await websocket.close()
   
    def run_task(lt: LoadedTask, zs: ZeroShotTask, ws_manager: WebSocketsManager, log: Callable[[str], None]) -> TaskResult:
        def write(b: bytes):
            asyncio.run(ws_manager.write(b))
        
        start = datetime.now()
        try:
            messages = rnnr.run(lt, cmd, write, ws_manager.is_closed, log)
            status = lt.to_result_status(messages)
        except Exception as e:
            tb = traceback.print_exc(file=sys.stdout)
            if tb is not None:
                log(tb)
            else:
                log(str(e))
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
        
    with run_background_server(app), ThreadPoolExecutor(max_workers=nworkers) as pool:
        done_event = threading.Event()

        def make_update_fns(id: str):
            def start():
                asyncio.run(updates_manager.write_json({"task_id": id, "payload": {"tag": "started"}}))

            def log(s: str):
                asyncio.run(updates_manager.write_json({"task_id": id, "payload": {"tag": "log", "message": s}}))

            def done(result: TaskResult):
                asyncio.run(updates_manager.write_json({"task_id": id, "payload": {"tag": "done", "status": result["status"]}}))

                with results_lock:
                    task_results.append(result)
                task_managers[id].close()
                if len(task_results) >= len(all_tasks):
                    done_event.set()

            return start, log, done

        futures: List[Future[TaskResult]] = []
        for lt, zs in zs_tasks:
            start, log, done = make_update_fns(zs["id"])
            tlc = TaskLifecycle[TaskResult]()
            tlc.add_start_fn(start)
            tlc.add_done_fn(done)
            args = (lt, zs, task_managers[zs["id"]], log)
            futures.append(pool.submit(tlc.wrap(run_task), *args))

        done_event.wait()

        for manager in task_managers.values():
            manager.close()
        updates_manager.close()

        logger.debug("Hold CTRL+C to close the server.")

        try:
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            ...
        
        logger.debug("exiting server...")
    logger.debug("exited!")

    return task_results


@dataclass
class OIBenchmarks:
    tasks: TasksStore
    command: OpenInterpreterCommand
    runner: BenchmarkRunner = field(default_factory=DockerBenchmarkRunner)
    modifier: TaskSetModifier = field(default_factory=IdModifier)
    nworkers: Optional[int] = None
    server: bool = False

    def run(self) -> List[TaskResult]:
        if self.server:
            results = run_benchmark_worker_pool_with_server(self.tasks, self.modifier, self.command, self.runner, self.nworkers)
        else:
            results = run_benchmark_worker_pool(self.tasks, self.modifier, self.command, self.runner, self.nworkers)
        print("at end of run function!")
        return results
