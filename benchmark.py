import json
import logging
import subprocess
import sys
import tempfile
import traceback
import uuid
import shlex
from contextlib import contextmanager
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from queue import Queue
from threading import Thread
from abc import ABC, abstractmethod
from datetime import datetime
from typing import Callable, Dict, Generic, List, Literal, NotRequired, Optional, Tuple, TypeVar, TypedDict, cast

import worker
from interpreter import OpenInterpreter


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


class TaskResult(TypedDict):
    task_id: str
    command: OpenInterpreterCommand
    prompt: str
    start: Optional[datetime]
    end: Optional[datetime]
    messages: List[LMC]
    status: ResultStatus


@dataclass
class Benchmark(Generic[Task]):
    get_tasks: Callable[[], List[Task]]
    task_to_id_prompt: Callable[[Task], ZeroShotTask]
    task_result_status: Callable[[Task, List[LMC]], ResultStatus]


class BenchmarkRunner(ABC):
    @abstractmethod
    def run(self, command: OpenInterpreterCommand, prompt: str, display: bool) -> List[LMC]:
        ...


class DefaultBenchmarkRunner(BenchmarkRunner):
    def run(self, command: OpenInterpreterCommand, prompt: str, display: bool) -> List[LMC]:
        return worker.run(command, prompt, display) # type: ignore


class DockerBenchmarkRunner(BenchmarkRunner):
    WORKER_NAME = "worker"

    def run(self, command: OpenInterpreterCommand, prompt: str, display: bool) -> List[LMC]:
        with tempfile.TemporaryDirectory() as temp_dir:
            # with self.temp_docker_volume() as volume_name:
            command_json_str = json.dumps(command)
            dcmd = [
                "docker", "run", "-t",
                "-v", f"{temp_dir}:{temp_dir}",
                DockerBenchmarkRunner.WORKER_NAME,
                command_json_str, f"{shlex.quote(prompt)}", temp_dir
            ]
            print(" ".join(dcmd))
            subprocess.run(dcmd)
            with open(Path(temp_dir) / worker.OUTPUT_PATH) as f:
                messages = json.load(f)
                return messages
    
    @contextmanager
    def temp_docker_volume(self):
        result = subprocess.run(["docker", "volume", "create"], stdout=subprocess.PIPE, text=True)
        volume_name = result.stdout.strip()
        yield volume_name
        subprocess.run(["docker", "volume", "rm", volume_name], stdout=subprocess.DEVNULL)
    

def run_benchmark(benchmark: Benchmark, command: OpenInterpreterCommand, display: bool = False) -> List[TaskResult]:
    all_tasks = benchmark.get_tasks()
    runner = DefaultBenchmarkRunner()
    results: List[TaskResult] = []

    logger.debug(f"Running {len(all_tasks)} task(s)...")

    for task in all_tasks:
        zstask = benchmark.task_to_id_prompt(task)

        logger.debug(f"  Running task {zstask['id']}...")
        start = datetime.now()
        messages  = runner.run(command, zstask["prompt"], display=display)
        end = datetime.now()

        status = benchmark.task_result_status(task, messages)
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


def run_benchmark_threaded_pool(benchmark: Benchmark[Task], command: OpenInterpreterCommand, runner: BenchmarkRunner, n_threads: Optional[int] = None, display: bool = False) -> List[TaskResult]:
    all_tasks = benchmark.get_tasks()
    task_results: List[TaskResult] = []

    def run_task(task: Task) -> TaskResult:
        zstask = benchmark.task_to_id_prompt(task)
        logger.debug(f"  task {zstask['id']}: RUNNING...")
        start = datetime.now()
        try:
            messages = runner.run(command, zstask["prompt"], display=display)
            status = benchmark.task_result_status(task, messages)
        except Exception as e:
            logger.debug(f"  task {zstask['id']}: EXCEPTION!")
            logger.debug(traceback.print_exc(file=sys.stdout))
            status = "error"
            messages = []
        finally:
            end = datetime.now()
            logger.debug(f"  task {zstask['id']}: DONE!")
            return {
                "task_id": zstask["id"],
                "command": command,
                "prompt": zstask["prompt"],
                "start": start,
                "end": end,
                "messages": messages,
                "status": status
            }

    with ThreadPoolExecutor(max_workers=n_threads) as pool:
        logger.debug(f"Running {len(all_tasks)} tasks across {pool._max_workers} threads...")
        pool._max_workers
        results = pool.map(run_task, all_tasks)
        for r in results:
            task_results.append(r)
        logger.debug(f"Done!")
    
    return task_results


def run_benchmark_threaded(benchmark: Benchmark[Task], command: OpenInterpreterCommand, n_threads: int = 2, display: bool = False) -> List[TaskResult]:
    all_tasks = benchmark.get_tasks()
    runner = DefaultBenchmarkRunner()
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
            zstask = benchmark.task_to_id_prompt(task)
            logger.debug(f"  task {zstask['id']} on thread {thread_id}: RUNNING...")
            start = datetime.now()
            messages = runner.run(command, zstask["prompt"], display=display)
            end = datetime.now()
            status = benchmark.task_result_status(task, messages)
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
