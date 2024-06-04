import json
import logging
import os
import subprocess
import sys
import tempfile
import traceback
import uuid
import shlex
import time
from pathlib import Path
from concurrent.futures import Future, ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from queue import Queue
from threading import Thread
from abc import ABC, abstractmethod
from datetime import datetime
from typing import Dict, Generic, List, Literal, NotRequired, Optional, Tuple, TypeVar, TypedDict, cast
from fsspec import AbstractFileSystem
from interpreter import OpenInterpreter

from utils import LocalBasedFS, change_working_dir, wrapping_offset
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
    def run(self, lt: LoadedTask[Task], command: OpenInterpreterCommand, prompt: str) -> List[LMC]:
        ...


class DefaultBenchmarkRunner(BenchmarkRunner):
    def run(self, lt: LoadedTask, command: OpenInterpreterCommand, prompt: str) -> List[LMC]:
        with tempfile.TemporaryDirectory() as worker_dir:
            input_dir = Path(worker_dir) / Path("input")
            output_dir = Path(worker_dir) / Path("output")
            input_dir.mkdir(parents=True, exist_ok=True)
            lt.setup_input_dir(LocalBasedFS(str(input_dir)))

            command_json_str = json.dumps(command)
            subprocess.run(["python", "-m", "worker.run", command_json_str, f"{shlex.quote(prompt)}", output_dir], cwd=worker_dir)
            messages_path = output_dir / worker.OUTPUT_PATH
            with open(messages_path, "r") as f:
                messages = json.load(f)
                return messages
            # with change_working_dir(worker_dir):
            #     result = worker.run(command, prompt) # type: ignore
            # return result


class DockerBenchmarkRunner(BenchmarkRunner):
    WORKER_NAME = "worker"

    def run(self, lt: LoadedTask[Task], command: OpenInterpreterCommand, prompt: str) -> List[LMC]:
        with tempfile.TemporaryDirectory() as temp_dir:
            output_dir = Path(temp_dir) / Path("output")
            input_dir = Path(temp_dir) / Path("input")
            command_json_str = json.dumps(command)
            input_dir.mkdir(parents=True, exist_ok=True)
            output_dir.mkdir(parents=True, exist_ok=True)
            lt.setup_input_dir(LocalBasedFS(str(input_dir)))
            container_name = f"{lt.to_zero_shot()["id"]}_{time.time()}"
            dcmd = [
                "docker", "run", "-t",
                "-v", f"{input_dir}:/input", "-v", f"{output_dir}:/output",
                "--name", container_name,
                DockerBenchmarkRunner.WORKER_NAME,
                command_json_str, f"{shlex.quote(prompt)}", "/output"
            ]
            subprocess.run(dcmd, stdout=subprocess.DEVNULL)
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
    runner = DefaultBenchmarkRunner()
    results: List[TaskResult] = []

    logger.debug(f"Running {len(all_tasks)} task(s)...")

    for task in all_tasks:
        lt = benchmark.load_task(task)
        zstask = lt.to_zero_shot()

        logger.debug(f"  Running task {zstask['id']}...")
        start = datetime.now()
        messages  = runner.run(lt, command, zstask["prompt"])
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
    logger.debug(f"  task {zstask['id']}: RUNNING...")
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
        logger.debug(f"  task {zstask['id']}: {status}!")
        return {
            "task_id": zstask["id"],
            "command": command,
            "prompt": zstask["prompt"],
            "start": start,
            "end": end,
            "messages": messages,
            "status": status
        }


def run_benchmark_worker_pool(benchmark: TasksStore[Task], mod: TaskSetModifier[Task], command: OpenInterpreterCommand, runner: BenchmarkRunner, n_workers: Optional[int] = None) -> List[TaskResult]:
    all_tasks = mod.modify(benchmark.get_tasks())
    task_results: List[TaskResult] = []

    actual_n_workers = n_workers or os.cpu_count()
    with ThreadPoolExecutor(max_workers=actual_n_workers) as pool:
        logger.debug(f"Running {len(all_tasks)} tasks across {actual_n_workers} threads...")
        run_task_args = [(benchmark.load_task(t), command, runner) for t in all_tasks]
        futures = [pool.submit(run_task, *args) for args in run_task_args]
        for f in as_completed(futures):
            task_results.append(f.result())
        logger.debug(f"Done!")
    
    return task_results


def run_benchmark_threaded(benchmark: TasksStore[Task], mod: TaskSetModifier, command: OpenInterpreterCommand, n_threads: int = 2) -> List[TaskResult]:
    all_tasks = mod.modify(benchmark.get_tasks())
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
            lt = benchmark.load_task(task)
            zstask = lt.to_zero_shot()
            logger.debug(f"  task {zstask['id']} on thread {thread_id}: RUNNING...")
            start = datetime.now()
            messages = runner.run(lt, command, zstask["prompt"])
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
    
    judge_msgs = cast(List[LMC], judge.chat(q, display=False))
    assert len(judge_msgs) > 0, "the judge is speechless!"

    judge_result = judge_msgs[0]["content"].strip().lower()
    assert judge_result in {"correct", "incorrect", "unknown", "error"}, f"the judge's response was unexpected! response: {judge_result}"

    judge.computer.terminate()
    return judge_result  # type: ignore


@dataclass
class OIBenchmarks:
    tasks: TasksStore
    command: OpenInterpreterCommand
    runner: BenchmarkRunner = field(default_factory=DockerBenchmarkRunner)
    modifier: TaskSetModifier = field(default_factory=IdModifier)
    nworkers: Optional[int] = None

    def run(self) -> List[TaskResult]:
        results = run_benchmark_worker_pool(self.tasks, self.modifier, self.command, self.runner, self.nworkers)
        return results
