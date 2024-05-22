from concurrent.futures import ThreadPoolExecutor
import logging
import traceback
from dataclasses import dataclass
from queue import Queue
from threading import Thread
from abc import ABC, abstractmethod
from datetime import datetime
from typing import Callable, Dict, Generic, List, Literal, NotRequired, Optional, Tuple, TypeVar, TypedDict, cast
import uuid

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


def command_to_interpreter(cmd: OpenInterpreterCommand) -> OpenInterpreter:
    interpreter = OpenInterpreter(import_computer_api=True)
    interpreter.llm.model = cmd.get("model", interpreter.llm.model)  # type: ignore
    interpreter.llm.context_window = cmd.get("context_window", interpreter.llm.context_window)  # type: ignore
    interpreter.llm.api_base = cmd.get("api_base", interpreter.llm.api_base)  # type: ignore
    interpreter.llm.api_key = cmd.get("api_key", interpreter.llm.api_key)  # type: ignore
    interpreter.auto_run = cmd.get("auto_run", interpreter.auto_run)  # type: ignore
    interpreter.os = cmd.get("os_mode", interpreter.os)  # type: ignore
    interpreter.custom_instructions = cmd.get("custom_instructions", interpreter.custom_instructions)  # type: ignore
    return interpreter


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
    def run(self, command: OpenInterpreterCommand, prompt: str) -> List[LMC]:
        ...


class DefaultBenchmarkRunner(BenchmarkRunner):
    def run(self, command: OpenInterpreterCommand, prompt: str) -> Tuple[datetime, List[LMC], datetime]:
        interpreter = command_to_interpreter(command)
        start = datetime.now()

        try:
            output = cast(List, interpreter.chat(prompt, display=False, stream=False))
        except KeyboardInterrupt:
            output = [*interpreter.messages, { "role": "error", "content": "KeyboardInterrupt" }]
        except Exception as e:
            trace = traceback.format_exc()
            output = [*interpreter.messages, { "role": "error", "content": trace }]
        finally:
            end = datetime.now()
            interpreter.computer.terminate()
            return start, output, end


def run_benchmark(benchmark: Benchmark, command: OpenInterpreterCommand) -> List[TaskResult]:
    all_tasks = benchmark.get_tasks()
    runner = DefaultBenchmarkRunner()
    results: List[TaskResult] = []

    logger.debug(f"Running {len(all_tasks)} task(s)...")

    for task in all_tasks:
        zstask = benchmark.task_to_id_prompt(task)

        logger.debug(f"  Running task {zstask['id']}...")
        start, messages, end  = runner.run(command, zstask["prompt"])

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


def run_benchmark_threaded_pool(benchmark: Benchmark[Task], command: OpenInterpreterCommand, n_threads: Optional[int] = None) -> List[TaskResult]:
    all_tasks = benchmark.get_tasks()
    runner = DefaultBenchmarkRunner()
    task_results: List[TaskResult] = []

    def run_task(task: Task) -> TaskResult:
        zstask = benchmark.task_to_id_prompt(task)
        logger.debug(f"  task {zstask['id']}: RUNNING...")
        try:
            start, messages, end = runner.run(command, zstask["prompt"])
            status = benchmark.task_result_status(task, messages)
        except Exception as e:
            logger.debug(f"  task {zstask['id']}: EXCEPTION!")
            logger.debug(e)
            status = "error"
            start = None
            end = None
        finally:
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


def run_benchmark_threaded(benchmark: Benchmark[Task], command: OpenInterpreterCommand, n_threads: int = 2) -> List[TaskResult]:
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
            start, messages, end = runner.run(command, zstask["prompt"])
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
