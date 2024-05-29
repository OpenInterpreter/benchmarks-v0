from dataclasses import dataclass
import re
from typing import Callable, Dict, List, Optional, TypedDict, cast
from datasets import load_dataset
from fsspec import AbstractFileSystem, filesystem
from interpreter import OpenInterpreter

from benchmark import LMC, Benchmark, LoadedTask, ResultStatus, TaskResult, ZeroShotTask
from constants import DATASETS, GAIA
from utils import copy_between_fss, wrapping_offset


GAIATask = TypedDict("GAIATask", {
    "task_id": str,
    "Question": str,
    "Level": str,
    "Final answer": str,
    "file_name": str,
    "file_path": str,
    "Annotator Metadata": Optional[Dict[str, str]]
})


def judge_result(initial_prompt: str, last_msg: str, expected: str) -> ResultStatus:
    judge = OpenInterpreter()
    judge.llm.model = "gpt-4-turbo"
    judge.llm.context_window = 128000  # type: ignore

    guide = "Answer with the single word 'correct', 'incorrect', 'unknown', or 'error', and do NOT answer in markdown"
    q = f"Does the message '{last_msg}' contain the answer '{expected}' to the prompt '{initial_prompt}'?"
    prompt = f"{guide}: {q}"
    judge_msgs = cast(List[LMC], judge.chat(prompt, display=False))
    assert len(judge_msgs) > 0, "the judge is speechless!"

    judge_result = judge_msgs[0]["content"]
    assert judge_result in {"correct", "incorrect", "unknown", "error"}, f"the judge's response was unexpected! response: {judge_result}"

    judge.computer.terminate()
    return judge_result  # type: ignore


@dataclass
class LoadedGAIATask(LoadedTask[GAIATask]):
    task: GAIATask

    def setup_input_dir(self, fs: AbstractFileSystem):
        if self.task["file_path"] == "":
            return
        local_fs = filesystem("file")
        copy_between_fss(local_fs, self.task["file_path"], fs, self.task["file_name"])
    
    def to_zero_shot(self) -> ZeroShotTask:
        file_path = f"input/{self.task['file_name']}"
        prompt = self.task["Question"] if self.task["file_name"] == "" else f"file_path: {file_path}\n{self.task['Question']}"
        return {"id": self.task["task_id"], "prompt": prompt}
    
    def to_result_status(self, messages: List[LMC]) -> ResultStatus:
        if len(messages) == 0:
            return "unknown"
        final_message = messages[-1]
        if "role" not in final_message:
            return "unknown"
        if final_message["role"] == "error":
            return "error"
        if "content" not in final_message:
            return "unknown"
        
        expected = self.task["Final answer"]
        prompt = self.to_zero_shot()["prompt"]
        return judge_result(prompt, final_message["content"], expected)

        # final_answer_re = re.search("FINAL ANSWER: (.+)", final_message["content"])
        # if final_answer_re is None:
        #     return "unknown"

        # actual = final_answer_re.group(1).strip()
        # if actual.lower() != expected.lower():
        #     return "incorrect"
        # else:
        #     return "correct"


class GAIABenchmark(Benchmark[GAIATask]):
    def __init__(self, first_n: Optional[int] = None, offset: int = 0, predicates: List[Callable[[GAIATask], bool]] = []):
        self.first_n = first_n
        self.offset = offset
        self.predicates = predicates

    def get_tasks(self) -> List[GAIATask]:
        ds = load_dataset(str(GAIA), "2023_all", split="validation", data_dir=str(DATASETS), trust_remote_code=True)
        tasks = cast(List[GAIATask], list(ds))
        n_tasks = self.first_n or len(tasks)
        return wrapping_offset([t for t in tasks if all(p(t) for p in self.predicates)], self.offset, n_tasks)
        
    def load_task(self, task: GAIATask) -> LoadedTask[GAIATask]:
        return LoadedGAIATask(task)
