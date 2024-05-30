"""
For defining a custom set of benchmarks.
"""

import csv
from dataclasses import dataclass, field
from os import PathLike
from pathlib import Path
from typing import List, TypedDict, cast

import jsonschema
import jsonschema.validators

from benchmark import LMC, Benchmark, LoadedTask, ResultStatus, ZeroShotTask, judge_result


CUSTOM_TASK_SCHEMA = {
    "type": "object",
    "properties": {
        "id": { "type": "string" },
        "prompt": { "type": "string" },
        "answer": { "type": "string" },
    }
}


class CustomTask(TypedDict):
    id: str
    prompt: str
    answer: str


@dataclass
class LoadedCustomTask(LoadedTask[CustomTask]):
    task: CustomTask

    def to_zero_shot(self) -> ZeroShotTask:
        return {"id": self.task["id"], "prompt": self.task["prompt"]}

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
        
        expected = self.task["answer"]
        prompt = self.to_zero_shot()["prompt"]
        return judge_result(prompt, final_message["content"], expected)


@dataclass
class CustomBenchmark(Benchmark[CustomTask]):
    tasks: List[CustomTask] = field(default_factory=list)

    @staticmethod
    def from_list(l: List[CustomTask]) -> "CustomBenchmark":
        return CustomBenchmark(l)
    
    @staticmethod
    def from_csv(path: PathLike) -> "CustomBenchmark":
        rows: List[CustomTask] = []
        if Path(path).exists():
            with open(path, "r") as file:
                reader = csv.DictReader(file)
                for row in reader:
                    jsonschema.validate(row, CUSTOM_TASK_SCHEMA)
                    rows.append(cast(CustomTask, row))
        else:
            raise FileNotFoundError(f"'{path}' does not exist, so it can't be used to load a CustomBenchmark.")
        return CustomBenchmark(rows)

    def get_tasks(self) -> List[CustomTask]:
        return self.tasks
        
    def load_task(self, task: CustomTask) -> LoadedTask[CustomTask]:
        return LoadedCustomTask(task)
