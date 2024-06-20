from typing import List, Literal, TypedDict, cast
from fsspec import AbstractFileSystem
from datasets import load_dataset

from task import LMC, LoadedTask, ResultStatus, TasksStore, ZeroShotTask


class SWEBenchTask(TypedDict):
    """
    An explanation behind each field can be found at https://huggingface.co/datasets/princeton-nlp/SWE-bench.
    You hear that GAIA!?
    """
    instance_id: str
    patch: str
    repo: str
    base_commit: str
    hints_text: str
    created_at: str
    test_patch: str
    problem_statement: str
    version: str
    environment_setup_commit: str
    FAIL_TO_PASS: str
    PASS_TO_FAIL: str


class LoadedSWEBenchTask(LoadedTask[SWEBenchTask]):
    def to_result_status(self, messages: List[LMC]) -> ResultStatus:
        return "unknown"

    def to_zero_shot(self) -> ZeroShotTask:
        """
        Initially, all the setup will be done by the LM itself.
        This is to make my job easier for the time being.
        """
        lines = [
            f"I want you to solve this github repo for me.",
            f"GITHUB REPO: {self.task['repo']}",
            f"START COMMIT: {self.task['base_commit']}",
            f"ENVIRONMENT COMMIT: {self.task['environment_setup_commit']}",
            f"PROBLEM: {self.task['problem_statement']}"
            f"HINT: {self.task['hints_text']}",
        ]
        return {
            "id": self.task["instance_id"],
            "prompt": "\n".join(lines)
        }
    

class SWEBench(TasksStore[SWEBenchTask]):
    def get_tasks(self) -> List[SWEBenchTask]:
        ds = load_dataset('princeton-nlp/SWE-bench', split='test')
        return cast(List[SWEBenchTask], list(ds))

    def load_task(self, task: SWEBenchTask) -> LoadedTask[SWEBenchTask]:
        return LoadedSWEBenchTask(task)
    
    def custom_instructions(self) -> None | str:
        lines = [
            "You will be solving a GitHub issue.",
            "You are NOT allowed to push code to any repository.",
            "The last thing you should do is output a patch to standard out."
        ]
        return "/n".join(lines)
