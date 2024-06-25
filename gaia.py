from typing import Dict, List, Optional, TypedDict, cast
from datasets import load_dataset
from fsspec import AbstractFileSystem, filesystem

from coordinators import LMC, TasksStore, LoadedTask, ResultStatus, TaskSetModifier, ZeroShotTask, judge_result
from constants import DATASETS, GAIA
from environment import Environment
from utils import copy_between_fss


GAIATask = TypedDict("GAIATask", {
    "task_id": str,
    "Question": str,
    "Level": str,
    "Final answer": str,
    "file_name": str,
    "file_path": str,
    "Annotator Metadata": Optional[Dict[str, str]]
})


class LoadedGAIATask(LoadedTask[GAIATask]):
    def setup_input_dir(self, fs: AbstractFileSystem):
        if self.task["file_path"] == "":
            return
        local_fs = filesystem("file")
        copy_between_fss(local_fs, self.task["file_path"], fs, self.task["file_name"])
    
    def setup_env(self, env: Environment):
        self.setup_input_dir(env.get_fs())
    
    def to_zero_shot(self) -> ZeroShotTask:
        file_path = self.task['file_name']
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

    def judge(self, env: Environment, messages: List[LMC]) -> ResultStatus:
        return self.to_result_status(messages)


class GAIATasks(TasksStore[GAIATask]):
    def get_tasks(self) -> List[GAIATask]:
        ds = load_dataset(str(GAIA), "2023_all", split="validation", data_dir=str(DATASETS), trust_remote_code=True)
        tasks = cast(List[GAIATask], list(ds))
        return tasks
        
    def load_task(self, task: GAIATask) -> LoadedTask[GAIATask]:
        return LoadedGAIATask(task)

    def custom_instructions(self) -> None | str:
        return """I will ask you a question. Report your thoughts, and finish your answer with the following template: FINAL ANSWER: [YOUR FINAL ANSWER].
YOUR FINAL ANSWER should be a number OR as few words as possible OR a comma separated list of numbers and/or strings.
If you are asked for a number, don’t use comma to write your number neither use units such as $ or percent sign unless specified otherwise.
If you are asked for a string, don’t use articles, neither abbreviations (e.g. for cities), and write the digits in plain text unless specified otherwise.
If you are asked for a comma separated list, apply the above rules depending of whether the element to be put in the list is a number or a string.
        """


class GAIAFilesOnlyModifier(TaskSetModifier[GAIATask]):
    def modify(self, task_set: List[GAIATask]) -> List[GAIATask]:
        return [t for t in task_set if t["file_name"] != ""]
