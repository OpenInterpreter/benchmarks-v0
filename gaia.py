import re
from typing import Callable, Dict, List, Optional, TypedDict, cast
from datasets import load_dataset
from fsspec import AbstractFileSystem, filesystem

from benchmark import Benchmark, ResultStatus, ZeroShotTask
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


def benchmark(first_n: Optional[int] = None, offset: int = 0, predicates: List[Callable[[GAIATask], bool]] = []) -> Benchmark[GAIATask]:
    def get_tasks() -> List[GAIATask]:
        ds = load_dataset(str(GAIA), "2023_all", split="validation", data_dir=str(DATASETS), trust_remote_code=True)
        tasks = cast(List[GAIATask], list(ds))
        n_tasks = first_n or len(tasks)
        return wrapping_offset([t for t in tasks if all(p(t) for p in predicates)], offset, n_tasks)
        
    def setup_input_dir(task: GAIATask, fs: AbstractFileSystem):
        if task["file_path"] == "":
            return
        local_fs = filesystem("file")
        copy_between_fss(local_fs, task["file_path"], fs, task["file_name"])
    
    def task_to_id_prompt(task: GAIATask) -> ZeroShotTask:
        file_path = f"input/{task['file_name']}"
        prompt = task["Question"] if task["file_name"] == "" else f"file_path: {file_path}\n{task['Question']}"
        return {"id": task["task_id"], "prompt": prompt}
    
    def task_result_status(task: GAIATask, messages: List[Dict[str, str]]) -> ResultStatus:
        if len(messages) == 0:
            return "unknown"
        final_message = messages[-1]
        if "role" not in final_message:
            return "unknown"
        if final_message["role"] == "error":
            return "error"
        if "content" not in final_message:
            return "unknown"
        final_answer_re = re.search("FINAL ANSWER: (.+)", final_message["content"])
        if final_answer_re is None:
            return "unknown"

        expected = task["Final answer"]
        actual = final_answer_re.group(1).strip()
        if actual.lower() != expected.lower():
            return "incorrect"
        else:
            return "correct"

    return Benchmark(
        get_tasks,
        setup_input_dir,
        task_to_id_prompt,
        task_result_status
    )
