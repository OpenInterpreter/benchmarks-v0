import re
from typing import Dict, List, Optional, TypedDict, cast
from datasets import load_dataset

from benchmark import Benchmark, ResultStatus, ZeroShotTask


GAIATask = TypedDict("GAIATask", {
    "task_id": str,
    "Question": str,
    "Level": str,
    "Final answer": str,
    "file_name": str,
    "file_path": str,
    "Annotator Metadata": Optional[Dict[str, str]]
})


def benchmark(first_n: Optional[int] = None) -> Benchmark[GAIATask]:
    def get_tasks() -> List[GAIATask]:
        ds = load_dataset("gaia-benchmark/GAIA", "2023_all", split="validation")
        as_list = cast(List[GAIATask], list(ds))
        if first_n is not None:
            return as_list[:first_n]
        else:
            return as_list
    
    def task_to_id_prompt(task: GAIATask) -> ZeroShotTask:
        file_path = f"files/{task['file_name']}"
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
        task_to_id_prompt,
        task_result_status
    )
