from pathlib import Path
import re
import shutil
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
        ds = load_dataset("./.datasets/GAIA", "2023_all", split="validation", data_dir="./datasets", trust_remote_code=True)
        as_list = cast(List[GAIATask], list(ds))
        tasks = [t for t in as_list if t["file_path"] != "" and t["Level"] == "1"]
        n_tasks = first_n or len(tasks)
        return tasks[:n_tasks]
        
    def setup_input_dir(task: GAIATask, dir_path: Path):
        src_file_path = Path(task["file_path"])
        dst_file_path = dir_path / Path(task["file_name"])
        shutil.copyfile(src_file_path, dst_file_path)
    
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
