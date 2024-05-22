import os
import re
import io
import csv
from typing import TypedDict, Optional, Dict, cast, List
from datasets import load_dataset

from benchmark import Benchmark, OpenInterpreterCommand, ResultStatus, TaskResult, ZeroShotTask, run_benchmark, run_benchmark_threaded, run_benchmark_threaded_pool


GAIATask = TypedDict("GAIATask", {
    "task_id": str,
    "Question": str,
    "Level": str,
    "Final answer": str,
    "file_name": str,
    "file_path": str,
    "Annotator Metadata": Optional[Dict[str, str]]
})


def gaia_benchmark(first_n: Optional[int] = None) -> Benchmark[GAIATask]:
    def get_tasks() -> List[GAIATask]:
        ds = load_dataset("gaia-benchmark/GAIA", "2023_all", split="validation")
        as_list = cast(List[GAIATask], list(ds))
        if first_n is not None:
            return as_list[:first_n]
        else:
            return as_list
        # data = load_dataset("gaia-benchmark/GAIA", "2023_all", split="validation")
        # tfel = [d for d in data if "tfel" in d["Question"]]
        # return tfel
    
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


custom_instructions = """I will ask you a question. Report your thoughts, and finish your answer with the following template: FINAL ANSWER: [YOUR FINAL ANSWER].
YOUR FINAL ANSWER should be a number OR as few words as possible OR a comma separated list of numbers and/or strings.
If you are asked for a number, don’t use comma to write your number neither use units such as $ or percent sign unless specified otherwise.
If you are asked for a string, don’t use articles, neither abbreviations (e.g. for cities), and write the digits in plain text unless specified otherwise.
If you are asked for a comma separated list, apply the above rules depending of whether the element to be put in the list is a number or a string.
"""

openapi_key = os.getenv("OPENAI_API_KEY", "")


commands: Dict[str, OpenInterpreterCommand] = {
    "oai_default": {
        "auto_run": True,
        "api_key": openapi_key,
        "custom_instructions": custom_instructions
    },
    "gpt4o": {
        "auto_run": True,
        "model": "openai/gpt-4o",
        "context_window": 128000,
        "api_key": openapi_key,
        "custom_instructions": custom_instructions
    },
    "gpt4": {
        "auto_run": True,
        "model": "openai/gpt-4",
        "context_window": 8192,
        "api_key": openapi_key,
        "custom_instructions": custom_instructions
    },
    "gpt35turbo": {
        "auto_run": True,
        "model": "openai/gpt-3.5-turbo-0125",
        "context_window": 16385,
        "api_key": openapi_key,
        "custom_instructions": custom_instructions
    },
    "llama3": {
        "auto_run": True,
        "model": "ollama/llama3",
        "context_window": 2048,
        "api_base": "http://192.168.1.86:11434",
        "custom_instructions": custom_instructions
    },
}


def consume_results(results: List[TaskResult]):
    if len(results) > 0:
        f = io.StringIO("")
        with io.StringIO("") as f:
            writer = csv.DictWriter(f, results[0].keys())
            writer.writeheader()
            writer.writerows(results)
            with open("output.csv", "w") as csv_file:
                v = f.getvalue()
                csv_file.write(v)


b = gaia_benchmark()
# results = run_benchmark(b, commands["gpt4"])
results = run_benchmark_threaded_pool(b, commands["gpt35turbo"])
consume_results(results)
