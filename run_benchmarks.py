import io
import csv
import argparse
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Optional

from constants import DATASETS
from custom import CustomTasks
from benchmark import DefaultBenchmarkRunner, DockerBenchmarkRunner, ModifierPipe, OIBenchmarks, SizeOffsetModifier, TaskResult
from commands import commands
from gaia import GAIAFilesOnlyModifier, GAIATasks


def save_results(results: List[TaskResult], filepath: Path):
    if len(results) > 0:
        f = io.StringIO("")
        if not filepath.parent.exists():
            filepath.parent.mkdir(parents=True, exist_ok=True)
        with io.StringIO("") as f:
            writer = csv.DictWriter(f, results[0].keys())
            writer.writeheader()
            writer.writerows(results)
            with open(filepath, "w") as csv_file:
                v = f.getvalue()
                csv_file.write(v)


def dt_to_str(dt: datetime) -> str:
    return dt.strftime("%Y-%m-%dT%H-%M-%SZ")


class ArgumentsNamespace(argparse.Namespace):
    list: bool
    command: str
    output: str
    ntasks: Optional[int]
    task_offset: int
    nworkers: Optional[int]


if __name__ == "__main__":
    default_command_id = ""
    default_output_file_dir = Path(".local/results")

    parser = argparse.ArgumentParser()
    parser.add_argument("-l", "--list", action="store_true")
    parser.add_argument("-c", "--command", action="store", type=str, default=default_command_id)
    parser.add_argument("-nt", "--ntasks", action="store", type=int)
    parser.add_argument("-nw", "--nworkers", action="store", type=int)
    parser.add_argument("-to", "--task-offset", action="store", type=int, default=0)
    args = parser.parse_args(namespace=ArgumentsNamespace())

    if args.list:
        print("possible commands configurations:", list(commands.keys()))
        exit(0)
    if args.command not in commands:
        print(f"'{args.command}' not recognized as a command configuration id")
        exit(1)
    
    print("command configuration:", args.command)
    now_utc = datetime.now(timezone.utc)
    save_path = default_output_file_dir / Path(f"{dt_to_str(now_utc)}-{args.command}.csv")
    print("output file:", save_path)

    results = OIBenchmarks(
        # tasks=CustomTasks.from_list([
        #     {"id": "simple", "prompt": "what is 3 + 4?", "answer": "7"},
        #     {"id": "hard", "prompt": "who do you think you are??", "answer": "laptop"},
        # ]),
        tasks=GAIATasks(),
        modifier=ModifierPipe([
            GAIAFilesOnlyModifier(),
            SizeOffsetModifier(ntasks=args.ntasks, offset=args.task_offset)
        ]),
        # modifier=SizeOffsetModifier(ntasks=args.ntasks, offset=args.task_offset),
        command=commands[args.command],
        nworkers=args.nworkers,
        runner=DefaultBenchmarkRunner()
    ).run()

    correct_count = sum(1 for result in results if result['status'] == 'correct')
    print(f"Number of correct results: {correct_count}/{len(results)}")
    save_results(results, save_path)
