import io
import csv
import argparse
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Optional

from gaia import GAIABenchmark
from benchmark import DefaultBenchmarkRunner, DockerBenchmarkRunner, TaskResult, run_benchmark_worker_pool
from commands import commands


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
    default_command_id = "gpt35turbo"
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

    if args.ntasks is None:
        b = GAIABenchmark()
    else:
        print("number of tasks:", args.ntasks)
        b = GAIABenchmark(args.ntasks, args.task_offset, [
            # lambda t: t["file_name"] != ""
        ])
    
    command = commands[args.command]
    runner = DockerBenchmarkRunner()
    
    if args.nworkers is None:
        results = run_benchmark_worker_pool(b, command, runner)
    else:
        print("number of workers:", args.nworkers)
        results = run_benchmark_worker_pool(b, command, runner, args.nworkers)

    save_results(results, save_path)
