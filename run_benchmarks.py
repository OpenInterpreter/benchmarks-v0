import io
import csv
import argparse
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Optional

from constants import DATASETS
from custom import CustomTasks
from benchmark import DefaultBenchmarkRunner, DockerBenchmarkRunner, ModifierPipe, OIBenchmarks, SizeOffsetModifier, TaskResult, runners
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
    server: bool
    runner: str
    benchmark: str
    bfile: Optional[str]


task_stores = {
    "gaia": GAIATasks()
}


if __name__ == "__main__":
    default_command_id = ""
    default_output_file_dir = Path(".local/results")
    default_runner = "docker"
    default_benchmark = "gaia"

    parser = argparse.ArgumentParser()
    parser.add_argument("-l", "--list", action="store_true", help="list the possible command configuration ids")
    parser.add_argument("-c", "--command", action="store", type=str, default=default_command_id, help=f"change the command configuration ({', '.join(commands.keys())})")
    parser.add_argument("-nt", "--ntasks", action="store", type=int, help="run the first n tasks for the selected benchmark")
    parser.add_argument("-nw", "--nworkers", action="store", type=int, help="run the benchmarks across n workers (docker containers, processes, E2B instances, etc.)")
    parser.add_argument("-to", "--task-offset", action="store", type=int, default=0)
    parser.add_argument("-s", "--server", action="store_true", help="launch a server that keeps track of and displays task starts, stops, and logging")
    parser.add_argument("-r", "--runner", action="store", type=str, default=default_runner, help=f"the kind of worker to run each task on ({', '.join(runners.keys())})")
    parser.add_argument("-b", "--benchmark", action="store", default=default_benchmark, help=f"where to retreive the list of tasks to run from ({', '.join(task_stores.keys())})")
    parser.add_argument("-bf", "--bfile", action="store", type=str, help="only works when '--benchmark custom' is used")
    args = parser.parse_args(namespace=ArgumentsNamespace())

    if args.list:
        print("possible commands configurations:", list(commands.keys()))
        exit(0)
    if args.command not in commands:
        print(f"'{args.command}' not recognized as a command configuration id")
        print("possible command configuration ids:", list(commands.keys()))
        exit(1)
    if args.runner not in runners:
        print(f"'{args.runner}' not recognized as a runner id")
        print("possible runner ids:", list(runners.keys()))
        exit(1)
    if args.benchmark not in task_stores and args.benchmark != "custom":
        print(f"'{args.benchmark}' not recognized as a benchmark id")
        print("possible benchmark ids:", [*task_stores.keys(), "custom"])
        exit(1)
    if args.benchmark == "custom" and args.bfile is None:
        print(f"'--benchmark custom' can only be used if '--bflag <file-path>' is also used.")
        exit(1)
    tasks = task_stores[args.benchmark] if args.benchmark in task_stores else CustomTasks.from_csv(args.bfile)  # type: ignore
    runner = runners[args.runner]
    
    print("command configuration:", args.command)
    now_utc = datetime.now(timezone.utc)
    save_path = default_output_file_dir / Path(f"{dt_to_str(now_utc)}-{args.command}.csv")
    print("output file:", save_path)

    results = OIBenchmarks(
        tasks=tasks,
        modifier=SizeOffsetModifier(ntasks=args.ntasks, offset=args.task_offset),
        command=commands[args.command],
        nworkers=args.nworkers,
        runner=runner,
        server=args.server
    ).run()

    correct_count = sum(1 for result in results if result['status'] == 'correct')
    print(f"Number of correct results: {correct_count}/{len(results)}")
    save_results(results, save_path)
