import io
import csv
import argparse
from typing import List, Optional

import gaia
from benchmark import TaskResult, run_benchmark_threaded_pool
from commands import commands


def save_results(results: List[TaskResult], filepath: str = "output.csv"):
    if len(results) > 0:
        f = io.StringIO("")
        with io.StringIO("") as f:
            writer = csv.DictWriter(f, results[0].keys())
            writer.writeheader()
            writer.writerows(results)
            with open(filepath, "w") as csv_file:
                v = f.getvalue()
                csv_file.write(v)


class ArgumentsNamespace(argparse.Namespace):
    list: bool
    command: str
    output: str
    ntasks: Optional[int]
    nworkers: Optional[int]


if __name__ == "__main__":
    default_command_id = "gpt35turbo"
    default_output_filepath = "output.csv"

    parser = argparse.ArgumentParser()
    parser.add_argument("-l", "--list", action="store_true")
    parser.add_argument("-c", "--command", action="store", type=str, default=default_command_id)
    parser.add_argument("-o", "--output", action="store", type=str, default=default_output_filepath)
    parser.add_argument("-nt", "--ntasks", action="store", type=int)
    parser.add_argument("-nw", "--nworkers", action="store", type=int)
    args = parser.parse_args(namespace=ArgumentsNamespace())

    print("args", args)

    if args.list:
        print("possible commands configurations:", commands.keys())
        exit(0)
    if args.command not in commands:
        print(f"'{args.command}' not recognized as a command configuration id")
        exit(1)
    
    print("command configuration:", args.command)
    print("output file:", args.output)

    b = gaia.benchmark(8)
    results = run_benchmark_threaded_pool(b, args.command)
    save_results(results, args.output)
