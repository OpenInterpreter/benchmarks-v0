import subprocess
from pathlib import Path


"""
This script is responsible for downloading benchmarks.  It's imperative that the files this script
downloads are NOT committed to this repository.
"""


color = lambda c: lambda s: f"\033[{c}m{s}\033[0m"
red = color(91)
green = color(92)


def run_cmd(cmd: str):
    subprocess.run(cmd.split(" "), check=True)


print(green("logging into huggingface"))
run_cmd("huggingface-cli login")

datasets = Path(".datasets")
if not datasets.exists():
    print(green("creating .datasets directory"))
    datasets.mkdir(parents=True)

if not Path(".datasets/GAIA").exists():
    print(green("downloading gaia"))
    run_cmd("git clone https://huggingface.co/datasets/gaia-benchmark/GAIA .datasets/GAIA")
