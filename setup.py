import subprocess
from pathlib import Path


color = lambda c: lambda s: f"\\033[{c}m{s}\\033[0m"
red = color(91)
green = color(92)


def run_cmd(cmd: str):
    subprocess.run(cmd.split(" "), stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True)


print(green("logging into huggingface"))
run_cmd("huggingface-cli login")

if not Path(".datasets"):
    print(green("creating .datasets directory"))
    run_cmd("mkdir -p .datasets")

if not Path(".datasets/GAIA").exists():
    print(green("downloading gaia"))
    run_cmd("git clone https://huggingface.co/datasets/gaia-benchmark/GAIA .datasets/GAIA")
