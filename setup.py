import sys
import subprocess
import shutil
from pathlib import Path


color = lambda c: lambda s: f"\\033[{c}m{s}\\033[0m"
red = color(91)
green = color(92)

def run_cmd(description: str, cmd: str, iff: bool = True):
    if iff:
        print(green(description))
        subprocess.run(cmd.split(" "), stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True)

def program_exists(name: str) -> bool:
    return shutil.which(name) is not None

def in_venv() -> bool:
    return sys.prefix != sys.base_prefix


assert program_exists("docker"), "Install docker before re-running this script."
assert program_exists("git"), "Install git before re-running this script."

run_cmd("creating virtual environment", "python -m venv .venv", iff=not Path(".venv").exists())
run_cmd("activating virtual environment", "source .venv/bin/activate", iff=not in_venv())
run_cmd("installing requirements", "python -m pip install -r requirements.txt")
run_cmd("logging into huggingface", "huggingface-cli login")
run_cmd("creating .datasets directory", "mkdir -p .datasets", iff=not Path(".datasets").exists())
run_cmd("downloading gaia", "git clone https://huggingface.co/datasets/gaia-benchmark/GAIA .datasets/GAIA", iff=not Path(".datasets/GAIA").exists())
run_cmd("building docker container", "docker build -t worker .")
