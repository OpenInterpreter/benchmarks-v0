This repo is used to run various AI benchmarks on the [Open Interpreter project](https://github.com/OpenInterpreter/open-interpreter).  Only [GAIA](https://huggingface.co/gaia-benchmark) is currently supported (although image tasks are broken I'll fix it soon promise).

---

## Setup

1. Make sure the following software is installed on your computer.

- [Git](https://git-scm.com)
- [Git-LFS](https://git-lfs.com)
- [Python](https://www.python.org)
- [Docker](https://www.docker.com/)

2. Copy-paste the following lines into your terminal if you're feeling dangerous.

```bash
git clone https://github.com/OpenInterpreter/benchmarks.git \
  && cd oi-benchmarks \
  && python -m venv .venv \
  && source .venv/bin/activate \
  && python -m pip install -r requirements.txt \
  && docker build -t worker . \
  && python setup.py
```

## Running Benchmarks

This section assumes:
- `oi-benchmarks` (downloaded via git in the preview section) is set as the current working directory.
- You've activated the virtualenv with the installed prerequisite packages.
- Your `OPENAI_API_KEY` environment variable is set with a valid OpenAI API key.

### Example: gpt-3.5-turbo, first 16 GAIA tasks, 8 docker containers

This command will output a file called `output.csv` containing the results of the benchmark.

```bash
python run_benchmarks.py \
  --command gpt35turbo \
  --ntasks 16 \
  --nworkers 8
```

- `--command gpt35turbo`: Replace gpt35turbo with any existing key in the commands `Dict` in commands.py.  Defaults to gpt35turbo.
- `--ntasks 16`: Grabs the first 16 GAIA tasks to run.  Defaults to all 165 GAIA validation tasks.
- `--nworkers 8`: Number of docker containers to run at once.  Defaults to whatever max_workers defaults to when constructing a ThreadPoolExecutor.

## Troubleshooting

- `ModuleNotFoundError: No module named '_lzma'` when running example.
    - If you're using `pyenv` to manage python versions, [this stackoverflow post](https://stackoverflow.com/questions/59690698/modulenotfounderror-no-module-named-lzma-when-building-python-using-pyenv-on) might help.
- `ModuleNotFoundError: No module named 'pkg_resources'` when running example.
    - Refer to [this stackoverflow post](https://stackoverflow.com/questions/7446187/no-module-named-pkg-resources) for now.
    - OpenInterpreter should probably include `setuptools` in its list of dependencies, or should switch to another module that's in python's standard library.
