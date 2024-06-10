from abc import ABC, abstractmethod
from asyncio import AbstractEventLoop
from io import RawIOBase
import json
import os
from pathlib import Path
import shlex
import subprocess
from tempfile import TemporaryDirectory
import time
from typing import Any, Callable, List, TypeVar
from rich.console import Console

from fsspec import AbstractFileSystem
from commands import OpenInterpreterCommand
from task import LMC, LoadedTask
from utils import LocalBasedFS
import worker
from e2b import Sandbox
from e2b_desktop import Desktop


Task = TypeVar("Task")


class BenchmarkRunner(ABC):
    @abstractmethod
    def run(self, lt: LoadedTask[Task], command: OpenInterpreterCommand, write: Callable[[bytes], None], should_stop: Callable[[], bool], log: Callable[[str], None]) -> List[LMC]:
        """
        Should stop is a boolean that will return True if something external wants to stop the running process.  It should be checked periodically.
        """
        raise NotImplementedError()


class DefaultBenchmarkRunner(BenchmarkRunner):
    def run(self, lt: LoadedTask[Task], command: OpenInterpreterCommand, write: Callable[[bytes], None], should_stop: Callable[[], bool], log: Callable[[str], None]) -> List[LMC]:
        with TemporaryDirectory() as worker_dir:
            output_dir = Path(worker_dir) / Path("output")
            input_dir = Path(worker_dir) / Path("input")
            input_dir.mkdir(parents=True, exist_ok=True)
            lt.setup_input_dir(LocalBasedFS(str(input_dir)))
            prompt = lt.to_zero_shot()["prompt"]

            command_json_str = json.dumps(command)
            p = subprocess.Popen([
                "python", "-m", "worker.run",
                command_json_str, f"{shlex.quote(prompt)}", worker_dir, output_dir
            ], stdout=subprocess.PIPE, stderr=subprocess.STDOUT)

            assert p.stdout is not None, "stdout stream is None!  Did you forget to set stdout=subprocess.PIPE?"
            while p.poll() is None and not should_stop():
                write(p.stdout.readline())
            
            if should_stop():
                # stopped before completion!
                p.kill()
                log("stopped!")
                return []
            
            # If the process p finishes without a newline, the above loop will not write the rest of the process.
            # Hence the following line.
            write(p.stdout.read())

            messages_path = worker_dir / worker.OUTPUT_PATH
            with open(messages_path, "r") as f:
                messages = json.load(f)
                return messages


class DockerBenchmarkRunner(BenchmarkRunner):
    WORKER_NAME = "worker"

    def run(self, lt: LoadedTask[Task], command: OpenInterpreterCommand, write: Callable[[bytes], None], should_stop: Callable[[], bool], log: Callable[[str], None]) -> List[LMC]:
        with TemporaryDirectory() as worker_dir:
            output_dir = Path(worker_dir) / Path("output")
            input_dir = Path(worker_dir) / Path("input")
            command_json_str = json.dumps(command)
            input_dir.mkdir(parents=True, exist_ok=True)
            output_dir.mkdir(parents=True, exist_ok=True)
            lt.setup_input_dir(LocalBasedFS(str(input_dir)))
            zs = lt.to_zero_shot()
            container_name = f"{zs["id"]}_{time.time()}"
            dcmd = [
                "docker", "run", "-t",
                "-v", f"{input_dir}:/input", "-v", f"{output_dir}:/output",
                "--name", container_name,
                DockerBenchmarkRunner.WORKER_NAME,
                command_json_str, f"{shlex.quote(zs["prompt"])}", "/", "/output"
            ]

            p = subprocess.Popen(dcmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL)
            assert p.stdout is not None

            while p.poll() is None and not should_stop():
                write(p.stdout.readline())
            
            if should_stop():
                # stopped before completion!
                log("stopped!")
                p.kill()
                return []
            
            write(p.stdout.read())

            messages_path = output_dir / worker.OUTPUT_PATH
            if not messages_path.exists():
                # this will happen if (for example) the container is stopped before it finishes.
                log(f"couldn't find {messages_path}!")
                return []
            with open(messages_path) as f:
                messages = json.load(f)
                return messages


class E2BFile(RawIOBase):
    def __init__(self, sandbox: Sandbox, path: os.PathLike, mode: str):
        self.sandbox = sandbox
        self.path = path
        self.mode = mode
    
    def read(self, size=-1) -> bytes:
        bs = self.sandbox.download_file(str(self.path))
        if size == -1:
            return bs
        return bs[:size]
    
    def readinto(self, b):
        bs = self.sandbox.download_file(str(self.path))
        b[:len(bs)] = bs
        return bs

    def write(self, b):
        self.sandbox.filesystem.write_bytes(str(self.path), b)
        return len(b)
    
    def readable(self) -> bool:
        return "r" in self.mode or "+" in self.mode

    def writable(self) -> bool:
        return "w" in self.mode or "+" in self.mode or "a" in self.mode


class E2BFilesystem(AbstractFileSystem):
    def __init__(self, sandbox: Sandbox, base: os.PathLike = Path("")):
        self.sandbox = sandbox
        self.base = base

    def _full_path(self, path: str) -> str:
        return f"{self.sandbox.get_hostname()}/{self.base}/{path}"
    
    def open(self, path: str, mode: str = 'r', **kwargs: Any) -> Any:
        return E2BFile(self.sandbox, self.base / Path(path), mode)
    
    def ls(self, path='', detail=True, **kwargs):
        return self.sandbox.filesystem.list(path)


class E2BTerminalBenchmarkRunner(BenchmarkRunner):
    rc = Console()

    # default timeout is 10 minutes.
    def __init__(self, timeout: int = 10 * 60 * 1000):
        self._timeout = timeout

    def run_cmd_blocking(self, sandbox: Sandbox, cmd: str, write: Callable[[bytes], None], should_stop: Callable[[], bool], log: Callable[[str], None], display: bool = True):
        if display:
            p = sandbox.process.start(
                cmd,
                on_stdout=lambda output: write(f"{output.line}\n".encode("utf-8")),
                on_stderr=lambda output: write(f"{output.line}\n".encode("utf-8")),
                timeout=self._timeout
            )
            while p.exit_code is None and not should_stop():
                time.sleep(1)
            if should_stop():
                log("stopped!")
                p.kill()
        else:
            sandbox.process.start_and_wait(cmd)

    def run(self, lt: LoadedTask[Task], command: OpenInterpreterCommand, write: Callable[[bytes], None], should_stop: Callable[[], bool], log: Callable[[str], None]) -> List[LMC]:
        with Sandbox(template="worker", cwd="/") as sandbox:
            input_dir = "/input"
            output_dir = "/output"
            sandbox.filesystem.make_dir(input_dir)
            sandbox.filesystem.make_dir(output_dir)
            fs = E2BFilesystem(sandbox, Path("/input"))
            lt.setup_input_dir(fs)
            prompt = lt.to_zero_shot()["prompt"]
            command_json_str = json.dumps(command)
            log(f"sandbox id: {sandbox.id}")
            self.run_cmd_blocking(
                sandbox,
                f"sudo python -m worker.run '{command_json_str}' {shlex.quote(prompt)} '/' {output_dir}",
                write,
                should_stop,
                log,
                display=True)
            try:
                messages_file = sandbox.download_file(str(Path(output_dir) / worker.OUTPUT_PATH))
                messages_str = messages_file.decode()
                return json.loads(messages_str)
            except Exception as e:  # download file doesn't throw an exception any less generic than Exception.
                log(str(e))
                return []


class E2BDesktopBenchmarkRunner(BenchmarkRunner):
    def run(self, lt: LoadedTask[Task], command: OpenInterpreterCommand, write: Callable[[bytes], None], should_stop: Callable[[], bool], log: Callable[[str], None]) -> List[LMC]:
        with Desktop(template="screen") as desktop:
            desktop.screenshot("screenshot-1.png")

            # Create file and open text editor
            file = "/home/user/test.txt"
            desktop.filesystem.write(file, "world!")
            
            # Normally, we would use `desktop.process.start_and_wait()` to run a new process
            # and wait until it finishes.
            # However, the mousepad command does not exit until you close the window so we
            # we need to just start the process and run it in the background so it doesn't
            # block our code.
            mousepad = desktop.process.start(
                f"mousepad {file}",
                env_vars={"DISPLAY": desktop.DISPLAY},
                on_stderr=lambda stderr: print(stderr),
                on_stdout=lambda stdout: print(stdout),
                cwd="/home/user",
            )
            time.sleep(2)  
            #####

            desktop.screenshot("screenshot-2.png")

            # Write "Hello, " in the text editor
            desktop.pyautogui(
                """
        pyautogui.write("Hello, ")
        """
            )
            desktop.screenshot("screenshot-3.png")
            mousepad.kill()
            desktop.screenshot("screenshot-4.png")

            desktop.process.start_and_wait(
                "python3 -c 'import interpreter'",
                on_stdout=lambda output: print("screen:", output.line),
                on_stderr=lambda output: print("screen:", output.line)
            )

        return []
