from abc import ABC, abstractmethod
from dataclasses import dataclass
from io import RawIOBase, StringIO
import json
import os
from pathlib import Path
import shlex
import subprocess
from tempfile import TemporaryDirectory
import time
from typing import Any, Callable, List, TypeVar, cast
from rich.console import Console
from websockets import ConnectionClosed
from websockets.sync.client import connect
from websockets.sync.connection import Connection
from fsspec import AbstractFileSystem
from websocket import WebSocket
from e2b import Sandbox, SandboxException
from e2b_desktop import Desktop

import worker
from accumulator import Accumulator
from commands import OpenInterpreterCommand
from task import LMC, LoadedTask
from utils import LocalBasedFS


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

            messages_path = output_dir / worker.OUTPUT_PATH
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
            container_name = f"{zs['id']}_{time.time()}"
            dcmd = [
                "docker", "run", "-t",
                "-v", f"{input_dir}:/input", "-v", f"{output_dir}:/output",
                "--name", container_name,
                DockerBenchmarkRunner.WORKER_NAME,
                command_json_str, f"{shlex.quote(zs['prompt'])}", "/", "/output"
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


def get_free_port():
    # sort of cursed but its fine.
    # basis from https://stackoverflow.com/questions/1365265/on-localhost-how-do-i-pick-a-free-port-number.
    import socket
    with socket.socket() as sock:
        sock.bind(('', 0))
        return sock.getsockname()[1]


class FakeBenchmarkRunner(BenchmarkRunner):
    def run(self, lt, command, write, should_stop, log) -> List[LMC]:
        # write(b"how could you.")
        return [{"role": "assistant", "type": "message", "content": "im sowwy."}]


class DockerServerBenchmarkRunner(BenchmarkRunner):
    WORKER_NAME = "server-worker"

    def run(self, lt: LoadedTask[Task], command: OpenInterpreterCommand, write: Callable[[bytes], None], should_stop: Callable[[], bool], log: Callable[[str], None]) -> List[LMC]:
        with TemporaryDirectory() as worker_dir:
            output_dir = Path(worker_dir) / Path("output")
            input_dir = Path(worker_dir) / Path("input")
            command_json_str = json.dumps(command)
            input_dir.mkdir(parents=True, exist_ok=True)
            output_dir.mkdir(parents=True, exist_ok=True)
            lt.setup_input_dir(LocalBasedFS(str(input_dir)))
            zs = lt.to_zero_shot()
            container_name = f"{zs['id']}_{time.time()}"
            port = get_free_port()
            dcmd = [
                "docker", "run", "-t",
                "-v", f"{input_dir}:/input", "-v", f"{output_dir}:/output",
                "--name", container_name,
                "-p", f"{port}:8000",
                DockerServerBenchmarkRunner.WORKER_NAME,
                command_json_str, "/"
            ]

            p = subprocess.Popen(dcmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL)
            assert p.stdout is not None

            while p.poll() is None and "Uvicorn running" not in p.stdout.readline().decode("utf-8"):
                time.sleep(0.5)
            
            def is_done(msg: LMC) -> bool:
                return (
                    "role" in msg and msg["role"] == "server"
                    and "type" in msg and msg["type"] == "completion"
                    and "content" in msg and "DONE" in msg["content"]
                )
            
            def recv(c: Connection, timeout: int) -> LMC | None:
                try:
                    return cast(LMC, json.loads(c.recv(timeout)))
                except ConnectionClosed:
                    return None

            messages: List[LMC] = []

            with connect(f"ws://localhost:{port}") as c:
                c.send(json.dumps({"role": "user", "type": "message", "start": True}))
                c.send(json.dumps({"role": "user", "type": "message", "content": zs["prompt"]}))
                c.send(json.dumps({"role": "user", "type": "message", "end": True}))

                timeout = 10 * 60
                current_msg = recv(c, timeout)
                acc = Accumulator()
                while current_msg is not None and not is_done(current_msg):
                    if current_msg["role"] != "server" and "content" in current_msg and isinstance(current_msg["content"], str):
                        write(str(current_msg["content"]).encode("utf-8"))
                    messages.append(current_msg)
                    full_msg = acc.accumulate(current_msg)
                    if full_msg is not None:
                        messages.append(full_msg)
                        acc = Accumulator()
                    current_msg = recv(c, timeout)

            # log("stopping...")
            p.kill()
            subprocess.run(["docker", "stop", container_name], stdout=subprocess.DEVNULL)
            # log("stopped!")
            
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


class WorkerRunner(ABC):
    @abstractmethod
    def run(self):
        ...


class E2BTerminalBenchmarkRunner(BenchmarkRunner):
    rc = Console()
    limit = 18

    # default timeout is 10 minutes.
    def __init__(self, wr: WorkerRunner, timeout: int = 10 * 60 * 1000):
        self._timeout = timeout

    def run_cmd_blocking(self, sandbox: Sandbox, cmd: str, write: Callable[[bytes], None], should_stop: Callable[[], bool], log: Callable[[str], None], display: bool = True):
        if display:
            try:
                p = sandbox.process.start_and_wait(
                    cmd,
                    on_stdout=lambda output: write(f"{output.line}\n".encode("utf-8")),
                    on_stderr=lambda output: write(f"{output.line}\n".encode("utf-8")),
                    timeout=self._timeout
                )
            except SandboxException as e:
                print("e", str(e))
        else:
            sandbox.process.start_and_wait(cmd)

    def run(self, lt: LoadedTask[Task], command: OpenInterpreterCommand, write: Callable[[bytes], None], should_stop: Callable[[], bool], log: Callable[[str], None]) -> List[LMC]:
        messages = []
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
                messages.extend(json.loads(messages_str))
            except Exception as e:  # download file doesn't throw an exception any less generic than Exception.
                log(str(e))
            finally:
                log("closing!")
        Sandbox.kill(sandbox_id=sandbox.id)
        return messages


class E2BServerTerminalBenchmarkRunner(BenchmarkRunner):
    WORKER_NAME = "server-worker"

    def __init__(self, timeout=10):
        self._timeout = timeout

    def run_cmd_blocking(self, sandbox: Sandbox, cmd: str, write: Callable[[bytes], None], should_stop: Callable[[], bool], log: Callable[[str], None], display: bool = True):
        if display:
            try:
                p = sandbox.process.start_and_wait(
                    cmd,
                    on_stdout=lambda output: write(f"{output.line}\n".encode("utf-8")),
                    on_stderr=lambda output: write(f"{output.line}\n".encode("utf-8")),
                    timeout=self._timeout
                )
            except SandboxException as e:
                print("e", str(e))
        else:
            sandbox.process.start_and_wait(cmd)

    def run(self, lt: LoadedTask[Task], command: OpenInterpreterCommand, write: Callable[[bytes], None], should_stop: Callable[[], bool], log: Callable[[str], None]) -> List[LMC]:
        messages: List[LMC] = []
        with Sandbox(template="server-worker", cwd="/") as sandbox:
            input_dir = "/input"
            output_dir = "/output"
            sandbox.filesystem.make_dir(input_dir)
            sandbox.filesystem.make_dir(output_dir)
            fs = E2BFilesystem(sandbox, Path("/input"))
            lt.setup_input_dir(fs)
            zs = lt.to_zero_shot()
            prompt = zs["prompt"]
            command_json_str = json.dumps(command)
            log(f"sandbox id: {sandbox.id}")

            def is_done(msg: LMC) -> bool:
                return (
                    "role" in msg and msg["role"] == "server"
                    and "type" in msg and msg["type"] == "completion"
                    and "content" in msg and "DONE" in msg["content"]
                )

            def recv(c: Connection, timeout: int) -> LMC | None:
                try:
                    return cast(LMC, json.loads(c.recv(timeout)))
                except ConnectionClosed:
                    return None
                
            strio = StringIO("")
            started = False

            def write_out(s: str):
                if not started:
                    strio.write(s)
                    write(s.encode("utf-8"))

            p = sandbox.process.start(
                f"sudo python -m worker.run '{command_json_str}' '/'",
                on_stdout=lambda output: write_out(output.line),
                on_stderr=lambda output: write_out(output.line),
                timeout=self._timeout
            )

            while "Uvicorn running" not in strio.getvalue():
                time.sleep(1)

            started = True
            hn = f"wss://{sandbox.get_hostname(port=8000)}"

            with connect(hn, open_timeout=2 * 60) as c:
                write(b"\n\n")
                log(f"connected to hostname '{hn}'!")
                c.send(json.dumps({"role": "user", "type": "message", "start": True}))
                c.send(json.dumps({"role": "user", "type": "message", "content": prompt}))
                c.send(json.dumps({"role": "user", "type": "message", "end": True}))

                timeout = 4 * 60
                current_msg = recv(c, timeout)
                acc = Accumulator()
                while current_msg is not None and not is_done(current_msg):
                    if current_msg["role"] != "server" and "content" in current_msg and isinstance(current_msg["content"], str):
                        write(str(current_msg["content"]).encode("utf-8"))
                    messages.append(current_msg)
                    full_msg = acc.accumulate(current_msg)
                    if full_msg is not None:
                        messages.append(full_msg)
                        acc = Accumulator()
                    current_msg = recv(c, timeout)

            p.kill()
            p.wait()

        Sandbox.kill(sandbox_id=sandbox.id)
        return messages


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
