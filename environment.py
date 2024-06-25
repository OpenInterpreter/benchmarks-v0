import subprocess
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Callable, Iterator, List, Tuple

from fsspec import AbstractFileSystem


class Executing(ABC):
    @abstractmethod
    def wait(self):
        raise NotImplementedError()

    @abstractmethod
    def wait_with(self, stdout: Callable[[bytes], None] | None = None, stderr: Callable[[bytes], None] | None = None):
        raise NotImplementedError()

    @abstractmethod
    def communicate(self, stdin: bytes, timeout: float | None = None):
        raise NotImplementedError()

    @abstractmethod
    def kill(self):
        raise NotImplementedError()
    
    @abstractmethod
    def generate_lines(self) -> Iterator[bytes]:
        raise NotImplementedError()


class PopenExecuting(Executing):
    def __init__(self, p: subprocess.Popen):
        self._p = p

    def wait(self):
        return self._p.wait()

    def wait_with(self, stdout = None, stderr = None):
        while self._p.poll() is None:
            if self._p.stdout is not None and stdout is not None:
                stdout(self._p.stdout.readline())
            if self._p.stderr is not None and stderr is not None:
                stderr(self._p.stderr.readline())
        if self._p.stdout is not None and stdout is not None:
            stdout(self._p.stdout.read())
        if self._p.stderr is not None and stderr is not None:
            stderr(self._p.stderr.read())

    def communicate(self, stdin: bytes, timeout: float | None = None):
        return self._p.communicate(stdin, timeout)

    def kill(self):
        self._p.kill()
    
    def generate_lines(self) -> Iterator[bytes]:
        while self._p.poll() is None:
            if self._p.stdout is not None:
                yield self._p.stdout.readline()
        if self._p.stdout is not None:
            yield self._p.stdout.read()


class Environment(ABC):
    @abstractmethod
    def exec(self, cmd: List[str], out: Callable[[bytes], None] | None = None) -> Executing:
        raise NotImplementedError()

    @abstractmethod
    def get_fs(self) -> AbstractFileSystem:
        raise NotImplementedError()


@dataclass(frozen=True)
class StructuredEnvironment(Environment):
    fs: AbstractFileSystem
    exec_f: Callable[[List[str]], Executing]

    def exec(self, cmd: List[str]) -> Executing:
        return self.exec_f(cmd)

    def get_fs(self) -> AbstractFileSystem:
        return self.fs
