from contextlib import contextmanager
import os
from pathlib import Path
from typing import Any, List, TypeVar
from fsspec import AbstractFileSystem, filesystem


class LocalBasedFS(AbstractFileSystem):
    def __init__(self, base: str):
        self.base = base
        self.fs: AbstractFileSystem = filesystem("file")

    def _full_path(self, path: str) -> str:
        return f"{self.base}/{path}"
    
    def open(self, path: str, mode: str = 'r', **kwargs: Any) -> Any:
        return self.fs.open(self._full_path(path), mode=mode, **kwargs)
    
    def ls(self, path='', detail=True, **kwargs):
        return self.fs.ls(self._full_path(path), detail=detail, **kwargs)


def copy_between_fss(src_fs: AbstractFileSystem, src_path: str, dst_fs: AbstractFileSystem, dst_path: str):
    with src_fs.open(src_path, "rb") as src_f:
        with dst_fs.open(dst_path, "wb") as dst_f:
            src = src_f.read()
            dst_f.write(src)  # type: ignore


@contextmanager
def change_working_dir(new_wd: Path | str):
    old_wd = os.getcwd()
    os.chdir(new_wd)
    yield
    os.chdir(old_wd)


T = TypeVar("T")
def wrapping_offset(a: List[T], offset: int, n: int) -> List[T]:
    """
    Will grab the first n elements of a rotated by offset.
    """
    if len(a) == 0 or n == 0:
        return []
    
    actual_n = min(len(a), n)
    actual_offset = offset % len(a)

    return (a[actual_offset:] + a[:actual_offset])[:actual_n]
