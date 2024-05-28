from typing import Any
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
