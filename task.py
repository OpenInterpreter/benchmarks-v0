from abc import abstractmethod
from dataclasses import dataclass
from datetime import datetime
from typing import Dict, Generic, List, Literal, Optional, TypeVar, TypedDict

from fsspec import AbstractFileSystem

from commands import OpenInterpreterCommand


Task = TypeVar("Task")
LMC = Dict[str, str]
ResultStatus = Literal["correct", "incorrect", "unknown", "error"]


class ZeroShotTask(TypedDict):
    id: str
    prompt: str


class TaskResult(TypedDict):
    task_id: str
    command: OpenInterpreterCommand
    prompt: str
    start: Optional[datetime]
    end: Optional[datetime]
    messages: List[LMC]
    status: ResultStatus


@dataclass(frozen=True)
class LoadedTask(Generic[Task]):
    task: Task

    @abstractmethod
    def setup_input_dir(self, fs: AbstractFileSystem):
        ...
    
    @abstractmethod
    def to_zero_shot(self) -> ZeroShotTask:
        raise NotImplementedError()
    
    @abstractmethod
    def to_result_status(self, messages: List[LMC]) -> ResultStatus:
        raise NotImplementedError()
   

class TasksStore(Generic[Task]):
    @abstractmethod
    def get_tasks(self) -> List[Task]:
        raise NotImplementedError()

    @abstractmethod
    def load_task(self, task: Task) -> LoadedTask[Task]:
        raise NotImplementedError()
    
    def custom_instructions(self) -> None | str:
        return None
