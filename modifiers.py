from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Callable, Generic, List, Optional, TypeVar

from utils import wrapping_offset


Task = TypeVar("Task")


class TaskSetModifier(ABC, Generic[Task]):
    @abstractmethod
    def modify(self, task_set: List[Task]) -> List[Task]:
        ...


class IdModifier(TaskSetModifier[Task]):
    def modify(self, task_set: List[Task]) -> List[Task]:
        return task_set


class PredModifier(TaskSetModifier[Task]):
    def __init__(self, pred: Callable[[Task], bool]):
        self._pred = pred

    def modify(self, task_set: List[Task]) -> List[Task]:
        return list(filter(self._pred, task_set))


@dataclass
class SizeOffsetModifier(TaskSetModifier[Task]):
    offset: int
    ntasks: Optional[int]

    def modify(self, task_set: List[Task]) -> List[Task]:
        return wrapping_offset([t for t in task_set], self.offset, self.ntasks or len(task_set))


@dataclass
class ModifierPipe(Generic[Task], TaskSetModifier[Task]):
    mods: List[TaskSetModifier[Task]]

    def modify(self, task_set: List[Task]) -> List[Task]:
        current = task_set
        for mod in self.mods:
            if len(current) == 0:
                break
            current = mod.modify(current)
        return current


