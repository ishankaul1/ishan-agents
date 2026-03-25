from abc import ABC, abstractmethod
from pathlib import Path


class Sandbox(ABC):
    def __init__(self, work_dir: str):
        self.work_dir = Path(work_dir).resolve()
        if not self.work_dir.is_dir():
            raise ValueError(f"work_dir does not exist: {self.work_dir}")

    def _resolve(self, path: str) -> Path:
        resolved = (self.work_dir / path).resolve()
        if not resolved.is_relative_to(self.work_dir):
            raise ValueError(f"Path escapes sandbox: {path}")
        return resolved

    @abstractmethod
    def read(self, path: str) -> str: ...

    @abstractmethod
    def write(self, path: str, content: str) -> str: ...

    @abstractmethod
    def bash(self, cmd: str) -> str: ...

    @abstractmethod
    def glob(self, pattern: str) -> list[str]: ...

    @abstractmethod
    def grep(self, pattern: str, path: str) -> str: ...
