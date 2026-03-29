from abc import ABC, abstractmethod

from ishan_agents.sandbox.base import Sandbox


class BaseTool(ABC):
    name: str
    namespace: str  # e.g. "claude_code", "openai" — used for logging
    description: str
    parameters: dict  # JSON Schema

    def __init__(self, sandbox: Sandbox):
        self._sandbox = sandbox

    @abstractmethod
    def execute(self, *args, **kwargs) -> str: ...

    @property
    def loggable_name(self) -> str:
        return f"{self.namespace}.{self.name}"

    def to_claude(self) -> dict:
        return {
            "name": self.name,
            "description": self.description,
            "input_schema": self.parameters,
        }

    def to_openai(self) -> dict:
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.parameters,
            },
        }
