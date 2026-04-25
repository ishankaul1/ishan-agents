from __future__ import annotations

import json
from abc import ABC, abstractmethod
from typing import Literal

from pydantic import BaseModel

from ishan_agents.tools.base import BaseTool


class ToolCall(BaseModel, frozen=True):
    id: str
    name: str
    input: dict


class ToolResult(BaseModel, frozen=True):
    tool_call_id: str
    content: str
    is_error: bool = False


class UsageInfo(BaseModel, frozen=True):
    input_tokens: int
    output_tokens: int
    cache_read_tokens: int = 0
    cache_write_tokens: int = 0


class LLMResponse(BaseModel, frozen=True):
    content: str | None
    tool_calls: list[ToolCall]
    stop_reason: Literal["end_turn", "tool_use", "max_tokens"]
    usage: UsageInfo


class Message(BaseModel, frozen=True):
    """A single turn in the conversation.

    Exactly one of (content, tool_results) is set on user messages.
    Assistant messages have content and/or tool_calls.
    """

    role: Literal["user", "assistant"]
    content: str | None = None
    tool_calls: list[ToolCall] | None = None
    tool_results: list[ToolResult] | None = None
    usage: UsageInfo | None = None

    @staticmethod
    def user(text: str) -> Message:
        return Message(role="user", content=text)

    @staticmethod
    def from_response(response: LLMResponse) -> Message:
        return Message(
            role="assistant",
            content=response.content,
            tool_calls=response.tool_calls or None,
            usage=response.usage,
        )

    @staticmethod
    def with_tool_results(results: list[ToolResult]) -> Message:
        return Message(role="user", tool_results=results)

    def to_anthropic(self) -> dict:
        if self.tool_results:
            return {
                "role": "user",
                "content": [
                    {
                        "type": "tool_result",
                        "tool_use_id": r.tool_call_id,
                        "content": r.content,
                        "is_error": r.is_error,
                    }
                    for r in self.tool_results
                ],
            }
        if self.role == "assistant":
            blocks = []
            if self.content:
                blocks.append({"type": "text", "text": self.content})
            for tc in self.tool_calls or []:
                blocks.append({"type": "tool_use", "id": tc.id, "name": tc.name, "input": tc.input})
            return {"role": "assistant", "content": blocks}
        return {"role": "user", "content": self.content or ""}

    def to_openai_parts(self) -> list[dict]:
        """Expand into one-or-more OpenAI message dicts.

        OpenAI needs one 'tool' message per result; all other roles produce a single dict.
        """
        if self.tool_results:
            return [
                {"role": "tool", "tool_call_id": r.tool_call_id, "content": r.content}
                for r in self.tool_results
            ]
        if self.role == "assistant":
            msg: dict = {"role": "assistant", "content": self.content}
            if self.tool_calls:
                msg["tool_calls"] = [
                    {
                        "id": tc.id,
                        "type": "function",
                        "function": {"name": tc.name, "arguments": json.dumps(tc.input)},
                    }
                    for tc in self.tool_calls
                ]
            return [msg]
        return [{"role": "user", "content": self.content or ""}]


class LLMClient(ABC):
    @abstractmethod
    async def call(
        self,
        messages: list[Message],
        tools: list[BaseTool],
        system: str | None = None,
        max_tokens: int = 16000,
    ) -> LLMResponse: ...
