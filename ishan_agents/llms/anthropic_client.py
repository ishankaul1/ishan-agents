import anthropic

from ishan_agents.llms.base import LLMClient, LLMResponse, Message, ToolCall, UsageInfo
from ishan_agents.tools.base import BaseTool


class AnthropicClient(LLMClient):
    def __init__(self, model: str):
        self._model = model
        self._client = anthropic.AsyncAnthropic()

    async def call(
        self,
        messages: list[Message],
        tools: list[BaseTool],
        system: str | None = None,
        max_tokens: int = 16000,
    ) -> LLMResponse:
        kwargs: dict = dict(
            model=self._model,
            max_tokens=max_tokens,
            tools=[t.to_claude() for t in tools],
            messages=[m.to_anthropic() for m in messages],
        )
        if system:
            kwargs["system"] = system

        raw = await self._client.messages.create(**kwargs)

        content = None
        tool_calls = []
        for block in raw.content:
            if block.type == "text":
                content = block.text
            elif block.type == "tool_use":
                tool_calls.append(ToolCall(id=block.id, name=block.name, input=dict(block.input)))

        usage = UsageInfo(
            input_tokens=raw.usage.input_tokens,
            output_tokens=raw.usage.output_tokens,
            cache_read_tokens=getattr(raw.usage, "cache_read_input_tokens", 0) or 0,
            cache_write_tokens=getattr(raw.usage, "cache_creation_input_tokens", 0) or 0,
        )

        return LLMResponse(
            content=content,
            tool_calls=tool_calls,
            stop_reason=raw.stop_reason,
            usage=usage,
        )
