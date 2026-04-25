import json

from openai import AsyncOpenAI

from ishan_agents.llms.base import LLMClient, LLMResponse, Message, ToolCall, UsageInfo
from ishan_agents.tools.base import BaseTool


class OpenAICompatClient(LLMClient):
    """OpenAI-compatible client — works for Tinker, Fireworks, Together, local vLLM, etc.

    Pass base_url + api_key to redirect away from api.openai.com.
    """

    def __init__(self, model: str, base_url: str | None = None, api_key: str | None = None):
        self._model = model
        self._client = AsyncOpenAI(base_url=base_url, api_key=api_key or "placeholder")

    async def call(
        self,
        messages: list[Message],
        tools: list[BaseTool],
        system: str | None = None,
        max_tokens: int = 16000,
    ) -> LLMResponse:
        openai_messages = []
        if system:
            openai_messages.append({"role": "system", "content": system})
        for m in messages:
            openai_messages.extend(m.to_openai_parts())

        raw = await self._client.chat.completions.create(
            model=self._model,
            max_tokens=max_tokens,
            tools=[t.to_openai() for t in tools],
            messages=openai_messages,
        )

        choice = raw.choices[0]
        msg = choice.message

        tool_calls = [
            ToolCall(id=tc.id, name=tc.function.name, input=json.loads(tc.function.arguments))
            for tc in (msg.tool_calls or [])
        ]

        if choice.finish_reason == "length":
            stop_reason = "max_tokens"
        elif tool_calls:
            stop_reason = "tool_use"
        else:
            stop_reason = "end_turn"

        usage = UsageInfo(
            input_tokens=raw.usage.prompt_tokens if raw.usage else 0,
            output_tokens=raw.usage.completion_tokens if raw.usage else 0,
        )

        return LLMResponse(
            content=msg.content,
            tool_calls=tool_calls,
            stop_reason=stop_reason,
            usage=usage,
        )
