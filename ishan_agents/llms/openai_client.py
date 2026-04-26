import json

from openai import AsyncOpenAI

from ishan_agents.llms.base import LLMClient, LLMResponse, Message, ToolCall, UsageInfo
from ishan_agents.tools.base import BaseTool


class OpenAICompatClient(LLMClient):
    """OpenAI-compatible client — works for Tinker, Fireworks, Together, local vLLM, etc.

    Pass base_url + api_key to redirect away from api.openai.com.
    Always streams to avoid provider-specific max_tokens restrictions.
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

        content_parts: list[str] = []
        # tool call accumulators keyed by chunk index
        tc_acc: dict[int, dict] = {}
        finish_reason: str | None = None
        prompt_tokens = 0
        completion_tokens = 0

        stream = await self._client.chat.completions.create(
            model=self._model,
            max_tokens=max_tokens,
            tools=[t.to_openai() for t in tools],
            messages=openai_messages,
            stream=True,
            stream_options={"include_usage": True},
        )

        async for chunk in stream:
            if chunk.usage:
                prompt_tokens = chunk.usage.prompt_tokens or 0
                completion_tokens = chunk.usage.completion_tokens or 0

            if not chunk.choices:
                continue

            choice = chunk.choices[0]
            if choice.finish_reason:
                finish_reason = choice.finish_reason

            delta = choice.delta
            if delta.content:
                content_parts.append(delta.content)

            for tc_delta in delta.tool_calls or []:
                idx = tc_delta.index
                if idx not in tc_acc:
                    tc_acc[idx] = {"id": "", "name": "", "arguments": ""}
                if tc_delta.id:
                    tc_acc[idx]["id"] = tc_delta.id
                if tc_delta.function:
                    if tc_delta.function.name:
                        tc_acc[idx]["name"] += tc_delta.function.name
                    if tc_delta.function.arguments:
                        tc_acc[idx]["arguments"] += tc_delta.function.arguments

        tool_calls = [
            ToolCall(
                id=tc_acc[i]["id"],
                name=tc_acc[i]["name"],
                input=json.loads(tc_acc[i]["arguments"]) if tc_acc[i]["arguments"] else {},
            )
            for i in sorted(tc_acc)
        ]

        if finish_reason == "length":
            stop_reason = "max_tokens"
        elif tool_calls:
            stop_reason = "tool_use"
        else:
            stop_reason = "end_turn"

        return LLMResponse(
            content="".join(content_parts) or None,
            tool_calls=tool_calls,
            stop_reason=stop_reason,
            usage=UsageInfo(input_tokens=prompt_tokens, output_tokens=completion_tokens),
        )
