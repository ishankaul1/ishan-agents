from dotenv import load_dotenv
from loguru import logger

from ishan_agents.llms.base import LLMClient, Message, ToolResult
from ishan_agents.sandbox.base import Sandbox
from ishan_agents.tools import resolve_tools
from ishan_agents.tools.base import BaseTool

load_dotenv()


def default_system_prompt(work_dir) -> str:
    return (
        f"You are a coding agent. You must try your best to solve the task at hand autonomously; "
        f"the user will never respond to you. All file paths and bash commands run from {work_dir}. "
        f"Use relative paths from there or absolute paths."
    )


class LoopResult:
    def __init__(self, messages: list[Message], tools: list[BaseTool]):
        self.messages = messages
        self.tools = tools

    @property
    def turn_usages(self):
        return [m.usage for m in self.messages if m.role == "assistant" and m.usage is not None]


async def run_agent_loop(
    client: LLMClient,
    sandbox: Sandbox,
    tools: list[str],
    user_message: str,
    system_prompt: str | None = None,
    max_turns: int = 50,
) -> LoopResult:
    assert max_turns > 0, f"max_turns must be > 0, got {max_turns}"

    resolved = resolve_tools(tools, sandbox)
    tool_map = {t.name: t for t in resolved}
    messages: list[Message] = [Message.user(user_message)]

    logger.info(f"Starting agent loop | tools={[t.name for t in resolved]} | max_turns={max_turns}")

    # TODO Context management, max token guardrails
    # Multi-env evals with agent pointing at MCPS could be pretty powerful too.

    for turn in range(max_turns):
        response = await client.call(messages=messages, tools=resolved, system=system_prompt)
        messages.append(Message.from_response(response))

        usage = response.usage
        logger.info(
            f"[turn {turn + 1}] stop={response.stop_reason} | "
            f"in={usage.input_tokens} out={usage.output_tokens} cache={usage.cache_read_tokens}"
        )
        if response.content:
            preview = response.content[:300].replace("\n", " ")
            logger.info(f"[turn {turn + 1}] text: {preview}{'...' if len(response.content) > 300 else ''}")

        if response.stop_reason == "end_turn" or not response.tool_calls:
            break

        tool_results = []
        for tc in response.tool_calls:
            input_preview = str(tc.input)[:200]
            logger.info(f"[turn {turn + 1}] → {tc.name}({input_preview})")

            tool = tool_map.get(tc.name)
            if tool is None:
                content = f"Error: unknown tool '{tc.name}'"
                is_error = True
            else:
                try:
                    content = await tool.execute(**tc.input)
                    is_error = False
                except Exception as e:
                    content = f"Error: {e}"
                    is_error = True

            result_preview = content[:300].replace("\n", " ")
            level = "warning" if is_error else "debug"
            logger.log(level.upper(), f"[turn {turn + 1}] ← {tc.name}: {result_preview}{'...' if len(content) > 300 else ''}")

            tool_results.append(ToolResult(tool_call_id=tc.id, content=content, is_error=is_error))

        messages.append(Message.with_tool_results(tool_results))

    total_in = sum(u.input_tokens for u in [m.usage for m in messages if m.usage])
    total_out = sum(u.output_tokens for u in [m.usage for m in messages if m.usage])
    logger.info(f"Agent loop done | turns={turn + 1} | total_in={total_in} total_out={total_out}")

    return LoopResult(messages=messages, tools=resolved)
