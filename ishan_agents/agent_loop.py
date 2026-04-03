import anthropic
from dotenv import load_dotenv

from ishan_agents.sandbox.base import Sandbox
from ishan_agents.tools.base import BaseTool

load_dotenv()


def _base_system_prompt(sandbox: Sandbox) -> str:
    return (
        f"You are a coding agent. You must try your best to solve the task at hand autonomously; "
        f"the user will never respond to you. All file paths and bash commands run from {sandbox.work_dir}. "
        f"Use relative paths from there or absolute paths."
    )


async def run_agent_loop(
    model: str,
    sandbox: Sandbox,
    tools: list[BaseTool],
    user_message: str,
    system_prompt: str | None = None,
    max_turns: int = 50,
) -> list:
    assert max_turns > 0, f"max_turns must be > 0, got {max_turns}"

    client = anthropic.AsyncAnthropic()
    tool_map = {t.name: t for t in tools}
    system = system_prompt or _base_system_prompt(sandbox)
    messages = [{"role": "user", "content": user_message}]

    for _ in range(max_turns):
        response = await client.messages.create(
            model=model,
            max_tokens=16000,
            system=system,
            tools=[t.to_claude() for t in tools],
            messages=messages,
        )

        messages.append({"role": "assistant", "content": response.content})

        if response.stop_reason == "end_turn":
            break

        tool_use_blocks = [b for b in response.content if b.type == "tool_use"]
        if not tool_use_blocks:
            break

        tool_results = []

        # TODO: parallelize parallel tool calls
        for block in tool_use_blocks:
            tool = tool_map.get(block.name)
            if tool is None:
                result = f"Error: unknown tool '{block.name}'"
                is_error = True
            else:
                try:
                    result = await tool.execute(**block.input)
                    is_error = False
                except Exception as e:
                    result = f"Error: {e}"
                    is_error = True

            tool_results.append({
                "type": "tool_result",
                "tool_use_id": block.id,
                "content": result,
                "is_error": is_error,
            })

        messages.append({"role": "user", "content": tool_results})

    return messages
