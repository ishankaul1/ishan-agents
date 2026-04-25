import asyncio

import click

from ishan_agents.agent_loop import default_system_prompt, run_agent_loop
from ishan_agents.llms.factory import _PROVIDERS, make_client
from ishan_agents.log import configure_stdout_logging
from ishan_agents.sandbox.local import LocalSandbox

DEFAULT_MODEL = "claude-sonnet-4-6"
DEFAULT_WORK_DIR = "test-work-dir"


@click.command()
@click.option("--model", default=DEFAULT_MODEL, show_default=True, help="Model name.")
@click.option(
    "--provider",
    default="anthropic",
    show_default=True,
    type=click.Choice(list(_PROVIDERS)),
    help="LLM provider.",
)
@click.option("--work-dir", default=DEFAULT_WORK_DIR, show_default=True, help="Sandbox working directory.")
@click.option(
    "--tools",
    multiple=True,
    default=("claude_code",),
    show_default=True,
    help="Tools to enable. Accepts a namespace (e.g. claude_code) or specific tool (e.g. claude_code.Bash).",
)
@click.option("--system-prompt", default=None, help="Optional system prompt override.")
@click.argument("message")
def main(
    model: str,
    provider: str,
    work_dir: str,
    tools: tuple[str, ...],
    system_prompt: str | None,
    message: str,
):
    configure_stdout_logging()
    sandbox = LocalSandbox(work_dir)
    client = make_client(provider, model)
    system = system_prompt or default_system_prompt(sandbox.work_dir)

    click.echo(f"Provider: {provider}")
    click.echo(f"Model:    {model}")
    click.echo(f"WorkDir:  {sandbox.work_dir}")
    click.echo(f"Tools:    {list(tools)}")
    click.echo(f"Message:  {message}\n")

    result = asyncio.run(
        run_agent_loop(
            client=client,
            sandbox=sandbox,
            tools=list(tools),
            user_message=message,
            system_prompt=system,
        )
    )

    for msg in reversed(result.messages):
        if msg.role == "assistant" and msg.content:
            click.echo(msg.content)
            break


if __name__ == "__main__":
    main()
