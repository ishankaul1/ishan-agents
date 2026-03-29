import click

from ishan_agents.agent_loop import run_agent_loop
from ishan_agents.sandbox.local import LocalSandbox
from ishan_agents.tools.claude_code import claude_code_tools

DEFAULT_MODEL = "claude-sonnet-4-6"
DEFAULT_WORK_DIR = "test-work-dir"

_NAMESPACE_REGISTRY = {
    "claude_code": claude_code_tools,
}


def resolve_tools(tool_specs: tuple[str, ...], sandbox: LocalSandbox) -> list:
    all_namespace_tools = {
        t.loggable_name: t for name, factory in _NAMESPACE_REGISTRY.items() for t in factory(sandbox)
    }
    all_namespace_tools_by_ns = {name: factory(sandbox) for name, factory in _NAMESPACE_REGISTRY.items()}

    selected = {}
    for spec in tool_specs:
        if "." in spec:
            # specific tool e.g. claude_code.Bash
            tool = all_namespace_tools.get(spec)
            if tool is None:
                raise click.BadParameter(f"Unknown tool '{spec}'. Available: {list(all_namespace_tools)}")
            selected[spec] = tool
        else:
            # whole namespace e.g. claude_code
            ns_tools = all_namespace_tools_by_ns.get(spec)
            if ns_tools is None:
                raise click.BadParameter(f"Unknown namespace '{spec}'. Available: {list(_NAMESPACE_REGISTRY)}")
            for t in ns_tools:
                selected[t.loggable_name] = t

    return list(selected.values())


@click.command()
@click.option("--model", default=DEFAULT_MODEL, show_default=True, help="Model to use.")
@click.option("--work-dir", default=DEFAULT_WORK_DIR, show_default=True, help="Sandbox working directory.")
@click.option(
    "--tools",
    multiple=True,
    default=("claude_code",),
    show_default=True,
    help="Tools to enable. Accepts a namespace (e.g. claude_code) or specific tools "
    "(e.g. claude_code.Bash). Additive, can be passed multiple times.",
)
@click.option("--system-prompt", default=None, help="Optional system prompt override.")
@click.argument("message")
def main(model: str, work_dir: str, tools: tuple[str, ...], system_prompt: str | None, message: str):
    sandbox = LocalSandbox(work_dir)
    resolved_tools = resolve_tools(tools, sandbox)

    click.echo(f"Model:   {model}")
    click.echo(f"WorkDir: {sandbox.work_dir}")
    click.echo(f"Tools:   {[t.loggable_name for t in resolved_tools]}")
    click.echo(f"Message: {message}\n")

    messages = run_agent_loop(
        model=model,
        sandbox=sandbox,
        tools=resolved_tools,
        user_message=message,
        system_prompt=system_prompt,
    )

    # Print final assistant text response
    for msg in reversed(messages):
        if msg["role"] == "assistant":
            for block in msg["content"]:
                if hasattr(block, "type") and block.type == "text":
                    click.echo(block.text)
            break


if __name__ == "__main__":
    main()
