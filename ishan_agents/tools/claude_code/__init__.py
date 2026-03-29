from ishan_agents.sandbox.base import Sandbox
from ishan_agents.tools.claude_code.tools import BashTool, EditTool, GlobTool, GrepTool, ReadTool, WriteTool


def claude_code_tools(sandbox: Sandbox):
    """Return the full claude_code tool set bound to a sandbox."""
    return [
        ReadTool(sandbox),
        WriteTool(sandbox),
        EditTool(sandbox),
        GlobTool(sandbox),
        GrepTool(sandbox),
        BashTool(sandbox),
    ]
