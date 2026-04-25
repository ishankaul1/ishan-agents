import importlib
import inspect

from ishan_agents.sandbox.base import Sandbox
from ishan_agents.tools.base import BaseTool


def resolve_tools(specs: list[str], sandbox: Sandbox) -> list[BaseTool]:
    """Resolve tool specs to BaseTool instances bound to a sandbox.

    "claude_code"      -> all tools from ishan_agents.tools.claude_code (via its factory)
    "claude_code.Read" -> just the tool with name=="Read" from that namespace
    """
    result: list[BaseTool] = []
    seen: set[str] = set()

    def _add(tool: BaseTool) -> None:
        if tool.loggable_name not in seen:
            result.append(tool)
            seen.add(tool.loggable_name)

    for spec in specs:
        if "." not in spec:
            namespace = spec
            module = importlib.import_module(f"ishan_agents.tools.{namespace}")
            factory = getattr(module, f"{namespace}_tools", None)
            if factory is None:
                raise ValueError(f"No factory '{namespace}_tools' found in ishan_agents.tools.{namespace}")
            for tool in factory(sandbox):
                _add(tool)
        else:
            namespace, tool_name = spec.split(".", 1)
            tools_module = importlib.import_module(f"ishan_agents.tools.{namespace}.tools")
            for _, cls in inspect.getmembers(tools_module, inspect.isclass):
                if issubclass(cls, BaseTool) and cls is not BaseTool and getattr(cls, "name", None) == tool_name:
                    _add(cls(sandbox))
                    break
            else:
                raise ValueError(f"Tool '{spec}' not found in ishan_agents.tools.{namespace}.tools")

    return result
