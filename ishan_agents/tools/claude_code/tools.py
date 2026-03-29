from ishan_agents.tools.base import BaseTool


class ReadTool(BaseTool):
    name = "Read"
    namespace = "claude_code"
    description = "Read a file from the filesystem. Returns the file contents with line numbers."
    parameters = {
        "type": "object",
        "properties": {
            "file_path": {"type": "string", "description": "The absolute or relative path to the file to read."},
            "offset": {"type": "integer", "description": "Line number to start reading from (1-indexed)."},
            "limit": {"type": "integer", "description": "Number of lines to read."},
        },
        "required": ["file_path"],
    }

    def execute(self, file_path: str, offset: int | None = None, limit: int | None = None) -> str:
        content = self._sandbox.read(file_path)
        if offset is not None or limit is not None:
            lines = content.splitlines(keepends=True)
            if offset is not None:
                lines = lines[offset - 1 :]
            if limit is not None:
                lines = lines[:limit]
            return "".join(lines)
        return content


class WriteTool(BaseTool):
    name = "Write"
    namespace = "claude_code"
    description = "Create or overwrite a file with the given content."
    parameters = {
        "type": "object",
        "properties": {
            "file_path": {"type": "string", "description": "The path to write to."},
            "content": {"type": "string", "description": "The content to write."},
        },
        "required": ["file_path", "content"],
    }

    def execute(self, file_path: str, content: str) -> str:
        return self._sandbox.write(file_path, content)


class EditTool(BaseTool):
    name = "Edit"
    namespace = "claude_code"
    description = (
        "Make a precise string replacement in a file. "
        "old_string must appear exactly once unless replace_all is true."
    )
    parameters = {
        "type": "object",
        "properties": {
            "file_path": {"type": "string", "description": "The file to edit."},
            "old_string": {"type": "string", "description": "The exact string to replace."},
            "new_string": {"type": "string", "description": "The string to replace it with."},
            "replace_all": {"type": "boolean", "description": "Replace all occurrences. Defaults to false."},
        },
        "required": ["file_path", "old_string", "new_string"],
    }

    def execute(self, file_path: str, old_string: str, new_string: str, replace_all: bool = False) -> str:
        content = self._sandbox.read(file_path)
        if old_string not in content:
            raise ValueError(f"old_string not found in {file_path}")
        if not replace_all and content.count(old_string) > 1:
            raise ValueError(f"old_string matches multiple locations in {file_path}; use replace_all=true")
        count = None if replace_all else 1
        new_content = content.replace(old_string, new_string, count or -1)
        return self._sandbox.write(file_path, new_content)


class GlobTool(BaseTool):
    name = "Glob"
    namespace = "claude_code"
    description = "Find files matching a glob pattern. Returns matching paths sorted by modification time."
    parameters = {
        "type": "object",
        "properties": {
            "pattern": {"type": "string", "description": "Glob pattern, e.g. '**/*.py' or 'src/**/*.ts'."},
            "path": {"type": "string", "description": "Directory to search in. Defaults to the sandbox root."},
        },
        "required": ["pattern"],
    }

    def execute(self, pattern: str, path: str | None = None) -> str:
        if path:
            pattern = f"{path.rstrip('/')}/{pattern}"
        results = self._sandbox.glob(pattern)
        return "\n".join(results) if results else "(no matches)"


class GrepTool(BaseTool):
    name = "Grep"
    namespace = "claude_code"
    description = "Search file contents using a regex pattern."
    parameters = {
        "type": "object",
        "properties": {
            "pattern": {"type": "string", "description": "Regex pattern to search for."},
            "path": {"type": "string", "description": "File or directory to search. Defaults to sandbox root."},
            "include": {"type": "string", "description": "Glob pattern to filter files, e.g. '*.py'."},
        },
        "required": ["pattern"],
    }

    def execute(self, pattern: str, path: str = ".", include: str | None = None) -> str:
        return self._sandbox.grep(pattern, path)


class BashTool(BaseTool):
    name = "Bash"
    namespace = "claude_code"
    description = "Execute a shell command and return its output."
    parameters = {
        "type": "object",
        "properties": {
            "command": {"type": "string", "description": "The shell command to run."},
            "description": {"type": "string", "description": "Short description of what the command does."},
            "timeout": {"type": "integer", "description": "Timeout in milliseconds."},
        },
        "required": ["command"],
    }

    def execute(self, command: str, description: str | None = None, timeout: int | None = None) -> str:
        return self._sandbox.bash(command)
