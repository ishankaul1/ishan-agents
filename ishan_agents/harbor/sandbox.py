import base64
import os
import shlex
from pathlib import PurePosixPath

from harbor.environments.base import BaseEnvironment

from ishan_agents.sandbox.base import Sandbox


class HarborSandbox(Sandbox):
    """Sandbox implementation backed by a Harbor BaseEnvironment (container exec)."""

    def __init__(self, environment: BaseEnvironment, work_dir: str = "/testbed"):
        # Skip local filesystem checks — work_dir lives inside the container.
        self._env = environment
        self.work_dir = PurePosixPath(work_dir)

    def _resolve(self, path: str) -> str:
        """Resolve path relative to work_dir; raise if it escapes the sandbox."""
        work = str(self.work_dir)
        full = os.path.normpath(os.path.join(work, path) if not os.path.isabs(path) else path)
        if full != work and not full.startswith(work + "/"):
            raise ValueError(f"Path escapes sandbox: {path}")
        return full

    async def read(self, path: str) -> str:
        resolved = self._resolve(path)
        result = await self._env.exec(f"cat {shlex.quote(resolved)}")
        if result.return_code != 0:
            raise FileNotFoundError(f"File not found: {path}")
        return result.stdout or ""

    async def write(self, path: str, content: str) -> str:
        resolved = self._resolve(path)
        b64 = base64.b64encode(content.encode()).decode()
        await self._env.exec(
            f"mkdir -p {shlex.quote(os.path.dirname(resolved))} && "
            f"echo {shlex.quote(b64)} | base64 -d > {shlex.quote(resolved)}"
        )
        return f"Written {path}"

    async def bash(self, cmd: str) -> str:
        result = await self._env.exec(cmd, cwd=str(self.work_dir))
        return (result.stdout or "") + (result.stderr or "")

    async def glob(self, pattern: str) -> list[str]:
        work = str(self.work_dir)
        # Use `find` with -path matching — universally available, no python3 dependency.
        # find's -path flag matches against the full path, so prefix the pattern with work_dir.
        find_pattern = os.path.join(work, pattern)
        result = await self._env.exec(
            f"find {shlex.quote(work)} -path {shlex.quote(find_pattern)} | sort"
        )
        lines = [line for line in (result.stdout or "").splitlines() if line]
        prefix = work.rstrip("/") + "/"
        return [line[len(prefix):] if line.startswith(prefix) else line for line in lines]

    async def grep(self, pattern: str, path: str) -> str:
        resolved = self._resolve(path)
        result = await self._env.exec(
            f"grep -rn {shlex.quote(pattern)} {shlex.quote(resolved)}",
            cwd=str(self.work_dir),
        )
        return result.stdout or ""
