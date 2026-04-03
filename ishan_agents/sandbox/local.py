import asyncio

import aiofiles
import aiofiles.os

from ishan_agents.sandbox.base import Sandbox


class LocalSandbox(Sandbox):
    async def read(self, path: str) -> str:
        p = self._resolve(path)
        if not await aiofiles.os.path.isfile(str(p)):
            raise FileNotFoundError(f"File not found: {path}")
        async with aiofiles.open(p) as f:
            return await f.read()

    async def write(self, path: str, content: str) -> str:
        p = self._resolve(path)
        await aiofiles.os.makedirs(str(p.parent), exist_ok=True)
        async with aiofiles.open(p, "w") as f:
            await f.write(content)
        return f"Written {path}"

    async def bash(self, cmd: str) -> str:
        proc = await asyncio.create_subprocess_shell(
            cmd,
            cwd=self.work_dir,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        stdout, stderr = await proc.communicate()
        return stdout.decode() + stderr.decode()

    async def glob(self, pattern: str) -> list[str]:
        work_dir = self.work_dir
        return await asyncio.to_thread(
            lambda: [str(p.relative_to(work_dir)) for p in work_dir.glob(pattern)]
        )

    async def grep(self, pattern: str, path: str) -> str:
        self._resolve(path)
        proc = await asyncio.create_subprocess_exec(
            "grep", "-rn", pattern, path,
            cwd=self.work_dir,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        stdout, _ = await proc.communicate()
        return stdout.decode()
