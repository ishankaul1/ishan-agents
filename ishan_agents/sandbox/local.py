import subprocess

from ishan_agents.sandbox.base import Sandbox


class LocalSandbox(Sandbox):
    pass

    def read(self, path: str) -> str:
        p = self._resolve(path)
        if not p.is_file():
            raise FileNotFoundError(f"File not found: {path}")
        return p.read_text()

    def write(self, path: str, content: str) -> str:
        p = self._resolve(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(content)
        return f"Written {path}"

    def bash(self, cmd: str) -> str:
        result = subprocess.run(cmd, shell=True, cwd=self.work_dir, capture_output=True, text=True)
        return result.stdout + result.stderr

    def glob(self, pattern: str) -> list[str]:
        return [str(p.relative_to(self.work_dir)) for p in self.work_dir.glob(pattern)]

    def grep(self, pattern: str, path: str) -> str:
        self._resolve(path)
        result = subprocess.run(
            ["grep", "-rn", pattern, path], cwd=self.work_dir, capture_output=True, text=True
        )
        return result.stdout
