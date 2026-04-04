"""Tests for HarborSandbox using mocked BaseEnvironment."""

import pytest
from unittest.mock import AsyncMock, MagicMock

from ishan_agents.harbor.sandbox import HarborSandbox


def make_result(stdout: str = "", stderr: str = "", return_code: int = 0):
    r = MagicMock()
    r.stdout = stdout
    r.stderr = stderr
    r.return_code = return_code
    return r


def make_env(stdout: str = "", stderr: str = "", return_code: int = 0):
    env = MagicMock()
    env.exec = AsyncMock(return_value=make_result(stdout, stderr, return_code))
    return env


# ── _resolve ──────────────────────────────────────────────────────────────────

def test_resolve_relative():
    sb = HarborSandbox(make_env(), "/testbed")
    assert sb._resolve("foo/bar.py") == "/testbed/foo/bar.py"


def test_resolve_absolute_inside():
    sb = HarborSandbox(make_env(), "/testbed")
    assert sb._resolve("/testbed/foo.py") == "/testbed/foo.py"


def test_resolve_traversal_raises():
    sb = HarborSandbox(make_env(), "/testbed")
    with pytest.raises(ValueError, match="escapes sandbox"):
        sb._resolve("../../etc/passwd")


def test_resolve_absolute_outside_raises():
    sb = HarborSandbox(make_env(), "/testbed")
    with pytest.raises(ValueError, match="escapes sandbox"):
        sb._resolve("/etc/passwd")


# ── read ──────────────────────────────────────────────────────────────────────

async def test_read_returns_stdout():
    env = make_env(stdout="hello world\n")
    sb = HarborSandbox(env, "/testbed")
    result = await sb.read("foo.txt")
    assert result == "hello world\n"
    env.exec.assert_awaited_once_with("cat /testbed/foo.txt")


async def test_read_not_found_raises():
    env = make_env(return_code=1)
    sb = HarborSandbox(env, "/testbed")
    with pytest.raises(FileNotFoundError):
        await sb.read("missing.txt")


# ── write ─────────────────────────────────────────────────────────────────────

async def test_write_returns_message():
    env = make_env()
    sb = HarborSandbox(env, "/testbed")
    result = await sb.write("out.txt", "content")
    assert result == "Written out.txt"
    env.exec.assert_awaited_once()
    cmd = env.exec.call_args[0][0]
    assert "base64" in cmd
    assert "/testbed/out.txt" in cmd


# ── bash ──────────────────────────────────────────────────────────────────────

async def test_bash_combines_stdout_stderr():
    env = MagicMock()
    env.exec = AsyncMock(return_value=make_result(stdout="out\n", stderr="err\n"))
    sb = HarborSandbox(env, "/testbed")
    result = await sb.bash("echo hi")
    assert result == "out\nerr\n"
    env.exec.assert_awaited_once_with("echo hi", cwd="/testbed")


# ── glob ──────────────────────────────────────────────────────────────────────

async def test_glob_strips_work_dir_prefix():
    env = make_env(stdout="/testbed/src/foo.py\n/testbed/src/bar.py\n")
    sb = HarborSandbox(env, "/testbed")
    result = await sb.glob("src/*.py")
    assert result == ["src/foo.py", "src/bar.py"]


async def test_glob_empty():
    env = make_env(stdout="")
    sb = HarborSandbox(env, "/testbed")
    result = await sb.glob("*.nonexistent")
    assert result == []


async def test_glob_uses_find():
    env = make_env(stdout="")
    sb = HarborSandbox(env, "/testbed")
    await sb.glob("**/*.py")
    cmd = env.exec.call_args[0][0]
    assert cmd.startswith("find ")


# ── grep ──────────────────────────────────────────────────────────────────────

async def test_grep_returns_matches():
    env = make_env(stdout="foo.py:1: def hello()\n")
    sb = HarborSandbox(env, "/testbed")
    result = await sb.grep("hello", ".")
    assert "hello" in result


async def test_grep_traversal_raises():
    sb = HarborSandbox(make_env(), "/testbed")
    with pytest.raises(ValueError, match="escapes sandbox"):
        await sb.grep("pattern", "../../etc")
