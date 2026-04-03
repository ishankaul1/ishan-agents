import pytest

from ishan_agents.sandbox.local import LocalSandbox


@pytest.fixture
def sandbox(tmp_path):
    return LocalSandbox(str(tmp_path))


# --- _resolve ---


def test_resolve_valid_path(sandbox):
    sandbox._resolve("foo.txt")


def test_resolve_nested_valid_path(sandbox):
    sandbox._resolve("a/b/c.txt")


def test_resolve_path_traversal_raises(sandbox):
    with pytest.raises(ValueError, match="escapes sandbox"):
        sandbox._resolve("../escape.txt")


def test_resolve_absolute_path_outside_raises(sandbox):
    with pytest.raises(ValueError, match="escapes sandbox"):
        sandbox._resolve("/etc/passwd")


# --- LocalSandbox ---


@pytest.mark.asyncio
async def test_read_write_roundtrip(sandbox):
    await sandbox.write("hello.txt", "world")
    assert await sandbox.read("hello.txt") == "world"


@pytest.mark.asyncio
async def test_write_creates_nested_dirs(sandbox):
    await sandbox.write("a/b/c.txt", "nested")
    assert await sandbox.read("a/b/c.txt") == "nested"


@pytest.mark.asyncio
async def test_read_missing_file_raises(sandbox):
    with pytest.raises(FileNotFoundError):
        await sandbox.read("nope.txt")


@pytest.mark.asyncio
async def test_bash_runs_command(sandbox):
    out = await sandbox.bash("echo hello")
    assert out.strip() == "hello"


@pytest.mark.asyncio
async def test_bash_stderr_captured(sandbox):
    out = await sandbox.bash("echo err >&2")
    assert "err" in out


@pytest.mark.asyncio
async def test_glob_finds_files(sandbox):
    await sandbox.write("a.py", "")
    await sandbox.write("b.py", "")
    await sandbox.write("c.txt", "")
    results = await sandbox.glob("*.py")
    assert set(results) == {"a.py", "b.py"}


@pytest.mark.asyncio
async def test_grep_finds_match(sandbox):
    await sandbox.write("foo.txt", "hello world\nbye world\n")
    out = await sandbox.grep("hello", "foo.txt")
    assert "hello" in out
    assert "bye" not in out
