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


def test_read_write_roundtrip(sandbox):
    sandbox.write("hello.txt", "world")
    assert sandbox.read("hello.txt") == "world"


def test_write_creates_nested_dirs(sandbox):
    sandbox.write("a/b/c.txt", "nested")
    assert sandbox.read("a/b/c.txt") == "nested"


def test_read_missing_file_raises(sandbox):
    with pytest.raises(FileNotFoundError):
        sandbox.read("nope.txt")


def test_bash_runs_command(sandbox):
    out = sandbox.bash("echo hello")
    assert out.strip() == "hello"


def test_bash_stderr_captured(sandbox):
    out = sandbox.bash("echo err >&2")
    assert "err" in out


def test_glob_finds_files(sandbox):
    sandbox.write("a.py", "")
    sandbox.write("b.py", "")
    sandbox.write("c.txt", "")
    results = sandbox.glob("*.py")
    assert set(results) == {"a.py", "b.py"}


def test_grep_finds_match(sandbox):
    sandbox.write("foo.txt", "hello world\nbye world\n")
    out = sandbox.grep("hello", "foo.txt")
    assert "hello" in out
    assert "bye" not in out
