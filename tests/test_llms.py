import json

import pytest

from ishan_agents.llms.base import LLMResponse, Message, ToolCall, ToolResult, UsageInfo


@pytest.fixture
def usage():
    return UsageInfo(input_tokens=10, output_tokens=5)


@pytest.fixture
def tool_call():
    return ToolCall(id="tc_1", name="Read", input={"path": "foo.txt"})


# --- Message factories ---


def test_user_message():
    m = Message.user("hello")
    assert m.role == "user"
    assert m.content == "hello"
    assert m.tool_calls is None
    assert m.tool_results is None


def test_from_response_text_only(usage):
    r = LLMResponse(content="hi", tool_calls=[], stop_reason="end_turn", usage=usage)
    m = Message.from_response(r)
    assert m.role == "assistant"
    assert m.content == "hi"
    assert m.tool_calls is None
    assert m.usage == usage


def test_from_response_with_tool_calls(usage, tool_call):
    r = LLMResponse(content=None, tool_calls=[tool_call], stop_reason="tool_use", usage=usage)
    m = Message.from_response(r)
    assert m.tool_calls == [tool_call]


def test_with_tool_results():
    results = [ToolResult(tool_call_id="tc_1", content="file contents")]
    m = Message.with_tool_results(results)
    assert m.role == "user"
    assert m.tool_results == results
    assert m.content is None


# --- to_anthropic ---


def test_to_anthropic_user_text():
    assert Message.user("hi").to_anthropic() == {"role": "user", "content": "hi"}


def test_to_anthropic_assistant_text_only(usage):
    r = LLMResponse(content="hello", tool_calls=[], stop_reason="end_turn", usage=usage)
    d = Message.from_response(r).to_anthropic()
    assert d["role"] == "assistant"
    assert d["content"] == [{"type": "text", "text": "hello"}]


def test_to_anthropic_assistant_with_tool_call(usage, tool_call):
    r = LLMResponse(content=None, tool_calls=[tool_call], stop_reason="tool_use", usage=usage)
    d = Message.from_response(r).to_anthropic()
    assert d["role"] == "assistant"
    assert d["content"] == [{"type": "tool_use", "id": "tc_1", "name": "Read", "input": {"path": "foo.txt"}}]


def test_to_anthropic_tool_results():
    m = Message.with_tool_results([ToolResult(tool_call_id="tc_1", content="ok")])
    d = m.to_anthropic()
    assert d["role"] == "user"
    assert d["content"] == [{"type": "tool_result", "tool_use_id": "tc_1", "content": "ok", "is_error": False}]


def test_to_anthropic_tool_result_error():
    m = Message.with_tool_results([ToolResult(tool_call_id="tc_1", content="boom", is_error=True)])
    block = m.to_anthropic()["content"][0]
    assert block["is_error"] is True


# --- to_openai_parts ---


def test_to_openai_user_text():
    parts = Message.user("hi").to_openai_parts()
    assert parts == [{"role": "user", "content": "hi"}]


def test_to_openai_assistant_text_only(usage):
    r = LLMResponse(content="hello", tool_calls=[], stop_reason="end_turn", usage=usage)
    parts = Message.from_response(r).to_openai_parts()
    assert parts == [{"role": "assistant", "content": "hello"}]


def test_to_openai_assistant_with_tool_call(usage, tool_call):
    r = LLMResponse(content=None, tool_calls=[tool_call], stop_reason="tool_use", usage=usage)
    parts = Message.from_response(r).to_openai_parts()
    assert len(parts) == 1
    msg = parts[0]
    assert msg["role"] == "assistant"
    assert msg["tool_calls"][0]["id"] == "tc_1"
    assert msg["tool_calls"][0]["function"]["name"] == "Read"
    assert json.loads(msg["tool_calls"][0]["function"]["arguments"]) == {"path": "foo.txt"}


def test_to_openai_tool_results_fan_out():
    results = [
        ToolResult(tool_call_id="tc_1", content="a"),
        ToolResult(tool_call_id="tc_2", content="b"),
    ]
    parts = Message.with_tool_results(results).to_openai_parts()
    assert len(parts) == 2
    assert parts[0] == {"role": "tool", "tool_call_id": "tc_1", "content": "a"}
    assert parts[1] == {"role": "tool", "tool_call_id": "tc_2", "content": "b"}


# --- immutability ---


def test_message_is_frozen():
    m = Message.user("hi")
    with pytest.raises(Exception):
        m.content = "bye"  # type: ignore[misc]
