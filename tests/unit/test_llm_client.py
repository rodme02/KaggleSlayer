"""Tests for kaggle_slayer.agent.llm_client.

This task covers the protocol + dataclasses. Gemini implementation tests
land in Task 9.
"""

from __future__ import annotations

from kaggle_slayer.agent import llm_client as llm


def test_message_dataclass():
    m = llm.Message(role="user", content="hello")
    assert m.role == "user"
    assert m.content == "hello"


def test_response_dataclass_defaults():
    r = llm.Response(text="ok", tool_calls=[], usage=llm.Usage(0, 0, 0))
    assert r.text == "ok"
    assert r.tool_calls == []
    assert r.usage.input_tokens == 0


def test_usage_dataclass():
    u = llm.Usage(input_tokens=10, output_tokens=5, cached_tokens=2)
    assert u.input_tokens == 10
    assert u.output_tokens == 5
    assert u.cached_tokens == 2
    assert u.total == 15  # input + output (cached is included in input but tracked separately)


def test_tool_call_dataclass():
    tc = llm.ToolCall(id="call_1", name="train_cv", args={"fe": "agent/fe.py"})
    assert tc.id == "call_1"
    assert tc.name == "train_cv"
    assert tc.args == {"fe": "agent/fe.py"}


def test_llm_client_protocol_exposes_call():
    """LLMClient is a Protocol with a call method.

    Structural conformance is verified end-to-end in test_fake_llm.py
    (FakeLLMClient is isinstance(LLMClient)). Here we just confirm the
    protocol object exists and exposes the expected name.
    """
    assert hasattr(llm.LLMClient, "call")
