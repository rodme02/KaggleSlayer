"""Tests for tests/fixtures/fake_llm.py.

This is test-support code, but it gets a real test file because the
integration tier and Week 3 agent loop will depend on it being correct.
"""

from __future__ import annotations

import pytest

from kaggle_slayer.agent.llm_client import LLMClient, Message
from tests.fixtures.fake_llm import FakeLLMClient, ScriptedResponse


def test_fake_llm_returns_next_scripted_response():
    fake = FakeLLMClient(
        script=[
            ScriptedResponse(text="hello"),
            ScriptedResponse(text="world"),
        ]
    )
    r1 = fake.call(messages=[Message(role="user", content="say hi")])
    r2 = fake.call(messages=[Message(role="user", content="say more")])
    assert r1.text == "hello"
    assert r2.text == "world"


def test_fake_llm_records_messages():
    fake = FakeLLMClient(script=[ScriptedResponse(text="ok")])
    fake.call(messages=[Message(role="user", content="probe")])
    assert len(fake.calls) == 1
    assert fake.calls[0].messages[-1].content == "probe"


def test_fake_llm_raises_on_script_exhaustion():
    fake = FakeLLMClient(script=[ScriptedResponse(text="only one")])
    fake.call(messages=[Message(role="user", content="first")])
    with pytest.raises(RuntimeError, match="exhausted"):
        fake.call(messages=[Message(role="user", content="second")])


def test_fake_llm_implements_protocol():
    """Confirm FakeLLMClient is a structural match for LLMClient."""
    fake = FakeLLMClient(script=[])
    assert isinstance(fake, LLMClient)


def test_fake_llm_default_usage_is_zero():
    fake = FakeLLMClient(script=[ScriptedResponse(text="x")])
    resp = fake.call(messages=[Message(role="user", content="probe")])
    assert resp.usage.input_tokens == 0
    assert resp.usage.output_tokens == 0
