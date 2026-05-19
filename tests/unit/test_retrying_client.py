"""Unit tests for RetryingLLMClient.

The adapter retries TransientLLMError with exponential backoff and lets
any other exception bubble up immediately. These tests pin the retry
semantics that the chaos tier depends on (spec §11.3 / §13).
"""

from __future__ import annotations

from typing import Any

import pytest

from kaggle_slayer.agent.llm_client import (
    Message,
    Response,
    ToolCall,
    TransientLLMError,
    Usage,
)
from kaggle_slayer.agent.retrying_client import RetryingLLMClient


class _RecordingClient:
    """LLMClient stub: enqueue responses or exceptions to be returned in order."""

    def __init__(self, outcomes: list[Any]) -> None:
        self._outcomes = list(outcomes)
        self.calls: list[dict[str, Any]] = []

    def call(
        self,
        messages: list[Message],
        *,
        tools: list[dict[str, Any]] | None = None,
        model: str | None = None,
    ) -> Response:
        self.calls.append({"messages": messages, "tools": tools, "model": model})
        if not self._outcomes:
            raise AssertionError("no more scripted outcomes")
        outcome = self._outcomes.pop(0)
        if isinstance(outcome, Exception):
            raise outcome
        return outcome


def _ok_response(text: str = "ok") -> Response:
    return Response(
        text=text,
        tool_calls=[ToolCall(id="t1", name="done", args={})],
        usage=Usage(1, 1, 0),
    )


def test_success_on_first_attempt_no_retry():
    """Happy path: inner returns immediately, sleep is never called."""
    sleeps: list[float] = []
    inner = _RecordingClient([_ok_response("first")])
    client = RetryingLLMClient(
        inner,
        retry_max=3,
        retry_base_delay_s=1.0,
        sleep=sleeps.append,
    )

    resp = client.call([Message(role="user", content="hi")])

    assert resp.text == "first"
    assert len(inner.calls) == 1
    assert sleeps == []


def test_transient_then_success_sleeps_once():
    """One TransientLLMError, then success; one sleep of base_delay."""
    sleeps: list[float] = []
    inner = _RecordingClient([
        TransientLLMError("first failure"),
        _ok_response("second"),
    ])
    client = RetryingLLMClient(
        inner,
        retry_max=3,
        retry_base_delay_s=0.5,
        sleep=sleeps.append,
    )

    resp = client.call([Message(role="user", content="hi")])

    assert resp.text == "second"
    assert len(inner.calls) == 2
    # First retry sleeps base_delay * 2**0 = 0.5
    assert sleeps == [0.5]


def test_gives_up_after_retry_max_with_exponential_backoff():
    """retry_max consecutive transients → raises; sleeps follow base*2**n schedule."""
    sleeps: list[float] = []
    # retry_max=3 means 4 total attempts. 4 transients → adapter exhausts retries.
    inner = _RecordingClient([
        TransientLLMError("1"),
        TransientLLMError("2"),
        TransientLLMError("3"),
        TransientLLMError("4"),
    ])
    client = RetryingLLMClient(
        inner,
        retry_max=3,
        retry_base_delay_s=1.0,
        sleep=sleeps.append,
    )

    with pytest.raises(TransientLLMError, match="4"):
        client.call([Message(role="user", content="hi")])

    # 4 attempts total (initial + 3 retries); sleep is called only between
    # attempts, so 3 sleeps with the exponential schedule.
    assert len(inner.calls) == 4
    assert sleeps == [1.0, 2.0, 4.0]


def test_non_transient_exception_propagates_immediately():
    """A non-TransientLLMError exception must NOT be retried."""
    sleeps: list[float] = []
    inner = _RecordingClient([RuntimeError("boom — not transient")])
    client = RetryingLLMClient(
        inner,
        retry_max=5,
        retry_base_delay_s=1.0,
        sleep=sleeps.append,
    )

    with pytest.raises(RuntimeError, match="boom"):
        client.call([Message(role="user", content="hi")])

    assert len(inner.calls) == 1
    assert sleeps == []


def test_sleep_injection_accepts_lambda_noop():
    """Tests can inject `lambda _: None` to avoid actual sleeping."""
    inner = _RecordingClient([
        TransientLLMError("flap"),
        _ok_response("done"),
    ])
    client = RetryingLLMClient(
        inner,
        retry_max=3,
        retry_base_delay_s=10.0,  # would be 10s in real time
        sleep=lambda _: None,
    )

    resp = client.call([Message(role="user", content="hi")])
    assert resp.text == "done"
    assert len(inner.calls) == 2


def test_kwargs_forwarded_to_inner():
    """messages, tools, and model must reach the wrapped client unchanged."""
    inner = _RecordingClient([_ok_response()])
    client = RetryingLLMClient(inner, sleep=lambda _: None)

    msgs = [Message(role="user", content="hi")]
    tools = [{"name": "done", "parameters": {"type": "object"}}]
    client.call(msgs, tools=tools, model="gemini-2.5-flash")

    assert inner.calls[0]["messages"] is msgs
    assert inner.calls[0]["tools"] is tools
    assert inner.calls[0]["model"] == "gemini-2.5-flash"


def test_retry_max_zero_means_no_retries():
    """retry_max=0 → exactly one attempt; transient raises with no sleeping."""
    sleeps: list[float] = []
    inner = _RecordingClient([TransientLLMError("immediate")])
    client = RetryingLLMClient(
        inner,
        retry_max=0,
        retry_base_delay_s=1.0,
        sleep=sleeps.append,
    )

    with pytest.raises(TransientLLMError, match="immediate"):
        client.call([Message(role="user", content="hi")])

    assert len(inner.calls) == 1
    assert sleeps == []
