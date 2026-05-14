"""Tests for kaggle_slayer.agent.llm_client.

This task covers the protocol + dataclasses. Gemini implementation tests
land in Task 9.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from kaggle_slayer.agent import llm_client as llm
from kaggle_slayer.agent.cost_ledger import CostLedger


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


def _fake_genai_response(text: str, in_tok: int = 10, out_tok: int = 5, cached_tok: int = 0):
    resp = MagicMock()
    resp.text = text
    # Single text part, no function_call — mirrors what the real Gemini SDK
    # returns for plain text responses.
    part = MagicMock()
    part.function_call = None
    part.text = text
    candidate = MagicMock()
    candidate.content.parts = [part]
    resp.candidates = [candidate]
    usage = MagicMock()
    usage.prompt_token_count = in_tok
    usage.candidates_token_count = out_tok
    usage.cached_content_token_count = cached_tok
    resp.usage_metadata = usage
    return resp


def test_gemini_client_call_returns_response(tmp_path):
    ledger = CostLedger(path=tmp_path / "cost.jsonl")
    with patch("kaggle_slayer.agent.llm_client._make_genai_client") as mock_factory:
        client_impl = MagicMock()
        client_impl.models.generate_content.return_value = _fake_genai_response("hello", 100, 50)
        mock_factory.return_value = client_impl

        client = llm.GeminiClient(api_key="fake", ledger=ledger, competition="test-comp")
        resp = client.call(
            messages=[llm.Message(role="user", content="hi")],
            model="gemini-2.5-flash",
        )

    assert resp.text == "hello"
    assert resp.usage.input_tokens == 100
    assert resp.usage.output_tokens == 50
    # Ledger should have one entry attributed to test-comp
    assert ledger.total_for(competition="test-comp") > 0


def test_gemini_client_retries_on_transient_error(tmp_path):
    """Transient errors (rate limit, 5xx) should retry up to 3 times."""
    ledger = CostLedger(path=tmp_path / "cost.jsonl")
    with patch("kaggle_slayer.agent.llm_client._make_genai_client") as mock_factory:
        client_impl = MagicMock()
        # Two transient errors, then success
        good = _fake_genai_response("ok", 5, 2)
        client_impl.models.generate_content.side_effect = [
            llm.TransientLLMError("rate limit"),
            llm.TransientLLMError("temporarily unavailable"),
            good,
        ]
        mock_factory.return_value = client_impl

        client = llm.GeminiClient(
            api_key="fake", ledger=ledger, competition="x",
            retry_max=3, retry_base_delay_s=0.0,
        )
        resp = client.call(messages=[llm.Message(role="user", content="hi")])

    assert resp.text == "ok"
    assert client_impl.models.generate_content.call_count == 3


def test_gemini_client_gives_up_after_retry_max(tmp_path):
    ledger = CostLedger(path=tmp_path / "cost.jsonl")
    with patch("kaggle_slayer.agent.llm_client._make_genai_client") as mock_factory:
        client_impl = MagicMock()
        client_impl.models.generate_content.side_effect = llm.TransientLLMError("nope")
        mock_factory.return_value = client_impl

        client = llm.GeminiClient(
            api_key="fake", ledger=ledger, competition="x",
            retry_max=2, retry_base_delay_s=0.0,
        )
        with pytest.raises(llm.TransientLLMError):
            client.call(messages=[llm.Message(role="user", content="hi")])
    assert client_impl.models.generate_content.call_count == 3  # initial + 2 retries


def test_gemini_client_does_not_retry_on_permanent_error(tmp_path):
    """Auth errors, malformed requests, etc. should NOT retry."""
    ledger = CostLedger(path=tmp_path / "cost.jsonl")

    class _AuthError(Exception):
        pass

    with patch("kaggle_slayer.agent.llm_client._make_genai_client") as mock_factory:
        client_impl = MagicMock()
        client_impl.models.generate_content.side_effect = _AuthError("invalid key")
        mock_factory.return_value = client_impl

        client = llm.GeminiClient(
            api_key="fake", ledger=ledger, competition="x",
            retry_max=3, retry_base_delay_s=0.0,
        )
        with pytest.raises(_AuthError):
            client.call(messages=[llm.Message(role="user", content="hi")])
    assert client_impl.models.generate_content.call_count == 1


def test_is_transient_uses_status_code_attribute_first():
    """If an exception has a status_code attribute, prefer it over substring matching."""
    class HttpError(Exception):
        def __init__(self, code):
            super().__init__("http error")
            self.status_code = code
    assert llm._is_transient(HttpError(429)) is True
    assert llm._is_transient(HttpError(503)) is True
    assert llm._is_transient(HttpError(500)) is True
    # 4xx other than 429 should not retry
    assert llm._is_transient(HttpError(400)) is False
    assert llm._is_transient(HttpError(401)) is False
    assert llm._is_transient(HttpError(403)) is False
    assert llm._is_transient(HttpError(404)) is False


def test_is_transient_does_not_false_positive_on_model_name():
    """A permanent error mentioning a model name with '503' in it must not retry."""
    err = ValueError("model 'gemini-503-test' is not supported")
    # No status_code attribute, no transient keyword — must be permanent
    assert llm._is_transient(err) is False


def test_is_transient_still_matches_keyword_when_no_status_code():
    """Backstop substring match still works for SDKs without status codes."""
    assert llm._is_transient(Exception("rate limit exceeded")) is True
    assert llm._is_transient(Exception("temporarily unavailable")) is True
    assert llm._is_transient(Exception("connection timeout")) is True
