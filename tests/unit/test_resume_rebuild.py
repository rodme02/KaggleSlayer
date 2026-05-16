"""Tests for resume.rebuild_conversation."""

from __future__ import annotations

import pytest

from kaggle_slayer.harness import resume
from kaggle_slayer.harness.journal import Journal
from kaggle_slayer.harness.workspace import Workspace


@pytest.fixture
def ws(tmp_path):
    return Workspace.create(root=tmp_path / "comp")


def test_rebuild_empty_journal_returns_empty_list(ws):
    """No prior runs → empty resume list."""
    assert resume.rebuild_conversation(ws) == []


def test_rebuild_includes_one_model_one_tool_per_call(ws):
    """Each tool_call record becomes a model(function_call) + tool(function_response)."""
    j = Journal(ws)
    j.log_tool_call(tool="read_context", args={}, result_summary="# Comp\nMetric: accuracy")
    j.log_tool_call(tool="write_file", args={"path": "agent/fe.py", "content": "..."},
                    result_summary="wrote 3 bytes")
    msgs = resume.rebuild_conversation(ws)
    # 2 tool calls → 4 messages (2 × (model + tool))
    assert len(msgs) == 4
    assert msgs[0].role == "model"
    assert msgs[0].tool_calls[0].name == "read_context"
    assert msgs[1].role == "tool"
    assert "Comp" in msgs[1].content
    assert msgs[2].role == "model"
    assert msgs[2].tool_calls[0].name == "write_file"
    assert msgs[2].tool_calls[0].args == {"path": "agent/fe.py", "content": "..."}
    assert msgs[3].role == "tool"
    assert "3 bytes" in msgs[3].content


def test_rebuild_handles_tool_errors_as_model_call_plus_error_response(ws):
    """tool_error records also emit model+tool pairs (result is the error message)."""
    j = Journal(ws)
    j.log_tool_error(tool="write_file", args={"path": "context.md", "content": "x"},
                     error="ToolError: path 'context.md' is protected")
    msgs = resume.rebuild_conversation(ws)
    assert len(msgs) == 2
    assert msgs[0].role == "model"
    assert msgs[0].tool_calls[0].name == "write_file"
    assert msgs[1].role == "tool"
    assert "protected" in msgs[1].content


def test_rebuild_skips_checkpoint_records(ws):
    """checkpoint records are journalled but not part of the LLM conversation."""
    j = Journal(ws)
    j._append(ws.run_log_path, {  # noqa: SLF001
        "ts": "2026-05-15", "kind": "checkpoint", "trigger": "set_metric",
        "action": "change metric", "evidence": {}, "decision": "approve",
    })
    j.log_tool_call(tool="train_cv", args={}, result_summary="mean=0.82")
    msgs = resume.rebuild_conversation(ws)
    # Only the train_cv call should produce messages — checkpoint stays in the log.
    assert len(msgs) == 2
    assert msgs[0].tool_calls[0].name == "train_cv"


def test_rebuild_raises_when_done_already_called(ws):
    """A workspace whose last tool_call was 'done' has nothing to resume."""
    j = Journal(ws)
    j.log_tool_call(tool="write_file", args={"path": "x", "content": "y"}, result_summary="ok")
    j.log_tool_call(tool="done", args={"summary": "all done"}, result_summary="ack")
    with pytest.raises(resume.ResumeError, match="already finished"):
        resume.rebuild_conversation(ws)


def test_rebuild_handles_missing_log_file(ws):
    """A workspace without run_log.jsonl returns an empty list."""
    # Don't write anything; run_log_path doesn't exist
    assert not ws.run_log_path.exists()
    assert resume.rebuild_conversation(ws) == []


def test_rebuild_preserves_original_tool_call_id(ws):
    """F8: when the journal record stores a tool_call_id, rebuild uses it
    instead of fabricating 'resume_<n>'. A stricter LLM provider would
    reject fabricated IDs whose paired function_response doesn't reference
    the original function_call id.
    """
    j = Journal(ws)
    j.log_tool_call(
        tool="take_note",
        args={"category": "observation", "content": "x"},
        result_summary="noted",
        tool_call_id="orig_42",
    )
    msgs = resume.rebuild_conversation(ws)
    assert msgs[0].tool_calls[0].id == "orig_42"


def test_rebuild_falls_back_to_resume_id_when_missing(ws):
    """F8: back-compat with old journals that don't carry tool_call_id."""
    j = Journal(ws)
    # log_tool_call without tool_call_id (default behaviour)
    j.log_tool_call(tool="take_note", args={"category": "observation", "content": "x"}, result_summary="noted")
    msgs = resume.rebuild_conversation(ws)
    assert msgs[0].tool_calls[0].id == "resume_0"
