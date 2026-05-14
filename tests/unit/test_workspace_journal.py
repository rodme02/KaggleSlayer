"""Tests for kaggle_slayer.harness.journal.Journal."""

from __future__ import annotations

import json

import pytest

from kaggle_slayer.harness import journal as journal_mod
from kaggle_slayer.harness.workspace import Workspace


@pytest.fixture
def fresh_workspace(tmp_path):
    return Workspace.create(root=tmp_path / "comp")


def test_log_tool_call_appends_one_line(fresh_workspace):
    j = journal_mod.Journal(fresh_workspace)
    j.log_tool_call(
        tool="load_competition",
        args={"name": "titanic"},
        result_summary="loaded 891 train rows, 418 test rows",
    )
    lines = fresh_workspace.run_log_path.read_text().splitlines()
    assert len(lines) == 1
    rec = json.loads(lines[0])
    assert rec["kind"] == "tool_call"
    assert rec["tool"] == "load_competition"
    assert rec["args"] == {"name": "titanic"}
    assert rec["result_summary"].startswith("loaded 891")
    assert "ts" in rec  # ISO timestamp


def test_log_multiple_tool_calls_are_appended(fresh_workspace):
    j = journal_mod.Journal(fresh_workspace)
    for i in range(3):
        j.log_tool_call(tool="probe", args={"i": i}, result_summary=f"r{i}")
    lines = fresh_workspace.run_log_path.read_text().splitlines()
    assert len(lines) == 3
    assert [json.loads(line)["args"]["i"] for line in lines] == [0, 1, 2]


def test_log_tool_error_records_failure(fresh_workspace):
    j = journal_mod.Journal(fresh_workspace)
    j.log_tool_error(
        tool="submit_kaggle",
        args={"comp": "titanic"},
        error="403 Forbidden: rules not accepted",
    )
    rec = json.loads(fresh_workspace.run_log_path.read_text().splitlines()[0])
    assert rec["kind"] == "tool_error"
    assert rec["error"] == "403 Forbidden: rules not accepted"


def test_take_note_writes_to_notes_jsonl(fresh_workspace):
    j = journal_mod.Journal(fresh_workspace)
    j.take_note(category="observation", content="target column is heavily imbalanced (10% positive)")
    lines = fresh_workspace.notes_path.read_text().splitlines()
    assert len(lines) == 1
    rec = json.loads(lines[0])
    assert rec["category"] == "observation"
    assert rec["content"].startswith("target column")


def test_take_note_rejects_unknown_category(fresh_workspace):
    j = journal_mod.Journal(fresh_workspace)
    with pytest.raises(ValueError, match="unknown category"):
        j.take_note(category="random", content="x")


def test_list_notes_filters_by_category(fresh_workspace):
    j = journal_mod.Journal(fresh_workspace)
    j.take_note(category="observation", content="a")
    j.take_note(category="decision", content="b")
    j.take_note(category="observation", content="c")
    obs = j.list_notes(category="observation")
    assert len(obs) == 2
    assert [n["content"] for n in obs] == ["a", "c"]


def test_iter_tool_calls_returns_dicts(fresh_workspace):
    j = journal_mod.Journal(fresh_workspace)
    j.log_tool_call(tool="a", args={}, result_summary="x")
    j.log_tool_call(tool="b", args={}, result_summary="y")
    records = list(j.iter_records())
    assert [r["tool"] for r in records] == ["a", "b"]


def test_append_before_return_is_durable(fresh_workspace, monkeypatch):
    """If we crash immediately after a journal call, the record must be on disk."""
    j = journal_mod.Journal(fresh_workspace)
    j.log_tool_call(tool="profile_data", args={}, result_summary="ok")
    # Read the file back from a fresh Journal — should see the entry
    j2 = journal_mod.Journal(fresh_workspace)
    records = list(j2.iter_records())
    assert len(records) == 1
    assert records[0]["tool"] == "profile_data"


def test_iter_records_skips_truncated_trailing_line(fresh_workspace):
    """If the process crashed mid-write, the last line may be partial JSON.
    iter_records must skip it rather than crashing."""
    j = journal_mod.Journal(fresh_workspace)
    j.log_tool_call(tool="a", args={}, result_summary="ok")
    j.log_tool_call(tool="b", args={}, result_summary="ok")
    # Simulate a truncated trailing line (crash before the newline+fsync)
    with fresh_workspace.run_log_path.open("a") as f:
        f.write('{"tool":"c","kind":"tool_ca')  # NO newline, NO closing brace
    records = list(j.iter_records())
    # Should yield the two complete records and skip the partial one
    assert [r["tool"] for r in records] == ["a", "b"]
