"""Tests for kaggle_slayer.agent.handlers.files."""

from __future__ import annotations

import json
from dataclasses import dataclass

import pandas as pd
import pytest

from kaggle_slayer.agent.handlers import files as fh
from kaggle_slayer.agent.tools import ToolError
from kaggle_slayer.harness.journal import Journal
from kaggle_slayer.harness.workspace import Workspace


@dataclass
class _Ctx:
    """Minimal SolverContext stand-in: just a workspace + journal for these handlers."""
    workspace: Workspace
    journal: Journal


@pytest.fixture
def ctx(tmp_path):
    ws = Workspace.create(root=tmp_path / "comp")
    return _Ctx(workspace=ws, journal=Journal(ws))


def test_read_context_returns_file_contents(ctx):
    ctx.workspace.context_path.write_text("# Test Comp\n\nMetric: accuracy.\n")
    result = fh.read_context(ctx)
    assert "Test Comp" in result
    assert "Metric: accuracy" in result


def test_read_context_missing_raises_tool_error(ctx):
    with pytest.raises(ToolError, match="context.md"):
        fh.read_context(ctx)


def test_read_file_inside_workspace(ctx):
    p = ctx.workspace.agent_dir / "fe.py"
    p.write_text("def fit_feature_transformer(df, t): pass")
    result = fh.read_file(ctx, path="agent/fe.py")
    assert "fit_feature_transformer" in result


def test_read_file_outside_workspace_rejected(ctx):
    """Path traversal must be blocked."""
    with pytest.raises(ToolError, match="outside workspace"):
        fh.read_file(ctx, path="../../etc/passwd")


def test_read_file_missing_raises_tool_error(ctx):
    with pytest.raises(ToolError, match="not found"):
        fh.read_file(ctx, path="agent/does_not_exist.py")


def test_write_file_creates_under_agent_dir(ctx):
    fh.write_file(ctx, path="agent/fe.py", content="def fit_feature_transformer(df, t): pass\n")
    assert (ctx.workspace.agent_dir / "fe.py").read_text() == "def fit_feature_transformer(df, t): pass\n"


def test_write_file_overwrites(ctx):
    fh.write_file(ctx, path="agent/fe.py", content="v1")
    fh.write_file(ctx, path="agent/fe.py", content="v2")
    assert (ctx.workspace.agent_dir / "fe.py").read_text() == "v2"


def test_write_file_creates_parent_dirs(ctx):
    fh.write_file(ctx, path="agent/scratch/probe.py", content="x = 1")
    assert (ctx.workspace.agent_dir / "scratch" / "probe.py").exists()


def test_write_file_outside_workspace_rejected(ctx):
    with pytest.raises(ToolError, match="outside workspace"):
        fh.write_file(ctx, path="../escape.py", content="x = 1")


def test_write_file_rejects_protected_paths(ctx):
    """run_log.jsonl, notes.jsonl, context.md must not be writable from a tool."""
    for forbidden in ("run_log.jsonl", "notes.jsonl", "context.md"):
        with pytest.raises(ToolError, match="protected"):
            fh.write_file(ctx, path=forbidden, content="x")


@pytest.mark.parametrize(
    "name",
    ["Run_Log.jsonl", "RUN_LOG.JSONL", "NOTES.JSONL", "Context.MD"],
)
def test_write_file_rejects_protected_paths_case_insensitive(ctx, name):
    """F3: protected basenames must match regardless of case."""
    with pytest.raises(ToolError, match="protected"):
        fh.write_file(ctx, path=name, content="x")


def test_write_file_rejects_empty_path(ctx):
    """F17: empty path resolves to the workspace root — reject with a
    clear ToolError, not a generic IsADirectoryError."""
    with pytest.raises(ToolError, match=r"directory|file path"):
        fh.write_file(ctx, path="", content="x")


def test_write_file_rejects_directory_path(ctx):
    """F17: a path that resolves to an existing directory must be rejected."""
    with pytest.raises(ToolError, match=r"directory|file path"):
        fh.write_file(ctx, path="agent", content="x")


def test_write_file_rejects_leaderboard_jsonl(ctx):
    """submissions/leaderboard.jsonl is the submit gate's evidence — the
    agent must not be able to rewrite it via write_file."""
    with pytest.raises(ToolError, match="protected"):
        fh.write_file(ctx, path="submissions/leaderboard.jsonl", content="{}")


def test_write_file_rejects_leaderboard_jsonl_case_insensitive(ctx):
    with pytest.raises(ToolError, match="protected"):
        fh.write_file(ctx, path="submissions/Leaderboard.JSONL", content="{}")


def test_write_file_rejects_raw_paths(ctx):
    """Competition data is read-only for the agent; overwriting raw/train.csv
    would silently corrupt every later train_cv."""
    with pytest.raises(ToolError, match="read-only"):
        fh.write_file(ctx, path="raw/train.csv", content="id\n1\n")


def test_write_file_still_allows_submission_csvs(ctx):
    """Only leaderboard.jsonl is protected under submissions/ — ordinary
    CSVs the agent stages there are fine."""
    fh.write_file(ctx, path="submissions/preds.csv", content="id,target\n1,0\n")
    assert (ctx.workspace.submissions_dir / "preds.csv").exists()


def test_sample_rows_returns_first_n_rows(ctx):
    df = pd.DataFrame({"a": range(20), "b": list("abcdefghijklmnopqrst")})
    df.to_csv(ctx.workspace.raw_dir / "train.csv", index=False)
    result = fh.sample_rows(ctx, table="train", n=5)
    # Result is a stringified table with the first 5 rows
    assert "a" in result and "b" in result
    assert "0" in result and "4" in result
    # Should not include rows 5+
    assert "10" not in result or result.count("\n") <= 8  # tolerate header + 5 data lines


def test_sample_rows_random_sampling_is_deterministic_with_seed(ctx):
    """sample_rows(random=True) uses a fixed random_state, so repeated calls
    return identical samples — important for reproducible debugging."""
    df = pd.DataFrame({"a": range(100)})
    df.to_csv(ctx.workspace.raw_dir / "train.csv", index=False)
    r1 = fh.sample_rows(ctx, table="train", n=5, random=True)
    r2 = fh.sample_rows(ctx, table="train", n=5, random=True)
    assert r1 == r2, "expected deterministic sampling for reproducibility"
    # And sanity-check that random sampling differs from head() for a large df.
    r_head = fh.sample_rows(ctx, table="train", n=5, random=False)
    assert r1 != r_head, "random sample should not match head() for 100 rows"


def test_sample_rows_missing_table_raises(ctx):
    with pytest.raises(ToolError, match="train.csv"):
        fh.sample_rows(ctx, table="train", n=3)


def test_sample_rows_rejects_path_traversal(ctx):
    """`table` must not be allowed to escape raw/ via '../'."""
    with pytest.raises(ToolError, match=r"outside|traversal|invalid"):
        fh.sample_rows(ctx, table="../../etc/passwd", n=5)


def test_sample_rows_rejects_absolute_path(ctx):
    """Absolute paths in `table` must be rejected the same way."""
    with pytest.raises(ToolError, match=r"outside|traversal|invalid|absolute"):
        fh.sample_rows(ctx, table="/etc/passwd", n=5)


def test_sample_rows_caps_at_table_size(ctx):
    df = pd.DataFrame({"a": [1, 2, 3]})
    df.to_csv(ctx.workspace.raw_dir / "train.csv", index=False)
    result = fh.sample_rows(ctx, table="train", n=100)
    # Should just return all 3 rows rather than raise
    assert "1" in result and "3" in result


def test_take_note_appends_to_notes_jsonl(ctx):
    fh.take_note(ctx, category="observation", content="target is imbalanced (3%)")
    lines = ctx.workspace.notes_path.read_text().splitlines()
    rec = json.loads(lines[0])
    assert rec["category"] == "observation"
    assert "imbalanced" in rec["content"]


def test_take_note_rejects_unknown_category(ctx):
    with pytest.raises(ToolError, match="category"):
        fh.take_note(ctx, category="invalid_cat", content="x")
