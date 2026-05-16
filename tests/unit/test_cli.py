"""Tests for kaggle_slayer.cli."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from kaggle_slayer import cli
from kaggle_slayer.harness.journal import Journal
from kaggle_slayer.harness.workspace import Workspace


def test_cli_parses_args(tmp_path):
    """Parser accepts a workspace path and optional --target/--metric/--problem-type."""
    args = cli._parse_args([
        str(tmp_path / "comp"),
        "--target", "Survived",
        "--metric", "accuracy",
        "--problem-type", "classification",
        "--max-iterations", "10",
    ])
    assert args.workspace_path == str(tmp_path / "comp")
    assert args.target == "Survived"
    assert args.metric == "accuracy"
    assert args.problem_type == "classification"
    assert args.max_iterations == 10


def test_cli_requires_workspace_path():
    with pytest.raises(SystemExit):
        cli._parse_args([])


def test_cli_default_model_is_flash(tmp_path):
    """F5: --model defaults to gemini-2.5-flash (the validated slow-tier model).

    Pro is unvalidated and 10x more expensive, plus it hits free-tier quota
    walls. Flash is the safe default; Pro is opt-in via --model gemini-2.5-pro.
    """
    args = cli._parse_args([str(tmp_path / "comp")])
    assert args.model == "gemini-2.5-flash"


def test_cli_model_can_be_overridden(tmp_path):
    """F5: --model can be overridden to a different Gemini model id."""
    args = cli._parse_args([str(tmp_path), "--model", "gemini-2.5-pro"])
    assert args.model == "gemini-2.5-pro"


def test_cli_run_creates_workspace_and_calls_solver(tmp_path):
    """run() with a fake LLMClient creates the workspace and invokes the solver."""
    comp_path = tmp_path / "comp"

    # Pre-populate raw/train.csv and raw/test.csv so the workspace looks real
    comp_path.mkdir()
    raw = comp_path / "raw"
    raw.mkdir()
    pd.DataFrame({"x1": [1, 2, 3], "Survived": [0, 1, 0]}).to_csv(raw / "train.csv", index=False)
    pd.DataFrame({"id": [1, 2], "x1": [1, 2]}).to_csv(raw / "test.csv", index=False)

    # Mock context builder and Solver to avoid real Kaggle/LLM calls
    with patch("kaggle_slayer.cli.Solver") as mock_solver_cls, \
         patch("kaggle_slayer.cli.build_context"), \
         patch("kaggle_slayer.cli.KaggleClient"), \
         patch("kaggle_slayer.cli.GeminiClient"), \
         patch("kaggle_slayer.cli.os.environ", {"GEMINI_API_KEY": "fake"}):

        mock_solver = MagicMock()
        mock_solver.solve.return_value = MagicMock(
            status="done", iterations=3, summary="best CV=0.85"
        )
        mock_solver_cls.return_value = mock_solver

        exit_code = cli.run([
            str(comp_path),
            "--target", "Survived",
            "--metric", "accuracy",
            "--problem-type", "classification",
        ])

    assert exit_code == 0
    mock_solver_cls.assert_called_once()
    mock_solver.solve.assert_called_once()
    # Workspace structure was created
    assert (comp_path / "agent").is_dir()
    assert (comp_path / "submissions").is_dir()


def test_cli_run_exits_nonzero_when_solver_does_not_finish(tmp_path):
    comp_path = tmp_path / "comp"
    comp_path.mkdir()
    (comp_path / "raw").mkdir()
    pd.DataFrame({"x": [1], "y": [0]}).to_csv(comp_path / "raw" / "train.csv", index=False)
    pd.DataFrame({"x": [1]}).to_csv(comp_path / "raw" / "test.csv", index=False)

    with patch("kaggle_slayer.cli.Solver") as mock_solver_cls, \
         patch("kaggle_slayer.cli.build_context"), \
         patch("kaggle_slayer.cli.KaggleClient"), \
         patch("kaggle_slayer.cli.GeminiClient"), \
         patch("kaggle_slayer.cli.os.environ", {"GEMINI_API_KEY": "fake"}):

        mock_solver = MagicMock()
        mock_solver.solve.return_value = MagicMock(
            status="max_iterations", iterations=25, summary=""
        )
        mock_solver_cls.return_value = mock_solver

        exit_code = cli.run([str(comp_path), "--target", "y"])
    assert exit_code != 0


def test_cli_parses_resume_flag(tmp_path):
    args = cli._parse_args([str(tmp_path / "comp"), "--resume", "--target", "y"])
    assert args.resume is True


def test_cli_parses_cost_budget_flag(tmp_path):
    args = cli._parse_args([str(tmp_path / "comp"), "--cost-budget", "0.25", "--target", "y"])
    assert args.cost_budget == 0.25


def test_cli_parses_auto_approve_flag(tmp_path):
    args = cli._parse_args([str(tmp_path / "comp"), "--auto-approve", "safe", "--target", "y"])
    assert args.auto_approve == "safe"


def test_cli_auto_approve_all_requires_i_know_what_im_doing(tmp_path):
    """F5: --auto-approve all without --i-know-what-im-doing must exit 2.

    The 'all' mode auto-approves every checkpoint trigger — including the
    Kaggle daily-submit cap, cost-budget overrun, and metric overrides.
    Forcing a second flag prevents a copy-pasted command line from
    silently bypassing those gates.
    """
    with pytest.raises(SystemExit) as ex:
        cli._parse_args([str(tmp_path / "comp"), "--target", "y", "--auto-approve", "all"])
    assert ex.value.code == 2


def test_cli_auto_approve_all_with_acknowledgement_parses(tmp_path):
    """F5: --auto-approve all + --i-know-what-im-doing parses successfully."""
    args = cli._parse_args([
        str(tmp_path / "comp"),
        "--target", "y",
        "--auto-approve", "all",
        "--i-know-what-im-doing",
    ])
    assert args.auto_approve == "all"
    assert args.i_know_what_im_doing is True


def test_cli_resume_skips_context_build_by_default(tmp_path):
    """F6: --resume should NOT rebuild context.md by default — doing so
    would regenerate context.md while the resumed conversation history
    still references the original, causing replay drift.
    """
    comp_path = tmp_path / "comp"
    comp_path.mkdir()
    (comp_path / "raw").mkdir()
    pd.DataFrame({"x": [1, 2], "y": [0, 1]}).to_csv(comp_path / "raw" / "train.csv", index=False)
    pd.DataFrame({"id": [1], "x": [1]}).to_csv(comp_path / "raw" / "test.csv", index=False)
    ws = Workspace.create(root=comp_path)
    Journal(ws).log_tool_call(tool="take_note", args={"category": "observation", "content": "x"}, result_summary="noted")

    with patch("kaggle_slayer.cli.Solver") as mock_solver_cls, \
         patch("kaggle_slayer.cli.build_context") as mock_bc, \
         patch("kaggle_slayer.cli.KaggleClient"), \
         patch("kaggle_slayer.cli.GeminiClient"), \
         patch("kaggle_slayer.cli.os.environ", {"GEMINI_API_KEY": "fake"}):
        mock_solver = MagicMock()
        mock_solver.solve.return_value = MagicMock(status="done", iterations=1, summary="ok")
        mock_solver_cls.return_value = mock_solver

        cli.run([str(comp_path), "--target", "y", "--resume"])

    mock_bc.assert_not_called()


def test_cli_resume_with_rebuild_context_runs_context_build(tmp_path):
    """F6: --resume --rebuild-context is the explicit opt-in for re-running
    build_context on a resumed run.
    """
    comp_path = tmp_path / "comp"
    comp_path.mkdir()
    (comp_path / "raw").mkdir()
    pd.DataFrame({"x": [1, 2], "y": [0, 1]}).to_csv(comp_path / "raw" / "train.csv", index=False)
    pd.DataFrame({"id": [1], "x": [1]}).to_csv(comp_path / "raw" / "test.csv", index=False)
    ws = Workspace.create(root=comp_path)
    Journal(ws).log_tool_call(tool="take_note", args={"category": "observation", "content": "x"}, result_summary="noted")

    with patch("kaggle_slayer.cli.Solver") as mock_solver_cls, \
         patch("kaggle_slayer.cli.build_context") as mock_bc, \
         patch("kaggle_slayer.cli.KaggleClient"), \
         patch("kaggle_slayer.cli.GeminiClient"), \
         patch("kaggle_slayer.cli.os.environ", {"GEMINI_API_KEY": "fake"}):
        mock_solver = MagicMock()
        mock_solver.solve.return_value = MagicMock(status="done", iterations=1, summary="ok")
        mock_solver_cls.return_value = mock_solver

        cli.run([str(comp_path), "--target", "y", "--resume", "--rebuild-context"])

    mock_bc.assert_called_once()


def test_cli_resume_passes_rebuilt_history_to_solver(tmp_path):
    """When --resume is set and run_log.jsonl has prior tool calls, those
    messages are passed via solve(resume_from=...)."""
    comp_path = tmp_path / "comp"
    comp_path.mkdir()
    (comp_path / "raw").mkdir()
    pd.DataFrame({"x": [1, 2], "y": [0, 1]}).to_csv(comp_path / "raw" / "train.csv", index=False)
    pd.DataFrame({"id": [1], "x": [1]}).to_csv(comp_path / "raw" / "test.csv", index=False)
    # Seed a journal entry that the resume should pick up
    ws = Workspace.create(root=comp_path)
    Journal(ws).log_tool_call(tool="take_note", args={"category": "observation", "content": "x"}, result_summary="noted")

    with patch("kaggle_slayer.cli.Solver") as mock_solver_cls, \
         patch("kaggle_slayer.cli.build_context"), \
         patch("kaggle_slayer.cli.KaggleClient"), \
         patch("kaggle_slayer.cli.GeminiClient"), \
         patch("kaggle_slayer.cli.os.environ", {"GEMINI_API_KEY": "fake"}):
        mock_solver = MagicMock()
        mock_solver.solve.return_value = MagicMock(status="done", iterations=1, summary="ok")
        mock_solver_cls.return_value = mock_solver

        cli.run([str(comp_path), "--target", "y", "--resume"])

    # solve() was called with resume_from=<rebuilt history>
    call_kwargs = mock_solver.solve.call_args.kwargs
    assert "resume_from" in call_kwargs
    assert call_kwargs["resume_from"] is not None
    assert len(call_kwargs["resume_from"]) >= 2  # at least one model + tool message
