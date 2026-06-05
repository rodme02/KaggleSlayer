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


def test_cli_resume_with_empty_journal_warns_but_continues(tmp_path, capsys):
    """F13: --resume against an empty journal must warn (not silently start
    fresh). The Solver still runs — exit code reflects solver outcome, not
    the empty journal.
    """
    comp_path = tmp_path / "comp"
    comp_path.mkdir()
    (comp_path / "raw").mkdir()
    pd.DataFrame({"x": [1, 2], "y": [0, 1]}).to_csv(comp_path / "raw" / "train.csv", index=False)
    pd.DataFrame({"id": [1], "x": [1]}).to_csv(comp_path / "raw" / "test.csv", index=False)
    # Workspace exists but no journal entries
    Workspace.create(root=comp_path)
    assert not (comp_path / "run_log.jsonl").exists()

    with patch("kaggle_slayer.cli.Solver") as mock_solver_cls, \
         patch("kaggle_slayer.cli.build_context"), \
         patch("kaggle_slayer.cli.KaggleClient"), \
         patch("kaggle_slayer.cli.GeminiClient"), \
         patch("kaggle_slayer.cli.os.environ", {"GEMINI_API_KEY": "fake"}):
        mock_solver = MagicMock()
        mock_solver.solve.return_value = MagicMock(status="done", iterations=1, summary="ok")
        mock_solver_cls.return_value = mock_solver

        exit_code = cli.run([str(comp_path), "--target", "y", "--resume"])

    captured = capsys.readouterr()
    assert "starting fresh" in captured.err
    # Solver still ran — exit code should be 0 (done), not 3 (resume failure)
    assert exit_code != 3
    mock_solver.solve.assert_called_once()


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


def test_cli_captures_unhandled_exception_to_errors_dir(tmp_path, monkeypatch):
    """An unhandled exception inside run() writes a crash report and returns 4."""
    from kaggle_slayer.harness.telemetry import errors as errors_mod
    err_dir = tmp_path / "errors"
    monkeypatch.setattr(errors_mod, "DEFAULT_DIR", err_dir)

    comp_path = tmp_path / "comp"
    comp_path.mkdir()
    (comp_path / "raw").mkdir()
    pd.DataFrame({"x": [1], "y": [0]}).to_csv(comp_path / "raw" / "train.csv", index=False)
    pd.DataFrame({"x": [1]}).to_csv(comp_path / "raw" / "test.csv", index=False)

    # Make GeminiClient construction blow up so we exercise the catch path.
    def bad_gemini(*args, **kwargs):
        raise RuntimeError("gemini boom")

    monkeypatch.setenv("GEMINI_API_KEY", "fake")
    with patch("kaggle_slayer.cli.GeminiClient", side_effect=bad_gemini), \
         patch("kaggle_slayer.cli.build_context"), \
         patch("kaggle_slayer.cli.KaggleClient"):
        exit_code = cli.run([
            str(comp_path),
            "--target", "y",
            "--auto-approve", "all",
            "--i-know-what-im-doing",
        ])

    assert exit_code == 4, f"expected exit code 4, got {exit_code}"
    captured = list(err_dir.glob("*.json"))
    assert len(captured) == 1
    import json
    rec = json.loads(captured[0].read_text())
    assert rec["exception"]["type"] == "RuntimeError"
    assert "gemini boom" in rec["exception"]["message"]


def test_cli_parses_download_flags(tmp_path):
    args = cli._parse_args([
        str(tmp_path / "comp"), "--target", "y",
        "--no-download", "--competition", "titanic",
    ])
    assert args.no_download is True
    assert args.competition == "titanic"


def test_cli_download_flags_default(tmp_path):
    args = cli._parse_args([str(tmp_path / "comp"), "--target", "y"])
    assert args.no_download is False
    assert args.competition is None


def test_cli_no_download_passes_enabled_false(tmp_path):
    """--no-download must reach ensure_competition_data as enabled=False."""
    comp_path = tmp_path / "comp"
    comp_path.mkdir()
    raw = comp_path / "raw"
    raw.mkdir()
    pd.DataFrame({"x": [1], "y": [0]}).to_csv(raw / "train.csv", index=False)
    pd.DataFrame({"x": [1]}).to_csv(raw / "test.csv", index=False)

    with patch("kaggle_slayer.cli.ensure_competition_data") as mock_ensure, \
         patch("kaggle_slayer.cli.Solver") as mock_solver_cls, \
         patch("kaggle_slayer.cli.build_context"), \
         patch("kaggle_slayer.cli.KaggleClient"), \
         patch("kaggle_slayer.cli.GeminiClient"), \
         patch("kaggle_slayer.cli.os.environ", {"GEMINI_API_KEY": "fake"}):
        mock_solver = MagicMock()
        mock_solver.solve.return_value = MagicMock(status="done", iterations=1, summary="")
        mock_solver_cls.return_value = mock_solver

        cli.run([str(comp_path), "--target", "y", "--no-download"])

    mock_ensure.assert_called_once()
    assert mock_ensure.call_args.kwargs["enabled"] is False


def test_cli_download_failure_exits_2(tmp_path):
    """A DownloadError from a needed fetch hard-exits with code 2."""
    comp_path = tmp_path / "comp"

    with patch(
        "kaggle_slayer.cli.ensure_competition_data",
        side_effect=cli.DownloadError("titanic", RuntimeError("403 Forbidden")),
    ), \
         patch("kaggle_slayer.cli.KaggleClient"), \
         patch("kaggle_slayer.cli.os.environ", {"GEMINI_API_KEY": "fake"}):
        exit_code = cli.run([str(comp_path), "--target", "y"])

    assert exit_code == 2
