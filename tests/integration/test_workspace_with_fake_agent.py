"""Week 2 acceptance integration test.

Threads together Workspace + Journal + KaggleClient (mocked) + context
builder + FakeLLMClient. The fake "agent" makes one LLM call, then the
harness journals it; we verify the journal is correct and complete.

This is plumbing-level, not yet the agent loop (which lands in Week 3).
"""

from __future__ import annotations

import json

import pandas as pd
import pytest

from kaggle_slayer.agent.llm_client import Message
from kaggle_slayer.harness.context import build_context
from kaggle_slayer.harness.journal import Journal
from kaggle_slayer.harness.kaggle_client import CompetitionFile, CompetitionInfo, LBEntry
from kaggle_slayer.harness.workspace import Workspace
from tests.fixtures.fake_llm import FakeLLMClient, ScriptedResponse

pytestmark = pytest.mark.integration


class _FakeKaggle:
    def view_competition(self, name):
        return CompetitionInfo(
            title="Fake Comp", description="A synthetic competition.",
            metric="accuracy",
        )

    def list_files(self, name):
        return [CompetitionFile(name="train.csv", size=12345)]

    def get_leaderboard(self, name, *, top_n=10):
        return [LBEntry(team_name="alpha", score=0.95)]


def test_fake_agent_loop_journals_each_step(tmp_path):
    # --- setup ---
    workspace = Workspace.create(root=tmp_path / "competitions" / "fake")

    # Write a tiny train.csv so context builder can profile it
    pd.DataFrame({
        "x1": range(10),
        "Survived": [0, 1] * 5,
    }).to_csv(workspace.raw_dir / "train.csv", index=False)

    kaggle = _FakeKaggle()
    journal = Journal(workspace)

    fake_llm = FakeLLMClient(script=[
        ScriptedResponse(text="I have read the context. My plan: train LightGBM."),
        ScriptedResponse(text="train_cv returned cv=0.82. Submitting."),
    ])

    # --- run ---
    # Step 1: build context, log it as a tool call
    ctx_path = build_context(workspace=workspace, kaggle_client=kaggle)
    journal.log_tool_call(
        tool="build_context",
        args={"competition": "fake"},
        result_summary=f"wrote {ctx_path.name}",
    )

    # Step 2: pretend "agent" reads the context and makes a planning call
    resp1 = fake_llm.call(messages=[
        Message(role="system", content=ctx_path.read_text()),
        Message(role="user", content="Plan a solution."),
    ])
    journal.log_tool_call(
        tool="llm_call",
        args={"role": "planner"},
        result_summary=resp1.text[:80],
    )

    # Step 3: pretend an action happened
    journal.log_tool_call(
        tool="train_cv",
        args={"fe": "agent/fe.py", "model": "agent/model.py"},
        result_summary="cv=0.82",
    )

    # Step 4: a second LLM call reading the result
    resp2 = fake_llm.call(messages=[
        Message(role="user", content="train_cv returned cv=0.82. What now?"),
    ])
    journal.log_tool_call(
        tool="llm_call",
        args={"role": "post_train"},
        result_summary=resp2.text[:80],
    )

    # --- asserts ---
    # context.md exists and has the right structure
    body = ctx_path.read_text()
    assert "Fake Comp" in body
    assert "Survived" in body

    # run_log has exactly 4 entries in order
    records = [json.loads(line) for line in workspace.run_log_path.read_text().splitlines()]
    assert len(records) == 4
    assert [r["tool"] for r in records] == [
        "build_context", "llm_call", "train_cv", "llm_call",
    ]
    assert all(r["kind"] == "tool_call" for r in records)

    # FakeLLMClient was called exactly twice
    assert len(fake_llm.calls) == 2
    assert fake_llm.calls[0].messages[-1].content == "Plan a solution."


def test_fake_agent_loop_resume_summary(tmp_path):
    """After a partial run, resume.summarize() should describe what happened."""
    workspace = Workspace.create(root=tmp_path / "competitions" / "fake")
    journal = Journal(workspace)

    journal.log_tool_call(tool="build_context", args={}, result_summary="ok")
    journal.log_tool_call(tool="train_cv", args={"v": 1}, result_summary="cv=0.7")
    journal.log_tool_error(tool="submit_kaggle", args={}, error="rules not accepted")

    from kaggle_slayer.harness.resume import summarize
    summary = summarize(workspace)
    assert summary.total_calls == 3
    assert summary.error_count == 1
    assert summary.last_call["tool"] == "submit_kaggle"
    assert summary.tool_counts == {
        "build_context": 1,
        "train_cv": 1,
        "submit_kaggle": 1,
    }
