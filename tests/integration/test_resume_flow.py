"""Integration: aborted Solver run can be resumed via rebuild_conversation."""

from __future__ import annotations

import pytest

from kaggle_slayer.agent.llm_client import Response, ToolCall, Usage
from kaggle_slayer.agent.solver import Solver
from kaggle_slayer.harness import resume
from tests.fixtures.synthetic_comp import make_synthetic_comp

pytestmark = pytest.mark.integration


class _Scripted:
    def __init__(self, responses):
        self._r, self._i = list(responses), 0
        self.captured = []
    def call(self, messages, *, tools=None, model=None):
        self.captured.append(list(messages))
        r = self._r[self._i]
        self._i += 1
        return r


def test_resume_picks_up_after_three_tool_calls(tmp_path):
    workspace = make_synthetic_comp(tmp_path / "synthetic")

    # Phase 1: run three steps, then halt at max_iterations=3.
    phase1_responses = [
        Response(text="",
                 tool_calls=[ToolCall(id="p1_t1", name="take_note",
                                      args={"category": "observation", "content": "binary target"})],
                 usage=Usage(0, 0, 0)),
        Response(text="",
                 tool_calls=[ToolCall(id="p1_t2", name="sample_rows",
                                      args={"table": "train", "n": 5})],
                 usage=Usage(0, 0, 0)),
        Response(text="",
                 tool_calls=[ToolCall(id="p1_t3", name="take_note",
                                      args={"category": "decision", "content": "use logistic regression"})],
                 usage=Usage(0, 0, 0)),
    ]
    phase1_client = _Scripted(phase1_responses)
    phase1 = Solver(
        workspace=workspace, llm_client=phase1_client,
        target_col="Survived", max_iterations=3,
    )
    r1 = phase1.solve()
    # Loop exited at max_iterations (no done call yet)
    assert r1.status == "max_iterations"

    # Phase 2: rebuild history, then a fresh Solver finishes the comp.
    history = resume.rebuild_conversation(workspace)
    # 3 prior tool calls → 6 messages (model + tool per call)
    assert len(history) == 6

    phase2_responses = [
        Response(text="",
                 tool_calls=[ToolCall(id="p2_t1", name="done",
                                      args={"summary": "resumed and finished"})],
                 usage=Usage(0, 0, 0)),
    ]
    phase2_client = _Scripted(phase2_responses)
    phase2 = Solver(
        workspace=workspace, llm_client=phase2_client,
        target_col="Survived", max_iterations=5,
    )
    r2 = phase2.solve(resume_from=history)
    assert r2.status == "done"
    assert "resumed" in r2.summary

    # The first LLM call in phase 2 must have included the 6 resumed messages.
    first_call_msgs = phase2_client.captured[0]
    # system + user(context) + 6 resumed = 8 minimum
    assert len(first_call_msgs) >= 8
    # The 3rd message (after system+user) should be a model-role with take_note
    assert first_call_msgs[2].role == "model"
    assert first_call_msgs[2].tool_calls[0].name == "take_note"


def test_resume_raises_when_done_already_called(tmp_path):
    workspace = make_synthetic_comp(tmp_path / "synthetic")
    phase1_client = _Scripted([
        Response(text="", tool_calls=[ToolCall(id="t1", name="done",
                                               args={"summary": "first run done"})],
                 usage=Usage(0, 0, 0)),
    ])
    Solver(workspace=workspace, llm_client=phase1_client, target_col="Survived").solve()

    with pytest.raises(resume.ResumeError, match="already finished"):
        resume.rebuild_conversation(workspace)
