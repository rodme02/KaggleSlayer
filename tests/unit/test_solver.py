"""Tests for kaggle_slayer.agent.solver.Solver."""

from __future__ import annotations

from kaggle_slayer.agent.llm_client import Message, Response, ToolCall, Usage
from kaggle_slayer.agent.solver import Solver, SolverContext, SolveResult
from kaggle_slayer.harness.journal import Journal
from kaggle_slayer.harness.workspace import Workspace


class _CannedClient:
    """Bare-bones LLMClient stand-in that returns a fixed sequence of Responses."""

    def __init__(self, responses):
        self._responses = list(responses)
        self._i = 0
        self.captured: list[list[Message]] = []

    def call(self, messages, *, tools=None, model=None):
        self.captured.append(list(messages))
        r = self._responses[self._i]
        self._i += 1
        return r


def _make_workspace_and_ctx(tmp_path):
    ws = Workspace.create(root=tmp_path / "comp")
    ws.context_path.write_text("# Fake Comp\n\nMetric: accuracy. Target: target.")
    return ws


def test_solver_exits_on_done_response(tmp_path):
    """A tool call to 'done' must stop the loop and produce status=done."""
    ws = _make_workspace_and_ctx(tmp_path)
    client = _CannedClient(responses=[
        Response(
            text="",
            tool_calls=[ToolCall(id="tc1", name="done", args={"summary": "fake done"})],
            usage=Usage(0, 0, 0),
        ),
    ])
    solver = Solver(workspace=ws, llm_client=client, max_iterations=5)
    result = solver.solve()
    assert isinstance(result, SolveResult)
    assert result.status == "done"
    assert "fake done" in result.summary


def test_solver_exits_on_max_iterations(tmp_path):
    """If the agent never calls done, the loop must terminate at max_iterations."""
    ws = _make_workspace_and_ctx(tmp_path)
    # Endless empty-text replies — no done call
    client = _CannedClient(responses=[
        Response(text="thinking...", tool_calls=[], usage=Usage(0, 0, 0))
        for _ in range(20)
    ])
    solver = Solver(workspace=ws, llm_client=client, max_iterations=3)
    result = solver.solve()
    assert result.status == "max_iterations"
    assert result.iterations == 3


def test_solver_dispatches_tool_call_and_feeds_result_back(tmp_path):
    """When the LLM returns a tool call, the solver invokes it and feeds the
    result back as a tool-role message on the next turn."""
    ws = _make_workspace_and_ctx(tmp_path)
    client = _CannedClient(responses=[
        Response(
            text="",
            tool_calls=[ToolCall(id="tc1", name="take_note", args={"category": "observation", "content": "x"})],
            usage=Usage(0, 0, 0),
        ),
        Response(
            text="",
            tool_calls=[ToolCall(id="tc2", name="done", args={"summary": "ok"})],
            usage=Usage(0, 0, 0),
        ),
    ])
    solver = Solver(workspace=ws, llm_client=client, max_iterations=5)
    result = solver.solve()
    assert result.status == "done"

    # On the second call, the messages must include a tool-role message with
    # the take_note result
    second_msgs = client.captured[1]
    tool_roles = [m for m in second_msgs if m.role == "tool"]
    assert len(tool_roles) >= 1
    # Note was actually written
    assert ws.notes_path.exists()


def test_solver_journals_each_tool_call(tmp_path):
    """Every tool call (success or error) lands in run_log.jsonl."""
    ws = _make_workspace_and_ctx(tmp_path)
    client = _CannedClient(responses=[
        Response(text="",
                 tool_calls=[ToolCall(id="t1", name="take_note", args={"category": "observation", "content": "noted"})],
                 usage=Usage(0, 0, 0)),
        Response(text="",
                 tool_calls=[ToolCall(id="t2", name="done", args={"summary": "fin"})],
                 usage=Usage(0, 0, 0)),
    ])
    solver = Solver(workspace=ws, llm_client=client, max_iterations=5)
    solver.solve()

    import json
    log_records = [json.loads(line) for line in ws.run_log_path.read_text().splitlines()]
    tool_calls_logged = [r for r in log_records if r["tool"] in ("take_note", "done")]
    assert len(tool_calls_logged) == 2


def test_solver_handles_tool_error_and_feeds_message_back(tmp_path):
    """If a tool raises ToolError, the solver journals it and feeds the error
    back to the LLM so it can correct itself."""
    ws = _make_workspace_and_ctx(tmp_path)
    client = _CannedClient(responses=[
        # First: invalid call — write_file with a protected path
        Response(text="",
                 tool_calls=[ToolCall(id="t1", name="write_file", args={"path": "context.md", "content": "x"})],
                 usage=Usage(0, 0, 0)),
        # Second: done
        Response(text="",
                 tool_calls=[ToolCall(id="t2", name="done", args={"summary": "bailing"})],
                 usage=Usage(0, 0, 0)),
    ])
    solver = Solver(workspace=ws, llm_client=client, max_iterations=5)
    result = solver.solve()
    assert result.status == "done"

    # Second LLM call must include a tool-role message reporting the error
    second = client.captured[1]
    tool_msgs = [m for m in second if m.role == "tool"]
    assert any("protected" in m.content.lower() for m in tool_msgs)


def test_solver_context_carries_target_metric_problem_type(tmp_path):
    """The SolverContext defaults can be overridden via kwargs."""
    ws = _make_workspace_and_ctx(tmp_path)
    ctx = SolverContext(workspace=ws, journal=Journal(ws), target_col="my_target",
                        problem_type="regression", metric_name="rmse")
    assert ctx.target_col == "my_target"
    assert ctx.problem_type == "regression"
    assert ctx.metric_name == "rmse"
    assert ctx.finished is False
