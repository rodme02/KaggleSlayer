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


def test_solver_journals_original_tool_call_id(tmp_path):
    """F8: the Solver passes ToolCall.id through to the journal so resume
    can preserve the original LLM-issued id instead of fabricating one.
    """
    ws = _make_workspace_and_ctx(tmp_path)
    client = _CannedClient(responses=[
        Response(text="",
                 tool_calls=[ToolCall(id="call_xyz_42",
                                      name="take_note",
                                      args={"category": "observation", "content": "x"})],
                 usage=Usage(0, 0, 0)),
        Response(text="",
                 tool_calls=[ToolCall(id="call_done_1", name="done", args={"summary": "fin"})],
                 usage=Usage(0, 0, 0)),
    ])
    solver = Solver(workspace=ws, llm_client=client, max_iterations=5)
    solver.solve()

    import json
    records = [json.loads(line) for line in ws.run_log_path.read_text().splitlines()]
    take_note_rec = next(r for r in records if r["tool"] == "take_note")
    assert take_note_rec["tool_call_id"] == "call_xyz_42"
    done_rec = next(r for r in records if r["tool"] == "done")
    assert done_rec["tool_call_id"] == "call_done_1"


def test_solver_appends_model_message_with_tool_calls(tmp_path):
    """When the LLM returns a tool call, the Solver must append a
    Message(role='model', tool_calls=[...]) before the tool-result message
    so the next turn's history mirrors Gemini's call/response protocol."""
    ws = _make_workspace_and_ctx(tmp_path)
    client = _CannedClient(responses=[
        Response(
            text="",
            tool_calls=[ToolCall(id="tc1", name="take_note",
                                 args={"category": "observation", "content": "x"})],
            usage=Usage(0, 0, 0),
        ),
        Response(
            text="",
            tool_calls=[ToolCall(id="tc2", name="done", args={"summary": "ok"})],
            usage=Usage(0, 0, 0),
        ),
    ])
    solver = Solver(workspace=ws, llm_client=client, max_iterations=5)
    solver.solve()

    # Second LLM call's history should contain a model-role Message whose
    # tool_calls list reflects the take_note call.
    second_msgs = client.captured[1]
    model_msgs = [m for m in second_msgs if m.role == "model"]
    assert any(getattr(m, "tool_calls", None) for m in model_msgs), \
        "expected at least one model message carrying tool_calls"
    matching = [
        m for m in model_msgs
        if getattr(m, "tool_calls", None) and m.tool_calls[0].name == "take_note"
    ]
    assert matching, "expected a model message whose first tool_call is take_note"


def test_solver_truncates_long_tool_results_in_response_to_llm(tmp_path):
    """A 20KB tool result must be capped to ~8KB before being fed back to the LLM,
    with a visible truncation marker so the model knows content was dropped."""
    from kaggle_slayer.agent.tools import Tool, ToolRegistry

    ws = _make_workspace_and_ctx(tmp_path)

    big_result = "A" * 20_000

    def _big_handler(ctx, **_args):
        return big_result

    registry = ToolRegistry()
    registry.register(Tool(
        name="big",
        description="produces a huge result",
        schema={"type": "object", "properties": {}, "additionalProperties": False},
        handler=_big_handler,
    ))
    # Need 'done' so the loop can exit cleanly
    from kaggle_slayer.agent.handlers import ml as ml_h
    registry.register(Tool(
        name="done",
        description="finish",
        schema={
            "type": "object",
            "properties": {"summary": {"type": "string"}},
            "required": ["summary"],
            "additionalProperties": False,
        },
        handler=ml_h.done,
    ))

    client = _CannedClient(responses=[
        Response(text="",
                 tool_calls=[ToolCall(id="t1", name="big", args={})],
                 usage=Usage(0, 0, 0)),
        Response(text="",
                 tool_calls=[ToolCall(id="t2", name="done", args={"summary": "fin"})],
                 usage=Usage(0, 0, 0)),
    ])
    solver = Solver(workspace=ws, llm_client=client, registry=registry, max_iterations=5)
    solver.solve()

    # The tool-role message in the second turn must be capped
    second = client.captured[1]
    tool_msgs = [m for m in second if m.role == "tool"]
    assert tool_msgs, "expected a tool-role message in the second call"
    # The serialized payload is JSON-wrapped; even so, the content length must
    # be well under the raw 20KB and must contain the truncation marker.
    assert len(tool_msgs[0].content) <= 8500, \
        f"expected the tool message to be capped (~8KB), got {len(tool_msgs[0].content)}"
    assert "truncated" in tool_msgs[0].content


def test_solver_model_message_precedes_tool_response(tmp_path):
    """The model(tool_call) message must come BEFORE the tool(result) message
    in the second call's history."""
    ws = _make_workspace_and_ctx(tmp_path)
    client = _CannedClient(responses=[
        Response(
            text="",
            tool_calls=[ToolCall(id="tc1", name="take_note",
                                 args={"category": "observation", "content": "x"})],
            usage=Usage(0, 0, 0),
        ),
        Response(
            text="",
            tool_calls=[ToolCall(id="tc2", name="done", args={"summary": "ok"})],
            usage=Usage(0, 0, 0),
        ),
    ])
    solver = Solver(workspace=ws, llm_client=client, max_iterations=5)
    solver.solve()

    second_msgs = client.captured[1]
    roles = [m.role for m in second_msgs]
    # We expect: [system, user, model(tool_calls), tool(result)] as the prefix
    assert roles[:4] == ["system", "user", "model", "tool"], (
        f"expected [system, user, model, tool] prefix, got {roles[:4]}"
    )
    # The model message in slot 2 must be the one carrying tool_calls
    assert second_msgs[2].tool_calls and second_msgs[2].tool_calls[0].name == "take_note"


def test_solver_wall_clock_checkpoint_extends_on_approve(tmp_path):
    """When time_budget_s elapses, checkpoint fires; APPROVE extends by another budget."""
    from kaggle_slayer.harness import checkpoints as cp
    from kaggle_slayer.harness.journal import Journal

    ws = _make_workspace_and_ctx(tmp_path)
    journal = Journal(ws)
    # The handler approves the first wall-clock checkpoint, denies the second.
    decisions = iter([cp.Decision.APPROVE, cp.Decision.DENY])

    def prompt(_req):
        return next(decisions)

    handler = cp.CheckpointHandler(
        mode=cp.HandlerMode.CALLABLE, journal=journal, prompt_fn=prompt
    )
    # Endless thinking-aloud responses; never call done.
    responses = [Response(text="thinking", tool_calls=[], usage=Usage(0, 0, 0)) for _ in range(50)]
    client = _CannedClient(responses=responses)
    solver = Solver(
        workspace=ws,
        llm_client=client,
        max_iterations=50,
        time_budget_s=0.0,  # immediate — fires on every iteration deterministically
        checkpoint_handler=handler,
    )
    result = solver.solve()
    # First wall-clock checkpoint approved → continued; second denied → exit.
    assert result.status == "time_exceeded"
    # Journal must contain exactly two wall_clock_budget checkpoints.
    import json
    cp_records = [
        json.loads(line) for line in ws.run_log_path.read_text().splitlines()
        if json.loads(line).get("kind") == "checkpoint"
    ]
    wall_cp = [r for r in cp_records if r["trigger"] == "wall_clock_budget"]
    assert len(wall_cp) == 2
    assert wall_cp[0]["decision"] == "approve"
    assert wall_cp[1]["decision"] == "deny"


def test_solver_cost_budget_checkpoint(tmp_path):
    """When cost_ledger.total_for(competition) > cost_budget_usd, checkpoint fires."""
    from unittest.mock import MagicMock

    from kaggle_slayer.harness import checkpoints as cp
    from kaggle_slayer.harness.journal import Journal

    ws = _make_workspace_and_ctx(tmp_path)
    journal = Journal(ws)
    handler = cp.CheckpointHandler(
        mode=cp.HandlerMode.STUB, journal=journal, stub_decision=cp.Decision.DENY
    )

    ledger = MagicMock()
    ledger.total_for = MagicMock(return_value=0.10)  # already over a $0.05 budget

    responses = [
        Response(text="thinking", tool_calls=[], usage=Usage(0, 0, 0)) for _ in range(5)
    ]
    client = _CannedClient(responses=responses)
    solver = Solver(
        workspace=ws,
        llm_client=client,
        max_iterations=10,
        checkpoint_handler=handler,
        cost_ledger=ledger,
        cost_budget_usd=0.05,
    )
    result = solver.solve()
    # Denied at first cost-budget check → exit with cost_budget_exceeded.
    assert result.status == "cost_budget_exceeded"


def test_solver_cost_budget_approval_doubles_budget(tmp_path):
    """F4: APPROVE on the cost-budget gate must DOUBLE cost_budget_usd so it
    doesn't re-fire on the very next iteration.

    Without the fix, every subsequent iteration sees `spent > budget` again
    and the user must answer 'yes' every single turn.
    """
    from unittest.mock import MagicMock

    from kaggle_slayer.harness import checkpoints as cp
    from kaggle_slayer.harness.journal import Journal

    ws = _make_workspace_and_ctx(tmp_path)
    journal = Journal(ws)
    handler = cp.CheckpointHandler(
        mode=cp.HandlerMode.STUB, journal=journal, stub_decision=cp.Decision.APPROVE
    )
    ledger = MagicMock()
    ledger.total_for = MagicMock(return_value=0.10)  # spent

    # Three thinking responses, then a done call so the loop exits cleanly.
    responses = [
        Response(text="t1", tool_calls=[], usage=Usage(0, 0, 0)),
        Response(text="t2", tool_calls=[], usage=Usage(0, 0, 0)),
        Response(
            text="",
            tool_calls=[ToolCall(id="x", name="done", args={"summary": "fin"})],
            usage=Usage(0, 0, 0),
        ),
    ]
    client = _CannedClient(responses=responses)
    solver = Solver(
        workspace=ws,
        llm_client=client,
        max_iterations=10,
        checkpoint_handler=handler,
        cost_ledger=ledger,
        cost_budget_usd=0.05,
    )
    result = solver.solve()
    assert result.status == "done", (
        f"expected loop to finish on done, got {result.status} — "
        f"gate likely re-fired and exited early"
    )
    # Budget doubled exactly once (spent 0.10 stayed flat → only first gate fires).
    assert solver.cost_budget_usd == 0.10
    # The journal must contain exactly one cost_budget checkpoint record.
    import json
    cp_records = [
        json.loads(line) for line in ws.run_log_path.read_text().splitlines()
        if json.loads(line).get("kind") == "checkpoint"
    ]
    cost_cps = [r for r in cp_records if r["trigger"] == "cost_budget"]
    assert len(cost_cps) == 1, (
        f"expected one cost_budget checkpoint after doubling, got {len(cost_cps)}"
    )


def test_solver_cost_budget_evidence_includes_doubled_target(tmp_path):
    """F4: the cost-budget checkpoint request should surface the post-approval
    budget in its evidence dict so the prompt can show the user what 'yes' costs."""
    from unittest.mock import MagicMock

    from kaggle_slayer.harness import checkpoints as cp
    from kaggle_slayer.harness.journal import Journal

    ws = _make_workspace_and_ctx(tmp_path)
    journal = Journal(ws)

    captured: list[cp.CheckpointRequest] = []

    def prompt(req: cp.CheckpointRequest) -> cp.Decision:
        captured.append(req)
        return cp.Decision.APPROVE

    handler = cp.CheckpointHandler(mode=cp.HandlerMode.CALLABLE, journal=journal, prompt_fn=prompt)
    ledger = MagicMock()
    ledger.total_for = MagicMock(return_value=0.10)

    responses = [
        Response(
            text="",
            tool_calls=[ToolCall(id="x", name="done", args={"summary": "fin"})],
            usage=Usage(0, 0, 0),
        ),
    ]
    client = _CannedClient(responses=responses)
    solver = Solver(
        workspace=ws,
        llm_client=client,
        max_iterations=5,
        checkpoint_handler=handler,
        cost_ledger=ledger,
        cost_budget_usd=0.05,
    )
    solver.solve()
    assert captured, "expected at least one checkpoint request"
    cost_req = next(r for r in captured if r.trigger == cp.CheckpointTrigger.COST_BUDGET)
    assert cost_req.evidence["spent_usd"] == 0.10
    assert cost_req.evidence["budget_usd"] == 0.05
    assert cost_req.evidence["if_approved_new_budget_usd"] == 0.10


def test_solver_wall_clock_approval_extends_budget(tmp_path):
    """F17: symmetry test — APPROVE on the wall-clock gate must let the loop
    make at least one LLM call past the original deadline.

    Without the extension (DENY on first gate), the loop exits with zero LLM
    calls. With APPROVE, the loop continues into iteration 1 (one LLM call)
    before the next gate (DENY) ends it on iteration 2. This is the symmetry
    test that would have caught F4 in the cost-budget path.
    """
    from kaggle_slayer.harness import checkpoints as cp
    from kaggle_slayer.harness.journal import Journal

    ws = _make_workspace_and_ctx(tmp_path)
    journal = Journal(ws)
    # APPROVE first, then DENY — the loop should continue past gate 1 and exit on gate 2.
    decisions = iter([cp.Decision.APPROVE, cp.Decision.DENY])

    def prompt(_req: cp.CheckpointRequest) -> cp.Decision:
        return next(decisions)

    handler = cp.CheckpointHandler(mode=cp.HandlerMode.CALLABLE, journal=journal, prompt_fn=prompt)

    responses = [Response(text="t", tool_calls=[], usage=Usage(0, 0, 0)) for _ in range(20)]
    client = _CannedClient(responses=responses)
    solver = Solver(
        workspace=ws,
        llm_client=client,
        max_iterations=20,
        time_budget_s=0.0,  # immediate; fires on the very first iteration
        checkpoint_handler=handler,
    )
    solver.solve()
    # With APPROVE on gate 1, the deadline resets and iteration 1 calls the LLM.
    # If APPROVE didn't extend, no LLM call would have happened.
    assert len(client.captured) >= 1, (
        f"expected APPROVE to allow at least one LLM call past the deadline, "
        f"got {len(client.captured)}"
    )
    # And exactly two wall-clock checkpoint records: APPROVE, then DENY.
    import json
    cp_records = [
        json.loads(line) for line in ws.run_log_path.read_text().splitlines()
        if json.loads(line).get("kind") == "checkpoint"
    ]
    wall_cp = [r for r in cp_records if r["trigger"] == "wall_clock_budget"]
    assert len(wall_cp) == 2
    assert wall_cp[0]["decision"] == "approve"
    assert wall_cp[1]["decision"] == "deny"


def test_solver_journals_full_capped_tool_result_for_resume_fidelity(tmp_path):
    """The run_log result_summary must store up to ~8KB so resume can
    reconstruct what the LLM actually saw on the original turn."""
    ws = _make_workspace_and_ctx(tmp_path)

    big_text = "x" * 4096  # 4 KB — well above the old 200-char cap, below the new 8 KB cap
    # Custom registry with a single handler that returns big_text
    from kaggle_slayer.agent.tools import Tool, ToolRegistry
    reg = ToolRegistry()
    reg.register(Tool(
        name="echo",
        description="returns the supplied content",
        schema={"type": "object", "properties": {"content": {"type": "string"}}, "required": ["content"]},
        handler=lambda ctx, content: content,
    ))
    reg.register(Tool(
        name="done",
        description="signal finished",
        schema={"type": "object", "properties": {"summary": {"type": "string"}}, "required": ["summary"]},
        handler=lambda ctx, summary: (setattr(ctx, "finished", True), setattr(ctx, "final_summary", summary))[0],
    ))

    client = _CannedClient(responses=[
        Response(text="", tool_calls=[ToolCall(id="t1", name="echo", args={"content": big_text})], usage=Usage(0, 0, 0)),
        Response(text="", tool_calls=[ToolCall(id="t2", name="done", args={"summary": "done"})], usage=Usage(0, 0, 0)),
    ])
    solver = Solver(workspace=ws, llm_client=client, max_iterations=5, registry=reg)
    solver.solve()

    import json
    records = [json.loads(line) for line in ws.run_log_path.read_text().splitlines()]
    echo_record = next(r for r in records if r.get("tool") == "echo")
    # Must contain the full 4 KB (or close to it), not be truncated at 200 chars.
    assert len(echo_record["result_summary"]) >= 4000


def test_solver_writes_otel_trace_to_workspace(tmp_path):
    """Every Solver run emits an OTel trace; root span + child spans per
    LLM call + per tool dispatch."""
    import json
    ws = _make_workspace_and_ctx(tmp_path)
    client = _CannedClient(responses=[
        Response(text="",
                 tool_calls=[ToolCall(id="t1", name="take_note",
                                      args={"category": "observation", "content": "x"})],
                 usage=Usage(input_tokens=5, output_tokens=3, cached_tokens=0)),
        Response(text="",
                 tool_calls=[ToolCall(id="t2", name="done", args={"summary": "fin"})],
                 usage=Usage(0, 0, 0)),
    ])
    solver = Solver(workspace=ws, llm_client=client, max_iterations=5)
    solver.solve()

    path = ws.root / "otel.jsonl"
    assert path.exists()
    spans = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    span_names = [s["name"] for s in spans]
    # Root marker for the run, plus LLM-call spans, plus tool-dispatch spans
    assert any(n.startswith("run:") for n in span_names)
    assert any(n == "llm.call" for n in span_names)
    assert any(n.startswith("tool:") for n in span_names)
    # The tool-dispatch span carries the tool name as an attribute
    tool_spans = [s for s in spans if s["name"].startswith("tool:")]
    assert any(s["attributes"].get("tool.name") in ("take_note", "done") for s in tool_spans)
