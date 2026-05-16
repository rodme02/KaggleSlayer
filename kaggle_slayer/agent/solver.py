"""Solver — the agent loop.

Per-turn: pass the message history + tool declarations to the LLM,
parse the response. If the response has tool_calls, dispatch each via the
ToolRegistry, append the result (or error) as a tool-role Message, and
loop. If the response is plain text with no tool calls, treat it as
"thinking aloud" and continue. Exit on:
  - the agent calls `done`
  - max_iterations exhausted
  - wall-clock budget exhausted
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from kaggle_slayer.agent.handlers import make_builtin_registry
from kaggle_slayer.agent.llm_client import LLMClient, Message
from kaggle_slayer.agent.prompts import load_system_prompt
from kaggle_slayer.agent.tools import ToolError, ToolRegistry
from kaggle_slayer.harness.journal import Journal
from kaggle_slayer.harness.workspace import Workspace

if TYPE_CHECKING:
    # F11: these are real types — no actual circular import exists, but we
    # keep them under TYPE_CHECKING so the module's import-time graph stays
    # minimal. `from __future__ import annotations` (above) makes all
    # annotations lazy strings, so dataclass field types resolve fine.
    from kaggle_slayer.agent.cost_ledger import CostLedger
    from kaggle_slayer.harness.checkpoints import CheckpointHandler
    from kaggle_slayer.harness.kaggle_client import KaggleClient

# Cap on the text we feed back to the LLM per tool result. A 20KB raw output
# (e.g., a long DataFrame.to_string()) would burn tokens and crowd context.
# We keep the full result in the registry's actual return value, but show the
# model only the first ~8KB plus a truncation marker.
_TOOL_RESULT_CAP_CHARS: int = 8192
_TOOL_RESULT_KEEP_CHARS: int = 8000


@dataclass
class SolverContext:
    """State the tool handlers read and write."""

    workspace: Workspace
    journal: Journal
    target_col: str = "target"
    problem_type: str = "classification"
    metric_name: str = "accuracy"
    cv_kind: str | None = None
    cv_params: dict[str, Any] = field(default_factory=dict)
    finished: bool = False
    final_summary: str = ""
    checkpoint_handler: CheckpointHandler | None = None
    best_cv_mean: float | None = None
    kaggle_client: KaggleClient | None = None
    competition: str = ""


@dataclass
class SolveResult:
    status: str  # "done" | "max_iterations" | "time_exceeded" | "cost_budget_exceeded"
    iterations: int
    summary: str


class Solver:
    """Runs the agent loop against a single competition workspace."""

    def __init__(
        self,
        *,
        workspace: Workspace,
        llm_client: LLMClient,
        target_col: str = "target",
        problem_type: str = "classification",
        metric_name: str = "accuracy",
        max_iterations: int = 25,
        time_budget_s: float = 900.0,
        registry: ToolRegistry | None = None,
        checkpoint_handler: CheckpointHandler | None = None,
        cost_ledger: CostLedger | None = None,
        cost_budget_usd: float | None = None,
        kaggle_client: KaggleClient | None = None,
    ) -> None:
        self.workspace = workspace
        self.llm = llm_client
        self.max_iterations = max_iterations
        self.time_budget_s = time_budget_s
        self.registry = registry or make_builtin_registry()
        self.journal = Journal(workspace)
        self.checkpoint_handler = checkpoint_handler
        self.cost_ledger = cost_ledger
        self.cost_budget_usd = cost_budget_usd
        self.ctx = SolverContext(
            workspace=workspace,
            journal=self.journal,
            target_col=target_col,
            problem_type=problem_type,
            metric_name=metric_name,
            checkpoint_handler=checkpoint_handler,
            kaggle_client=kaggle_client,
            competition=workspace.name,
        )

    def solve(self, *, resume_from: list[Message] | None = None) -> SolveResult:
        system_prompt = load_system_prompt()
        context_md = (
            self.workspace.context_path.read_text()
            if self.workspace.context_path.exists()
            else "(no context.md yet)"
        )

        messages: list[Message]
        if resume_from:
            messages = [
                Message(role="system", content=system_prompt),
                Message(role="user", content=context_md),
                *resume_from,
            ]
        else:
            messages = [
                Message(role="system", content=system_prompt),
                Message(role="user", content=context_md),
            ]
        tool_decls = self.registry.to_function_declarations()

        started = time.perf_counter()
        for iteration in range(1, self.max_iterations + 1):
            if time.perf_counter() - started > self.time_budget_s:
                if not self._gate_wall_clock(iteration - 1):
                    return SolveResult(status="time_exceeded", iterations=iteration - 1, summary="")
                # Approved — extend the budget by one more cycle of the original cap.
                started = time.perf_counter()

            if self._cost_budget_exceeded() and not self._gate_cost_budget():
                return SolveResult(status="cost_budget_exceeded", iterations=iteration - 1, summary="")

            response = self.llm.call(messages=messages, tools=tool_decls)

            if not response.tool_calls:
                # Pure-text response. Keep the model's words in history for
                # continuity, then loop until the agent calls done or max iter.
                if response.text:
                    messages.append(Message(role="model", content=response.text))
                continue

            # Tool-call response. Gemini's multi-turn protocol requires the
            # model(function_call) turn to precede the tool(function_response)
            # turn in history — append it before dispatching so the next call's
            # history is well-formed.
            messages.append(Message(
                role="model",
                content=response.text or "",
                tool_calls=response.tool_calls,
            ))

            for tc in response.tool_calls:
                tool_result_text = self._dispatch(tc.name, tc.args)
                # Feed the result back as a tool-role message. We serialize as a
                # small JSON payload so the LLMClient knows which tool the
                # function_response Part should attribute to.
                payload = json.dumps({"tool": tc.name, "result": tool_result_text})
                messages.append(Message(role="tool", content=payload))

                if self.ctx.finished:
                    return SolveResult(
                        status="done",
                        iterations=iteration,
                        summary=self.ctx.final_summary,
                    )

        return SolveResult(status="max_iterations", iterations=self.max_iterations, summary="")

    def _dispatch(self, name: str, args: dict[str, Any]) -> str:
        """Invoke a tool, journal it, return a string result (success or error)."""
        try:
            result = self.registry.invoke(name, ctx=self.ctx, args=args)
            text_result = str(result)
            self.journal.log_tool_call(
                tool=name,
                args=args,
                # 8 KB matches the LLM-visible cap, so resume can replay
                # exactly what the LLM originally saw.
                result_summary=text_result[:8000],
            )
            return _cap_tool_result(text_result)
        except ToolError as e:
            err_msg = f"ToolError: {e}"
            self.journal.log_tool_error(tool=name, args=args, error=err_msg)
            return err_msg
        except Exception as e:  # noqa: BLE001
            err_msg = f"unexpected error in {name}: {e!r}"
            self.journal.log_tool_error(tool=name, args=args, error=err_msg)
            return err_msg

    def _gate_wall_clock(self, iterations_so_far: int) -> bool:
        """Ask the checkpoint handler to extend the wall-clock budget. True = continue."""
        if self.checkpoint_handler is None:
            return False
        from kaggle_slayer.harness import checkpoints as cp  # noqa: PLC0415

        decision = self.checkpoint_handler.request(cp.CheckpointRequest(
            trigger=cp.CheckpointTrigger.WALL_CLOCK_BUDGET,
            action=f"extend wall-clock budget (iter {iterations_so_far})",
            evidence={"budget_s": self.time_budget_s, "iter": iterations_so_far},
        ))
        return decision in (cp.Decision.APPROVE, cp.Decision.SKIP_CHECK)

    def _cost_budget_exceeded(self) -> bool:
        if self.cost_ledger is None or self.cost_budget_usd is None:
            return False
        spent = float(self.cost_ledger.total_for(competition=self.workspace.name))
        return spent > self.cost_budget_usd

    def _gate_cost_budget(self) -> bool:
        """Ask the user to raise the cost budget. APPROVE doubles it (F4).

        Without the doubling, every subsequent iteration sees `spent > budget`
        again and the user is re-prompted on every turn. Mirrors the wall-
        clock pattern, which resets the timer on APPROVE.
        """
        if self.checkpoint_handler is None or self.cost_ledger is None:
            return False
        from kaggle_slayer.harness import checkpoints as cp  # noqa: PLC0415

        spent = float(self.cost_ledger.total_for(competition=self.workspace.name))
        new_budget = self.cost_budget_usd * 2.0 if self.cost_budget_usd is not None else None
        decision = self.checkpoint_handler.request(cp.CheckpointRequest(
            trigger=cp.CheckpointTrigger.COST_BUDGET,
            action="cost budget exceeded — raise it by 2x?",
            evidence={
                "spent_usd": spent,
                "budget_usd": self.cost_budget_usd,
                "if_approved_new_budget_usd": new_budget,
            },
        ))
        approved = decision in (cp.Decision.APPROVE, cp.Decision.SKIP_CHECK)
        if approved and self.cost_budget_usd is not None:
            self.cost_budget_usd *= 2.0
        return approved


def _cap_tool_result(text: str) -> str:
    """If text exceeds the cap, truncate to _TOOL_RESULT_KEEP_CHARS and append
    a marker telling the model how many bytes were dropped."""
    if len(text) <= _TOOL_RESULT_CAP_CHARS:
        return text
    dropped = len(text) - _TOOL_RESULT_KEEP_CHARS
    return text[:_TOOL_RESULT_KEEP_CHARS] + f"\n\n[truncated, {dropped} more chars]"
