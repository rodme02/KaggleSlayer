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
from typing import Any

from kaggle_slayer.agent.handlers import make_builtin_registry
from kaggle_slayer.agent.llm_client import LLMClient, Message
from kaggle_slayer.agent.prompts import load_system_prompt
from kaggle_slayer.agent.tools import ToolError, ToolRegistry
from kaggle_slayer.harness.journal import Journal
from kaggle_slayer.harness.workspace import Workspace


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


@dataclass
class SolveResult:
    status: str  # "done" | "max_iterations" | "time_exceeded"
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
    ) -> None:
        self.workspace = workspace
        self.llm = llm_client
        self.max_iterations = max_iterations
        self.time_budget_s = time_budget_s
        self.registry = registry or make_builtin_registry()
        self.journal = Journal(workspace)
        self.ctx = SolverContext(
            workspace=workspace,
            journal=self.journal,
            target_col=target_col,
            problem_type=problem_type,
            metric_name=metric_name,
        )

    def solve(self) -> SolveResult:
        system_prompt = load_system_prompt()
        context_md = (
            self.workspace.context_path.read_text()
            if self.workspace.context_path.exists()
            else "(no context.md yet)"
        )

        messages: list[Message] = [
            Message(role="system", content=system_prompt),
            Message(role="user", content=context_md),
        ]
        tool_decls = self.registry.to_function_declarations()

        started = time.perf_counter()
        for iteration in range(1, self.max_iterations + 1):
            if time.perf_counter() - started > self.time_budget_s:
                return SolveResult(status="time_exceeded", iterations=iteration - 1, summary="")

            response = self.llm.call(messages=messages, tools=tool_decls)

            # If the LLM produced text alongside or instead of tool calls, append
            # it as a model-role message so the next turn has continuity.
            if response.text:
                messages.append(Message(role="model", content=response.text))

            if not response.tool_calls:
                # Pure-text response. Keep looping until the agent calls done or max iter.
                continue

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
                result_summary=text_result[:200],
            )
            return text_result
        except ToolError as e:
            err_msg = f"ToolError: {e}"
            self.journal.log_tool_error(tool=name, args=args, error=err_msg)
            return err_msg
        except Exception as e:  # noqa: BLE001
            err_msg = f"unexpected error in {name}: {e!r}"
            self.journal.log_tool_error(tool=name, args=args, error=err_msg)
            return err_msg
