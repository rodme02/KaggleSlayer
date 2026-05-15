"""Checkpoint gate — typed pause-points where the harness blocks the agent.

Spec §9 defines six triggers (submit_kaggle first / submit_kaggle regression /
set_metric / wall-clock budget / cost budget / agent-initiated). Each emits a
CheckpointRequest; the CheckpointHandler turns that into a Decision.

The handler has four modes:
  INTERACTIVE   — rich CLI prompt (used by the real CLI; not unit-tested)
  AUTO_SAFE     — auto-approves the spec's 'auto-approve' triggers, denies the rest
  STUB          — returns a fixed Decision (used by tests)
  CALLABLE      — calls a user-supplied prompt_fn (used by the CLI; testable)

Every Decision is journalled to run_log.jsonl as kind='checkpoint'.
"""

from __future__ import annotations

import datetime as dt
import enum
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

from kaggle_slayer.harness.journal import Journal


class CheckpointTrigger(enum.Enum):
    """The named gate points from spec §9 + agent-initiated."""

    SUBMIT_KAGGLE_FIRST = "submit_kaggle_first"
    SUBMIT_KAGGLE_REGRESSION = "submit_kaggle_regression"
    SUBMIT_KAGGLE_NON_REGRESSION = "submit_kaggle_non_regression"
    SET_METRIC = "set_metric"
    WALL_CLOCK_BUDGET = "wall_clock_budget"
    COST_BUDGET = "cost_budget"
    MEMORY_SUSTAINED = "memory_sustained"
    AGENT_INITIATED = "agent_initiated"


class Decision(enum.Enum):
    APPROVE = "approve"
    DENY = "deny"
    ABORT = "abort"
    SKIP_CHECK = "skip_check"  # approve + don't ask again this run


class HandlerMode(enum.Enum):
    INTERACTIVE = "interactive"
    AUTO_SAFE = "auto_safe"
    STUB = "stub"
    CALLABLE = "callable"


# Triggers that auto_safe approves without prompting.
_AUTO_SAFE_APPROVES: frozenset[CheckpointTrigger] = frozenset({
    CheckpointTrigger.SUBMIT_KAGGLE_NON_REGRESSION,
})


@dataclass(frozen=True)
class CheckpointRequest:
    trigger: CheckpointTrigger
    action: str
    evidence: dict[str, Any] = field(default_factory=dict)


class CheckpointHandler:
    """Bound to one Journal; dispatches per-trigger to the configured mode."""

    def __init__(
        self,
        *,
        mode: HandlerMode,
        journal: Journal,
        stub_decision: Decision | None = None,
        prompt_fn: Callable[[CheckpointRequest], Decision] | None = None,
    ) -> None:
        self.mode = mode
        self.journal = journal
        self.stub_decision = stub_decision
        self.prompt_fn = prompt_fn
        self._skipped: set[CheckpointTrigger] = set()
        if mode == HandlerMode.STUB and stub_decision is None:
            raise ValueError("STUB mode requires stub_decision")
        if mode == HandlerMode.CALLABLE and prompt_fn is None:
            raise ValueError("CALLABLE mode requires prompt_fn")

    def request(self, req: CheckpointRequest) -> Decision:
        decision = self._decide(req)
        if decision == Decision.SKIP_CHECK:
            self._skipped.add(req.trigger)
        self._journal(req, decision)
        return decision

    def _decide(self, req: CheckpointRequest) -> Decision:
        if req.trigger in self._skipped:
            return Decision.APPROVE
        if self.mode == HandlerMode.STUB:
            assert self.stub_decision is not None  # checked in __init__
            return self.stub_decision
        if self.mode == HandlerMode.AUTO_SAFE:
            return Decision.APPROVE if req.trigger in _AUTO_SAFE_APPROVES else Decision.DENY
        if self.mode == HandlerMode.CALLABLE:
            assert self.prompt_fn is not None
            return self.prompt_fn(req)
        if self.mode == HandlerMode.INTERACTIVE:
            return _interactive_prompt(req)
        raise RuntimeError(f"unhandled mode: {self.mode}")

    def _journal(self, req: CheckpointRequest, decision: Decision) -> None:
        self.journal._append(  # noqa: SLF001 — checkpoint kind is part of journal contract
            self.journal.workspace.run_log_path,
            {
                "ts": dt.datetime.now(dt.UTC).isoformat(timespec="seconds"),
                "kind": "checkpoint",
                "trigger": req.trigger.value,
                "action": req.action,
                "evidence": req.evidence,
                "decision": decision.value,
            },
        )


def _interactive_prompt(req: CheckpointRequest) -> Decision:
    """Default rich-styled prompt — never reached in tests."""
    from rich.console import Console  # noqa: PLC0415
    from rich.panel import Panel  # noqa: PLC0415
    from rich.prompt import Prompt  # noqa: PLC0415

    console = Console()
    body = f"[bold]{req.action}[/]\n\n"
    for k, v in req.evidence.items():
        body += f"  • {k}: {v}\n"
    console.print(Panel(body, title=f"Checkpoint: {req.trigger.value}", border_style="yellow"))
    choice = Prompt.ask(
        "Decision",
        choices=["y", "n", "a", "s"],
        default="n",
        show_choices=True,
    )
    return {
        "y": Decision.APPROVE,
        "n": Decision.DENY,
        "a": Decision.ABORT,
        "s": Decision.SKIP_CHECK,
    }[choice]
