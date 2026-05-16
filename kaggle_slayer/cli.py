"""kaggle-slayer CLI entry point.

Usage:
    kaggle-slayer <workspace-path> --target <col> [--metric <m>] [--problem-type <p>]
                  [--max-iterations N] [--model <gemini-id>]

This is intentionally thin: parse args, ensure the workspace exists,
maybe build context.md, then invoke the Solver. Heavy lifting lives in
kaggle_slayer.agent.solver.Solver.

The Gemini model defaults to `gemini-2.5-flash` (the validated slow-tier
model). Use `--model gemini-2.5-pro` to opt into Pro (10x cost, subject to
free-tier quota walls).
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

from dotenv import load_dotenv
from rich.console import Console
from rich.panel import Panel

from kaggle_slayer.agent.cost_ledger import CostLedger
from kaggle_slayer.agent.llm_client import GeminiClient
from kaggle_slayer.agent.solver import Solver
from kaggle_slayer.harness.context import build_context
from kaggle_slayer.harness.journal import Journal
from kaggle_slayer.harness.kaggle_client import KaggleClient
from kaggle_slayer.harness.workspace import Workspace


def _parse_args(argv: list[str]) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        prog="kaggle-slayer",
        description=(
            "LLM-agent harness for tabular Kaggle competitions. "
            "Defaults to Gemini Flash; override with --model for Pro or other ids."
        ),
    )
    p.add_argument("workspace_path", help="Path to per-competition workspace (e.g., competitions/titanic)")
    p.add_argument("--target", default=None, help="Target column name")
    p.add_argument("--metric", default="accuracy", help="Metric (accuracy, auc, logloss, rmse, mae, r2)")
    p.add_argument("--problem-type", default="classification", choices=["classification", "regression"])
    p.add_argument("--max-iterations", type=int, default=25)
    p.add_argument("--time-budget-s", type=float, default=900.0)
    p.add_argument(
        "--model",
        default="gemini-2.5-flash",
        help=(
            "Gemini model id. Defaults to gemini-2.5-flash (the validated "
            "slow-tier default). Pass gemini-2.5-pro to opt into Pro (10x "
            "cost, subject to free-tier quota)."
        ),
    )
    p.add_argument("--no-context-build", action="store_true",
                   help="Skip rebuilding context.md (use existing one)")
    p.add_argument("--resume", action="store_true",
                   help="Resume from run_log.jsonl (rebuilds conversation history)")
    p.add_argument("--rebuild-context", action="store_true",
                   help="With --resume, force a fresh build_context (default: skip on resume to avoid replay drift)")
    p.add_argument("--cost-budget", type=float, default=None,
                   help="USD cost cap; checkpoint fires when exceeded")
    p.add_argument("--auto-approve", choices=["off", "safe", "all"], default="off",
                   help="Checkpoint mode: off=interactive, safe=auto-approve non-regression submits only, all=auto-approve everything (tests only)")
    p.add_argument("--i-know-what-im-doing", action="store_true",
                   help="Required acknowledgement when using --auto-approve all (bypasses every checkpoint gate)")
    args = p.parse_args(argv)

    # F5: --auto-approve all bypasses *every* checkpoint, including the Kaggle
    # daily-submit cap, cost-budget overrun, and metric overrides. Require an
    # explicit second flag so a copied command line cannot silently disarm
    # those gates.
    if args.auto_approve == "all" and not args.i_know_what_im_doing:
        print(
            "ERROR: --auto-approve all disables the Kaggle daily-submit cap, "
            "cost budget, and metric overrides. Pass --i-know-what-im-doing "
            "to acknowledge.",
            file=sys.stderr,
        )
        sys.exit(2)

    return args


def run(argv: list[str]) -> int:
    args = _parse_args(argv)
    load_dotenv()

    # F5: surface the safety bypass at run start so it's visible in logs/CI.
    if args.auto_approve == "all":
        Console(stderr=True).print(Panel(
            "WARNING: --auto-approve all is ENABLED. Kaggle daily submit cap, "
            "cost budget, and metric overrides will NOT prompt. Use this only "
            "in scripted tests.",
            title="safety gate off",
            style="yellow",
        ))

    comp_path = Path(args.workspace_path)
    workspace = Workspace.create(root=comp_path)

    api_key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
    if not api_key:
        print("ERROR: no GEMINI_API_KEY / GOOGLE_API_KEY in env", file=sys.stderr)
        return 2

    # Build context.md unless user opts out. On --resume, default to skipping
    # the rebuild — regenerating context.md while the resumed conversation
    # history references the original would cause replay drift. Opt back in
    # via --rebuild-context.
    skip_context = args.no_context_build or (args.resume and not args.rebuild_context)
    if not skip_context:
        try:
            kaggle = KaggleClient()
            build_context(workspace=workspace, kaggle_client=kaggle)
        except Exception as e:  # noqa: BLE001
            # Non-fatal — the agent can still read whatever context.md exists,
            # or none at all. Surface the warning.
            print(f"warning: context build failed: {e!r}", file=sys.stderr)

    ledger = CostLedger()
    llm = GeminiClient(
        api_key=api_key,
        ledger=ledger,
        competition=workspace.name,
        default_model=args.model,
    )

    # Checkpoint handler
    from kaggle_slayer.harness import checkpoints as cp  # noqa: PLC0415
    if args.auto_approve == "safe":
        handler = cp.CheckpointHandler(mode=cp.HandlerMode.AUTO_SAFE, journal=Journal(workspace))
    elif args.auto_approve == "all":
        handler = cp.CheckpointHandler(
            mode=cp.HandlerMode.STUB,
            journal=Journal(workspace),
            stub_decision=cp.Decision.APPROVE,
        )
    else:
        handler = cp.CheckpointHandler(mode=cp.HandlerMode.INTERACTIVE, journal=Journal(workspace))

    # Resume?
    resume_from = None
    if args.resume:
        from kaggle_slayer.harness import resume as resume_mod  # noqa: PLC0415
        try:
            resume_from = resume_mod.rebuild_conversation(workspace)
        except resume_mod.ResumeError as e:
            print(f"resume failed: {e}", file=sys.stderr)
            return 3

    solver = Solver(
        workspace=workspace,
        llm_client=llm,
        target_col=args.target or "target",
        problem_type=args.problem_type,
        metric_name=args.metric,
        max_iterations=args.max_iterations,
        time_budget_s=args.time_budget_s,
        checkpoint_handler=handler,
        cost_ledger=ledger,
        cost_budget_usd=args.cost_budget,
        kaggle_client=KaggleClient(),
    )
    result = solver.solve(resume_from=resume_from)

    print(f"\nstatus: {result.status}")
    print(f"iterations: {result.iterations}")
    print(f"summary: {result.summary}")
    print(f"spent: ${ledger.total_for(competition=workspace.name):.4f}")

    return 0 if result.status == "done" else 1


def main() -> None:
    sys.exit(run(sys.argv[1:]))


if __name__ == "__main__":
    main()
