# CLAUDE.md

Guidance for Claude Code when working in this repo.

## What this is

KaggleSlayer V2 — an LLM-agent harness for tabular Kaggle competitions. A Gemini-driven Solver reads the competition, plans a strategy, writes feature-engineering and model code, and the trusted Python harness runs leak-free cross-validation, journals every tool call, gates submissions through a checkpoint, and ships a CSV.

Full design in `docs/superpowers/specs/2026-05-14-llm-agent-harness-design.md`. Per-week plans in `docs/superpowers/plans/`. The legacy V1 AutoML pipeline was archived and removed; only the leak-free-CV idea was carried forward.

**Current state:** Weeks 1–5 of 6 shipped. The harness is end-to-end runnable — `kaggle-slayer <workspace> --target <col>` invokes Gemini, the agent writes `agent/fe.py`/`agent/model.py`, calls `train_cv` and `submit_local`, and (gated) `submit_kaggle`. Every run also emits an OpenTelemetry trace to `<workspace>/otel.jsonl`, logs one MLflow run per `train_cv`, appends a row to `~/.kaggle_slayer/calibration.jsonl` on each successful `submit_kaggle`, and captures unhandled exceptions as JSON crash reports under `~/.kaggle_slayer/errors/`. Real-Gemini acceptance gate is green on gemini-2.5-flash (~$0.005–0.02 per synthetic-comp solve). 375 non-slow tests, 8 slow-tier (opt-in).

## Hard rules

1. **Leak-free CV is the inviolable contract.** `train_cv` only ever passes one fold's training data to the agent's `fit_feature_transformer`. Never add a code path that hands the full dataset (or any val/test data) to agent-written code during fit. See `kaggle_slayer/harness/cv.py` and spec §6.
2. **Agent code is never trusted at import time.** Any file that will be loaded as agent-written (`fe.py`, `model.py`, anything under `agent/`) MUST pass `sandbox.lint_module()` first. The lint is wired into `cv.train_cv` and `submit_local`; don't bypass.
3. **The harness owns metrics, CV strategies, and checkpoint triggers.** Adding a metric or splitter goes in `kaggle_slayer/harness/registry/`. Adding a checkpoint trigger goes in `kaggle_slayer/harness/checkpoints.py:CheckpointTrigger`. Not inline at the call site.
4. **Kaggle is touched only through `submit_kaggle`.** The tool is checkpoint-gated (first submit always blocks; subsequent submits block on score regression). Direct `kaggle.api.*` calls outside `harness/kaggle_client.py` are forbidden.
5. **The agent never journals directly.** Tool-call records go through `Journal.log_tool_call` / `log_tool_error`; checkpoint records go through `CheckpointHandler.request`. Submit history lives in `submissions/leaderboard.jsonl`, not in `run_log.jsonl`.
6. **Telemetry never crashes the agent.** Failures from MLflow, OTel, and the calibration log are caught, logged via stdlib `logging` and `<workspace>/mlflow_errors.log` (for MLflow), and execution continues. The Solver loop must not abort because a tracking surface is down.

## Layout

```
kaggle_slayer/
├─ cli.py                          # kaggle-slayer entry point (run() wraps _run_inner with error capture)
├─ harness/                        # trusted, no LLM
│   ├─ cv.py                       # train_cv leak-free contract + lint gate
│   ├─ workspace.py                # per-comp Workspace dataclass
│   ├─ journal.py                  # append-only run_log.jsonl + notes.jsonl
│   ├─ resume.py                   # rebuild_conversation from run_log (delegates stuck-loop to telemetry.behavior)
│   ├─ checkpoints.py              # typed gate, four modes, journalled decisions
│   ├─ context.py                  # context.md builder
│   ├─ kaggle_client.py            # KaggleClient wrapper (view/list/download/submit)
│   ├─ sandbox.py                  # AST lint + run_subprocess (RLIMIT_AS/CPU/NPROC/FSIZE)
│   ├─ registry/
│   │   ├─ metrics.py              # 6 metrics with .kind + .higher_is_better
│   │   └─ cv_strategies.py        # KFold/StratifiedKFold/TimeSeriesSplit/GroupKFold
│   └─ telemetry/                  # Week 5
│       ├─ otel.py                 # JSONL tracer + Span / Tracer / make_tracer
│       ├─ calibration.py          # ~/.kaggle_slayer/calibration.jsonl record/read_history
│       ├─ errors.py               # ~/.kaggle_slayer/errors/<ts>.json + redaction + 100-file rotation
│       ├─ behavior.py             # BehaviorMetrics + compute_metrics + detect_stuck_loop
│       └─ mlflow_logger.py        # one experiment per comp, one run per train_cv, tags + wall_clock_s
├─ agent/                          # LLM side
│   ├─ solver.py                   # reason-act loop + SolverContext (last_cv_mean + best_cv_mean) + OTel spans
│   ├─ llm_client.py               # GeminiClient (Content/Part + function_call/response) + TransientLLMError
│   ├─ retrying_client.py          # LLMClient adapter: TransientLLMError + exponential backoff (Week 5)
│   ├─ tools.py                    # Tool / ToolRegistry / ToolError
│   ├─ cost_ledger.py              # per-call USD ledger
│   ├─ handlers/
│   │   ├─ files.py                # read_context, read_file, write_file, sample_rows, take_note
│   │   ├─ ml.py                   # set_cv, train_cv (+MLflow), submit_local, done, set_metric, submit_kaggle (+calibration), request_human_approval
│   │   └─ python.py               # run_python (sandboxed escape hatch)
│   └─ prompts/system.md           # Solver system prompt
└─ dashboard/                      # Week 5 — Streamlit, read-only over disk artifacts
    ├─ app.py                      # kaggle-slayer-dashboard entry; routes Portfolio / Competition detail
    ├─ portfolio.py                # list comps + best CV + cost + tool count
    └─ comp_detail.py              # timeline + cost + calibration + notes + submissions + behavior metrics

tests/
├─ unit/                           # ~360 unit tests
├─ integration/                    # fake-LLM + real-Gemini (slow tier, opt-in)
├─ chaos/                          # Week 5 — FailureInjectingLLMClient + 5% rate, seeded determinism
└─ fixtures/                       # fake_llm.py + synthetic_comp.py + stubs
```

## Common commands

```bash
pip install -e ".[dev,dashboard]"
pytest -m "not slow"                                       # ~5s, 375 tests
pytest --cov=kaggle_slayer/harness --cov=kaggle_slayer/agent -m "not slow"
ruff check kaggle_slayer tests
mypy kaggle_slayer/harness kaggle_slayer/agent             # strict on both
kaggle-slayer competitions/<name> --target <col> --metric <m>  # full solve
kaggle-slayer-dashboard                                    # Streamlit portfolio + comp-detail
pytest -m slow                                             # opt-in real-Gemini ($)
pytest -m chaos                                            # chaos tier (5% LLM failure injection)
```

CLI flags worth knowing:
- `--resume` — replay `run_log.jsonl` and continue an aborted run (auto-implies skip context rebuild; pass `--rebuild-context` to override).
- `--cost-budget <USD>` — checkpoint when the cost ledger crosses; APPROVE doubles the budget.
- `--auto-approve {off,safe,all}` — gate mode. `all` requires `--i-know-what-im-doing`.
- `--model <name>` — default `gemini-2.5-flash` (Pro requires Tier 1 billing on Google AI Studio).

## Conventions

- Mypy strict on `kaggle_slayer/harness` AND `kaggle_slayer/agent`. Tests aren't type-checked.
- TDD: write the failing test first, then the implementation. Commits demonstrate the pattern week by week.
- Errors at boundaries raise typed exceptions: `CVError`, `ToolError`, `ResumeError`, `TransientLLMError`.
- CLI output uses `rich`. Library code uses stdlib `logging`. No `print` in library code.
- Tool handlers take `ctx` as the first positional arg (structural contract, not nominal — see `SolverContext`).
- Don't add metrics/CV strategies/checkpoint triggers inline. Extend the registry / enum.
- Don't add features outside the active plan. Each week's scope is enforced by its own plan.

## What's coming (do not pre-build)

- **Week 6** — three live Kaggle Playground comps + docs (`docs/architecture.md`, `docs/adr/*.md`, `.claude/commands/*.md`, `.claude/agents/harness-reviewer.md`), MLflow artifact logging (fe.py / model.py / oof_preds.npy), LB-score backfill into the calibration log, the fe_v01↔fe_v02 dashboard diff page, and the cross-comp dashboard page.

## Files that pin the architecture (read these before touching the load-bearing parts)

- `kaggle_slayer/harness/cv.py` — the leak-free contract. Don't refactor without re-reading spec §6.
- `kaggle_slayer/harness/checkpoints.py` — the gate enum lists every trigger; don't add ad-hoc human-in-the-loop calls elsewhere.
- `kaggle_slayer/agent/solver.py` — the reason-act loop. SolverContext is the typed state object every handler mutates.
- `kaggle_slayer/agent/llm_client.py` — Gemini multi-turn protocol (`Content`/`Part`/`function_call`/`function_response`). Tool schemas are stripped of Gemini-unsupported keys (`additionalProperties`, `$schema`, …) before send.
- `kaggle_slayer/harness/telemetry/otel.py` — JSONL tracer, single-process / single-threaded by design. `make_tracer` writes a `run:<name>` marker; Solver wraps the loop in a `solve.loop` span, each LLM call in `llm.call`, each tool dispatch in `tool:<name>`.
- `kaggle_slayer/harness/telemetry/mlflow_logger.py` — context manager that defaults to `file:~/.kaggle_slayer/mlruns` (respects `MLFLOW_TRACKING_URI`). MLflow failures route to `<workspace>/mlflow_errors.log` and never crash the agent.
