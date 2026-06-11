# CLAUDE.md

Guidance for Claude Code when working in this repo.

## What this is

KaggleSlayer V2 — an LLM-agent harness for tabular Kaggle competitions. A Gemini-driven
Solver writes `agent/fe.py` + `agent/model.py`; the trusted Python harness runs leak-free
cross-validation, journals every tool call, gates Kaggle submissions, and ships a CSV.

- **Structure & data flow:** `docs/architecture.md` (module map, solve walkthrough, telemetry, test tiers)
- **Why-decisions:** `docs/adr/` (trust split, leak-free CV, checkpoint gate, journal, telemetry, LLM client)
- **Status, scope, roadmap:** `GOALS.md` — read it before deciding what is in or out of scope
- **History:** specs in `docs/superpowers/specs/`, week-by-week plans in `docs/superpowers/plans/`

## Hard rules

1. **Leak-free CV is the inviolable contract.** `train_cv` only ever passes one fold's
   training data to the agent's `fit_feature_transformer`. Never add a code path that hands
   the full dataset (or any val/test data) to agent-written code during fit. See
   `kaggle_slayer/harness/cv.py`, ADR 0002, and spec §6.
2. **Agent code is never trusted at import time.** Anything loaded as agent-written code
   MUST pass `sandbox.lint_module()` first — load it via `cv.load_agent_module`. See ADR 0001.
3. **The harness owns metrics, CV strategies, and checkpoint triggers.** Extend
   `kaggle_slayer/harness/registry/` or `checkpoints.CheckpointTrigger` — never inline at
   the call site.
4. **Kaggle is touched only through `KaggleClient`** (`harness/kaggle_client.py`), and
   submissions flow only through the checkpoint-gated `submit_kaggle` tool. Direct
   `kaggle.api.*` calls anywhere else are forbidden. See ADR 0003.
5. **The agent never journals directly.** Tool records via `Journal.log_tool_call` /
   `log_tool_error`; checkpoint records via `CheckpointHandler.request`; submit history in
   `submissions/leaderboard.jsonl`, not `run_log.jsonl`. See ADR 0004.
6. **Telemetry never crashes the agent.** MLflow/OTel/calibration failures are caught,
   logged (stdlib `logging`; MLflow → `<workspace>/mlflow_errors.log`), and execution
   continues. See ADR 0005.

## Common commands

```bash
pip install -e ".[dev,dashboard]"
pytest -m "not slow"                                  # ~5s, 419 tests, no keys — the merge gate
pytest -m slow                                        # opt-in real-Gemini ($)
pytest -m chaos                                       # chaos tier (injected LLM failures)
ruff check kaggle_slayer tests
mypy kaggle_slayer/harness                            # what CI runs
mypy kaggle_slayer/harness kaggle_slayer/agent        # local standard — keep both clean
kaggle-slayer competitions/<name> --target <col> --metric <m>   # full solve (--target required)
kaggle-slayer-dashboard                               # read-only Streamlit dashboard
```

Or just run `/gates` for the full merge gate. CLI flags worth knowing: `--resume`,
`--cost-budget <USD>`, `--auto-approve {off,safe,all}`, `--model`, `--no-download`,
`--competition <slug>`.

## Conventions

- TDD: write the failing test first. Four test tiers (unit/integration/chaos/slow) — keep them intact.
- Mypy strict and clean on `harness/` AND `agent/` locally; **CI enforces harness only**.
- Errors at boundaries are typed: `CVError`, `ToolError`, `ResumeError`, `TransientLLMError`, `DownloadError`.
- CLI output uses `rich`; library code uses stdlib `logging`; no `print` in library code.
- Tool handlers take `ctx` first (structural contract — see `SolverContext`).
- **Secret hygiene:** `.env` is gitignored and must never be committed (`git add -f` included);
  commit `.env.example` instead. If a key leaks, rotate it provider-side — scrubbing files
  does not revoke keys.
- Don't pre-build roadmap items from `GOALS.md`.

## Workflows

- `/gates` — run the merge gate (tests + ruff + mypy)
- `/solve <workspace> <target> [metric]` — run a competition end-to-end
- `/harness-review` — audit the current diff against the hard rules
- `/new-adr <topic>` — scaffold the next ADR

## Files that pin the architecture

Read `docs/architecture.md` §"Files that pin the architecture" before touching
`harness/cv.py`, `harness/checkpoints.py`, `agent/solver.py`, `agent/llm_client.py`,
or `harness/telemetry/`.
