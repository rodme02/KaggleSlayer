# KaggleSlayer Architecture

How the system fits together: the trust boundary, the module map, what happens
during a solve, what lands on disk, and how the test tiers are cut. The *why*
behind each decision lives in [`docs/adr/`](adr/README.md); status and scope in
[`GOALS.md`](../GOALS.md); day-to-day rules in [`CLAUDE.md`](../CLAUDE.md).

## Trust boundary

The repo is split into two worlds:

- **Trusted harness** (`kaggle_slayer/harness/`) — plain Python written by humans,
  type-checked strict, owns everything with consequences: cross-validation, metrics,
  Kaggle access, the submission gate, the journal, telemetry.
- **Untrusted agent output** — `agent/fe.py`, `agent/model.py`, and anything else the
  LLM writes into a workspace. It is **never trusted at import time**: every load goes
  through the sandbox AST lint (`sandbox.lint_module`, reached via
  `cv.load_agent_module`), `run_python` executes under POSIX resource limits
  (`RLIMIT_AS/CPU/NPROC/FSIZE` in `sandbox.run_subprocess`), and during CV fit the
  agent's code only ever sees one fold's training rows (`cv.train_cv`).

The threat model is *accidental* damage, not a malicious human: an LLM that
helpfully reads the full dataset inside a transformer (leakage), writes outside its
workspace, or spins in a resource-eating loop. The lint is a denylist — novel
escapes are possible — so the invariant is enforced at the loader: nothing imports
agent code without passing the lint first (ADR 0001, ADR 0002).

## Module map

```
kaggle_slayer/
├─ cli.py                          # kaggle-slayer entry point; run() wraps _run_inner with crash capture; exit codes 0/1/2/3/4/130
├─ harness/                        # trusted, no LLM
│   ├─ cv.py                       # train_cv leak-free contract; load_agent_module = lint + import (the ONE sanctioned loader); CVError
│   ├─ workspace.py                # Workspace dataclass: raw/, agent/(versions/, scratch/), artifacts/, submissions/, context.md, run_log/notes paths
│   ├─ journal.py                  # append-only fsync'd run_log.jsonl + notes.jsonl; iter_records/list_notes skip corrupt trailing lines
│   ├─ resume.py                   # summarize (delegates counting to telemetry.behavior) + rebuild_conversation (replay journal as LLM history)
│   ├─ checkpoints.py              # CheckpointTrigger enum + Decision + four HandlerModes (interactive/auto_safe/stub/callable); journals kind='checkpoint'
│   ├─ context.py                  # context.md builder (competition brief from raw/ + Kaggle metadata)
│   ├─ data.py                     # ensure_competition_data — auto-fetch + unzip into raw/, skip when CSVs present; DownloadError
│   ├─ kaggle_client.py            # KaggleClient wrapper (view/list/download/submit) — the ONLY module allowed to touch kaggle.api.*
│   ├─ sandbox.py                  # AST denylist lint (imports/builtins/path prefixes, alias-aware) + run_subprocess with rlimits
│   ├─ registry/
│   │   ├─ metrics.py              # 6 metrics, each with kind / higher_is_better / needs_proba; get() + list_metrics()
│   │   └─ cv_strategies.py        # kfold / stratified_kfold / time_series / group_kfold; get() + list_strategies() + auto_select()
│   └─ telemetry/
│       ├─ otel.py                 # in-tree JSONL tracer (Span/Tracer/make_tracer) — no opentelemetry dependency; single-process by design
│       ├─ calibration.py          # ~/.kaggle_slayer/calibration.jsonl record/read_history (CV↔LB pairs; lb_score backfill is roadmap)
│       ├─ errors.py               # crash reports to ~/.kaggle_slayer/errors/<ts>.json; env-key redaction; keeps last 100
│       ├─ behavior.py             # BehaviorMetrics over the journal: turns, error rate, metric-direction-aware best CV; detect_stuck_loop
│       └─ mlflow_logger.py        # one MLflow experiment per comp, one run per train_cv; failures → <workspace>/mlflow_errors.log, never raise
├─ agent/                          # LLM side
│   ├─ solver.py                   # reason-act loop; SolverContext (typed state all handlers mutate); OTel spans; tool-result caps
│   ├─ llm_client.py               # GeminiClient — Content/Part + function_call/function_response protocol; TransientLLMError; schema stripping
│   ├─ retrying_client.py          # LLMClient adapter: retries TransientLLMError with exponential backoff
│   ├─ tools.py                    # Tool / ToolRegistry / ToolError
│   ├─ cost_ledger.py              # per-call USD ledger (~/.kaggle_slayer/cost_ledger.jsonl); price table per model
│   ├─ handlers/
│   │   ├─ files.py                # read_context, read_file, write_file, sample_rows, take_note (paths resolved under the workspace)
│   │   ├─ ml.py                   # set_cv, train_cv (+MLflow +versions archive), submit_local, submit_kaggle (gated, +calibration), set_metric, done, request_human_approval
│   │   └─ python.py               # run_python — sandboxed subprocess escape hatch
│   └─ prompts/system.md           # Solver system prompt
└─ dashboard/                      # read-only Streamlit over disk artifacts; never touches Kaggle or Gemini
    ├─ app.py                      # kaggle-slayer-dashboard entry; re-execs under streamlit (forwards CLI flags)
    ├─ portfolio.py                # comp cards: best CV (via behavior.compute_metrics), cost, tool calls
    └─ comp_detail.py              # timeline, behavior metrics, calibration, notes, submissions
```

## Anatomy of a solve

`kaggle-slayer competitions/titanic --target Survived --metric accuracy`

1. **CLI** (`cli.run` → `_run_inner`): parse args (`--target` is required), check
   `GEMINI_API_KEY`/`GOOGLE_API_KEY` *before* creating any directories, resolve the
   workspace path (so `kaggle-slayer .` gets a real name), `Workspace.create`.
2. **Data** (`harness/data.py:ensure_competition_data`): if `raw/` has no top-level
   CSV and `--no-download` wasn't passed, download the competition (slug = dir name
   or `--competition`), unzip, delete the archives. A needed download that fails
   exits 2 with an actionable message — better than burning Gemini tokens on empty data.
3. **Context** (`harness/context.py:build_context`): write `context.md` (data brief,
   metric, rules). Non-fatal on failure — the agent can run with a thinner brief.
4. **Solver loop** (`agent/solver.py:Solver.solve`): system prompt + `context.md`
   seed the history; each iteration calls the LLM (`llm.call` inside an `llm.call`
   OTel span), dispatches any tool calls through `ToolRegistry.invoke`
   (`tool:<name>` spans), journals every call via `Journal.log_tool_call`/
   `log_tool_error`, and feeds capped results back as tool messages. Exits on
   `done`, `max_iterations`, wall-clock budget, or cost budget (the latter two can
   be extended through checkpoints).
5. **Training** (`handlers/ml.py:train_cv` → `harness/cv.py:train_cv`): lint
   `fe.py`/`model.py`, pick the CV strategy (agent's `set_cv` or `auto_select`),
   then per fold: call `fit_feature_transformer` **with that fold's training rows
   only**, transform both sides, `fit_model`, score with the registry metric.
   Success archives the code into `agent/versions/` and logs one MLflow run.
6. **Submission** (`submit_local` → gated `submit_kaggle`): `submit_local` fits on
   the full train set and writes a CSV under `submissions/`. `submit_kaggle` is
   checkpoint-gated: the first submit always blocks for a human; later submits
   block when CV regressed (direction-aware per metric). On success it appends one
   record to `submissions/leaderboard.jsonl` and a CV↔LB calibration row.
7. **Exit codes:** 0 done · 1 finished without `done` (e.g. max_iterations) ·
   2 setup error (keys/download) · 3 resume failure · 4 unhandled crash (report
   written) · 130 interrupted.

`--resume` replays `run_log.jsonl` through `resume.rebuild_conversation` so an
aborted run continues with its full tool history (checkpoint records are skipped;
context.md is not rebuilt unless `--rebuild-context`).

## On-disk layout

Per workspace (`competitions/<name>/` — gitignored):

```
raw/                  # train.csv, test.csv, ... (auto-downloaded or manual)
context.md            # the brief the agent reads first
agent/fe.py           # agent-written feature engineering (current)
agent/model.py        # agent-written model factory (current)
agent/versions/       # fe_vNN.py / model_vNN.py archived per successful train_cv
agent/scratch/        # run_python workspace
artifacts/            # harness-owned artifacts
submissions/*.csv     # submit_local outputs
submissions/leaderboard.jsonl   # one record per successful submit_kaggle
run_log.jsonl         # append-only journal: tool_call / tool_error / checkpoint
notes.jsonl           # agent's take_note scratchpad
otel.jsonl            # span trace for the run
mlflow_errors.log     # MLflow failures (telemetry never crashes the agent)
```

Global (`~/.kaggle_slayer/`): `cost_ledger.jsonl` (every LLM call, USD),
`calibration.jsonl` (CV↔LB rows), `errors/` (last 100 crash reports),
`mlruns/` (MLflow file store unless `MLFLOW_TRACKING_URI` is set).

## Telemetry surfaces

| Surface | Where | Failure behavior |
| --- | --- | --- |
| OTel-style spans | `<workspace>/otel.jsonl` (in-tree tracer, single-process) | logged, loop continues |
| MLflow | `~/.kaggle_slayer/mlruns` — one run per `train_cv` | `<workspace>/mlflow_errors.log`, loop continues |
| Calibration | `~/.kaggle_slayer/calibration.jsonl` per `submit_kaggle` | logged, loop continues |
| Crash reports | `~/.kaggle_slayer/errors/<ts>.json`, env redacted, 100 kept | reported on stderr, exit code preserved |
| Behavior metrics | computed on demand from the journal | n/a (pure reads) |

The contract (hard rule #6, ADR 0005): **no tracking failure may abort a solve.**
All JSONL readers tolerate a corrupt trailing line (crash mid-write).

## Test architecture

| Tier | Marker | May touch | Size |
| --- | --- | --- | --- |
| Unit | (none) | tmp dirs only — no network, no keys | ~370 |
| Integration | `integration` | fake LLM + synthetic comps, still offline | ~30 |
| Chaos | `chaos` | fake LLM with 5% injected failures, seeded | small |
| Slow | `slow` (deselected by default) | **real Gemini** ($), opt-in | 8 |

`pytest -m "not slow"` (408 passed, 1 env-gated skip, ~5s) is the merge gate and
exactly what CI runs on Linux 3.11 + 3.12, plus `ruff check` and
`mypy kaggle_slayer/harness`. Fixtures: `tests/fixtures/fake_llm.py` (scriptable
LLM client) and `tests/fixtures/synthetic_comp.py` (generates toy competitions).

## Files that pin the architecture

Read these before touching the load-bearing parts:

- `kaggle_slayer/harness/cv.py` — the leak-free contract. Don't refactor without re-reading spec §6 / ADR 0002.
- `kaggle_slayer/harness/checkpoints.py` — the gate enum lists every trigger; don't add ad-hoc human-in-the-loop calls elsewhere.
- `kaggle_slayer/agent/solver.py` — the reason-act loop. `SolverContext` is the typed state object every handler mutates.
- `kaggle_slayer/agent/llm_client.py` — Gemini multi-turn protocol (`Content`/`Part`/`function_call`/`function_response`). Tool schemas are stripped of Gemini-unsupported keys before send.
- `kaggle_slayer/harness/telemetry/otel.py` — JSONL tracer, single-process/single-threaded by design.
- `kaggle_slayer/harness/telemetry/mlflow_logger.py` — context manager defaulting to `file:~/.kaggle_slayer/mlruns`; failures must keep routing to `mlflow_errors.log`.
