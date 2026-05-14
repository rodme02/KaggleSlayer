# KaggleSlayer V2 — LLM-Agent Harness Design

**Status:** Approved design, ready for implementation planning
**Date:** 2026-05-14
**Author:** Rodrigo Medeiros (with Claude)
**Supersedes:** KaggleSlayer V1 (the existing AutoML pipeline; see `legacy/` after rebuild)

---

## 1. Goal

Build a Kaggle competition harness driven by an LLM agent that, given a competition name, autonomously reads the problem, plans a strategy, writes feature-engineering and model code, evaluates it with leak-free cross-validation, and produces a submission that reliably scores at or above the public-leaderboard median.

The agent runs autonomously by default but **pauses at named checkpoints** (mixed autonomy) — most importantly, before any Kaggle submission.

## 2. Scope

**In scope (V1):** tabular supervised competitions — binary classification, multi-class classification, regression, ordinal regression, time-series tabular forecasting, multi-table tabular.

**Out of scope (V1):** NLP, computer vision, audio, multimodal, RL, code competitions. The architecture is designed so these can be added as parallel "tracks" later, but no Phase-2/3 work is committed here.

**Success bar:** reliably hit the **median public-leaderboard score (top 50%)** on tabular competitions, end-to-end autonomous from `kaggle-slayer <comp>` to `submission.csv`.

## 3. Non-goals

- Gold/silver medal performance. The pattern recognition required for top finishes is beyond what a generic LLM-driven harness can deliver consistently. Aiming for it would over-engineer the system.
- Cloud-first execution. Single-machine (MacBook Pro M5, 24 GB RAM) is the target.
- Distributed training, multi-GPU, fancy infra. Cloud-burst stays as a future flag.
- Supporting "any" LLM at launch. Gemini Pro is the V1 stack; the abstraction allows future swaps but isn't a polyglot harness.

## 4. Constraints

| Constraint | Value |
|---|---|
| LLM stack | Google Gemini Pro (tier 1), behind a thin `LLMClient` abstraction |
| Compute | MacBook Pro M5, 24 GB RAM, no NVIDIA GPU |
| Autonomy | Mixed — runs autonomously, blocks at named checkpoints |
| Wall-clock per comp | 90 min default; extendable via checkpoint approval |
| Cost budget | `--cost-budget` flag, default $5/comp, checkpoint when hit |
| Code-execution surface | LLM-written Python runs in a sandboxed subprocess (workspace-scoped CWD + AST lint + resource limits) |

## 5. Architecture

### 5.1 Shape

One LLM agent ("Solver") runs a tool-use loop. A trusted Python harness owns the things that must not be left to LLM judgment (leak-free CV, metric scoring, Kaggle API guardrails) and exposes them as typed tool calls. The agent owns everything else.

```
┌──────────────────────────────────────────────────────────────────┐
│  Solver Agent  (Gemini Pro · tool-use loop · long context)       │
│   - reads competition_context.md (auto-generated brief)          │
│   - reason → call tool → read result → reason → ...              │
│   - persists scratchpad notes between turns                      │
└──────────────────────────────────────────────────────────────────┘
         │  typed tool calls
         ▼
┌──────────────────────────────────────────────────────────────────┐
│  Trusted Harness (pure Python, no LLM)                           │
│   ├─ Competition tools   load_comp · profile · rules · metric    │
│   ├─ Code tools          write_file · run_python · diff · grep   │
│   ├─ ML tools            train_cv · evaluate · submit_local      │
│   │                       submit_kaggle · read_leaderboard       │
│   ├─ Memory tools        take_note · list_notes · summarize_log  │
│   └─ Checkpoint gate     request_human_approval                  │
└──────────────────────────────────────────────────────────────────┘
         │  file I/O, MLflow logs, sandboxed subprocesses
         ▼
┌──────────────────────────────────────────────────────────────────┐
│  Per-competition workspace  competitions/<name>/                 │
│     raw/  context.md  agent/  artifacts/  submissions/           │
│     notes.jsonl  run_log.jsonl                                   │
└──────────────────────────────────────────────────────────────────┘
```

### 5.2 Agent loop

On `kaggle-slayer <comp>`:

1. **Bootstrap context** — Kaggle API pulls competition metadata; harness parses overview/data/evaluation pages, samples the data, writes `context.md` with: problem framing, target/ID candidates, parsed metric, data shape, missingness, public LB top scores.
2. **Spawn Solver** — Gemini Pro instance with `context.md` in the system message and the toolbox bound.
3. **Iterate** — agent plans CV strategy, writes `agent/fe.py` and `agent/model.py`, calls `train_cv`, reads result, fixes/iterates, calls `submit_local` to dry-run, hits the **checkpoint gate**, on approval calls `submit_kaggle`.
4. **Terminate** on: wall-clock budget, N submissions without improvement, explicit agent stop, or human abort.

### 5.3 Why one agent, not a stage pipeline

Stage boundaries (DataScout → FE → Model) leak information across stages anyway: FE choices depend on model family, model choice depends on metric and data shape, metric handling depends on data quirks. A single agent with the full picture makes fewer dumb decisions than a relay of agents that each see a slice. Modern Gemini/Claude tool-use is reliable enough that the old multi-agent gymnastics are dead weight. See [ADR-0001](../../adr/0001-single-agent.md).

## 6. The leak-free CV contract

The most consequential correctness property. See [ADR-0003](../../adr/0003-leak-free-contract.md) for the rationale.

### 6.1 Contract

The agent writes two files per competition: `fe.py` and `model.py`. Each exposes a single fit function. The harness — not the agent — runs CV, calling the agent's fit functions **with one fold's training data at a time**. The agent literally cannot see the full dataset inside a fit call.

```python
# agent/fe.py — written by Solver
def fit_feature_transformer(train_df, target_col):
    """Fit on train fold only. Return an object with .transform(df) -> df."""
    ...

# agent/model.py — written by Solver
def fit_model(X_train, y_train, problem_type, metric_name):
    """Return an object with .predict(X). predict_proba if metric needs it."""
    ...
```

```python
# harness/cv.py — trusted, never written by agent
def train_cv(fe_path, model_path, train_df, target_col, cv, metric) -> CVResult:
    oof = np.empty(len(train_df))
    for train_idx, val_idx in cv.split(train_df, target_col):
        train_fold, val_fold = train_df.iloc[train_idx], train_df.iloc[val_idx]

        fe = load(fe_path).fit_feature_transformer(train_fold, target_col)
        X_tr = fe.transform(train_fold.drop(columns=[target_col]))
        X_va = fe.transform(val_fold.drop(columns=[target_col]))

        m = load(model_path).fit_model(X_tr, train_fold[target_col], ...)
        preds = m.predict_proba(X_va) if metric.needs_proba else m.predict(X_va)
        oof[val_idx] = preds
    return CVResult(oof=oof, score=metric.score(train_df[target_col], oof))
```

### 6.2 Why this beats the legacy `TransformerMixin`

The legacy fix was structural — easy to subvert (any global `fit` outside the pipeline undoes it, which is what happened with target encoding and `KBinsDiscretizer` in the original codebase). The new contract is **temporal**: the harness physically only passes the train fold during fit. If the agent wants to compute global statistics, it must do it inside `fit_feature_transformer`, on the data it was given. There is no other entry point.

### 6.3 Registries owned by the harness

- **CV strategy registry** — auto-picks from problem type + data shape: classification → `StratifiedKFold`; regression → `KFold`; time-indexed → `TimeSeriesSplit`; grouped → `GroupKFold`. Agent override via `set_cv(kind, **params)` tool call; choice logged.
- **Metric registry** — pre-registers Kaggle metrics that matter for tabular: `RMSE`, `RMSLE`, `MAE`, `R2`, `LogLoss`, `AUC`, `Accuracy`, `F1` (binary/macro/weighted), `QWK`, `MAP@K`, `MCRMSE`, `Pinball`. Harness picks from parsed competition metric. Agent override is checkpointed (changing scoring is a big deal).

### 6.4 Final fit

After the agent locks in the final `fe.py`/`model.py`, harness runs **one final fit on the full training set** with the same code, persists `artifacts/pipeline.pkl`, generates test predictions. Leak-free only applies to CV-score interpretation, not to using all training signal for the deployed predictor.

### 6.5 Anti-cheat

The harness calls `fit_feature_transformer` in-process for performance (subprocess-per-fold would multiply overhead). The leak-free guarantee comes from three things:

1. **AST lint forbids the agent's `fe.py`/`model.py` from reading competition data files directly** (no `pd.read_csv("raw/*")`, no `Path("raw").glob(...)`, no `open("raw/...", ...)`). The only data the fit functions ever see is what the harness passes in. Enforced at module-load time; rejected files don't run.
2. **Harness verifies `fe.transform(df).shape[0] == df.shape[0]`** (no row dropping that would change splits).
3. **Optional permuted-target sanity check** after the first run: if shuffling `y` and re-running keeps the same score, something is badly leaking.

If the threat model ever changes (untrusted LLM provider, multi-tenant execution), upgrade to subprocess-per-fold isolation with the train fold passed as a pickled DataFrame over stdin. That's out of scope for V1.

## 7. The toolbox

Five groups. Concrete signatures. JSON schemas live in `agent/tools.py` as the single source of truth.

```python
# Competition (read-only)
load_competition(name)                    -> CompetitionContext
read_overview() / read_evaluation() / read_data_description()
list_tables() / profile_data(table) / sample_rows(table, n, random)
read_leaderboard(top_n=50)

# Code (workspace-scoped)
write_file(path, content) / read_file(path) / apply_diff(path, diff) / grep(pattern, path)
run_python(code, timeout_s=60)   # for plotting, peeks, debug — NOT for CV

# ML
set_cv(kind, **params)             # CVStrategy
set_metric(name)                   # CHECKPOINTED — changing scoring is a big deal
train_cv(fe="agent/fe.py", model="agent/model.py", subsample=None) -> CVResult
submit_local(label=None)           # full-data fit + test predict
submit_kaggle(message=None)        # CHECKPOINTED
read_my_submissions()

# Memory
take_note(category, content)       # observation / decision / hypothesis / todo
list_notes(category=None)
summarize_log(window="last_5_calls")  # context management

# Checkpoint gate
request_human_approval(action, summary, evidence) -> Decision
```

**Design rules:**
- The agent never touches Kaggle directly — `submit_kaggle` is the only path, and it's checkpointed.
- `train_cv` is the only path to a CV score; `run_python` cannot shortcut to fake leak-free.
- `run_python` is sandboxed to the comp workspace; useful for plotting, peeks, debugging.
- Notes are first-class — long-running comps have agent context drift; structured notes survive turns better than raw conversation.

## 8. Sandbox

Threat model: "agent typos itself into deleting another competition's folder," not "agent is adversarial." Three layers:

1. **CWD lock** — subprocess `cwd=competitions/<name>/`. Agent code addresses files relative to that.
2. **AST lint pass before execution** — rejects: `os.remove`, `shutil.rmtree`, `os.system`, arbitrary `subprocess.*`, `requests` to non-Kaggle hosts, `open(..., 'w')` outside the workspace. Implemented as AST walk, not regex.
3. **Resource limits** — `resource.setrlimit` for memory (≤80% of RAM) and CPU time per call; wall-clock timeout via `subprocess.run(timeout=...)`.

If this ever runs on someone else's machine or with an untrusted LLM provider, the threat model changes and Docker/gVisor would be needed. For local M5 use, AST lint is enough.

## 9. Checkpoints (mixed autonomy)

Harness blocks the agent until human responds at these points:

| Trigger | When | Default if non-interactive |
|---|---|---|
| First `submit_kaggle` of a comp | Always | Block |
| Subsequent `submit_kaggle` | Score regression vs best CV | Block on regression, auto-approve otherwise |
| `set_metric` override | Always | Block |
| Wall-clock budget hit (default 90 min) | Always | Block (with "extend by 30 min?" option) |
| Memory >90% sustained | Always | Block (with subsample suggestion) |
| Cost-budget hit | Always | Block (with "raise budget by $X?" option) |
| Agent-initiated `request_human_approval` | Always | Block |

UI: rich CLI prompt with action summary + evidence dict + `[y]es / [n]o / [a]bort / [s]kip-this-check`. Decisions journalled to `run_log.jsonl`. Non-interactive mode `--auto-approve=safe` auto-approves only the auto-approve cases above.

## 10. Storage & per-comp workspace

```
competitions/<name>/
├─ raw/                  Kaggle download (gitignored)
├─ context.md            auto-generated brief (problem, metric, data shape, LB context)
├─ agent/                ALL agent-written code
│   ├─ fe.py             current best feature engineering
│   ├─ model.py          current best model
│   ├─ versions/         fe_v01.py, model_v01.py, ... (every attempt kept)
│   └─ scratch/          one-off scripts via run_python
├─ artifacts/
│   ├─ pipeline.pkl      last full-data fit
│   ├─ oof_preds.npy     out-of-fold predictions
│   ├─ feature_importance.json
│   └─ cv_results.json
├─ submissions/
│   ├─ 2026-05-14_v01_cv0.842.csv
│   ├─ 2026-05-14_v02_cv0.851.csv
│   └─ leaderboard.jsonl     {csv → LB score, after submit_kaggle}
├─ notes.jsonl           agent's scratchpad
└─ run_log.jsonl         tool-call audit log
```

The per-comp workspace is the unit of memory. Resumable, debuggable, portfolio-ready.

## 11. Observability

### 11.1 MLflow

One experiment per competition. One **run** per `train_cv` call.

- **Params** — `cv_strategy`, `cv_n_splits`, `metric`, `fe_version`, `model_version`, `subsample`, `agent_decision_id`
- **Metrics** — `cv_mean`, `cv_std`, per-fold scores, `wall_clock_s`, `public_lb_score` (filled in after `submit_kaggle`)
- **Tags** — `problem_type`, `kaggle_competition`, `agent_iteration`
- **Artifacts** — `fe.py`, `model.py`, `oof_preds.npy`, `feature_importance.json`, stdout/stderr

### 11.2 Telemetry layer

| Layer | What | Why |
|---|---|---|
| Cost ledger | Per LLM call: input/output/cached tokens, USD cost, aggregated per tool-call/run/comp/day. Cap via `--cost-budget`. | LLM agents fail silently into the wallet. Non-negotiable. |
| OpenTelemetry → local file exporter | One trace per agent run. Tool calls = spans. LLM calls = nested spans with token counts. | Massive debugging leverage for the cost of one decorator. |
| CV↔LB calibration tracker | Every `submit_kaggle` writes `{cv_score, lb_score, problem_type, metric, cv_strategy}` to a global log. Dashboard shows regression and residuals. | The killer signal: tells us if the harness is trustworthy. |
| Agent behavior metrics | Counters: turns/run, turns_to_first_submission, turns_to_best_score, tool_call_failure_rate, stuck-loop detection. | Catches degenerate agents before they burn budget. |
| Health checks | Preflight: disk free, RAM available, Kaggle API reachable, Gemini API reachable, `kaggle.json` present. | Saves an hour of "why did this hang." |
| Error capture | Unhandled exception → `~/.kaggle_slayer/errors/<ts>.json` (traceback, last 20 tool calls, env). Rotation: keep last 100. | Local "Sentry" without infra. |
| Notification hook (optional) | `--notify=webhook_url` for checkpoint requests + run completion. | Walk-away-from-terminal during long comps. |

### 11.3 Failure-recovery semantics

- Tool-call exception → returned as `ToolError`. Agent decides retry.
- LLM call exception → exponential backoff (max 3). Surfaced as critical after.
- Sandbox kill (timeout, OOM, lint rejection) → returned as `SandboxError`. Agent learns and retries.
- Kaggle API failure during `submit_kaggle` → CSV intact, LB score marked `pending`, retryable.
- Harness internal crash → captured to error log, exit non-zero. `--resume` brings it back from last journalled tool call.

### 11.4 Dashboard

Streamlit, redesigned around the agent loop:

| Page | Content |
|---|---|
| Portfolio | Card per competition: best CV, best LB, status, gap from top public LB, last touched. Sortable. |
| Comp detail | Agent-decision timeline overlaid with CV + LB curves. Side-by-side diff of `fe_v01` ↔ `fe_v02`. Feature importance (latest). Notes browser filterable by category. Tool-call log search over `run_log.jsonl`. |
| Cross-comp *(nice-to-have)* | Strategy reuse patterns. CV↔LB correlation by comp type. |

### 11.5 Logging discipline

- `run_log.jsonl` — machine-readable audit, one tool call per line.
- Python `logging` (JSON formatter) — harness internals.
- `rich` — human-facing console output only. No print soup.

## 12. Resumability

Crucial property. Any run can be aborted (Ctrl-C, crash) and resumed.

- Every tool call appends to `run_log.jsonl` **before** returning to the agent.
- `kaggle-slayer <comp> --resume` rebuilds the conversation from `run_log.jsonl` (compressed via `summarize_log` once it crosses a token threshold), reloads `notes.jsonl`, and the current `agent/fe.py`/`agent/model.py` are on disk.
- Crashes don't lose progress past the last completed tool call.

## 13. Testing strategy

Three tiers, balancing realism and CI cost.

1. **Unit** — harness components (metric registry, CV registry, AST sandbox lint, file IO, MLflow logger). Fast. Every push.
2. **Integration with fake agent** — replaces the LLM with a scripted agent that issues canned tool-call sequences. Exercises full harness contracts (leak-free CV, checkpoint gate, resume) on synthetic data, **without** LLM API calls. The workhorse tier. Every push.
3. **E2E with real LLM** on micro-comps — three tiny synthetic competitions, each with a known optimal solution: tabular binary classification (LGBM beats DT baseline), regression with one obvious nonlinear feature (must be engineered), time-series (TimeSeriesSplit beats KFold by a known margin). Each runs <2 min. Marked `slow`; nightly when `GEMINI_API_KEY` is set, skipped otherwise.
4. **Chaos tier** — integration tier with random tool-call failures injected at 5% rate. Pipeline must survive without corrupting state. Run before declaring V1 done.

CI extensions on the existing `ci.yml`:
- Tighten ruff (drop `continue-on-error`).
- `mypy` on **harness only** (not agent-written code).
- Coverage via `pytest-cov`, target ≥ 80% on harness.
- Nightly schedule for E2E tier (secret-gated Gemini key).

## 14. Migration plan

### 14.1 Code disposition

**Delete (most of it):**
- `agents/` (data_scout, feature_engineer, model_selector, coordinator, base_agent)
- `core/` (data, features, models — replaced by agent-written code under the §6 contract + harness registries)
- `kaggle_slayer.py` (rewritten as a proper package entry)
- `tests/test_data.py`, `tests/test_models.py`, `tests/test_pipeline_e2e.py`

**Carry forward (infra and ideas):**
- `LICENSE`, `.gitignore`, `.pre-commit-config.yaml`
- `Dockerfile`, `docker-compose.yml` (refactored)
- `pyproject.toml` (heavily revised — add `google-generativeai`, `opentelemetry-api/sdk`, `jsonschema`)
- `utils/logging.py`, `utils/io.py`, `utils/config.py` (light refactor)
- `utils/kaggle_api.py` (significantly extended — add overview/eval/data-description/leaderboard)
- `utils/tracking.py` (refactor for new MLflow shape — one run per `train_cv`)
- `dashboard/app.py` (rewrite around new pages)
- `tests/conftest.py` (keep synthetic-fixture idea, rewrite)
- `competition_data/` directory idea, renamed to `competitions/`
- The leak-free-CV **idea**, re-implemented as the §6 contract
- `.github/workflows/ci.yml` (extend)

### 14.2 New top-level layout

```
kaggle_slayer/                 package root
├─ cli.py                      entry point
├─ harness/                    trusted side (no LLM)
│   ├─ context.py              Kaggle metadata enrichment + context.md builder
│   ├─ cv.py                   train_cv leak-free contract
│   ├─ registry/
│   │   ├─ metrics.py
│   │   └─ cv_strategies.py
│   ├─ sandbox.py              subprocess + AST lint + resource limits
│   ├─ kaggle_client.py        extended Kaggle API
│   ├─ workspace.py            per-comp folder, journalling
│   ├─ resume.py               rebuild conversation from run_log
│   └─ telemetry/
│       ├─ cost_ledger.py
│       ├─ otel.py
│       ├─ calibration.py      CV↔LB tracker
│       ├─ health.py
│       └─ errors.py
├─ agent/                      LLM side
│   ├─ solver.py               the loop
│   ├─ llm_client.py           Gemini wrapper, retry, cost tracking
│   ├─ prompts/{system,context_template}.md
│   └─ tools.py                JSON schemas (single source of truth)
├─ dashboard/app.py
└─ utils/{logging,config}.py
competitions/                  workspaces (per-comp)
tests/{unit,integration,e2e,chaos}/
docs/superpowers/specs/        this design doc
docs/adr/                      architecture decision records
docs/architecture.md           cross-cutting tech reference
legacy/                        old code, archived for reference
.claude/{commands,agents}/     Claude Code surfaces
```

### 14.3 Build order

| Week | Deliverable | Acceptance criterion |
|---|---|---|
| 1 — Foundations | `harness/cv.py`, `registry/`, `sandbox.py`, unit tests, new `pyproject.toml` + CI | Unit tier green; leak-free CV runs on a stub `fe.py`/`model.py` |
| 2 — Workspace + LLM | `workspace.py`, `kaggle_client.py` (extended), `context.py`, `llm_client.py`, integration tier with **fake agent** | Fake agent runs canned tool sequence on synthetic comp; harness journals correctly |
| 3 — Agent loop + tools | `solver.py`, `tools.py`, `prompts/`, first real Gemini call | Real Gemini agent solves simplest synthetic micro-comp end-to-end |
| 4 — Checkpoints + resume | Checkpoint gate UI, `resume.py`, hardened sandbox lint, `submit_kaggle` safety | Can abort mid-run and resume; checkpoint gate blocks the right calls |
| 5 — Observability | `telemetry/*`, redesigned dashboard pages, chaos-test mode | Cost ledger / OTel / CV-LB calibration in dashboard; chaos tier green |
| 6 — Polish + real comps | Run on 3 live Kaggle Playground comps, iterate, docs (README, CLAUDE.md, ADRs, slash commands, subagent) | ≥ 2 of 3 comps autonomously reach median public LB |

### 14.4 Repo strategy

Branch `v2-rebuild`. **First commit moves all existing source under `legacy/`** (a single `git mv` so blame history is preserved, no semantic change to existing files). The new layout is then built from scratch alongside `legacy/` — the carry-forward list in §14.1 is reference for *patterns and ideas*, not direct copy-paste; carried-forward files are rewritten or significantly refactored inside the new package layout. When V1 acceptance criteria (§18) are met, `v2-rebuild` merges to `main`. `legacy/` stays in `main` until V1 has been used on ~5 real comps; then deleted in a follow-up commit.

### 14.5 Implementation-plan structure

This spec is intentionally large (six build weeks). For the writing-plans phase, **each week is its own implementation plan**: six plans total, sequentially gated by their acceptance criteria. The first plan (week 1, foundations) is written immediately after spec approval; subsequent plans are written at the start of their week so they can incorporate learnings from the previous one. This avoids planning theatre and keeps each plan focused enough to execute well.

## 15. Documentation deliverables

Researched against Claude Code best practices: keep `CLAUDE.md` ≤ 200 lines and scoped to non-derivable knowledge; ADRs terse; slash commands and subagents only where they solve real friction.

| File | Purpose |
|---|---|
| `CLAUDE.md` | Project rules. Encodes the leak-free contract as a hard rule, points to deeper docs. |
| `README.md` | Human-facing description. Replaces the legacy README. |
| `docs/superpowers/specs/2026-05-14-llm-agent-harness-design.md` | This spec. |
| `docs/architecture.md` | Cross-cutting tech reference (sandbox lint patterns, OTel schema, tool JSON schemas). |
| `docs/adr/0001-single-agent.md` | Why Approach A (single-agent tool-use loop). |
| `docs/adr/0002-gemini-pro.md` | Why Gemini Pro, not Claude. |
| `docs/adr/0003-leak-free-contract.md` | Why agent-writes-code + harness-runs-CV. |
| `.claude/commands/run-comp.md` | `/run-comp <name>` — bundles preflight + start. |
| `.claude/commands/resume-comp.md` | `/resume-comp <name>` — resume from journal. |
| `.claude/agents/harness-reviewer.md` | Subagent invoked on harness changes; verifies the leak-free contract. |

**Deliberately skipped:** `AGENTS.md` (CLAUDE.md is enough), `.claude/skills/` (over-engineering for V1), `docs/runbook.md` and `docs/contributing.md` (write when needed), issue/PR templates, `/dash` slash command.

## 16. Risks and mitigations

| Risk | Likelihood | Mitigation |
|---|---|---|
| Gemini tool-use reliability lags Claude on long multi-step loops | Medium | Schema-strict tools, retry on malformed calls, use Gemini's structured-output mode where available. Re-evaluate at end of week 3. |
| Synthetic E2E micro-comps don't reflect real Kaggle weirdness | High | Week 6 mandates ≥ 2 real Playground comps. Plan to surface unanticipated failures and treat as V1.1. |
| AST sandbox lint is permissive | Low (current threat model) | Documented; revisit if threat model changes (cloud, untrusted LLM). |
| Context bloat on long runs | Medium | Aggressive `summarize_log` once past ~600 k tokens; Gemini 1M context buys headroom. |
| Agent gets stuck in retry loops | Medium | Stuck-loop detector (same tool ≥ 5× with similar args in 10-turn window) flags and breaks the loop. |
| Cost runaway | Medium | `--cost-budget` flag with default $5/comp, checkpoint when hit. |
| CV doesn't predict LB | High in some comp types | CV↔LB calibration tracker surfaces it. If a comp type is reliably miscalibrated, agent is prompted to be more conservative. |

## 17. Out-of-scope, parking lot

Items considered and explicitly deferred:
- Multi-agent (planner + executor + critic) — revisit if single-agent hits a quality ceiling.
- NLP, CV, audio tracks — Phase 2/3, separate specs.
- Cloud-burst execution — `--compute=cloud` flag designed in but not implemented.
- AGENTS.md / cross-tool LLM-agent compat layer.
- Slash command `/dash`, skills, contributing/runbook docs.
- Kaggle API rate-limit handling beyond simple retry.
- Multi-target / multi-label classification — out of V1 scope; can be added under §6 contract without rearchitecture.

## 18. Acceptance criteria for V1

- Unit + integration tiers green; chaos tier green.
- E2E tier passes on all three synthetic micro-comps.
- Run on 3 live Kaggle Playground competitions, ≥ 2 reach median public LB autonomously.
- Dashboard surfaces: cost ledger, CV↔LB calibration, per-comp timeline, agent notes.
- All Section 15 documentation files exist and reference each other correctly.
- `CLAUDE.md` is ≤ 200 lines and accurately reflects the new architecture.
- `legacy/` archive removed from `main` after acceptance.
