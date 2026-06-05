# KaggleSlayer

[![CI](https://github.com/rodme02/KaggleSlayer/actions/workflows/ci.yml/badge.svg)](https://github.com/rodme02/KaggleSlayer/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)
[![Python 3.11+](https://img.shields.io/badge/python-3.11%2B-blue.svg)](pyproject.toml)

> **An LLM-agent harness for tabular Kaggle competitions. The agent owns creativity; the harness owns correctness.**

KaggleSlayer drops a Gemini-driven agent into a sandboxed workspace, hands it a competition brief, and lets it iterate on `fe.py` and `model.py` until it's ready to submit. The harness — the parts that cannot be left to LLM judgment — owns leak-free cross-validation, the tool surface, the checkpoint gate on Kaggle submissions, and the journal that makes any run resumable after a crash or abort.

<!--
Hero media is pending capture (a terminal solve and/or the Streamlit dashboard).
See docs/media/README.md — drop the file in at docs/media/hero.gif and uncomment the
line below and it renders with no further edits.
-->
<!-- ![KaggleSlayer in action](docs/media/hero.gif) -->

> **Hero media:** pending — see [`docs/media/README.md`](docs/media/README.md).

Full design lives in [`docs/superpowers/specs/2026-05-14-llm-agent-harness-design.md`](docs/superpowers/specs/2026-05-14-llm-agent-harness-design.md). Per-week implementation plans live in [`docs/superpowers/plans/`](docs/superpowers/plans/). Scope, status, and roadmap live in one place: [`GOALS.md`](GOALS.md).

## Status

End-to-end runnable. A clean clone runs the full test suite — **375 tests, ~5s, no API keys** — which is exactly what CI enforces (Linux 3.11 + 3.12). A *real* solve needs a Gemini API key plus Kaggle credentials; on the latest validation, real `gemini-2.5-flash` solved a synthetic binary-classification micro-comp in **6 iterations, 10 seconds, $0.0013**, writing both `agent/fe.py` and `agent/model.py`, running leak-free CV, producing a submission CSV, routing the (mocked) Kaggle push through the checkpoint gate — and emitting an OpenTelemetry trace, an MLflow run per `train_cv`, a CV↔LB calibration row, and a cost-ledger row along the way.

> **Headline v1 goal:** a credential-free demo so a clean clone runs the agent loop end-to-end with **no keys** (fake-LLM + synthetic-comp path). This is **not wired yet** — until it is, a real solve requires Gemini. See [`GOALS.md`](GOALS.md) for the full v1 scope boundary.

What's shipped:

- ✅ **Leak-free CV contract** (`harness/cv.py`) — harness owns the splits, agent's `fit_feature_transformer` only ever sees one fold's train data
- ✅ **Trusted harness** — metric registry (6 metrics with `higher_is_better` direction), CV-strategy registry (KFold / StratifiedKFold / TimeSeriesSplit / GroupKFold with auto-select), AST lint, resource-limited subprocess sandbox (RLIMIT_AS/CPU/NPROC/FSIZE on POSIX)
- ✅ **Per-comp workspace + journal** — `competitions/<name>/raw/`, `agent/`, `agent/versions/`, `agent/scratch/`, `submissions/`, `run_log.jsonl`, `notes.jsonl`, `submissions/leaderboard.jsonl`, `otel.jsonl`
- ✅ **GeminiClient** (`agent/llm_client.py`) — structured `Content`/`Part` multi-turn with `function_call`/`function_response` round-trips, automatic schema sanitization for Gemini's OpenAPI subset, retry with exponential backoff, status-code-aware transient classification
- ✅ **RetryingLLMClient** (`agent/retrying_client.py`) — portable LLMClient adapter with TransientLLMError retry + exponential backoff (chaos-tier exercises this; GeminiClient retains its own internal retry)
- ✅ **Solver loop + 13 tools** — `read_context`, `read_file`, `write_file`, `sample_rows`, `take_note`, `set_cv`, `train_cv`, `submit_local`, `done`, `run_python`, `set_metric`, `submit_kaggle`, `request_human_approval`
- ✅ **Checkpoint gate** — typed `CheckpointTrigger` enum, four handler modes (interactive / auto-safe / stub / callable), journalled decisions; covers first submit, regression submit, set_metric, wall-clock budget, cost budget, memory pressure, agent-initiated
- ✅ **Resume** — `kaggle-slayer ... --resume` rebuilds the conversation from `run_log.jsonl` and seeds the next solve
- ✅ **CLI** — `kaggle-slayer <workspace> --target <col>` does the full thing; unhandled exceptions land as JSON crash reports under `~/.kaggle_slayer/errors/` (with env-var redaction + 100-file rotation) and exit code 4
- ✅ **OpenTelemetry tracing** — `<workspace>/otel.jsonl` per run, one span for the loop + one per LLM call + one per tool dispatch (`harness/telemetry/otel.py`)
- ✅ **CV↔LB calibration tracker** — every successful `submit_kaggle` appends to `~/.kaggle_slayer/calibration.jsonl` with cv_score (of *this* submission, not best-ever), problem_type, metric, cv_strategy; lb_score backfill is Week 6
- ✅ **MLflow logging** — one experiment per competition (`kaggleslayer/<comp>`), one run per `train_cv` call, with tags (`problem_type`, `kaggle_competition`), params (`cv_strategy`, `metric`, `fe_version`, `model_version`), metrics (`cv_mean`, `cv_std`, `fold_N`, `wall_clock_s`). Failures route to `<workspace>/mlflow_errors.log` and never crash the agent
- ✅ **Agent behavior metrics** — `turns_per_run`, `turns_to_first_submission`, `turns_to_best_score`, `tool_call_failure_rate`, stuck-loop detector (consolidated from `resume.py`)
- ✅ **Streamlit dashboard** — `kaggle-slayer-dashboard` console_script: portfolio page (list comps + best CV + cost + tool count) + comp-detail page (journal timeline + cost + calibration + behavior metrics + notes + submission CSV downloads). Read-only over disk
- ✅ **Chaos tier** — `FailureInjectingLLMClient` fixture (seeded, configurable rate) + integration test asserting `result.status == "done"` deterministically under 5% transient injection
- ✅ **375 unit + integration + chaos tests** (pass with no keys). ruff clean; mypy strict on `harness/` locally and `agent/` too. **CI type-checks the harness only** (`mypy kaggle_slayer/harness`) — see [`CLAUDE.md`](CLAUDE.md). ~95% coverage on new code.

What's next (deferred to the post-v1 roadmap — see [`GOALS.md`](GOALS.md)):

- The **credential-free demo** (the headline v1 goal): expose `tests/fixtures/synthetic_comp.py` + a fake-LLM path so a clean clone solves a comp with no keys.
- Live leaderboard / benchmark across real Kaggle Playground comps, full docs (`docs/architecture.md`, ADRs, `.claude/` commands and subagents), MLflow artifact logging (fe.py / model.py / oof_preds), LB-score backfill into the calibration log, the fe_v01↔fe_v02 dashboard diff page, and the cross-comp dashboard page.

## Why "leak-free CV" is the headline

The agent writes `fe.py` (a `fit_feature_transformer(train_df, target_col)`) and `model.py` (a `fit_model(X_train, y_train, problem_type, metric_name)`). The harness — not the agent — runs CV: each fold is iterated, and `fit_feature_transformer` is called with **only the train-fold data**. Anti-cheat is layered:

1. The harness physically only passes the train fold during fit. No other entry point exists.
2. `fe.transform(df).shape[0] == df.shape[0]` is verified (no row dropping that would change splits).
3. AST lint forbids the agent's code from reading `raw/` paths or hitting the network. The lint runs before module load; rejected files never execute.
4. Code passed to `run_python` runs in a resource-limited subprocess (memory, CPU, process count, file size, network blocked at the lint).

This is the *temporal* version of leak-free CV. V1 used a sklearn `TransformerMixin` (a *structural* guarantee), which was easy to subvert by accident — and did, in three places, before the rebuild.

## Quickstart

No API keys are needed to clone, install, and run the full test suite:

```bash
git clone https://github.com/rodme02/KaggleSlayer.git
cd KaggleSlayer
pip install -e ".[dev,dashboard]"
pytest -m "not slow"                                # ~5s, 375 tests, no keys
```

To run against a real competition you need a Gemini API key (Tier 1 billing recommended — Tier 0 free tier has 0 daily quota for `gemini-2.5-pro` and only 20/day for `gemini-2.5-flash`) and Kaggle API credentials. Copy [`.env.example`](.env.example) to `.env` and fill it in:

```bash
cp .env.example .env               # then fill in GEMINI_API_KEY + KAGGLE_API_TOKEN
python scripts/preflight.py        # verifies both credentials work

# kaggle-slayer auto-downloads competition data into raw/ on first run
# (the slug is the workspace dir name; pass --no-download to use your own data)
kaggle-slayer competitions/titanic --target Survived --metric accuracy
```

> Per-competition workspaces under `competitions/` are gitignored — a clean clone starts empty and you point the CLI at one you set up. A **no-key demo** that skips this setup entirely is the headline v1 goal (see [`GOALS.md`](GOALS.md)); it is not wired yet.

Useful flags:

| Flag | What it does |
| --- | --- |
| `--target <col>` | Target column name (required for non-trivial runs) |
| `--metric {accuracy,auc,logloss,rmse,mae,r2}` | Scoring metric. Defaults to `accuracy`. |
| `--problem-type {classification,regression}` | Defaults to `classification`. |
| `--max-iterations N` | Solver iteration cap (default 25). |
| `--time-budget-s S` | Wall-clock cap; triggers a checkpoint, not a hard exit (default 900). |
| `--cost-budget USD` | Cost ledger cap; APPROVE at the gate doubles the budget. |
| `--model <id>` | Gemini model id. Defaults to `gemini-2.5-flash`. |
| `--auto-approve {off,safe,all}` | Checkpoint mode. `all` requires `--i-know-what-im-doing`. |
| `--resume` | Replay `run_log.jsonl` to seed a continuation. Auto-skips context rebuild. |
| `--rebuild-context` | Force `context.md` regeneration even when resuming. |
| `--no-context-build` | Skip context rebuild entirely (manual `context.md`). |
| `--no-download` | Skip auto-downloading competition data into `raw/` (use data you placed there yourself). |
| `--competition <slug>` | Kaggle competition slug to download. Defaults to the workspace directory name. |

The agent's running history lives in `competitions/<name>/run_log.jsonl` and its scratchpad in `notes.jsonl`. After each `train_cv` the current `agent/fe.py` and `agent/model.py` are archived under `agent/versions/`.

## Stack

| Choice | Why |
| --- | --- |
| **Google Gemini Flash (default), Pro on opt-in** | Single LLM behind a thin `LLMClient` Protocol. Flash has real Tier 0 quota and is ~10× cheaper; Pro is one flag away when budget allows. Swapping providers is a single file. |
| **Python 3.11+ + scikit-learn** | The harness wraps sklearn splitters and metrics; agent code is free to import whatever it wants subject to the AST lint and the resource-limited subprocess. |
| **JSON-Schema for tool args** | `jsonschema.validate` runs before the handler, so a malformed call returns a typed `ToolError` the agent can self-correct on. Schemas are sanitized for Gemini's OpenAPI subset on the wire. |
| **AST sandbox lint + RLIMIT subprocess** | Local M5 use; threat model is "agent typos itself," not adversarial. The lint catches `os.remove`, `subprocess`, `requests`, network, raw/* reads, symlinks. The subprocess sandbox caps memory, CPU, process count, and file size. |
| **MLflow + Streamlit** | Per-comp run tracking (one experiment per comp, one run per `train_cv`) + per-comp dashboard pages (`kaggle-slayer-dashboard`). |
| **Custom OTel JSONL exporter** | We use a thin in-tree tracer instead of `opentelemetry-sdk` — saves 1 MB+ of transitive deps; one swap to OTLP if/when we need it. |

## Repo layout

```
kaggle_slayer/
├─ cli.py                      # kaggle-slayer entry point; run() wraps _run_inner with error capture
├─ harness/                    # trusted, no LLM — owns the contracts
│   ├─ cv.py                   # train_cv leak-free contract
│   ├─ workspace.py            # per-comp directory dataclass
│   ├─ journal.py              # durable run_log.jsonl + notes.jsonl
│   ├─ resume.py               # rebuild_conversation from journal (stuck-loop delegated to telemetry.behavior)
│   ├─ checkpoints.py          # typed gate, journalled decisions
│   ├─ context.py              # context.md builder
│   ├─ kaggle_client.py        # extended Kaggle API
│   ├─ sandbox.py              # AST lint + run_subprocess
│   ├─ registry/               # metrics + CV strategies
│   └─ telemetry/              # OTel tracer, calibration, error capture, behavior metrics, MLflow logger
├─ agent/                      # LLM side
│   ├─ solver.py               # reason-act loop + SolverContext (best/last cv) + OTel spans
│   ├─ llm_client.py           # GeminiClient + LLMClient Protocol + TransientLLMError
│   ├─ retrying_client.py      # LLMClient adapter for portable transient retry
│   ├─ tools.py                # Tool / ToolRegistry / ToolError
│   ├─ cost_ledger.py          # per-call USD ledger
│   ├─ handlers/               # files.py + ml.py (incl. MLflow + calibration hooks) + python.py
│   └─ prompts/system.md       # Solver system prompt
└─ dashboard/                  # Streamlit, read-only over disk
    ├─ app.py                  # kaggle-slayer-dashboard entry; routes Portfolio / Competition detail
    ├─ portfolio.py            # list comps + best CV + cost + tool count
    └─ comp_detail.py          # timeline + cost + calibration + notes + submissions + behavior metrics

tests/{unit,integration,chaos,fixtures}/
docs/superpowers/{specs,plans}/
scripts/preflight.py            # verify Gemini + Kaggle creds
```

## Status of testing

- **Unit tier** — `pytest -m "not slow"`. 375 tests (1 environment-gated skip). Runs on every push. Linux 3.11 + 3.12 matrix in CI.
- **Integration tier** — fake-LLM-driven scripted runs against a synthetic micro-comp (`tests/fixtures/synthetic_comp.py`). Also runs in CI under `-m integration` (selected automatically by file location).
- **Chaos tier** — `pytest -m chaos`. Scripted Solver run wrapped in `FailureInjectingLLMClient` (5% transient injection, seeded) + `RetryingLLMClient` adapter. Verifies spec §11.3 / §13: the pipeline reaches `done` deterministically and the journal stays parseable. Runs in CI under the default `-m "not slow"` invocation.
- **Slow tier (opt-in)** — real Gemini calls. `pytest -m slow`. 8 tests; ~$0.005–0.02 per run; skipped automatically when `GEMINI_API_KEY` is missing. Not part of CI.

## Honest limitations

- Phase-1 scope is tabular only — binary, multi-class, regression, time-series tabular. NLP / CV / audio are deferred to later phases.
- The sandbox is "best effort." Threat model is non-adversarial: AST lint catches typos; subprocess rlimits cap memory/CPU. Truly adversarial code or an untrusted LLM provider would need Docker/gVisor.
- macOS rejects `RLIMIT_AS` with "current limit exceeds maximum limit" — the cap is best-effort on Darwin, hard on Linux. CPU cap is enforced on both (with a 2s buffer above the wall-clock timeout).
- CV↔leaderboard tracker writes the *CV side* on every successful `submit_kaggle`; the *LB side* (`lb_score`) is left null pending the post-v1 Kaggle-leaderboard backfill (see [`GOALS.md`](GOALS.md)). Treat early calibration history as one-sided until then.
- MLflow tracking defaults to a file store at `~/.kaggle_slayer/mlruns` (overridable via `MLFLOW_TRACKING_URI`). Artifact logging (fe.py / model.py / oof_preds.npy) is post-v1 roadmap (see [`GOALS.md`](GOALS.md)); only params + metrics + tags land today.
- The OTel exporter is a custom JSONL writer (single-process, single-threaded). Concurrent writers against the same workspace would interleave records; the Solver is serial by design, so this is fine in practice.
- No Phase-2/3 features (multi-agent, cloud-burst, NLP track) are wired in.

## License

MIT — see [`LICENSE`](LICENSE).
