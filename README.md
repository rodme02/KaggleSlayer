# KaggleSlayer

[![CI](https://github.com/rodme02/KaggleSlayer/actions/workflows/ci.yml/badge.svg)](https://github.com/rodme02/KaggleSlayer/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)
[![Python 3.11+](https://img.shields.io/badge/python-3.11%2B-blue.svg)](pyproject.toml)

> **An LLM-agent harness for tabular Kaggle competitions. The agent owns creativity; the harness owns correctness.**

KaggleSlayer drops a Gemini-driven agent into a sandboxed workspace, hands it a competition brief, and lets it iterate on `fe.py` and `model.py` until it's ready to submit. The harness — the parts that cannot be left to LLM judgment — owns leak-free cross-validation, the tool surface, the checkpoint gate on Kaggle submissions, and the journal that makes any run resumable after a crash or abort.

Full design lives in [`docs/superpowers/specs/2026-05-14-llm-agent-harness-design.md`](docs/superpowers/specs/2026-05-14-llm-agent-harness-design.md). Per-week implementation plans live in [`docs/superpowers/plans/`](docs/superpowers/plans/).

## Status — Week 4 of 6

End-to-end runnable. On the latest validation, real `gemini-2.5-flash` solved a synthetic binary-classification micro-comp in **6 iterations, 10 seconds, $0.0013**, writing both `agent/fe.py` and `agent/model.py`, running leak-free CV, producing a submission CSV, and routing the (mocked) Kaggle push through the checkpoint gate.

What's shipped:

- ✅ **Leak-free CV contract** (`harness/cv.py`) — harness owns the splits, agent's `fit_feature_transformer` only ever sees one fold's train data
- ✅ **Trusted harness** — metric registry (6 metrics with `higher_is_better` direction), CV-strategy registry (KFold / StratifiedKFold / TimeSeriesSplit / GroupKFold with auto-select), AST lint, resource-limited subprocess sandbox (RLIMIT_AS/CPU/NPROC/FSIZE on POSIX)
- ✅ **Per-comp workspace + journal** — `competitions/<name>/raw/`, `agent/`, `agent/versions/`, `agent/scratch/`, `submissions/`, `run_log.jsonl`, `notes.jsonl`, `submissions/leaderboard.jsonl`
- ✅ **GeminiClient** (`agent/llm_client.py`) — structured `Content`/`Part` multi-turn with `function_call`/`function_response` round-trips, automatic schema sanitization for Gemini's OpenAPI subset, retry with exponential backoff, status-code-aware transient classification
- ✅ **Solver loop + 13 tools** — `read_context`, `read_file`, `write_file`, `sample_rows`, `take_note`, `set_cv`, `train_cv`, `submit_local`, `done`, `run_python`, `set_metric`, `submit_kaggle`, `request_human_approval`
- ✅ **Checkpoint gate** — typed `CheckpointTrigger` enum, four handler modes (interactive / auto-safe / stub / callable), journalled decisions; covers first submit, regression submit, set_metric, wall-clock budget, cost budget, memory pressure, agent-initiated
- ✅ **Resume** — `kaggle-slayer ... --resume` rebuilds the conversation from `run_log.jsonl` and seeds the next solve
- ✅ **CLI** — `kaggle-slayer <workspace> --target <col>` does the full thing
- ✅ **316 unit + integration tests**, ruff + mypy strict on `harness/` and `agent/`, ~95% coverage on new code

What's next:

- ⏳ **Week 5** — telemetry surfaced (cost-ledger / OpenTelemetry / CV↔LB calibration), chaos-tier tests, redesigned Streamlit dashboard
- ⏳ **Week 6** — three real Kaggle Playground comps, full docs (`docs/architecture.md`, ADRs, `.claude/` commands and subagents)

## Why "leak-free CV" is the headline

The agent writes `fe.py` (a `fit_feature_transformer(train_df, target_col)`) and `model.py` (a `fit_model(X_train, y_train, problem_type, metric_name)`). The harness — not the agent — runs CV: each fold is iterated, and `fit_feature_transformer` is called with **only the train-fold data**. Anti-cheat is layered:

1. The harness physically only passes the train fold during fit. No other entry point exists.
2. `fe.transform(df).shape[0] == df.shape[0]` is verified (no row dropping that would change splits).
3. AST lint forbids the agent's code from reading `raw/` paths or hitting the network. The lint runs before module load; rejected files never execute.
4. Code passed to `run_python` runs in a resource-limited subprocess (memory, CPU, process count, file size, network blocked at the lint).

This is the *temporal* version of leak-free CV. V1 used a sklearn `TransformerMixin` (a *structural* guarantee), which was easy to subvert by accident — and did, in three places, before the rebuild.

## Quickstart

```bash
git clone https://github.com/rodme02/KaggleSlayer.git
cd KaggleSlayer
pip install -e ".[dev,dashboard]"
pytest -m "not slow"                                # ~5s, 316 tests
```

To run against a real competition you need a Gemini API key (Tier 1 billing recommended — Tier 0 free tier has 0 daily quota for `gemini-2.5-pro` and only 20/day for `gemini-2.5-flash`) and Kaggle API credentials:

```bash
# .env at the repo root
GEMINI_API_KEY=...
KAGGLE_API_TOKEN=KGAT_...          # or the legacy KAGGLE_USERNAME + KAGGLE_KEY

python scripts/preflight.py        # verifies both credentials work

# competitions/<name>/raw/ should contain Kaggle's train.csv + test.csv
kaggle-slayer competitions/titanic --target Survived --metric accuracy
```

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

The agent's running history lives in `competitions/<name>/run_log.jsonl` and its scratchpad in `notes.jsonl`. After each `train_cv` the current `agent/fe.py` and `agent/model.py` are archived under `agent/versions/`.

## Stack

| Choice | Why |
| --- | --- |
| **Google Gemini Flash (default), Pro on opt-in** | Single LLM behind a thin `LLMClient` Protocol. Flash has real Tier 0 quota and is ~10× cheaper; Pro is one flag away when budget allows. Swapping providers is a single file. |
| **Python 3.11+ + scikit-learn** | The harness wraps sklearn splitters and metrics; agent code is free to import whatever it wants subject to the AST lint and the resource-limited subprocess. |
| **JSON-Schema for tool args** | `jsonschema.validate` runs before the handler, so a malformed call returns a typed `ToolError` the agent can self-correct on. Schemas are sanitized for Gemini's OpenAPI subset on the wire. |
| **AST sandbox lint + RLIMIT subprocess** | Local M5 use; threat model is "agent typos itself," not adversarial. The lint catches `os.remove`, `subprocess`, `requests`, network, raw/* reads, symlinks. The subprocess sandbox caps memory, CPU, process count, and file size. |
| **MLflow + Streamlit** (Week 5) | Per-comp run tracking + per-comp dashboard pages. |

## Repo layout

```
kaggle_slayer/
├─ cli.py                      # kaggle-slayer entry point
├─ harness/                    # trusted, no LLM — owns the contracts
│   ├─ cv.py                   # train_cv leak-free contract
│   ├─ workspace.py            # per-comp directory dataclass
│   ├─ journal.py              # durable run_log.jsonl + notes.jsonl
│   ├─ resume.py               # rebuild_conversation from journal
│   ├─ checkpoints.py          # typed gate, journalled decisions
│   ├─ context.py              # context.md builder
│   ├─ kaggle_client.py        # extended Kaggle API
│   ├─ sandbox.py              # AST lint + run_subprocess
│   └─ registry/               # metrics + CV strategies
└─ agent/                      # LLM side
    ├─ solver.py               # reason-act loop
    ├─ llm_client.py           # GeminiClient + LLMClient Protocol
    ├─ tools.py                # Tool / ToolRegistry / ToolError
    ├─ cost_ledger.py          # per-call USD ledger
    ├─ handlers/               # files.py + ml.py + python.py
    └─ prompts/system.md       # Solver system prompt

tests/{unit,integration,fixtures}/
docs/superpowers/{specs,plans}/
scripts/preflight.py            # verify Gemini + Kaggle creds
```

## Status of testing

- **Unit tier** — `pytest -m "not slow"`. ~316 tests. Runs on every push. Linux 3.11 + 3.12 matrix in CI.
- **Integration tier** — fake-LLM-driven scripted runs against a synthetic micro-comp (`tests/fixtures/synthetic_comp.py`). Also runs in CI under `-m integration` (selected automatically by file location).
- **Slow tier (opt-in)** — real Gemini calls. `pytest -m slow`. 8 tests; ~$0.001–0.05 per run; skipped automatically when `GEMINI_API_KEY` is missing. Not part of CI.

## Honest limitations

- Phase-1 scope is tabular only — binary, multi-class, regression, time-series tabular. NLP / CV / audio are deferred to later phases.
- The sandbox is "best effort." Threat model is non-adversarial: AST lint catches typos; subprocess rlimits cap memory/CPU. Truly adversarial code or an untrusted LLM provider would need Docker/gVisor.
- macOS rejects `RLIMIT_AS` with "current limit exceeds maximum limit" — the cap is best-effort on Darwin, hard on Linux. CPU cap is enforced on both (with a 2s buffer above the wall-clock timeout).
- CV↔leaderboard correlation tracking lands in Week 5. Until then, treat CV-on-real-comps with some skepticism.
- No Phase-2/3 features (multi-agent, cloud-burst, NLP track) are wired in.

## License

MIT — see [`LICENSE`](LICENSE).
