# CLAUDE.md

Guidance for Claude Code when working in this repo.

## What this is

KaggleSlayer V2 — an LLM-agent harness for tabular Kaggle competitions, under construction. The full design is in `docs/superpowers/specs/2026-05-14-llm-agent-harness-design.md`; the active sub-plan is in `docs/superpowers/plans/`.

V1 (a plain AutoML pipeline) was archived and removed. The leak-free CV idea was the only thing carried forward.

**Current state:** Week 1 (Foundations) complete. The trusted harness ships: `harness/cv.py`, `harness/registry/{metrics,cv_strategies}.py`, `harness/sandbox.py`. No LLM integration yet (that's Week 2-3).

## Hard rules

1. **Leak-free CV is the inviolable contract.** `train_cv` only ever passes one fold's training data to the agent's `fit_feature_transformer`. Never add a code path that hands the full dataset (or any val/test data) to agent-written code during fit. See `kaggle_slayer/harness/cv.py` and spec §6.
2. **Agent code is never trusted at import time.** Any file that will be loaded as agent-written (`fe.py`, `model.py`, future `*.py` under `competitions/<name>/agent/`) MUST pass `sandbox.lint_module()` first. Don't bypass.
3. **The harness owns metrics and CV strategies.** Adding a metric or splitter goes in `kaggle_slayer/harness/registry/`, not inline at the call site. The registry has `.kind` and `.needs_proba` metadata the contract depends on.

## Layout

```
kaggle_slayer/
├─ __init__.py
└─ harness/                    # trusted, no LLM
    ├─ cv.py                   # train_cv leak-free contract
    ├─ sandbox.py              # AST lint
    └─ registry/
        ├─ metrics.py          # 6 metrics: accuracy, auc, logloss, rmse, mae, r2
        └─ cv_strategies.py    # KFold + StratifiedKFold + auto_select
tests/
├─ unit/                       # harness-level tests
├─ integration/                # CV with stub agent code
├─ fixtures/                   # canonical fe_stub.py + model_stub.py
└─ conftest.py                 # synthetic binary/regression/time-series data
docs/superpowers/
├─ specs/                      # the V2 design
└─ plans/                      # weekly implementation plans
```

## Common commands

```bash
pip install -e ".[dev,dashboard]"
pytest -m "not slow"
pytest --cov=kaggle_slayer/harness --cov-report=term-missing -m "not slow"
ruff check kaggle_slayer tests
mypy kaggle_slayer/harness          # strict on harness only; tests/ + future agent code are not type-checked
```

## Conventions

- Type-hint public function signatures in `kaggle_slayer/harness/`. Mypy is strict here.
- TDD: write the failing test first, then the implementation. Any commit on the harness shows the pattern.
- Errors at boundaries (file I/O, future Kaggle API, future LLM client) get caught and surfaced as a typed exception. The harness raises `CVError` from `cv.py`; future modules raise their own.
- CLI output: `rich`. Structured output: stdlib `logging` with a JSON formatter (added in Week 5). No `print` in library code.
- Don't add metrics/CV strategies inline — extend the registry.
- Don't add features that aren't in the active plan. The spec is large; scope is enforced week by week.

## What's coming (do not pre-build)

- **Week 2** — workspace journalling (`competitions/<name>/`, `run_log.jsonl`), extended Kaggle API client, `context.md` builder, `LLMClient` Gemini wrapper, integration tier with a fake agent.
- **Week 3** — agent loop (`solver.py`), tool JSON schemas, first real Gemini call.
- **Week 4** — checkpoint gate, resume, hardened sandbox.
- **Week 5** — telemetry (cost ledger, OTel, CV↔LB calibration), redesigned Streamlit dashboard.
- **Week 6** — real Kaggle comps + docs (`docs/architecture.md`, ADRs, slash commands, harness-reviewer subagent).
