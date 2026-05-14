# KaggleSlayer

[![CI](https://github.com/rodme02/KaggleSlayer/actions/workflows/ci.yml/badge.svg)](https://github.com/rodme02/KaggleSlayer/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)
[![Python 3.11+](https://img.shields.io/badge/python-3.11%2B-blue.svg)](pyproject.toml)

> **An LLM-agent harness for tabular Kaggle competitions. Under construction.**

KaggleSlayer V2 is being rebuilt from scratch as an LLM-driven Kaggle harness: an agent reads the competition, plans a strategy, writes feature-engineering and model code, and the harness runs leak-free cross-validation on it. The agent owns creativity; the harness owns correctness.

The full design is in [`docs/superpowers/specs/2026-05-14-llm-agent-harness-design.md`](docs/superpowers/specs/2026-05-14-llm-agent-harness-design.md). Weekly implementation plans live in [`docs/superpowers/plans/`](docs/superpowers/plans/).

## Status — Week 1 of 6

The **trusted harness** (the parts that cannot be left to LLM judgment) is shipped:

- ✅ Leak-free CV contract (`kaggle_slayer/harness/cv.py`)
- ✅ Metric registry — accuracy, AUC, logloss, RMSE, MAE, R²
- ✅ CV strategy registry — KFold + StratifiedKFold (+ auto-select by problem type)
- ✅ AST sandbox lint for agent-written modules
- ✅ 62 tests, 95% coverage on the harness, ruff + mypy strict green
- ⏳ LLM integration, Kaggle API extension, workspace journalling — Week 2
- ⏳ Agent loop, tool JSON schemas, checkpoints, resume — Weeks 3-4
- ⏳ Telemetry + dashboard + first real Kaggle run — Weeks 5-6

## What "leak-free CV" means here

The agent writes `fe.py` (a `fit_feature_transformer`) and `model.py` (a `fit_model`). The harness — not the agent — runs CV: each fold is iterated by the harness, and `fit_feature_transformer` is called with **only the train-fold data**. Anti-cheat: row count must be preserved through `.transform()`; agent code is AST-linted against direct filesystem reads and network calls before it ever loads.

This is the *temporal* version of leak-free CV. V1 used a sklearn `TransformerMixin` (a *structural* guarantee), which was easy to subvert by accident — and did, in three places, before the rebuild.

## Quickstart (today)

```bash
git clone https://github.com/rodme02/KaggleSlayer.git
cd KaggleSlayer
pip install -e ".[dev,dashboard]"
pytest -m "not slow"
```

There is no CLI yet (lands in Week 3). The harness is library-shaped and exercised through the test suite. See [`docs/superpowers/plans/2026-05-14-week1-foundations.md`](docs/superpowers/plans/2026-05-14-week1-foundations.md) for what's in this milestone.

## Stack

| Choice | Why |
| --- | --- |
| **Google Gemini Pro** (planned) | Single LLM, cheap, behind a thin `LLMClient` abstraction so swapping later costs nothing. |
| **Python 3.11+ + scikit-learn** | The harness wraps sklearn splitters and metrics; agent code is free to import whatever it wants, subject to the AST lint. |
| **MLflow + Streamlit** (planned) | Per-comp run tracking + per-comp dashboard pages. Same `mlruns/` store across the project. |
| **AST sandbox lint** | Local M5 use; threat model is "agent typos itself," not adversarial. Subprocess sandbox upgrade documented for later. |

## License

MIT — see [`LICENSE`](LICENSE).
