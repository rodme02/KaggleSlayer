# KaggleSlayer

[![CI](https://github.com/rodme02/KaggleSlayer/actions/workflows/ci.yml/badge.svg)](https://github.com/rodme02/KaggleSlayer/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)
[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](pyproject.toml)

> **AutoML pipeline for tabular Kaggle competitions — leak-free CV, MLflow-tracked, dashboarded.**

KaggleSlayer takes a raw Kaggle competition folder and gives you back a leaderboard-ready `submission.csv`. Every run is tracked in MLflow and surfaced in a Streamlit dashboard so you can compare experiments across competitions.

```bash
# One command, fresh clone → dashboard with a pre-loaded run
docker compose up dashboard
```

## What it does

1. **Data Scout** — loads `train.csv` / `test.csv`, detects the target & ID columns, classifies each feature (numeric / categorical / ordinal / date / identifier), handles missing values and outliers.
2. **Feature Engineer** — generates numerical, categorical, polynomial, and statistical features; selects via variance, correlation, and univariate tests; transforms (scale + encode).
3. **Model Selector** — evaluates RF / Extra Trees / Logistic / Ridge / Lasso / ElasticNet / XGBoost / LightGBM / CatBoost / KNN / SVM, optionally tunes the best with **Optuna** and stacks/votes ensembles.
4. **Submission** — refits the chosen pipeline on the full training set, predicts test, writes `submission.csv` (and optionally pushes via the Kaggle API).

Every run logs params, metrics, and artifacts to MLflow.

## Why it's interesting

- **Leak-free CV.** `FeatureEngineeringPipeline` (in [`agents/feature_engineer.py`](agents/feature_engineer.py)) is a sklearn-compatible `TransformerMixin`, so feature generation/selection/scaling runs *inside* every CV fold. Train-fold statistics never leak into the validation fold — CV scores actually correlate with the leaderboard.
- **Real MLOps surface.** Coordinator → MLflow run → Streamlit dashboard, containerised end-to-end. No local Jupyter babysitting.
- **String-label classification handled end-to-end.** `LabelEncoder` is fit during training, persisted (`target_encoder.pkl`), and inverse-transformed at submission time so the CSV matches Kaggle's expected labels.
- **Smart sampling for large datasets.** Stratified down-sample for CV, then refit the chosen pipeline on the full training set before predicting.

## Tech stack

| Choice | Why |
| --- | --- |
| **scikit-learn pipelines** | Lets feature engineering live *inside* CV folds — single most important correctness feature. |
| **XGBoost + LightGBM + CatBoost** | All three because each wins on different competition shapes; cheap to run them in parallel and pick the best. |
| **Optuna** | TPE search beats grid/random for the kind of conditional spaces tree boosters need. |
| **MLflow** | Local file store by default → zero infra; same code logs to a remote server in prod. |
| **Streamlit + Plotly** | Fastest path from "run finished" to a screenshot a recruiter can read. |
| **Docker Compose** | One command for `cli`, `dashboard`, and `mlflow ui`. |
| **rich** | CLI output you can read in a terminal screenshot. |

## Quickstart

### Option A — Docker (recommended)
```bash
git clone https://github.com/rodme02/KaggleSlayer.git
cd KaggleSlayer
docker compose up dashboard          # http://localhost:8501
docker compose run --rm cli titanic --data-path competition_data/titanic
```

### Option B — Local Python
```bash
git clone https://github.com/rodme02/KaggleSlayer.git
cd KaggleSlayer
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dashboard,dev]"

python scripts/download_competition.py titanic        # needs ~/.kaggle/kaggle.json
kaggle-slayer titanic --data-path competition_data/titanic
streamlit run dashboard/app.py
```

## Usage

```bash
# Single competition
kaggle-slayer titanic --data-path competition_data/titanic

# Submit to Kaggle as well
kaggle-slayer titanic --data-path competition_data/titanic --submit

# Every downloaded competition, non-interactive
kaggle-slayer --all --yes
```

Configuration lives in [`config.yaml`](config.yaml) — CV folds, Optuna trials, missing/correlation thresholds, model metric.

## Project layout

```
agents/                 # DataScout → FeatureEngineer → ModelSelector → Coordinator
core/{data,features,models}/   # Building blocks (loaders, generators, factory, ...)
utils/                  # config, io, kaggle_api, logging, tracking (MLflow)
dashboard/              # Streamlit app reading from MLflow
scripts/                # Standalone Kaggle download helpers
tests/                  # pytest suite (unit + e2e smoke on synthetic data)
```

## Outputs

After a run:

```
competition_data/<comp>/
  raw/             # Original Kaggle download
  processed/       # train_cleaned.csv, test_cleaned.csv (+ engineered if small)
  results/         # data_scout_results.json, model_selector_results.json, ...
  models/          # best_model.pkl, target_encoder.pkl
  submission.csv   # Ready to upload
mlruns/            # MLflow runs (local file store by default)
```

## Roadmap & known limitations

- Tabular only — no NN/transformer support today.
- Single-machine; no distributed training. The dashboard is a viewer, not a scheduler.
- Optuna runs sequentially per model; multi-model HP search is on the roadmap.
- Kaggle API push assumes you've accepted the competition rules.

Pull requests welcome.

## License

MIT — see [`LICENSE`](LICENSE).
