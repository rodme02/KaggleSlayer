# CLAUDE.md

Guidance for Claude Code when working in this repo.

## What this is
KaggleSlayer is an AutoML pipeline for tabular Kaggle competitions. The differentiator is **leak-free cross-validation**: feature engineering runs *inside* CV folds via a sklearn-compatible pipeline, so train-fold statistics never leak into validation folds.

## Layout
```
kaggle_slayer.py         # CLI entry point (also exposed as `kaggle-slayer`)
agents/                  # Orchestration layer
  base_agent.py          # Shared logging + file I/O
  coordinator.py         # Runs DataScout → FeatureEngineer → ModelSelector → Submission
  data_scout.py          # Loads, validates, cleans data
  feature_engineer.py    # Wraps generators/selectors/transformers as a sklearn Pipeline
  model_selector.py      # Trains models, runs Optuna, picks best
core/                    # Building blocks
  data/                  # loaders, preprocessors, validators
  features/              # generators, selectors, transformers
  models/                # factory, evaluators, optimizers, ensembles
utils/
  config.py              # YAML config manager
  io.py                  # FileManager (per-competition results/processed/models layout)
  kaggle_api.py          # Kaggle CLI wrapper
  logging.py             # Logger setup
  tracking.py            # MLflow integration
dashboard/               # Streamlit dashboard (reads MLflow store)
scripts/                 # Standalone Kaggle download scripts
tests/                   # pytest suite (synthetic fixtures)
```

## Pipeline flow
1. `DataScoutAgent` loads `train.csv` / `test.csv`, detects target & ID columns, cleans, writes `processed/{train,test}_cleaned.csv` + `results/data_scout_results.json`.
2. `FeatureEngineerAgent` (small datasets only — skipped >100k rows) writes `processed/{train,test}_engineered.csv`.
3. `ModelSelectorAgent` loads cleaned data, wraps `FeatureEngineeringPipeline` + model in a sklearn `Pipeline`, evaluates each model with K-fold CV (engineering happens inside the fold), optionally runs Optuna and ensembling, persists `models/best_model.pkl` (and `target_encoder.pkl` for string labels).
4. `PipelineCoordinator.create_submission()` loads the persisted pipeline, predicts on `test_cleaned.csv`, writes `submission.csv`. Optionally submits via Kaggle API.

Per-run artifacts live under `competition_data/<comp>/{raw,processed,results,models}/`. MLflow runs are written to `./mlruns/` by default (override with `MLFLOW_TRACKING_URI`).

## Running
```
pip install -e ".[dashboard,dev]"
kaggle-slayer titanic --data-path competition_data/titanic
streamlit run dashboard/app.py
pytest -q
```

## Conventions
- Type-hint public function signatures.
- Errors at boundaries (CLI, file I/O, Kaggle API) get caught and logged; everything else propagates.
- New runtime deps go in `pyproject.toml`, not `requirements.txt` (which is regenerated from it).
- CLI output uses `rich`; structured output uses the `logging` module — never bare `print` for production code paths.
