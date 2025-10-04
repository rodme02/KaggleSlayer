# KaggleSlayer

Simple, effective AutoML pipeline for Kaggle competitions.

## Features

- Automated data processing and cleaning
- Smart feature engineering with train/test consistency
- Multiple model training with hyperparameter optimization
- Automatic ensemble creation
- One-command execution

## Installation

```bash
pip install -r requirements.txt
```

## Quick Start

```bash
python kaggle_slayer.py titanic --data-path competition_data/titanic
```

## Project Structure

```
KaggleSlayer/
├── kaggle_slayer.py        # Main entry point
├── agents/                  # Pipeline components
│   ├── coordinator.py      # Orchestrates pipeline
│   ├── data_scout.py       # Data processing
│   ├── feature_engineer.py # Feature engineering
│   └── model_selector.py   # Model training
├── core/                    # Core functionality
│   ├── data/               # Data loaders & preprocessors
│   ├── features/           # Feature generators & selectors
│   └── models/             # Model factory & training
└── utils/                   # Utilities
```

## How It Works

1. **Data Scout**: Analyzes and cleans the dataset
2. **Feature Engineer**: Creates and selects features
3. **Model Selector**: Trains models and creates ensembles
4. **Output**: Generates submission.csv

## Configuration

Edit `config.yaml` to customize:

```yaml
pipeline:
  cv_folds: 5
  optuna_trials: 20
  max_features_to_create: 25

data:
  correlation_threshold: 0.95
  variance_threshold: 0.01
```

## Requirements

- Python 3.8+
- scikit-learn, pandas, numpy
- xgboost, lightgbm, catboost
- optuna

## License

MIT
