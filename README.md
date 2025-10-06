# KaggleSlayer

**Automated Machine Learning Pipeline for Kaggle Competitions**

KaggleSlayer is an end-to-end AutoML system that automatically handles data loading, preprocessing, feature engineering, model training, and submission creation for tabular Kaggle competitions.

## Features

- **Automatic Competition Discovery**: Finds and downloads tabular competitions from Kaggle
- **Smart Data Processing**: Handles missing values, outliers, duplicates, and datetime features
- **Advanced Feature Engineering**: Creates numerical, categorical, polynomial, and statistical features
- **Multi-Model Training**: Supports Random Forest, XGBoost, LightGBM, CatBoost, and more
- **Leak-Free CV**: Feature engineering inside cross-validation folds to prevent data leakage
- **String Label Support**: Automatically handles multi-class string classification targets
- **Submission Generation**: Creates properly formatted submission files ready for Kaggle

## Quick Start

### 1. Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/KaggleSlayer.git
cd KaggleSlayer

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Setup Kaggle API

1. Create a Kaggle API token at https://www.kaggle.com/account
2. Download `kaggle.json` and place it in:
   - Linux/Mac: `~/.kaggle/kaggle.json`
   - Windows: `C:\Users\<username>\.kaggle\kaggle.json`

### 3. Download Competitions

Most competitions in Kaggle require you to accept the rules before downloading data. Make sure to visit the competition page and accept the rules first.

```bash
# Download a single competition
python download_competition.py titanic

# Batch download all tabular competitions (auto-discovery)
python download_all_competitions.py
```

### 4. Run Pipeline

```bash
# Run on a competition
python kaggle_slayer.py titanic --data-path competition_data/titanic

# Automatically submits to Kaggle
python kaggle_slayer.py titanic --data-path competition_data/titanic --submit
```

## Usage Examples

### Single Competition

```bash
# Download and run pipeline
python download_competition.py house-prices-advanced-regression-techniques
python kaggle_slayer.py house-prices-advanced-regression-techniques --data-path competition_data/house-prices-advanced-regression-techniques
```

### Batch Processing

```bash
# Download all entered competitions
python download_all_competitions.py
```

### Configuration

Edit `config.yaml` to customize pipeline behavior:

```yaml
pipeline:
  cv_folds: 5                    # Cross-validation folds
  cv_random_state: 42            # Random seed
  max_features_to_create: 25     # Max engineered features
  polynomial_degree: 2           # Polynomial feature degree

data:
  drop_missing_threshold: 0.9    # Drop columns with >90% missing
  correlation_threshold: 0.95    # Drop highly correlated features
  variance_threshold: 0.01       # Drop low-variance features

model_selection:
  classification_metric: 'accuracy'
  regression_metric: 'neg_mean_squared_error'
```

## Project Structure

```
KaggleSlayer/
├── agents/                 # Pipeline orchestration
│   ├── coordinator.py      # Main pipeline coordinator
│   ├── data_scout.py       # Data exploration and cleaning
│   ├── feature_engineer.py # Feature engineering
│   └── model_selector.py   # Model training and selection
├── core/                   # Core functionality
│   ├── data/              # Data loading and preprocessing
│   ├── features/          # Feature generation and transformation
│   └── models/            # Model factory, evaluation, optimization
├── utils/                  # Utilities
│   ├── config.py          # Configuration management
│   ├── io.py              # File I/O operations
│   ├── kaggle_api.py      # Kaggle API integration
│   └── logging.py         # Logging utilities
├── kaggle_slayer.py       # Main entry point
├── download_competition.py # Download single competition
├── download_all_competitions.py # Batch download
├── config.yaml            # Configuration file
└── requirements.txt       # Python dependencies
```

## Key Features Explained

### Automatic Competition Discovery

The batch download script uses multiple strategies to find tabular competitions:
- **Entered competitions**: Competitions you've joined
- **Search keywords**: "tabular", "classification", "regression", "structured", "prediction", etc.
- **Categories**: gettingStarted, playground, featured
- **Known-good list**: 42 popular tabular competitions

### Data Processing Pipeline

1. **Data Scout**: Loads data, detects target column, analyzes feature types
2. **Preprocessing**: Handles missing values, removes duplicates (train only), detects outliers
3. **Feature Engineering**: Creates 25+ engineered features per competition
4. **Model Training**: Trains multiple models with leak-free cross-validation
5. **Submission**: Generates properly formatted submission file

### String Label Handling

KaggleSlayer automatically handles multi-class string classification:
- Encodes labels during training (e.g., 'Dropout' → 0, 'Enrolled' → 1, 'Graduate' → 2)
- Trains models with numeric labels
- Decodes predictions back to original strings for submission
- Saves target encoder for consistent encoding/decoding

### Leak-Free Cross-Validation

Feature engineering happens **inside** CV folds to prevent data leakage:
- Training data statistics never leak to validation folds
- Ensures honest cross-validation scores
- Better generalization to test data

## Output

After running the pipeline, you'll find:

```
competition_data/{competition_name}/
├── raw/                       # Original data
│   ├── train.csv
│   ├── test.csv
│   └── sample_submission.csv
├── processed/                 # Cleaned data
│   ├── train_cleaned.csv
│   └── test_cleaned.csv
├── results/                   # Pipeline outputs
│   ├── data_scout_results.json
│   ├── feature_engineer_results.json
│   ├── model_selector_results.json
│   └── pipeline_results.json
├── models/                    # Saved models
│   ├── best_model.pkl
│   └── target_encoder.pkl (if string labels)
└── submission.csv            # Ready for upload
```

## Supported Models

**Classification:**
- Random Forest, Extra Trees
- Logistic Regression
- XGBoost, LightGBM, CatBoost
- K-Nearest Neighbors, SVM

**Regression:**
- Random Forest, Extra Trees
- Ridge, Lasso, ElasticNet
- XGBoost, LightGBM, CatBoost
- K-Nearest Neighbors, SVR

## Requirements

- Python 3.8+
- pandas, numpy, scikit-learn
- xgboost, lightgbm, catboost
- kaggle (Kaggle API)
- PyYAML, joblib

See `requirements.txt` for complete list.

## Tips

1. **Accept competition rules** on Kaggle before downloading
2. **Start with small competitions** (Titanic, House Prices) to test
3. **Use batch download** to discover new competitions automatically
4. **Check config.yaml** to tune pipeline behavior
5. **Monitor CV scores** - they should correlate with leaderboard scores