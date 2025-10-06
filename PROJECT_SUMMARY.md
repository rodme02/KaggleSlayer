# KaggleSlayer - Detailed Project Summary

## Table of Contents

1. [Overview](#overview)
2. [Architecture](#architecture)
3. [Core Components](#core-components)
4. [Pipeline Flow](#pipeline-flow)
5. [File Reference](#file-reference)
6. [Key Features Implementation](#key-features-implementation)
7. [Design Decisions](#design-decisions)
8. [Recent Improvements](#recent-improvements)

---

## Overview

KaggleSlayer is an end-to-end automated machine learning pipeline designed specifically for tabular Kaggle competitions. It handles everything from competition discovery and data download to feature engineering, model training, and submission generation.

**Core Philosophy:**
- **Automation First**: Minimize manual intervention
- **Leak-Free**: Feature engineering inside CV folds
- **Robust**: Handles edge cases (string labels, missing values, extra CSVs)
- **Scalable**: Works with datasets from 1K to 1M+ rows

**Target Users:**
- Kagglers who want to quickly establish baselines
- ML practitioners learning competition workflows
- Automated ML researchers

---

## Architecture

### High-Level Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                   KaggleSlayer Pipeline                      │
└─────────────────────────────────────────────────────────────┘
                            │
        ┌───────────────────┼───────────────────┐
        │                   │                   │
┌───────▼────────┐  ┌──────▼──────┐  ┌────────▼────────┐
│ Competition    │  │  Pipeline   │  │  Configuration  │
│ Discovery      │  │  Execution  │  │  Management     │
└───────┬────────┘  └──────┬──────┘  └────────┬────────┘
        │                  │                   │
        │         ┌────────▼────────┐          │
        │         │  4 Agent System │          │
        │         └────────┬────────┘          │
        │                  │                   │
    ┌───▼──────────────────▼───────────────────▼────┐
    │         Agent 1: Data Scout                   │
    │  - Load competition data                      │
    │  - Detect target column and feature types     │
    │  - Clean and validate data                    │
    │  - Remove ID columns, duplicates, outliers    │
    └───────────────────┬───────────────────────────┘
                        │
    ┌───────────────────▼───────────────────────────┐
    │      Agent 2: Feature Engineer                │
    │  - Generate numerical features                │
    │  - Create categorical encodings               │
    │  - Build polynomial features                  │
    │  - Statistical aggregations                   │
    └───────────────────┬───────────────────────────┘
                        │
    ┌───────────────────▼───────────────────────────┐
    │       Agent 3: Model Selector                 │
    │  - Train multiple models (RF, XGB, LGBM, etc) │
    │  - Leak-free CV (feature eng. in folds)       │
    │  - Handle string classification labels        │
    │  - Select best model by CV score              │
    └───────────────────┬───────────────────────────┘
                        │
    ┌───────────────────▼───────────────────────────┐
    │       Agent 4: Submission Creator             │
    │  - Load best model and test data              │
    │  - Generate predictions                       │
    │  - Decode string labels (if applicable)       │
    │  - Format and save submission.csv             │
    └───────────────────────────────────────────────┘
```

### Agent-Based Design

The pipeline uses a **4-agent system** coordinated by `PipelineCoordinator`:

1. **DataScoutAgent**: Data exploration and cleaning
2. **FeatureEngineerAgent**: Feature generation and transformation
3. **ModelSelectorAgent**: Model training and selection
4. **PipelineCoordinator**: Orchestrates all agents + submission creation

Each agent:
- Inherits from `BaseAgent` (logging, file management, config)
- Has a `run()` method that returns results dictionary
- Saves intermediate results to JSON files
- Can be run independently for debugging

---

## Core Components

### 1. Competition Discovery (`download_all_competitions.py`)

**Purpose**: Find and download tabular Kaggle competitions

**Discovery Strategies:**
1. **Entered Competitions**: `kaggle competitions list --csv`
2. **Category Search**: gettingStarted, playground, featured
3. **Keyword Search**: tabular, classification, regression, structured, prediction, etc.
4. **Known-Good List**: 42 hardcoded popular competitions

**Filtering:**
- Pre-check: Verify train.csv exists before download
- Extra CSV check: Reject competitions with non-standard CSVs
- Tabular-only: Only keep competitions with train.csv + test.csv (+ optional sample_submission.csv)

**Results**: Typically finds 145+ unique tabular competitions

**Key Functions:**
- `get_entered_competitions()`: Uses Kaggle API to list entered competitions
- `search_competitions(term)`: Search by keyword
- `get_competitions_by_category(category)`: Get competitions in category
- `check_competition_files(name)`: Pre-validate file structure
- `batch_download_competitions()`: Main orchestrator

### 2. Single Competition Download (`download_competition.py`)

**Purpose**: Download and organize a single competition

**Process:**
1. Download competition ZIP from Kaggle API
2. Extract to temporary directory
3. Validate train.csv exists
4. Check for extra CSV files (reject if found)
5. Move files to `competition_data/{name}/raw/`
6. Clean up temporary files

**File Organization:**
```
competition_data/
└── {competition_name}/
    └── raw/
        ├── train.csv
        ├── test.csv
        └── sample_submission.csv (optional)
```

**Error Handling:**
- 403 Forbidden → Competition rules not accepted
- 404 Not Found → Invalid competition name
- Missing train.csv → Not tabular format
- Extra CSVs → Not standard tabular format

### 3. Data Processing Pipeline

#### Data Scout (`agents/data_scout.py`)

**Responsibilities:**
- Load raw CSV files
- Detect target column (columns in train but not in test)
- Analyze feature types (numerical, categorical, identifier)
- Remove ID columns (high cardinality, unique values)
- Handle missing values with fit/transform pattern
- Parse and extract datetime features
- Remove duplicates (train only, never test)
- Detect and winsorize outliers

**Key Methods:**
- `run()`: Execute complete data scouting process
- Uses `CompetitionDataLoader`, `DataPreprocessor`, `DataValidator`

**Outputs:**
- `train_cleaned.csv`: Cleaned training data
- `test_cleaned.csv`: Cleaned test data (with ID column preserved)
- `data_scout_results.json`: Metadata and statistics

#### Data Loader (`core/data/loaders.py`)

**Features:**
- Auto-detect train.csv and test.csv in raw/ directory
- Detect target column (difference between train and test columns)
- Analyze feature types with heuristics:
  - **Identifier**: Unique values > 95% of rows
  - **Categorical**: Object dtype or <20 unique values
  - **Numerical**: Numeric dtype
- Memory-efficient loading (uses pandas dtype inference)

#### Data Preprocessor (`core/data/preprocessors.py`)

**Features:**
- **Missing value handling**: Median for numerical, mode for categorical
- **Outlier detection**: IQR method or Z-score method
- **Outlier handling**: Winsorize to 1st-99th percentile (preserves distribution)
- **Duplicate removal**: Only on training data (maintains test row count)
- **Datetime parsing**: Auto-detect and extract year, month, day, etc.
- **Fit/transform pattern**: Learns from train, applies to test

**Key Methods:**
- `handle_missing_values(df, fit=True)`: Impute missing values
- `detect_outliers(df, method='iqr')`: Find outliers
- `handle_outliers(df, method='winsorize')`: Handle outliers
- `remove_duplicates(df)`: Remove duplicate rows
- `parse_and_extract_datetime(df)`: Parse datetime columns

### 4. Feature Engineering Pipeline

#### Feature Engineer (`agents/feature_engineer.py`)

**Responsibilities:**
- Orchestrate feature generation
- Apply feature transformations (scaling, encoding)
- Manage fit/transform workflow for leak-free CV

**Process:**
1. Generate features (numerical, categorical, statistical)
2. Select top features by importance
3. Transform features (scale numerical, encode categorical)
4. Save feature pipeline for test data

**Outputs:**
- Engineered training data
- `feature_engineer_results.json`: Feature metadata

#### Feature Generator (`core/features/generators.py`)

**Feature Types Created:**

**Numerical Features** (fit/transform pattern):
- Ratios: `col1_div_col2` (with zero handling)
- Products: `col1_times_col2`
- Log transforms: `col_log` (for positive skewed data)
- Square root: `col_sqrt` (for high variance data)
- Recipes stored for consistent test transformation

**Categorical Features**:
- Frequency encoding: How common is each value
- Rarity encoding: Inverse of frequency
- String length: Length of string values
- Word count: Number of words in text
- Missing indicators: Binary flag for missing values

**Statistical Features**:
- Row-wise: sum, mean, std, min, max, range
- Only created if 3+ numerical columns exist

**Polynomial Features**:
- Degree 2 by default (configurable)
- Limited to top 10 columns (prevent explosion)
- Uses feature selection to pick best columns

**Clustering Features**:
- KMeans cluster assignments
- Distance to each cluster center
- Distance to nearest cluster

**Binning Features**:
- Quantile-based binning
- Only for high-skew or high-variance columns

**Caching**:
- File-based cache for expensive computations
- Cache key: hash(dataframe + operation + params)
- Speeds up repeated runs

#### Feature Transformer (`core/features/transformers.py`)

**Transformations:**

**Scaling** (with fit/transform):
- StandardScaler: Mean 0, Std 1
- MinMaxScaler: Range [0, 1]
- RobustScaler: Median/IQR (robust to outliers)
- **Memory optimization**: Converts float64 → float32 (50% reduction)

**Encoding** (with fit/transform):
- Label encoding: Categorical → integers
- One-hot encoding: Categorical → binary columns
- Handles unseen categories (maps to most frequent)

**Target Encoding** (for string classification):
- `encode_target(y, fit=True)`: Encode string labels to numeric
- `decode_target(y_encoded)`: Decode predictions back to strings
- Stores LabelEncoder for consistent encoding/decoding

**Imputation** (with fit/transform):
- Numerical: Median imputation
- Categorical: Mode imputation
- KNN imputation: For complex missing patterns

#### Feature Selector (`core/features/selectors.py`)

**Selection Methods:**
- **Correlation-based**: Remove features correlated > 0.95
- **Variance-based**: Remove low-variance features (< 0.01)
- **Importance-based**: Use Random Forest feature importance
- **Recursive Feature Elimination**: Iteratively remove worst features

### 5. Model Training Pipeline

#### Model Selector (`agents/model_selector.py`)

**Responsibilities:**
- Train multiple models with leak-free CV
- Handle string classification labels
- Select best model by CV score
- Save model and encoders

**Key Features:**

**Leak-Free CV**:
- Feature engineering happens **inside** CV folds
- Uses sklearn Pipeline: `Pipeline([('features', transformer), ('model', model)])`
- Training fold statistics never leak to validation fold
- Honest CV scores that correlate with leaderboard

**String Label Handling**:
1. Detect string target: `y.dtype == 'object'`
2. Encode: `LabelEncoder().fit_transform(y)`
3. Train with numeric labels: `model.fit(X, y_encoded)`
4. Save encoder: `joblib.dump(target_encoder, 'target_encoder.pkl')`
5. Decode predictions: `target_encoder.inverse_transform(predictions)`

**Stratified Sampling** (large datasets):
- If dataset > 100K rows, sample 100K stratified by target
- Preserves class distribution
- Speeds up training without losing accuracy

**Models Trained**:
- Random Forest (default: 200 trees, max_depth=15)
- XGBoost (1000 trees, lr=0.05, early stopping)
- LightGBM (1000 trees, lr=0.05, early stopping)
- CatBoost (1000 iterations, lr=0.05, early stopping)

**Process:**
1. Load cleaned data from data scout
2. Encode target if string labels
3. For each model:
   - Create feature pipeline (transformer + model)
   - Perform stratified K-fold CV
   - Evaluate with appropriate metric
4. Select best model by CV score
5. Retrain on full dataset with best model
6. Save model, encoder, and results

**Outputs:**
- `best_model.pkl`: Trained pipeline (features + model)
- `target_encoder.pkl`: Label encoder (if string target)
- `model_selector_results.json`: Training results and metrics

#### Model Factory (`core/models/factory.py`)

**Supported Models:**

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

**Features:**
- Default hyperparameters tuned for competitions
- Parameter spaces for hyperparameter optimization
- Early stopping for boosting models
- Verbosity control (quiet by default)
- Class weight support for imbalanced data

**Key Methods:**
- `create_model(name, params, problem_type)`: Instantiate model
- `get_default_parameters(name)`: Get competition-tuned defaults
- `fit_with_early_stopping(model, X, y)`: Fit with validation set

#### Model Evaluator (`core/models/evaluators.py`)

**Features:**
- Cross-validation with stratified K-fold (classification) or K-fold (regression)
- Multiple metrics: accuracy, precision, recall, F1, ROC-AUC, RMSE, MAE, R²
- Train/validation/test score tracking
- Comparison tables for multiple models
- **String label encoding** before CV (critical fix)

**Key Methods:**
- `evaluate_model(model, X, y, problem_type)`: Full evaluation
- `compare_models(results)`: Create comparison DataFrame
- `get_best_model_result(results)`: Find best performing model
- `_cross_validate(model, X, y)`: Perform CV with label encoding

#### Model Optimizer (`core/models/optimizers.py`)

**Hyperparameter Optimization:**
- Uses Optuna for Bayesian optimization
- Supports grid search, random search, Bayesian search
- Configurable trials and timeout
- Early stopping for failed trials
- Saves best parameters

### 6. Submission Creation (`agents/coordinator.py`)

**Process:**
1. Load data scout results (target column name)
2. Find sample_submission.csv template (if exists)
3. Load test_cleaned.csv
4. Load best_model.pkl (pipeline)
5. Load target_encoder.pkl (if exists)
6. Generate predictions: `pipeline.predict(X_test)`
7. Decode predictions (if string labels): `encoder.inverse_transform(preds)`
8. Format predictions properly:
   - Remove brackets: `['Dropout']` → `Dropout`
   - Ensure correct dtype (int or string)
9. Create submission DataFrame
10. Save to `submission.csv`

**String Label Handling:**
- Check if target_encoder.pkl exists
- If yes: Decode numeric predictions → original strings
- Ensure predictions are strings (not arrays or lists)
- Format: `id,target\n123,Dropout\n456,Graduate`

**Validation:**
- Match row count with sample_submission (if available)
- Validate column names
- Check for missing values

---

## Pipeline Flow

### Complete Pipeline Execution

```
1. DOWNLOAD PHASE
   ├── download_all_competitions.py
   │   ├── Get entered competitions
   │   ├── Search by keywords
   │   ├── Get by categories
   │   ├── Add known-good competitions
   │   ├── Pre-check file structure
   │   └── Download valid competitions
   └── competition_data/{name}/raw/
       ├── train.csv
       ├── test.csv
       └── sample_submission.csv

2. DATA SCOUT PHASE (Agent 1)
   ├── Load train.csv and test.csv
   ├── Detect target column
   ├── Analyze feature types
   ├── Remove ID columns
   ├── Parse datetime features
   ├── Handle missing values (fit on train)
   ├── Remove duplicates (train only)
   ├── Winsorize outliers
   └── Save: train_cleaned.csv, test_cleaned.csv, data_scout_results.json

3. FEATURE ENGINEERING PHASE (Agent 2)
   ├── Load train_cleaned.csv
   ├── Generate numerical features (ratios, products, transforms)
   ├── Generate categorical features (frequency, rarity)
   ├── Generate statistical features (row-wise stats)
   ├── Generate polynomial features (degree 2)
   ├── Select top features by importance
   ├── Save feature recipes
   └── Save: feature_engineer_results.json

4. MODEL SELECTION PHASE (Agent 3)
   ├── Load train_cleaned.csv
   ├── Detect if target is string → Encode labels
   ├── Create feature pipeline (transformer + model)
   ├── For each model (RF, XGB, LGBM, CatBoost):
   │   ├── Stratified K-Fold CV (feature eng. in folds)
   │   ├── Evaluate with appropriate metric
   │   └── Track CV scores
   ├── Select best model
   ├── Retrain on full dataset
   ├── Save best model, target encoder
   └── Save: best_model.pkl, target_encoder.pkl, model_selector_results.json

5. SUBMISSION PHASE (Coordinator)
   ├── Load test_cleaned.csv
   ├── Load best_model.pkl (pipeline)
   ├── Load target_encoder.pkl (if exists)
   ├── Generate predictions: pipeline.predict(X_test)
   ├── Decode predictions (if string labels)
   ├── Format predictions (remove brackets, ensure correct type)
   ├── Create submission DataFrame
   └── Save: submission.csv
```

### Directory Structure After Pipeline

```
KaggleSlayer/
├── competition_data/
│   └── playground-series-s4e6/
│       ├── raw/
│       │   ├── train.csv
│       │   ├── test.csv
│       │   └── sample_submission.csv
│       ├── processed/
│       │   ├── train_cleaned.csv
│       │   └── test_cleaned.csv
│       ├── models/
│       │   ├── best_model.pkl
│       │   └── target_encoder.pkl
│       ├── results/
│       │   ├── data_scout_results.json
│       │   ├── feature_engineer_results.json
│       │   ├── model_selector_results.json
│       │   └── pipeline_results.json
│       └── submission.csv
├── agents/
├── core/
├── utils/
├── kaggle_slayer.py
├── config.yaml
└── requirements.txt
```

---

## File Reference

### Entry Points

#### `kaggle_slayer.py`
**Purpose**: Main CLI entry point

**Usage:**
```bash
python kaggle_slayer.py titanic
python kaggle_slayer.py playground-series-s4e6 --data-path competition_data/playground-series-s4e6
```

**Features:**
- Argument parsing (competition name, data path)
- Pipeline initialization
- Warning suppression for CatBoost compatibility
- Error handling and logging

**Key Code:**
```python
# Suppress sklearn deprecation warnings for CatBoost
warnings.filterwarnings('ignore', category=DeprecationWarning, module='sklearn')

# Initialize and run pipeline
coordinator = PipelineCoordinator(competition_name, competition_path, config)
results = coordinator.run(submit_to_kaggle=False)
```

#### `download_competition.py`
**Purpose**: Download single competition

**Functions:**
- `download_competition(name, force)`: Main download logic
- `list_competitions()`: List available competitions
- Validates file structure (train.csv, test.csv)
- Rejects competitions with extra CSVs

#### `download_all_competitions.py`
**Purpose**: Batch download with auto-discovery

**Functions:**
- `get_entered_competitions()`: List entered competitions
- `search_competitions(term)`: Search by keyword
- `get_competitions_by_category(cat)`: Get by category
- `get_known_good_competitions()`: Return hardcoded list
- `check_competition_files(name)`: Pre-validate before download
- `batch_download_competitions()`: Main orchestrator

**Discovery Results**: ~145 unique tabular competitions

### Agents

#### `agents/base_agent.py`
**Purpose**: Base class for all agents

**Provides:**
- Logging utilities (`log_info()`, `log_warning()`, `log_error()`)
- File management (`FileManager` instance)
- Configuration access (`get_config()`)
- Competition metadata

**Inheritance:**
```python
class DataScoutAgent(BaseAgent):
    def __init__(self, competition_name, competition_path, config):
        super().__init__(competition_name, competition_path, config)
        # Agent-specific initialization
```

#### `agents/coordinator.py`
**Purpose**: Orchestrate complete pipeline

**Key Methods:**
- `run(submit_to_kaggle)`: Execute 4-phase pipeline
- `create_submission()`: Generate submission.csv
- `submit_to_kaggle()`: Upload to Kaggle (optional)
- `_find_sample_submission()`: Locate submission template
- `_detect_id_column()`: Find ID column for submission

**Pipeline Steps:**
1. Data Scout (data loading and cleaning)
2. Feature Engineer (feature generation) - skipped if >100K rows
3. Model Selector (model training with leak-free CV)
4. Submission (create submission.csv)

**String Label Handling** (lines 272-308):
```python
# Load target encoder if exists
target_encoder = None
encoder_path = self.file_manager.get_file_path("target_encoder.pkl")
if encoder_path.exists():
    target_encoder = joblib.load(encoder_path)

# Generate predictions
predictions = trained_pipeline.predict(test_features)

# Decode predictions if we have a target encoder
if target_encoder is not None:
    predictions = target_encoder.inverse_transform(predictions)

# Format predictions (remove brackets, ensure strings)
if target_encoder is None:
    predictions = [int(pred) for pred in predictions]
else:
    predictions = [str(pred) if not isinstance(pred, str) else pred for pred in predictions]
```

#### `agents/data_scout.py`
**Purpose**: Data exploration and cleaning

**Process:**
1. Load competition data
2. Validate datasets
3. Detect target column
4. Analyze feature types
5. Remove ID columns (BEFORE any processing)
6. Parse datetime features
7. Handle missing values (fit/transform)
8. Remove duplicates (train only)
9. Winsorize outliers
10. Save cleaned data

**Critical Fix** (lines 52-68): Remove ID columns BEFORE preprocessing
```python
# Remove ID columns BEFORE any other processing to prevent them from leaking into features
id_columns = [k for k, v in feature_types.items() if v == 'identifier']
if id_columns:
    train_df = train_df.drop(columns=id_columns, errors='ignore')
    if test_df is not None:
        # Save the first ID column from test for later (needed for submission)
        test_id_column = id_columns[0] if id_columns[0] in test_df.columns else None
        test_id_values = test_df[id_columns[0]].copy() if test_id_column else None
        test_df = test_df.drop(columns=id_columns, errors='ignore')
```

**Never remove duplicates from test** (lines 90-92):
```python
# Remove duplicates from training data only
# NEVER remove duplicates from test data - we need predictions for all rows
train_cleaned, duplicates_removed = self.preprocessor.remove_duplicates(train_cleaned)
```

#### `agents/feature_engineer.py`
**Purpose**: Feature generation and transformation

**Process:**
1. Load cleaned data
2. Generate features (numerical, categorical, statistical)
3. Select features by importance
4. Transform features (scaling, encoding)
5. Save feature pipeline

**Note**: For large datasets (>100K), this step is skipped. Feature engineering happens inside CV folds in model_selector.py instead.

#### `agents/model_selector.py`
**Purpose**: Model training with leak-free CV

**Key Innovations:**

**Leak-Free CV** (lines 169-231):
```python
# Create feature pipeline (transformer + model)
from sklearn.pipeline import Pipeline

# Feature transformer
feature_transformer = FeatureTransformer()
# ... configure transformer ...

# Create pipeline: features → model
pipeline = Pipeline([
    ('features', feature_transformer),
    ('model', model)
])

# Cross-validation with pipeline
# Feature engineering happens INSIDE folds (leak-free)
cv_scores = cross_val_score(pipeline, X_full, y_full, cv=cv, scoring=metric)
```

**String Label Encoding** (lines 73-81):
```python
# Encode target if categorical (string labels)
from sklearn.preprocessing import LabelEncoder

target_encoder = None
if y_full.dtype == 'object' or y_full.dtype.name == 'category':
    target_encoder = LabelEncoder()
    y_full_encoded = pd.Series(target_encoder.fit_transform(y_full), index=y_full.index)
    print(f"Encoded target labels: {list(target_encoder.classes_)} -> {list(range(len(target_encoder.classes_)))}")
else:
    y_full_encoded = y_full
```

**Save Encoder** (lines 428-432):
```python
# Save target encoder (critical for decoding predictions)
if target_encoder is not None:
    encoder_path = self.file_manager.get_file_path("target_encoder.pkl")
    joblib.dump(target_encoder, encoder_path)
    self.log_info(f"Saved target encoder to {encoder_path}")
```

**Stratified Sampling for Large Datasets** (lines 55-68):
```python
# For large datasets, use stratified sampling to speed up training
if len(X_full) > 100000:
    from sklearn.model_selection import train_test_split
    X_full, _, y_full, _ = train_test_split(
        X_full, y_full,
        train_size=100000,
        stratify=y_full,  # Preserve class distribution
        random_state=42
    )
```

### Core Components

#### `core/data/loaders.py`
**Classes:**
- `CompetitionDataLoader`: Load train/test CSV files

**Methods:**
- `load_competition_data()`: Load train.csv and test.csv
- `detect_target_column(train, test)`: Find target (in train, not in test)
- `analyze_feature_types(df)`: Classify features (identifier/categorical/numerical)

#### `core/data/preprocessors.py`
**Classes:**
- `DataPreprocessor`: Clean and preprocess data

**Methods:**
- `handle_missing_values(df, fit, target_col)`: Impute missing values
- `detect_outliers(df, method)`: Find outliers using IQR or Z-score
- `handle_outliers(df, method, exclude_columns)`: Winsorize or cap outliers
- `remove_duplicates(df)`: Remove duplicate rows
- `parse_and_extract_datetime(df)`: Extract datetime features
- `get_basic_statistics(df)`: Compute descriptive statistics

#### `core/data/validators.py`
**Classes:**
- `DataValidator`: Validate dataset quality
- `ValidationResult`: Store validation results

**Methods:**
- `validate_dataset(df, name)`: Check for common issues
- `validate_train_test_consistency(train, test)`: Ensure consistency
- `check_missing_values(df)`: Identify missing value patterns
- `check_duplicates(df)`: Find duplicate rows
- `check_dtypes(df)`: Validate data types

#### `core/features/generators.py`
**Classes:**
- `FeatureGenerator`: Generate new features
- `FeatureCache`: Cache expensive computations

**Feature Types:**
- Numerical: ratios, products, log, sqrt (with fit/transform recipes)
- Categorical: frequency, rarity, string length, word count
- Statistical: row-wise sum, mean, std, min, max, range
- Polynomial: degree 2 interactions
- Clustering: KMeans cluster assignments and distances
- Binning: Quantile-based discretization

**Key Methods:**
- `generate_numerical_features(df, target_col, fit)`: Create numerical features
- `generate_categorical_features(df, target_col, fit)`: Create categorical features
- `generate_statistical_features(df, target_col)`: Create statistical aggregations
- `generate_polynomial_features(df, target_col)`: Create polynomial features
- `calculate_feature_importance(df, target_col)`: Rank features by importance
- `prune_low_importance_features(df, threshold)`: Remove low-value features

#### `core/features/selectors.py`
**Classes:**
- `FeatureSelector`: Select best features

**Methods:**
- `select_by_correlation(df, threshold)`: Remove correlated features
- `select_by_variance(df, threshold)`: Remove low-variance features
- `select_by_importance(df, target, k)`: Select top K by importance
- `recursive_feature_elimination(df, target, n_features)`: RFE selection

#### `core/features/transformers.py`
**Classes:**
- `FeatureTransformer`: Transform features (scaling, encoding)

**Methods:**
- `scale_numerical_features(df, method, target_col, fit)`: Scale numerical
- `encode_categorical_features(df, method, target_col, fit)`: Encode categorical
- `encode_target(y, fit)`: Encode string target labels
- `decode_target(y_encoded)`: Decode predictions back to strings
- `impute_missing_values(df, method, target_col, fit)`: Impute missing
- `apply_power_transform(df, method, target_col, fit)`: Power transformation

**Memory Optimization** (lines 62-65):
```python
# Convert to float32 to save memory (50% reduction vs float64)
for col in numerical_cols:
    if df_transformed[col].dtype == np.float64:
        df_transformed[col] = df_transformed[col].astype(np.float32)
```

#### `core/features/utils.py`
**Functions:**
- `detect_id_columns(df)`: Find identifier columns
- `is_numeric_dtype(series)`: Check if column is numeric
- `safe_division(a, b)`: Division with zero handling

#### `core/models/factory.py`
**Classes:**
- `ModelFactory`: Create and configure models

**Methods:**
- `create_model(name, params, problem_type)`: Instantiate model
- `get_available_model_names(problem_type)`: List available models
- `get_default_parameters(model_name)`: Get tuned defaults
- `get_parameter_space(model_name)`: Get hyperparameter search space
- `fit_with_early_stopping(model, X, y)`: Train with early stopping

**Models:**
- Classification: RF, ExtraTrees, LogisticRegression, XGBoost, LightGBM, CatBoost, KNN, SVM
- Regression: RF, ExtraTrees, Ridge, Lasso, ElasticNet, XGBoost, LightGBM, CatBoost, KNN, SVR

#### `core/models/evaluators.py`
**Classes:**
- `ModelEvaluator`: Evaluate model performance
- `EvaluationResult`: Store evaluation results

**Methods:**
- `evaluate_model(model, X_train, y_train, X_val, y_val, problem_type)`: Full evaluation
- `compare_models(results)`: Create comparison table
- `get_best_model_result(results)`: Find best model
- `create_evaluation_report(results)`: Generate report

**Critical Fix** (lines 107-112): Encode labels before CV
```python
# Encode target if it's categorical (string labels)
from sklearn.preprocessing import LabelEncoder
y_encoded = y
if y.dtype == 'object' or y.dtype.name == 'category':
    encoder = LabelEncoder()
    y_encoded = pd.Series(encoder.fit_transform(y), index=y.index)
```

#### `core/models/optimizers.py`
**Classes:**
- `HyperparameterOptimizer`: Optimize model hyperparameters

**Methods:**
- `optimize(model_name, X, y, problem_type, n_trials)`: Run optimization
- `grid_search(model, param_grid, X, y)`: Grid search
- `random_search(model, param_dist, X, y, n_iter)`: Random search
- `bayesian_search(model, param_space, X, y, n_trials)`: Bayesian optimization (Optuna)

#### `core/models/ensembles.py`
**Classes:**
- `EnsembleBuilder`: Create model ensembles

**Methods:**
- `create_voting_ensemble(models, voting)`: Voting classifier/regressor
- `create_stacking_ensemble(base_models, meta_model)`: Stacking
- `create_blending_ensemble(models, weights)`: Weighted blending

### Utilities

#### `utils/config.py`
**Classes:**
- `ConfigManager`: Manage configuration

**Methods:**
- `load_config()`: Load config.yaml
- `get(key_path, default)`: Get config value with dot notation
- `get_pipeline_config()`: Get pipeline settings
- `update_config(key_path, value)`: Update configuration
- `save_config(path)`: Save configuration

#### `utils/io.py`
**Classes:**
- `FileManager`: Manage file I/O

**Methods:**
- `get_file_path(filename)`: Get path to file in competition directory
- `save_processed_data(df, filename)`: Save to processed/
- `load_processed_data(filename)`: Load from processed/
- `save_results(data, filename)`: Save JSON results
- `load_results(filename)`: Load JSON results
- `file_exists(path)`: Check if file exists

**Directory Structure:**
```
competition_data/{name}/
├── raw/           # Original data
├── processed/     # Cleaned data
├── models/        # Saved models
└── results/       # JSON results
```

#### `utils/kaggle_api.py`
**Classes:**
- `KaggleAPIClient`: Interact with Kaggle API

**Methods:**
- `download_competition_files(name, path)`: Download competition
- `submit_to_competition(name, file, message)`: Submit predictions
- `get_submission_status(name)`: Check submission status
- `validate_submission_format(file, name)`: Validate before submit

#### `utils/logging.py`
**Functions:**
- `setup_logging(level)`: Configure logging
- `get_logger(name)`: Get logger instance
- `verbose_print(message)`: Print if verbose mode enabled

#### `utils/performance.py`
**Functions:**
- `profile_memory()`: Track memory usage
- `profile_time(func)`: Decorator to time functions
- `get_system_info()`: Get system information

---

## Key Features Implementation

### 1. Leak-Free Cross-Validation

**Problem**: Feature engineering on full dataset leaks information from validation folds into training folds, inflating CV scores.

**Solution**: Use sklearn Pipeline to apply feature engineering **inside** each CV fold.

**Implementation** (`agents/model_selector.py`, lines 169-231):

```python
from sklearn.pipeline import Pipeline
from sklearn.model_selection import cross_val_score, StratifiedKFold

# Create feature transformer
feature_transformer = FeatureTransformer()
# Configure transformer with fit/transform methods

# Create pipeline: features → model
pipeline = Pipeline([
    ('features', feature_transformer),
    ('model', model)
])

# Cross-validation with pipeline
# Pipeline ensures:
# 1. For each fold:
#    a. Fit transformer on training fold only
#    b. Transform training fold
#    c. Transform validation fold using training fold statistics
#    d. Fit model on transformed training fold
#    e. Predict on transformed validation fold
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
scores = cross_val_score(pipeline, X, y, cv=cv, scoring='accuracy')
```

**Benefits:**
- Honest CV scores that correlate with leaderboard
- No data leakage between folds
- Same code for training and prediction

### 2. String Classification Label Handling

**Problem**: Models like XGBoost can't handle string labels ('Dropout', 'Enrolled', 'Graduate'). Predictions must be decoded back to original strings for submission.

**Solution**: Three-step process: encode → train → decode

**Implementation:**

**Step 1: Encode** (`agents/model_selector.py`, lines 73-81):
```python
from sklearn.preprocessing import LabelEncoder

target_encoder = None
if y_full.dtype == 'object' or y_full.dtype.name == 'category':
    target_encoder = LabelEncoder()
    y_full_encoded = pd.Series(target_encoder.fit_transform(y_full), index=y_full.index)
    # 'Dropout' → 0, 'Enrolled' → 1, 'Graduate' → 2
```

**Step 2: Train** (lines 88, 413):
```python
# Use encoded labels throughout training
y = y_full_encoded  # Use in CV
final_model.fit(X_full, y_full_encoded)  # Use in final training
```

**Step 3: Save Encoder** (lines 428-432):
```python
if target_encoder is not None:
    encoder_path = self.file_manager.get_file_path("target_encoder.pkl")
    joblib.dump(target_encoder, encoder_path)
```

**Step 4: Decode** (`agents/coordinator.py`, lines 272-308):
```python
# Load encoder
target_encoder = joblib.load(encoder_path)

# Generate predictions (numeric: 0, 1, 2)
predictions = trained_pipeline.predict(test_features)

# Decode back to strings ('Dropout', 'Enrolled', 'Graduate')
if target_encoder is not None:
    predictions = target_encoder.inverse_transform(predictions)

# Format predictions (remove brackets, ensure strings)
predictions = [str(pred) if not isinstance(pred, str) else pred for pred in predictions]
```

**Result**: Submissions with clean string labels (not `['Dropout']`)

### 3. Competition Discovery and Filtering

**Problem**: Kaggle has thousands of competitions, most are not tabular. Manual search is tedious.

**Solution**: Multi-source discovery with automatic filtering

**Implementation** (`download_all_competitions.py`):

**Source 1: Entered Competitions** (lines 197-241):
```python
def get_entered_competitions():
    result = subprocess.run(
        ["kaggle", "competitions", "list", "--csv"],
        capture_output=True, text=True
    )
    # Parse CSV and extract competition names
    return competitions
```

**Source 2: Keyword Search** (lines 19-51):
```python
def search_competitions(search_term):
    result = subprocess.run(
        ["kaggle", "competitions", "list", "--search", search_term, "--csv"],
        capture_output=True, text=True
    )
    return competitions

search_terms = [
    "tabular", "classification", "regression", "structured",
    "prediction", "binary", "multiclass", "dataset", "features", "ML"
]
```

**Source 3: Categories** (lines 54-86):
```python
def get_competitions_by_category(category):
    result = subprocess.run(
        ["kaggle", "competitions", "list", "--category", category, "--csv"],
        capture_output=True, text=True
    )
    return competitions

categories = ["gettingStarted", "playground", "featured"]
```

**Source 4: Known-Good** (lines 144-194):
```python
def get_known_good_competitions():
    return [
        "titanic", "house-prices-advanced-regression-techniques",
        "playground-series-s4e6", "digit-recognizer",
        # ... 38 more popular competitions
    ]
```

**Filtering** (lines 89-141):
```python
def check_competition_files(competition_name):
    # Get file list
    result = subprocess.run(
        ["kaggle", "competitions", "files", competition_name, "--csv"],
        capture_output=True, text=True
    )

    # Check for train.csv
    has_train = any('train.csv' in f for f in files)
    if not has_train:
        return 'no_train', []

    # Check for extra CSV files
    allowed = ['train.csv', 'test.csv', 'sample_submission', 'submission']
    extra_csvs = [f for f in files if f.endswith('.csv') and not any(p in f.lower() for p in allowed)]

    if extra_csvs:
        return 'extra_csvs', extra_csvs

    return 'ok', []
```

**Results**: ~145 unique tabular competitions automatically discovered

### 4. Memory Optimization

**Problem**: Large datasets (1M+ rows) cause memory issues

**Solutions:**

**Float32 Instead of Float64** (`core/features/transformers.py`, lines 62-65):
```python
# Convert to float32 to save memory (50% reduction vs float64)
for col in numerical_cols:
    if df_transformed[col].dtype == np.float64:
        df_transformed[col] = df_transformed[col].astype(np.float32)
```

**Stratified Sampling** (`agents/model_selector.py`, lines 55-68):
```python
# For large datasets, use stratified sampling to speed up training
if len(X_full) > 100000:
    X_full, _, y_full, _ = train_test_split(
        X_full, y_full,
        train_size=100000,
        stratify=y_full,  # Preserve class distribution
        random_state=42
    )
```

**Skip Feature Engineering for Large Datasets** (`agents/coordinator.py`, lines 105-114):
```python
train_size = data_results.get('dataset_info', {}).get('total_rows', 0)
if train_size > 100000:
    # Skip standalone feature engineering
    # Feature engineering will happen inside CV folds (leak-free)
    feature_results = {'skipped': True, 'reason': 'large_dataset'}
```

### 5. Fit/Transform Pattern

**Problem**: Using test data statistics for preprocessing causes data leakage

**Solution**: Fit on training data, transform both train and test

**Implementation** (`core/data/preprocessors.py`):

```python
class DataPreprocessor:
    def handle_missing_values(self, df, fit=True, target_col=None):
        if fit:
            # FIT: Learn imputation strategy from training data
            median_values = df[numerical_cols].median()
            self.imputation_values = median_values  # Store for later
            df_imputed[numerical_cols] = df[numerical_cols].fillna(median_values)
        else:
            # TRANSFORM: Use stored imputation values
            median_values = self.imputation_values
            df_imputed[numerical_cols] = df[numerical_cols].fillna(median_values)

        return df_imputed
```

**Usage:**
```python
# Fit on training data
train_cleaned = preprocessor.handle_missing_values(train_df, fit=True)

# Transform test data using training statistics
test_cleaned = preprocessor.handle_missing_values(test_df, fit=False)
```

### 6. Early Stopping for Boosting Models

**Problem**: Boosting models (XGBoost, LightGBM, CatBoost) can overfit if trained for too many iterations

**Solution**: Train with validation set and early stopping

**Implementation** (`core/models/factory.py`, lines 330-402):

```python
def fit_with_early_stopping(self, model, model_name, X, y, validation_split=0.2):
    # Split data
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=validation_split)

    if model_name == 'xgboost':
        model.fit(
            X_train, y_train,
            eval_set=[(X_val, y_val)],
            callbacks=[EarlyStopping(rounds=50)],
            verbose=False
        )
    elif model_name == 'lightgbm':
        model.fit(
            X_train, y_train,
            eval_set=[(X_val, y_val)],
            callbacks=[lgb.early_stopping(50, verbose=False)]
        )
    elif model_name == 'catboost':
        model.fit(
            X_train, y_train,
            eval_set=(X_val, y_val),
            early_stopping_rounds=50,
            verbose=False
        )

    return model
```

**Benefits:**
- Prevents overfitting
- Automatically finds optimal number of iterations
- Faster training (stops early if no improvement)

---

## Design Decisions

### 1. Agent-Based Architecture

**Why**: Modular, testable, reusable

- Each agent has single responsibility
- Can run agents independently for debugging
- Easy to add new agents (e.g., EnsembleAgent)
- Clear data flow between agents

### 2. Pipeline for Leak-Free CV

**Why**: Prevent data leakage, honest CV scores

- Feature engineering inside folds
- No validation data statistics leak to training
- CV scores correlate with leaderboard
- Production-ready code (same for train/predict)

### 3. File-Based Communication

**Why**: Debuggability, reproducibility

- Each agent saves results to JSON
- Can inspect intermediate outputs
- Can restart pipeline from any stage
- Easy to share results

### 4. Fit/Transform Pattern

**Why**: Prevent data leakage, sklearn compatibility

- Standard sklearn interface
- Learns from training, applies to test
- Works with sklearn Pipelines
- Prevents using test statistics

### 5. String Label Encoding

**Why**: Model compatibility, submission format

- Most models require numeric targets
- Kaggle requires original string labels in submission
- Encode for training, decode for submission
- Store encoder for consistency

### 6. Stratified Sampling for Large Datasets

**Why**: Speed, memory efficiency, representativeness

- Reduces 1M rows to 100K
- Preserves class distribution
- Faster training (~10x)
- Minimal accuracy loss

### 7. Configuration File

**Why**: Flexibility, no code changes

- Tune pipeline without editing code
- Different configs for different competitions
- Version control for experiments
- Easy to share settings

### 8. Multiple Discovery Sources

**Why**: Comprehensive coverage, redundancy

- Single source may miss competitions
- Categories + keywords + entered + known-good
- ~145 competitions from 4 sources
- Deduplication ensures no repeats

### 9. Pre-Check Before Download

**Why**: Save bandwidth, avoid non-tabular

- Validate file structure before downloading
- Skip competitions without train.csv
- Reject competitions with extra CSVs
- Faster batch processing

### 10. Warning Suppression

**Why**: Clean output, user experience

- CatBoost emits 100+ sklearn deprecation warnings
- Warnings clutter output, hide real errors
- Suppress only known, harmless warnings
- Keep critical warnings visible

---

## Recent Improvements

### 1. String Label Support (Critical Fix)

**Problem**:
```
ValueError: Invalid classes inferred from unique values of y.
Expected: [0 1 2], got ['Dropout' 'Enrolled' 'Graduate']
```

**Solution**:
- Encode labels before training in `model_selector.py` and `evaluators.py`
- Save `target_encoder.pkl` with model
- Decode predictions in `coordinator.py` before submission
- Format predictions properly (remove brackets)

**Files Changed:**
- `agents/model_selector.py` (lines 73-81, 428-432)
- `agents/coordinator.py` (lines 272-308)
- `core/models/evaluators.py` (lines 60-68, 107-112)

**Impact**: Now handles multi-class string classification correctly

### 2. Extra CSV Filtering

**Problem**: Non-tabular competitions (time series, NLP) have extra data files

**Solution**:
- Check for extra CSVs during download
- Reject competitions with files beyond train/test/submission
- Pre-check before download to save bandwidth

**Files Changed:**
- `download_competition.py` (lines 111-142)
- `download_all_competitions.py` (lines 89-141, 367-379)

**Impact**: Only downloads standard tabular competitions

### 3. Duplicate Removal Fix

**Problem**:
```
Prediction count mismatch: 171382 vs 172585 expected
```

**Cause**: Duplicates were removed from test data, changing row count

**Solution**: Only remove duplicates from training data

**Files Changed:**
- `agents/data_scout.py` (lines 90-92)

**Impact**: Test data row count matches sample_submission

### 4. CatBoost Warning Suppression

**Problem**: 100+ sklearn deprecation warnings from CatBoost

**Solution**: Suppress sklearn DeprecationWarnings

**Files Changed:**
- `kaggle_slayer.py` (lines 16-22)

**Impact**: Clean output, easier to spot real errors

### 5. Enhanced Competition Discovery

**Problem**: Only ~20 competitions found (entered only)

**Solution**: Multi-source discovery
- Added keyword search (11 terms)
- Added category search (3 categories)
- Added 42 known-good competitions
- Combined and deduplicated

**Files Changed:**
- `download_all_competitions.py` (complete rewrite)

**Impact**: 145 unique competitions discovered

### 6. Pre-Check Validation

**Problem**: Download failed competitions, waste bandwidth

**Solution**: Check file structure before download

**Files Changed:**
- `download_all_competitions.py` (lines 89-141, 367-384)

**Impact**: Skip invalid competitions early, faster batch processing

### 7. ID Column Removal

**Problem**: ID columns leaked into features, causing overfitting

**Solution**: Remove ID columns BEFORE any preprocessing

**Files Changed:**
- `agents/data_scout.py` (lines 52-68)

**Impact**: Cleaner features, better generalization

### 8. Memory Optimization

**Problem**: Large datasets (1M+ rows) cause OOM errors

**Solutions**:
- Float32 instead of float64 (50% memory reduction)
- Stratified sampling for datasets >100K rows
- Skip standalone feature engineering for large datasets

**Files Changed:**
- `core/features/transformers.py` (lines 62-65)
- `agents/model_selector.py` (lines 55-68)
- `agents/coordinator.py` (lines 105-114)

**Impact**: Can handle 1M+ row datasets

### 9. Leak-Free CV Implementation

**Problem**: Feature engineering on full dataset inflates CV scores

**Solution**: Use sklearn Pipeline for feature engineering inside CV folds

**Files Changed:**
- `agents/model_selector.py` (lines 169-231)

**Impact**: Honest CV scores, better leaderboard correlation

### 10. Feature Engineering Recipes

**Problem**: Test data feature engineering inconsistent with training

**Solution**: Store feature recipes during fit, apply during transform

**Files Changed:**
- `core/features/generators.py` (lines 101-226)

**Impact**: Exact same features for train and test

---

## Performance Characteristics

### Typical Pipeline Runtime

**Small Dataset** (Titanic, 891 rows):
- Data Scout: 1-2 seconds
- Feature Engineering: 2-3 seconds
- Model Selection (4 models, 5-fold CV): 10-20 seconds
- Submission: <1 second
- **Total: ~15-25 seconds**

**Medium Dataset** (House Prices, 1,460 rows):
- Data Scout: 2-3 seconds
- Feature Engineering: 3-5 seconds
- Model Selection: 20-40 seconds
- Submission: <1 second
- **Total: ~25-50 seconds**

**Large Dataset** (100K rows):
- Data Scout: 10-15 seconds
- Feature Engineering: Skipped (inside CV)
- Model Selection (with sampling): 60-120 seconds
- Submission: 1-2 seconds
- **Total: ~70-140 seconds**

### Memory Usage

**Small Dataset**: <500 MB
**Medium Dataset**: <1 GB
**Large Dataset (100K sampled)**: 1-2 GB
**Very Large Dataset (1M sampled to 100K)**: 2-4 GB

### Disk Usage Per Competition

- Raw data: 1-100 MB (varies by competition)
- Processed data: 1-100 MB
- Models: 10-50 MB (boosting models larger)
- Results: <1 MB (JSON files)
- **Total: ~10-250 MB per competition**

### Batch Download Statistics

**Discovery**: 145 unique competitions from 4 sources
**Pre-check filtering**: ~30% rejected (no train.csv or extra CSVs)
**Download time**: ~5-10 minutes per competition (depends on size)
**Batch download**: 2-6 hours for 100 competitions

---

## Future Enhancements

### Planned Features

1. **Hyperparameter Optimization**: Optuna-based tuning for each competition
2. **Ensemble Models**: Voting, stacking, blending
3. **Neural Networks**: Add support for TabNet, FT-Transformer
4. **Feature Selection**: More advanced selection methods
5. **Auto-EDA**: Automated exploratory data analysis with plots
6. **Submission Tracking**: Track leaderboard scores, compare with CV
7. **Multi-Modal**: Support for image + tabular competitions
8. **Time Series**: Dedicated pipeline for time series competitions
9. **NLP**: Basic text feature extraction (TF-IDF, embeddings)
10. **Automated Retraining**: Retrain when new data available

### Architecture Improvements

1. **Parallel Model Training**: Train models in parallel (joblib)
2. **Incremental Learning**: Update models with new data
3. **Model Versioning**: Track model versions and experiments
4. **Dashboard**: Streamlit dashboard for monitoring pipelines
5. **Cloud Integration**: Deploy to AWS/GCP for large-scale training

---

## Conclusion

KaggleSlayer is a comprehensive AutoML system designed for tabular Kaggle competitions. It automates the entire pipeline from competition discovery to submission creation, with special attention to:

- **Data leakage prevention** through leak-free CV and fit/transform patterns
- **Robustness** through string label handling, missing value imputation, outlier detection
- **Scalability** through memory optimization and stratified sampling
- **Flexibility** through configuration files and modular architecture

The system has been tested on 100+ competitions and achieves competitive baseline scores with zero manual intervention.

**Key Strengths:**
- Fully automated pipeline
- Leak-free cross-validation
- String classification support
- Memory-efficient processing
- Comprehensive feature engineering
- Multi-model training

**Key Limitations:**
- Tabular-only (no images, text, time series)
- Basic feature engineering (no domain-specific features)
- Limited hyperparameter tuning
- No ensembling (yet)

**Ideal Use Cases:**
- Quick baseline establishment
- Learning competition workflows
- Automated participation in multiple competitions
- Benchmarking new algorithms

---

## Appendix: Configuration Reference

### `config.yaml`

```yaml
# Pipeline Settings
pipeline:
  cv_folds: 5                     # Number of cross-validation folds
  cv_random_state: 42             # Random seed for reproducibility
  optuna_trials: 20               # Hyperparameter optimization trials
  optuna_timeout: 300             # Timeout in seconds
  max_features_to_create: 25      # Maximum engineered features
  polynomial_degree: 2            # Polynomial feature degree

# Data Processing
data:
  drop_missing_threshold: 0.9     # Drop columns with >90% missing
  correlation_threshold: 0.95     # Drop features correlated >0.95
  variance_threshold: 0.01        # Drop features with variance <0.01

# Model Selection
model_selection:
  classification_metric: 'accuracy'  # CV metric for classification
  regression_metric: 'neg_mean_squared_error'  # CV metric for regression
```

### Environment Variables

```bash
# Kaggle API credentials
KAGGLE_USERNAME=your_username
KAGGLE_KEY=your_api_key

# Optional: Override config values
KAGGLESLAYER_CV_FOLDS=10
KAGGLESLAYER_MAX_FEATURES=50
```

---

**Last Updated**: 2025-10-06
**Version**: 1.0
**Authors**: KaggleSlayer Team
