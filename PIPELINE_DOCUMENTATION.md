# KaggleSlayer Pipeline Documentation

## Complete End-to-End AutoML Pipeline

This document provides a comprehensive overview of the KaggleSlayer pipeline from data ingestion through model selection.

---

## **Pipeline Architecture**

The pipeline consists of **4 main stages** coordinated by `PipelineCoordinator`:

```
Data Loading → Data Scout → Feature Engineering → Model Selection → Submission
```

---

## **Stage 1: Data Scout** 📊

**Agent**: `DataScoutAgent` (`agents/data_scout.py`)

### Purpose
Explore, validate, clean, and prepare raw competition data for feature engineering.

### Components Used

#### 1. **CompetitionDataLoader** (`core/data/loaders.py`)
- Loads `train.csv` and `test.csv` from competition directory
- Searches in both `raw/` subdirectory and root
- **Target detection**: Identifies target column by finding column present in train but not test
- **Feature type analysis**: Categorizes each column as:
  - `identifier`: High uniqueness (>95%) + name contains 'id'/'index'
  - `binary`: Exactly 2 unique values
  - `ordinal`: Sequential integers with low cardinality
  - `categorical_numeric`: Low cardinality numeric (<20 unique, <5% of total)
  - `numerical`: Standard numeric features
  - `categorical`: Object type with moderate cardinality
  - `high_cardinality_text`: >90% unique text values
  - `text`: 50-90% unique text values
  - `date_string`: >50% parseable as dates
  - `datetime`: Already datetime type

#### 2. **DataValidator** (`core/data/validators.py`)
Comprehensive validation checks:
- Empty dataset detection
- Duplicate column names
- Data type consistency (numeric stored as object)
- Missing data patterns (>50% missing flagged)
- Infinite values in numerical columns
- Constant (zero variance) features
- **Train/test consistency**: Column alignment, dtype matching, value range comparison

#### 3. **DataPreprocessor** (`core/data/preprocessors.py`)

**Operations (in order)**:

1. **DateTime Parsing & Extraction**
   - Auto-detects datetime columns (>50% parseable as dates)
   - Extracts 11 temporal features per datetime column:
     - Basic: year, month, day, dayofweek, quarter, is_weekend
     - Cyclical encodings: month_sin, month_cos, dayofweek_sin, dayofweek_cos
   - Drops original datetime columns

2. **Missing Value Handling**
   - **Missingness indicators**: Creates `{col}_was_missing` binary flags for columns with 5-80% missing
   - **Column dropping**: Removes columns with >80% missing (configurable)
   - **Numerical imputation**: Median imputation using `SimpleImputer` (fit/transform pattern)
   - **Categorical imputation**: Fills with explicit `'MISSING_VALUE'` category

3. **Duplicate Removal**
   - Drops exact duplicate rows
   - Reports count of duplicates removed

4. **Outlier Handling**
   - Method: Winsorization (clipping to 1st-99th percentile)
   - Excludes target column from outlier handling
   - Prevents data loss while handling extreme values

### Output Files
- `train_cleaned.csv`: Cleaned training data
- `test_cleaned.csv`: Cleaned test data (if available)
- `data_scout_results.json`: Validation results, statistics, recommendations

### Key Features
✅ **Fit/transform pattern**: Train data fitted, test data transformed
✅ **Preserves information**: Missingness indicators + explicit MISSING category
✅ **Temporal features**: Cyclical encoding for seasonality
✅ **Robust outlier handling**: Winsorization instead of removal

---

## **Stage 2: Feature Engineering** 🔧

**Agent**: `FeatureEngineerAgent` (`agents/feature_engineer.py`)

### Architecture: 4-Phase Pipeline

The pipeline has been **reordered for efficiency** - cheap filtering happens BEFORE expensive generation.

---

### **Phase 1: Early Filtering** (Lines 55-75)

**Purpose**: Remove bad features BEFORE expensive operations

**Operations**:
1. **Low Variance Removal**
   - Uses `VarianceThreshold` (threshold=0.01)
   - Removes features with near-zero variance
   - Applied to numerical features only

2. **Distribution Stability Check** (if test data available)
   - Compares train/test distributions using:
     - **Mean difference**: Normalized by train std (threshold: 0.5)
     - **Std ratio**: Max/min ratio (threshold: 2.0)
     - **KS test**: Kolmogorov-Smirnov p-value (threshold: 0.01)
   - Removes features that are unstable across train/test
   - **NEW**: Uses scipy.stats.ks_2samp for rigorous statistical testing

3. **Correlation Removal**
   - Removes features with correlation > 0.95 (configurable)
   - Uses upper triangle to avoid duplicates

**Why early?** Prevents generating polynomial/interaction features from useless base features.

---

### **Phase 2: Feature Generation** (Lines 77-119)

**Components**: `FeatureGenerator` (`core/features/generators.py`)

#### **New Auto-ID Detection** ✨
- No longer uses hardcoded list `['PassengerId', 'id', ...]`
- Uses `detect_id_columns()` utility:
  - Checks uniqueness >95%
  - Checks correlation with row index >0.95
  - Checks for 'id'/'index' in column name
- Applied consistently across all generation methods

#### **A. Numerical Features** (max 25 features)
- **Ratio features**: `col1 / col2` (with zero handling, scale checks)
- **Product features**: `col1 * col2` (with scale limits to prevent explosion)
- **Log transformation**: `log1p(col)` for skewed data (skew > 1)
- **Square root**: `sqrt(col)` for high variance data

#### **B. Categorical Features**
For **non-high-cardinality** (<10% unique):
- **Frequency encoding**: Normalized value counts (fit/transform aware)
- **Rarity encoding**: Inverse of frequency
- **Missing indicators**: Binary flags for missing values

For **high-cardinality** (>10% unique):
- **String length**: Character count
- **Word count**: Number of words
- **Unique characters**: Set size
- High-cardinality columns **dropped** after extraction

#### **C. Target Encoding** ✨ NEW
Applied to high-cardinality categoricals (>10 unique values):
- **Smoothed mean encoding**: Prevents overfitting
  - Formula: `(count × category_mean + smoothing × global_mean) / (count + smoothing)`
  - Default smoothing: 1.0
- **Fit/transform pattern**: Train fitted, test uses stored mappings
- **Unseen categories**: Filled with global mean

#### **D. Statistical Features** (requires 3+ numerical cols)
Row-wise aggregations:
- sum, mean, std, min, max, range
- Created across all numerical features

#### **E. Binning/Discretization** ✨ NEW
- Uses `KBinsDiscretizer` with quantile strategy
- Only bins columns with:
  - High skewness (>0.5) OR high variance (std > mean)
  - Enough unique values (>n_bins)
- Limit: Top 10 columns to prevent explosion
- Creates ordinal bins (0, 1, 2, ..., n_bins-1)

#### **F. Clustering Features** ✨ NEW
- KMeans clustering (default: 5 clusters)
- Features generated:
  - `cluster_label`: Cluster assignment
  - `cluster_dist_0` to `cluster_dist_N`: Distance to each centroid
  - `cluster_min_dist`: Distance to nearest centroid
- Uses top 10 features by variance to prevent overfitting
- Fit/transform pattern for train/test

#### **G. Polynomial Features** (Last!)
- Degree 2 polynomial combinations
- Adaptive column limit (3-5 cols based on degree)
- Uses `SelectKBest` to pick most important columns first
- Sanitizes feature names (`^` → `_pow_`, spaces → `_`)

#### **H. Time-Series Features** ✨ NEW (Available but not auto-applied)
For temporal data:
- **Lag features**: 1, 2, 3, 7 periods
- **Rolling statistics**: mean/std over windows (3, 7, 14)
- **Difference features**: First difference, percent change

**Total Features Created**: Typically 50-150 depending on data

---

### **Phase 3: Final Feature Selection** (Lines 121-155)

**Component**: `FeatureSelector` (`core/features/selectors.py`)

**Univariate Selection**:
- Uses `SelectKBest` with F-test (f_classif or f_regression)
- **Adaptive k**: Defaults to 50% of features (min 10, max 100)
- Scores all features and keeps top k

**Test Data Handling**:
- Applies same variance filter (transform mode)
- Aligns features with training data
- **FIXED**: Uses `is_numeric_dtype()` instead of `in [np.number]`
- Fills missing features with median (numerical) or 0 (categorical)

---

### **Phase 4: Feature Transformation** (Lines 157-170)

**Component**: `FeatureTransformer` (`core/features/transformers.py`)

**Order of operations**:

1. **Imputation** (method='simple')
   - Numerical: Median via `SimpleImputer`
   - Categorical: Most frequent value

2. **Categorical Encoding** ✨ **IMPROVED** (method='auto')

   **Auto-selection strategy**:
   - **Low cardinality** (≤5 unique): **One-hot encoding**
     - Creates binary columns per category
     - Sparse=False for better compatibility
     - Handles unseen categories with `handle_unknown='ignore'`

   - **Medium/high cardinality** (>5 unique): **Label encoding**
     - Integer encoding (0, 1, 2, ...)
     - Handles unseen categories by mapping to most frequent class

   - **Very high cardinality** (>10 unique): Uses **target encoding** from Phase 2

3. **Numerical Scaling** (method='standard')
   - StandardScaler (mean=0, std=1)
   - Excludes target column
   - Fit/transform pattern

### Output Files
- `train_engineered.csv`: Final feature matrix for training
- `test_engineered.csv`: Final feature matrix for prediction
- `feature_engineer_results.json`: Contains:
  - Feature generation report (all created features)
  - Feature selection report (scores, top features, distribution stats, KS test results)
  - Transformation report (scalers/encoders fitted)
  - **Performance report** (memory usage, time per phase)

### Key Features
✅ **Reordered pipeline**: Early filtering before expensive generation
✅ **Auto-ID detection**: No hardcoded lists
✅ **Target encoding**: Smoothed encoding for high-cardinality
✅ **Smart categorical encoding**: Auto-selects strategy by cardinality
✅ **Clustering features**: KMeans-based distance features
✅ **Binning features**: Quantile discretization for skewed data
✅ **KS test**: Rigorous distribution stability testing
✅ **Memory monitoring**: Tracks RAM and time per phase

---

## **Stage 3: Model Selection** 🤖

**Agent**: `ModelSelectorAgent` (`agents/model_selector.py`)

### Components Used

#### 1. **ModelFactory** (`core/models/factory.py`)

**Available Models**:
- **Sklearn**: RandomForest, ExtraTrees, LogisticRegression, Ridge, Lasso, ElasticNet, KNN, SVM/SVR
- **XGBoost**: XGBClassifier, XGBRegressor (if installed)
- **LightGBM**: LGBMClassifier, LGBMRegressor (if installed)
- **CatBoost**: CatBoostClassifier, CatBoostRegressor (if installed)

**Default Parameters**: Tuned for good baseline performance
- Example (XGBoost): 200 estimators, max_depth=6, lr=0.05, subsample=0.8

**Verbosity Control** ✨ **IMPROVED**:
- `_apply_verbosity_defaults()` method auto-silences all models:
  - CatBoost: `verbose=False`
  - LightGBM: `verbose=-1`
  - XGBoost: `verbosity=0`
- Applied during model creation, even in hyperparameter optimization

#### 2. **ModelEvaluator** (`core/models/evaluators.py`)

**Evaluation Method**:
- **Cross-validation**: 5-fold (configurable)
- **Stratified** for classification, **KFold** for regression
- Returns: CV mean, CV std, individual fold scores

**Metrics**:
- Classification: Accuracy (f_classif)
- Regression: Negative MSE (f_regression)

#### 3. **HyperparameterOptimizer** (`core/models/optimizers.py`)

**Method**: Optuna with TPE sampler
- **Trials**: 20 (configurable)
- **Timeout**: 300s (5 minutes)
- **CV folds**: 5
- **Silent**: `show_progress_bar=False`

**Fallback**: Grid search if Optuna unavailable (limits to 50 combinations)

#### 4. **EnsembleBuilder** (`core/models/ensembles.py`)

**Ensemble Types**:
1. **Stacking Ensemble**
   - Base models: Top 3 models
   - Meta-learner: LogisticRegression (classification) or Ridge (regression)
   - Uses CV predictions as meta-features

2. **Voting Ensemble**
   - Soft voting (average probabilities for classification)
   - Average predictions (for regression)

**Selection**: Compares stacking vs voting, uses better performing one

### Pipeline Workflow

1. **Load Data**: Engineered train data from Phase 2
2. **Detect Problem Type**: Auto-detect classification (<20 unique) vs regression
3. **Adaptive Sampling** (for faster evaluation):
   - <10K rows: No sampling
   - 10K-100K rows: Sample to 10K
   - >100K rows: Sample to 5K
4. **Evaluate All Models**: Default parameters, 5-fold CV
5. **Select Top 3**: Based on CV scores
6. **Optimize Best Model**: Hyperparameter tuning with Optuna (20 trials, 5 min timeout)
7. **Create Ensembles**: Both stacking and voting
8. **Select Final Model**: Best of (optimized_model, stacking, voting)
9. **Retrain on Full Data**: Using best configuration
10. **Save Model**: Both specific (`best_model_{name}.pkl`) and generic (`best_model.pkl`)

### Output Files
- `best_model.pkl`: Final trained model (ensemble or single)
- `best_model_{name}.pkl`: Specific model file
- `model_selector_results.json`: Contains:
  - Problem type, target column
  - Models evaluated count
  - Best model name and score
  - Optimized parameters
  - Ensemble performance
  - Model comparison table
  - Data shape info

### Key Features
✅ **Adaptive sampling**: Faster evaluation on large datasets
✅ **Auto problem type detection**: No manual specification
✅ **Ensemble comparison**: Stacking vs voting, uses best
✅ **Silent training**: All models silenced (CatBoost, LightGBM, XGBoost)
✅ **Fallback retraining**: If saved model not found, retrains automatically

---

## **Stage 4: Submission Creation** 📤

**Coordinator Method**: `create_submission()` (`agents/coordinator.py:131-226`)

### Process

1. **Load Target Info**: Gets target column from data scout results
2. **Find Submission Template**:
   - Searches for `*submission*.csv` in `raw/` and root
   - If found, uses that structure (ID column + target column)
   - If not found, extracts ID column from original test.csv

3. **Load Test Features**: Engineered test data from Phase 2
4. **Generate Predictions**: Uses `model_selector.predict_with_best_model()`
5. **Format Predictions**:
   - Classification: Cast to int
   - Regression: Keep as float

6. **Create Submission DataFrame**:
   - Column 1: ID column (from sample submission or test.csv)
   - Column 2: Target column with predictions

7. **Save**: Writes to `competition_path/submission.csv`

### Kaggle Submission (Optional)

**Method**: `submit_to_kaggle()` (`agents/coordinator.py:228-276`)

Uses `KaggleAPIClient` (`utils/kaggle_api.py`):
- Validates submission format
- Creates message: `"KaggleSlayer submission - Model: {name}, CV Score: {score}"`
- Submits via Kaggle API
- Checks submission status

---

## **Utility Modules**

### **core/features/utils.py** ✨ NEW

**1. FeatureEngineeringMonitor**
- Tracks memory usage (MB) at each checkpoint
- Tracks elapsed time per phase
- Uses `psutil` for process memory monitoring

**2. detect_id_columns()**
- Auto-detects ID columns using:
  - Uniqueness ratio >95%
  - Correlation with index >0.95
  - Name patterns ('id', 'index', 'key')

**3. is_numeric_dtype()**
- Proper numeric dtype checking using `np.issubdtype()`

**4. auto_detect_problem_type()**
- Uses dtype + unique count + unique ratio
- More sophisticated than simple `nunique() < 20`

### **utils/performance.py**

**PerformanceTimer**: Context manager and decorator for timing
**PerformanceProfiler**: Tracks metrics across operations (count, total/avg/min/max time)
**Global profiler**: Singleton pattern for app-wide profiling

### **utils/io.py**

**FileManager**: Centralized file I/O
- Handles paths for: raw/, processed/, models/, results/
- Save/load: CSV, JSON, pickle (models)
- Ensures directory structure

### **utils/config.py**

**ConfigManager**: Loads and manages `config.yaml`
- Nested dict access with defaults
- Used throughout pipeline for thresholds, hyperparameters

---

## **Pipeline Execution Flow**

```
┌─────────────────────────────────────────────────────────────┐
│ PipelineCoordinator.run()                                    │
└─────────────────────────────────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────────┐
│ STAGE 1: DATA SCOUT                                          │
│ • Load train.csv, test.csv                                   │
│ • Validate data quality                                      │
│ • Detect target column                                       │
│ • Analyze feature types                                      │
│ • Parse datetime → Extract 11 temporal features             │
│ • Handle missing → Indicators + imputation                   │
│ • Remove duplicates                                          │
│ • Handle outliers → Winsorization (1-99 percentile)         │
│ Output: train_cleaned.csv, test_cleaned.csv                  │
└─────────────────────────────────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────────┐
│ STAGE 2: FEATURE ENGINEERING                                 │
│                                                              │
│ PHASE 1: Early Filtering (NEW order!)                       │
│ • Remove low variance features                              │
│ • Check distribution stability (KS test)                     │
│ • Remove highly correlated features                         │
│                                                              │
│ PHASE 2: Feature Generation                                 │
│ • Auto-detect ID columns (NEW!)                             │
│ • Generate numerical features (ratios, products, log, sqrt) │
│ • Generate categorical features (frequency, rarity)         │
│ • Target encoding for high-cardinality (NEW!)               │
│ • Statistical features (row aggregations)                   │
│ • Binning features (quantile discretization) (NEW!)         │
│ • Clustering features (KMeans distances) (NEW!)             │
│ • Polynomial features (degree 2, adaptive)                  │
│                                                              │
│ PHASE 3: Final Selection                                    │
│ • Univariate selection (SelectKBest, adaptive k)            │
│ • Align train/test features                                 │
│                                                              │
│ PHASE 4: Transformation                                     │
│ • Impute missing values (median/mode)                       │
│ • Smart categorical encoding (auto-select by cardinality)   │
│ • Scale numerical features (StandardScaler)                 │
│                                                              │
│ Output: train_engineered.csv, test_engineered.csv            │
│         + performance report (memory, time)                  │
└─────────────────────────────────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────────┐
│ STAGE 3: MODEL SELECTION                                     │
│ • Load engineered data                                       │
│ • Auto-detect problem type (classification/regression)      │
│ • Adaptive sampling (10K for medium, 5K for large datasets) │
│ • Evaluate all models (RF, XGB, LightGBM, CatBoost, etc.)   │
│ • Select top 3 models                                        │
│ • Optimize best model (Optuna, 20 trials, 5 min)           │
│ • Create ensembles (stacking + voting)                      │
│ • Select final model (best of single/ensemble)              │
│ • Retrain on full data                                      │
│ • Save model                                                │
│ Output: best_model.pkl, model_selector_results.json          │
└─────────────────────────────────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────────┐
│ STAGE 4: SUBMISSION                                          │
│ • Find sample submission or detect ID column                │
│ • Load engineered test features                             │
│ • Generate predictions with best model                      │
│ • Format submission (ID + predictions)                      │
│ • Save submission.csv                                       │
│ • [Optional] Submit to Kaggle via API                       │
│ Output: submission.csv                                       │
└─────────────────────────────────────────────────────────────┘
```

---

## **Configuration** (`config.yaml`)

```yaml
pipeline:
  cv_folds: 5                    # Cross-validation folds
  cv_random_state: 42            # Random seed
  optuna_trials: 20              # Hyperparameter tuning trials
  optuna_timeout: 300            # 5 minutes
  max_features_to_create: 25     # Limit for numerical feature generation
  polynomial_degree: 2           # Polynomial feature degree

data:
  drop_missing_threshold: 0.8    # Drop columns with >80% missing
  correlation_threshold: 0.95    # Remove features with >0.95 correlation
  variance_threshold: 0.01       # Minimum variance threshold
```

---

## **Key Improvements Summary**

### **Data Scout**
✅ Cyclical encoding for temporal features (seasonality)
✅ Missingness indicators preserve information
✅ Winsorization instead of outlier removal
✅ Comprehensive validation with train/test consistency checks

### **Feature Engineering**
✅ **Reordered pipeline**: Early filtering BEFORE expensive generation
✅ **Auto-ID detection**: No hardcoded lists, uses heuristics
✅ **Target encoding**: Smoothed mean encoding for high-cardinality
✅ **Smart categorical encoding**: Auto-selects one-hot vs label by cardinality
✅ **Clustering features**: KMeans-based distance features
✅ **Binning features**: Quantile discretization for skewed distributions
✅ **KS test**: Rigorous statistical test for distribution stability
✅ **Memory monitoring**: Tracks RAM usage and time per phase
✅ **Fixed dtype bug**: Uses `is_numeric_dtype()` instead of `in [np.number]`

### **Model Selection**
✅ **Silent training**: All models (CatBoost, LightGBM, XGBoost) auto-silenced
✅ **Adaptive sampling**: Faster on large datasets
✅ **Ensemble comparison**: Tests both stacking and voting
✅ **Fallback retraining**: Auto-retrains if model file missing

---

## **File Structure**

```
KaggleSlayer/
├── agents/
│   ├── base_agent.py          # Base class with logging, file management
│   ├── data_scout.py          # Stage 1: Data exploration & cleaning
│   ├── feature_engineer.py    # Stage 2: Feature engineering (4 phases)
│   ├── model_selector.py      # Stage 3: Model training & selection
│   └── coordinator.py         # Pipeline orchestration
│
├── core/
│   ├── data/
│   │   ├── loaders.py         # CSV loading, target detection, feature typing
│   │   ├── preprocessors.py   # Missing values, outliers, datetime parsing
│   │   └── validators.py      # Data quality validation
│   │
│   ├── features/
│   │   ├── generators.py      # Feature creation (numerical, categorical, clustering, etc.)
│   │   ├── selectors.py       # Feature selection (variance, correlation, univariate, KS test)
│   │   ├── transformers.py    # Imputation, encoding, scaling
│   │   └── utils.py           # Helper functions (ID detection, monitoring)
│   │
│   └── models/
│       ├── factory.py         # Model creation with default params
│       ├── evaluators.py      # Cross-validation evaluation
│       ├── optimizers.py      # Hyperparameter tuning (Optuna)
│       └── ensembles.py       # Stacking and voting ensembles
│
├── utils/
│   ├── io.py                  # File management (FileManager)
│   ├── config.py              # Configuration loading
│   ├── kaggle_api.py          # Kaggle API integration
│   ├── logging.py             # Logging utilities
│   └── performance.py         # Performance profiling
│
├── kaggle_slayer.py           # CLI entry point
├── config.yaml                # Configuration file
└── requirements.txt           # Dependencies
```

---

## **Usage Example**

```bash
# Run full pipeline
python kaggle_slayer.py titanic --data-path competition_data/titanic

# Run with Kaggle submission
python kaggle_slayer.py titanic --data-path competition_data/titanic --submit
```

**Expected Output**:
```
======================================================================
  KAGGLESLAYER PIPELINE - TITANIC
======================================================================

======================================================================
  STEP 1/4: DATA SCOUT - Exploring and Cleaning Data
======================================================================
Loading data for competition: titanic
Loaded training data from raw: (891, 12)
Loaded test data from raw: (418, 11)
Detected target column: Survived
...
[OK] Data Scout completed in 2.3s

======================================================================
  STEP 2/4: FEATURE ENGINEER - Creating Powerful Features
======================================================================
Phase 1: Early feature filtering...
Removing low variance features...
Checking train/test distribution stability (with KS test)...
Removing highly correlated features...

Phase 2: Generating new features...
Generating numerical features from 5 columns...
Generating categorical features from 3 columns...
Applying target encoding to 2 high-cardinality features
...
Phase 3: Final feature selection...
Selecting top 50 features using univariate tests...

Phase 4: Transforming features...
Auto-selecting encoding strategy for 3 categorical features...
  One-hot encoding 2 low-cardinality features
  Label encoding 1 medium/high-cardinality features
...
[OK] Feature Engineering completed in 8.7s

======================================================================
  STEP 3/4: MODEL SELECTOR - Training and Optimizing Models
======================================================================
Detected problem type: classification
Available models: ['random_forest', 'extra_trees', 'logistic_regression',
                   'xgboost', 'lightgbm', 'catboost']
Evaluating random_forest...
random_forest CV score: 0.8215
...
Optimizing hyperparameters for xgboost...
Optimization completed. Best score: 0.8456
...
Creating ensembles...
Stacking ensemble score: 0.8512
Voting ensemble score: 0.8489
Using stacking ensemble (better score)
[OK] Model Selection completed in 45.2s

======================================================================
  STEP 4/4: SUBMISSION - Creating Kaggle Submission File
======================================================================
Found sample submission: gender_submission.csv
Generated 418 predictions using ensemble_stacking
Created submission: 418 rows
Columns: ['PassengerId', 'Survived']
Saved to: competition_data/titanic/submission.csv
[OK] Submission created in 0.8s

======================================================================
  PIPELINE COMPLETED SUCCESSFULLY!
======================================================================
  Competition: titanic
  Best Model: ensemble_stacking
  CV Score: 0.8512
  Total Time: 57.0s (0.9 minutes)
  Steps: data_scout -> feature_engineer -> model_selector -> submission
======================================================================
```

---

## **Performance Characteristics**

- **Small datasets** (<10K rows): ~1-2 minutes total
- **Medium datasets** (10K-100K rows): ~5-10 minutes total
- **Large datasets** (>100K rows): ~10-20 minutes total

**Memory usage**: Typically 200-500 MB for medium datasets

---

## **Dependencies**

All required packages are in `requirements.txt`:
- Core ML: pandas, numpy, scikit-learn, scipy
- Advanced models: xgboost, lightgbm, catboost
- Optimization: optuna
- Monitoring: psutil
- Kaggle: kaggle (API)

---

## **Error Handling & Robustness**

- **Missing files**: Auto-creates directory structure
- **Missing test data**: Pipeline continues without test data
- **Missing models**: Fallback retraining if model file not found
- **Unseen categories**: Handled in all encoding methods
- **Train/test feature mismatch**: Auto-aligns features, fills missing
- **Failed model training**: Continues with other models
- **Optuna failures**: Falls back to grid search

---

## **Outputs Generated**

### Data Files
- `processed/train_cleaned.csv`
- `processed/test_cleaned.csv`
- `processed/train_engineered.csv`
- `processed/test_engineered.csv`
- `submission.csv`

### Model Files
- `models/best_model.pkl`
- `models/best_model_{name}.pkl`

### Result Files (JSON)
- `results/data_scout_results.json`
- `results/feature_engineer_results.json`
- `results/model_selector_results.json`
- `results/pipeline_results.json`

---

**End of Pipeline Documentation**
