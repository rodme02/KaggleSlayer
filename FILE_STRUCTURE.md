# KaggleSlayer File Structure Guide

Complete reference for every file and directory in the KaggleSlayer project.

---

## 📁 Root Directory

### Configuration Files

#### `config.yaml`
**Purpose:** Main configuration file for the entire pipeline
**Contains:**
- LLM model settings (model name, tokens, temperature)
- Pipeline parameters (CV folds, Optuna trials, feature limits)
- Data processing thresholds (missing data, correlation, variance)
- Logging configuration

**Usage:**
```python
from utils.config import ConfigManager
config = ConfigManager()  # Automatically loads config.yaml
```

#### `.env`
**Purpose:** Environment variables and sensitive credentials
**Contains:**
- `OPENAI_API_KEY` - OpenRouter API key for LLM
- `KAGGLE_USERNAME` - Kaggle username
- `KAGGLE_KEY` - Kaggle API key

**⚠️ Note:** This file is gitignored and should never be committed.

#### `.env.example`
**Purpose:** Template for creating `.env` file
**Usage:** Copy to `.env` and fill in your credentials

#### `.gitignore`
**Purpose:** Specifies which files Git should ignore
**Ignores:**
- Python cache (`__pycache__`, `*.pyc`)
- Virtual environments (`venv/`, `env/`)
- Data files (`competition_data/`)
- Logs (`*.log`, `llm_logs/`)
- IDE files (`.vscode/`, `.idea/`)
- System files (`.DS_Store`, `Thumbs.db`)

#### `requirements.txt`
**Purpose:** Python package dependencies
**Key Dependencies:**
- `pandas`, `numpy`, `scikit-learn` - Data processing and ML
- `xgboost`, `lightgbm`, `catboost` - Advanced ML models
- `optuna` - Hyperparameter optimization
- `kaggle` - Kaggle API integration
- `openai`, `tenacity` - LLM integration
- `streamlit`, `plotly` - Dashboard and visualization
- `pyyaml` - Configuration parsing

### Entry Point Files

#### `run_pipeline.py`
**Purpose:** Wrapper script for main pipeline execution
**Usage:**
```bash
python run_pipeline.py
```
**Note:** This is a convenience wrapper that calls `scripts/run_pipeline.py`

#### `run_dashboard.py`
**Purpose:** Launches the Streamlit web dashboard
**Usage:**
```bash
python run_dashboard.py
# or
streamlit run dashboard/app.py
```

#### `test_pipeline_simple.py`
**Purpose:** Simple test that runs the Data Scout agent on Titanic data
**Tests:**
- Data loading
- Target detection
- Basic statistics
- Problem type identification

**Usage:**
```bash
python test_pipeline_simple.py
```

#### `__init__.py`
**Purpose:** Makes the root directory a Python package
**Contains:** Package metadata (version, author)

---

## 📁 `agents/` - Orchestration Layer

High-level agents that coordinate the ML pipeline workflow.

### `__init__.py`
**Exports:** All agent classes for easy importing
```python
from agents import DataScoutAgent, FeatureEngineerAgent, ModelSelectorAgent, PipelineCoordinator
```

### `base_agent.py`
**Purpose:** Abstract base class for all agents
**Provides:**
- Common initialization (competition name, path, config)
- Logging functionality (via `LoggerMixin`)
- File I/O utilities (via `FileManager`)
- Result saving/loading (JSON format)
- Configuration access

**Key Methods:**
- `run()` - Abstract method that each agent must implement
- `save_results()` - Save agent results to JSON
- `load_results()` - Load previous agent results
- `get_config()` - Get configuration values

### `data_scout.py`
**Purpose:** Data exploration, cleaning, and initial analysis
**Responsibilities:**
- Load training and test datasets
- Validate data quality and consistency
- Detect target column automatically
- Analyze feature types (numeric, categorical)
- Identify missing values and outliers
- Clean and preprocess data
- Generate initial insights

**Output:** `data_scout_results.json` containing:
- Dataset information (rows, columns, types)
- Target column and problem type
- Data quality metrics
- Cleaning actions taken
- Recommendations

**Usage:**
```python
scout = DataScoutAgent("titanic", Path("competition_data/titanic"))
results = scout.run()
```

### `feature_engineer.py`
**Purpose:** Automated feature engineering pipeline
**Responsibilities:**
- Generate new features (polynomial, interactions, aggregations)
- Select important features (variance, correlation, statistical tests)
- Transform features (scaling, encoding, imputation)
- Use LLM for domain-specific feature suggestions

**Output:** `feature_engineer_results.json` containing:
- Features created and their descriptions
- Feature selection results
- Transformation methods applied
- LLM recommendations

**Key Features:**
- Polynomial features (degree 2, 3)
- Mathematical combinations (add, subtract, multiply, divide)
- Statistical aggregations (mean, std, min, max)
- Variance thresholding
- Correlation filtering
- Univariate feature selection

### `model_selector.py`
**Purpose:** Model training, evaluation, and selection
**Responsibilities:**
- Train multiple ML models in parallel
- Perform cross-validation scoring
- Optimize hyperparameters with Optuna
- Create ensemble models
- Evaluate and compare models
- Select best performing model

**Supported Models:**
- Random Forest
- XGBoost
- LightGBM
- CatBoost
- Logistic/Linear Regression
- Support Vector Machines

**Output:** `model_selector_results.json` containing:
- Model comparison results
- Best model name and score
- Hyperparameters used
- Cross-validation scores
- Feature importance

### `coordinator.py`
**Purpose:** Orchestrates the complete end-to-end pipeline
**Responsibilities:**
- Run Data Scout → Feature Engineer → Model Selector
- Create submission files
- Submit to Kaggle API
- Track pipeline progress
- Handle errors and recovery

**Key Methods:**
- `run()` - Execute complete pipeline
- `create_submission()` - Generate Kaggle submission file
- `submit_to_kaggle()` - Upload submission via API

**Usage:**
```python
coordinator = PipelineCoordinator("titanic", Path("competition_data/titanic"))
results = coordinator.run(submit_to_kaggle=True)
```

---

## 📁 `core/` - Business Logic Layer

Pure business logic without orchestration concerns. Modular, testable components.

### `__init__.py`
**Exports:** Nothing (subpackages have their own exports)

---

### 📁 `core/data/` - Data Management

#### `__init__.py`
**Exports:**
```python
from core.data import CompetitionDataLoader, DataPreprocessor, DataValidator
```

#### `loaders.py`
**Purpose:** Load and parse competition datasets
**Classes:**
- `DataLoader` - Base loader with CSV reading utilities
- `CompetitionDataLoader` - Specialized for Kaggle competitions
- `DatasetInfo` - Dataclass with dataset metadata

**Key Methods:**
- `load_competition_data()` - Load train/test CSVs
- `detect_target_column()` - Automatically identify target
- `analyze_feature_types()` - Classify features as numeric/categorical
- `get_dataset_info()` - Extract comprehensive metadata

#### `preprocessors.py`
**Purpose:** Clean and preprocess data
**Classes:**
- `DataPreprocessor` - Main preprocessing class

**Key Methods:**
- `clean_data()` - Remove duplicates, handle missing values
- `handle_missing_values()` - Imputation strategies
- `detect_outliers()` - Statistical outlier detection
- `get_basic_statistics()` - Descriptive statistics
- `impute_missing()` - Fill missing values

**Strategies:**
- Drop columns with >threshold% missing
- Impute numeric: mean/median
- Impute categorical: mode/constant

#### `validators.py`
**Purpose:** Validate data quality
**Classes:**
- `DataValidator` - Data quality checks

**Key Methods:**
- `validate_dataset()` - Comprehensive validation
- `validate_train_test_consistency()` - Ensure train/test match
- `check_missing_values()` - Missing data analysis
- `check_duplicates()` - Duplicate detection
- `check_data_types()` - Type validation

**Checks:**
- Required columns present
- Consistent schemas
- Reasonable data ranges
- Type compatibility

---

### 📁 `core/features/` - Feature Engineering

#### `__init__.py`
**Exports:**
```python
from core.features import FeatureGenerator, FeatureSelector, FeatureTransformer
```

#### `generators.py`
**Purpose:** Generate new features from existing ones
**Classes:**
- `FeatureGenerator` - Main feature generation class

**Key Methods:**
- `generate_numerical_features()` - Math combinations
- `generate_polynomial_features()` - Polynomial expansion
- `generate_statistical_features()` - Aggregations
- `generate_interaction_features()` - Feature interactions

**Feature Types:**
- Addition, subtraction, multiplication, division
- Polynomial features (degree 2, 3)
- Statistical: mean, std, min, max, quantiles
- Interactions between categorical and numeric

#### `selectors.py`
**Purpose:** Select most important features
**Classes:**
- `FeatureSelector` - Feature selection algorithms

**Key Methods:**
- `select_by_variance()` - Remove low variance features
- `select_by_correlation()` - Remove highly correlated features
- `select_by_importance()` - Model-based importance
- `select_univariate()` - Statistical tests (chi2, f_classif)
- `recursive_feature_elimination()` - RFE with cross-validation

**Strategies:**
- Variance threshold filtering
- Correlation analysis
- Univariate statistical tests
- Recursive feature elimination
- Model-based importance (tree models)

#### `transformers.py`
**Purpose:** Transform features (scaling, encoding, imputation)
**Classes:**
- `FeatureTransformer` - Feature transformation pipeline

**Key Methods:**
- `fit_transform()` - Fit and transform training data
- `transform()` - Transform test data
- `encode_categorical()` - OneHot/Label encoding
- `scale_numerical()` - StandardScaler/MinMaxScaler
- `handle_missing()` - Missing value imputation

**Transformations:**
- Scaling: StandardScaler, MinMaxScaler, RobustScaler
- Encoding: OneHotEncoder, LabelEncoder, TargetEncoder
- Imputation: Mean, median, mode, constant
- Power transforms: Log, Box-Cox

---

### 📁 `core/models/` - Model Management

#### `__init__.py`
**Exports:**
```python
from core.models import ModelFactory, ModelEvaluator, HyperparameterOptimizer, EnsembleBuilder
```

#### `factory.py`
**Purpose:** Create and configure ML models
**Classes:**
- `ModelFactory` - Factory for model creation

**Key Methods:**
- `create_model()` - Instantiate model by name
- `get_available_models()` - List supported models
- `get_default_params()` - Get default hyperparameters

**Supported Models:**
- **Classification:** LogisticRegression, RandomForest, XGBoost, LightGBM, CatBoost, SVM
- **Regression:** LinearRegression, Ridge, Lasso, RandomForest, XGBoost, LightGBM, CatBoost

#### `evaluators.py`
**Purpose:** Evaluate model performance
**Classes:**
- `ModelEvaluator` - Comprehensive model evaluation

**Key Methods:**
- `cross_validate_model()` - K-fold cross-validation
- `evaluate_classification()` - Classification metrics
- `evaluate_regression()` - Regression metrics
- `get_feature_importance()` - Extract feature importance
- `compare_models()` - Compare multiple models

**Metrics:**
- **Classification:** Accuracy, Precision, Recall, F1, ROC-AUC, Log Loss
- **Regression:** MAE, MSE, RMSE, R², MAPE

#### `optimizers.py`
**Purpose:** Hyperparameter optimization
**Classes:**
- `HyperparameterOptimizer` - Optuna-based optimization

**Key Methods:**
- `optimize()` - Run optimization trials
- `get_best_params()` - Get optimal hyperparameters
- `get_optimization_history()` - View trial history

**Features:**
- Bayesian optimization via Optuna
- Cross-validation during optimization
- Early stopping support
- Parallel trial execution

#### `ensembles.py`
**Purpose:** Create ensemble models
**Classes:**
- `EnsembleBuilder` - Build model ensembles

**Key Methods:**
- `create_voting_ensemble()` - Voting classifier/regressor
- `create_weighted_ensemble()` - Weighted averaging
- `create_stacking_ensemble()` - Stacked generalization

**Ensemble Types:**
- Hard voting (classification)
- Soft voting (classification with probabilities)
- Weighted averaging (regression)
- Stacking with meta-learner

---

### 📁 `core/analysis/` - Performance Analysis

#### `__init__.py`
**Exports:**
```python
from core.analysis import PerformanceAnalyzer, InsightGenerator
```

#### `performance.py`
**Purpose:** Analyze model performance in depth
**Classes:**
- `PerformanceAnalyzer` - Detailed performance analysis

**Key Methods:**
- `analyze_predictions()` - Prediction quality analysis
- `get_confusion_matrix()` - Classification confusion matrix
- `get_residual_analysis()` - Regression residuals
- `identify_problem_areas()` - Find weak spots
- `generate_report()` - Comprehensive report

**Analyses:**
- Error distribution
- Residual plots
- Learning curves
- Calibration curves
- Feature importance ranking

#### `insights.py`
**Purpose:** Generate actionable insights using LLM
**Classes:**
- `InsightGenerator` - LLM-powered insights

**Key Methods:**
- `generate_data_insights()` - Data exploration insights
- `generate_feature_recommendations()` - Feature engineering suggestions
- `generate_model_insights()` - Model improvement suggestions
- `generate_competition_strategy()` - Competition-specific advice

**Uses LLM For:**
- Domain-specific feature ideas
- Model selection recommendations
- Error analysis interpretation
- Competition strategy advice

---

## 📁 `utils/` - Infrastructure & Utilities

Cross-cutting concerns and shared utilities.

### `__init__.py`
**Exports:**
```python
from utils import (
    ConfigManager, setup_logging, FileManager,
    KaggleSlayerError, DataLoadError, cached, timer, profile
)
```

### `config.py`
**Purpose:** Configuration management
**Classes:**
- `ConfigManager` - Load and access configuration

**Key Methods:**
- `load_config()` - Load from config.yaml
- `get()` - Get config value with dot notation
- `update_config()` - Update config value
- `save_config()` - Save to file
- `get_env_or_config()` - Check env var, fallback to config

**Features:**
- Dot notation access: `config.get("pipeline.cv_folds")`
- Environment variable override
- Default values
- Type-safe access

### `logging.py`
**Purpose:** Logging utilities
**Functions:**
- `setup_logging()` - Configure logging
- `get_logger()` - Get logger instance

**Classes:**
- `LoggerMixin` - Mixin for easy logging in classes

**Features:**
- Consistent log formatting
- Multiple log levels
- File and console output
- Colored console output

### `io.py`
**Purpose:** File I/O operations
**Classes:**
- `FileManager` - Manage file operations for competitions

**Key Methods:**
- `get_file_path()` - Get path in competition directory
- `save_json()` - Save dict to JSON
- `load_json()` - Load JSON to dict
- `save_dataframe()` - Save DataFrame to CSV/pickle
- `load_dataframe()` - Load DataFrame from file

**Features:**
- Automatic directory creation
- JSON serialization
- DataFrame persistence
- Path management

### `exceptions.py`
**Purpose:** Custom exception hierarchy
**Exceptions:**
- `KaggleSlayerError` - Base exception
- `DataLoadError` - Data loading failures
- `DataValidationError` - Data validation failures
- `FeatureEngineeringError` - Feature engineering failures
- `ModelTrainingError` - Model training failures
- `ConfigurationError` - Configuration issues
- `KaggleAPIError` - Kaggle API errors
- `LLMError` - LLM operation failures

**Usage:**
```python
from utils import DataLoadError
raise DataLoadError("Failed to load train.csv")
```

### `cache.py`
**Purpose:** Caching for expensive operations
**Classes:**
- `DiskCache` - Persistent file-based cache with TTL
- `LRUCache` - In-memory LRU cache

**Functions:**
- `get_disk_cache()` - Get global disk cache instance
- `get_memory_cache()` - Get global memory cache instance
- `cached()` - Decorator for caching function results

**Features:**
- Disk and memory caching
- TTL (time-to-live) support
- LRU eviction policy
- Decorator interface
- Automatic cache key generation

**Usage:**
```python
from utils import cached

@cached(use_disk=True, ttl=3600)
def expensive_function():
    return result
```

### `performance.py`
**Purpose:** Performance monitoring and profiling
**Classes:**
- `PerformanceTimer` - Timer context manager and decorator
- `PerformanceProfiler` - Multi-operation profiler

**Functions:**
- `timer()` - Context manager for timing code blocks
- `timed()` - Decorator to time function execution
- `profile()` - Decorator to profile function execution
- `get_profiler()` - Get global profiler instance

**Features:**
- Function timing
- Operation profiling
- Statistical summaries
- Bottleneck identification

**Usage:**
```python
from utils import timer, timed, profile

with timer("Data loading"):
    df = load_data()

@timed
def process_data():
    pass

@profile("feature_engineering")
def engineer_features():
    pass
```

### `kaggle_api.py`
**Purpose:** Kaggle API integration
**Classes:**
- `KaggleAPIClient` - Wrapper for Kaggle API

**Key Methods:**
- `download_competition_files()` - Download competition data
- `submit_to_competition()` - Submit predictions
- `get_submission_status()` - Check submission status
- `get_leaderboard()` - Get competition leaderboard
- `validate_submission_format()` - Validate submission file

**Features:**
- API credential management
- File downloads
- Submission handling
- Error handling and retries

---

### 📁 `utils/llm/` - LLM Integration

#### `__init__.py`
**Exports:**
```python
from utils.llm import LLMClient, LLMCoordinator
```

#### `client.py`
**Purpose:** LLM API client
**Classes:**
- `LLMClient` - OpenRouter API client

**Key Methods:**
- `chat_completion()` - Send chat completion request
- `stream_completion()` - Stream responses
- `count_tokens()` - Estimate token count

**Features:**
- OpenRouter API integration
- Automatic retries with exponential backoff
- Error handling
- Token counting
- Response caching

#### `coordinator.py`
**Purpose:** High-level LLM coordination
**Classes:**
- `LLMCoordinator` - Orchestrate LLM operations

**Key Methods:**
- `get_data_insights()` - Get data exploration insights
- `get_feature_recommendations()` - Get feature suggestions
- `get_model_recommendations()` - Get model selection advice
- `analyze_errors()` - Interpret model errors

**Features:**
- Structured prompts
- Response parsing
- Context management
- Retry logic

#### `prompts.py`
**Purpose:** LLM prompt templates
**Contains:**
- System prompts
- User prompts for different tasks
- Few-shot examples
- Response formatting instructions

**Prompt Categories:**
- Data exploration prompts
- Feature engineering prompts
- Model selection prompts
- Error analysis prompts
- Competition strategy prompts

---

## 📁 `dashboard/` - Web Interface

### `__init__.py`
**Purpose:** Package initialization (empty)

### `app.py`
**Purpose:** Streamlit dashboard application
**Features:**
- Competition selection
- Pipeline execution control
- Real-time progress monitoring
- Result visualization
- Model comparison
- Feature importance plots
- Submission management

**Pages:**
- Home: Competition overview
- Data Explorer: Dataset inspection
- Feature Engineering: Feature analysis
- Model Training: Model comparison
- Submission: Submit to Kaggle

**Usage:**
```bash
streamlit run dashboard/app.py
```

---

## 📁 `scripts/` - Entry Points & Tools

Collection of executable scripts for various tasks.

### `__init__.py`
**Purpose:** Package initialization (empty)

### `run_pipeline.py`
**Purpose:** Main entry point for pipeline execution
**Usage:**
```bash
python scripts/run_pipeline.py titanic --competition-path competition_data/titanic
```

**Arguments:**
- `competition_name` - Name of the competition
- `--competition-path` - Path to competition data
- `--config` - Custom config file
- `--skip-steps` - Steps to skip
- `--submit` - Submit to Kaggle
- `--log-level` - Logging level

### `run_data_scout.py`
**Purpose:** Run only the Data Scout agent
**Usage:**
```bash
python scripts/run_data_scout.py titanic --competition-path competition_data/titanic
```

**Use Cases:**
- Quick data exploration
- Testing data loading
- Debugging data issues

### `test_pipeline.py`
**Purpose:** Comprehensive test suite
**Tests:**
- Import tests (all modules)
- Configuration tests
- Core component tests
- Agent tests

**Usage:**
```bash
python scripts/test_pipeline.py
```

**Output:**
- Test results summary
- Pass/fail status
- Error messages

### `setup_kaggle.py`
**Purpose:** Interactive Kaggle API setup
**Features:**
- Check for existing credentials
- Prompt for username and API key
- Create/update `.kaggle/kaggle.json`
- Set correct file permissions
- Verify API connection

**Usage:**
```bash
python scripts/setup_kaggle.py
```

### `download_titanic_data.py`
**Purpose:** Download Titanic competition data
**Features:**
- Check for existing data
- Download via Kaggle API
- Extract ZIP files
- Verify file integrity

**Usage:**
```bash
python scripts/download_titanic_data.py
```

### `fix_titanic_data.py`
**Purpose:** Fix common Titanic dataset issues
**Fixes:**
- Missing value handling
- Data type corrections
- Column name normalization
- Sample submission creation

**Usage:**
```bash
python scripts/fix_titanic_data.py
```

---

## 📁 `competition_data/` - Competition Datasets

**Purpose:** Store competition datasets
**Structure:**
```
competition_data/
├── titanic/
│   ├── train.csv
│   ├── test.csv
│   ├── sample_submission.csv
│   ├── submission.csv (generated)
│   ├── data_scout_results.json (generated)
│   ├── feature_engineer_results.json (generated)
│   ├── model_selector_results.json (generated)
│   └── pipeline_results.json (generated)
└── house-prices/
    └── ...
```

**⚠️ Note:** This entire directory is gitignored.

**Generated Files:**
- `*_results.json` - Agent output files
- `submission.csv` - Final predictions
- `*.pkl` / `*.joblib` - Cached models and transformers

---

## 📁 Documentation Files

### `README.md`
**Purpose:** Main project documentation
**Sections:**
- Features overview
- Installation instructions
- Quick start guide
- Configuration reference
- Usage examples
- Troubleshooting

### `ARCHITECTURE.md`
**Purpose:** Detailed architecture documentation
**Topics:**
- System design
- Component relationships
- Design patterns
- Extensibility

### `INSTALLATION.md`
**Purpose:** Step-by-step installation guide
**Covers:**
- Prerequisites
- Virtual environment setup
- Dependency installation
- Configuration
- Verification

### `KAGGLE_SUBMISSION.md`
**Purpose:** Kaggle submission walkthrough
**Topics:**
- API setup
- Submission process
- Format requirements
- Troubleshooting

### `PROJECT_SUMMARY.md`
**Purpose:** Project review and optimization report
**Contains:**
- Review summary
- Cleanup actions
- New features added
- Performance improvements
- Recommendations

### `FILE_STRUCTURE.md` (This File)
**Purpose:** Complete file and directory reference

---

## 🗂️ File Naming Conventions

### Python Modules
- **Lowercase with underscores:** `data_loader.py`, `feature_generator.py`
- **Classes:** PascalCase (`DataLoader`, `FeatureGenerator`)
- **Functions/methods:** snake_case (`load_data()`, `generate_features()`)
- **Constants:** UPPER_CASE (`MAX_FEATURES`, `DEFAULT_THRESHOLD`)

### Configuration Files
- **YAML:** `config.yaml`
- **Environment:** `.env`, `.env.example`
- **JSON:** `*_results.json`

### Documentation
- **ALL CAPS:** `README.md`, `LICENSE`
- **PascalCase:** `ARCHITECTURE.md`, `INSTALLATION.md`

### Data Files
- **CSV:** `train.csv`, `test.csv`, `submission.csv`
- **Pickled models:** `*.pkl`, `*.joblib`

---

## 📊 File Size Guidelines

**Small Files (<50 lines):**
- `__init__.py` files
- Simple configuration
- Basic utilities

**Medium Files (50-300 lines):**
- Most classes
- Agent implementations
- Core components

**Large Files (300-600 lines):**
- Complex factories
- LLM prompts
- Comprehensive analyzers

**Very Large Files (>600 lines):**
- Should be split if possible
- Current exceptions: `prompts.py` (622 lines)

---

## 🔄 File Dependencies

### Most Imported Files
1. `utils/config.py` - Used by almost everything
2. `utils/logging.py` - Used by all agents and core components
3. `agents/base_agent.py` - Inherited by all agents
4. `core/data/loaders.py` - Used by all agents

### No Dependencies (Utilities Only)
- `utils/exceptions.py`
- `utils/cache.py`
- `utils/performance.py`

### High Dependency (Framework Code)
- `agents/coordinator.py` - Depends on all agents
- `dashboard/app.py` - Depends on all agents and core components

---

## 🧹 Maintenance

### Files Requiring Regular Updates
- `requirements.txt` - When adding new dependencies
- `config.yaml` - When adding new configuration options
- `README.md` - When adding major features
- Test files - When adding new functionality

### Files That Should Rarely Change
- `base_agent.py` - Core abstraction
- `exceptions.py` - Exception hierarchy
- `__init__.py` files - Export definitions

### Auto-Generated Files (Never Edit Directly)
- `__pycache__/` directories
- `*.pyc` files
- `*_results.json` files
- `.cache/` directory

---

**Last Updated:** 2025-09-30
**Version:** 2.0.0
