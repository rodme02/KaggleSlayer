# KaggleSlayer

**Simple, effective AutoML pipeline for Kaggle competitions**

KaggleSlayer is a streamlined, end-to-end machine learning pipeline that automates the process of data analysis, feature engineering, model selection, and submission generation for Kaggle competitions.

## ✨ Features

- **Automated Data Exploration**: Intelligent data type detection, missing value analysis, and statistical insights
- **Smart Feature Engineering**: Automatic creation of numerical, categorical, and statistical features with leak-free cross-validation
- **Multi-Model Evaluation**: Tests 8 different algorithms (Random Forest, XGBoost, LightGBM, CatBoost, and more)
- **Ensemble Methods**: Automatic creation and evaluation of voting and stacking ensembles
- **Kaggle Integration**: Direct submission to Kaggle competitions via API
- **Clean Output**: Minimal, informative logging for easy progress tracking

## 🚀 Quick Start

### Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/KaggleSlayer.git
cd KaggleSlayer

# Install dependencies
pip install -r requirements.txt
```

### Basic Usage

```bash
# Run the full pipeline
python kaggle_slayer.py <competition_name> --data-path <path_to_data>

# Example: Titanic competition
python kaggle_slayer.py titanic --data-path competition_data/titanic

# With automatic submission to Kaggle
python kaggle_slayer.py titanic --data-path competition_data/titanic --submit
```

### Data Structure

Your competition data directory should contain:
```
competition_data/titanic/
├── raw/
│   ├── train.csv
│   ├── test.csv
│   └── sample_submission.csv (optional)
```

The pipeline will automatically create:
```
competition_data/titanic/
├── processed/          # Cleaned and engineered data
├── results/            # Model results and metrics
└── submission.csv      # Final submission file
```

## 📋 Pipeline Stages

### 1. Data Scout
- Loads and analyzes raw data
- Detects data types (numerical, categorical, datetime, text)
- Identifies target column and ID columns
- Handles missing values with intelligent imputation
- Creates missingness indicators for important features

### 2. Feature Engineer
- **Numerical Features**: Ratios, differences, polynomials, log transforms
- **Categorical Features**: Frequency encoding, target encoding, one-hot encoding
- **Statistical Features**: Row-wise statistics, aggregations
- **Feature Selection**: Variance thresholds, correlation filtering, univariate selection
- **Leak Prevention**: Feature engineering happens inside CV folds

### 3. Model Selector
- Evaluates 8 models with 5-fold cross-validation:
  - Random Forest
  - Extra Trees
  - Logistic Regression / Linear Regression
  - K-Nearest Neighbors
  - Support Vector Machine
  - XGBoost
  - LightGBM
  - CatBoost
- Optional hyperparameter optimization
- Optional ensemble creation (voting/stacking)
- Saves best model for predictions

### 4. Submission Creator
- Generates predictions on test data
- Formats submission file correctly
- Optionally submits directly to Kaggle

## ⚙️ Configuration

Edit `config.yaml` to customize pipeline behavior:

```yaml
pipeline:
  cv_folds: 5                    # Number of CV folds
  cv_random_state: 42            # Random seed for reproducibility
  max_features_to_create: 25     # Maximum new features to generate

data:
  correlation_threshold: 0.95    # Remove features with correlation > threshold
  variance_threshold: 0.01       # Remove features with variance < threshold
```

## 📊 Example Output

```
======================================================================
  KAGGLESLAYER PIPELINE - TITANIC
======================================================================

======================================================================
  STEP 1/4: DATA SCOUT - Exploring and Cleaning Data
======================================================================
[OK] Data Scout completed in 0.1s

======================================================================
  STEP 2/4: FEATURE ENGINEER - Creating Powerful Features
======================================================================
[OK] Feature Engineering completed in 0.2s

======================================================================
  STEP 3/4: MODEL SELECTOR - Training and Optimizing Models
======================================================================

  [1/8] Evaluating random_forest... CV: 0.8384
  [2/8] Evaluating extra_trees... CV: 0.8215
  [3/8] Evaluating logistic_regression... CV: 0.7980
  [4/8] Evaluating knn... CV: 0.8148
  [5/8] Evaluating svm... CV: 0.8125
  [6/8] Evaluating xgboost... CV: 0.8160
  [7/8] Evaluating lightgbm... CV: 0.8036
  [8/8] Evaluating catboost... CV: 0.8283

[OK] Model Selection completed in 35.2s

======================================================================
  PIPELINE COMPLETED SUCCESSFULLY!
======================================================================
  Competition: titanic
  Best Model: random_forest
  CV Score: 0.8384
  Total Time: 35.2s (0.6 minutes)
======================================================================
```

## 🏗️ Project Structure

```
KaggleSlayer/
├── agents/                 # Pipeline orchestration
│   ├── base_agent.py      # Base class for all agents
│   ├── coordinator.py     # Main pipeline coordinator
│   ├── data_scout.py      # Data exploration and cleaning
│   ├── feature_engineer.py # Feature generation and selection
│   └── model_selector.py  # Model training and evaluation
├── core/                  # Core ML components
│   ├── data/             # Data loading and preprocessing
│   ├── features/         # Feature engineering utilities
│   └── models/           # Model factories and ensembles
├── utils/                # Utility functions
│   ├── config.py        # Configuration management
│   ├── io.py            # File I/O operations
│   ├── logging.py       # Logging utilities
│   ├── kaggle_api.py    # Kaggle API integration
│   └── performance.py   # Performance monitoring
├── config.yaml          # Pipeline configuration
├── requirements.txt     # Python dependencies
└── kaggle_slayer.py    # Main entry point
```

## 🔧 Advanced Usage

### Custom Feature Engineering

The pipeline uses a scikit-learn compatible feature engineering pipeline that prevents data leakage:

```python
from agents.feature_engineer import FeatureEngineerAgent

engineer = FeatureEngineerAgent(competition_name, competition_path)
feature_pipeline = engineer.create_feature_pipeline()

# Use in your own pipeline
from sklearn.pipeline import Pipeline
model_pipeline = Pipeline([
    ('features', feature_pipeline),
    ('model', your_model)
])
```

### Programmatic Access

```python
from pathlib import Path
from agents.coordinator import PipelineCoordinator

# Run pipeline programmatically
coordinator = PipelineCoordinator("titanic", Path("competition_data/titanic"))
results = coordinator.run(submit_to_kaggle=False)

# Access results
print(f"Best model: {results['best_model']}")
print(f"CV Score: {results['final_score']}")
```

## 📝 Requirements

- Python 3.8+
- scikit-learn
- pandas
- numpy
- xgboost
- lightgbm
- catboost
- PyYAML
- kaggle (for API submissions)

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## 📄 License

This project is open source and available under the MIT License.

## 🎯 Roadmap

- [ ] Support for more competition types (object detection, NLP, time series)
- [ ] Neural network models (PyTorch/TensorFlow)
- [ ] Automated hyperparameter tuning with Optuna
- [ ] Feature importance analysis and visualization
- [ ] Model interpretation tools (SHAP, LIME)
- [ ] Multi-target and multi-class support
- [ ] Automated report generation

## 💡 Tips for Best Results

1. **Data Quality**: Ensure your train.csv and test.csv are clean and properly formatted
2. **Target Column**: Name your target column clearly (e.g., 'Survived', 'SalePrice')
3. **ID Columns**: Use standard naming (e.g., 'PassengerId', 'Id') for automatic detection
4. **Feature Engineering**: The pipeline works best with a mix of numerical and categorical features
5. **Cross-Validation**: The CV score is a good indicator of test performance when data leakage is prevented

## 📧 Contact

For questions, suggestions, or issues, please open an issue on GitHub.

---

**Happy Kaggling! 🏆**
