#!/usr/bin/env python3
"""
Model Selection Agent - Advanced model selection and hyperparameter tuning with LLM intelligence

This agent analyzes competition characteristics and dataset features to select optimal models
and hyperparameters using LLM insights combined with systematic optimization.
"""

import argparse
import json
import sys
import warnings
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, asdict
from datetime import datetime
import pandas as pd
import numpy as np

# ML libraries
from sklearn.model_selection import cross_val_score, StratifiedKFold, KFold
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import accuracy_score, mean_squared_error, roc_auc_score

# Advanced models
try:
    import xgboost as xgb
    XGB_AVAILABLE = True
except ImportError:
    XGB_AVAILABLE = False

try:
    import lightgbm as lgb
    LGB_AVAILABLE = True
except ImportError:
    LGB_AVAILABLE = False

try:
    from catboost import CatBoostClassifier, CatBoostRegressor
    CATBOOST_AVAILABLE = True
except ImportError:
    CATBOOST_AVAILABLE = False

try:
    import optuna
    OPTUNA_AVAILABLE = True
    optuna.logging.set_verbosity(optuna.logging.WARNING)
except ImportError:
    OPTUNA_AVAILABLE = False

# Suppress warnings
warnings.filterwarnings('ignore')

try:
    from agents.llm_coordinator import LLMCoordinator
    from utils.llm_utils import PromptTemplates, LLMUtils
    DEPENDENCIES_AVAILABLE = True
except ImportError:
    DEPENDENCIES_AVAILABLE = False


@dataclass
class ModelResult:
    """Structure for individual model performance"""
    model_name: str
    model_type: str  # classifier/regressor
    cv_mean: float
    cv_std: float
    best_params: Dict[str, Any]
    feature_importance: Dict[str, float]
    training_time: float
    prediction_time: float
    confidence_score: float


@dataclass
class ModelSelection:
    """Structure to hold model selection results"""
    competition_name: str
    problem_type: str  # classification/regression
    best_model: str
    model_results: List[ModelResult]
    ensemble_recommendation: Dict[str, Any]
    hyperparameter_insights: Dict[str, Any]
    validation_strategy: str
    feature_selection_results: Dict[str, Any]
    timestamp: str
    total_models_tested: int


@dataclass
class LLMModelInsights:
    """Structure for LLM-generated model selection insights"""
    competition_name: str
    recommended_models: List[str]
    model_rationale: Dict[str, str]  # model -> reasoning
    hyperparameter_suggestions: Dict[str, Dict[str, Any]]  # model -> params
    ensemble_strategy: str
    validation_approach: str
    feature_selection_advice: List[str]
    optimization_priorities: List[str]
    risk_assessment: List[str]
    confidence_score: float
    analysis_timestamp: str


class ModelSelector:
    """
    Model Selection Agent with LLM-powered recommendations and optimization
    """

    def __init__(self, competition_path: Path, enable_llm: bool = True):
        self.competition_path = Path(competition_path)
        self.competition_name = self.competition_path.name
        self.enable_llm = enable_llm

        # Initialize output directory
        self.output_dir = self.competition_path / "model_selection"
        self.output_dir.mkdir(exist_ok=True)

        # Initialize LLM coordinator if available and enabled
        self.llm_coordinator = None
        if self.enable_llm and DEPENDENCIES_AVAILABLE:
            try:
                self.llm_coordinator = LLMCoordinator(log_dir=self.output_dir / "llm_logs")
                print("LLM coordinator initialized for model selection")
            except Exception as e:
                print(f"Warning: Could not initialize LLM coordinator: {e}")
                self.enable_llm = False

        # Load competition and dataset insights
        self.competition_insights = None
        self.dataset_info = None
        self.feature_engineering_info = None
        self.problem_type = None
        self.is_classification = True

        # Storage for results
        self.model_selection = None
        self.llm_insights = None
        self.trained_models = {}

    def load_insights_and_data(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Load data and existing insights from previous agents"""
        print("Loading data and insights from previous agents...")

        # Try to load engineered features first, fall back to original data
        train_path = self.competition_path / "feature_engineering" / "train_engineered.csv"
        test_path = self.competition_path / "feature_engineering" / "test_engineered.csv"

        if not train_path.exists():
            print("No engineered features found, using original data")
            train_path = self.competition_path / "train.csv"
            test_path = self.competition_path / "test.csv"

        if not train_path.exists():
            raise FileNotFoundError(f"Training data not found: {train_path}")

        train_df = pd.read_csv(train_path)
        test_df = pd.read_csv(test_path) if test_path.exists() else pd.DataFrame()

        print(f"Loaded train: {train_df.shape}, test: {test_df.shape}")

        # Load competition insights
        comp_insights_path = self.competition_path / "competition_understanding.json"
        if comp_insights_path.exists():
            with open(comp_insights_path, 'r') as f:
                self.competition_insights = json.load(f)
            print("Loaded competition insights")

        # Load dataset insights
        dataset_info_path = self.competition_path / "data_scout_outputs" / "dataset_info.json"
        if dataset_info_path.exists():
            with open(dataset_info_path, 'r') as f:
                self.dataset_info = json.load(f)
            print("Loaded dataset insights")

        # Load feature engineering insights
        feature_info_path = self.competition_path / "feature_engineering" / "feature_engineering.json"
        if feature_info_path.exists():
            with open(feature_info_path, 'r') as f:
                self.feature_engineering_info = json.load(f)
            print("Loaded feature engineering insights")

        # Determine problem type
        self._determine_problem_type(train_df)

        return train_df, test_df

    def analyze_models_with_llm(self, train_df: pd.DataFrame) -> Optional[LLMModelInsights]:
        """Use LLM to analyze dataset and recommend optimal models and parameters"""
        if not self.enable_llm or not self.llm_coordinator:
            return None

        print("Analyzing models with LLM...")

        # Prepare context for LLM
        dataset_summary = self._prepare_dataset_summary(train_df)
        competition_context = self._prepare_competition_context()
        problem_context = self._prepare_problem_context(train_df)

        # Create prompt for model analysis
        prompt = PromptTemplates.model_selection_analysis(
            dataset_summary=dataset_summary,
            competition_context=competition_context,
            problem_context=problem_context,
            available_models=self._get_available_models()
        )

        # Get structured insights from LLM
        llm_response = self.llm_coordinator.structured_output(
            prompt,
            agent="model_selector",
            model_type="primary",
            temperature=0.3,  # Lower temperature for consistent recommendations
            max_tokens=3500
        )

        if not llm_response:
            print("Failed to get LLM model analysis")
            return None

        # Validate response structure
        required_keys = ["recommended_models", "model_rationale", "hyperparameter_suggestions"]
        if not LLMUtils.validate_json_structure(llm_response, required_keys):
            print("LLM response missing required fields for model analysis")
            return None

        # Calculate confidence score
        confidence = self._calculate_model_confidence(llm_response)

        # Create structured insights
        insights = LLMModelInsights(
            competition_name=self.competition_name,
            recommended_models=llm_response.get("recommended_models", [])[:5],
            model_rationale=llm_response.get("model_rationale", {}),
            hyperparameter_suggestions=llm_response.get("hyperparameter_suggestions", {}),
            ensemble_strategy=llm_response.get("ensemble_strategy", "Simple averaging"),
            validation_approach=llm_response.get("validation_approach", "K-fold cross-validation"),
            feature_selection_advice=llm_response.get("feature_selection_advice", [])[:5],
            optimization_priorities=llm_response.get("optimization_priorities", [])[:3],
            risk_assessment=llm_response.get("risk_assessment", [])[:3],
            confidence_score=confidence,
            analysis_timestamp=datetime.now().isoformat()
        )

        return insights

    def prepare_data_for_modeling(self, train_df: pd.DataFrame, test_df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Prepare data for model training"""
        print("Preparing data for modeling...")

        # Identify target column
        target_col = self._identify_target_column(train_df)

        # Separate features and target
        feature_cols = [col for col in train_df.columns if col != target_col and col not in ['id', 'Id', 'ID']]

        X = train_df[feature_cols].copy()
        y = train_df[target_col].copy()
        X_test = test_df[feature_cols].copy() if not test_df.empty else pd.DataFrame()

        # Handle categorical variables
        categorical_cols = X.select_dtypes(include=['object']).columns
        for col in categorical_cols:
            if col in X.columns:
                le = LabelEncoder()
                X[col] = le.fit_transform(X[col].astype(str))
                if not X_test.empty and col in X_test.columns:
                    # Handle unseen categories in test set
                    X_test[col] = X_test[col].astype(str)
                    mask = X_test[col].isin(le.classes_)
                    X_test[col][mask] = le.transform(X_test[col][mask])
                    X_test[col][~mask] = 0  # Assign 0 to unseen categories

        # Handle missing values
        X = X.fillna(X.median())
        if not X_test.empty:
            X_test = X_test.fillna(X.median())

        # Convert to numpy arrays
        X_array = X.values.astype(np.float32)
        y_array = y.values
        X_test_array = X_test.values.astype(np.float32) if not X_test.empty else np.array([])

        print(f"Prepared data: X{X_array.shape}, y{y_array.shape}, X_test{X_test_array.shape if len(X_test_array) > 0 else '(0,)'}")

        return X_array, y_array, X_test_array, feature_cols

    def evaluate_models(self, X: np.ndarray, y: np.ndarray, feature_cols: List[str]) -> List[ModelResult]:
        """Evaluate multiple models with cross-validation"""
        print("Evaluating models...")

        model_results = []

        # Determine models to test based on LLM recommendations or defaults
        models_to_test = self._get_models_to_test()
        print(f"Models to test: {list(models_to_test.keys())}")

        # Setup cross-validation
        if self.is_classification:
            cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
            scoring = 'roc_auc' if len(np.unique(y)) == 2 else 'accuracy'
        else:
            cv = KFold(n_splits=5, shuffle=True, random_state=42)
            scoring = 'neg_root_mean_squared_error'

        for model_name, model_config in models_to_test.items():
            try:
                print(f"Evaluating {model_name}...")
                start_time = datetime.now()

                model = model_config['model']
                print(f"Created model: {model}")

                # Cross-validation
                cv_scores = cross_val_score(model, X, y, cv=cv, scoring=scoring, n_jobs=-1)

                # Train model for feature importance
                model.fit(X, y)

                # Get feature importance
                feature_importance = self._get_feature_importance(model, feature_cols)

                training_time = (datetime.now() - start_time).total_seconds()

                # Calculate confidence based on CV scores and model characteristics
                confidence = self._calculate_model_confidence_score(cv_scores, model_name)

                result = ModelResult(
                    model_name=model_name,
                    model_type="classifier" if self.is_classification else "regressor",
                    cv_mean=cv_scores.mean(),
                    cv_std=cv_scores.std(),
                    best_params=model_config.get('params', {}),
                    feature_importance=feature_importance,
                    training_time=training_time,
                    prediction_time=0.0,  # Would measure in production
                    confidence_score=confidence
                )

                model_results.append(result)

                # Store trained model
                self.trained_models[model_name] = model

                print(f"{model_name}: CV {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")

            except Exception as e:
                print(f"Error evaluating {model_name}: {e}")
                import traceback
                traceback.print_exc()
                continue

        # Sort by performance
        model_results.sort(key=lambda x: x.cv_mean, reverse=True)

        return model_results

    def optimize_best_model(self, X: np.ndarray, y: np.ndarray, best_model_name: str) -> Dict[str, Any]:
        """Optimize hyperparameters for the best model using Optuna"""
        if not OPTUNA_AVAILABLE:
            print("Optuna not available, skipping hyperparameter optimization")
            return {}

        print(f"Optimizing hyperparameters for {best_model_name}...")

        def objective(trial):
            # Get hyperparameter suggestions from LLM or use defaults
            params = self._get_optimization_params(trial, best_model_name)

            # Create model with suggested parameters
            model = self._create_model_with_params(best_model_name, params)

            # Cross-validation
            if self.is_classification:
                cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
                scoring = 'roc_auc' if len(np.unique(y)) == 2 else 'accuracy'
            else:
                cv = KFold(n_splits=3, shuffle=True, random_state=42)
                scoring = 'neg_root_mean_squared_error'

            scores = cross_val_score(model, X, y, cv=cv, scoring=scoring, n_jobs=-1)
            return scores.mean()

        # Create study
        direction = 'maximize'  # Always maximize (even for RMSE as we use neg_root_mean_squared_error)
        study = optuna.create_study(direction=direction)

        # Optimize
        n_trials = 50 if not self.enable_llm else 30  # Fewer trials when using LLM guidance
        study.optimize(objective, n_trials=n_trials, show_progress_bar=False)

        optimization_results = {
            "best_params": study.best_params,
            "best_score": study.best_value,
            "n_trials": len(study.trials),
            "optimization_time": sum(trial.duration.total_seconds() for trial in study.trials if trial.duration)
        }

        print(f"Optimization complete: {study.best_value:.4f} with {len(study.trials)} trials")

        return optimization_results

    def save_results(self, model_selection: ModelSelection):
        """Save model selection results"""
        print("Saving model selection results...")

        # Save model selection metadata
        selection_path = self.output_dir / "model_selection.json"
        with open(selection_path, 'w') as f:
            # Convert ModelResult objects to dicts for JSON serialization
            selection_dict = asdict(model_selection)
            json.dump(selection_dict, f, indent=2, default=str)
        print(f"Saved model selection results: {selection_path}")

        # Save LLM insights if available
        if self.llm_insights:
            insights_path = self.output_dir / "llm_model_insights.json"
            with open(insights_path, 'w') as f:
                json.dump(asdict(self.llm_insights), f, indent=2)
            print(f"Saved LLM model insights: {insights_path}")

        # Save model comparison
        comparison_data = []
        for result in model_selection.model_results:
            comparison_data.append({
                'Model': result.model_name,
                'CV_Mean': result.cv_mean,
                'CV_Std': result.cv_std,
                'Training_Time': result.training_time,
                'Confidence': result.confidence_score
            })

        comparison_df = pd.DataFrame(comparison_data)
        comparison_path = self.output_dir / "model_comparison.csv"
        comparison_df.to_csv(comparison_path, index=False)
        print(f"Saved model comparison: {comparison_path}")

    def run_model_selection(self) -> Tuple[ModelSelection, Dict[str, Any]]:
        """Run complete model selection pipeline"""
        print(f"Model Selection Agent: {self.competition_name}")
        print("=" * 60)

        # Load data and insights
        train_df, test_df = self.load_insights_and_data()

        # Analyze models with LLM if enabled
        if self.enable_llm and self.llm_coordinator:
            print("\nStep 1: Analyzing models with LLM...")
            self.llm_insights = self.analyze_models_with_llm(train_df)

        # Prepare data
        print("\nStep 2: Preparing data for modeling...")
        X, y, X_test, feature_cols = self.prepare_data_for_modeling(train_df, test_df)

        # Evaluate models
        print("\nStep 3: Evaluating models...")
        model_results = self.evaluate_models(X, y, feature_cols)

        if not model_results:
            raise Exception("No models could be evaluated successfully")

        # Get best model
        best_model = model_results[0]
        print(f"\nBest model: {best_model.model_name} (CV: {best_model.cv_mean:.4f})")

        # Optimize best model
        print("\nStep 4: Optimizing hyperparameters...")
        optimization_results = self.optimize_best_model(X, y, best_model.model_name)

        # Create model selection summary
        model_selection = ModelSelection(
            competition_name=self.competition_name,
            problem_type="classification" if self.is_classification else "regression",
            best_model=best_model.model_name,
            model_results=model_results,
            ensemble_recommendation={"strategy": "simple_average", "models": [r.model_name for r in model_results[:3]]},
            hyperparameter_insights=optimization_results,
            validation_strategy="5-fold cross-validation",
            feature_selection_results={"n_features": len(feature_cols)},
            timestamp=datetime.now().isoformat(),
            total_models_tested=len(model_results)
        )

        # Save results
        self.save_results(model_selection)

        print(f"\nModel Selection Complete!")
        print(f"Best model: {best_model.model_name}")
        print(f"CV Score: {best_model.cv_mean:.4f} (+/- {best_model.cv_std * 2:.4f})")
        print(f"Models tested: {len(model_results)}")

        return model_selection, self.trained_models

    # Helper methods
    def _determine_problem_type(self, train_df: pd.DataFrame):
        """Determine if this is a classification or regression problem"""
        target_col = self._identify_target_column(train_df)

        if self.competition_insights:
            problem_type = self.competition_insights.get('problem_type', '').lower()
            self.is_classification = 'classification' in problem_type
        else:
            # Heuristic: if target has few unique values, it's likely classification
            n_unique = train_df[target_col].nunique()
            self.is_classification = n_unique <= 20

        self.problem_type = "classification" if self.is_classification else "regression"
        print(f"Problem type determined: {self.problem_type}")

    def _identify_target_column(self, train_df: pd.DataFrame) -> str:
        """Identify the target column in training data"""
        common_targets = ['target', 'label', 'y', 'class', 'survived', 'price', 'saleprice']

        for col in train_df.columns:
            if col.lower() in common_targets:
                return col

        # Return last column as default
        return train_df.columns[-1]

    def _prepare_dataset_summary(self, train_df: pd.DataFrame) -> str:
        """Prepare dataset summary for LLM"""
        summary_parts = [
            f"Dataset: {train_df.shape[0]} rows, {train_df.shape[1]} columns",
            f"Problem type: {self.problem_type}",
            f"Features: {len(train_df.select_dtypes(include=[np.number]).columns)} numerical, {len(train_df.select_dtypes(include=['object']).columns)} categorical"
        ]

        if self.dataset_info:
            summary_parts.append(f"Missing values: {len([k for k, v in self.dataset_info.get('missing_percentages', {}).items() if v > 0])} columns")

        return " | ".join(summary_parts)

    def _prepare_competition_context(self) -> str:
        """Prepare competition context for LLM"""
        if not self.competition_insights:
            return f"Competition: {self.competition_name}"

        context_parts = [
            f"Competition: {self.competition_insights.get('competition_name', self.competition_name)}",
            f"Key strategies: {', '.join(self.competition_insights.get('key_strategies', [])[:2])}"
        ]

        return " | ".join(context_parts)

    def _prepare_problem_context(self, train_df: pd.DataFrame) -> str:
        """Prepare problem-specific context"""
        target_col = self._identify_target_column(train_df)

        if self.is_classification:
            n_classes = train_df[target_col].nunique()
            context = f"Classification problem with {n_classes} classes"
        else:
            target_range = train_df[target_col].max() - train_df[target_col].min()
            context = f"Regression problem, target range: {target_range:.2f}"

        return context

    def _get_available_models(self) -> List[str]:
        """Get list of available models"""
        models = ["RandomForest", "LogisticRegression", "Ridge"]

        if XGB_AVAILABLE:
            models.append("XGBoost")
        if LGB_AVAILABLE:
            models.append("LightGBM")
        if CATBOOST_AVAILABLE:
            models.append("CatBoost")

        return models

    def _get_models_to_test(self) -> Dict[str, Dict[str, Any]]:
        """Get models to test based on LLM recommendations or defaults"""
        models = {}

        # Default models based on problem type
        if self.is_classification:
            models["RandomForest"] = {"model": RandomForestClassifier(n_estimators=100, random_state=42)}
            models["LogisticRegression"] = {"model": LogisticRegression(random_state=42, max_iter=1000)}

            if XGB_AVAILABLE:
                models["XGBoost"] = {"model": xgb.XGBClassifier(random_state=42, eval_metric='logloss')}
            if LGB_AVAILABLE:
                models["LightGBM"] = {"model": lgb.LGBMClassifier(random_state=42, verbose=-1)}
            if CATBOOST_AVAILABLE:
                models["CatBoost"] = {"model": CatBoostClassifier(random_state=42, verbose=False)}
        else:
            models["RandomForest"] = {"model": RandomForestRegressor(n_estimators=100, random_state=42)}
            models["Ridge"] = {"model": Ridge(random_state=42)}

            if XGB_AVAILABLE:
                models["XGBoost"] = {"model": xgb.XGBRegressor(random_state=42)}
            if LGB_AVAILABLE:
                models["LightGBM"] = {"model": lgb.LGBMRegressor(random_state=42, verbose=-1)}
            if CATBOOST_AVAILABLE:
                models["CatBoost"] = {"model": CatBoostRegressor(random_state=42, verbose=False)}

        # Filter based on LLM recommendations if available
        if self.llm_insights and self.llm_insights.recommended_models:
            recommended = self.llm_insights.recommended_models
            filtered_models = {}

            for model_name, model_config in models.items():
                # Check if this model matches any LLM recommendation
                for rec in recommended:
                    if (model_name.lower() in rec.lower() or
                        rec.lower() in model_name.lower() or
                        any(word in rec.lower() for word in model_name.lower().split())):
                        filtered_models[model_name] = model_config
                        break

            # If filtering results in no models, use all available models as fallback
            if filtered_models:
                models = filtered_models
                print(f"Using LLM-recommended models: {list(models.keys())}")
            else:
                print(f"LLM recommendations didn't match available models, using all available models")

        return models

    def _get_feature_importance(self, model, feature_cols: List[str]) -> Dict[str, float]:
        """Extract feature importance from trained model"""
        importance_dict = {}

        try:
            if hasattr(model, 'feature_importances_'):
                importances = model.feature_importances_
            elif hasattr(model, 'coef_'):
                importances = np.abs(model.coef_[0]) if len(model.coef_.shape) > 1 else np.abs(model.coef_)
            else:
                return importance_dict

            # Create importance dictionary
            for i, col in enumerate(feature_cols):
                if i < len(importances):
                    importance_dict[col] = float(importances[i])

        except Exception as e:
            print(f"Could not extract feature importance: {e}")

        return importance_dict

    def _calculate_model_confidence(self, llm_response: Dict) -> float:
        """Calculate confidence score for LLM model recommendations"""
        confidence = 0.0

        # Check if we have good model recommendations
        if len(llm_response.get("recommended_models", [])) >= 3:
            confidence += 0.4

        # Check if we have model rationale
        if llm_response.get("model_rationale"):
            confidence += 0.3

        # Check if we have hyperparameter suggestions
        if llm_response.get("hyperparameter_suggestions"):
            confidence += 0.3

        return min(confidence, 1.0)

    def _calculate_model_confidence_score(self, cv_scores: np.ndarray, model_name: str) -> float:
        """Calculate confidence score for model performance"""
        # Base confidence on CV score stability
        cv_stability = 1.0 - (cv_scores.std() / max(abs(cv_scores.mean()), 1e-6))
        cv_stability = max(0.0, cv_stability)

        # Boost confidence for generally reliable models
        model_reliability = {
            "RandomForest": 0.1,
            "XGBoost": 0.15,
            "LightGBM": 0.15,
            "CatBoost": 0.15,
            "LogisticRegression": 0.05,
            "Ridge": 0.05
        }

        reliability_boost = model_reliability.get(model_name, 0.0)

        return min(cv_stability + reliability_boost, 1.0)

    def _get_optimization_params(self, trial, model_name: str) -> Dict[str, Any]:
        """Get hyperparameter suggestions for optimization"""
        # Use LLM suggestions if available
        if self.llm_insights and model_name in self.llm_insights.hyperparameter_suggestions:
            llm_params = self.llm_insights.hyperparameter_suggestions[model_name]
            # Convert LLM suggestions to Optuna trial suggestions
            params = {}
            for param, value in llm_params.items():
                if isinstance(value, dict) and 'range' in value:
                    if value.get('type') == 'int':
                        params[param] = trial.suggest_int(param, value['range'][0], value['range'][1])
                    else:
                        params[param] = trial.suggest_float(param, value['range'][0], value['range'][1])
                elif isinstance(value, list):
                    params[param] = trial.suggest_categorical(param, value)
                else:
                    params[param] = value
            return params

        # Default hyperparameter ranges
        if model_name == "RandomForest":
            return {
                'n_estimators': trial.suggest_int('n_estimators', 50, 300),
                'max_depth': trial.suggest_int('max_depth', 5, 20),
                'min_samples_split': trial.suggest_int('min_samples_split', 2, 20)
            }
        elif model_name == "XGBoost":
            return {
                'n_estimators': trial.suggest_int('n_estimators', 50, 300),
                'max_depth': trial.suggest_int('max_depth', 3, 10),
                'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3),
                'subsample': trial.suggest_float('subsample', 0.6, 1.0)
            }
        elif model_name == "LightGBM":
            return {
                'n_estimators': trial.suggest_int('n_estimators', 50, 300),
                'num_leaves': trial.suggest_int('num_leaves', 10, 100),
                'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3),
                'feature_fraction': trial.suggest_float('feature_fraction', 0.4, 1.0)
            }
        else:
            return {}

    def _create_model_with_params(self, model_name: str, params: Dict[str, Any]):
        """Create model instance with specified parameters"""
        base_params = {'random_state': 42}
        all_params = {**base_params, **params}

        if model_name == "RandomForest":
            if self.is_classification:
                return RandomForestClassifier(**all_params)
            else:
                return RandomForestRegressor(**all_params)
        elif model_name == "XGBoost" and XGB_AVAILABLE:
            if self.is_classification:
                all_params['eval_metric'] = 'logloss'
                return xgb.XGBClassifier(**all_params)
            else:
                return xgb.XGBRegressor(**all_params)
        elif model_name == "LightGBM" and LGB_AVAILABLE:
            all_params['verbose'] = -1
            if self.is_classification:
                return lgb.LGBMClassifier(**all_params)
            else:
                return lgb.LGBMRegressor(**all_params)
        elif model_name == "CatBoost" and CATBOOST_AVAILABLE:
            all_params['verbose'] = False
            if self.is_classification:
                return CatBoostClassifier(**all_params)
            else:
                return CatBoostRegressor(**all_params)
        else:
            # Fallback to basic models
            if self.is_classification:
                return RandomForestClassifier(**base_params)
            else:
                return RandomForestRegressor(**base_params)


def main():
    """Main entry point for Model Selection Agent"""
    parser = argparse.ArgumentParser(description="Model Selection Agent with LLM and Optuna")
    parser.add_argument("competition_path", type=Path,
                       help="Path to competition directory")
    parser.add_argument("--no-llm", action="store_true",
                       help="Disable LLM and use basic model selection only")

    args = parser.parse_args()

    # Validate competition path
    if not args.competition_path.exists():
        print(f"ERROR: Competition path does not exist: {args.competition_path}")
        return 1

    train_csv = args.competition_path / "train.csv"
    if not train_csv.exists():
        print(f"ERROR: Training data not found: {train_csv}")
        return 1

    try:
        # Initialize and run model selection
        selector = ModelSelector(args.competition_path, enable_llm=not args.no_llm)
        model_selection, trained_models = selector.run_model_selection()

        print(f"\nModel Selection Success!")
        print(f"Best model: {model_selection.best_model}")
        print(f"Models tested: {model_selection.total_models_tested}")
        print(f"Results saved to: {selector.output_dir}")
        return 0

    except Exception as e:
        print(f"ERROR: Model selection failed: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    exit(main())