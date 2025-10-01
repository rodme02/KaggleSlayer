"""
Model factory for creating and configuring different ML models.
"""

from typing import Dict, Any, Optional, List, Union
import pandas as pd
import numpy as np
from sklearn.ensemble import (
    RandomForestClassifier, RandomForestRegressor,
    ExtraTreesClassifier, ExtraTreesRegressor
)
from sklearn.linear_model import (
    LogisticRegression, Ridge, Lasso, ElasticNet
)
from sklearn.svm import SVC, SVR
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
import warnings

warnings.filterwarnings('ignore')

# Optional advanced models
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


class ModelFactory:
    """Factory for creating and configuring ML models."""

    def __init__(self, problem_type: str = "auto", random_state: int = 42):
        self.problem_type = problem_type
        self.random_state = random_state
        self.available_models = self._get_available_models()

    def _get_available_models(self) -> Dict[str, bool]:
        """Check which models are available."""
        return {
            'sklearn': True,
            'xgboost': XGB_AVAILABLE,
            'lightgbm': LGB_AVAILABLE,
            'catboost': CATBOOST_AVAILABLE
        }

    def get_available_model_names(self, problem_type: Optional[str] = None) -> List[str]:
        """Get list of available model names for the problem type."""
        if problem_type is None:
            problem_type = self.problem_type

        if problem_type == "classification":
            models = [
                'random_forest', 'extra_trees', 'logistic_regression',
                'knn', 'svm'
            ]
            if self.available_models['xgboost']:
                models.append('xgboost')
            if self.available_models['lightgbm']:
                models.append('lightgbm')
            if self.available_models['catboost']:
                models.append('catboost')

        elif problem_type == "regression":
            models = [
                'random_forest', 'extra_trees', 'ridge', 'lasso',
                'elastic_net', 'knn', 'svr'
            ]
            if self.available_models['xgboost']:
                models.append('xgboost')
            if self.available_models['lightgbm']:
                models.append('lightgbm')
            if self.available_models['catboost']:
                models.append('catboost')

        else:
            models = []

        return models

    def create_model(self, model_name: str, parameters: Optional[Dict[str, Any]] = None,
                    problem_type: Optional[str] = None) -> Any:
        """Create a model instance with given parameters."""
        if problem_type is None:
            problem_type = self.problem_type

        if parameters is None:
            parameters = self.get_default_parameters(model_name, problem_type)

        # Add random state to parameters if not present
        if 'random_state' not in parameters and self._model_supports_random_state(model_name):
            parameters['random_state'] = self.random_state

        try:
            if problem_type == "classification":
                return self._create_classification_model(model_name, parameters)
            elif problem_type == "regression":
                return self._create_regression_model(model_name, parameters)
            else:
                raise ValueError(f"Unknown problem type: {problem_type}")

        except Exception as e:
            raise ValueError(f"Failed to create {model_name} model: {e}")

    def _create_classification_model(self, model_name: str, parameters: Dict[str, Any]) -> Any:
        """Create a classification model."""
        if model_name == 'random_forest':
            return RandomForestClassifier(**parameters)
        elif model_name == 'extra_trees':
            return ExtraTreesClassifier(**parameters)
        elif model_name == 'logistic_regression':
            return LogisticRegression(**parameters)
        elif model_name == 'knn':
            return KNeighborsClassifier(**parameters)
        elif model_name == 'svm':
            return SVC(**parameters)
        elif model_name == 'xgboost' and XGB_AVAILABLE:
            return xgb.XGBClassifier(**parameters)
        elif model_name == 'lightgbm' and LGB_AVAILABLE:
            return lgb.LGBMClassifier(**parameters)
        elif model_name == 'catboost' and CATBOOST_AVAILABLE:
            return CatBoostClassifier(**parameters)
        else:
            raise ValueError(f"Unknown classification model: {model_name}")

    def _create_regression_model(self, model_name: str, parameters: Dict[str, Any]) -> Any:
        """Create a regression model."""
        if model_name == 'random_forest':
            return RandomForestRegressor(**parameters)
        elif model_name == 'extra_trees':
            return ExtraTreesRegressor(**parameters)
        elif model_name == 'ridge':
            return Ridge(**parameters)
        elif model_name == 'lasso':
            return Lasso(**parameters)
        elif model_name == 'elastic_net':
            return ElasticNet(**parameters)
        elif model_name == 'knn':
            return KNeighborsRegressor(**parameters)
        elif model_name == 'svr':
            return SVR(**parameters)
        elif model_name == 'xgboost' and XGB_AVAILABLE:
            return xgb.XGBRegressor(**parameters)
        elif model_name == 'lightgbm' and LGB_AVAILABLE:
            return lgb.LGBMRegressor(**parameters)
        elif model_name == 'catboost' and CATBOOST_AVAILABLE:
            return CatBoostRegressor(**parameters)
        else:
            raise ValueError(f"Unknown regression model: {model_name}")

    def get_default_parameters(self, model_name: str, problem_type: str) -> Dict[str, Any]:
        """Get default parameters for a model."""
        defaults = {
            'random_forest': {'n_estimators': 200, 'max_depth': 15, 'min_samples_split': 5, 'min_samples_leaf': 2, 'n_jobs': -1, 'max_samples': 0.8},
            'extra_trees': {'n_estimators': 200, 'max_depth': 15, 'min_samples_split': 5, 'min_samples_leaf': 2, 'n_jobs': -1, 'bootstrap': True, 'max_samples': 0.8},
            'logistic_regression': {'max_iter': 2000, 'solver': 'lbfgs', 'C': 1.0},
            'ridge': {'alpha': 1.0, 'solver': 'auto'},
            'lasso': {'alpha': 0.1, 'max_iter': 2000},
            'elastic_net': {'alpha': 0.1, 'l1_ratio': 0.5, 'max_iter': 2000},
            'knn': {'n_neighbors': 7, 'weights': 'distance'},
            'svm': {'kernel': 'rbf', 'C': 10.0, 'gamma': 'scale'},
            'svr': {'kernel': 'rbf', 'C': 10.0, 'gamma': 'scale'},
            'xgboost': {
                'n_estimators': 200,
                'max_depth': 6,
                'learning_rate': 0.05,
                'subsample': 0.8,
                'colsample_bytree': 0.8,
                'min_child_weight': 3,
                'gamma': 0.1,
                'n_jobs': -1
            },
            'lightgbm': {
                'n_estimators': 200,
                'max_depth': 7,
                'learning_rate': 0.05,
                'feature_fraction': 0.8,
                'bagging_fraction': 0.8,
                'num_leaves': 50,
                'min_child_samples': 20,
                'n_jobs': -1,
                'verbose': -1
            },
            'catboost': {
                'iterations': 200,
                'depth': 6,
                'learning_rate': 0.05,
                'l2_leaf_reg': 3,
                'verbose': False,
                'thread_count': -1
            }
        }

        return defaults.get(model_name, {}).copy()

    def get_parameter_space(self, model_name: str, problem_type: str) -> Dict[str, List[Any]]:
        """Get parameter space for hyperparameter optimization."""
        if problem_type == "classification":
            spaces = {
                'random_forest': {
                    'n_estimators': [100, 200, 300],
                    'max_depth': [5, 10, 15, None],
                    'min_samples_split': [2, 5, 10],
                    'min_samples_leaf': [1, 2, 4]
                },
                'extra_trees': {
                    'n_estimators': [100, 200, 300],
                    'max_depth': [5, 10, 15, None],
                    'min_samples_split': [2, 5, 10],
                    'min_samples_leaf': [1, 2, 4]
                },
                'logistic_regression': {
                    'C': [0.001, 0.01, 0.1, 1.0, 10.0],
                    'solver': ['liblinear', 'lbfgs'],
                    'penalty': ['l1', 'l2']
                },
                'xgboost': {
                    'n_estimators': [100, 300, 500],
                    'max_depth': [3, 6, 9],
                    'learning_rate': [0.01, 0.1, 0.2],
                    'subsample': [0.8, 0.9, 1.0],
                    'colsample_bytree': [0.8, 0.9, 1.0]
                },
                'lightgbm': {
                    'n_estimators': [100, 300, 500],
                    'max_depth': [3, 6, 9],
                    'learning_rate': [0.01, 0.1, 0.2],
                    'feature_fraction': [0.8, 0.9, 1.0],
                    'bagging_fraction': [0.8, 0.9, 1.0]
                },
                'catboost': {
                    'iterations': [100, 300, 500],
                    'depth': [3, 6, 9],
                    'learning_rate': [0.01, 0.1, 0.2],
                    'l2_leaf_reg': [1, 3, 5]
                }
            }
        else:  # regression
            spaces = {
                'random_forest': {
                    'n_estimators': [100, 200, 300],
                    'max_depth': [5, 10, 15, None],
                    'min_samples_split': [2, 5, 10],
                    'min_samples_leaf': [1, 2, 4]
                },
                'ridge': {
                    'alpha': [0.1, 1.0, 10.0, 100.0],
                    'solver': ['auto', 'svd', 'cholesky']
                },
                'lasso': {
                    'alpha': [0.001, 0.01, 0.1, 1.0],
                    'max_iter': [1000, 5000, 10000]
                },
                'xgboost': {
                    'n_estimators': [100, 300, 500],
                    'max_depth': [3, 6, 9],
                    'learning_rate': [0.01, 0.1, 0.2],
                    'subsample': [0.8, 0.9, 1.0],
                    'colsample_bytree': [0.8, 0.9, 1.0]
                }
            }

        return spaces.get(model_name, {})

    def _model_supports_random_state(self, model_name: str) -> bool:
        """Check if model supports random_state parameter."""
        models_with_random_state = [
            'random_forest', 'extra_trees', 'logistic_regression',
            'xgboost', 'lightgbm'
        ]
        return model_name in models_with_random_state