"""
Hyperparameter optimization utilities using Optuna.
"""

from typing import Dict, List, Any, Optional, Callable
import pandas as pd
import numpy as np
from sklearn.model_selection import cross_val_score, StratifiedKFold, KFold
from dataclasses import dataclass
import warnings

warnings.filterwarnings('ignore')

# Optional Optuna integration
try:
    import optuna
    OPTUNA_AVAILABLE = True
    optuna.logging.set_verbosity(optuna.logging.WARNING)
except ImportError:
    OPTUNA_AVAILABLE = False


@dataclass
class OptimizationResult:
    """Result of hyperparameter optimization."""
    best_params: Dict[str, Any]
    best_score: float
    n_trials: int
    optimization_time: float
    study: Optional[Any] = None


class HyperparameterOptimizer:
    """Optimizes model hyperparameters using Optuna."""

    def __init__(self, n_trials: int = 20, timeout: int = 300, cv_folds: int = 5, random_state: int = 42):
        self.n_trials = n_trials
        self.timeout = timeout
        self.cv_folds = cv_folds
        self.random_state = random_state

        if not OPTUNA_AVAILABLE:
            print("Warning: Optuna not available. Hyperparameter optimization will use grid search fallback.")

    def optimize(self, model_factory_func: Callable, param_space: Dict[str, List[Any]],
                X_train: pd.DataFrame, y_train: pd.Series, problem_type: str = "auto") -> OptimizationResult:
        """Optimize hyperparameters for a model."""
        import time
        start_time = time.time()

        if problem_type == "auto":
            problem_type = self._determine_problem_type(y_train)

        if OPTUNA_AVAILABLE:
            result = self._optuna_optimize(model_factory_func, param_space, X_train, y_train, problem_type)
        else:
            result = self._grid_search_optimize(model_factory_func, param_space, X_train, y_train, problem_type)

        result.optimization_time = time.time() - start_time
        return result

    def _optuna_optimize(self, model_factory_func: Callable, param_space: Dict[str, List[Any]],
                        X_train: pd.DataFrame, y_train: pd.Series, problem_type: str) -> OptimizationResult:
        """Optimize using Optuna."""
        def objective(trial):
            # Suggest parameters
            params = {}
            for param_name, param_values in param_space.items():
                # Filter out None values
                clean_values = [v for v in param_values if v is not None]
                if not clean_values:
                    continue

                if isinstance(clean_values[0], (int, np.integer)):
                    params[param_name] = trial.suggest_int(param_name, min(clean_values), max(clean_values))
                elif isinstance(clean_values[0], (float, np.floating)):
                    params[param_name] = trial.suggest_float(param_name, min(clean_values), max(clean_values))
                else:
                    params[param_name] = trial.suggest_categorical(param_name, clean_values)

            # Create model with suggested parameters
            model = model_factory_func(params)

            # Evaluate using cross-validation
            try:
                cv_scores = self._cross_validate(model, X_train, y_train, problem_type)
                return np.mean(cv_scores)
            except Exception as e:
                print(f"Trial failed: {e}")
                return float('-inf') if problem_type == "classification" else float('inf')

        # Create study
        direction = "maximize" if problem_type == "classification" else "minimize"
        study = optuna.create_study(direction=direction, sampler=optuna.samplers.TPESampler(seed=self.random_state))

        # Optimize
        study.optimize(objective, n_trials=self.n_trials, timeout=self.timeout, show_progress_bar=False)

        return OptimizationResult(
            best_params=study.best_params,
            best_score=study.best_value,
            n_trials=len(study.trials),
            optimization_time=0.0,  # Will be set by caller
            study=study
        )

    def _grid_search_optimize(self, model_factory_func: Callable, param_space: Dict[str, List[Any]],
                            X_train: pd.DataFrame, y_train: pd.Series, problem_type: str) -> OptimizationResult:
        """Fallback grid search optimization."""
        from itertools import product

        # Generate all parameter combinations
        param_names = list(param_space.keys())
        param_values = list(param_space.values())

        best_score = float('-inf') if problem_type == "classification" else float('inf')
        best_params = {}
        n_trials = 0

        # Limit combinations to prevent explosion
        max_combinations = min(self.n_trials, 50)
        combinations = list(product(*param_values))[:max_combinations]

        for combination in combinations:
            params = dict(zip(param_names, combination))

            try:
                model = model_factory_func(params)
                cv_scores = self._cross_validate(model, X_train, y_train, problem_type)
                score = np.mean(cv_scores)

                if problem_type == "classification" and score > best_score:
                    best_score = score
                    best_params = params
                elif problem_type == "regression" and score < best_score:
                    best_score = score
                    best_params = params

                n_trials += 1

            except Exception as e:
                print(f"Parameter combination failed: {e}")
                continue

        return OptimizationResult(
            best_params=best_params,
            best_score=best_score,
            n_trials=n_trials,
            optimization_time=0.0
        )

    def _cross_validate(self, model: Any, X: pd.DataFrame, y: pd.Series, problem_type: str) -> List[float]:
        """Perform cross-validation."""
        try:
            if problem_type == "classification":
                cv = StratifiedKFold(n_splits=self.cv_folds, shuffle=True, random_state=self.random_state)
                scoring = 'accuracy'
            else:
                cv = KFold(n_splits=self.cv_folds, shuffle=True, random_state=self.random_state)
                scoring = 'neg_mean_squared_error'

            scores = cross_val_score(model, X, y, cv=cv, scoring=scoring)

            if scoring == 'neg_mean_squared_error':
                scores = -scores

            return scores.tolist()

        except Exception as e:
            print(f"Cross-validation failed: {e}")
            return [0.0] * self.cv_folds

    def _determine_problem_type(self, y: pd.Series) -> str:
        """Determine problem type from target variable."""
        unique_values = y.nunique()
        if unique_values < 20 and y.dtype in ['int64', 'object', 'bool']:
            return "classification"
        else:
            return "regression"