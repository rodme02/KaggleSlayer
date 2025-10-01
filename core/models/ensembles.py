"""
Ensemble building utilities for combining multiple models.
"""

from typing import List, Dict, Any, Optional, Tuple
import pandas as pd
import numpy as np
from sklearn.ensemble import VotingClassifier, VotingRegressor, StackingClassifier, StackingRegressor
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.model_selection import cross_val_score
from dataclasses import dataclass
import warnings

warnings.filterwarnings('ignore')


@dataclass
class EnsembleResult:
    """Result of ensemble creation."""
    ensemble_model: Any
    member_models: List[Tuple[str, Any]]
    ensemble_score: float
    member_scores: Dict[str, float]
    weights: Optional[List[float]] = None


class EnsembleBuilder:
    """Builds ensemble models from multiple base models."""

    def __init__(self, cv_folds: int = 5, random_state: int = 42):
        self.cv_folds = cv_folds
        self.random_state = random_state

    def create_voting_ensemble(self, models: List[Tuple[str, Any]], X_train: pd.DataFrame,
                             y_train: pd.Series, problem_type: str = "auto",
                             voting: str = "auto") -> EnsembleResult:
        """Create a voting ensemble from multiple models."""
        if problem_type == "auto":
            problem_type = self._determine_problem_type(y_train)

        # Determine voting strategy
        if voting == "auto":
            if problem_type == "classification":
                # Use soft voting if all models support predict_proba
                if all(hasattr(model, 'predict_proba') for _, model in models):
                    voting = 'soft'
                else:
                    voting = 'hard'
            else:
                voting = 'hard'  # Only option for regression

        print(f"Creating {voting} voting ensemble with {len(models)} models...")

        # Evaluate individual models
        member_scores = {}
        for name, model in models:
            try:
                scores = self._cross_validate(model, X_train, y_train, problem_type)
                member_scores[name] = np.mean(scores)
                print(f"  {name}: {member_scores[name]:.4f}")
            except Exception as e:
                print(f"  {name}: Failed to evaluate - {e}")
                member_scores[name] = 0.0

        # Create ensemble
        if problem_type == "classification":
            ensemble = VotingClassifier(estimators=models, voting=voting)
        else:
            ensemble = VotingRegressor(estimators=models)

        # Evaluate ensemble
        try:
            ensemble_scores = self._cross_validate(ensemble, X_train, y_train, problem_type)
            ensemble_score = np.mean(ensemble_scores)
            print(f"Ensemble score: {ensemble_score:.4f}")
        except Exception as e:
            print(f"Ensemble evaluation failed: {e}")
            ensemble_score = 0.0

        # Fit ensemble on full training data
        print("Fitting ensemble on full training data...")
        ensemble.fit(X_train, y_train)

        return EnsembleResult(
            ensemble_model=ensemble,
            member_models=models,
            ensemble_score=ensemble_score,
            member_scores=member_scores
        )

    def create_weighted_ensemble(self, models: List[Tuple[str, Any]], X_train: pd.DataFrame,
                               y_train: pd.Series, problem_type: str = "auto") -> EnsembleResult:
        """Create a weighted ensemble based on individual model performance."""
        if problem_type == "auto":
            problem_type = self._determine_problem_type(y_train)

        print(f"Creating weighted ensemble with {len(models)} models...")

        # Evaluate individual models and calculate weights
        member_scores = {}
        scores = []

        for name, model in models:
            try:
                cv_scores = self._cross_validate(model, X_train, y_train, problem_type)
                score = np.mean(cv_scores)
                member_scores[name] = score
                scores.append(score)
                print(f"  {name}: {score:.4f}")
            except Exception as e:
                print(f"  {name}: Failed to evaluate - {e}")
                member_scores[name] = 0.0
                scores.append(0.0)

        # Calculate weights based on performance
        if problem_type == "classification":
            # Higher scores are better
            weights = np.array(scores) / np.sum(scores) if np.sum(scores) > 0 else np.ones(len(scores)) / len(scores)
        else:
            # Lower scores (MSE) are better, so invert
            inv_scores = 1.0 / (np.array(scores) + 1e-8)
            weights = inv_scores / np.sum(inv_scores)

        weights = weights.tolist()

        # Create weighted ensemble
        models_with_weights = [(name, model) for name, model in models]

        if problem_type == "classification":
            ensemble = WeightedVotingClassifier(estimators=models_with_weights, weights=weights)
        else:
            ensemble = WeightedVotingRegressor(estimators=models_with_weights, weights=weights)

        # Evaluate ensemble
        try:
            ensemble_scores = self._cross_validate(ensemble, X_train, y_train, problem_type)
            ensemble_score = np.mean(ensemble_scores)
            print(f"Weighted ensemble score: {ensemble_score:.4f}")
        except Exception as e:
            print(f"Weighted ensemble evaluation failed: {e}")
            ensemble_score = 0.0

        # Fit ensemble on full training data
        print("Fitting weighted ensemble on full training data...")
        ensemble.fit(X_train, y_train)

        return EnsembleResult(
            ensemble_model=ensemble,
            member_models=models,
            ensemble_score=ensemble_score,
            member_scores=member_scores,
            weights=weights
        )

    def create_stacking_ensemble(self, models: List[Tuple[str, Any]], X_train: pd.DataFrame,
                                y_train: pd.Series, problem_type: str = "auto") -> EnsembleResult:
        """Create a stacking ensemble with a meta-learner."""
        if problem_type == "auto":
            problem_type = self._determine_problem_type(y_train)

        print(f"Creating stacking ensemble with {len(models)} base models...")

        # Evaluate individual models
        member_scores = {}
        for name, model in models:
            try:
                scores = self._cross_validate(model, X_train, y_train, problem_type)
                member_scores[name] = np.mean(scores)
                print(f"  {name}: {member_scores[name]:.4f}")
            except Exception as e:
                print(f"  {name}: Failed to evaluate - {e}")
                member_scores[name] = 0.0

        # Create stacking ensemble with appropriate meta-learner
        if problem_type == "classification":
            meta_learner = LogisticRegression(max_iter=2000, C=1.0)
            ensemble = StackingClassifier(
                estimators=models,
                final_estimator=meta_learner,
                cv=3,
                stack_method='auto'
            )
        else:
            meta_learner = Ridge(alpha=1.0)
            ensemble = StackingRegressor(
                estimators=models,
                final_estimator=meta_learner,
                cv=3
            )

        # Evaluate stacking ensemble
        try:
            ensemble_scores = self._cross_validate(ensemble, X_train, y_train, problem_type)
            ensemble_score = np.mean(ensemble_scores)
            print(f"Stacking ensemble score: {ensemble_score:.4f}")
        except Exception as e:
            print(f"Stacking ensemble evaluation failed: {e}")
            ensemble_score = 0.0

        # Fit ensemble on full training data
        print("Fitting stacking ensemble on full training data...")
        ensemble.fit(X_train, y_train)

        return EnsembleResult(
            ensemble_model=ensemble,
            member_models=models,
            ensemble_score=ensemble_score,
            member_scores=member_scores
        )

    def select_best_models(self, model_results: List[Tuple[str, Any, float]],
                         top_n: int = 3, min_score_threshold: Optional[float] = None) -> List[Tuple[str, Any]]:
        """Select the best performing models for ensemble."""
        # Sort by score (descending for classification, ascending for regression)
        # Assuming higher scores are better (accuracy, r2) and lower are worse (mse)
        sorted_results = sorted(model_results, key=lambda x: x[2], reverse=True)

        # Filter by minimum score threshold if provided
        if min_score_threshold is not None:
            sorted_results = [r for r in sorted_results if r[2] >= min_score_threshold]

        # Select top N models
        selected = sorted_results[:top_n]
        selected_models = [(name, model) for name, model, _ in selected]

        print(f"Selected {len(selected_models)} models for ensemble:")
        for name, _, score in selected:
            print(f"  {name}: {score:.4f}")

        return selected_models

    def _cross_validate(self, model: Any, X: pd.DataFrame, y: pd.Series, problem_type: str) -> List[float]:
        """Perform cross-validation."""
        try:
            from sklearn.model_selection import StratifiedKFold, KFold

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


class WeightedVotingClassifier:
    """Custom weighted voting classifier."""

    def __init__(self, estimators: List[Tuple[str, Any]], weights: List[float]):
        self.estimators = estimators
        self.weights = weights

    def fit(self, X, y):
        for _, estimator in self.estimators:
            estimator.fit(X, y)
        return self

    def predict(self, X):
        predictions = []
        for _, estimator in self.estimators:
            predictions.append(estimator.predict(X))

        # Weighted voting
        weighted_predictions = np.zeros(len(X))
        for i, (pred, weight) in enumerate(zip(predictions, self.weights)):
            weighted_predictions += pred * weight

        return np.round(weighted_predictions).astype(int)

    def predict_proba(self, X):
        if not all(hasattr(est, 'predict_proba') for _, est in self.estimators):
            raise AttributeError("Not all estimators support predict_proba")

        probabilities = []
        for _, estimator in self.estimators:
            probabilities.append(estimator.predict_proba(X))

        # Weighted average of probabilities
        weighted_proba = np.zeros_like(probabilities[0])
        for proba, weight in zip(probabilities, self.weights):
            weighted_proba += proba * weight

        return weighted_proba


class WeightedVotingRegressor:
    """Custom weighted voting regressor."""

    def __init__(self, estimators: List[Tuple[str, Any]], weights: List[float]):
        self.estimators = estimators
        self.weights = weights

    def fit(self, X, y):
        for _, estimator in self.estimators:
            estimator.fit(X, y)
        return self

    def predict(self, X):
        predictions = []
        for _, estimator in self.estimators:
            predictions.append(estimator.predict(X))

        # Weighted average
        weighted_predictions = np.zeros(len(X))
        for pred, weight in zip(predictions, self.weights):
            weighted_predictions += pred * weight

        return weighted_predictions