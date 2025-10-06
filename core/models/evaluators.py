"""
Model evaluation utilities for scoring and validation.
"""

from typing import Dict, List, Any, Optional, Tuple, Union
import pandas as pd
import numpy as np
from sklearn.model_selection import cross_val_score, StratifiedKFold, KFold
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, roc_auc_score,
    mean_squared_error, mean_absolute_error, r2_score, confusion_matrix,
    classification_report
)
from dataclasses import dataclass
import warnings

warnings.filterwarnings('ignore')


@dataclass
class EvaluationResult:
    """Result of model evaluation."""
    model_name: str
    problem_type: str
    cv_scores: List[float]
    cv_mean: float
    cv_std: float
    validation_score: Optional[float]
    train_score: Optional[float]
    metrics: Dict[str, float]
    evaluation_time: float


class ModelEvaluator:
    """Evaluates model performance using various metrics and validation strategies."""

    def __init__(self, cv_folds: int = 5, random_state: int = 42,
                 classification_metric: str = 'accuracy', regression_metric: str = 'neg_root_mean_squared_error'):
        self.cv_folds = cv_folds
        self.random_state = random_state
        self.classification_metric = classification_metric
        self.regression_metric = regression_metric  # Using RMSE to match Kaggle competitions

    def evaluate_model(self, model: Any, X_train: pd.DataFrame, y_train: pd.Series,
                      X_val: Optional[pd.DataFrame] = None, y_val: Optional[pd.Series] = None,
                      problem_type: str = "auto", model_name: str = "unknown") -> EvaluationResult:
        """Evaluate a model using cross-validation and optional validation set."""
        import time
        start_time = time.time()

        # Determine problem type if auto
        if problem_type == "auto":
            problem_type = self._determine_problem_type(y_train)

        # Perform cross-validation
        cv_scores = self._cross_validate(model, X_train, y_train, problem_type)
        cv_mean = np.mean(cv_scores)
        cv_std = np.std(cv_scores)

        # Encode target if it's categorical (string labels)
        from sklearn.preprocessing import LabelEncoder
        y_train_encoded = y_train
        y_val_encoded = y_val
        if y_train.dtype == 'object' or y_train.dtype.name == 'category':
            encoder = LabelEncoder()
            y_train_encoded = pd.Series(encoder.fit_transform(y_train), index=y_train.index)
            if y_val is not None:
                y_val_encoded = pd.Series(encoder.transform(y_val), index=y_val.index)

        # Evaluate on validation set if provided
        validation_score = None
        if X_val is not None and y_val_encoded is not None:
            model.fit(X_train, y_train_encoded)
            validation_score = self._score_model(model, X_val, y_val_encoded, problem_type)

        # Calculate train score
        train_score = None
        try:
            model.fit(X_train, y_train_encoded)
            train_score = self._score_model(model, X_train, y_train_encoded, problem_type)
        except:
            pass

        # Calculate detailed metrics
        metrics = {}
        if X_val is not None and y_val is not None:
            metrics = self._calculate_detailed_metrics(model, X_val, y_val, problem_type)

        evaluation_time = time.time() - start_time

        return EvaluationResult(
            model_name=model_name,
            problem_type=problem_type,
            cv_scores=cv_scores,
            cv_mean=cv_mean,
            cv_std=cv_std,
            validation_score=validation_score,
            train_score=train_score,
            metrics=metrics,
            evaluation_time=evaluation_time
        )

    def _cross_validate(self, model: Any, X: pd.DataFrame, y: pd.Series,
                       problem_type: str) -> List[float]:
        """Perform cross-validation."""
        try:
            # Encode target if it's categorical (string labels)
            from sklearn.preprocessing import LabelEncoder
            y_encoded = y
            if y.dtype == 'object' or y.dtype.name == 'category':
                encoder = LabelEncoder()
                y_encoded = pd.Series(encoder.fit_transform(y), index=y.index)

            # Choose appropriate cross-validation strategy and metric
            if problem_type == "classification":
                cv = StratifiedKFold(n_splits=self.cv_folds, shuffle=True, random_state=self.random_state)
                scoring = self.classification_metric
            else:
                cv = KFold(n_splits=self.cv_folds, shuffle=True, random_state=self.random_state)
                scoring = self.regression_metric

            scores = cross_val_score(model, X, y_encoded, cv=cv, scoring=scoring)

            # Convert negative metrics to positive for consistency
            # This allows us to always use "higher is better" logic
            if scoring.startswith('neg_'):
                scores = -scores

            return scores.tolist()

        except Exception as e:
            print(f"Cross-validation failed: {e}")
            return [0.0] * self.cv_folds

    def _score_model(self, model: Any, X: pd.DataFrame, y: pd.Series,
                    problem_type: str) -> float:
        """Score a fitted model."""
        try:
            y_pred = model.predict(X)

            if problem_type == "classification":
                return accuracy_score(y, y_pred)
            else:
                return r2_score(y, y_pred)

        except Exception as e:
            print(f"Scoring failed: {e}")
            return 0.0

    def _calculate_detailed_metrics(self, model: Any, X: pd.DataFrame, y: pd.Series,
                                  problem_type: str) -> Dict[str, float]:
        """Calculate detailed metrics for the model."""
        metrics = {}

        try:
            y_pred = model.predict(X)

            if problem_type == "classification":
                metrics['accuracy'] = accuracy_score(y, y_pred)

                # For binary classification
                if len(np.unique(y)) == 2:
                    metrics['precision'] = precision_score(y, y_pred, average='binary')
                    metrics['recall'] = recall_score(y, y_pred, average='binary')
                    metrics['f1'] = f1_score(y, y_pred, average='binary')

                    # ROC AUC if predict_proba is available
                    if hasattr(model, 'predict_proba'):
                        try:
                            y_proba = model.predict_proba(X)[:, 1]
                            metrics['roc_auc'] = roc_auc_score(y, y_proba)
                        except:
                            pass

                # For multiclass classification
                else:
                    metrics['precision_macro'] = precision_score(y, y_pred, average='macro')
                    metrics['recall_macro'] = recall_score(y, y_pred, average='macro')
                    metrics['f1_macro'] = f1_score(y, y_pred, average='macro')

            else:  # regression
                metrics['r2'] = r2_score(y, y_pred)
                metrics['mse'] = mean_squared_error(y, y_pred)
                metrics['rmse'] = np.sqrt(mean_squared_error(y, y_pred))
                metrics['mae'] = mean_absolute_error(y, y_pred)

        except Exception as e:
            print(f"Detailed metrics calculation failed: {e}")

        return metrics

    def _determine_problem_type(self, y: pd.Series) -> str:
        """Automatically determine if problem is classification or regression."""
        unique_values = y.nunique()

        # If target has very few unique values, likely classification
        if unique_values < 20 and y.dtype in ['int64', 'object', 'bool']:
            return "classification"
        else:
            return "regression"

    def compare_models(self, evaluation_results: List[EvaluationResult]) -> pd.DataFrame:
        """Compare multiple model evaluation results."""
        comparison_data = []

        for result in evaluation_results:
            row = {
                'Model': result.model_name,
                'Problem_Type': result.problem_type,
                'CV_Mean': result.cv_mean,
                'CV_Std': result.cv_std,
                'Validation_Score': result.validation_score,
                'Train_Score': result.train_score,
                'Evaluation_Time': result.evaluation_time
            }

            # Add specific metrics
            row.update(result.metrics)
            comparison_data.append(row)

        comparison_df = pd.DataFrame(comparison_data)

        # Sort by CV_Mean (descending for classification accuracy, ascending for regression MSE)
        if len(evaluation_results) > 0:
            if evaluation_results[0].problem_type == "classification":
                comparison_df = comparison_df.sort_values('CV_Mean', ascending=False)
            else:
                comparison_df = comparison_df.sort_values('CV_Mean', ascending=True)

        return comparison_df

    def get_best_model_result(self, evaluation_results: List[EvaluationResult]) -> Optional[EvaluationResult]:
        """Get the best performing model from evaluation results."""
        if not evaluation_results:
            return None

        # Determine sorting criteria based on problem type
        if evaluation_results[0].problem_type == "classification":
            # Higher is better for classification
            best_result = max(evaluation_results, key=lambda x: x.cv_mean)
        else:
            # Lower is better for regression (assuming MSE-based scoring)
            best_result = min(evaluation_results, key=lambda x: x.cv_mean)

        return best_result

    def create_evaluation_report(self, evaluation_results: List[EvaluationResult]) -> Dict[str, Any]:
        """Create a comprehensive evaluation report."""
        if not evaluation_results:
            return {'error': 'No evaluation results provided'}

        comparison_df = self.compare_models(evaluation_results)
        best_result = self.get_best_model_result(evaluation_results)

        report = {
            'summary': {
                'total_models_evaluated': len(evaluation_results),
                'problem_type': evaluation_results[0].problem_type,
                'cv_folds': self.cv_folds,
                'best_model': best_result.model_name if best_result else None,
                'best_cv_score': best_result.cv_mean if best_result else None
            },
            'comparison_table': comparison_df.to_dict('records'),
            'detailed_results': [
                {
                    'model_name': result.model_name,
                    'cv_scores': result.cv_scores,
                    'cv_mean': result.cv_mean,
                    'cv_std': result.cv_std,
                    'metrics': result.metrics
                }
                for result in evaluation_results
            ]
        }

        return report