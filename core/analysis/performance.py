"""
Performance analysis utilities for model evaluation and improvement identification.
"""

from typing import Dict, List, Any, Optional, Tuple
import pandas as pd
import numpy as np
from sklearn.metrics import confusion_matrix, classification_report
from dataclasses import dataclass
import warnings

warnings.filterwarnings('ignore')


@dataclass
class PerformanceAnalysis:
    """Comprehensive performance analysis results."""
    model_name: str
    current_score: float
    performance_trend: List[float]
    error_analysis: Dict[str, Any]
    feature_importance: Dict[str, float]
    improvement_suggestions: List[str]
    analysis_timestamp: str


class PerformanceAnalyzer:
    """Analyzes model performance and identifies improvement opportunities."""

    def __init__(self):
        self.analysis_history = []

    def analyze_model_performance(self, model: Any, X_train: pd.DataFrame, y_train: pd.Series,
                                X_val: pd.DataFrame, y_val: pd.Series,
                                model_name: str = "unknown", problem_type: str = "auto") -> PerformanceAnalysis:
        """Perform comprehensive performance analysis."""
        from datetime import datetime

        if problem_type == "auto":
            problem_type = self._determine_problem_type(y_val)

        # Get predictions
        train_pred = model.predict(X_train)
        val_pred = model.predict(X_val)

        # Calculate scores
        if problem_type == "classification":
            from sklearn.metrics import accuracy_score
            train_score = accuracy_score(y_train, train_pred)
            val_score = accuracy_score(y_val, val_pred)
        else:
            from sklearn.metrics import r2_score
            train_score = r2_score(y_train, train_pred)
            val_score = r2_score(y_val, val_pred)

        # Error analysis
        error_analysis = self._analyze_errors(y_val, val_pred, problem_type)

        # Feature importance
        feature_importance = self._get_feature_importance(model, X_train.columns)

        # Generate improvement suggestions
        improvement_suggestions = self._generate_improvement_suggestions(
            train_score, val_score, error_analysis, feature_importance, problem_type
        )

        # Track performance trend
        performance_trend = self._update_performance_trend(model_name, val_score)

        analysis = PerformanceAnalysis(
            model_name=model_name,
            current_score=val_score,
            performance_trend=performance_trend,
            error_analysis=error_analysis,
            feature_importance=feature_importance,
            improvement_suggestions=improvement_suggestions,
            analysis_timestamp=datetime.now().isoformat()
        )

        self.analysis_history.append(analysis)
        return analysis

    def _analyze_errors(self, y_true: pd.Series, y_pred: np.ndarray, problem_type: str) -> Dict[str, Any]:
        """Analyze prediction errors."""
        error_analysis = {}

        if problem_type == "classification":
            # Confusion matrix analysis
            cm = confusion_matrix(y_true, y_pred)
            error_analysis['confusion_matrix'] = cm.tolist()

            # Classification report
            report = classification_report(y_true, y_pred, output_dict=True)
            error_analysis['classification_report'] = report

            # Most confused classes
            if cm.shape[0] > 2:  # Multiclass
                most_confused = []
                for i in range(cm.shape[0]):
                    for j in range(cm.shape[1]):
                        if i != j and cm[i, j] > 0:
                            most_confused.append((i, j, cm[i, j]))
                most_confused.sort(key=lambda x: x[2], reverse=True)
                error_analysis['most_confused_pairs'] = most_confused[:5]

        else:  # Regression
            residuals = y_true - y_pred
            error_analysis['mean_residual'] = float(np.mean(residuals))
            error_analysis['std_residual'] = float(np.std(residuals))
            error_analysis['max_absolute_error'] = float(np.max(np.abs(residuals)))

            # Identify outliers (large residuals)
            threshold = np.std(residuals) * 2
            outlier_indices = np.where(np.abs(residuals) > threshold)[0]
            error_analysis['outlier_count'] = len(outlier_indices)
            error_analysis['outlier_percentage'] = (len(outlier_indices) / len(residuals)) * 100

        return error_analysis

    def _get_feature_importance(self, model: Any, feature_names: List[str]) -> Dict[str, float]:
        """Extract feature importance from the model."""
        importance_dict = {}

        try:
            if hasattr(model, 'feature_importances_'):
                importances = model.feature_importances_
                for name, importance in zip(feature_names, importances):
                    importance_dict[name] = float(importance)
            elif hasattr(model, 'coef_'):
                # For linear models
                coef = model.coef_
                if coef.ndim > 1:
                    coef = np.mean(np.abs(coef), axis=0)
                else:
                    coef = np.abs(coef)
                for name, importance in zip(feature_names, coef):
                    importance_dict[name] = float(importance)
        except Exception as e:
            print(f"Could not extract feature importance: {e}")

        return importance_dict

    def _generate_improvement_suggestions(self, train_score: float, val_score: float,
                                        error_analysis: Dict[str, Any],
                                        feature_importance: Dict[str, float],
                                        problem_type: str) -> List[str]:
        """Generate suggestions for model improvement."""
        suggestions = []

        # Overfitting detection
        score_diff = train_score - val_score
        if problem_type == "classification":
            if score_diff > 0.1:
                suggestions.append("Model appears to be overfitting. Consider regularization or reducing model complexity.")
        else:
            if score_diff > 0.2:
                suggestions.append("Model shows signs of overfitting. Try L1/L2 regularization or feature selection.")

        # Underfitting detection
        if problem_type == "classification" and val_score < 0.7:
            suggestions.append("Low validation score suggests underfitting. Consider more complex models or feature engineering.")
        elif problem_type == "regression" and val_score < 0.5:
            suggestions.append("Low R² score indicates poor fit. Consider polynomial features or different algorithms.")

        # Feature importance analysis
        if feature_importance:
            sorted_features = sorted(feature_importance.items(), key=lambda x: x[1], reverse=True)
            top_features = sorted_features[:5]
            low_importance_features = [f for f, imp in sorted_features if imp < 0.01]

            if len(low_importance_features) > 5:
                suggestions.append(f"Consider removing {len(low_importance_features)} low-importance features to reduce noise.")

            if len(sorted_features) > 20:
                suggestions.append("High feature count detected. Consider feature selection to improve performance.")

        # Error analysis suggestions
        if problem_type == "classification":
            if 'outlier_percentage' in error_analysis and error_analysis['outlier_percentage'] > 10:
                suggestions.append("High number of prediction outliers. Consider data cleaning or outlier detection.")

            if 'most_confused_pairs' in error_analysis and len(error_analysis['most_confused_pairs']) > 0:
                suggestions.append("Certain classes are frequently confused. Consider feature engineering for these specific classes.")

        else:  # regression
            if 'outlier_percentage' in error_analysis and error_analysis['outlier_percentage'] > 15:
                suggestions.append("High residual outliers detected. Consider robust regression methods or outlier removal.")

        # General suggestions
        if val_score > 0 and len(suggestions) == 0:
            suggestions.append("Model performance is reasonable. Consider ensemble methods for further improvement.")

        return suggestions

    def _update_performance_trend(self, model_name: str, current_score: float) -> List[float]:
        """Update and return performance trend for the model."""
        # Find existing trend for this model
        model_scores = []
        for analysis in self.analysis_history:
            if analysis.model_name == model_name:
                model_scores.append(analysis.current_score)

        # Add current score
        model_scores.append(current_score)

        # Keep only last 10 scores
        return model_scores[-10:]

    def _determine_problem_type(self, y: pd.Series) -> str:
        """Determine problem type from target variable."""
        unique_values = y.nunique()
        if unique_values < 20 and y.dtype in ['int64', 'object', 'bool']:
            return "classification"
        else:
            return "regression"

    def get_performance_summary(self) -> Dict[str, Any]:
        """Get summary of all performance analyses."""
        if not self.analysis_history:
            return {'message': 'No performance analyses available'}

        latest_analyses = {}
        for analysis in self.analysis_history:
            latest_analyses[analysis.model_name] = analysis

        summary = {
            'total_models_analyzed': len(latest_analyses),
            'best_performing_model': max(latest_analyses.values(), key=lambda x: x.current_score).model_name,
            'worst_performing_model': min(latest_analyses.values(), key=lambda x: x.current_score).model_name,
            'average_score': np.mean([a.current_score for a in latest_analyses.values()]),
            'latest_analyses': {name: {
                'score': analysis.current_score,
                'suggestions_count': len(analysis.improvement_suggestions),
                'timestamp': analysis.analysis_timestamp
            } for name, analysis in latest_analyses.items()}
        }

        return summary