"""
Feature selection utilities for choosing the most relevant features.

This module provides the FeatureSelector class which offers multiple
feature selection strategies:
- Variance-based filtering (remove low-variance features)
- Correlation-based filtering (remove highly correlated features)
- Univariate statistical tests (f_classif, f_regression)
- Model-based selection (Random Forest importance)
- Recursive feature elimination (RFE)
- Distribution stability checking (train/test consistency)

All methods support both classification and regression tasks.
"""

from typing import List, Dict, Any, Optional, Tuple
import pandas as pd
import numpy as np
from sklearn.feature_selection import (
    SelectKBest, f_classif, f_regression, mutual_info_classif, mutual_info_regression,
    RFE, SelectFromModel, VarianceThreshold
)
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LassoCV, LogisticRegression
from utils.logging import verbose_print
import warnings

warnings.filterwarnings('ignore')


class FeatureSelector:
    """Selects the most relevant features using various methods.

    Provides multiple feature selection strategies with configurable thresholds.
    Maintains state about selected features and their scores for analysis.

    Attributes:
        correlation_threshold: Maximum correlation allowed between features (default 0.95)
        variance_threshold: Minimum variance required for features (default 0.01)
        selected_features: List of selected feature names
        feature_scores: Dictionary mapping features to their importance scores
        selection_methods: Dictionary tracking which method selected each feature
        variance_selector: Fitted VarianceThreshold selector
        selected_feature_names: Cached list of selected feature names
        distribution_stats: Statistics about train/test distribution differences
        unstable_features: Features removed due to unstable distributions
    """

    def __init__(self, correlation_threshold: float = 0.95, variance_threshold: float = 0.01):
        """Initialize feature selector with thresholds.

        Args:
            correlation_threshold: Max correlation for feature pairs (0-1)
            variance_threshold: Min variance required for features
        """
        self.correlation_threshold = correlation_threshold
        self.variance_threshold = variance_threshold
        self.selected_features = []
        self.feature_scores = {}
        self.selection_methods = {}
        self.variance_selector = None
        self.selected_feature_names = None
        self.distribution_stats = {}
        self.unstable_features = []

    def remove_low_variance_features(self, df: pd.DataFrame, target_col: Optional[str] = None, fit: bool = True) -> pd.DataFrame:
        """Remove features with low variance."""
        feature_cols = [col for col in df.columns if col != target_col]

        if len(feature_cols) == 0:
            return df

        if fit:
            verbose_print("Removing low variance features...")

        numerical_cols = df[feature_cols].select_dtypes(include=[np.number]).columns

        if len(numerical_cols) == 0:
            return df

        if fit:
            # Fit the variance selector
            self.variance_selector = VarianceThreshold(threshold=self.variance_threshold)
            self.variance_selector.fit(df[numerical_cols])
            selected_feature_mask = self.variance_selector.get_support()
            selected_numerical_cols = numerical_cols[selected_feature_mask]

            # Store selected feature names
            non_numerical_cols = [col for col in feature_cols if col not in numerical_cols]
            self.selected_feature_names = list(selected_numerical_cols) + non_numerical_cols

            removed_count = len(numerical_cols) - len(selected_numerical_cols)
            if removed_count > 0:
                verbose_print(f"Removed {removed_count} low variance features")
        else:
            # Use fitted selector
            if self.variance_selector is None or self.selected_feature_names is None:
                return df  # Return as-is if not fitted

        # Apply selection
        selected_cols = [col for col in self.selected_feature_names if col in df.columns]
        if target_col and target_col in df.columns:
            selected_cols.append(target_col)

        return df[selected_cols]

    def remove_highly_correlated_features(self, df: pd.DataFrame, target_col: Optional[str] = None) -> pd.DataFrame:
        """Remove highly correlated features."""
        feature_cols = [col for col in df.columns if col != target_col]
        numerical_cols = df[feature_cols].select_dtypes(include=[np.number]).columns

        if len(numerical_cols) < 2:
            return df

        verbose_print("Removing highly correlated features...")

        # Calculate correlation matrix
        corr_matrix = df[numerical_cols].corr().abs()

        # Find pairs of highly correlated features
        upper_triangle = corr_matrix.where(
            np.triu(np.ones(corr_matrix.shape), k=1).astype(bool)
        )

        # Find features to drop
        to_drop = [column for column in upper_triangle.columns
                  if any(upper_triangle[column] > self.correlation_threshold)]

        # Keep all non-numerical columns and target
        columns_to_keep = [col for col in df.columns if col not in to_drop]

        if len(to_drop) > 0:
            verbose_print(f"Removed {len(to_drop)} highly correlated features")

        return df[columns_to_keep]

    def select_univariate_features(self, df: pd.DataFrame, target_col: str,
                                 k: int = 20, problem_type: str = "auto") -> pd.DataFrame:
        """Select features using univariate statistical tests."""
        if target_col not in df.columns:
            return df

        feature_cols = [col for col in df.columns if col != target_col]

        # Remove ID-like columns that have no predictive value
        id_columns = ['PassengerId', 'id', 'Id', 'ID', 'index']
        feature_cols = [col for col in feature_cols if col not in id_columns]

        numerical_feature_cols = df[feature_cols].select_dtypes(include=[np.number]).columns

        if len(numerical_feature_cols) == 0:
            return df

        verbose_print(f"Selecting top {k} features using univariate tests...")

        # Determine problem type
        if problem_type == "auto":
            unique_values = df[target_col].nunique()
            problem_type = "classification" if unique_values < 20 else "regression"

        # Choose appropriate scoring function
        if problem_type == "classification":
            score_func = f_classif
        else:
            score_func = f_regression

        # Select features
        selector = SelectKBest(score_func=score_func, k=min(k, len(numerical_feature_cols)))

        try:
            selected_features = selector.fit_transform(df[numerical_feature_cols], df[target_col])
            selected_feature_mask = selector.get_support()
            selected_feature_names = numerical_feature_cols[selected_feature_mask]

            # Store feature scores
            feature_scores = selector.scores_
            for i, feature in enumerate(numerical_feature_cols):
                self.feature_scores[feature] = feature_scores[i]
                if feature in selected_feature_names:
                    self.selection_methods[feature] = "univariate"

            # Combine with non-numerical features and target
            non_numerical_cols = [col for col in feature_cols if col not in numerical_feature_cols]
            final_columns = list(selected_feature_names) + non_numerical_cols + [target_col]

            verbose_print(f"Selected {len(selected_feature_names)} numerical features")
            return df[final_columns]

        except Exception as e:
            verbose_print(f"Error in univariate selection: {e}")
            return df

    def select_model_based_features(self, df: pd.DataFrame, target_col: str,
                                  max_features: int = 20, problem_type: str = "auto") -> pd.DataFrame:
        """Select features using model-based importance."""
        if target_col not in df.columns:
            return df

        feature_cols = [col for col in df.columns if col != target_col]

        # Remove ID-like columns that have no predictive value
        id_columns = ['PassengerId', 'id', 'Id', 'ID', 'index']
        feature_cols = [col for col in feature_cols if col not in id_columns]

        numerical_feature_cols = df[feature_cols].select_dtypes(include=[np.number]).columns

        if len(numerical_feature_cols) == 0:
            return df

        verbose_print(f"Selecting top {max_features} features using model importance...")

        # Determine problem type
        if problem_type == "auto":
            unique_values = df[target_col].nunique()
            problem_type = "classification" if unique_values < 20 else "regression"

        try:
            # Choose appropriate model
            if problem_type == "classification":
                model = RandomForestClassifier(n_estimators=50, random_state=42)
            else:
                model = RandomForestRegressor(n_estimators=50, random_state=42)

            # Fit model and get feature importances
            model.fit(df[numerical_feature_cols], df[target_col])
            feature_importances = model.feature_importances_

            # Create feature importance ranking
            feature_importance_pairs = list(zip(numerical_feature_cols, feature_importances))
            feature_importance_pairs.sort(key=lambda x: x[1], reverse=True)

            # Select top features
            n_features = min(max_features, len(feature_importance_pairs))
            selected_features = [pair[0] for pair in feature_importance_pairs[:n_features]]

            # Store feature scores
            for feature, importance in feature_importance_pairs:
                self.feature_scores[feature] = importance
                if feature in selected_features:
                    self.selection_methods[feature] = "model_based"

            # Combine with non-numerical features and target
            non_numerical_cols = [col for col in feature_cols if col not in numerical_feature_cols]
            final_columns = selected_features + non_numerical_cols + [target_col]

            verbose_print(f"Selected {len(selected_features)} features based on model importance")
            return df[final_columns]

        except Exception as e:
            verbose_print(f"Error in model-based selection: {e}")
            return df

    def select_recursive_features(self, df: pd.DataFrame, target_col: str,
                                n_features: int = 15, problem_type: str = "auto") -> pd.DataFrame:
        """Select features using recursive feature elimination."""
        if target_col not in df.columns:
            return df

        feature_cols = [col for col in df.columns if col != target_col]

        # Remove ID-like columns that have no predictive value
        id_columns = ['PassengerId', 'id', 'Id', 'ID', 'index']
        feature_cols = [col for col in feature_cols if col not in id_columns]

        numerical_feature_cols = df[feature_cols].select_dtypes(include=[np.number]).columns

        if len(numerical_feature_cols) == 0:
            return df

        verbose_print(f"Selecting {n_features} features using recursive elimination...")

        # Determine problem type
        if problem_type == "auto":
            unique_values = df[target_col].nunique()
            problem_type = "classification" if unique_values < 20 else "regression"

        try:
            # Choose appropriate estimator
            if problem_type == "classification":
                estimator = LogisticRegression(random_state=42, max_iter=1000)
            else:
                estimator = LassoCV(random_state=42, max_iter=1000)

            # Apply RFE
            selector = RFE(estimator=estimator, n_features_to_select=min(n_features, len(numerical_feature_cols)))
            selector.fit(df[numerical_feature_cols], df[target_col])

            # Get selected features
            selected_feature_mask = selector.get_support()
            selected_features = numerical_feature_cols[selected_feature_mask]

            # Store selection information
            for i, feature in enumerate(numerical_feature_cols):
                ranking = selector.ranking_[i]
                self.feature_scores[feature] = 1.0 / ranking  # Higher score for lower ranking
                if feature in selected_features:
                    self.selection_methods[feature] = "recursive_elimination"

            # Combine with non-numerical features and target
            non_numerical_cols = [col for col in feature_cols if col not in numerical_feature_cols]
            final_columns = list(selected_features) + non_numerical_cols + [target_col]

            verbose_print(f"Selected {len(selected_features)} features using RFE")
            return df[final_columns]

        except Exception as e:
            verbose_print(f"Error in recursive feature elimination: {e}")
            return df

    def check_distribution_stability(self, train_df: pd.DataFrame, test_df: pd.DataFrame,
                                    target_col: Optional[str] = None,
                                    mean_threshold: float = 0.5, std_ratio_threshold: float = 2.0) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Remove features with unstable distributions between train and test sets.

        Args:
            train_df: Training dataframe
            test_df: Test dataframe
            target_col: Target column to exclude
            mean_threshold: Maximum normalized mean difference allowed
            std_ratio_threshold: Maximum ratio of std deviations allowed

        Returns:
            Tuple of (train_df, test_df) with only stable features
        """
        feature_cols = [col for col in train_df.columns if col != target_col]
        numerical_cols = train_df[feature_cols].select_dtypes(include=[np.number]).columns

        if len(numerical_cols) == 0:
            return train_df, test_df

        verbose_print("Checking train/test distribution stability...")

        stable_features = []
        self.unstable_features = []

        for col in numerical_cols:
            if col not in test_df.columns:
                stable_features.append(col)
                continue

            train_values = train_df[col].dropna()
            test_values = test_df[col].dropna()

            if len(train_values) == 0 or len(test_values) == 0:
                stable_features.append(col)
                continue

            # Calculate statistics
            train_mean = train_values.mean()
            test_mean = test_values.mean()
            train_std = train_values.std()
            test_std = test_values.std()

            # Avoid division by zero
            if train_std == 0 or test_std == 0:
                if train_mean == test_mean:
                    stable_features.append(col)
                else:
                    self.unstable_features.append(col)
                continue

            # Calculate differences
            mean_diff = abs(train_mean - test_mean)
            std_ratio = max(train_std, test_std) / min(train_std, test_std)
            normalized_mean_diff = mean_diff / train_std

            # Store statistics
            self.distribution_stats[col] = {
                'train_mean': train_mean,
                'test_mean': test_mean,
                'mean_diff': mean_diff,
                'normalized_mean_diff': normalized_mean_diff,
                'train_std': train_std,
                'test_std': test_std,
                'std_ratio': std_ratio
            }

            # Keep feature if distributions are similar
            if normalized_mean_diff < mean_threshold and std_ratio < std_ratio_threshold:
                stable_features.append(col)
            else:
                self.unstable_features.append(col)
                verbose_print(f"  Removing {col}: norm_mean_diff={normalized_mean_diff:.2f}, std_ratio={std_ratio:.2f}")

        # Combine stable features with non-numerical features and target
        non_numerical_cols = [col for col in feature_cols if col not in numerical_cols]
        final_columns = stable_features + non_numerical_cols
        if target_col and target_col in train_df.columns:
            final_columns.append(target_col)

        # Prepare test columns (exclude target, only keep columns that exist in test)
        test_final_columns = [col for col in final_columns if col != target_col and col in test_df.columns]

        if len(self.unstable_features) > 0:
            verbose_print(f"Removed {len(self.unstable_features)} unstable features")

        return train_df[final_columns], test_df[test_final_columns]

    def get_selection_report(self) -> Dict[str, Any]:
        """Get a report of feature selection results."""
        return {
            'total_features_evaluated': len(self.feature_scores),
            'selected_features': self.selected_features,
            'feature_scores': self.feature_scores,
            'selection_methods': self.selection_methods,
            'top_features': sorted(self.feature_scores.items(), key=lambda x: x[1], reverse=True)[:10],
            'distribution_stats': self.distribution_stats,
            'unstable_features': self.unstable_features
        }