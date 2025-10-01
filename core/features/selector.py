"""
Feature selection utilities for identifying and removing problematic features.
"""

from typing import List, Tuple, Dict, Any, Optional
import pandas as pd
import numpy as np
from sklearn.feature_selection import SelectKBest, f_classif, f_regression, mutual_info_classif, mutual_info_regression
from scipy import stats
import warnings

warnings.filterwarnings('ignore')


class FeatureSelector:
    """Selects the most important features and removes problematic ones."""

    def __init__(self, problem_type: str = "classification", max_features: Optional[int] = None):
        """
        Args:
            problem_type: "classification" or "regression"
            max_features: Maximum number of features to keep (None = no limit)
        """
        self.problem_type = problem_type
        self.max_features = max_features
        self.selected_features = []
        self.removed_features = []
        self.feature_scores = {}
        self.distribution_stats = {}

    def select_features(self, X_train: pd.DataFrame, y_train: pd.Series,
                       X_test: Optional[pd.DataFrame] = None,
                       method: str = "auto") -> pd.DataFrame:
        """
        Select the best features from training data.

        Args:
            X_train: Training features
            y_train: Training target
            X_test: Test features (optional, for distribution check)
            method: Selection method ("auto", "kbest", "mutual_info", "distribution_check")

        Returns:
            X_train with selected features only
        """
        if method == "auto":
            # Use multiple methods
            features_to_keep = set(X_train.columns)

            # 1. Remove features with train/test distribution mismatch (if test provided)
            if X_test is not None:
                stable_features = self._check_distribution_stability(X_train, X_test)
                features_to_keep &= set(stable_features)
                print(f"After distribution check: {len(features_to_keep)} features")

            # 2. Remove low-variance features
            variance_features = self._remove_low_variance(X_train)
            features_to_keep &= set(variance_features)
            print(f"After variance check: {len(features_to_keep)} features")

            # 3. Use statistical feature selection if we still have too many features
            if self.max_features and len(features_to_keep) > self.max_features:
                X_train_filtered = X_train[list(features_to_keep)]
                kbest_features = self._select_kbest(X_train_filtered, y_train, k=self.max_features)
                features_to_keep &= set(kbest_features)
                print(f"After KBest selection: {len(features_to_keep)} features")

            self.selected_features = list(features_to_keep)
            self.removed_features = [col for col in X_train.columns if col not in features_to_keep]

            return X_train[self.selected_features]

        elif method == "kbest":
            k = self.max_features or min(50, len(X_train.columns))
            self.selected_features = self._select_kbest(X_train, y_train, k=k)
            return X_train[self.selected_features]

        elif method == "mutual_info":
            k = self.max_features or min(50, len(X_train.columns))
            self.selected_features = self._select_mutual_info(X_train, y_train, k=k)
            return X_train[self.selected_features]

        else:
            raise ValueError(f"Unknown selection method: {method}")

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Transform data using previously selected features."""
        if not self.selected_features:
            raise ValueError("No features selected. Call select_features() first.")

        # Only keep features that exist in X
        available_features = [f for f in self.selected_features if f in X.columns]
        return X[available_features]

    def _check_distribution_stability(self, X_train: pd.DataFrame, X_test: pd.DataFrame,
                                     mean_threshold: float = 0.5, std_ratio_threshold: float = 2.0) -> List[str]:
        """
        Check if features have similar distributions in train and test sets.

        Returns features that are stable (not too different between train/test).
        """
        stable_features = []

        for col in X_train.columns:
            if col not in X_test.columns:
                continue

            train_values = X_train[col].dropna()
            test_values = X_test[col].dropna()

            if len(train_values) == 0 or len(test_values) == 0:
                continue

            # Calculate distribution statistics
            train_mean = train_values.mean()
            test_mean = test_values.mean()
            train_std = train_values.std()
            test_std = test_values.std()

            # Avoid division by zero
            if train_std == 0 or test_std == 0:
                if train_mean == test_mean:
                    stable_features.append(col)
                continue

            # Calculate differences
            mean_diff = abs(train_mean - test_mean)
            std_ratio = max(train_std, test_std) / min(train_std, test_std)

            # Normalize mean difference by standard deviation
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
                self.removed_features.append(col)
                print(f"Removing {col}: mean_diff={normalized_mean_diff:.2f}, std_ratio={std_ratio:.2f}")

        return stable_features

    def _remove_low_variance(self, X: pd.DataFrame, threshold: float = 0.01) -> List[str]:
        """Remove features with very low variance."""
        high_variance_features = []

        for col in X.columns:
            variance = X[col].var()
            if pd.isna(variance) or variance < threshold:
                self.removed_features.append(col)
                print(f"Removing {col}: low variance ({variance:.4f})")
            else:
                high_variance_features.append(col)

        return high_variance_features

    def _select_kbest(self, X: pd.DataFrame, y: pd.Series, k: int) -> List[str]:
        """Select k best features using univariate statistical tests."""
        if self.problem_type == "classification":
            selector = SelectKBest(score_func=f_classif, k=min(k, len(X.columns)))
        else:
            selector = SelectKBest(score_func=f_regression, k=min(k, len(X.columns)))

        selector.fit(X, y)

        # Get feature scores
        scores = selector.scores_
        for i, col in enumerate(X.columns):
            self.feature_scores[col] = scores[i] if not np.isnan(scores[i]) else 0.0

        # Get selected features
        selected_mask = selector.get_support()
        selected_features = X.columns[selected_mask].tolist()

        return selected_features

    def _select_mutual_info(self, X: pd.DataFrame, y: pd.Series, k: int) -> List[str]:
        """Select k best features using mutual information."""
        if self.problem_type == "classification":
            mi_scores = mutual_info_classif(X, y, random_state=42)
        else:
            mi_scores = mutual_info_regression(X, y, random_state=42)

        # Store scores
        for i, col in enumerate(X.columns):
            self.feature_scores[col] = mi_scores[i]

        # Select top k features
        top_k_indices = np.argsort(mi_scores)[-k:]
        selected_features = X.columns[top_k_indices].tolist()

        return selected_features

    def get_feature_importance_report(self) -> Dict[str, Any]:
        """Get a report of feature selection results."""
        return {
            'selected_features': self.selected_features,
            'removed_features': self.removed_features,
            'n_selected': len(self.selected_features),
            'n_removed': len(self.removed_features),
            'feature_scores': self.feature_scores,
            'distribution_stats': self.distribution_stats
        }
