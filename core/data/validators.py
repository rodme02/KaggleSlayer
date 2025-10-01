"""
Data validation utilities for ensuring data quality and consistency.
"""

from typing import Dict, List, Optional, Any, Tuple
import pandas as pd
import numpy as np
from dataclasses import dataclass


@dataclass
class ValidationResult:
    """Result of data validation checks."""
    is_valid: bool
    issues: List[str]
    warnings: List[str]
    statistics: Dict[str, Any]


class DataValidator:
    """Validates data quality and consistency."""

    def __init__(self):
        self.issues = []
        self.warnings = []

    def validate_dataset(self, df: pd.DataFrame, name: str = "dataset") -> ValidationResult:
        """Perform comprehensive dataset validation."""
        self.issues = []
        self.warnings = []

        # Basic validation checks
        self._check_empty_dataset(df, name)
        self._check_duplicate_columns(df, name)
        self._check_data_types(df, name)
        self._check_missing_data(df, name)
        self._check_infinite_values(df, name)
        self._check_constant_features(df, name)

        # Compute statistics
        statistics = self._compute_statistics(df)

        is_valid = len(self.issues) == 0

        return ValidationResult(
            is_valid=is_valid,
            issues=self.issues.copy(),
            warnings=self.warnings.copy(),
            statistics=statistics
        )

    def validate_train_test_consistency(self, train_df: pd.DataFrame,
                                      test_df: pd.DataFrame) -> ValidationResult:
        """Validate consistency between train and test datasets."""
        self.issues = []
        self.warnings = []

        # Check column consistency
        train_cols = set(train_df.columns)
        test_cols = set(test_df.columns)

        missing_in_test = train_cols - test_cols
        extra_in_test = test_cols - train_cols

        if missing_in_test:
            self.warnings.append(f"Columns in train but not in test: {missing_in_test}")

        if extra_in_test:
            self.warnings.append(f"Columns in test but not in train: {extra_in_test}")

        # Check data type consistency for common columns
        common_cols = train_cols.intersection(test_cols)
        for col in common_cols:
            if train_df[col].dtype != test_df[col].dtype:
                self.warnings.append(f"Data type mismatch for {col}: train={train_df[col].dtype}, test={test_df[col].dtype}")

        # Check value ranges for numerical columns
        for col in common_cols:
            if train_df[col].dtype in ['int64', 'float64']:
                train_min, train_max = train_df[col].min(), train_df[col].max()
                test_min, test_max = test_df[col].min(), test_df[col].max()

                if test_min < train_min or test_max > train_max:
                    self.warnings.append(f"Value range mismatch for {col}: train=[{train_min}, {train_max}], test=[{test_min}, {test_max}]")

        statistics = {
            'train_shape': train_df.shape,
            'test_shape': test_df.shape,
            'common_columns': len(common_cols),
            'train_only_columns': len(missing_in_test),
            'test_only_columns': len(extra_in_test)
        }

        is_valid = len(self.issues) == 0

        return ValidationResult(
            is_valid=is_valid,
            issues=self.issues.copy(),
            warnings=self.warnings.copy(),
            statistics=statistics
        )

    def _check_empty_dataset(self, df: pd.DataFrame, name: str):
        """Check if dataset is empty."""
        if df.empty:
            self.issues.append(f"{name} is empty")
        elif len(df) == 0:
            self.issues.append(f"{name} has no rows")
        elif len(df.columns) == 0:
            self.issues.append(f"{name} has no columns")

    def _check_duplicate_columns(self, df: pd.DataFrame, name: str):
        """Check for duplicate column names."""
        duplicate_cols = df.columns[df.columns.duplicated()].tolist()
        if duplicate_cols:
            self.issues.append(f"{name} has duplicate columns: {duplicate_cols}")

    def _check_data_types(self, df: pd.DataFrame, name: str):
        """Check for problematic data types."""
        for col in df.columns:
            if df[col].dtype == 'object':
                # Check if numeric data is stored as string
                try:
                    pd.to_numeric(df[col], errors='raise')
                    self.warnings.append(f"{name}.{col} appears to be numeric but stored as object")
                except (ValueError, TypeError):
                    pass

    def _check_missing_data(self, df: pd.DataFrame, name: str):
        """Check missing data patterns."""
        missing_counts = df.isnull().sum()
        total_missing = missing_counts.sum()

        if total_missing > 0:
            high_missing_cols = missing_counts[missing_counts > len(df) * 0.5].index.tolist()
            if high_missing_cols:
                self.warnings.append(f"{name} has columns with >50% missing data: {high_missing_cols}")

        # Check for completely missing columns
        completely_missing = missing_counts[missing_counts == len(df)].index.tolist()
        if completely_missing:
            self.issues.append(f"{name} has completely empty columns: {completely_missing}")

    def _check_infinite_values(self, df: pd.DataFrame, name: str):
        """Check for infinite values in numerical columns."""
        numerical_cols = df.select_dtypes(include=[np.number]).columns

        for col in numerical_cols:
            if np.isinf(df[col]).any():
                self.issues.append(f"{name}.{col} contains infinite values")

    def _check_constant_features(self, df: pd.DataFrame, name: str):
        """Check for constant (zero variance) features."""
        for col in df.columns:
            if df[col].nunique() <= 1:
                self.warnings.append(f"{name}.{col} is constant (no variance)")

    def _compute_statistics(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Compute dataset statistics."""
        return {
            'shape': df.shape,
            'memory_usage_mb': df.memory_usage(deep=True).sum() / 1024 / 1024,
            'total_missing': df.isnull().sum().sum(),
            'missing_percentage': (df.isnull().sum().sum() / df.size) * 100,
            'duplicated_rows': df.duplicated().sum(),
            'numerical_columns': len(df.select_dtypes(include=[np.number]).columns),
            'categorical_columns': len(df.select_dtypes(include=['object']).columns),
            'datetime_columns': len(df.select_dtypes(include=['datetime']).columns)
        }