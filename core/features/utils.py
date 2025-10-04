"""
Utility functions for feature engineering.
"""

from typing import List, Optional
import pandas as pd
import numpy as np
import time
import psutil
import warnings

warnings.filterwarnings('ignore')


class FeatureEngineeringMonitor:
    """Monitor memory and performance during feature engineering."""

    def __init__(self):
        self.process = psutil.Process()
        self.checkpoints = {}
        self.start_time = None
        self.start_memory = None

    def start(self):
        """Start monitoring."""
        self.start_time = time.time()
        self.start_memory = self.process.memory_info().rss / 1024 / 1024  # MB

    def checkpoint(self, name: str):
        """Record a checkpoint."""
        current_time = time.time()
        current_memory = self.process.memory_info().rss / 1024 / 1024  # MB

        self.checkpoints[name] = {
            'elapsed_time': current_time - self.start_time if self.start_time else 0,
            'memory_mb': current_memory,
            'memory_delta': current_memory - self.start_memory if self.start_memory else 0
        }

    def get_report(self):
        """Get monitoring report."""
        return self.checkpoints


def detect_id_columns(df: pd.DataFrame, threshold: float = 0.95) -> List[str]:
    """
    Auto-detect ID columns based on heuristics.

    PERFORMANCE FIX: Now detects both numeric and string IDs (e.g., "CUST_00123", UUIDs)

    Args:
        df: Input dataframe
        threshold: Uniqueness threshold for ID detection (default 95%)

    Returns:
        List of detected ID column names
    """
    id_columns = []

    for col in df.columns:
        uniqueness_ratio = df[col].nunique() / len(df)

        # ID column criteria:
        # 1. High uniqueness (>95% unique values)
        # 2. Name contains 'id', 'index', or 'key' (case-insensitive)

        col_lower = col.lower()
        has_id_name = any(pattern in col_lower for pattern in ['id', 'index', 'key'])

        # For ALL columns (numeric and non-numeric):
        if uniqueness_ratio > threshold:
            if has_id_name:
                id_columns.append(col)
                continue

            # Additional check for numeric columns: high correlation with row index
            if df[col].dtype in ['int64', 'float64', 'int32', 'float32']:
                try:
                    correlation_with_index = df[col].corr(pd.Series(range(len(df))))
                    if abs(correlation_with_index) > threshold:
                        id_columns.append(col)
                except:
                    pass

    return id_columns


def is_numeric_dtype(dtype) -> bool:
    """Check if a dtype is numeric."""
    return np.issubdtype(dtype, np.number)


def auto_detect_problem_type(y: pd.Series) -> str:
    """
    Automatically detect problem type (classification or regression).

    Unified logic used across all components for consistency.

    Args:
        y: Target series

    Returns:
        'classification' or 'regression'
    """
    # Check dtype first - non-numeric types are always classification
    if y.dtype in ['object', 'bool', 'category']:
        return 'classification'

    # For numeric types, check unique values
    unique_count = y.nunique()

    # Classification if:
    # 1. Few unique values (< 20), AND
    # 2. Integer-like dtype (int64, int32, etc.)
    if unique_count < 20 and y.dtype in ['int64', 'int32', 'int16', 'int8']:
        return 'classification'

    # Otherwise regression
    return 'regression'
