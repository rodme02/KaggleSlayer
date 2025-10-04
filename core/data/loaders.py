"""
Data loading utilities for Kaggle competition datasets.
"""

from pathlib import Path
from typing import Optional, Tuple, Dict, Any
import pandas as pd
import numpy as np
from dataclasses import dataclass


@dataclass
class DatasetInfo:
    """Information about a loaded dataset."""
    competition_name: str
    total_rows: int
    total_columns: int
    train_rows: int
    test_rows: int
    feature_types: Dict[str, str]
    missing_values: Dict[str, int]
    missing_percentages: Dict[str, float]
    target_column: Optional[str]
    target_type: Optional[str]
    duplicates_count: int
    memory_usage_mb: float
    analysis_timestamp: str


class DataLoader:
    """Base class for loading datasets."""

    def __init__(self, data_path: Path):
        self.data_path = Path(data_path)

    def load_csv(self, filename: str, **kwargs) -> pd.DataFrame:
        """Load a CSV file with error handling."""
        file_path = self.data_path / filename
        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")

        try:
            return pd.read_csv(file_path, **kwargs)
        except Exception as e:
            raise ValueError(f"Error loading {filename}: {e}")


class CompetitionDataLoader(DataLoader):
    """Specialized loader for Kaggle competition data."""

    def __init__(self, competition_path: Path):
        super().__init__(competition_path)
        self.competition_name = competition_path.name

    def load_competition_data(self) -> Tuple[pd.DataFrame, Optional[pd.DataFrame]]:
        """Load train and test datasets from competition directory.

        Looks for data in both root directory and raw/ subdirectory.
        """
        print(f"Loading data for competition: {self.competition_name}")

        # Check for raw/ subdirectory first, then root
        possible_paths = [
            self.data_path / "raw",
            self.data_path
        ]

        train_df = None
        test_df = None

        # Find and load training data
        for base_path in possible_paths:
            train_path = base_path / "train.csv"
            if train_path.exists():
                train_df = pd.read_csv(train_path)
                print(f"Loaded training data from {base_path.name or 'root'}: {train_df.shape}")
                break

        if train_df is None:
            raise FileNotFoundError("Could not find train.csv")

        # Find and load test data
        for base_path in possible_paths:
            test_path = base_path / "test.csv"
            if test_path.exists():
                test_df = pd.read_csv(test_path)
                print(f"Loaded test data from {base_path.name or 'root'}: {test_df.shape}")
                break

        if test_df is None:
            print("! No test data found")

        return train_df, test_df

    def detect_target_column(self, train_df: pd.DataFrame, test_df: Optional[pd.DataFrame] = None) -> Optional[str]:
        """Attempt to detect the target column."""
        # Check if target column is not in test set
        if test_df is not None:
            train_cols = set(train_df.columns)
            test_cols = set(test_df.columns)
            diff_cols = train_cols - test_cols

            if len(diff_cols) == 1:
                return list(diff_cols)[0]

        return None

    def analyze_feature_types(self, df: pd.DataFrame) -> Dict[str, str]:
        """Analyze and categorize feature types with improved detection."""
        feature_types = {}

        for col in df.columns:
            # Check if it's an identifier column first
            if self._is_identifier_column(col, df[col]):
                feature_types[col] = "identifier"
                continue

            dtype = str(df[col].dtype)

            if dtype in ['int64', 'int32', 'float64', 'float32']:
                unique_count = df[col].nunique()
                total_count = len(df)
                unique_ratio = unique_count / total_count
                uniques = set(df[col].dropna().unique())

                KNOWN_BOOL_SETS = [{0, 1}, {True, False}, {"Yes", "No"}, {-1, 1}, {0.0, 1.0}]

                # Very high uniqueness suggests identifier
                if unique_ratio > 0.95:
                    feature_types[col] = "identifier"
                # Binary detection
                elif (unique_count == 2 and df[col].value_counts().min() > 1) or uniques in KNOWN_BOOL_SETS:
                    feature_types[col] = "binary"
                # Ordinal: sequential integers with low cardinality
                elif unique_count < 10 and self._is_sequential(df[col]):
                    feature_types[col] = "ordinal"
                # Categorical numeric
                elif unique_count < 20 and unique_count < total_count * 0.05:
                    feature_types[col] = "categorical_numeric"
                else:
                    feature_types[col] = "numerical"

            elif dtype == 'object':
                unique_count = df[col].nunique()
                total_count = len(df)
                unique_ratio = unique_count / total_count

                # Check if it's a date string
                if self._is_date_string(df[col]):
                    feature_types[col] = "date_string"
                # High cardinality text (likely IDs or unique descriptions)
                elif unique_ratio > 0.9:
                    feature_types[col] = "high_cardinality_text"
                # Regular text
                elif unique_ratio > 0.5:
                    feature_types[col] = "text"
                # Categorical
                else:
                    feature_types[col] = "categorical"

            elif dtype.startswith('datetime'):
                feature_types[col] = "datetime"
            else:
                feature_types[col] = "other"

        return feature_types

    def _is_identifier_column(self, col_name: str, series: pd.Series) -> bool:
        """Detect if a column is an identifier (ID, index, etc.)."""
        # Name-based detection
        id_keywords = ['id', 'ID', 'Id', 'index', 'key', 'passenger', 'customer', 'user', 'order']
        if any(keyword.lower() in col_name.lower() for keyword in id_keywords):
            # Also check uniqueness to confirm
            if series.nunique() / len(series) > 0.9:
                return True
        return False

    def _is_sequential(self, series: pd.Series) -> bool:
        """Check if numeric values are sequential (1,2,3,4,5...)."""
        unique_vals = sorted(series.dropna().unique())
        if len(unique_vals) < 2:
            return False
        # Check if all differences are 1
        diffs = np.diff(unique_vals)
        return np.all(diffs == 1) or np.all(diffs == -1)

    def _is_date_string(self, series: pd.Series) -> bool:
        """Detect if object column contains date strings."""
        if len(series) == 0:
            return False

        # Sample first non-null values
        sample = series.dropna().head(100)
        if len(sample) == 0:
            return False

        try:
            # Try to parse as datetime
            parsed = pd.to_datetime(sample, errors='coerce')
            valid_ratio = parsed.notna().sum() / len(sample)
            return valid_ratio > 0.5  # >50% parseable as dates
        except:
            return False