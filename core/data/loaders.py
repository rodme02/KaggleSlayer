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
        """Load train and test datasets from competition directory."""
        print(f"Loading data for competition: {self.competition_name}")

        # Load training data
        train_df = self.load_csv("train.csv")
        print(f"Loaded training data: {train_df.shape}")

        # Load test data if available
        test_path = self.data_path / "test.csv"
        test_df = None
        if test_path.exists():
            test_df = self.load_csv("test.csv")
            print(f"Loaded test data: {test_df.shape}")
        else:
            print("! No test data found")

        return train_df, test_df

    def detect_target_column(self, train_df: pd.DataFrame, test_df: Optional[pd.DataFrame] = None) -> Optional[str]:
        """Attempt to detect the target column."""
        common_targets = ['target', 'label', 'y', 'class', 'outcome', 'prediction']

        # Check for common target column names
        for col in train_df.columns:
            if col.lower() in common_targets:
                return col

        # Check if target column is not in test set
        if test_df is not None:
            train_cols = set(train_df.columns)
            test_cols = set(test_df.columns)
            diff_cols = train_cols - test_cols

            if len(diff_cols) == 1:
                return list(diff_cols)[0]

        # Check for columns with limited unique values that might be targets
        for col in train_df.columns:
            unique_ratio = train_df[col].nunique() / len(train_df)
            if 0.001 < unique_ratio < 0.1 and train_df[col].dtype in ['int64', 'float64']:
                return col

        return None

    def analyze_feature_types(self, df: pd.DataFrame) -> Dict[str, str]:
        """Analyze and categorize feature types."""
        feature_types = {}

        for col in df.columns:
            dtype = str(df[col].dtype)

            if dtype in ['int64', 'int32', 'float64', 'float32']:
                unique_count = df[col].nunique()
                total_count = len(df)

                if unique_count == 2:
                    feature_types[col] = "binary"
                elif unique_count < 10 and unique_count < total_count * 0.05:
                    feature_types[col] = "categorical_numeric"
                else:
                    feature_types[col] = "numerical"
            elif dtype == 'object':
                unique_count = df[col].nunique()
                total_count = len(df)

                if unique_count < total_count * 0.5:
                    feature_types[col] = "categorical"
                else:
                    feature_types[col] = "text"
            elif dtype.startswith('datetime'):
                feature_types[col] = "datetime"
            else:
                feature_types[col] = "other"

        return feature_types