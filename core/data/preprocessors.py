"""
Data preprocessing utilities for cleaning and preparing datasets.
"""

from typing import Dict, List, Optional, Tuple, Any
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
import warnings

warnings.filterwarnings('ignore')


class DataPreprocessor:
    """Handles data cleaning and preprocessing operations."""

    def __init__(self, missing_threshold: float = 0.8, outlier_threshold: float = 3.0):
        self.missing_threshold = missing_threshold
        self.outlier_threshold = outlier_threshold
        self.label_encoders = {}
        self.scalers = {}

    def handle_missing_values(self, df: pd.DataFrame, method: str = "auto") -> pd.DataFrame:
        """Handle missing values in the dataset."""
        df_cleaned = df.copy()

        # Drop columns with too many missing values
        missing_percentages = df_cleaned.isnull().sum() / len(df_cleaned)
        cols_to_drop = missing_percentages[missing_percentages > self.missing_threshold].index

        if len(cols_to_drop) > 0:
            print(f"Dropping columns with >{self.missing_threshold*100}% missing: {list(cols_to_drop)}")
            df_cleaned = df_cleaned.drop(columns=cols_to_drop)

        # Fill remaining missing values
        for col in df_cleaned.columns:
            if df_cleaned[col].isnull().sum() > 0:
                if df_cleaned[col].dtype in ['int64', 'float64']:
                    # Numerical: fill with median
                    df_cleaned[col] = df_cleaned[col].fillna(df_cleaned[col].median())
                else:
                    # Categorical: fill with mode or 'Unknown'
                    mode_val = df_cleaned[col].mode()
                    fill_val = mode_val[0] if len(mode_val) > 0 else 'Unknown'
                    df_cleaned[col] = df_cleaned[col].fillna(fill_val)

        return df_cleaned

    def remove_duplicates(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, int]:
        """Remove duplicate rows from the dataset."""
        initial_count = len(df)
        df_deduplicated = df.drop_duplicates()
        duplicates_removed = initial_count - len(df_deduplicated)

        if duplicates_removed > 0:
            print(f"Removed {duplicates_removed} duplicate rows")

        return df_deduplicated, duplicates_removed

    def detect_outliers(self, df: pd.DataFrame, columns: Optional[List[str]] = None) -> Dict[str, List[int]]:
        """Detect outliers using Z-score method."""
        if columns is None:
            columns = df.select_dtypes(include=[np.number]).columns

        outliers = {}

        for col in columns:
            if col in df.columns:
                z_scores = np.abs((df[col] - df[col].mean()) / df[col].std())
                outlier_indices = df[z_scores > self.outlier_threshold].index.tolist()
                if len(outlier_indices) > 0:
                    outliers[col] = outlier_indices

        return outliers

    def encode_categorical_features(self, df: pd.DataFrame, fit: bool = True) -> pd.DataFrame:
        """Encode categorical features using label encoding."""
        df_encoded = df.copy()

        categorical_cols = df_encoded.select_dtypes(include=['object']).columns

        for col in categorical_cols:
            if fit:
                encoder = LabelEncoder()
                df_encoded[col] = encoder.fit_transform(df_encoded[col].astype(str))
                self.label_encoders[col] = encoder
            else:
                if col in self.label_encoders:
                    # Handle unseen categories
                    encoder = self.label_encoders[col]
                    known_classes = set(encoder.classes_)
                    df_col = df_encoded[col].astype(str)

                    # Replace unseen categories with the most frequent class
                    most_frequent = encoder.classes_[0]  # Assuming first class is most frequent
                    df_col = df_col.apply(lambda x: x if x in known_classes else most_frequent)
                    df_encoded[col] = encoder.transform(df_col)

        return df_encoded

    def scale_numerical_features(self, df: pd.DataFrame, fit: bool = True,
                               exclude_columns: Optional[List[str]] = None) -> pd.DataFrame:
        """Scale numerical features using StandardScaler."""
        df_scaled = df.copy()

        if exclude_columns is None:
            exclude_columns = []

        numerical_cols = df_scaled.select_dtypes(include=[np.number]).columns
        numerical_cols = [col for col in numerical_cols if col not in exclude_columns]

        if fit:
            scaler = StandardScaler()
            df_scaled[numerical_cols] = scaler.fit_transform(df_scaled[numerical_cols])
            self.scalers['numerical'] = scaler
        else:
            if 'numerical' in self.scalers:
                scaler = self.scalers['numerical']
                df_scaled[numerical_cols] = scaler.transform(df_scaled[numerical_cols])

        return df_scaled

    def get_basic_statistics(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Get basic statistics about the dataset."""
        return {
            'shape': df.shape,
            'memory_usage_mb': df.memory_usage(deep=True).sum() / 1024 / 1024,
            'missing_values': df.isnull().sum().to_dict(),
            'missing_percentages': (df.isnull().sum() / len(df)).to_dict(),
            'duplicates_count': df.duplicated().sum(),
            'numerical_columns': df.select_dtypes(include=[np.number]).columns.tolist(),
            'categorical_columns': df.select_dtypes(include=['object']).columns.tolist()
        }