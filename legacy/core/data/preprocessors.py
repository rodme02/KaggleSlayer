"""
Data preprocessing utilities for cleaning and preparing datasets.
"""

from typing import Dict, List, Optional, Tuple, Any
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.impute import SimpleImputer
import warnings

warnings.filterwarnings('ignore')


class DataPreprocessor:
    """Handles data cleaning and preprocessing operations."""

    def __init__(self, missing_threshold: float = 0.8, outlier_threshold: float = 3.0):
        self.missing_threshold = missing_threshold
        self.outlier_threshold = outlier_threshold
        self.label_encoders = {}
        self.scalers = {}
        self.numeric_imputer = None
        self.categorical_imputer = None
        self.missing_indicators = {}  # Store which columns had missing values

    def handle_missing_values(self, df: pd.DataFrame, method: str = "auto",
                             fit: bool = True, target_col: Optional[str] = None) -> pd.DataFrame:
        """Handle missing values with improved strategies and missingness indicators.

        Args:
            df: Input dataframe
            method: Imputation method ('auto', 'median', 'mean', 'mode')
            fit: If True, fit imputer on this data. If False, use stored imputer.
            target_col: Target column to exclude from analysis
        """
        df_cleaned = df.copy()

        # CRITICAL FIX: Identify columns excluding target FIRST to prevent data leakage
        feature_cols = [col for col in df_cleaned.columns if col != target_col]

        # Identify columns with missing values (excluding target)
        missing_percentages = df_cleaned[feature_cols].isnull().sum() / len(df_cleaned)
        cols_with_missing = missing_percentages[missing_percentages > 0].index.tolist()

        # Create missingness indicators BEFORE imputation
        for col in cols_with_missing:
            missing_pct = missing_percentages[col]
            # Only create indicator if missingness is between 5% and threshold
            if 0.05 < missing_pct <= self.missing_threshold:
                indicator_name = f'{col}_was_missing'
                df_cleaned[indicator_name] = df_cleaned[col].isnull().astype(int)
                if fit:
                    self.missing_indicators[col] = indicator_name
                print(f"Created missingness indicator for {col} ({missing_pct*100:.1f}% missing)")

        # Drop columns with excessive missing values (already excludes target)
        cols_to_drop = missing_percentages[missing_percentages > self.missing_threshold].index.tolist()

        if len(cols_to_drop) > 0:
            print(f"Dropping {len(cols_to_drop)} columns with >{self.missing_threshold*100}% missing: {cols_to_drop[:5]}...")
            df_cleaned = df_cleaned.drop(columns=cols_to_drop)

        # Fill remaining missing values (target already excluded from feature_cols)
        numeric_cols = df_cleaned.select_dtypes(include=['int64', 'float64']).columns.tolist()
        categorical_cols = df_cleaned.select_dtypes(include=['object']).columns.tolist()

        # Ensure target is not in these lists (safety check)
        if target_col:
            numeric_cols = [col for col in numeric_cols if col != target_col]
            categorical_cols = [col for col in categorical_cols if col != target_col]

        # Impute numerical columns
        if len(numeric_cols) > 0:
            if fit:
                # Fit imputer on training data
                self.numeric_imputer = SimpleImputer(strategy='median')
                df_cleaned[numeric_cols] = self.numeric_imputer.fit_transform(df_cleaned[numeric_cols])
            else:
                # Transform test data using fitted imputer
                if self.numeric_imputer is not None:
                    df_cleaned[numeric_cols] = self.numeric_imputer.transform(df_cleaned[numeric_cols])

        # Impute categorical columns - use explicit 'MISSING' category
        for col in categorical_cols:
            if df_cleaned[col].isnull().sum() > 0:
                # Fill with explicit MISSING category to preserve information
                df_cleaned[col] = df_cleaned[col].fillna('MISSING_VALUE')

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

    def handle_outliers(self, df: pd.DataFrame, method: str = "winsorize",
                       exclude_columns: Optional[List[str]] = None) -> pd.DataFrame:
        """Handle outliers without removing data points.

        Args:
            df: Input dataframe
            method: 'winsorize' (clip to percentiles), 'flag' (create indicators), or 'none'
            exclude_columns: Columns to exclude from outlier handling
        """
        df_handled = df.copy()
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()

        if exclude_columns:
            numeric_cols = [col for col in numeric_cols if col not in exclude_columns]

        for col in numeric_cols:
            if method == "winsorize":
                # Clip to 1st-99th percentile to handle extreme outliers
                lower = df[col].quantile(0.01)
                upper = df[col].quantile(0.99)
                df_handled[col] = df[col].clip(lower, upper)

            elif method == "flag":
                # Create binary outlier indicator
                if df[col].std() > 0:  # Avoid division by zero
                    z_scores = np.abs((df[col] - df[col].mean()) / df[col].std())
                    df_handled[f'{col}_is_outlier'] = (z_scores > self.outlier_threshold).astype(int)

        return df_handled

    def parse_and_extract_datetime(self, df: pd.DataFrame) -> pd.DataFrame:
        """Parse datetime columns and extract temporal features.

        Args:
            df: Input dataframe

        Returns:
            Dataframe with datetime features extracted
        """
        df_temporal = df.copy()
        datetime_cols = []

        # Find datetime columns
        datetime_cols.extend(df.select_dtypes(include=['datetime', 'datetime64']).columns.tolist())

        # Check object columns for date patterns
        for col in df.select_dtypes(include=['object']).columns:
            try:
                # Sample to check if parseable
                sample = df[col].dropna().head(100)
                if len(sample) > 0:
                    parsed = pd.to_datetime(sample, errors='coerce')
                    valid_ratio = parsed.notna().sum() / len(sample)
                    if valid_ratio > 0.5:  # >50% parseable as dates
                        df_temporal[col] = pd.to_datetime(df[col], errors='coerce')
                        datetime_cols.append(col)
                        print(f"Parsed {col} as datetime ({valid_ratio*100:.1f}% valid)")
            except:
                pass

        # Extract features from datetime columns
        for col in datetime_cols:
            if col in df_temporal.columns and df_temporal[col].dtype in ['datetime64[ns]', 'datetime64']:
                # Basic temporal features
                df_temporal[f'{col}_year'] = df_temporal[col].dt.year
                df_temporal[f'{col}_month'] = df_temporal[col].dt.month
                df_temporal[f'{col}_day'] = df_temporal[col].dt.day
                df_temporal[f'{col}_dayofweek'] = df_temporal[col].dt.dayofweek
                df_temporal[f'{col}_quarter'] = df_temporal[col].dt.quarter
                df_temporal[f'{col}_is_weekend'] = (df_temporal[col].dt.dayofweek >= 5).astype(int)

                # Cyclical encoding for month (captures seasonal patterns)
                df_temporal[f'{col}_month_sin'] = np.sin(2 * np.pi * df_temporal[col].dt.month / 12)
                df_temporal[f'{col}_month_cos'] = np.cos(2 * np.pi * df_temporal[col].dt.month / 12)

                # Cyclical encoding for day of week
                df_temporal[f'{col}_dayofweek_sin'] = np.sin(2 * np.pi * df_temporal[col].dt.dayofweek / 7)
                df_temporal[f'{col}_dayofweek_cos'] = np.cos(2 * np.pi * df_temporal[col].dt.dayofweek / 7)

                print(f"Extracted {11} temporal features from {col}")

                # Drop original datetime column
                df_temporal = df_temporal.drop(columns=[col])

        return df_temporal

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