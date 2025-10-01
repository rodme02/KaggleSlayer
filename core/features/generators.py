"""
Feature generation utilities for creating new features from existing data.
"""

from typing import List, Dict, Any, Optional, Tuple
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder, PolynomialFeatures
from sklearn.feature_selection import SelectKBest, f_classif, f_regression
import warnings

warnings.filterwarnings('ignore')


class FeatureGenerator:
    """Generates new features from existing data."""

    def __init__(self, max_features: int = 25, polynomial_degree: int = 2):
        self.max_features = max_features
        self.polynomial_degree = polynomial_degree
        self.created_features = []
        self.creation_methods = {}
        self.frequency_encodings = {}  # Store frequency mappings from training data

    def generate_numerical_features(self, df: pd.DataFrame, target_col: Optional[str] = None) -> pd.DataFrame:
        """Generate numerical features from existing numerical columns."""
        df_engineered = df.copy()
        numerical_cols = df.select_dtypes(include=[np.number]).columns.tolist()

        # Exclude target column and ID columns from feature generation
        if target_col and target_col in numerical_cols:
            numerical_cols.remove(target_col)

        # Remove ID-like columns that have no predictive value
        id_columns = ['PassengerId', 'id', 'Id', 'ID', 'index']
        for id_col in id_columns:
            if id_col in numerical_cols:
                numerical_cols.remove(id_col)

        if len(numerical_cols) < 2:
            return df_engineered

        print(f"Generating numerical features from {len(numerical_cols)} columns...")

        # Create more meaningful features based on data types and relationships
        feature_count = 0

        # First, create features that commonly make sense
        for i, col1 in enumerate(numerical_cols):
            if feature_count >= self.max_features:
                break

            for j, col2 in enumerate(numerical_cols[i+1:], i+1):
                if feature_count >= self.max_features:
                    break

                # Only create ratio if denominator is never zero and has meaningful variance
                col2_values = df[col2].replace(0, np.nan).dropna()
                if len(col2_values) > 0 and col2_values.std() > 0.01:
                    new_col = f"{col1}_div_{col2}"
                    df_engineered[new_col] = df[col1] / df[col2].replace(0, np.nan)
                    df_engineered[new_col] = df_engineered[new_col].fillna(df_engineered[new_col].median())
                    self._record_feature(new_col, f"Ratio of {col1} and {col2}")
                    feature_count += 1

                    if feature_count >= self.max_features:
                        break

                # Create product only if both columns have reasonable scale
                col1_scale = abs(df[col1].mean())
                col2_scale = abs(df[col2].mean())
                if col1_scale > 0 and col2_scale > 0 and col1_scale * col2_scale < 1e6:
                    new_col = f"{col1}_times_{col2}"
                    df_engineered[new_col] = df[col1] * df[col2]
                    self._record_feature(new_col, f"Product of {col1} and {col2}")
                    feature_count += 1

                    if feature_count >= self.max_features:
                        break

        # Add individual column transformations
        for col in numerical_cols[:5]:  # Limit to top 5 numerical columns
            if feature_count >= self.max_features:
                break

            # Log transformation for positive skewed data
            if (df[col] > 0).all() and df[col].skew() > 1:
                new_col = f"{col}_log"
                df_engineered[new_col] = np.log1p(df[col])
                self._record_feature(new_col, f"Log transformation of {col}")
                feature_count += 1

            # Square root for positive data with high variance
            if (df[col] >= 0).all() and df[col].std() > df[col].mean():
                new_col = f"{col}_sqrt"
                df_engineered[new_col] = np.sqrt(df[col])
                self._record_feature(new_col, f"Square root of {col}")
                feature_count += 1

            if feature_count >= self.max_features:
                break

        return df_engineered

    def generate_polynomial_features(self, df: pd.DataFrame, target_col: Optional[str] = None) -> pd.DataFrame:
        """Generate polynomial features for numerical columns."""
        df_engineered = df.copy()
        numerical_cols = df.select_dtypes(include=[np.number]).columns.tolist()

        # Exclude target column and ID columns from feature generation
        if target_col and target_col in numerical_cols:
            numerical_cols.remove(target_col)

        # Remove ID-like columns that have no predictive value
        id_columns = ['PassengerId', 'id', 'Id', 'ID', 'index']
        for id_col in id_columns:
            if id_col in numerical_cols:
                numerical_cols.remove(id_col)

        if len(numerical_cols) == 0:
            return df_engineered

        print(f"Generating polynomial features (degree {self.polynomial_degree})...")

        # Limit columns to prevent explosion
        if len(numerical_cols) > 5:
            # Use feature selection to pick most important columns
            if target_col and target_col in df.columns:
                selector = SelectKBest(score_func=f_regression, k=5)
                X_selected = selector.fit_transform(df[numerical_cols], df[target_col])
                selected_indices = selector.get_support(indices=True)
                numerical_cols = [numerical_cols[i] for i in selected_indices]
            else:
                numerical_cols = numerical_cols[:5]

        poly = PolynomialFeatures(degree=self.polynomial_degree, include_bias=False)
        poly_features = poly.fit_transform(df[numerical_cols])
        poly_feature_names = poly.get_feature_names_out(numerical_cols)

        # Add only new polynomial features (not the original ones)
        original_features = set(numerical_cols)
        for i, feature_name in enumerate(poly_feature_names):
            if feature_name not in original_features:
                # Sanitize feature name by replacing spaces with underscores
                sanitized_name = f"poly_{feature_name}".replace(" ", "_").replace("^", "_pow_")
                df_engineered[sanitized_name] = poly_features[:, i]
                self._record_feature(sanitized_name, f"Polynomial feature: {feature_name}")

        return df_engineered

    def generate_categorical_features(self, df: pd.DataFrame, target_col: Optional[str] = None, fit: bool = True) -> pd.DataFrame:
        """Generate features from categorical columns.

        Args:
            df: Input dataframe
            target_col: Target column to exclude
            fit: If True, compute frequency encodings from this data. If False, use stored encodings.
        """
        df_engineered = df.copy()
        categorical_cols = df.select_dtypes(include=['object']).columns.tolist()

        # Exclude target column from feature generation
        if target_col and target_col in categorical_cols:
            categorical_cols.remove(target_col)

        if len(categorical_cols) == 0:
            return df_engineered

        print(f"Generating categorical features from {len(categorical_cols)} columns...")

        for col in categorical_cols:
            # Only create frequency features for columns with reasonable cardinality
            unique_values = df[col].nunique()

            if unique_values > 1 and unique_values < len(df) * 0.5:  # Avoid high-cardinality columns
                # Frequency encoding (how common is this value)
                if fit:
                    # FIT: Compute frequency encoding from training data and store it
                    freq_encoding = df[col].value_counts(normalize=True)
                    self.frequency_encodings[col] = freq_encoding
                    df_engineered[f"{col}_freq"] = df[col].map(freq_encoding)
                    self._record_feature(f"{col}_freq", f"Frequency encoding of {col}")
                else:
                    # TRANSFORM: Use stored frequency encoding from training data
                    if col in self.frequency_encodings:
                        freq_encoding = self.frequency_encodings[col]
                        # For unseen categories, use a default small frequency
                        df_engineered[f"{col}_freq"] = df[col].map(freq_encoding).fillna(freq_encoding.min() / 2)
                    else:
                        # Fallback if not fitted (shouldn't happen)
                        freq_encoding = df[col].value_counts(normalize=True)
                        df_engineered[f"{col}_freq"] = df[col].map(freq_encoding)

                # Rarity encoding (inverse of frequency)
                df_engineered[f"{col}_rarity"] = 1 / df_engineered[f"{col}_freq"]
                if not fit:
                    self._record_feature(f"{col}_rarity", f"Rarity encoding of {col}")

            # For high-cardinality string columns, extract length and word count
            if unique_values > len(df) * 0.1:  # High cardinality
                # String length
                df_engineered[f"{col}_len"] = df[col].astype(str).str.len()
                self._record_feature(f"{col}_len", f"Length of {col} string")

                # Word count (for text-like columns)
                df_engineered[f"{col}_words"] = df[col].astype(str).str.split().str.len()
                self._record_feature(f"{col}_words", f"Word count in {col}")

                # Number of unique characters
                df_engineered[f"{col}_unique_chars"] = df[col].astype(str).apply(lambda x: len(set(x)))
                self._record_feature(f"{col}_unique_chars", f"Unique characters in {col}")

            # Missing value indicator
            if df[col].isna().any():
                df_engineered[f"{col}_is_missing"] = df[col].isna().astype(int)
                self._record_feature(f"{col}_is_missing", f"Missing value indicator for {col}")

        # DROP original categorical columns after extracting features
        # This prevents them from being label-encoded and scaled, which destroys information
        # and causes overfitting (especially for high-cardinality columns like Name, Ticket)
        print(f"Dropping {len(categorical_cols)} original categorical columns after feature extraction...")
        df_engineered = df_engineered.drop(columns=categorical_cols, errors='ignore')

        return df_engineered

    def generate_datetime_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Generate features from datetime columns."""
        df_engineered = df.copy()

        # Auto-detect datetime columns
        datetime_cols = []
        for col in df.columns:
            if df[col].dtype == 'datetime64[ns]':
                datetime_cols.append(col)
            elif df[col].dtype == 'object':
                # Try to convert object columns to datetime
                try:
                    pd.to_datetime(df[col].head(100), errors='raise')
                    datetime_cols.append(col)
                except:
                    continue

        if len(datetime_cols) == 0:
            return df_engineered

        print(f"Generating datetime features from {len(datetime_cols)} columns...")

        for col in datetime_cols:
            # Convert to datetime if not already
            if df[col].dtype != 'datetime64[ns]':
                df_engineered[col] = pd.to_datetime(df[col])

            # Extract components
            df_engineered[f"{col}_year"] = df_engineered[col].dt.year
            df_engineered[f"{col}_month"] = df_engineered[col].dt.month
            df_engineered[f"{col}_day"] = df_engineered[col].dt.day
            df_engineered[f"{col}_dayofweek"] = df_engineered[col].dt.dayofweek
            df_engineered[f"{col}_quarter"] = df_engineered[col].dt.quarter
            df_engineered[f"{col}_is_weekend"] = (df_engineered[col].dt.dayofweek >= 5).astype(int)

            # Record features
            for suffix in ['year', 'month', 'day', 'dayofweek', 'quarter', 'is_weekend']:
                self._record_feature(f"{col}_{suffix}", f"Extracted {suffix} from {col}")

        return df_engineered

    def generate_statistical_features(self, df: pd.DataFrame, target_col: Optional[str] = None) -> pd.DataFrame:
        """Generate statistical features for numerical columns."""
        df_engineered = df.copy()
        numerical_cols = df.select_dtypes(include=[np.number]).columns.tolist()

        # Exclude target column and ID columns from feature generation
        if target_col and target_col in numerical_cols:
            numerical_cols.remove(target_col)

        # Remove ID-like columns that have no predictive value
        id_columns = ['PassengerId', 'id', 'Id', 'ID', 'index']
        for id_col in id_columns:
            if id_col in numerical_cols:
                numerical_cols.remove(id_col)

        if len(numerical_cols) == 0:
            return df_engineered

        print("Generating statistical features...")

        # Row-wise statistics
        df_engineered['row_sum'] = df[numerical_cols].sum(axis=1)
        df_engineered['row_mean'] = df[numerical_cols].mean(axis=1)
        df_engineered['row_std'] = df[numerical_cols].std(axis=1)
        df_engineered['row_min'] = df[numerical_cols].min(axis=1)
        df_engineered['row_max'] = df[numerical_cols].max(axis=1)
        df_engineered['row_range'] = df_engineered['row_max'] - df_engineered['row_min']

        # Record features
        for stat in ['sum', 'mean', 'std', 'min', 'max', 'range']:
            self._record_feature(f"row_{stat}", f"Row-wise {stat} of numerical features")

        return df_engineered

    def _record_feature(self, feature_name: str, description: str):
        """Record a created feature and its description."""
        self.created_features.append(feature_name)
        self.creation_methods[feature_name] = description

    def get_feature_report(self) -> Dict[str, Any]:
        """Get a report of all created features."""
        return {
            'total_features_created': len(self.created_features),
            'features': self.created_features,
            'creation_methods': self.creation_methods
        }