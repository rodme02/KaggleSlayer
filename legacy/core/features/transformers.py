"""
Feature transformation utilities for scaling, encoding, and preprocessing features.

This module provides the FeatureTransformer class which handles:
- Scaling numerical features (standard, minmax, robust)
- Encoding categorical features (label, one-hot)
- Missing value imputation (simple, KNN)
- Power transformations for normality
- Interaction feature creation

All transformers support fit/transform pattern for proper train/test separation.
"""

from typing import Dict, List, Any, Optional, Tuple
import pandas as pd
import numpy as np
from sklearn.preprocessing import (
    StandardScaler, MinMaxScaler, RobustScaler, LabelEncoder,
    OneHotEncoder, OrdinalEncoder, PowerTransformer
)
from sklearn.impute import SimpleImputer, KNNImputer
from utils.logging import verbose_print
import warnings

warnings.filterwarnings('ignore')


class FeatureTransformer:
    """Transforms features using various scaling and encoding methods.

    Provides stateful transformation operations that can be fit on training data
    and applied to test data. All fitted transformers are stored for reuse.

    Attributes:
        scalers: Dictionary of fitted scaling transformers
        encoders: Dictionary of fitted encoding transformers
        imputers: Dictionary of fitted imputation transformers
        transformations_applied: Dictionary tracking which transformations were applied
    """

    def __init__(self):
        """Initialize the feature transformer."""
        self.scalers = {}
        self.encoders = {}
        self.imputers = {}
        self.transformations_applied = {}

    def scale_numerical_features(self, df: pd.DataFrame, method: str = "standard",
                                target_col: Optional[str] = None, fit: bool = True) -> pd.DataFrame:
        """Scale numerical features using specified method.

        Memory optimization: Uses float32 instead of float64 to reduce memory usage by 50%.
        """
        # Memory optimization: Use copy=False when safe, and convert to float32
        df_transformed = df.copy()
        feature_cols = [col for col in df.columns if col != target_col]
        numerical_cols = df[feature_cols].select_dtypes(include=[np.number]).columns

        if len(numerical_cols) == 0:
            return df_transformed

        # Convert to float32 to save memory (50% reduction vs float64)
        for col in numerical_cols:
            if df_transformed[col].dtype == np.float64:
                df_transformed[col] = df_transformed[col].astype(np.float32)

        verbose_print(f"Scaling numerical features using {method} scaling...")

        # Choose scaler
        if method == "standard":
            scaler = StandardScaler()
        elif method == "minmax":
            scaler = MinMaxScaler()
        elif method == "robust":
            scaler = RobustScaler()
        else:
            raise ValueError(f"Unknown scaling method: {method}")

        if fit:
            df_transformed[numerical_cols] = scaler.fit_transform(df[numerical_cols])
            self.scalers[f'numerical_{method}'] = scaler
            self.scalers[f'numerical_{method}_columns'] = list(numerical_cols)
        else:
            if f'numerical_{method}' in self.scalers:
                scaler = self.scalers[f'numerical_{method}']
                fitted_columns = self.scalers[f'numerical_{method}_columns']
                # Only transform columns that exist in both fitted and current data
                columns_to_scale = [col for col in fitted_columns if col in numerical_cols]
                if columns_to_scale:
                    df_transformed[columns_to_scale] = scaler.transform(df[columns_to_scale])

        self.transformations_applied[f'scaling_{method}'] = list(numerical_cols)
        return df_transformed

    def encode_target(self, y: pd.Series, fit: bool = True) -> pd.Series:
        """Encode target column if it's categorical (string labels).

        Args:
            y: Target series
            fit: If True, fit the encoder. If False, use stored encoder.

        Returns:
            Encoded target series (numeric)
        """
        # Check if target is string/object type
        if y.dtype == 'object' or y.dtype.name == 'category':
            if fit:
                encoder = LabelEncoder()
                y_encoded = pd.Series(encoder.fit_transform(y), index=y.index)
                self.encoders['target_label'] = encoder
                # Store mapping for later reference
                self.target_classes_ = encoder.classes_
                verbose_print(f"Encoded target labels: {list(encoder.classes_)} -> {list(range(len(encoder.classes_)))}")
            else:
                if 'target_label' in self.encoders:
                    encoder = self.encoders['target_label']
                    # Handle unseen categories
                    known_classes = set(encoder.classes_)
                    most_frequent = encoder.classes_[0]
                    y_cleaned = y.apply(lambda x: x if x in known_classes else most_frequent)
                    y_encoded = pd.Series(encoder.transform(y_cleaned), index=y.index)
                else:
                    # Fallback if not fitted
                    y_encoded = y
            return y_encoded
        else:
            # Already numeric
            return y

    def decode_target(self, y_encoded) -> pd.Series:
        """Decode target predictions back to original labels if they were encoded."""
        if 'target_label' in self.encoders:
            encoder = self.encoders['target_label']
            return encoder.inverse_transform(y_encoded)
        else:
            return y_encoded

    def encode_categorical_features(self, df: pd.DataFrame, method: str = "label",
                                  target_col: Optional[str] = None, fit: bool = True) -> pd.DataFrame:
        """Encode categorical features using specified method."""
        df_transformed = df.copy()
        feature_cols = [col for col in df.columns if col != target_col]
        categorical_cols = df[feature_cols].select_dtypes(include=['object']).columns

        if len(categorical_cols) == 0:
            return df_transformed

        verbose_print(f"Encoding categorical features using {method} encoding...")

        if method == "label":
            if fit:
                # Store which columns were fitted for later reference
                self.encoders['categorical_columns_fitted'] = list(categorical_cols)

            for col in categorical_cols:
                if fit:
                    encoder = LabelEncoder()
                    df_transformed[col] = encoder.fit_transform(df[col].astype(str))
                    self.encoders[f'{col}_label'] = encoder
                else:
                    if f'{col}_label' in self.encoders:
                        encoder = self.encoders[f'{col}_label']
                        # Handle unseen categories
                        known_classes = set(encoder.classes_)
                        series = df[col].astype(str)
                        most_frequent = encoder.classes_[0]
                        series = series.apply(lambda x: x if x in known_classes else most_frequent)
                        df_transformed[col] = encoder.transform(series)

        elif method == "onehot":
            if fit:
                encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
                encoded_features = encoder.fit_transform(df[categorical_cols])
                feature_names = encoder.get_feature_names_out(categorical_cols)

                # Create new dataframe with encoded features
                encoded_df = pd.DataFrame(encoded_features, columns=feature_names, index=df.index)

                # Drop original categorical columns and add encoded ones
                df_transformed = df_transformed.drop(columns=categorical_cols)
                df_transformed = pd.concat([df_transformed, encoded_df], axis=1)

                self.encoders['categorical_onehot'] = encoder
            else:
                if 'categorical_onehot' in self.encoders:
                    encoder = self.encoders['categorical_onehot']
                    encoded_features = encoder.transform(df[categorical_cols])
                    feature_names = encoder.get_feature_names_out(categorical_cols)

                    # Create new dataframe with encoded features
                    encoded_df = pd.DataFrame(encoded_features, columns=feature_names, index=df.index)

                    # Drop original categorical columns and add encoded ones
                    df_transformed = df_transformed.drop(columns=categorical_cols)
                    df_transformed = pd.concat([df_transformed, encoded_df], axis=1)

        self.transformations_applied[f'encoding_{method}'] = list(categorical_cols)
        return df_transformed

    def impute_missing_values(self, df: pd.DataFrame, method: str = "simple",
                            target_col: Optional[str] = None, fit: bool = True) -> pd.DataFrame:
        """Impute missing values using specified method."""
        df_imputed = df.copy()
        feature_cols = [col for col in df.columns if col != target_col]

        # Check if there are any missing values
        missing_counts = df[feature_cols].isnull().sum()
        cols_with_missing = missing_counts[missing_counts > 0].index

        if len(cols_with_missing) == 0:
            return df_imputed

        verbose_print(f"Imputing missing values using {method} method...")

        numerical_cols = df[feature_cols].select_dtypes(include=[np.number]).columns
        categorical_cols = df[feature_cols].select_dtypes(include=['object']).columns

        # Impute numerical columns
        numerical_missing = [col for col in numerical_cols if col in cols_with_missing]
        if len(numerical_missing) > 0:
            if method == "simple":
                if fit:
                    imputer = SimpleImputer(strategy='median')
                    df_imputed[numerical_missing] = imputer.fit_transform(df[numerical_missing])
                    self.imputers['numerical_simple'] = imputer
                    self.imputers['numerical_simple_columns'] = list(numerical_missing)
                else:
                    if 'numerical_simple' in self.imputers:
                        imputer = self.imputers['numerical_simple']
                        fitted_columns = self.imputers['numerical_simple_columns']
                        # Only impute columns that exist in both fitted and current data
                        columns_to_impute = [col for col in fitted_columns if col in numerical_missing]
                        if columns_to_impute:
                            df_imputed[columns_to_impute] = imputer.transform(df[columns_to_impute])

            elif method == "knn":
                if fit:
                    imputer = KNNImputer(n_neighbors=5)
                    df_imputed[numerical_missing] = imputer.fit_transform(df[numerical_missing])
                    self.imputers['numerical_knn'] = imputer
                    self.imputers['numerical_knn_columns'] = list(numerical_missing)
                else:
                    if 'numerical_knn' in self.imputers:
                        imputer = self.imputers['numerical_knn']
                        fitted_columns = self.imputers['numerical_knn_columns']
                        columns_to_impute = [col for col in fitted_columns if col in numerical_missing]
                        if columns_to_impute:
                            df_imputed[columns_to_impute] = imputer.transform(df[columns_to_impute])

        # Impute categorical columns
        categorical_missing = [col for col in categorical_cols if col in cols_with_missing]
        if len(categorical_missing) > 0:
            if fit:
                imputer = SimpleImputer(strategy='most_frequent')
                df_imputed[categorical_missing] = imputer.fit_transform(df[categorical_missing])
                self.imputers['categorical_simple'] = imputer
                self.imputers['categorical_simple_columns'] = list(categorical_missing)
            else:
                if 'categorical_simple' in self.imputers:
                    imputer = self.imputers['categorical_simple']
                    fitted_columns = self.imputers['categorical_simple_columns']
                    columns_to_impute = [col for col in fitted_columns if col in categorical_missing]
                    if columns_to_impute:
                        df_imputed[columns_to_impute] = imputer.transform(df[columns_to_impute])

        self.transformations_applied['imputation'] = list(cols_with_missing)
        return df_imputed

    def apply_power_transform(self, df: pd.DataFrame, method: str = "yeo-johnson",
                            target_col: Optional[str] = None, fit: bool = True) -> pd.DataFrame:
        """Apply power transformation to make features more Gaussian."""
        df_transformed = df.copy()
        feature_cols = [col for col in df.columns if col != target_col]
        numerical_cols = df[feature_cols].select_dtypes(include=[np.number]).columns

        if len(numerical_cols) == 0:
            return df_transformed

        verbose_print(f"Applying {method} power transformation...")

        try:
            if fit:
                transformer = PowerTransformer(method=method)
                df_transformed[numerical_cols] = transformer.fit_transform(df[numerical_cols])
                self.scalers[f'power_{method}'] = transformer
            else:
                if f'power_{method}' in self.scalers:
                    transformer = self.scalers[f'power_{method}']
                    df_transformed[numerical_cols] = transformer.transform(df[numerical_cols])

            self.transformations_applied[f'power_{method}'] = list(numerical_cols)

        except Exception as e:
            print(f"Warning: Power transformation failed: {e}")

        return df_transformed

    def create_interaction_features(self, df: pd.DataFrame, target_col: Optional[str] = None,
                                  max_interactions: int = 10) -> pd.DataFrame:
        """Create interaction features between numerical columns."""
        df_interactions = df.copy()
        feature_cols = [col for col in df.columns if col != target_col]
        numerical_cols = df[feature_cols].select_dtypes(include=[np.number]).columns

        if len(numerical_cols) < 2:
            return df_interactions

        verbose_print(f"Creating up to {max_interactions} interaction features...")

        interaction_count = 0
        created_features = []

        for i, col1 in enumerate(numerical_cols):
            if interaction_count >= max_interactions:
                break

            for j, col2 in enumerate(numerical_cols[i+1:], i+1):
                if interaction_count >= max_interactions:
                    break

                # Create interaction feature
                interaction_name = f"{col1}_x_{col2}"
                df_interactions[interaction_name] = df[col1] * df[col2]
                created_features.append(interaction_name)
                interaction_count += 1

        self.transformations_applied['interactions'] = created_features
        verbose_print(f"Created {len(created_features)} interaction features")

        return df_interactions

    def get_transformation_report(self) -> Dict[str, Any]:
        """Get a report of all transformations applied."""
        return {
            'transformations_applied': self.transformations_applied,
            'scalers_fitted': list(self.scalers.keys()),
            'encoders_fitted': list(self.encoders.keys()),
            'imputers_fitted': list(self.imputers.keys())
        }