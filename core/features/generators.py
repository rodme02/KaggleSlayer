"""
Feature generation utilities for creating new features from existing data.
"""

from typing import List, Dict, Any, Optional, Tuple
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder, PolynomialFeatures, KBinsDiscretizer
from sklearn.feature_selection import SelectKBest, f_classif, f_regression
from sklearn.cluster import KMeans
from scipy.stats import skew
import warnings
from pathlib import Path
import hashlib
import pickle

from .utils import detect_id_columns, is_numeric_dtype
from utils.logging import verbose_print

warnings.filterwarnings('ignore')


class FeatureCache:
    """Simple file-based cache for expensive feature computations."""

    def __init__(self, cache_dir: Optional[Path] = None):
        """Initialize feature cache.

        Args:
            cache_dir: Directory for cache files (default: .cache/features)
        """
        if cache_dir is None:
            cache_dir = Path(".cache/features")
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def _get_cache_key(self, df: pd.DataFrame, operation: str, params: Dict[str, Any]) -> str:
        """Generate cache key from dataframe content and operation parameters."""
        # Hash the dataframe content + operation + params
        df_hash = hashlib.md5(pd.util.hash_pandas_object(df).values).hexdigest()[:8]
        params_str = str(sorted(params.items()))
        params_hash = hashlib.md5(params_str.encode()).hexdigest()[:8]
        return f"{operation}_{df_hash}_{params_hash}.pkl"

    def get(self, df: pd.DataFrame, operation: str, params: Dict[str, Any]) -> Optional[pd.DataFrame]:
        """Get cached result if available."""
        cache_key = self._get_cache_key(df, operation, params)
        cache_file = self.cache_dir / cache_key

        if cache_file.exists():
            try:
                with open(cache_file, 'rb') as f:
                    cached_df = pickle.load(f)
                verbose_print(f"[Cache] Loaded {operation} from cache")
                return cached_df
            except Exception as e:
                verbose_print(f"[Cache] Failed to load cache: {e}")
                return None
        return None

    def set(self, df: pd.DataFrame, operation: str, params: Dict[str, Any], result: pd.DataFrame):
        """Cache the result of an expensive operation."""
        cache_key = self._get_cache_key(df, operation, params)
        cache_file = self.cache_dir / cache_key

        try:
            with open(cache_file, 'wb') as f:
                pickle.dump(result, f)
            verbose_print(f"[Cache] Saved {operation} to cache")
        except Exception as e:
            verbose_print(f"[Cache] Failed to save cache: {e}")

    def clear(self):
        """Clear all cached results."""
        import shutil
        if self.cache_dir.exists():
            shutil.rmtree(self.cache_dir)
            self.cache_dir.mkdir(parents=True, exist_ok=True)
        verbose_print("[Cache] Cleared all cached features")


class FeatureGenerator:
    """Generates new features from existing data.

    Includes feature importance tracking to identify and prune low-value features.
    """

    def __init__(self, max_features: int = 25, polynomial_degree: int = 2, use_cache: bool = True):
        self.max_features = max_features
        self.polynomial_degree = polynomial_degree
        self.created_features = []
        self.creation_methods = {}
        self.frequency_encodings = {}  # Store frequency mappings from training data
        self.target_encodings = {}  # Store target encodings from training data
        self.high_cardinality_threshold = 0.1  # Columns with >10% unique values are high cardinality
        self.id_columns = []  # Will be auto-detected
        self.kmeans_models = {}  # Store KMeans models for clustering features
        self.binning_transformers = {}  # Store binning transformers
        self.feature_importances = {}  # Store feature importance scores
        self.cache = FeatureCache() if use_cache else None  # Feature computation cache
        self.numerical_feature_recipes = []  # Store recipes for numerical features (col1, col2, operation)

    def generate_numerical_features(self, df: pd.DataFrame, target_col: Optional[str] = None, fit: bool = True) -> pd.DataFrame:
        """Generate numerical features from existing numerical columns.

        Args:
            df: Input dataframe
            target_col: Target column to exclude
            fit: If True, determine which features to create. If False, use stored recipes.
        """
        df_engineered = df.copy()
        numerical_cols = df.select_dtypes(include=[np.number]).columns.tolist()

        # Exclude target column and ID columns from feature generation
        if target_col and target_col in numerical_cols:
            numerical_cols.remove(target_col)

        # Auto-detect and remove ID columns
        if not self.id_columns:  # Only detect once
            self.id_columns = detect_id_columns(df)

        for id_col in self.id_columns:
            if id_col in numerical_cols:
                numerical_cols.remove(id_col)

        if len(numerical_cols) < 2:
            return df_engineered

        if fit:
            # FIT mode: Determine which features to create and store recipes
            verbose_print(f"Generating numerical features from {len(numerical_cols)} columns...")
            self.numerical_feature_recipes = []  # Reset recipes
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
                        # Store recipe
                        self.numerical_feature_recipes.append((col1, col2, 'div'))

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
                        # Store recipe
                        self.numerical_feature_recipes.append((col1, col2, 'times'))

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
                    self.numerical_feature_recipes.append((col, None, 'log'))

                    new_col = f"{col}_log"
                    df_engineered[new_col] = np.log1p(df[col])
                    self._record_feature(new_col, f"Log transformation of {col}")
                    feature_count += 1

                # Square root for positive data with high variance
                if (df[col] >= 0).all() and df[col].std() > df[col].mean():
                    self.numerical_feature_recipes.append((col, None, 'sqrt'))

                    new_col = f"{col}_sqrt"
                    df_engineered[new_col] = np.sqrt(df[col])
                    self._record_feature(new_col, f"Square root of {col}")
                    feature_count += 1

                if feature_count >= self.max_features:
                    break

        else:
            # TRANSFORM mode: Apply stored recipes to create exact same features
            for col1, col2, operation in self.numerical_feature_recipes:
                # Skip if required columns don't exist
                if col1 not in df.columns:
                    continue
                if col2 is not None and col2 not in df.columns:
                    continue

                if operation == 'div':
                    new_col = f"{col1}_div_{col2}"
                    df_engineered[new_col] = df[col1] / df[col2].replace(0, np.nan)
                    df_engineered[new_col] = df_engineered[new_col].fillna(df_engineered[new_col].median())

                elif operation == 'times':
                    new_col = f"{col1}_times_{col2}"
                    df_engineered[new_col] = df[col1] * df[col2]

                elif operation == 'log':
                    new_col = f"{col1}_log"
                    df_engineered[new_col] = np.log1p(df[col1].clip(lower=0))  # Clip to avoid negatives

                elif operation == 'sqrt':
                    new_col = f"{col1}_sqrt"
                    df_engineered[new_col] = np.sqrt(df[col1].clip(lower=0))  # Clip to avoid negatives

        return df_engineered

    def generate_polynomial_features(self, df: pd.DataFrame, target_col: Optional[str] = None) -> pd.DataFrame:
        """Generate polynomial features for numerical columns."""
        df_engineered = df.copy()
        numerical_cols = df.select_dtypes(include=[np.number]).columns.tolist()

        # Exclude target column and ID columns from feature generation
        if target_col and target_col in numerical_cols:
            numerical_cols.remove(target_col)

        # Use auto-detected ID columns
        for id_col in self.id_columns:
            if id_col in numerical_cols:
                numerical_cols.remove(id_col)

        if len(numerical_cols) == 0:
            return df_engineered

        verbose_print(f"Generating polynomial features (degree {self.polynomial_degree})...")

        # Adaptive limit based on dataset size and degree
        # Polynomial features grow as C(n+d, d) - limit to prevent explosion
        max_poly_cols = 10 if self.polynomial_degree == 2 else 5

        if len(numerical_cols) > max_poly_cols:
            # Use feature selection to pick most important columns
            if target_col and target_col in df.columns:
                selector = SelectKBest(score_func=f_regression, k=max_poly_cols)
                X_selected = selector.fit_transform(df[numerical_cols], df[target_col])
                selected_indices = selector.get_support(indices=True)
                numerical_cols = [numerical_cols[i] for i in selected_indices]
            else:
                numerical_cols = numerical_cols[:max_poly_cols]
            verbose_print(f"Limited to {len(numerical_cols)} columns for polynomial features")

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

        if not hasattr(self, '_logged_categorical'):
            verbose_print(f"Generating categorical features from {len(categorical_cols)} columns...")
            self._logged_categorical = True

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
                if fit:
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

        # NOTE: Keep categorical columns for now - they will be encoded in the transformer
        # Only drop high-cardinality text columns that we've already extracted features from
        high_cardinality_cols = [col for col in categorical_cols
                                 if df[col].nunique() > len(df) * 0.1]
        if high_cardinality_cols:
            verbose_print(f"Dropping {len(high_cardinality_cols)} high-cardinality text columns after feature extraction...")
            df_engineered = df_engineered.drop(columns=high_cardinality_cols, errors='ignore')

        return df_engineered

    def generate_target_encoding(self, df: pd.DataFrame, target_col: str,
                                 categorical_cols: Optional[List[str]] = None,
                                 fit: bool = True, smoothing: float = 1.0) -> pd.DataFrame:
        """
        Generate target encoding for categorical features.
        Uses smoothing to prevent overfitting.

        Args:
            df: Input dataframe
            target_col: Target column name
            categorical_cols: List of categorical columns to encode (auto-detect if None)
            fit: If True, compute encodings from training data
            smoothing: Smoothing parameter (higher = more regularization)
        """
        if target_col not in df.columns:
            return df

        df_encoded = df.copy()

        if categorical_cols is None:
            categorical_cols = df.select_dtypes(include=['object']).columns.tolist()
            if target_col in categorical_cols:
                categorical_cols.remove(target_col)

        if len(categorical_cols) == 0:
            return df_encoded

        # Global mean for smoothing
        global_mean = df[target_col].mean() if fit else None

        for col in categorical_cols:
            if col not in df.columns:
                continue

            if fit:
                # Calculate mean target per category with smoothing
                category_stats = df.groupby(col)[target_col].agg(['mean', 'count'])

                # Smoothed mean = (count * category_mean + smoothing * global_mean) / (count + smoothing)
                category_stats['smoothed_mean'] = (
                    (category_stats['count'] * category_stats['mean'] + smoothing * global_mean) /
                    (category_stats['count'] + smoothing)
                )

                # Store encoding
                self.target_encodings[col] = {
                    'encoding': category_stats['smoothed_mean'].to_dict(),
                    'global_mean': global_mean
                }

                # Apply encoding
                df_encoded[f"{col}_target_enc"] = df[col].map(self.target_encodings[col]['encoding'])
                self._record_feature(f"{col}_target_enc", f"Target encoding of {col}")
            else:
                # Use stored encoding
                if col in self.target_encodings:
                    encoding = self.target_encodings[col]['encoding']
                    global_mean_stored = self.target_encodings[col]['global_mean']

                    # For unseen categories, use global mean
                    df_encoded[f"{col}_target_enc"] = df[col].map(encoding).fillna(global_mean_stored)

        return df_encoded

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

        verbose_print(f"Generating datetime features from {len(datetime_cols)} columns...")

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

        # Use auto-detected ID columns
        for id_col in self.id_columns:
            if id_col in numerical_cols:
                numerical_cols.remove(id_col)

        # Only create statistical features if we have enough columns
        if len(numerical_cols) < 3:
            verbose_print(f"Skipping statistical features (need 3+ numerical columns, have {len(numerical_cols)})")
            return df_engineered

        verbose_print(f"Generating statistical features from {len(numerical_cols)} columns...")

        # Row-wise statistics - only meaningful with multiple features
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

    def generate_clustering_features(self, df: pd.DataFrame, target_col: Optional[str] = None,
                                    n_clusters: int = 5, fit: bool = True) -> pd.DataFrame:
        """
        Generate clustering-based features using KMeans.

        Args:
            df: Input dataframe
            target_col: Target column to exclude
            n_clusters: Number of clusters
            fit: If True, fit new KMeans models
        """
        df_clustered = df.copy()
        numerical_cols = df.select_dtypes(include=[np.number]).columns.tolist()

        # Exclude target and ID columns
        if target_col and target_col in numerical_cols:
            numerical_cols.remove(target_col)

        for id_col in self.id_columns:
            if id_col in numerical_cols:
                numerical_cols.remove(id_col)

        if len(numerical_cols) < 2:
            return df_clustered

        verbose_print(f"Generating clustering features with {n_clusters} clusters...")

        # Use a subset of features to prevent overfitting (max 10 features)
        if len(numerical_cols) > 10:
            # Use features with highest variance
            variances = df[numerical_cols].var().sort_values(ascending=False)
            selected_cols = variances.head(10).index.tolist()
        else:
            selected_cols = numerical_cols

        try:
            if fit:
                # Fit KMeans
                kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
                df_clustered['cluster_label'] = kmeans.fit_predict(df[selected_cols].fillna(0))

                # Store model
                self.kmeans_models['default'] = {
                    'model': kmeans,
                    'features': selected_cols
                }

                # Distance to each cluster center
                distances = kmeans.transform(df[selected_cols].fillna(0))
                for i in range(n_clusters):
                    df_clustered[f'cluster_dist_{i}'] = distances[:, i]
                    self._record_feature(f'cluster_dist_{i}', f"Distance to cluster {i}")

                # Distance to nearest cluster
                df_clustered['cluster_min_dist'] = distances.min(axis=1)
                self._record_feature('cluster_min_dist', "Distance to nearest cluster")
                self._record_feature('cluster_label', "Cluster assignment")

            else:
                # Use stored model
                if 'default' in self.kmeans_models:
                    kmeans = self.kmeans_models['default']['model']
                    selected_cols = self.kmeans_models['default']['features']

                    # Ensure all features exist
                    available_cols = [col for col in selected_cols if col in df.columns]
                    if len(available_cols) > 0:
                        df_clustered['cluster_label'] = kmeans.predict(df[available_cols].fillna(0))

                        distances = kmeans.transform(df[available_cols].fillna(0))
                        for i in range(n_clusters):
                            df_clustered[f'cluster_dist_{i}'] = distances[:, i]

                        df_clustered['cluster_min_dist'] = distances.min(axis=1)

        except Exception as e:
            verbose_print(f"Warning: Clustering feature generation failed: {e}")

        return df_clustered

    def generate_binning_features(self, df: pd.DataFrame, target_col: Optional[str] = None,
                                 n_bins: int = 5, fit: bool = True) -> pd.DataFrame:
        """
        Generate binning/discretization features for numerical columns.

        Args:
            df: Input dataframe
            target_col: Target column to exclude
            n_bins: Number of bins
            fit: If True, fit new binning transformers
        """
        df_binned = df.copy()
        numerical_cols = df.select_dtypes(include=[np.number]).columns.tolist()

        # Exclude target and ID columns
        if target_col and target_col in numerical_cols:
            numerical_cols.remove(target_col)

        for id_col in self.id_columns:
            if id_col in numerical_cols:
                numerical_cols.remove(id_col)

        if len(numerical_cols) == 0:
            return df_binned

        verbose_print(f"Generating binning features ({n_bins} bins) for numerical columns...")

        # Only bin columns with high variance or skewness (more informative)
        cols_to_bin = []
        for col in numerical_cols:
            if df[col].nunique() > n_bins:  # Only bin if we have enough unique values
                col_skew = abs(df[col].skew())
                col_std = df[col].std()
                if col_skew > 0.5 or col_std > df[col].mean():  # High skew or variance
                    cols_to_bin.append(col)

        # Limit to top 10 columns to prevent feature explosion
        cols_to_bin = cols_to_bin[:10]

        if len(cols_to_bin) == 0:
            return df_binned

        for col in cols_to_bin:
            try:
                if fit:
                    # Fit binning transformer
                    binner = KBinsDiscretizer(n_bins=n_bins, encode='ordinal', strategy='quantile')
                    df_binned[f'{col}_binned'] = binner.fit_transform(df[[col]].fillna(df[col].median()))

                    self.binning_transformers[col] = binner
                    self._record_feature(f'{col}_binned', f"Binned version of {col} ({n_bins} quantile bins)")
                else:
                    # Use stored transformer
                    if col in self.binning_transformers:
                        binner = self.binning_transformers[col]
                        df_binned[f'{col}_binned'] = binner.transform(df[[col]].fillna(df[col].median()))

            except Exception as e:
                verbose_print(f"Warning: Binning failed for {col}: {e}")
                continue

        return df_binned

    def generate_timeseries_features(self, df: pd.DataFrame, datetime_col: str,
                                    value_cols: Optional[List[str]] = None) -> pd.DataFrame:
        """
        Generate time-series features like lags, rolling statistics, and differences.

        Args:
            df: Input dataframe (must be sorted by datetime_col)
            datetime_col: Name of the datetime column
            value_cols: Columns to generate time-series features for (auto-detect if None)
        """
        if datetime_col not in df.columns:
            return df

        df_ts = df.copy()

        # Auto-detect numerical columns if not specified
        if value_cols is None:
            value_cols = df.select_dtypes(include=[np.number]).columns.tolist()
            # Remove ID columns
            value_cols = [col for col in value_cols if col not in self.id_columns]

        if len(value_cols) == 0:
            return df_ts

        verbose_print(f"Generating time-series features for {len(value_cols)} columns...")

        # Limit to top 5 columns to prevent explosion
        value_cols = value_cols[:5]

        for col in value_cols:
            if col not in df.columns:
                continue

            try:
                # Lag features (previous values)
                for lag in [1, 2, 3, 7]:
                    df_ts[f'{col}_lag_{lag}'] = df[col].shift(lag)
                    self._record_feature(f'{col}_lag_{lag}', f"Lag {lag} of {col}")

                # Rolling statistics
                for window in [3, 7, 14]:
                    if len(df) >= window:
                        df_ts[f'{col}_rolling_mean_{window}'] = df[col].rolling(window=window, min_periods=1).mean()
                        df_ts[f'{col}_rolling_std_{window}'] = df[col].rolling(window=window, min_periods=1).std()
                        self._record_feature(f'{col}_rolling_mean_{window}', f"Rolling mean (window={window}) of {col}")
                        self._record_feature(f'{col}_rolling_std_{window}', f"Rolling std (window={window}) of {col}")

                # Difference features (change from previous)
                df_ts[f'{col}_diff_1'] = df[col].diff(1)
                df_ts[f'{col}_pct_change'] = df[col].pct_change(1)
                self._record_feature(f'{col}_diff_1', f"First difference of {col}")
                self._record_feature(f'{col}_pct_change', f"Percent change of {col}")

            except Exception as e:
                verbose_print(f"Warning: Time-series feature generation failed for {col}: {e}")
                continue

        return df_ts

    def _record_feature(self, feature_name: str, description: str):
        """Record a created feature and its description."""
        self.created_features.append(feature_name)
        self.creation_methods[feature_name] = description

    def calculate_feature_importance(self, df: pd.DataFrame, target_col: str,
                                    problem_type: str = "auto") -> Dict[str, float]:
        """Calculate importance scores for all generated features.

        Uses Random Forest feature importance to score features.
        Scores below 0.01 indicate features that could be pruned.

        Args:
            df: DataFrame with features and target
            target_col: Name of target column
            problem_type: 'classification', 'regression', or 'auto'

        Returns:
            Dictionary mapping feature names to importance scores
        """
        if target_col not in df.columns:
            return {}

        # Select only created features
        feature_cols = [col for col in self.created_features if col in df.columns]
        if len(feature_cols) == 0:
            return {}

        # Determine problem type
        if problem_type == "auto":
            problem_type = "classification" if df[target_col].nunique() < 20 else "regression"

        try:
            # Use Random Forest for importance
            if problem_type == "classification":
                from sklearn.ensemble import RandomForestClassifier
                model = RandomForestClassifier(n_estimators=50, max_depth=10, random_state=42)
            else:
                from sklearn.ensemble import RandomForestRegressor
                model = RandomForestRegressor(n_estimators=50, max_depth=10, random_state=42)

            # Prepare data
            X = df[feature_cols].fillna(0)
            y = df[target_col]

            # Handle categorical features (encode them)
            from sklearn.preprocessing import LabelEncoder
            for col in feature_cols:
                if X[col].dtype == 'object':
                    le = LabelEncoder()
                    X[col] = le.fit_transform(X[col].astype(str))

            # Fit model
            model.fit(X, y)

            # Get importances
            importances = model.feature_importances_
            self.feature_importances = dict(zip(feature_cols, importances))

            verbose_print(f"Feature importance calculated for {len(feature_cols)} features")

            # Identify low-value features
            low_value_features = [f for f, score in self.feature_importances.items() if score < 0.01]
            if low_value_features:
                verbose_print(f"Found {len(low_value_features)} low-importance features (score < 0.01)")

            return self.feature_importances

        except Exception as e:
            verbose_print(f"Warning: Feature importance calculation failed: {e}")
            return {}

    def prune_low_importance_features(self, df: pd.DataFrame, importance_threshold: float = 0.01) -> pd.DataFrame:
        """Remove features with importance below threshold.

        Args:
            df: DataFrame to prune
            importance_threshold: Minimum importance score to keep

        Returns:
            DataFrame with low-importance features removed
        """
        if not self.feature_importances:
            verbose_print("No feature importances available, skipping pruning")
            return df

        # Find features to drop
        features_to_drop = [
            feature for feature, score in self.feature_importances.items()
            if score < importance_threshold and feature in df.columns
        ]

        if len(features_to_drop) > 0:
            verbose_print(f"Pruning {len(features_to_drop)} low-importance features")
            df_pruned = df.drop(columns=features_to_drop)
            # Remove from created features list
            self.created_features = [f for f in self.created_features if f not in features_to_drop]
            return df_pruned

        return df

    def get_feature_importance(self) -> Dict[str, float]:
        """Get feature importance scores.

        Returns:
            Dictionary mapping feature names to importance scores
        """
        return self.feature_importances.copy()

    def get_feature_report(self) -> Dict[str, Any]:
        """Get a report of all created features."""
        report = {
            'total_features_created': len(self.created_features),
            'features': self.created_features,
            'creation_methods': self.creation_methods
        }

        if self.feature_importances:
            report['feature_importances'] = self.feature_importances
            report['low_importance_features'] = [
                f for f, score in self.feature_importances.items() if score < 0.01
            ]

        return report