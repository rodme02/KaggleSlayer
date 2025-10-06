"""
Streamlined Feature Engineer Agent using modular components.
"""

from typing import Dict, Any, Tuple, Optional, List
import pandas as pd
from pathlib import Path
from sklearn.base import BaseEstimator, TransformerMixin

from .base_agent import BaseAgent
from core.features import FeatureGenerator, FeatureSelector, FeatureTransformer
from utils.config import ConfigManager


class FeatureEngineeringPipeline(BaseEstimator, TransformerMixin):
    """Scikit-learn compatible pipeline for all feature engineering.

    This ensures feature engineering happens INSIDE CV folds to prevent data leakage.
    """

    # Class-level flag to suppress all print statements during CV
    _suppress_logs = False

    def __init__(self, generator: FeatureGenerator, selector: FeatureSelector,
                 transformer: FeatureTransformer, target_col: str, k_features: int = 20, verbose: bool = False):
        self.generator = generator
        self.selector = selector
        self.transformer = transformer
        self.target_col = target_col
        self.k_features = k_features
        self.verbose = verbose
        self.selected_features_ = None
        self._fit_count = 0

    def fit(self, X: pd.DataFrame, y=None):
        """Fit on training data only - NO information from validation folds."""
        self._fit_count += 1
        if self.verbose and self._fit_count == 1:
            print(f"[FeaturePipeline] Fitting on {len(X)} samples...")

        # Combine X and y for feature generation that needs target
        if y is not None and self.target_col:
            df = X.copy()
            df[self.target_col] = y
        else:
            df = X.copy()

        # Generate features
        df_generated = self.generator.generate_numerical_features(df, self.target_col, fit=True)
        df_generated = self.generator.generate_categorical_features(df_generated, self.target_col, fit=True)
        df_generated = self.generator.generate_statistical_features(df_generated, self.target_col)

        # Select features (fit only on training data)
        df_selected = self.selector.remove_low_variance_features(df_generated, self.target_col, fit=True)
        df_selected = self.selector.remove_highly_correlated_features(df_selected, self.target_col)
        df_selected = self.selector.select_univariate_features(df_selected, self.target_col, k=self.k_features)

        # Store which features were selected
        self.selected_features_ = [col for col in df_selected.columns if col != self.target_col]

        # Fit transformers
        df_transformed = self.transformer.impute_missing_values(df_selected, target_col=self.target_col, fit=True)
        df_transformed = self.transformer.encode_categorical_features(df_transformed, target_col=self.target_col, fit=True)
        df_transformed = self.transformer.scale_numerical_features(df_transformed, target_col=self.target_col, fit=True)

        if self.verbose and self._fit_count == 1:
            print(f"[FeaturePipeline] Selected {len(self.selected_features_)} features")

        return self

    def transform(self, X: pd.DataFrame):
        """Transform validation/test data using fitted parameters only."""
        if self.selected_features_ is None:
            raise RuntimeError("Pipeline must be fitted before transform")

        df = X.copy()

        # Generate features (same operations, but fit=False for categorical)
        df_generated = self.generator.generate_numerical_features(df, self.target_col, fit=False)
        df_generated = self.generator.generate_categorical_features(df_generated, self.target_col, fit=False)
        df_generated = self.generator.generate_statistical_features(df_generated, self.target_col)  # FIX: use df_generated not df

        # Apply same feature selection (only keep features that were selected during fit)
        available_features = [col for col in self.selected_features_ if col in df_generated.columns]
        df_selected = df_generated[available_features]

        # Apply fitted transformations
        df_transformed = self.transformer.impute_missing_values(df_selected, target_col=self.target_col, fit=False)
        df_transformed = self.transformer.encode_categorical_features(df_transformed, target_col=self.target_col, fit=False)
        df_transformed = self.transformer.scale_numerical_features(df_transformed, target_col=self.target_col, fit=False)

        return df_transformed


class FeatureEngineerAgent(BaseAgent):
    """Streamlined agent for feature engineering using modular components."""

    def __init__(self, competition_name: str, competition_path: Path,
                 config: Optional[ConfigManager] = None):
        super().__init__(competition_name, competition_path, config)

        # Initialize components
        self.generator = FeatureGenerator(
            max_features=self.get_config("pipeline.max_features_to_create", 25),
            polynomial_degree=self.get_config("pipeline.polynomial_degree", 2)
        )
        self.selector = FeatureSelector(
            correlation_threshold=self.get_config("data.correlation_threshold", 0.95),
            variance_threshold=self.get_config("data.variance_threshold", 0.01)
        )
        self.transformer = FeatureTransformer()

    def run(self, save_engineered_data: bool = True) -> Dict[str, Any]:
        """Run the complete feature engineering process with error handling and rollback.

        If feature engineering fails, attempts to load previous successful results.
        """
        self.log_info("Starting feature engineering process...")

        # Store backup of previous results for rollback
        previous_results = None
        try:
            previous_results = self.load_results("feature_engineer_results.json")
        except:
            pass

        try:
            # Load cleaned data
            train_df, test_df = self._load_cleaned_data()

            # Load data scout results for context
            data_results = self.load_results("data_scout_results.json")
            target_column = data_results.get('dataset_info', {}).get('target_column')

            if not target_column:
                self.log_warning("No target column identified, using last column")
                target_column = train_df.columns[-1]

            # Generate features
            self.log_info("Generating new features...")
            train_engineered = self.generator.generate_numerical_features(train_df, target_column, fit=True)
            train_engineered = self.generator.generate_categorical_features(train_engineered, target_column, fit=True)
            train_engineered = self.generator.generate_statistical_features(train_engineered, target_column)
            # Skip polynomial features to reduce overfitting
            # train_engineered = self.generator.generate_polynomial_features(train_engineered, target_column)

            if test_df is not None:
                test_engineered = self.generator.generate_numerical_features(test_df, target_column, fit=False)
                test_engineered = self.generator.generate_categorical_features(test_engineered, target_column, fit=False)
                test_engineered = self.generator.generate_statistical_features(test_engineered, target_column)
                # Skip polynomial features to reduce overfitting
                # test_engineered = self.generator.generate_polynomial_features(test_engineered, target_column)
            else:
                test_engineered = None

            # Feature selection
            self.log_info("Performing feature selection...")

            # Step 1: Check distribution stability (if test data available)
            if test_engineered is not None:
                train_selected, test_selected = self.selector.check_distribution_stability(
                    train_engineered, test_engineered, target_column,
                    mean_threshold=0.5, std_ratio_threshold=2.0
                )
            else:
                train_selected = train_engineered
                test_selected = None

            # Step 2: Remove low variance and high correlation from aligned data
            train_selected = self.selector.remove_low_variance_features(train_selected, target_column, fit=True)
            train_selected = self.selector.remove_highly_correlated_features(train_selected, target_column)

            # Step 3: Select best features (fewer features to reduce overfitting)
            train_selected = self.selector.select_univariate_features(train_selected, target_column, k=20)

            if test_selected is not None:
                # Apply same selection to test data (transform only) - already aligned from stability check
                test_selected = self.selector.remove_low_variance_features(test_selected, target_column, fit=False)

                # Get the exact features from training data (excluding target)
                train_features = [col for col in train_selected.columns if col != target_column]

                # Only keep features that exist in test data AND were selected in training
                common_features = [col for col in train_features if col in test_selected.columns]

                # For features in train but not in test, add them with zeros/NaN
                for col in train_features:
                    if col not in test_selected.columns:
                        test_selected[col] = 0.0  # Fill missing features with 0

                # Reorder test columns to match train order
                if target_column in test_selected.columns:
                    test_selected = test_selected[train_features + [target_column]]
                else:
                    test_selected = test_selected[train_features]

            # Transform features - ONLY on selected features
            self.log_info("Transforming features...")
            train_final = self.transformer.impute_missing_values(train_selected, target_col=target_column, fit=True)
            train_final = self.transformer.encode_categorical_features(train_final, target_col=target_column, fit=True)
            train_final = self.transformer.scale_numerical_features(train_final, target_col=target_column, fit=True)

            if test_selected is not None:
                test_final = self.transformer.impute_missing_values(test_selected, target_col=target_column, fit=False)
                test_final = self.transformer.encode_categorical_features(test_final, target_col=target_column, fit=False)
                test_final = self.transformer.scale_numerical_features(test_final, target_col=target_column, fit=False)
            else:
                test_final = None

            # Create results
            results = {
                'competition_name': self.competition_name,
                'original_features': len(train_df.columns),
                'generated_features': len(self.generator.created_features),
                'final_features': len(train_final.columns),
                'target_column': target_column,
                'feature_generation_report': self.generator.get_feature_report(),
                'feature_selection_report': self.selector.get_selection_report(),
                'transformation_report': self.transformer.get_transformation_report(),
                'data_shapes': {
                    'train_final': train_final.shape,
                    'test_final': test_final.shape if test_final is not None else None
                }
            }

            # Save engineered data
            if save_engineered_data:
                self.file_manager.save_processed_data(train_final, "train_engineered.csv")
                if test_final is not None:
                    self.file_manager.save_processed_data(test_final, "test_engineered.csv")
                self.log_info("Saved engineered datasets")

            # Save results
            self.save_results(results, "feature_engineer_results.json")

            self.log_info("Feature engineering completed successfully")
            return results

        except Exception as e:
            self.log_error(f"Feature engineering failed: {e}")

            # Rollback: Try to use previous successful results
            if previous_results is not None:
                self.log_warning("Rolling back to previous successful feature engineering results")
                return previous_results
            else:
                self.log_error("No previous results available for rollback")
                raise

    def create_feature_pipeline(self) -> FeatureEngineeringPipeline:
        """Create a scikit-learn compatible feature engineering pipeline.

        This pipeline prevents data leakage by fitting only on training folds during CV.

        Returns:
            FeatureEngineeringPipeline: A fitted transformer that can be used in scikit-learn pipelines
        """
        # Load data scout results to get target column
        data_results = self.load_results("data_scout_results.json")
        target_column = data_results.get('dataset_info', {}).get('target_column')

        if not target_column:
            self.log_warning("No target column identified, using last column")
            # Try to load cleaned data to infer target
            train_df, _ = self._load_cleaned_data()
            target_column = train_df.columns[-1]

        # Create fresh instances of feature engineering components
        # This ensures each pipeline has its own state
        generator = FeatureGenerator(
            max_features=self.get_config("pipeline.max_features_to_create", 25),
            polynomial_degree=self.get_config("pipeline.polynomial_degree", 2)
        )
        selector = FeatureSelector(
            correlation_threshold=self.get_config("data.correlation_threshold", 0.95),
            variance_threshold=self.get_config("data.variance_threshold", 0.01)
        )
        transformer = FeatureTransformer()

        return FeatureEngineeringPipeline(
            generator=generator,
            selector=selector,
            transformer=transformer,
            target_col=target_column,
            k_features=20,  # Reduced from 50 to prevent overfitting
            verbose=False  # Suppress logs during CV to reduce noise
        )

    def get_engineered_data(self) -> Tuple[pd.DataFrame, Optional[pd.DataFrame]]:
        """Load the engineered datasets."""
        try:
            train_df = self.file_manager.load_processed_data("train_engineered.csv")
            test_df = None

            if self.file_manager.file_exists(f"{self.file_manager.processed_dir}/test_engineered.csv"):
                test_df = self.file_manager.load_processed_data("test_engineered.csv")

            return train_df, test_df

        except FileNotFoundError:
            self.log_warning("Engineered data files not found, running feature engineering first...")
            self.run()
            return self.get_engineered_data()

    def _load_cleaned_data(self) -> Tuple[pd.DataFrame, Optional[pd.DataFrame]]:
        """Load cleaned data from data scout."""
        try:
            train_df = self.file_manager.load_processed_data("train_cleaned.csv")
            test_df = None

            if self.file_manager.file_exists(f"{self.file_manager.processed_dir}/test_cleaned.csv"):
                test_df = self.file_manager.load_processed_data("test_cleaned.csv")

            return train_df, test_df

        except FileNotFoundError:
            self.log_error("Cleaned data not found. Run data scout first.")
            raise