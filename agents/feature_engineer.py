"""
Streamlined Feature Engineer Agent using modular components.
"""

from typing import Dict, Any, Tuple, Optional
import pandas as pd
from pathlib import Path

from .base_agent import BaseAgent
from core.features import FeatureGenerator, FeatureSelector, FeatureTransformer
from utils.config import ConfigManager


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
        """Run the complete feature engineering process."""
        self.log_info("Starting feature engineering process...")

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
            train_engineered = self.generator.generate_numerical_features(train_df, target_column)
            train_engineered = self.generator.generate_categorical_features(train_engineered, target_column, fit=True)
            train_engineered = self.generator.generate_statistical_features(train_engineered, target_column)
            train_engineered = self.generator.generate_polynomial_features(train_engineered, target_column)

            if test_df is not None:
                test_engineered = self.generator.generate_numerical_features(test_df, target_column)
                test_engineered = self.generator.generate_categorical_features(test_engineered, target_column, fit=False)
                test_engineered = self.generator.generate_statistical_features(test_engineered, target_column)
                test_engineered = self.generator.generate_polynomial_features(test_engineered, target_column)
            else:
                test_engineered = None

            # Feature selection
            self.log_info("Performing feature selection...")

            # Step 1: Check distribution stability (if test data available)
            if test_engineered is not None:
                train_selected = self.selector.check_distribution_stability(
                    train_engineered, test_engineered, target_column,
                    mean_threshold=0.5, std_ratio_threshold=2.0
                )
            else:
                train_selected = train_engineered

            # Step 2: Remove low variance and high correlation
            train_selected = self.selector.remove_low_variance_features(train_selected, target_column, fit=True)
            train_selected = self.selector.remove_highly_correlated_features(train_selected, target_column)

            # Step 3: Select best features
            train_selected = self.selector.select_univariate_features(train_selected, target_column, k=50)

            if test_engineered is not None:
                # Apply same selection to test data (transform only)
                test_selected = self.selector.remove_low_variance_features(test_engineered, target_column, fit=False)

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
            else:
                test_selected = None

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
                self.file_manager.save_dataframe(train_final, "train_engineered.csv")
                if test_final is not None:
                    self.file_manager.save_dataframe(test_final, "test_engineered.csv")
                self.log_info("Saved engineered datasets")

            # Save results
            self.save_results(results, "feature_engineer_results.json")

            self.log_info("Feature engineering completed successfully")
            return results

        except Exception as e:
            self.log_error(f"Feature engineering failed: {e}")
            raise

    def get_engineered_data(self) -> Tuple[pd.DataFrame, Optional[pd.DataFrame]]:
        """Load the engineered datasets."""
        try:
            train_df = self.file_manager.load_dataframe("train_engineered.csv")
            test_df = None

            if self.file_manager.file_exists("test_engineered.csv"):
                test_df = self.file_manager.load_dataframe("test_engineered.csv")

            return train_df, test_df

        except FileNotFoundError:
            self.log_warning("Engineered data files not found, running feature engineering first...")
            self.run()
            return self.get_engineered_data()

    def _load_cleaned_data(self) -> Tuple[pd.DataFrame, Optional[pd.DataFrame]]:
        """Load cleaned data from data scout."""
        try:
            train_df = self.file_manager.load_dataframe("train_cleaned.csv")
            test_df = None

            if self.file_manager.file_exists("test_cleaned.csv"):
                test_df = self.file_manager.load_dataframe("test_cleaned.csv")

            return train_df, test_df

        except FileNotFoundError:
            self.log_error("Cleaned data not found. Run data scout first.")
            raise