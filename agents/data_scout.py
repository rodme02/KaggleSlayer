"""
Streamlined Data Scout Agent using modular components.
"""

from typing import Dict, Any, Tuple, Optional
import pandas as pd
from pathlib import Path

from .base_agent import BaseAgent
from core.data import CompetitionDataLoader, DataPreprocessor, DataValidator
from utils.config import ConfigManager


class DataScoutAgent(BaseAgent):
    """Streamlined agent for data loading, cleaning, and initial analysis."""

    def __init__(self, competition_name: str, competition_path: Path,
                 config: Optional[ConfigManager] = None):
        super().__init__(competition_name, competition_path, config)

        # Initialize components
        self.data_loader = CompetitionDataLoader(competition_path)
        self.preprocessor = DataPreprocessor(
            missing_threshold=self.get_config("data.drop_missing_threshold", 0.8),
            outlier_threshold=self.get_config("data.outlier_threshold", 3.0)
        )
        self.validator = DataValidator()

    def run(self, save_cleaned_data: bool = True) -> Dict[str, Any]:
        """Run the complete data scouting process."""
        self.log_info("Starting data scouting process...")

        try:
            # Load data
            train_df, test_df = self.data_loader.load_competition_data()

            # Validate data
            validation_result = self.validator.validate_dataset(train_df, "training")
            if test_df is not None:
                test_validation = self.validator.validate_train_test_consistency(train_df, test_df)

            # Detect target column
            target_column = self.data_loader.detect_target_column(train_df, test_df)
            self.log_info(f"Detected target column: {target_column}")

            # Analyze feature types
            feature_types = self.data_loader.analyze_feature_types(train_df)
            self.log_info(f"Feature types detected: {len([k for k,v in feature_types.items() if v == 'identifier'])} identifiers, "
                         f"{len([k for k,v in feature_types.items() if v == 'categorical'])} categorical, "
                         f"{len([k for k,v in feature_types.items() if v == 'numerical'])} numerical")

            # Clean data
            if not validation_result.is_valid:
                self.log_warning(f"Data validation issues found: {validation_result.issues}")

            # Parse datetime features first
            train_cleaned = self.preprocessor.parse_and_extract_datetime(train_df)
            if test_df is not None:
                test_cleaned = self.preprocessor.parse_and_extract_datetime(test_df)
            else:
                test_cleaned = None

            # Handle missing values with fit/transform pattern
            train_cleaned = self.preprocessor.handle_missing_values(
                train_cleaned, fit=True, target_col=target_column
            )
            if test_cleaned is not None:
                test_cleaned = self.preprocessor.handle_missing_values(
                    test_cleaned, fit=False, target_col=target_column
                )

            # Remove duplicates
            train_cleaned, duplicates_removed = self.preprocessor.remove_duplicates(train_cleaned)
            if test_cleaned is not None:
                test_cleaned, _ = self.preprocessor.remove_duplicates(test_cleaned)

            # Handle outliers (winsorize to 1-99 percentile)
            exclude_cols = [target_column] if target_column else []
            train_cleaned = self.preprocessor.handle_outliers(
                train_cleaned, method="winsorize", exclude_columns=exclude_cols
            )

            # Get statistics
            stats = self.preprocessor.get_basic_statistics(train_cleaned)

            # Generate insights
            dataset_info = {
                'total_rows': len(train_cleaned),
                'total_columns': len(train_cleaned.columns),
                'target_column': target_column,
                'feature_types': feature_types,
                'missing_values': stats['missing_values'],
                'duplicates_removed': duplicates_removed,
                'memory_usage_mb': stats['memory_usage_mb']
            }

            # Create results
            results = {
                'competition_name': self.competition_name,
                'dataset_info': dataset_info,
                'validation_results': {
                    'train_valid': validation_result.is_valid,
                    'train_issues': validation_result.issues,
                    'train_warnings': validation_result.warnings,
                    'statistics': stats
                },
                'preprocessing_applied': {
                    'missing_values_handled': True,
                    'duplicates_removed': duplicates_removed,
                    'outliers_detected': len(self.preprocessor.detect_outliers(train_cleaned))
                },
                'recommendations': {
                    'target_column': target_column,
                    'problem_type': 'classification' if target_column and train_cleaned[target_column].nunique() < 20 else 'regression',
                    'feature_count': len(train_cleaned.columns),
                    'sample_count': len(train_cleaned)
                }
            }

            # Save cleaned data if requested
            if save_cleaned_data:
                self.file_manager.save_processed_data(train_cleaned, "train_cleaned.csv")
                if test_cleaned is not None:
                    self.file_manager.save_processed_data(test_cleaned, "test_cleaned.csv")
                self.log_info("Saved cleaned datasets")

            # Save results
            self.file_manager.save_results(results, "data_scout_results.json")

            self.log_info("Data scouting completed successfully")
            return results

        except Exception as e:
            self.log_error(f"Data scouting failed: {e}")
            raise

    def get_cleaned_data(self) -> Tuple[pd.DataFrame, Optional[pd.DataFrame]]:
        """Load the cleaned datasets."""
        try:
            train_df = self.file_manager.load_processed_data("train_cleaned.csv")
            test_df = None

            if self.file_manager.file_exists(f"{self.file_manager.processed_dir}/test_cleaned.csv"):
                test_df = self.file_manager.load_processed_data("test_cleaned.csv")

            return train_df, test_df

        except FileNotFoundError:
            self.log_warning("Cleaned data files not found, running data scouting first...")
            self.run()
            return self.get_cleaned_data()