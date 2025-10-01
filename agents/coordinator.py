"""
Pipeline Coordinator - Orchestrates the entire KaggleSlayer pipeline.
"""

from typing import Dict, Any, Optional
from pathlib import Path
import pandas as pd

from .base_agent import BaseAgent
from .data_scout import DataScoutAgent
from .feature_engineer import FeatureEngineerAgent
from .model_selector import ModelSelectorAgent
from utils.config import ConfigManager
from utils.kaggle_api import KaggleAPIClient


class PipelineCoordinator(BaseAgent):
    """Coordinates the entire KaggleSlayer pipeline execution."""

    def __init__(self, competition_name: str, competition_path: Path,
                 config: Optional[ConfigManager] = None):
        super().__init__(competition_name, competition_path, config)

        # Initialize agents
        self.data_scout = DataScoutAgent(competition_name, competition_path, config)
        self.feature_engineer = FeatureEngineerAgent(competition_name, competition_path, config)
        self.model_selector = ModelSelectorAgent(competition_name, competition_path, config)

        # Initialize Kaggle API client
        self.kaggle_client = KaggleAPIClient()

    def run(self, skip_steps: Optional[list] = None, submit_to_kaggle: bool = False) -> Dict[str, Any]:
        """Run the complete pipeline."""
        if skip_steps is None:
            skip_steps = []

        pipeline_results = {
            'competition_name': self.competition_name,
            'pipeline_status': 'running',
            'steps_completed': [],
            'results': {}
        }

        # Print banner
        print("\n" + "="*70)
        print(f"  KAGGLESLAYER PIPELINE - {self.competition_name.upper()}")
        print("="*70)

        try:
            import time
            start_time = time.time()

            # Step 1: Data Scouting
            if 'data_scout' not in skip_steps:
                print(f"\n{'='*70}")
                print("  STEP 1/4: DATA SCOUT - Exploring and Cleaning Data")
                print("="*70)
                self.log_info("Analyzing dataset structure and quality...")
                step_start = time.time()
                data_results = self.data_scout.run()
                pipeline_results['results']['data_scout'] = data_results
                pipeline_results['steps_completed'].append('data_scout')
                print(f"[OK] Data Scout completed in {time.time() - step_start:.1f}s")

            # Step 2: Feature Engineering
            if 'feature_engineer' not in skip_steps:
                print(f"\n{'='*70}")
                print("  STEP 2/4: FEATURE ENGINEER - Creating Powerful Features")
                print("="*70)
                self.log_info("Generating, selecting, and transforming features...")
                step_start = time.time()
                feature_results = self.feature_engineer.run()
                pipeline_results['results']['feature_engineer'] = feature_results
                pipeline_results['steps_completed'].append('feature_engineer')
                print(f"[OK] Feature Engineering completed in {time.time() - step_start:.1f}s")

            # Step 3: Model Selection and Training
            if 'model_selector' not in skip_steps:
                print(f"\n{'='*70}")
                print("  STEP 3/4: MODEL SELECTOR - Training and Optimizing Models")
                print("="*70)
                self.log_info("Training multiple models and finding the best...")
                step_start = time.time()
                model_results = self.model_selector.run()
                pipeline_results['results']['model_selector'] = model_results
                pipeline_results['steps_completed'].append('model_selector')
                print(f"[OK] Model Selection completed in {time.time() - step_start:.1f}s")

            # Step 4: Create Submission
            if 'submission' not in skip_steps:
                print(f"\n{'='*70}")
                print("  STEP 4/4: SUBMISSION - Creating Kaggle Submission File")
                print("="*70)
                step_start = time.time()
                submission_df = self.create_submission()
                if submission_df is not None:
                    pipeline_results['results']['submission_created'] = True
                    pipeline_results['steps_completed'].append('submission')
                    print(f"[OK] Submission created in {time.time() - step_start:.1f}s")

                    # Step 5: Submit to Kaggle (optional)
                    if submit_to_kaggle:
                        print(f"\n{'='*70}")
                        print("  BONUS: Submitting to Kaggle API")
                        print("="*70)
                        submission_success = self.submit_to_kaggle()
                        pipeline_results['results']['kaggle_submission'] = submission_success
                        if submission_success:
                            pipeline_results['steps_completed'].append('kaggle_submission')

            # Create final summary
            pipeline_results['pipeline_status'] = 'completed'
            pipeline_results['final_score'] = model_results.get('best_model_score', 0.0)
            pipeline_results['best_model'] = model_results.get('best_model_name', 'unknown')
            total_time = time.time() - start_time

            self.save_results(pipeline_results, "pipeline_results.json")

            # Print final summary
            print(f"\n{'='*70}")
            print("  PIPELINE COMPLETED SUCCESSFULLY!")
            print("="*70)
            print(f"  Competition: {self.competition_name}")
            print(f"  Best Model: {pipeline_results['best_model']}")
            print(f"  CV Score: {pipeline_results['final_score']:.4f}")
            print(f"  Total Time: {total_time:.1f}s ({total_time/60:.1f} minutes)")
            print(f"  Steps: {' → '.join(pipeline_results['steps_completed'])}")
            print("="*70 + "\n")

            return pipeline_results

        except Exception as e:
            self.log_error(f"Pipeline failed: {e}")
            pipeline_results['pipeline_status'] = 'failed'
            pipeline_results['error'] = str(e)
            raise

    def create_submission(self) -> Optional[pd.DataFrame]:
        """Create final submission file."""
        try:
            # Load data scout results to get dataset info
            data_results = self.load_results("data_scout_results.json")
            if not data_results:
                self.log_error("No data scout results found")
                return None

            target_column = data_results.get('dataset_info', {}).get('target_column', 'target')

            # Load best model results
            model_results = self.load_results("model_selector_results.json")
            if not model_results:
                self.log_warning("No model results found, creating basic submission")

            # Load test data
            _, test_df = self.feature_engineer.get_engineered_data()
            if test_df is None:
                self.log_error("No test data available")
                return None

            # Also try to load original test data to get original IDs
            original_test_df = None
            try:
                from core.data import CompetitionDataLoader
                loader = CompetitionDataLoader(self.competition_path)
                _, original_test_df = loader.load_competition_data()
            except Exception:
                pass

            # Remove target column if it exists in test data
            feature_cols = [col for col in test_df.columns if col != target_column]
            test_features = test_df[feature_cols]

            # Try to get predictions from best model
            predictions = None
            if model_results and 'best_model_name' in model_results:
                try:
                    # Try to load and use the best model
                    predictions = self.model_selector.predict_with_best_model(test_features)
                    self.log_info(f"Generated predictions using {model_results['best_model_name']}")
                except Exception as e:
                    self.log_warning(f"Could not use best model for predictions: {e}")

            # Fallback to dummy predictions if model prediction fails
            if predictions is None:
                self.log_warning("Using dummy predictions (all zeros)")
                predictions = [0] * len(test_features)

            # Ensure predictions are in correct format for Kaggle
            # For classification problems, ensure integer predictions
            if data_results.get('recommendations', {}).get('problem_type') == 'classification':
                predictions = [int(pred) for pred in predictions]

            # Try to use sample submission format if available
            sample_submission_path = self.file_manager.get_file_path("sample_submission.csv")
            submission_df = None

            if sample_submission_path.exists():
                try:
                    sample_df = pd.read_csv(sample_submission_path)
                    if len(sample_df) == len(predictions):
                        # Use sample submission format - keep ID column, replace target
                        submission_df = sample_df.copy()
                        # Replace the second column (target) with our predictions
                        target_col = submission_df.columns[1]
                        submission_df[target_col] = predictions
                        self.log_info(f"Using sample submission format: {list(submission_df.columns)}")
                except Exception as e:
                    self.log_warning(f"Could not use sample submission format: {e}")

            # Fallback: create submission manually
            if submission_df is None:
                # Try to get IDs from original test data first
                id_values = None
                id_column_name = 'PassengerId'  # Default for Titanic

                if original_test_df is not None:
                    original_id_column = self._detect_id_column(original_test_df)
                    if original_id_column and original_id_column in original_test_df.columns:
                        id_values = original_test_df[original_id_column].tolist()
                        id_column_name = original_id_column

                # Fallback to engineered test data IDs
                if id_values is None:
                    id_column = self._detect_id_column(test_df)
                    if id_column and id_column in test_df.columns:
                        id_values = test_df[id_column].tolist()
                        id_column_name = id_column

                # Final fallback: generate IDs
                if id_values is None:
                    # For Titanic specifically, test IDs typically start from 892
                    start_id = 892 if self.competition_name.lower() == 'titanic' else 1
                    id_values = list(range(start_id, start_id + len(test_features)))

                # Create submission DataFrame
                submission_df = pd.DataFrame({
                    id_column_name: id_values,
                    target_column: predictions
                })

            # Save submission file
            submission_path = self.file_manager.get_file_path("submission.csv")
            submission_df.to_csv(submission_path, index=False)
            self.log_info(f"Created submission file: {submission_path}")
            self.log_info(f"Submission shape: {submission_df.shape}")
            self.log_info(f"Columns: {list(submission_df.columns)}")

            return submission_df

        except Exception as e:
            self.log_error(f"Submission creation failed: {e}")
            return None

    def _detect_id_column(self, df: pd.DataFrame) -> Optional[str]:
        """Detect the ID column in the dataset."""
        id_candidates = ['id', 'Id', 'ID', 'PassengerId', 'test_id', 'index']
        for col in id_candidates:
            if col in df.columns:
                return col
        return None

    def submit_to_kaggle(self) -> bool:
        """Submit the created submission file to Kaggle."""
        try:
            submission_path = self.file_manager.get_file_path("submission.csv")
            if not submission_path.exists():
                self.log_error("No submission file found to submit")
                return False

            # Validate submission format
            if not self.kaggle_client.validate_submission_format(submission_path, self.competition_name):
                self.log_error("Submission validation failed")
                return False

            # Create submission message
            pipeline_results = self.load_results("pipeline_results.json")
            best_model = pipeline_results.get('best_model', 'Unknown') if pipeline_results else 'Unknown'
            final_score = pipeline_results.get('final_score', 0.0) if pipeline_results else 0.0

            message = f"KaggleSlayer submission - Model: {best_model}, CV Score: {final_score:.4f}"

            # Submit to Kaggle
            success = self.kaggle_client.submit_to_competition(
                competition_name=self.competition_name,
                submission_file=submission_path,
                message=message
            )

            if success:
                self.log_info("✅ Successfully submitted to Kaggle!")
                # Check submission status
                self.kaggle_client.get_submission_status(self.competition_name)
            else:
                self.log_error("❌ Kaggle submission failed")

            return success

        except Exception as e:
            self.log_error(f"Kaggle submission error: {e}")
            return False