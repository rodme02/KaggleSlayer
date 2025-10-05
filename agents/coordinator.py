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
    """Coordinates the entire KaggleSlayer pipeline execution.

    Orchestrates all pipeline stages in sequence:
    1. Data Scout: Data exploration, cleaning, and initial analysis
    2. Feature Engineer: Feature generation, selection, and transformation
    3. Model Selector: Model training, evaluation, and selection
    4. Submission: Create and optionally submit predictions to Kaggle

    Attributes:
        data_scout: Agent for data exploration and cleaning
        feature_engineer: Agent for feature engineering
        model_selector: Agent for model training and selection
        kaggle_client: Client for Kaggle API interactions
    """

    def __init__(self, competition_name: str, competition_path: Path,
                 config: Optional[ConfigManager] = None):
        """Initialize the pipeline coordinator.

        Args:
            competition_name: Name of the Kaggle competition
            competition_path: Path to competition data directory
            config: Optional configuration manager
        """
        super().__init__(competition_name, competition_path, config)

        # Initialize agents
        self.data_scout = DataScoutAgent(competition_name, competition_path, config)
        self.feature_engineer = FeatureEngineerAgent(competition_name, competition_path, config)
        self.model_selector = ModelSelectorAgent(competition_name, competition_path, config)

        # Initialize Kaggle API client
        self.kaggle_client = KaggleAPIClient()

    def run(self, submit_to_kaggle: bool = False) -> Dict[str, Any]:
        """Run the complete pipeline from data loading to submission.

        Executes all pipeline stages in sequence and provides comprehensive
        progress updates and final summary.

        Args:
            submit_to_kaggle: Whether to submit results to Kaggle after completion

        Returns:
            Dictionary containing:
                - competition_name: Competition identifier
                - pipeline_status: 'completed' or 'failed'
                - steps_completed: List of completed stage names
                - results: Results from each pipeline stage
                - best_model: Name of the best performing model
                - final_score: Cross-validation score of best model

        Raises:
            Exception: Any error that occurs during pipeline execution
        """
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
            print(f"\n{'='*70}")
            print("  STEP 3/4: MODEL SELECTOR - Training and Optimizing Models")
            print("="*70)
            self.log_info("Training multiple models and finding the best...")
            step_start = time.time()
            # Disable hyperparameter optimization by default to keep output clean
            # (can be enabled by modifying this line)
            model_results = self.model_selector.run(optimize_hyperparameters=False)
            pipeline_results['results']['model_selector'] = model_results
            pipeline_results['steps_completed'].append('model_selector')
            print(f"[OK] Model Selection completed in {time.time() - step_start:.1f}s")

            # Step 4: Create Submission
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

            self.file_manager.save_results(pipeline_results, "pipeline_results.json")

            # Print final summary
            print(f"\n{'='*70}")
            print("  PIPELINE COMPLETED SUCCESSFULLY!")
            print("="*70)
            print(f"  Competition: {self.competition_name}")
            print(f"  Best Model: {pipeline_results['best_model']}")
            print(f"  CV Score: {pipeline_results['final_score']:.4f}")
            print(f"  Total Time: {total_time:.1f}s ({total_time/60:.1f} minutes)")
            print(f"  Steps: {' -> '.join(pipeline_results['steps_completed'])}")
            print("="*70 + "\n")

            return pipeline_results

        except Exception as e:
            self.log_error(f"Pipeline failed: {e}")
            pipeline_results['pipeline_status'] = 'failed'
            pipeline_results['error'] = str(e)
            raise

    def create_submission(self) -> Optional[pd.DataFrame]:
        """Create submission file with predictions for test data.

        Generates predictions using the trained pipeline (feature engineering + model)
        and formats them according to Kaggle submission requirements.

        Process:
        1. Load data scout results to get target column name
        2. Look for sample_submission.csv template
        3. Load cleaned test data from processed directory
        4. Load trained model pipeline
        5. Generate predictions
        6. Format and save submission.csv

        Returns:
            DataFrame with submission (ID column + predictions), or None if failed
        """
        try:
            # Load data scout results
            data_results = self.file_manager.load_results("data_scout_results.json")
            if not data_results:
                self.log_error("No data scout results found")
                return None

            target_column = data_results.get('dataset_info', {}).get('target_column')
            if not target_column:
                self.log_error("No target column identified")
                return None

            # Try to find sample submission file first
            sample_submission_df = self._find_sample_submission()

            # If no sample submission, load original test data for structure
            if sample_submission_df is None:
                from core.data import CompetitionDataLoader
                loader = CompetitionDataLoader(self.competition_path)
                _, test_df = loader.load_competition_data()

                if test_df is None:
                    self.log_error("No test data available")
                    return None

                # Detect ID column
                id_column = self._detect_id_column(test_df)
                if not id_column:
                    self.log_error("Could not detect ID column in test data")
                    return None
            else:
                # Use sample submission structure
                id_column = sample_submission_df.columns[0]
                test_df = None  # We'll use sample_submission_df for IDs
                self.log_info(f"Using sample submission template with columns: {list(sample_submission_df.columns)}")

            # Load CLEANED test data from processed directory (ID column already added back)
            _, test_cleaned = self.file_manager.load_processed_data("train_cleaned.csv"), None
            if self.file_manager.file_exists(f"{self.file_manager.processed_dir}/test_cleaned.csv"):
                test_cleaned = self.file_manager.load_processed_data("test_cleaned.csv")

            if test_cleaned is None:
                self.log_error("No cleaned test data available")
                return None

            # Store ID column values before dropping
            if id_column and id_column in test_cleaned.columns:
                id_values = test_cleaned[id_column].copy()
                test_features = test_cleaned.drop(columns=[id_column])
            else:
                id_values = None
                test_features = test_cleaned

            # Get model results
            model_results = self.file_manager.load_results("model_selector_results.json")
            if not model_results or 'best_model_name' not in model_results:
                self.log_error("No model results found")
                return None

            # Load the saved pipeline (includes feature engineering + model)
            import joblib
            model_path = self.file_manager.get_file_path("best_model.pkl")
            trained_pipeline = joblib.load(model_path)

            # Generate predictions using the full pipeline
            # The pipeline will handle feature engineering internally
            predictions = trained_pipeline.predict(test_features)
            self.log_info(f"Generated {len(predictions)} predictions using {model_results['best_model_name']}")

            # Format predictions for classification
            if data_results.get('recommendations', {}).get('problem_type') == 'classification':
                predictions = [int(pred) for pred in predictions]

            # Validate prediction count and create submission
            if sample_submission_df is not None:
                # Using sample submission template
                if len(predictions) != len(sample_submission_df):
                    self.log_error(f"Prediction count mismatch: {len(predictions)} vs {len(sample_submission_df)} expected")
                    return None

                submission_df = sample_submission_df.copy()
                submission_df.iloc[:, 1] = predictions  # Replace second column (target) with predictions
            else:
                # Using test_cleaned structure with stored ID values
                if id_values is None:
                    self.log_error("No ID column values available for submission")
                    return None

                if len(predictions) != len(id_values):
                    self.log_error(f"Prediction count mismatch: {len(predictions)} vs {len(id_values)} expected")
                    return None

                submission_df = pd.DataFrame({
                    id_column: id_values.values,
                    target_column: predictions
                })

            # Save submission
            submission_path = self.file_manager.get_file_path("submission.csv")
            submission_df.to_csv(submission_path, index=False)

            self.log_info(f"Created submission: {submission_df.shape[0]} rows")
            self.log_info(f"Columns: {list(submission_df.columns)}")
            self.log_info(f"Saved to: {submission_path}")

            return submission_df

        except Exception as e:
            self.log_error(f"Submission creation failed: {e}")
            import traceback
            self.log_error(traceback.format_exc())
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

            # Create submission message - try to load pipeline results, fall back to model results
            try:
                pipeline_results = self.file_manager.load_results("pipeline_results.json")
                best_model = pipeline_results.get('best_model', 'Unknown')
                final_score = pipeline_results.get('final_score', 0.0)
            except (FileNotFoundError, Exception):
                # Fallback to model selector results
                try:
                    model_results = self.file_manager.load_results("model_selector_results.json")
                    best_model = model_results.get('best_model_name', 'Unknown')
                    final_score = model_results.get('best_model_score', 0.0)
                except (FileNotFoundError, Exception):
                    best_model = 'Unknown'
                    final_score = 0.0

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

    def _find_sample_submission(self) -> Optional[pd.DataFrame]:
        """Find and load sample submission file.

        Looks for any CSV file containing 'submission' in the filename
        in both raw/ subdirectory and root directory.
        """
        possible_paths = [
            self.competition_path / "raw",
            self.competition_path
        ]

        for base_path in possible_paths:
            if not base_path.exists():
                continue

            # Find CSV files containing 'submission'
            for file_path in base_path.glob("*.csv"):
                if "submission" in file_path.name.lower() and file_path.name.lower() != "submission.csv":
                    try:
                        df = pd.read_csv(file_path)
                        self.log_info(f"Found sample submission: {file_path.name}")
                        return df
                    except Exception as e:
                        self.log_warning(f"Could not load {file_path.name}: {e}")

        return None

    def _detect_id_column(self, df: pd.DataFrame) -> Optional[str]:
        """Detect the ID column in the dataset.

        Checks both exact matches and if 'id' or 'ID' appears in the column name.
        """
        # First check exact matches (common patterns)
        exact_candidates = ['id', 'Id', 'ID', 'test_id', 'index']
        for col in exact_candidates:
            if col in df.columns:
                return col

        # Then check if any column contains 'id' or 'ID' (case-insensitive)
        for col in df.columns:
            col_lower = col.lower()
            if 'id' in col_lower or col_lower == 'index':
                return col

        return None