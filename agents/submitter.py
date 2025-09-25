#!/usr/bin/env python3
"""
Submitter Agent - Handles Kaggle API submissions

Capabilities:
- Format predictions according to Kaggle requirements
- Submit predictions via Kaggle API
- Log submission results and leaderboard scores
- Manage submission history
"""

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, asdict
from datetime import datetime

import pandas as pd

try:
    from kaggle.api.kaggle_api_extended import KaggleApi
    KAGGLE_AVAILABLE = True
except ImportError:
    print("Warning: Kaggle API not available. Install with: pip install kaggle")
    KaggleApi = None
    KAGGLE_AVAILABLE = False


@dataclass
class SubmissionResult:
    """Structure to hold submission result information"""
    competition_name: str
    submission_file: str
    submission_message: str
    submission_id: Optional[str]
    public_score: Optional[float]
    private_score: Optional[float]
    submission_status: str
    submission_timestamp: str
    error_message: Optional[str]


class Submitter:
    """
    Submitter Agent for handling Kaggle competition submissions
    """

    def __init__(self, competition_path: Path):
        self.competition_path = Path(competition_path)
        self.competition_name = self.competition_path.name
        self.submission_dir = self.competition_path / "submissions"
        self.submission_dir.mkdir(exist_ok=True)

        # Kaggle API
        self.api = None
        if KAGGLE_AVAILABLE:
            try:
                self.api = KaggleApi()
                self.api.authenticate()
                print("Kaggle API authenticated successfully")
            except Exception as e:
                print(f"Warning: Kaggle API authentication failed: {e}")
                self.api = None
        else:
            print("Kaggle API not available")

    def load_predictions(self, predictions_path: Path = None) -> pd.DataFrame:
        """Load predictions from file"""
        if predictions_path is None:
            # Look for predictions in baseline model directory
            baseline_dir = self.competition_path / "baseline_model"
            predictions_path = baseline_dir / "predictions.csv"

        if not predictions_path.exists():
            raise FileNotFoundError(f"Predictions file not found: {predictions_path}")

        predictions = pd.read_csv(predictions_path)
        print(f"Loaded predictions: {predictions.shape}")
        return predictions

    def get_competition_format(self, competition_ref: str = None) -> Dict:
        """Get competition submission format from Kaggle API"""
        if not self.api:
            return self._get_default_format()

        try:
            if not competition_ref:
                competition_ref = self.competition_name

            # Get competition details
            comp_info = self.api.competition_view(competition_ref)

            # Try to get sample submission format
            files = self.api.competition_download_files(
                competition_ref,
                path=self.competition_path,
                quiet=True
            )

            sample_sub_path = self.competition_path / "sample_submission.csv"
            if sample_sub_path.exists():
                sample_sub = pd.read_csv(sample_sub_path)
                return {
                    'id_column': sample_sub.columns[0],
                    'prediction_columns': list(sample_sub.columns[1:]),
                    'sample_format': sample_sub.head()
                }

        except Exception as e:
            print(f"Warning: Could not get competition format from API: {e}")

        return self._get_default_format()

    def _get_default_format(self) -> Dict:
        """Get default submission format based on common patterns"""
        return {
            'id_column': 'Id',  # Will be detected from predictions
            'prediction_columns': ['prediction'],  # Will be detected
            'sample_format': None
        }

    def format_submission(self, predictions: pd.DataFrame,
                         competition_ref: str = None) -> pd.DataFrame:
        """Format predictions according to competition requirements"""
        print("Formatting submission file...")

        # Get competition format
        format_info = self.get_competition_format(competition_ref)

        # Auto-detect ID column
        id_candidates = ['Id', 'id', 'ID', 'PassengerId', 'test_id']
        id_column = None
        for candidate in id_candidates:
            if candidate in predictions.columns:
                id_column = candidate
                break

        if id_column is None:
            # Use first column as ID
            id_column = predictions.columns[0]
            print(f"Warning: Using first column '{id_column}' as ID column")

        # Auto-detect prediction columns (non-ID columns)
        prediction_columns = [col for col in predictions.columns if col != id_column]

        # For most competitions, we want just the ID and main prediction
        if len(prediction_columns) > 1:
            # Look for the main prediction column
            main_pred_candidates = ['target', 'prediction', 'Survived', 'SalePrice']
            main_pred_col = None

            for candidate in main_pred_candidates:
                if candidate in prediction_columns:
                    main_pred_col = candidate
                    break

            if main_pred_col is None:
                # Use first non-ID column
                main_pred_col = prediction_columns[0]

            print(f"Using '{main_pred_col}' as main prediction column")
            formatted_sub = predictions[[id_column, main_pred_col]].copy()
        else:
            formatted_sub = predictions[[id_column] + prediction_columns].copy()

        print(f"Formatted submission shape: {formatted_sub.shape}")
        print(f"Columns: {list(formatted_sub.columns)}")

        # Show sample of formatted submission
        print("\nSubmission preview:")
        print(formatted_sub.head())

        return formatted_sub

    def save_submission(self, submission_df: pd.DataFrame,
                       message: str = "Baseline submission") -> Path:
        """Save submission to file"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"submission_{timestamp}.csv"
        submission_path = self.submission_dir / filename

        submission_df.to_csv(submission_path, index=False)
        print(f"Saved submission: {submission_path}")

        # Also save as latest submission
        latest_path = self.submission_dir / "latest_submission.csv"
        submission_df.to_csv(latest_path, index=False)
        print(f"Saved as latest: {latest_path}")

        return submission_path

    def submit_to_kaggle(self, submission_path: Path,
                        message: str = "Baseline submission",
                        competition_ref: str = None) -> SubmissionResult:
        """Submit predictions to Kaggle"""
        if not competition_ref:
            competition_ref = self.competition_name

        print(f"\nSubmitting to Kaggle competition: {competition_ref}")
        print(f"Submission file: {submission_path}")
        print(f"Message: {message}")

        result = SubmissionResult(
            competition_name=competition_ref,
            submission_file=str(submission_path),
            submission_message=message,
            submission_id=None,
            public_score=None,
            private_score=None,
            submission_status="failed",
            submission_timestamp=datetime.now().isoformat(),
            error_message=None
        )

        if not self.api:
            result.error_message = "Kaggle API not available"
            print("ERROR: Kaggle API not available")
            return result

        try:
            # Submit to competition
            submit_result = self.api.competition_submit(
                file_name=str(submission_path),
                message=message,
                competition=competition_ref
            )

            print("Submission successful!")
            result.submission_status = "submitted"

            # Try to get submission info
            try:
                submissions = self.api.competition_submissions(competition_ref)
                if submissions:
                    latest_sub = submissions[0]  # Most recent submission
                    result.submission_id = str(latest_sub.ref)
                    result.public_score = latest_sub.publicScore
                    result.private_score = latest_sub.privateScore

                    print(f"Submission ID: {result.submission_id}")
                    if result.public_score is not None:
                        print(f"Public Score: {result.public_score}")
                    else:
                        print("Public score not yet available")

            except Exception as e:
                print(f"Warning: Could not retrieve submission details: {e}")

        except Exception as e:
            result.error_message = str(e)
            print(f"ERROR: Submission failed: {e}")

        return result

    def save_submission_log(self, result: SubmissionResult):
        """Save submission result to log"""
        log_path = self.submission_dir / "submission_log.jsonl"

        # Append to log file
        with open(log_path, 'a') as f:
            json.dump(asdict(result), f)
            f.write('\n')

        print(f"Logged submission result: {log_path}")

    def get_submission_history(self) -> List[SubmissionResult]:
        """Get submission history from log"""
        log_path = self.submission_dir / "submission_log.jsonl"

        if not log_path.exists():
            return []

        history = []
        with open(log_path, 'r') as f:
            for line in f:
                if line.strip():
                    data = json.loads(line.strip())
                    history.append(SubmissionResult(**data))

        return history

    def print_submission_summary(self, result: SubmissionResult):
        """Print submission summary"""
        print("\n" + "="*50)
        print("SUBMISSION SUMMARY")
        print("="*50)
        print(f"Competition: {result.competition_name}")
        print(f"Status: {result.submission_status}")
        print(f"Timestamp: {result.submission_timestamp}")

        if result.submission_status == "submitted":
            print(f"Submission ID: {result.submission_id}")
            if result.public_score is not None:
                print(f"Public Score: {result.public_score}")
            else:
                print("Public Score: Pending...")
        else:
            print(f"Error: {result.error_message}")
        print("="*50)

    def run_submission_pipeline(self, predictions_path: Path = None,
                              message: str = "Baseline submission",
                              competition_ref: str = None,
                              dry_run: bool = False) -> SubmissionResult:
        """Run the complete submission pipeline"""
        print(f"Starting Submission Pipeline for {self.competition_name}")
        print("=" * 60)

        try:
            # Load predictions
            predictions = self.load_predictions(predictions_path)

            # Format submission
            formatted_sub = self.format_submission(predictions, competition_ref)

            # Save submission file
            submission_path = self.save_submission(formatted_sub, message)

            if dry_run:
                print("\nDRY RUN: Skipping actual submission to Kaggle")
                result = SubmissionResult(
                    competition_name=competition_ref or self.competition_name,
                    submission_file=str(submission_path),
                    submission_message=message,
                    submission_id="dry_run",
                    public_score=None,
                    private_score=None,
                    submission_status="dry_run",
                    submission_timestamp=datetime.now().isoformat(),
                    error_message=None
                )
            else:
                # Submit to Kaggle
                result = self.submit_to_kaggle(submission_path, message, competition_ref)

            # Save to log
            self.save_submission_log(result)

            # Print summary
            self.print_submission_summary(result)

            return result

        except Exception as e:
            error_result = SubmissionResult(
                competition_name=competition_ref or self.competition_name,
                submission_file="",
                submission_message=message,
                submission_id=None,
                public_score=None,
                private_score=None,
                submission_status="error",
                submission_timestamp=datetime.now().isoformat(),
                error_message=str(e)
            )

            self.save_submission_log(error_result)
            print(f"ERROR: Submission pipeline failed: {e}")
            return error_result


def main():
    """Main entry point for the Submitter Agent"""
    parser = argparse.ArgumentParser(description="Submitter Agent for Kaggle")
    parser.add_argument("competition_path", type=Path,
                       help="Path to competition directory")
    parser.add_argument("--predictions", type=Path, default=None,
                       help="Path to predictions CSV file")
    parser.add_argument("--message", type=str, default="Baseline submission",
                       help="Submission message")
    parser.add_argument("--competition", type=str, default=None,
                       help="Kaggle competition name (default: infer from path)")
    parser.add_argument("--dry-run", action="store_true",
                       help="Create submission file but don't submit to Kaggle")
    parser.add_argument("--history", action="store_true",
                       help="Show submission history")

    args = parser.parse_args()

    # Validate competition path
    if not args.competition_path.exists():
        print(f"ERROR: Competition path does not exist: {args.competition_path}")
        return 1

    try:
        # Initialize submitter
        submitter = Submitter(args.competition_path)

        # Show history if requested
        if args.history:
            history = submitter.get_submission_history()
            if history:
                print("\nSubmission History:")
                print("-" * 50)
                for i, result in enumerate(history[-5:], 1):  # Show last 5
                    print(f"{i}. {result.submission_timestamp}")
                    print(f"   Status: {result.submission_status}")
                    if result.public_score:
                        print(f"   Score: {result.public_score}")
                    print()
            else:
                print("No submission history found")
            return 0

        # Run submission pipeline
        result = submitter.run_submission_pipeline(
            predictions_path=args.predictions,
            message=args.message,
            competition_ref=args.competition,
            dry_run=args.dry_run
        )

        if result.submission_status in ["submitted", "dry_run"]:
            print("\nSubmission pipeline completed successfully!")
            return 0
        else:
            print("\nSubmission pipeline failed!")
            return 1

    except Exception as e:
        print(f"ERROR: Error during submission: {e}")
        return 1


if __name__ == "__main__":
    exit(main())