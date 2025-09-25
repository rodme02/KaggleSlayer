#!/usr/bin/env python3
"""
End-to-End Pipeline Runner
Orchestrates the complete KaggleSlayer pipeline: Data Scout → Baseline Model → Submitter
"""

import argparse
import sys
from pathlib import Path
from datetime import datetime

from agents.data_scout import DataScout
from agents.baseline_model import BaselineModel
from agents.submitter import Submitter


def run_complete_pipeline(competition_path: Path,
                         skip_scout: bool = False,
                         skip_model: bool = False,
                         skip_submit: bool = False,
                         dry_run: bool = False,
                         submission_message: str = None,
                         competition_ref: str = None):
    """Run the complete end-to-end pipeline"""

    competition_name = competition_path.name
    print(f"KAGGLE SLAYER - COMPLETE PIPELINE")
    print(f"Competition: {competition_name}")
    print(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 70)

    results = {
        'scout': None,
        'model': None,
        'submission': None,
        'success': False
    }

    try:
        # STEP 1: Data Scout Analysis
        if not skip_scout:
            print("\nSTEP 1: Data Scout Analysis")
            print("-" * 40)

            scout = DataScout(competition_path)
            scout_output_dir = competition_path / "scout_output"

            # Check if already analyzed
            if scout_output_dir.exists() and not skip_scout:
                print(f"Scout output already exists, using existing analysis")
            else:
                cleaned_df, dataset_info, quality_report = scout.run_full_analysis()
                results['scout'] = {
                    'dataset_info': dataset_info,
                    'quality_report': quality_report
                }
                print(f"Data Scout completed successfully")
        else:
            print("\nSTEP 1: Data Scout (SKIPPED)")

        # STEP 2: Baseline Model Training
        if not skip_model:
            print("\nSTEP 2: Baseline Model Training")
            print("-" * 40)

            baseline = BaselineModel(competition_path, use_scout_output=not skip_scout)
            model_results, predictions = baseline.run_baseline_pipeline()

            results['model'] = {
                'cv_score': model_results.cv_mean,
                'cv_std': model_results.cv_std,
                'model_type': model_results.model_type,
                'problem_type': model_results.problem_type
            }

            print(f"Baseline Model completed successfully")
            print(f"   Model: {model_results.model_type}")
            print(f"   CV Score: {model_results.cv_mean:.4f} (+/- {model_results.cv_std * 2:.4f})")
        else:
            print("\nSTEP 2: Baseline Model (SKIPPED)")

        # STEP 3: Submission
        if not skip_submit:
            print("\nSTEP 3: Kaggle Submission")
            print("-" * 40)

            submitter = Submitter(competition_path)

            if not submission_message:
                if results['model']:
                    submission_message = f"Baseline {results['model']['model_type']} - CV: {results['model']['cv_score']:.4f}"
                else:
                    submission_message = "Baseline submission"

            submission_result = submitter.run_submission_pipeline(
                message=submission_message,
                competition_ref=competition_ref,
                dry_run=dry_run
            )

            results['submission'] = {
                'status': submission_result.submission_status,
                'submission_id': submission_result.submission_id,
                'public_score': submission_result.public_score,
                'error_message': submission_result.error_message
            }

            if submission_result.submission_status in ['submitted', 'dry_run']:
                print(f"Submission completed successfully")
                if submission_result.public_score:
                    print(f"   Public Score: {submission_result.public_score}")
            else:
                print(f"Submission failed: {submission_result.error_message}")
        else:
            print("\nSTEP 3: Submission (SKIPPED)")

        # Pipeline Summary
        print("\n" + "=" * 70)
        print("PIPELINE SUMMARY")
        print("=" * 70)

        if results['scout']:
            print(f"Data Scout:")
            print(f"   Target: {results['scout']['dataset_info'].target_column}")
            print(f"   Problem: {results['scout']['dataset_info'].target_type}")
            print(f"   Features: {results['scout']['dataset_info'].total_columns}")
            print(f"   Rows: {results['scout']['dataset_info'].total_rows:,}")

        if results['model']:
            print(f"Baseline Model:")
            print(f"   Type: {results['model']['model_type']}")
            print(f"   CV Score: {results['model']['cv_score']:.4f} (+/- {results['model']['cv_std'] * 2:.4f})")

        if results['submission']:
            print(f"Submission:")
            print(f"   Status: {results['submission']['status']}")
            if results['submission']['public_score']:
                print(f"   Public Score: {results['submission']['public_score']}")
            elif results['submission']['error_message']:
                print(f"   Error: {results['submission']['error_message']}")

        # Determine overall success
        pipeline_success = True
        if not skip_scout and not results['scout']:
            pipeline_success = False
        if not skip_model and not results['model']:
            pipeline_success = False
        if not skip_submit and results['submission'] and results['submission']['status'] not in ['submitted', 'dry_run']:
            pipeline_success = False

        results['success'] = pipeline_success

        if pipeline_success:
            print(f"\nPIPELINE COMPLETED SUCCESSFULLY!")
        else:
            print(f"\nPIPELINE COMPLETED WITH ISSUES")

        print(f"Finished at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print("=" * 70)

        return results

    except Exception as e:
        print(f"\nPIPELINE FAILED: {e}")
        results['success'] = False
        return results


def main():
    """Main entry point for the complete pipeline"""
    parser = argparse.ArgumentParser(description="Complete KaggleSlayer Pipeline")
    parser.add_argument("competition_path", type=Path,
                       help="Path to competition directory")
    parser.add_argument("--skip-scout", action="store_true",
                       help="Skip data scout analysis")
    parser.add_argument("--skip-model", action="store_true",
                       help="Skip baseline model training")
    parser.add_argument("--skip-submit", action="store_true",
                       help="Skip Kaggle submission")
    parser.add_argument("--dry-run", action="store_true",
                       help="Create submission file but don't submit to Kaggle")
    parser.add_argument("--message", type=str, default=None,
                       help="Custom submission message")
    parser.add_argument("--competition", type=str, default=None,
                       help="Kaggle competition name (default: infer from path)")

    # Convenience flags
    parser.add_argument("--scout-only", action="store_true",
                       help="Run only data scout analysis")
    parser.add_argument("--model-only", action="store_true",
                       help="Run only baseline model training")
    parser.add_argument("--submit-only", action="store_true",
                       help="Run only submission")

    args = parser.parse_args()

    # Validate competition path
    if not args.competition_path.exists():
        print(f"ERROR: Competition path does not exist: {args.competition_path}")
        return 1

    train_csv = args.competition_path / "train.csv"
    if not train_csv.exists():
        print(f"ERROR: Training data not found: {train_csv}")
        return 1

    # Handle convenience flags
    if args.scout_only:
        args.skip_model = True
        args.skip_submit = True
    elif args.model_only:
        args.skip_scout = True
        args.skip_submit = True
    elif args.submit_only:
        args.skip_scout = True
        args.skip_model = True

    # Run the pipeline
    results = run_complete_pipeline(
        competition_path=args.competition_path,
        skip_scout=args.skip_scout,
        skip_model=args.skip_model,
        skip_submit=args.skip_submit,
        dry_run=args.dry_run,
        submission_message=args.message,
        competition_ref=args.competition
    )

    # Return appropriate exit code
    return 0 if results['success'] else 1


if __name__ == "__main__":
    sys.exit(main())