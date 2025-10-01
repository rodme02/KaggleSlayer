#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Main entry point for running the complete KaggleSlayer pipeline.
"""

import argparse
import sys
import io
from pathlib import Path

# Fix encoding issues on Windows
if sys.platform == 'win32':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')

# Add the parent directory to the path so we can import modules
sys.path.insert(0, str(Path(__file__).parent.parent))

from agents import PipelineCoordinator
from utils.config import ConfigManager
from utils.logging import setup_logging


def main():
    parser = argparse.ArgumentParser(description="Run KaggleSlayer pipeline")
    parser.add_argument("competition_name", help="Name of the competition")
    parser.add_argument("--competition-path", "-p",
                       help="Path to competition data directory")
    parser.add_argument("--config", "-c",
                       help="Path to configuration file")
    parser.add_argument("--skip-steps", nargs="*",
                       choices=["data_scout", "feature_engineer", "model_selector", "submission"],
                       help="Pipeline steps to skip")
    parser.add_argument("--submit", action="store_true",
                       help="Submit results to Kaggle competition")
    parser.add_argument("--log-level", default="INFO",
                       choices=["DEBUG", "INFO", "WARNING", "ERROR"],
                       help="Logging level")

    args = parser.parse_args()

    # Set up logging
    setup_logging(log_level=args.log_level)

    # Determine competition path
    if args.competition_path:
        competition_path = Path(args.competition_path)
    else:
        competition_path = Path("competition_data") / args.competition_name

    if not competition_path.exists():
        print(f"Error: Competition path does not exist: {competition_path}")
        return 1

    # Load configuration
    if args.config:
        config = ConfigManager(Path(args.config))
    else:
        config = ConfigManager()

    try:
        # Initialize and run pipeline
        coordinator = PipelineCoordinator(args.competition_name, competition_path, config)

        print(f"Starting KaggleSlayer pipeline for: {args.competition_name}")
        print(f"Competition path: {competition_path}")

        results = coordinator.run(skip_steps=args.skip_steps or [], submit_to_kaggle=args.submit)

        print("\n" + "="*50)
        print("PIPELINE COMPLETED SUCCESSFULLY!")
        print("="*50)
        print(f"Final Score: {results.get('final_score', 'N/A')}")
        print(f"Best Model: {results.get('best_model', 'N/A')}")
        print(f"Steps Completed: {', '.join(results.get('steps_completed', []))}")

        # Show submission status
        if 'submission_created' in results.get('results', {}):
            print(f"✅ Submission file created")
        if 'kaggle_submission' in results.get('results', {}):
            if results['results']['kaggle_submission']:
                print(f"✅ Successfully submitted to Kaggle!")
            else:
                print(f"❌ Kaggle submission failed")

        return 0

    except Exception as e:
        print(f"\nPipeline failed: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())