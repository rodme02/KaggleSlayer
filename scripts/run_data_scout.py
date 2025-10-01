#!/usr/bin/env python3
"""
Entry point for running just the data scouting phase.
"""

import argparse
import sys
from pathlib import Path

# Add the parent directory to the path so we can import modules
sys.path.insert(0, str(Path(__file__).parent.parent))

from agents import DataScoutAgent
from utils.config import ConfigManager
from utils.logging import setup_logging


def main():
    parser = argparse.ArgumentParser(description="Run KaggleSlayer Data Scout")
    parser.add_argument("competition_name", help="Name of the competition")
    parser.add_argument("--competition-path", "-p",
                       help="Path to competition data directory")
    parser.add_argument("--config", "-c",
                       help="Path to configuration file")
    parser.add_argument("--no-save", action="store_true",
                       help="Don't save cleaned data files")
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
        # Initialize and run data scout
        scout = DataScoutAgent(args.competition_name, competition_path, config)

        print(f"Starting Data Scout for: {args.competition_name}")
        print(f"Competition path: {competition_path}")

        results = scout.run(save_cleaned_data=not args.no_save)

        print("\n" + "="*50)
        print("DATA SCOUTING COMPLETED!")
        print("="*50)
        print(f"Dataset rows: {results['dataset_info']['total_rows']}")
        print(f"Dataset columns: {results['dataset_info']['total_columns']}")
        print(f"Target column: {results['dataset_info']['target_column']}")
        print(f"Problem type: {results['recommendations']['problem_type']}")
        print(f"Memory usage: {results['dataset_info']['memory_usage_mb']:.2f} MB")

        return 0

    except Exception as e:
        print(f"\nData scouting failed: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())