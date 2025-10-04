"""
KaggleSlayer - Simple AutoML Pipeline for Kaggle Competitions

Usage:
    python kaggle_slayer.py <competition_name> --data-path <path_to_data>

Example:
    python kaggle_slayer.py titanic --data-path competition_data/titanic
"""

import argparse
from pathlib import Path
from agents.coordinator import PipelineCoordinator


def main():
    parser = argparse.ArgumentParser(description="KaggleSlayer - AutoML for Kaggle")
    parser.add_argument("competition", help="Competition name")
    parser.add_argument("--data-path", required=True, help="Path to competition data directory")
    parser.add_argument("--submit", action="store_true", help="Submit to Kaggle after creating submission file")

    args = parser.parse_args()

    # Run pipeline
    print(f"\n{'='*60}")
    print(f"  KaggleSlayer - Running pipeline for: {args.competition}")
    print(f"{'='*60}\n")

    coordinator = PipelineCoordinator(args.competition, Path(args.data_path))
    results = coordinator.run(submit_to_kaggle=args.submit)

    print(f"\n{'='*60}")
    print(f"  Pipeline Complete!")
    print(f"  Best Model: {results.get('best_model', 'N/A')}")
    print(f"  CV Score: {results.get('final_score', 0):.4f}")
    print(f"  Submission saved to: {args.data_path}/submission.csv")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
