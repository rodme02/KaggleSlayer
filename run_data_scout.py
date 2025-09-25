#!/usr/bin/env python3
"""
Convenience script to run the Data Scout Agent on all downloaded competitions
"""

import argparse
import sys
from pathlib import Path
from agents.data_scout import DataScout


def run_scout_on_competition(competition_dir: Path, force: bool = False):
    """Run data scout on a single competition"""
    competition_name = competition_dir.name
    scout_output = competition_dir / "scout_output"

    # Check if already analyzed
    if scout_output.exists() and not force:
        print(f"Skipping {competition_name} (already analyzed, use --force to re-run)")
        return True

    try:
        print(f"\nRunning Data Scout on: {competition_name}")
        print("=" * 50)

        scout = DataScout(competition_dir)
        cleaned_df, dataset_info, quality_report = scout.run_full_analysis()

        return True

    except Exception as e:
        print(f"ERROR: Failed to analyze {competition_name}: {e}")
        return False


def main():
    parser = argparse.ArgumentParser(description="Run Data Scout on competition datasets")
    parser.add_argument("--datasets-dir", type=Path, default="downloaded_datasets",
                       help="Directory containing competition datasets")
    parser.add_argument("--competition", type=str, default=None,
                       help="Specific competition to analyze (default: all)")
    parser.add_argument("--force", action="store_true",
                       help="Force re-analysis even if already done")

    args = parser.parse_args()

    datasets_dir = Path(args.datasets_dir)
    if not datasets_dir.exists():
        print(f"ERROR: Datasets directory not found: {datasets_dir}")
        return 1

    success_count = 0
    total_count = 0

    if args.competition:
        # Analyze specific competition
        competition_dir = datasets_dir / args.competition
        if not competition_dir.exists():
            print(f"ERROR: Competition not found: {competition_dir}")
            return 1

        total_count = 1
        if run_scout_on_competition(competition_dir, args.force):
            success_count = 1
    else:
        # Analyze all competitions
        competition_dirs = [d for d in datasets_dir.iterdir()
                          if d.is_dir() and (d / "train.csv").exists()]

        if not competition_dirs:
            print("No competition directories with train.csv found")
            return 1

        total_count = len(competition_dirs)
        print(f"Found {total_count} competition(s) to analyze")

        for comp_dir in competition_dirs:
            if run_scout_on_competition(comp_dir, args.force):
                success_count += 1

    print(f"\nSummary: {success_count}/{total_count} competitions analyzed successfully")
    return 0 if success_count == total_count else 1


if __name__ == "__main__":
    sys.exit(main())