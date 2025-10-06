"""
KaggleSlayer - Simple AutoML Pipeline for Kaggle Competitions

Usage:
    # Single competition
    python kaggle_slayer.py <competition_name> --data-path <path_to_data>

    # All competitions
    python kaggle_slayer.py --all

Examples:
    python kaggle_slayer.py titanic --data-path competition_data/titanic
    python kaggle_slayer.py --all --submit
"""

import argparse
import warnings
from pathlib import Path
from agents.coordinator import PipelineCoordinator

# Suppress sklearn deprecation warnings for CatBoost compatibility
warnings.filterwarnings('ignore', category=DeprecationWarning, module='sklearn')


def get_all_competitions():
    """Get list of all downloaded competitions."""
    competition_data_dir = Path("competition_data")

    if not competition_data_dir.exists():
        return []

    competitions = []
    for comp_dir in competition_data_dir.iterdir():
        if comp_dir.is_dir():
            # Check if it has raw data
            raw_dir = comp_dir / "raw"
            if raw_dir.exists() and (raw_dir / "train.csv").exists():
                competitions.append(comp_dir.name)

    return sorted(competitions)


def run_single_competition(competition_name, data_path, submit_to_kaggle=False):
    """Run pipeline for a single competition."""
    print(f"\n{'='*60}")
    print(f"  KaggleSlayer - Running pipeline for: {competition_name}")
    print(f"{'='*60}\n")

    coordinator = PipelineCoordinator(competition_name, Path(data_path))
    results = coordinator.run(submit_to_kaggle=submit_to_kaggle)

    print(f"\n{'='*60}")
    print(f"  Pipeline Complete!")
    print(f"  Best Model: {results.get('best_model', 'N/A')}")
    print(f"  CV Score: {results.get('final_score', 0):.4f}")
    print(f"  Submission saved to: {data_path}/submission.csv")
    print(f"{'='*60}\n")

    return results


def run_all_competitions(submit_to_kaggle=False):
    """Run pipeline for all downloaded competitions."""
    competitions = get_all_competitions()

    if not competitions:
        print("\n[!] No competitions found in competition_data/")
        print("    Run: python download_all_competitions.py")
        return

    print(f"\n{'='*70}")
    print(f"  KaggleSlayer - Running pipeline for ALL competitions")
    print(f"{'='*70}")
    print(f"\nFound {len(competitions)} competitions:")
    for i, comp in enumerate(competitions, 1):
        print(f"  {i}. {comp}")

    # Ask for confirmation
    response = input(f"\nRun pipeline for all {len(competitions)} competitions? (y/N): ").strip().lower()
    if response != 'y':
        print("\n[!] Cancelled")
        return

    # Run pipeline for each competition
    all_results = []
    successful = []
    failed = []

    print(f"\n{'='*70}")
    print(f"  Starting batch pipeline execution...")
    print(f"{'='*70}\n")

    for i, comp_name in enumerate(competitions, 1):
        print(f"\n{'#'*70}")
        print(f"  [{i}/{len(competitions)}] Processing: {comp_name}")
        print(f"{'#'*70}")

        try:
            data_path = Path("competition_data") / comp_name
            results = run_single_competition(comp_name, data_path, submit_to_kaggle)

            all_results.append({
                'competition': comp_name,
                'best_model': results.get('best_model', 'N/A'),
                'cv_score': results.get('final_score', 0),
                'status': 'success'
            })
            successful.append(comp_name)

        except KeyboardInterrupt:
            print(f"\n\n[!] Batch execution interrupted by user")
            failed.append(comp_name)
            break
        except Exception as e:
            print(f"\n[X] Error processing {comp_name}: {e}")
            all_results.append({
                'competition': comp_name,
                'status': 'failed',
                'error': str(e)
            })
            failed.append(comp_name)

    # Final summary
    print(f"\n{'='*70}")
    print(f"  BATCH EXECUTION SUMMARY")
    print(f"{'='*70}")
    print(f"\nTotal competitions: {len(competitions)}")
    print(f"Successful: {len(successful)}")
    print(f"Failed: {len(failed)}")

    if successful:
        print(f"\n[OK] Successful ({len(successful)}):")
        print(f"\n{'Competition':<40} {'Model':<20} {'CV Score':<10}")
        print(f"{'-'*40} {'-'*20} {'-'*10}")
        for result in all_results:
            if result['status'] == 'success':
                comp = result['competition']
                model = result['best_model']
                score = result['cv_score']
                print(f"{comp:<40} {model:<20} {score:<10.4f}")

    if failed:
        print(f"\n[X] Failed ({len(failed)}):")
        for comp in failed:
            print(f"  - {comp}")

    print(f"\n{'='*70}")
    print(f"  All pipelines complete!")
    print(f"{'='*70}\n")


def main():
    parser = argparse.ArgumentParser(
        description="KaggleSlayer - AutoML for Kaggle Competitions",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run single competition
  python kaggle_slayer.py titanic --data-path competition_data/titanic

  # Run all downloaded competitions
  python kaggle_slayer.py --all

  # Run all and submit to Kaggle
  python kaggle_slayer.py --all --submit
        """
    )

    parser.add_argument("competition", nargs="?", help="Competition name (required unless using --all)")
    parser.add_argument("--data-path", help="Path to competition data directory")
    parser.add_argument("--submit", action="store_true", help="Submit to Kaggle after creating submission file")
    parser.add_argument("--all", action="store_true", help="Run pipeline for all downloaded competitions")

    args = parser.parse_args()

    # Handle --all flag
    if args.all:
        run_all_competitions(submit_to_kaggle=args.submit)
        return

    # Single competition mode - require competition name and data path
    if not args.competition:
        parser.print_help()
        print("\n[X] Error: Competition name required (or use --all)")
        return

    if not args.data_path:
        parser.print_help()
        print("\n[X] Error: --data-path required for single competition mode")
        return

    # Run single competition
    run_single_competition(args.competition, args.data_path, args.submit)


if __name__ == "__main__":
    main()
