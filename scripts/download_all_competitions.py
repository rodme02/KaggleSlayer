"""
Batch download all Kaggle competitions you've entered, plus tabular competitions.

This script:
1. Lists all competitions you've entered
2. Searches for competitions by keywords: playground, tabular, classification, regression, structured
3. Combines and downloads each one
4. Checks for train.csv and test.csv (tabular data)
5. If tabular: Saves to competition_data/{name}/raw/
6. If not tabular or has extra CSVs: Deletes everything
"""

import subprocess
import sys
from pathlib import Path
from download_competition import download_competition


def search_competitions(search_term):
    """Search for competitions by keyword."""
    result = subprocess.run(
        ["kaggle", "competitions", "list", "--search", search_term, "--csv"],
        capture_output=True,
        text=True
    )

    if result.returncode != 0:
        return []

    # Parse CSV output
    lines = result.stdout.strip().split('\n')
    if len(lines) <= 1:
        return []

    # Extract competition refs (first column)
    competitions = []
    for line in lines[1:]:  # Skip header
        parts = line.split(',')
        if len(parts) >= 1:
            comp_ref = parts[0].strip()
            if not comp_ref:
                continue
            if '/competitions/' in comp_ref:
                comp_name = comp_ref.split('/competitions/')[-1]
                if comp_name:
                    competitions.append(comp_name)
            else:
                if comp_ref:
                    competitions.append(comp_ref)

    return competitions


def get_competitions_by_category(category):
    """Get competitions by category (gettingStarted, featured, playground, etc.)."""
    result = subprocess.run(
        ["kaggle", "competitions", "list", "--category", category, "--csv"],
        capture_output=True,
        text=True
    )

    if result.returncode != 0:
        return []

    # Parse CSV output
    lines = result.stdout.strip().split('\n')
    if len(lines) <= 1:
        return []

    # Extract competition refs (first column)
    competitions = []
    for line in lines[1:]:  # Skip header
        parts = line.split(',')
        if len(parts) >= 1:
            comp_ref = parts[0].strip()
            if not comp_ref:
                continue
            if '/competitions/' in comp_ref:
                comp_name = comp_ref.split('/competitions/')[-1]
                if comp_name:
                    competitions.append(comp_name)
            else:
                if comp_ref:
                    competitions.append(comp_ref)

    return competitions


def check_competition_files(competition_name):
    """Check if a competition has train.csv and test.csv before downloading.

    Returns:
        tuple: (status: str, extra_csvs: list)
        status can be: 'ok', 'extra_csvs', 'no_train', 'cannot_access'
    """
    result = subprocess.run(
        ["kaggle", "competitions", "files", competition_name, "--csv"],
        capture_output=True,
        text=True
    )

    if result.returncode != 0:
        # Can't access files list (probably not entered/accepted rules yet)
        # Don't skip - let download attempt handle it
        return 'cannot_access', []

    # Parse CSV output
    lines = result.stdout.strip().split('\n')
    if len(lines) <= 1:
        return 'cannot_access', []

    # Extract file names (first column)
    files = []
    for line in lines[1:]:  # Skip header
        parts = line.split(',')
        if len(parts) >= 1:
            filename = parts[0].strip()
            if filename:
                files.append(filename)

    # Check for required files
    has_train = any('train.csv' in f for f in files)

    if not has_train:
        return 'no_train', []

    # Check for extra CSV files
    allowed_patterns = ['train.csv', 'test.csv', 'sample_submission', 'submission', 'samplesubmission']
    extra_csvs = []

    for f in files:
        if f.endswith('.csv'):
            # Check if it's an allowed file
            is_allowed = any(pattern in f.lower() for pattern in allowed_patterns)
            if not is_allowed:
                extra_csvs.append(f)

    if extra_csvs:
        return 'extra_csvs', extra_csvs

    return 'ok', []


def get_known_good_competitions():
    """Return a list of known popular tabular competitions."""
    return [
        # Classic getting-started competitions
        "titanic",
        "digit-recognizer",
        "house-prices-advanced-regression-techniques",
        "spaceship-titanic",
        "home-data-for-ml-course",

        # Popular tabular competitions
        "santander-customer-satisfaction",
        "santander-value-prediction-challenge",
        "home-credit-default-risk",
        "elo-merchant-category-recommendation",
        "ieee-fraud-detection",
        "microsoft-malware-prediction",
        "google-analytics-customer-revenue-prediction",
        "walmart-recruiting-trip-type-classification",
        "prudential-life-insurance-assessment",
        "allstate-claims-severity",
        "porto-seguro-safe-driver-prediction",
        "two-sigma-financial-modeling",
        "zillow-prize-1",
        "rossmann-store-sales",
        "competitive-data-science-predict-future-sales",

        # Playground series (recent ones)
        "playground-series-s5e1",
        "playground-series-s5e2",
        "playground-series-s5e3",
        "playground-series-s5e4",
        "playground-series-s5e5",
        "playground-series-s5e6",
        "playground-series-s5e7",
        "playground-series-s5e8",
        "playground-series-s5e9",
        "playground-series-s5e10",
        "playground-series-s4e1",
        "playground-series-s4e2",
        "playground-series-s4e3",
        "playground-series-s4e4",
        "playground-series-s4e5",
        "playground-series-s4e6",
        "playground-series-s4e7",
        "playground-series-s4e8",
        "playground-series-s4e9",
        "playground-series-s4e10",
        "playground-series-s4e11",
        "playground-series-s4e12",
    ]


def get_entered_competitions():
    """Get list of competitions the user has entered."""
    print("\n" + "="*70)
    print("  Fetching competitions you've entered...")
    print("="*70 + "\n")

    # Use Kaggle API to list competitions with --csv flag
    # The API shows entered competitions with different filters
    result = subprocess.run(
        ["kaggle", "competitions", "list", "--csv"],
        capture_output=True,
        text=True
    )

    if result.returncode != 0:
        print(f"[X] Error fetching competitions:")
        print(result.stderr)
        return []

    # Parse CSV output
    lines = result.stdout.strip().split('\n')
    if len(lines) <= 1:
        print("No competitions found")
        return []

    # Extract competition refs (first column)
    competitions = []
    for line in lines[1:]:  # Skip header
        parts = line.split(',')
        if len(parts) >= 1:
            # Extract competition name from URL
            comp_ref = parts[0].strip()
            if not comp_ref:  # Skip empty strings
                continue
            if '/competitions/' in comp_ref:
                comp_name = comp_ref.split('/competitions/')[-1]
                if comp_name:  # Only add if not empty
                    competitions.append(comp_name)
            else:
                # Sometimes it's just the name
                if comp_ref:  # Only add if not empty
                    competitions.append(comp_ref)

    return competitions


def batch_download_competitions(dry_run=False):
    """
    Download all entered competitions and filter for tabular data.
    Also searches for competitions using multiple keywords likely to find tabular data.

    Args:
        dry_run: If True, just show what would be downloaded
    """
    # Get list of entered competitions
    entered_competitions = get_entered_competitions()

    # Get known-good tabular competitions
    print("\n" + "="*70)
    print("  Adding known-good tabular competitions...")
    print("="*70 + "\n")

    known_good = get_known_good_competitions()
    print(f"[>>] Added {len(known_good)} known-good competitions")

    # Get competitions by category (gettingStarted, playground, featured)
    print("\n" + "="*70)
    print("  Fetching competitions by category...")
    print("="*70 + "\n")

    categories = ["gettingStarted", "playground", "featured"]
    category_competitions = []
    for category in categories:
        print(f"[>>] Category '{category}'...", end=" ")
        results = get_competitions_by_category(category)
        print(f"found {len(results)} competitions")
        category_competitions.extend(results)

    # Remove duplicates from category results
    category_competitions = list(set(category_competitions))

    # Search for competitions using multiple keywords
    search_terms = [
        "tabular", "classification", "regression", "structured",
        "prediction", "binary", "multiclass", "dataset", "features", "ML"
    ]

    print("\n" + "="*70)
    print("  Searching for tabular competitions by keyword...")
    print("="*70 + "\n")

    searched_competitions = []
    for term in search_terms:
        print(f"[>>] Searching for '{term}'...", end=" ")
        results = search_competitions(term)
        print(f"found {len(results)} competitions")
        searched_competitions.extend(results)

    # Remove duplicates from search results
    searched_competitions = list(set(searched_competitions))

    # Combine all sources and remove duplicates
    all_competitions = list(set(
        entered_competitions +
        known_good +
        category_competitions +
        searched_competitions
    ))

    # Sort for consistent ordering
    all_competitions.sort()

    if not all_competitions:
        print("\n[!] No competitions found to download")
        return

    print(f"\n[>>] Total unique competitions: {len(all_competitions)}")
    print(f"    - Entered: {len(entered_competitions)}")
    print(f"    - Known-good: {len(known_good)}")
    print(f"    - From categories: {len(category_competitions)}")
    print(f"    - Found by search: {len(searched_competitions)}")
    print(f"    - Combined (unique): {len(all_competitions)}")

    print("\nCompetitions to download:")
    for i, comp in enumerate(all_competitions, 1):
        # Mark if it's a search-found competition (not in entered)
        marker = ""
        if comp not in entered_competitions:
            if "playground" in comp.lower():
                marker = " [playground]"
            else:
                marker = " [searched]"
        print(f"  {i}. {comp}{marker}")

    # Use all_competitions for the rest of the function
    competitions = all_competitions

    if dry_run:
        print("\n[!] Dry run mode - no downloads performed")
        return

    # Ask for confirmation
    print("\n" + "="*70)
    response = input("Download all competitions? (y/N): ").strip().lower()
    if response != 'y':
        print("\n[!] Download cancelled")
        return

    # Download each competition
    successful = []
    already_exists = []
    failed = []
    non_tabular = []
    skipped_precheck = []

    print("\n" + "="*70)
    print("  Starting batch download...")
    print("="*70)

    for i, comp_name in enumerate(competitions, 1):
        print(f"\n[{i}/{len(competitions)}] Processing: {comp_name}")
        print("-" * 70)

        # Check if already exists
        comp_raw_dir = Path("competition_data") / comp_name / "raw"
        if comp_raw_dir.exists():
            already_exists.append(comp_name)
            print(f"[OK] {comp_name} - Already downloaded (skipped)")
            continue

        # Pre-check file structure before downloading (if accessible)
        print(f"    Checking file structure...", end=" ")
        status, extra_csvs = check_competition_files(comp_name)

        if status == 'no_train':
            skipped_precheck.append(comp_name)
            print(f"SKIP (no train.csv found)")
            continue

        if status == 'extra_csvs':
            non_tabular.append(comp_name)
            print(f"SKIP (extra CSVs: {', '.join(extra_csvs[:3])}{'...' if len(extra_csvs) > 3 else ''})")
            continue

        if status == 'cannot_access':
            print(f"UNKNOWN (will attempt download)")
        else:
            print(f"OK")

        try:
            # Download and organize competition
            success = download_competition(comp_name, force=False)

            if success:
                successful.append(comp_name)
                print(f"[OK] {comp_name} - Tabular data saved")
            else:
                # Check if it failed due to non-tabular format
                comp_dir = Path("competition_data") / comp_name
                if not comp_dir.exists():
                    # Was deleted because non-tabular
                    non_tabular.append(comp_name)
                    print(f"[!] {comp_name} - Not tabular (deleted)")
                else:
                    failed.append(comp_name)
                    print(f"[X] {comp_name} - Download failed")

        except KeyboardInterrupt:
            print(f"\n\n[!] Download interrupted by user")
            break
        except Exception as e:
            failed.append(comp_name)
            print(f"[X] {comp_name} - Error: {e}")

    # Summary
    print("\n" + "="*70)
    print("  BATCH DOWNLOAD SUMMARY")
    print("="*70)
    print(f"\nTotal competitions: {len(competitions)}")
    print(f"Successfully downloaded (new): {len(successful)}")
    print(f"Already existed (tabular): {len(already_exists)}")
    print(f"Skipped (pre-check): {len(skipped_precheck)}")
    print(f"Non-tabular (extra CSVs): {len(non_tabular)}")
    print(f"Failed: {len(failed)}")

    total_tabular = len(successful) + len(already_exists)

    if successful:
        print(f"\n[OK] Successfully downloaded ({len(successful)}):")
        for comp in successful:
            print(f"  - {comp}")

    if already_exists:
        print(f"\n[OK] Already downloaded ({len(already_exists)}):")
        for comp in already_exists:
            print(f"  - {comp}")

    if skipped_precheck:
        print(f"\n[!] Skipped (pre-check failed) ({len(skipped_precheck)}):")
        for comp in skipped_precheck[:10]:  # Show first 10
            print(f"  - {comp}")
        if len(skipped_precheck) > 10:
            print(f"  ... and {len(skipped_precheck) - 10} more")

    if non_tabular:
        print(f"\n[!] Non-tabular (extra CSVs) ({len(non_tabular)}):")
        for comp in non_tabular[:10]:  # Show first 10
            print(f"  - {comp}")
        if len(non_tabular) > 10:
            print(f"  ... and {len(non_tabular) - 10} more")

    if failed:
        print(f"\n[X] Failed downloads ({len(failed)}):")
        for comp in failed:
            print(f"  - {comp}")

    print("\n" + "="*70)
    print(f"Ready to run KaggleSlayer on {total_tabular} tabular competitions!")
    print("="*70 + "\n")


def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="Batch download all Kaggle competitions you've entered",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # See what would be downloaded (dry run)
  python download_all_competitions.py --dry-run

  # Download all entered competitions
  python download_all_competitions.py

Note: This script only downloads competitions with tabular data (train.csv/test.csv).
Non-tabular competitions are automatically deleted after download.
        """
    )

    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be downloaded without downloading"
    )

    args = parser.parse_args()

    # Check if Kaggle API is available
    try:
        result = subprocess.run(
            ["kaggle", "--version"],
            capture_output=True,
            text=True
        )
        if result.returncode != 0:
            print("[X] Kaggle API not found. Install with: pip install kaggle")
            sys.exit(1)
    except FileNotFoundError:
        print("[X] Kaggle API not found. Install with: pip install kaggle")
        sys.exit(1)

    # Run batch download
    batch_download_competitions(dry_run=args.dry_run)


if __name__ == "__main__":
    main()
