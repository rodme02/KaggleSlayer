"""
Download and organize Kaggle competition data.

This script:
1. Downloads a Kaggle competition zip file
2. Extracts it to a temporary location
3. Checks for train.csv and test.csv
4. If found, moves them to competition_data/{competition_name}/raw/
5. If not found, deletes everything and notifies user
"""

import argparse
import zipfile
import shutil
from pathlib import Path
import subprocess
import sys


def download_competition(competition_name: str, force: bool = False):
    """
    Download and organize Kaggle competition data.

    Args:
        competition_name: Name of the Kaggle competition
        force: If True, re-download even if data already exists
    """
    print(f"\n{'='*70}")
    print(f"  Downloading Kaggle Competition: {competition_name}")
    print(f"{'='*70}\n")

    # Setup paths
    competition_dir = Path("competition_data") / competition_name
    raw_dir = competition_dir / "raw"
    temp_dir = Path(".temp_download") / competition_name

    # Check if data already exists
    if raw_dir.exists() and not force:
        print(f"[!] Competition data already exists at: {raw_dir}")
        print(f"    Use --force to re-download")
        return False

    # Create temporary directory
    temp_dir.mkdir(parents=True, exist_ok=True)

    try:
        # Step 1: Download competition files using Kaggle API
        print(f"[>>] Downloading competition files...")
        result = subprocess.run(
            ["kaggle", "competitions", "download", "-c", competition_name, "-p", str(temp_dir)],
            capture_output=True,
            text=True
        )

        if result.returncode != 0:
            print(f"\n[X] Error downloading competition:")
            print(result.stderr)

            # Check for common errors
            if "403" in result.stderr or "Forbidden" in result.stderr:
                print(f"\nTip: Make sure you've accepted the competition rules at:")
                print(f"     https://www.kaggle.com/competitions/{competition_name}/rules")
            elif "404" in result.stderr or "Not Found" in result.stderr:
                print(f"\nTip: Competition '{competition_name}' not found.")
                print(f"     Check the competition name at kaggle.com")

            shutil.rmtree(temp_dir)
            return False

        print(f"[OK] Download complete")

        # Step 2: Extract zip files
        print(f"\n[>>] Extracting files...")
        zip_files = list(temp_dir.glob("*.zip"))

        if not zip_files:
            print(f"[X] No zip files found in download")
            shutil.rmtree(temp_dir)
            return False

        for zip_file in zip_files:
            print(f"     Extracting {zip_file.name}...")
            with zipfile.ZipFile(zip_file, 'r') as zip_ref:
                zip_ref.extractall(temp_dir)
            # Remove the zip file after extraction
            zip_file.unlink()

        print(f"[OK] Extraction complete")

        # Step 3: Check for train.csv and test.csv
        print(f"\n[>>] Checking for required files (train.csv, test.csv)...")

        train_csv = temp_dir / "train.csv"
        test_csv = temp_dir / "test.csv"

        if not train_csv.exists():
            print(f"[X] train.csv not found in competition files")
            print(f"\nFiles found:")
            for file in sorted(temp_dir.rglob("*")):
                if file.is_file():
                    print(f"  - {file.relative_to(temp_dir)}")

            print(f"\n[!] This competition doesn't have the standard train.csv/test.csv format.")
            print(f"    Cleaning up downloaded files...")
            shutil.rmtree(temp_dir)
            return False

        if not test_csv.exists():
            print(f"[!] test.csv not found (this is optional)")

        # Step 4: Check for extra CSV files (reject if found)
        print(f"\n[>>] Checking for extra CSV files...")

        all_csv_files = list(temp_dir.glob("*.csv"))
        csv_names = [f.name for f in all_csv_files]

        # Allowed files: train.csv, test.csv, and submission-related files
        allowed_files = {"train.csv", "test.csv"}
        submission_patterns = ["sample_submission", "submission", "samplesubmission"]

        # Check if there are any extra CSV files
        extra_csvs = []
        submission_file = None

        for csv_name in csv_names:
            if csv_name in allowed_files:
                continue
            # Check if it's a submission file
            name_lower = csv_name.lower()
            if any(pattern in name_lower for pattern in submission_patterns):
                submission_file = csv_name
            else:
                extra_csvs.append(csv_name)

        if extra_csvs:
            print(f"[X] Found extra CSV files beyond train/test/submission:")
            for csv_name in extra_csvs:
                print(f"  - {csv_name}")
            print(f"\n[!] This competition has additional data files (not standard tabular format).")
            print(f"    Cleaning up downloaded files...")
            shutil.rmtree(temp_dir)
            return False

        # Step 5: Move files to proper location
        print(f"\n[>>] Organizing files...")

        # Create competition directory structure
        raw_dir.mkdir(parents=True, exist_ok=True)

        # Move train.csv
        shutil.move(str(train_csv), str(raw_dir / "train.csv"))
        print(f"     Moved train.csv to {raw_dir}/")

        # Move test.csv if it exists
        if test_csv.exists():
            shutil.move(str(test_csv), str(raw_dir / "test.csv"))
            print(f"     Moved test.csv to {raw_dir}/")

        # Move submission file if it exists
        if submission_file:
            submission_path = temp_dir / submission_file
            if submission_path.exists():
                shutil.move(str(submission_path), str(raw_dir / submission_file))
                print(f"     Moved {submission_file} to {raw_dir}/")

        # Step 6: Clean up temporary directory
        print(f"\n[>>] Cleaning up temporary files...")
        shutil.rmtree(temp_dir)

        # Success message
        print(f"\n{'='*70}")
        print(f"  SUCCESS! Competition data ready")
        print(f"{'='*70}")
        print(f"\nData location: {raw_dir}")
        print(f"\nReady to run:")
        print(f"  python kaggle_slayer.py {competition_name} --data-path {competition_dir}")
        print()

        return True

    except Exception as e:
        print(f"\n❌ Error: {e}")
        # Clean up on error
        if temp_dir.exists():
            shutil.rmtree(temp_dir)
        return False


def list_competitions():
    """List user's Kaggle competitions."""
    print(f"\nFetching your Kaggle competitions...\n")

    result = subprocess.run(
        ["kaggle", "competitions", "list", "--csv"],
        capture_output=True,
        text=True
    )

    if result.returncode != 0:
        print(f"[X] Error fetching competitions:")
        print(result.stderr)
        return

    # Parse CSV output
    lines = result.stdout.strip().split('\n')
    if len(lines) <= 1:
        print("No competitions found")
        return

    # Print header
    print(f"{'Competition':<40} {'Deadline':<20} {'Category':<15}")
    print(f"{'-'*40} {'-'*20} {'-'*15}")

    # Print competitions (skip header)
    for line in lines[1:11]:  # Show first 10
        parts = line.split(',')
        if len(parts) >= 3:
            comp_name = parts[0]
            deadline = parts[2]
            category = parts[3] if len(parts) > 3 else "N/A"
            print(f"{comp_name:<40} {deadline:<20} {category:<15}")

    if len(lines) > 11:
        print(f"\n... and {len(lines) - 11} more")

    print(f"\nTo download a competition, run:")
    print(f"  python download_competition.py <competition-name>")
    print()


def main():
    parser = argparse.ArgumentParser(
        description="Download and organize Kaggle competition data",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Download a specific competition
  python download_competition.py titanic

  # Force re-download even if data exists
  python download_competition.py titanic --force

  # List your competitions
  python download_competition.py --list

Note: Make sure you've:
  1. Installed Kaggle API: pip install kaggle
  2. Setup API credentials: https://www.kaggle.com/docs/api
  3. Accepted competition rules on kaggle.com
        """
    )

    parser.add_argument(
        "competition",
        nargs="?",
        help="Competition name (e.g., 'titanic', 'playground-series-s5e9')"
    )

    parser.add_argument(
        "--force",
        action="store_true",
        help="Force re-download even if data already exists"
    )

    parser.add_argument(
        "--list",
        action="store_true",
        help="List your Kaggle competitions"
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

    # Handle list command
    if args.list:
        list_competitions()
        return

    # Require competition name
    if not args.competition:
        parser.print_help()
        print("\n[X] Error: Competition name required (or use --list)")
        sys.exit(1)

    # Download competition
    success = download_competition(args.competition, args.force)
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
