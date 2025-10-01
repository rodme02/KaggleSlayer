"""
Kaggle API utilities for competition submission and management.
"""

import os
import subprocess
from pathlib import Path
from typing import Optional, Dict, Any, List
import pandas as pd
import json


class KaggleAPIClient:
    """Client for interacting with Kaggle API."""

    def __init__(self):
        self.api_configured = self._check_api_configuration()

    def _check_api_configuration(self) -> bool:
        """Check if Kaggle API is properly configured."""
        try:
            # Check for kaggle credentials
            kaggle_config_path = Path.home() / ".kaggle" / "kaggle.json"

            if not kaggle_config_path.exists():
                print("Warning: Kaggle API credentials not found at ~/.kaggle/kaggle.json")
                print("Please download your kaggle.json from https://www.kaggle.com/settings")
                return False

            # Test API connection
            result = subprocess.run(
                ["kaggle", "--version"],
                capture_output=True,
                text=True,
                timeout=10
            )

            if result.returncode == 0:
                print(f"Kaggle API configured successfully: {result.stdout.strip()}")
                return True
            else:
                print(f"Kaggle API error: {result.stderr}")
                return False

        except subprocess.TimeoutExpired:
            print("Kaggle API timeout - check internet connection")
            return False
        except FileNotFoundError:
            print("Kaggle CLI not installed. Install with: pip install kaggle")
            return False
        except Exception as e:
            print(f"Kaggle API configuration error: {e}")
            return False

    def get_competition_info(self, competition_name: str) -> Optional[Dict[str, Any]]:
        """Get information about a competition."""
        if not self.api_configured:
            return None

        try:
            result = subprocess.run(
                ["kaggle", "competitions", "show", competition_name, "--json"],
                capture_output=True,
                text=True,
                timeout=30
            )

            if result.returncode == 0:
                return json.loads(result.stdout)
            else:
                print(f"Failed to get competition info: {result.stderr}")
                return None

        except Exception as e:
            print(f"Error getting competition info: {e}")
            return None

    def list_competition_files(self, competition_name: str) -> Optional[List[str]]:
        """List files available for a competition."""
        if not self.api_configured:
            return None

        try:
            result = subprocess.run(
                ["kaggle", "competitions", "files", competition_name],
                capture_output=True,
                text=True,
                timeout=30
            )

            if result.returncode == 0:
                lines = result.stdout.strip().split('\n')[1:]  # Skip header
                files = [line.split()[0] for line in lines if line.strip()]
                return files
            else:
                print(f"Failed to list competition files: {result.stderr}")
                return None

        except Exception as e:
            print(f"Error listing competition files: {e}")
            return None

    def download_competition_data(self, competition_name: str, download_path: Path) -> bool:
        """Download competition data."""
        if not self.api_configured:
            return False

        try:
            download_path.mkdir(parents=True, exist_ok=True)

            result = subprocess.run(
                ["kaggle", "competitions", "download", competition_name, "-p", str(download_path)],
                capture_output=True,
                text=True,
                timeout=300  # 5 minutes
            )

            if result.returncode == 0:
                print(f"Successfully downloaded competition data to {download_path}")

                # Unzip files if they exist
                for zip_file in download_path.glob("*.zip"):
                    import zipfile
                    with zipfile.ZipFile(zip_file, 'r') as zip_ref:
                        zip_ref.extractall(download_path)
                    zip_file.unlink()  # Remove zip file after extraction
                    print(f"Extracted {zip_file.name}")

                return True
            else:
                print(f"Failed to download competition data: {result.stderr}")
                return False

        except Exception as e:
            print(f"Error downloading competition data: {e}")
            return False

    def submit_to_competition(self,
                            competition_name: str,
                            submission_file: Path,
                            message: str = "KaggleSlayer submission") -> bool:
        """Submit predictions to a Kaggle competition."""
        if not self.api_configured:
            print("Kaggle API not configured. Cannot submit.")
            return False

        if not submission_file.exists():
            print(f"Submission file not found: {submission_file}")
            return False

        try:
            print(f"Submitting to {competition_name}...")
            print(f"File: {submission_file}")
            print(f"Message: {message}")

            result = subprocess.run([
                "kaggle", "competitions", "submit",
                competition_name,
                "-f", str(submission_file),
                "-m", message
            ], capture_output=True, text=True, timeout=60)

            if result.returncode == 0:
                print("Submission successful!")
                print(result.stdout)
                return True
            else:
                print(f"Submission failed: {result.stderr}")
                return False

        except subprocess.TimeoutExpired:
            print("Submission timeout - check internet connection")
            return False
        except Exception as e:
            print(f"Submission error: {e}")
            return False

    def get_submission_status(self, competition_name: str) -> Optional[List[Dict[str, Any]]]:
        """Get status of recent submissions."""
        if not self.api_configured:
            return None

        try:
            result = subprocess.run([
                "kaggle", "competitions", "submissions", competition_name
            ], capture_output=True, text=True, timeout=30)

            if result.returncode == 0:
                # Parse the output (it's in table format)
                lines = result.stdout.strip().split('\n')
                if len(lines) > 1:
                    print("Recent submissions:")
                    for line in lines[:6]:  # Show header + top 5
                        print(line)
                return []  # Could parse this into structured data if needed
            else:
                print(f"Failed to get submission status: {result.stderr}")
                return None

        except Exception as e:
            print(f"Error getting submission status: {e}")
            return None

    def validate_submission_format(self,
                                 submission_file: Path,
                                 competition_name: str) -> bool:
        """Validate submission file format."""
        try:
            df = pd.read_csv(submission_file)

            # Basic validation
            if df.empty:
                print("ERROR: Submission file is empty")
                return False

            # Generic validation - just check basic format requirements
            if df.shape[0] == 0:
                print("ERROR: Submission has no data rows")
                return False

            if df.shape[1] < 2:
                print("ERROR: Submission must have at least 2 columns (ID + prediction)")
                return False

            print(f"SUCCESS: Submission format validation passed")

            print(f"   Shape: {df.shape}")
            print(f"   Columns: {list(df.columns)}")

            return True

        except Exception as e:
            print(f"ERROR: Submission validation failed: {e}")
            return False


def setup_kaggle_credentials():
    """Helper function to set up Kaggle credentials."""
    print("\nKaggle API Setup Instructions:")
    print("1. Go to https://www.kaggle.com/settings")
    print("2. Click 'Create New API Token' to download kaggle.json")
    print("3. Place kaggle.json in ~/.kaggle/ directory")
    print("4. Set permissions: chmod 600 ~/.kaggle/kaggle.json")
    print("5. Install kaggle CLI: pip install kaggle")
    print("\nAlternatively, set environment variables:")
    print("export KAGGLE_USERNAME=your_username")
    print("export KAGGLE_KEY=your_api_key")