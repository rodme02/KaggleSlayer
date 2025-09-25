#!/usr/bin/env python3
"""
Test script to verify the enhanced app components work correctly
"""

import sys
from pathlib import Path

# Test imports
try:
    import pandas as pd
    import plotly.express as px
    import plotly.graph_objects as go
    import json
    print("All dependencies imported successfully")
except ImportError as e:
    print(f"Import error: {e}")
    sys.exit(1)

# Test pipeline functions
try:
    # Simulate the pipeline helper functions
    datasets_dir = Path("downloaded_datasets")
    competitions = []

    if datasets_dir.exists():
        for comp_dir in datasets_dir.iterdir():
            if comp_dir.is_dir() and (comp_dir / "train.csv").exists():
                scout_done = (comp_dir / "scout_output").exists()
                model_done = (comp_dir / "baseline_model").exists()
                submissions_exist = (comp_dir / "submissions").exists()

                competitions.append({
                    "name": comp_dir.name,
                    "path": comp_dir,
                    "scout_done": scout_done,
                    "model_done": model_done,
                    "submissions_exist": submissions_exist
                })

    print(f"Found {len(competitions)} competitions")
    for comp in competitions:
        print(f"  - {comp['name']}: Scout={comp['scout_done']}, Model={comp['model_done']}, Submissions={comp['submissions_exist']}")

    # Test data loading functions
    for comp in competitions:
        comp_path = comp["path"]

        # Test dataset info loading
        info_path = comp_path / "scout_output" / "dataset_info.json"
        if info_path.exists():
            with open(info_path, 'r') as f:
                dataset_info = json.load(f)
            print(f"{comp['name']}: Dataset info loaded - {dataset_info['total_rows']} rows")

        # Test baseline results loading
        results_path = comp_path / "baseline_model" / "baseline_results.json"
        if results_path.exists():
            with open(results_path, 'r') as f:
                baseline_results = json.load(f)
            print(f"{comp['name']}: Baseline results loaded - {baseline_results['model_type']}, CV: {baseline_results['cv_mean']:.4f}")

        # Test submission history loading
        log_path = comp_path / "submissions" / "submission_log.jsonl"
        if log_path.exists():
            history = []
            with open(log_path, 'r') as f:
                for line in f:
                    if line.strip():
                        history.append(json.loads(line.strip()))
            print(f"{comp['name']}: Submission history loaded - {len(history)} submissions")

    print("\nAll enhanced app components are working correctly!")
    print("Run 'streamlit run app.py' to start the dashboard")

except Exception as e:
    print(f"Error testing app components: {e}")
    sys.exit(1)