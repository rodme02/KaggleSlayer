#!/usr/bin/env python3
"""
Simple test script that bypasses YAML configuration.
"""

import sys
from pathlib import Path

# Add current directory to path
sys.path.insert(0, str(Path(__file__).parent))

def main():
    # Simple hardcoded config instead of YAML
    class SimpleConfig:
        def get(self, key, default=None):
            config_map = {
                "pipeline.cv_folds": 5,
                "pipeline.max_features_to_create": 25,
                "pipeline.polynomial_degree": 2,
                "data.drop_missing_threshold": 0.8,
                "data.correlation_threshold": 0.95,
                "data.variance_threshold": 0.01,
                "pipeline.cv_random_state": 42,
                "pipeline.optuna_trials": 20,
                "pipeline.optuna_timeout": 300
            }
            return config_map.get(key, default)

    # Test just the data scout
    from agents.data_scout import DataScoutAgent

    config = SimpleConfig()
    competition_path = Path("competition_data/titanic")

    print(f"Testing Data Scout for: titanic")
    print(f"Competition path: {competition_path}")

    try:
        # Create agent with simple config
        agent = DataScoutAgent("titanic", competition_path, config)

        # Run data scouting
        results = agent.run()

        print("\n" + "="*50)
        print("DATA SCOUTING COMPLETED!")
        print("="*50)
        print(f"Dataset rows: {results['dataset_info']['total_rows']}")
        print(f"Dataset columns: {results['dataset_info']['total_columns']}")
        print(f"Target column: {results['dataset_info']['target_column']}")
        print(f"Problem type: {results['recommendations']['problem_type']}")

        return 0

    except Exception as e:
        print(f"Error: {e}")
        return 1

if __name__ == "__main__":
    sys.exit(main())