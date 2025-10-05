"""
Configuration management utilities.
"""

import yaml
from pathlib import Path
from typing import Dict, Any, Optional
import os


class ConfigManager:
    """Manages configuration loading and access."""

    def __init__(self, config_path: Optional[Path] = None):
        if config_path is None:
            # Look for config.yaml in the current working directory first
            config_path = Path.cwd() / "config.yaml"
            if not config_path.exists():
                # Fallback to relative path from this file
                config_path = Path(__file__).parent.parent / "config.yaml"

        self.config_path = Path(config_path)
        self._config = None
        self.load_config()

    def load_config(self) -> Dict[str, Any]:
        """Load configuration from YAML file."""
        try:
            with open(self.config_path, 'r') as f:
                self._config = yaml.safe_load(f)
            return self._config
        except FileNotFoundError:
            print(f"Config file not found: {self.config_path}")
            self._config = self._get_default_config()
            return self._config
        except Exception as e:
            print(f"Error loading config: {e}")
            self._config = self._get_default_config()
            return self._config

    def get(self, key_path: str, default: Any = None) -> Any:
        """Get configuration value using dot notation."""
        if self._config is None:
            self.load_config()

        keys = key_path.split('.')
        value = self._config

        try:
            for key in keys:
                value = value[key]
            return value
        except (KeyError, TypeError):
            return default

    def get_pipeline_config(self) -> Dict[str, Any]:
        """Get pipeline configuration."""
        return self.get("pipeline", {})

    def get_model_config(self, model_name: str) -> Dict[str, Any]:
        """Get model-specific configuration."""
        return self.get(f"models.{model_name}", {})

    def _get_default_config(self) -> Dict[str, Any]:
        """Get default configuration if file is not available."""
        return {
            "pipeline": {
                "cv_folds": 5,
                "cv_random_state": 42,
                "optuna_trials": 20,
                "max_features_to_create": 25
            },
            "data": {
                "drop_missing_threshold": 0.9,
                "correlation_threshold": 0.95
            },
            "logging": {
                "level": "INFO",
                "format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
            }
        }

    def update_config(self, key_path: str, value: Any) -> None:
        """Update configuration value."""
        if self._config is None:
            self.load_config()

        keys = key_path.split('.')
        config = self._config

        # Navigate to the parent of the target key
        for key in keys[:-1]:
            if key not in config:
                config[key] = {}
            config = config[key]

        # Set the value
        config[keys[-1]] = value

    def save_config(self, path: Optional[Path] = None) -> None:
        """Save current configuration to file."""
        if path is None:
            path = self.config_path

        try:
            with open(path, 'w') as f:
                yaml.dump(self._config, f, default_flow_style=False, indent=2)
        except Exception as e:
            print(f"Error saving config: {e}")

    def get_env_or_config(self, env_var: str, config_path: str, default: Any = None) -> Any:
        """Get value from environment variable or config, with fallback to default."""
        env_value = os.getenv(env_var)
        if env_value is not None:
            return env_value

        return self.get(config_path, default)