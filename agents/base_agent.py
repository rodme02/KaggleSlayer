"""
Base agent class with common functionality.
"""

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Dict, Any, Optional
from utils.config import ConfigManager
from utils.logging import LoggerMixin
from utils.io import FileManager


class BaseAgent(LoggerMixin, ABC):
    """Base class for all KaggleSlayer agents."""

    def __init__(self, competition_name: str, competition_path: Path,
                 config: Optional[ConfigManager] = None):
        super().__init__()
        self.competition_name = competition_name
        self.competition_path = Path(competition_path)
        self.config = config or ConfigManager()
        self.file_manager = FileManager(self.competition_path)

        self.log_info(f"Initialized {self.__class__.__name__} for competition: {competition_name}")

    @abstractmethod
    def run(self, **kwargs) -> Dict[str, Any]:
        """Run the agent's main functionality."""
        pass

    def save_results(self, results: Dict[str, Any], filename: str) -> None:
        """Save agent results to file."""
        self.file_manager.save_json(results, filename)
        self.log_info(f"Saved results to {filename}")

    def load_results(self, filename: str) -> Dict[str, Any]:
        """Load agent results from file."""
        try:
            results = self.file_manager.load_json(filename)
            self.log_info(f"Loaded results from {filename}")
            return results
        except FileNotFoundError:
            self.log_warning(f"Results file not found: {filename}")
            return {}

    def get_config(self, key: str, default: Any = None) -> Any:
        """Get configuration value."""
        return self.config.get(key, default)