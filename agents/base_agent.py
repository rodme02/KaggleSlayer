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
    """Base class for all KaggleSlayer agents.

    Provides common functionality for all pipeline agents including:
    - Logging via LoggerMixin
    - Configuration management
    - File I/O operations via FileManager
    - Directory structure setup

    Attributes:
        competition_name: Name of the Kaggle competition
        competition_path: Path to competition data directory
        config: Configuration manager instance
        file_manager: File manager for I/O operations
    """

    def __init__(self, competition_name: str, competition_path: Path,
                 config: Optional[ConfigManager] = None):
        """Initialize the base agent.

        Args:
            competition_name: Name of the Kaggle competition
            competition_path: Path to competition data directory
            config: Optional configuration manager (creates default if None)
        """
        super().__init__()
        self.competition_name = competition_name
        self.competition_path = Path(competition_path)
        self.config = config or ConfigManager()
        self.file_manager = FileManager(self.competition_path)

        # Setup organized directory structure
        self.file_manager.setup_directories()

        self.log_info(f"Initialized {self.__class__.__name__} for competition: {competition_name}")

    @abstractmethod
    def run(self, **kwargs) -> Dict[str, Any]:
        """Run the agent's main functionality.

        This method must be implemented by all subclasses.

        Args:
            **kwargs: Agent-specific keyword arguments

        Returns:
            Dictionary containing agent execution results
        """
        pass

    def save_results(self, results: Dict[str, Any], filename: str) -> None:
        """Save agent results to JSON file.

        Args:
            results: Dictionary of results to save
            filename: Name of the file to save to (in results directory)
        """
        self.file_manager.save_results(results, filename)
        self.log_info(f"Saved results to {filename}")

    def load_results(self, filename: str) -> Dict[str, Any]:
        """Load agent results from JSON file.

        Args:
            filename: Name of the file to load from (in results directory)

        Returns:
            Dictionary of loaded results, or empty dict if file not found
        """
        try:
            results = self.file_manager.load_results(filename)
            self.log_info(f"Loaded results from {filename}")
            return results
        except FileNotFoundError:
            self.log_warning(f"Results file not found: {filename}")
            return {}

    def get_config(self, key: str, default: Any = None) -> Any:
        """Get configuration value from config manager.

        Args:
            key: Configuration key (supports nested keys like 'pipeline.cv_folds')
            default: Default value if key not found

        Returns:
            Configuration value or default
        """
        return self.config.get(key, default)