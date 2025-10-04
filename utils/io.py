"""
File I/O utilities for managing data and results.
"""

import json
import pickle
from pathlib import Path
from typing import Any, Dict, List, Optional, Union
import pandas as pd


class FileManager:
    """Manages file I/O operations for the pipeline with organized subdirectories.

    Directory structure:
        base_path/
        ├── raw/         # Downloaded CSV files
        ├── processed/   # Cleaned and engineered data
        ├── models/      # Saved models
        ├── results/     # JSON result files
        └── submission.csv  # Final submission
    """

    def __init__(self, base_path: Optional[Path] = None):
        if base_path is None:
            base_path = Path.cwd()
        self.base_path = Path(base_path)

        # Define subdirectories
        self.raw_dir = "raw"
        self.processed_dir = "processed"
        self.models_dir = "models"
        self.results_dir = "results"

    def save_json(self, data: Dict[str, Any], file_path: Union[str, Path]) -> None:
        """Save data as JSON file."""
        file_path = self.base_path / file_path
        file_path.parent.mkdir(parents=True, exist_ok=True)

        with open(file_path, 'w') as f:
            json.dump(data, f, indent=2, default=str)

    def load_json(self, file_path: Union[str, Path]) -> Dict[str, Any]:
        """Load data from JSON file."""
        file_path = self.base_path / file_path

        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")

        with open(file_path, 'r') as f:
            return json.load(f)

    def save_pickle(self, data: Any, file_path: Union[str, Path]) -> None:
        """Save data as pickle file."""
        file_path = self.base_path / file_path
        file_path.parent.mkdir(parents=True, exist_ok=True)

        with open(file_path, 'wb') as f:
            pickle.dump(data, f)

    def load_pickle(self, file_path: Union[str, Path]) -> Any:
        """Load data from pickle file."""
        file_path = self.base_path / file_path

        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")

        with open(file_path, 'rb') as f:
            return pickle.load(f)

    def save_dataframe(self, df: pd.DataFrame, file_path: Union[str, Path],
                      format: str = "csv") -> None:
        """Save DataFrame to file."""
        file_path = self.base_path / file_path
        file_path.parent.mkdir(parents=True, exist_ok=True)

        if format.lower() == "csv":
            df.to_csv(file_path, index=False)
        elif format.lower() == "parquet":
            df.to_parquet(file_path, index=False)
        elif format.lower() == "xlsx":
            df.to_excel(file_path, index=False)
        else:
            raise ValueError(f"Unsupported format: {format}")

    def load_dataframe(self, file_path: Union[str, Path],
                      format: Optional[str] = None) -> pd.DataFrame:
        """Load DataFrame from file."""
        file_path = self.base_path / file_path

        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")

        # Auto-detect format from extension if not specified
        if format is None:
            format = file_path.suffix.lower().lstrip('.')

        if format == "csv":
            return pd.read_csv(file_path)
        elif format == "parquet":
            return pd.read_parquet(file_path)
        elif format in ["xlsx", "xls"]:
            return pd.read_excel(file_path)
        else:
            raise ValueError(f"Unsupported format: {format}")

    def ensure_directory(self, dir_path: Union[str, Path]) -> Path:
        """Ensure directory exists, create if not."""
        dir_path = self.base_path / dir_path
        dir_path.mkdir(parents=True, exist_ok=True)
        return dir_path

    def list_files(self, pattern: str = "*", recursive: bool = False) -> List[Path]:
        """List files matching pattern."""
        if recursive:
            return list(self.base_path.rglob(pattern))
        else:
            return list(self.base_path.glob(pattern))

    def file_exists(self, file_path: Union[str, Path]) -> bool:
        """Check if file exists."""
        return (self.base_path / file_path).exists()

    def get_file_size(self, file_path: Union[str, Path]) -> int:
        """Get file size in bytes."""
        file_path = self.base_path / file_path
        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")
        return file_path.stat().st_size

    def delete_file(self, file_path: Union[str, Path]) -> None:
        """Delete file if it exists."""
        file_path = self.base_path / file_path
        if file_path.exists():
            file_path.unlink()

    def copy_file(self, source: Union[str, Path], destination: Union[str, Path]) -> None:
        """Copy file from source to destination."""
        import shutil
        source_path = self.base_path / source
        dest_path = self.base_path / destination

        dest_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source_path, dest_path)

    def get_file_path(self, file_path: Union[str, Path]) -> Path:
        """Get full path for a file."""
        return self.base_path / file_path

    # Organized directory helpers
    def save_processed_data(self, df: pd.DataFrame, filename: str) -> None:
        """Save processed data to processed/ subdirectory."""
        self.save_dataframe(df, f"{self.processed_dir}/{filename}")

    def load_processed_data(self, filename: str) -> pd.DataFrame:
        """Load processed data from processed/ subdirectory."""
        return self.load_dataframe(f"{self.processed_dir}/{filename}")

    def save_model(self, model: Any, filename: str) -> None:
        """Save model to models/ subdirectory."""
        self.save_pickle(model, f"{self.models_dir}/{filename}")

    def load_model(self, filename: str) -> Any:
        """Load model from models/ subdirectory."""
        return self.load_pickle(f"{self.models_dir}/{filename}")

    def save_results(self, data: Dict[str, Any], filename: str) -> None:
        """Save results to results/ subdirectory."""
        self.save_json(data, f"{self.results_dir}/{filename}")

    def load_results(self, filename: str) -> Dict[str, Any]:
        """Load results from results/ subdirectory."""
        return self.load_json(f"{self.results_dir}/{filename}")

    def get_raw_path(self) -> Path:
        """Get path to raw/ subdirectory."""
        return self.base_path / self.raw_dir

    def get_processed_path(self) -> Path:
        """Get path to processed/ subdirectory."""
        return self.base_path / self.processed_dir

    def get_models_path(self) -> Path:
        """Get path to models/ subdirectory."""
        return self.base_path / self.models_dir

    def get_results_path(self) -> Path:
        """Get path to results/ subdirectory."""
        return self.base_path / self.results_dir

    def setup_directories(self) -> None:
        """Create all organized subdirectories."""
        self.ensure_directory(self.raw_dir)
        self.ensure_directory(self.processed_dir)
        self.ensure_directory(self.models_dir)
        self.ensure_directory(self.results_dir)