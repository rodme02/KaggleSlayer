"""
File I/O utilities for managing data and results.
"""

import json
import pickle
from pathlib import Path
from typing import Any, Dict, List, Optional, Union
import pandas as pd


class FileManager:
    """Manages file I/O operations for the pipeline."""

    def __init__(self, base_path: Optional[Path] = None):
        if base_path is None:
            base_path = Path.cwd()
        self.base_path = Path(base_path)

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