"""File I/O utilities for CLO prediction.

This module provides utilities for:
- Model save/load operations
- File path validation
- Data file reading/writing helpers
"""

import json
from pathlib import Path
from typing import Any, Dict, Optional

import joblib
import pandas as pd

from ml_clo.utils.exceptions import ModelLoadError
from ml_clo.utils.logger import get_logger

logger = get_logger(__name__)


def ensure_directory_exists(file_path: str) -> Path:
    """Ensure directory for file path exists.

    Args:
        file_path: Path to file (directory will be created if needed)

    Returns:
        Path object for the file

    Examples:
        >>> path = ensure_directory_exists("models/my_model.joblib")
        >>> path.parent.exists()
        True
    """
    path = Path(file_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def validate_file_exists(file_path: str, file_type: str = "file") -> Path:
    """Validate that a file exists.

    Args:
        file_path: Path to file
        file_type: Type of file for error message (default: "file")

    Returns:
        Path object for the file

    Raises:
        FileNotFoundError: If file does not exist
    """
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"{file_type.capitalize()} not found: {file_path}")
    return path


def save_model(
    model: Any,
    file_path: str,
    metadata: Optional[Dict[str, Any]] = None,
    compress: int = 3,
) -> None:
    """Save a model to file using joblib.

    Args:
        model: Model object to save
        file_path: Path to save model file
        metadata: Optional metadata dictionary to save with model
        compress: Compression level (0-9, default: 3)

    Examples:
        >>> save_model(my_model, "models/model.joblib", metadata={"version": "1.0"})
    """
    path = ensure_directory_exists(file_path)

    model_data = {
        "model": model,
        "metadata": metadata or {},
    }

    try:
        joblib.dump(model_data, path, compress=compress)
        logger.info(f"Saved model to {file_path}")
    except Exception as e:
        logger.error(f"Failed to save model to {file_path}: {e}")
        raise


def load_model(file_path: str) -> Dict[str, Any]:
    """Load a model from file using joblib.

    Args:
        file_path: Path to model file

    Returns:
        Dictionary with 'model' and 'metadata' keys

    Raises:
        ModelLoadError: If model cannot be loaded

    Examples:
        >>> model_data = load_model("models/model.joblib")
        >>> model = model_data["model"]
        >>> metadata = model_data["metadata"]
    """
    path = validate_file_exists(file_path, "model file")

    try:
        model_data = joblib.load(path)
        logger.info(f"Loaded model from {file_path}")

        # Ensure model_data has expected structure
        if isinstance(model_data, dict):
            if "model" not in model_data:
                # Old format - model_data is the model itself
                return {"model": model_data, "metadata": {}}
            return model_data
        else:
            # Old format - model_data is the model itself
            return {"model": model_data, "metadata": {}}

    except Exception as e:
        logger.error(f"Failed to load model from {file_path}: {e}")
        raise ModelLoadError(f"Cannot load model from {file_path}: {e}") from e


def save_json(data: Dict[str, Any], file_path: str, indent: int = 2) -> None:
    """Save data to JSON file.

    Args:
        data: Dictionary to save
        file_path: Path to JSON file
        indent: JSON indentation level (default: 2)

    Examples:
        >>> save_json({"key": "value"}, "output.json")
    """
    path = ensure_directory_exists(file_path)

    try:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=indent, ensure_ascii=False)
        logger.info(f"Saved JSON to {file_path}")
    except Exception as e:
        logger.error(f"Failed to save JSON to {file_path}: {e}")
        raise


def load_json(file_path: str) -> Dict[str, Any]:
    """Load data from JSON file.

    Args:
        file_path: Path to JSON file

    Returns:
        Dictionary loaded from JSON file

    Raises:
        FileNotFoundError: If file does not exist
        json.JSONDecodeError: If file is not valid JSON

    Examples:
        >>> data = load_json("output.json")
    """
    path = validate_file_exists(file_path, "JSON file")

    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        logger.info(f"Loaded JSON from {file_path}")
        return data
    except json.JSONDecodeError as e:
        logger.error(f"Invalid JSON in {file_path}: {e}")
        raise
    except Exception as e:
        logger.error(f"Failed to load JSON from {file_path}: {e}")
        raise


def save_csv(df: pd.DataFrame, file_path: str, **kwargs) -> None:
    """Save DataFrame to CSV file.

    Args:
        df: DataFrame to save
        file_path: Path to CSV file
        **kwargs: Additional arguments passed to pd.DataFrame.to_csv()

    Examples:
        >>> save_csv(df, "output.csv", index=False)
    """
    path = ensure_directory_exists(file_path)

    try:
        df.to_csv(path, **kwargs)
        logger.info(f"Saved CSV to {file_path} ({len(df)} rows)")
    except Exception as e:
        logger.error(f"Failed to save CSV to {file_path}: {e}")
        raise


def load_csv(file_path: str, **kwargs) -> pd.DataFrame:
    """Load DataFrame from CSV file.

    Args:
        file_path: Path to CSV file
        **kwargs: Additional arguments passed to pd.read_csv()

    Returns:
        DataFrame loaded from CSV file

    Examples:
        >>> df = load_csv("data.csv", index_col=0)
    """
    path = validate_file_exists(file_path, "CSV file")

    try:
        df = pd.read_csv(path, **kwargs)
        logger.info(f"Loaded CSV from {file_path} ({len(df)} rows)")
        return df
    except Exception as e:
        logger.error(f"Failed to load CSV from {file_path}: {e}")
        raise


def get_file_size(file_path: str) -> int:
    """Get file size in bytes.

    Args:
        file_path: Path to file

    Returns:
        File size in bytes

    Examples:
        >>> size = get_file_size("models/model.joblib")
        >>> print(f"Model size: {size / 1024 / 1024:.2f} MB")
    """
    path = validate_file_exists(file_path)
    return path.stat().st_size


def format_file_size(size_bytes: int) -> str:
    """Format file size in human-readable format.

    Args:
        size_bytes: File size in bytes

    Returns:
        Human-readable file size string (e.g., "1.5 MB")

    Examples:
        >>> format_file_size(1572864)
        '1.5 MB'
    """
    for unit in ["B", "KB", "MB", "GB", "TB"]:
        if size_bytes < 1024.0:
            return f"{size_bytes:.2f} {unit}"
        size_bytes /= 1024.0
    return f"{size_bytes:.2f} PB"

