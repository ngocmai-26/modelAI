"""Logging configuration for ML CLO library."""

import logging
import sys
from typing import Optional


def get_logger(name: str, level: int = logging.INFO) -> logging.Logger:
    """Get a configured logger instance.

    Args:
        name: Logger name (typically __name__)
        level: Logging level (default: INFO)

    Returns:
        Configured logger instance
    """
    logger = logging.getLogger(name)
    logger.setLevel(level)

    # Avoid adding multiple handlers if logger already configured
    if logger.handlers:
        return logger

    # Create console handler
    handler = logging.StreamHandler(sys.stdout)
    handler.setLevel(level)

    # Create formatter
    formatter = logging.Formatter(
        fmt="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    handler.setFormatter(formatter)

    logger.addHandler(handler)
    logger.propagate = False

    return logger


def hash_sensitive_data(value: str, length: int = 8) -> str:
    """Hash sensitive data for logging purposes.

    Args:
        value: Value to hash
        length: Length of hash to return (default: 8)

    Returns:
        Hashed value
    """
    import hashlib

    if not value:
        return "N/A"

    hash_obj = hashlib.sha256(str(value).encode())
    hash_hex = hash_obj.hexdigest()
    return hash_hex[:length]

