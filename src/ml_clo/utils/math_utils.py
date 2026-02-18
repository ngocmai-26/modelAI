"""Math utilities for CLO prediction.

This module provides mathematical utility functions including:
- Score conversion between different scales
- Statistical calculations
- Range validation
"""

from typing import Optional, Union

import numpy as np
import pandas as pd

from ml_clo.utils.logger import get_logger

logger = get_logger(__name__)


def convert_score_10_to_6(score: Union[float, pd.Series, np.ndarray]) -> Union[float, pd.Series, np.ndarray]:
    """Convert a single score or array of scores from 10-point scale to 6-point scale (CLO scale).

    **CRITICAL**: This conversion is mandatory for all historical exam scores.
    Formula: CLO_6 = Score_10 / 10 × 6

    Args:
        score: Score(s) in 10-point scale (0-10). Can be float, Series, or array.

    Returns:
        Converted score(s) in 6-point scale (0-6). Same type as input.

    Examples:
        >>> convert_score_10_to_6(10.0)
        6.0
        >>> convert_score_10_to_6(5.0)
        3.0
        >>> convert_score_10_to_6([10.0, 5.0, 0.0])
        array([6.0, 3.0, 0.0])
    """
    if isinstance(score, pd.Series):
        result = score / 10.0 * 6.0
        result = result.clip(lower=0.0, upper=6.0)
        return result
    elif isinstance(score, np.ndarray):
        result = score / 10.0 * 6.0
        result = np.clip(result, 0.0, 6.0)
        return result
    else:
        # Single float value
        result = float(score) / 10.0 * 6.0
        return max(0.0, min(6.0, result))


def convert_score_6_to_10(score: Union[float, pd.Series, np.ndarray]) -> Union[float, pd.Series, np.ndarray]:
    """Convert a single score or array of scores from 6-point scale to 10-point scale.

    Formula: Score_10 = CLO_6 / 6 × 10

    Args:
        score: Score(s) in 6-point scale (0-6). Can be float, Series, or array.

    Returns:
        Converted score(s) in 10-point scale (0-10). Same type as input.

    Examples:
        >>> convert_score_6_to_10(6.0)
        10.0
        >>> convert_score_6_to_10(3.0)
        5.0
    """
    if isinstance(score, pd.Series):
        result = score / 6.0 * 10.0
        result = result.clip(lower=0.0, upper=10.0)
        return result
    elif isinstance(score, np.ndarray):
        result = score / 6.0 * 10.0
        result = np.clip(result, 0.0, 10.0)
        return result
    else:
        # Single float value
        result = float(score) / 6.0 * 10.0
        return max(0.0, min(10.0, result))


def validate_score_range(
    score: Union[float, pd.Series, np.ndarray],
    min_score: float = 0.0,
    max_score: float = 6.0,
    scale: str = "clo",
) -> bool:
    """Validate that score(s) are within valid range.

    Args:
        score: Score(s) to validate. Can be float, Series, or array.
        min_score: Minimum valid score (default: 0.0)
        max_score: Maximum valid score (default: 6.0 for CLO scale)
        scale: Scale name for error messages (default: "clo")

    Returns:
        True if all scores are in valid range, False otherwise

    Examples:
        >>> validate_score_range(4.5, min_score=0.0, max_score=6.0)
        True
        >>> validate_score_range(7.0, min_score=0.0, max_score=6.0)
        False
    """
    if isinstance(score, pd.Series):
        valid = ((score >= min_score) & (score <= max_score)).all()
    elif isinstance(score, np.ndarray):
        valid = np.all((score >= min_score) & (score <= max_score))
    else:
        valid = min_score <= float(score) <= max_score

    if not valid:
        if isinstance(score, (pd.Series, np.ndarray)):
            invalid_count = np.sum((score < min_score) | (score > max_score))
            logger.warning(
                f"Found {invalid_count} scores outside valid range "
                f"[{min_score}, {max_score}] for {scale} scale"
            )
        else:
            logger.warning(
                f"Score {score} is outside valid range "
                f"[{min_score}, {max_score}] for {scale} scale"
            )

    return valid


def clip_scores(
    score: Union[float, pd.Series, np.ndarray],
    min_score: float = 0.0,
    max_score: float = 6.0,
) -> Union[float, pd.Series, np.ndarray]:
    """Clip scores to valid range.

    Args:
        score: Score(s) to clip. Can be float, Series, or array.
        min_score: Minimum valid score (default: 0.0)
        max_score: Maximum valid score (default: 6.0)

    Returns:
        Clipped score(s). Same type as input.

    Examples:
        >>> clip_scores(7.0, min_score=0.0, max_score=6.0)
        6.0
        >>> clip_scores(-1.0, min_score=0.0, max_score=6.0)
        0.0
    """
    if isinstance(score, pd.Series):
        return score.clip(lower=min_score, upper=max_score)
    elif isinstance(score, np.ndarray):
        return np.clip(score, min_score, max_score)
    else:
        return max(min_score, min(max_score, float(score)))


def calculate_percentage(value: float, total: float) -> float:
    """Calculate percentage value.

    Args:
        value: Value to calculate percentage for
        total: Total value (denominator)

    Returns:
        Percentage value (0-100)

    Examples:
        >>> calculate_percentage(3, 10)
        30.0
        >>> calculate_percentage(0, 10)
        0.0
    """
    if total == 0:
        return 0.0
    return (value / total) * 100.0


def normalize_to_percentage(
    values: Union[pd.Series, np.ndarray],
    total: Optional[float] = None,
) -> Union[pd.Series, np.ndarray]:
    """Normalize values to percentage (0-100).

    Args:
        values: Values to normalize. Can be Series or array.
        total: Total value for normalization. If None, uses sum of values.

    Returns:
        Normalized values as percentages. Same type as input.

    Examples:
        >>> normalize_to_percentage([10, 20, 30])
        array([16.67, 33.33, 50.0])
    """
    if isinstance(values, pd.Series):
        if total is None:
            total = values.sum()
        if total == 0:
            return pd.Series([0.0] * len(values), index=values.index)
        return (values / total) * 100.0
    else:
        values = np.asarray(values)
        if total is None:
            total = values.sum()
        if total == 0:
            return np.zeros_like(values)
        return (values / total) * 100.0


def round_to_precision(value: Union[float, pd.Series, np.ndarray], precision: int = 2) -> Union[float, pd.Series, np.ndarray]:
    """Round value(s) to specified precision.

    Args:
        value: Value(s) to round. Can be float, Series, or array.
        precision: Number of decimal places (default: 2)

    Returns:
        Rounded value(s). Same type as input.

    Examples:
        >>> round_to_precision(3.14159, precision=2)
        3.14
        >>> round_to_precision([3.14159, 2.71828], precision=2)
        array([3.14, 2.72])
    """
    if isinstance(value, pd.Series):
        return value.round(precision)
    elif isinstance(value, np.ndarray):
        return np.round(value, precision)
    else:
        return round(float(value), precision)

