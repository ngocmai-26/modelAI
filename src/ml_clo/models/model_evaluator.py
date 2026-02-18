"""Model evaluation utilities.

This module provides functions to evaluate model performance using various metrics.
"""

from typing import Dict

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from ml_clo.utils.logger import get_logger

logger = get_logger(__name__)


def evaluate_model(
    y_true: pd.Series | np.ndarray,
    y_pred: pd.Series | np.ndarray,
    prefix: str = "",
) -> Dict[str, float]:
    """Evaluate model predictions.

    Calculates MAE, RMSE, and R² metrics.

    Args:
        y_true: True target values
        y_pred: Predicted values
        prefix: Prefix for metric names (e.g., "train_", "val_") (default: "")

    Returns:
        Dictionary of evaluation metrics
    """
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred)

    metrics = {
        f"{prefix}mae": mae,
        f"{prefix}rmse": rmse,
        f"{prefix}r2": r2,
    }

    logger.info(
        f"Evaluation metrics ({prefix}): "
        f"MAE={mae:.4f}, RMSE={rmse:.4f}, R²={r2:.4f}"
    )

    return metrics


def evaluate_by_score_range(
    y_true: pd.Series | np.ndarray,
    y_pred: pd.Series | np.ndarray,
    score_ranges: list = None,
) -> Dict[str, Dict[str, float]]:
    """Evaluate model performance by score ranges.

    Useful for understanding model performance across different CLO score levels.

    Args:
        y_true: True target values
        y_pred: Predicted values
        score_ranges: List of (min, max) tuples for score ranges (default: None, uses default ranges)

    Returns:
        Dictionary mapping range names to evaluation metrics
    """
    if score_ranges is None:
        # Default ranges for 0-6 CLO scale
        score_ranges = [
            (0.0, 2.0, "Low (0-2)"),
            (2.0, 4.0, "Medium (2-4)"),
            (4.0, 6.0, "High (4-6)"),
        ]

    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    results = {}

    for min_score, max_score, range_name in score_ranges:
        mask = (y_true >= min_score) & (y_true < max_score)
        if mask.sum() == 0:
            continue

        y_true_range = y_true[mask]
        y_pred_range = y_pred[mask]

        metrics = evaluate_model(y_true_range, y_pred_range, prefix=f"{range_name}_")
        metrics["count"] = len(y_true_range)
        results[range_name] = metrics

        logger.info(
            f"Score range {range_name}: {len(y_true_range)} samples, "
            f"MAE={metrics[f'{range_name}_mae']:.4f}"
        )

    return results


def calculate_prediction_errors(
    y_true: pd.Series | np.ndarray,
    y_pred: pd.Series | np.ndarray,
) -> pd.DataFrame:
    """Calculate detailed prediction errors.

    Args:
        y_true: True target values
        y_pred: Predicted values

    Returns:
        DataFrame with error analysis
    """
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    errors = y_pred - y_true
    abs_errors = np.abs(errors)

    error_df = pd.DataFrame({
        "true": y_true,
        "pred": y_pred,
        "error": errors,
        "abs_error": abs_errors,
        "squared_error": errors ** 2,
    })

    return error_df


def print_evaluation_summary(metrics: Dict[str, float]) -> None:
    """Print evaluation metrics summary.

    Args:
        metrics: Dictionary of evaluation metrics
    """
    print("\n" + "=" * 60)
    print("MODEL EVALUATION SUMMARY")
    print("=" * 60)

    for metric_name, value in sorted(metrics.items()):
        if isinstance(value, float):
            print(f"  {metric_name:20s}: {value:.4f}")
        else:
            print(f"  {metric_name:20s}: {value}")

    print("=" * 60)


