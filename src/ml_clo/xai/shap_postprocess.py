"""SHAP postprocessing utilities.

This module provides functions to filter, group, and normalize SHAP values
according to pedagogical categories for interpretability.
"""

from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from ml_clo.config.xai_config import IMPACT_CONFIG, SHAP_CONFIG
from ml_clo.features.feature_groups import group_features_by_pedagogy
from ml_clo.utils.logger import get_logger

logger = get_logger(__name__)


def filter_shap_values(
    shap_values: np.ndarray,
    feature_names: List[str],
    threshold: Optional[float] = None,
) -> Tuple[np.ndarray, List[str]]:
    """Filter SHAP values by absolute threshold.

    Args:
        shap_values: SHAP values array (n_samples, n_features)
        feature_names: List of feature names
        threshold: Absolute threshold for filtering (default: from config)

    Returns:
        Tuple of (filtered_shap_values, filtered_feature_names)
    """
    if threshold is None:
        # DESIGN-06: Single source of truth — read from xai_config.SHAP_CONFIG.
        # The fallback constant matches the config default and exists only as
        # a safety net if SHAP_CONFIG is missing the key entirely.
        threshold = SHAP_CONFIG["shap_threshold"]

    # Calculate mean absolute SHAP per feature across samples
    if len(shap_values.shape) == 1:
        mean_abs_shap = np.abs(shap_values)
    else:
        mean_abs_shap = np.mean(np.abs(shap_values), axis=0)

    # Filter features above threshold
    mask = mean_abs_shap >= threshold
    filtered_indices = np.where(mask)[0]

    # BUG-04: If threshold filters out every feature, fall back to the
    # top-K features by absolute SHAP so downstream reasoning still has
    # something to work with (instead of returning an empty reason list).
    if len(filtered_indices) == 0 and len(feature_names) > 0:
        fallback_k = min(5, len(feature_names))
        logger.warning(
            f"No SHAP values above threshold {threshold}; "
            f"falling back to top-{fallback_k} features by absolute SHAP."
        )
        filtered_indices = np.argsort(mean_abs_shap)[-fallback_k:][::-1]

    if len(shap_values.shape) == 1:
        filtered_shap = shap_values[filtered_indices]
    else:
        filtered_shap = shap_values[:, filtered_indices]

    filtered_features = [feature_names[i] for i in filtered_indices]

    logger.debug(
        f"Filtered SHAP values: {len(feature_names)} -> {len(filtered_features)} "
        f"features (threshold={threshold})"
    )

    return filtered_shap, filtered_features


def group_shap_by_pedagogy(
    shap_values: np.ndarray,
    feature_names: List[str],
    df: Optional[pd.DataFrame] = None,
) -> Dict[str, np.ndarray]:
    """Group SHAP values by pedagogical categories.

    Args:
        shap_values: SHAP values array (n_samples, n_features) or (n_features,)
        feature_names: List of feature names
        df: Optional DataFrame to auto-detect TM/EM columns (default: None)

    Returns:
        Dictionary mapping pedagogical group names to aggregated SHAP values
    """
    # Group features by pedagogy
    grouped_features = group_features_by_pedagogy(feature_names, df)

    if not grouped_features:
        logger.warning(
            f"No features grouped by pedagogy. "
            f"Feature names: {feature_names[:10]}... (showing first 10)"
        )
        return {}

    # Initialize result dictionary
    grouped_shap = {}

    # Create feature index mapping
    feature_to_idx = {name: idx for idx, name in enumerate(feature_names)}

    # Aggregate SHAP values for each group
    for group_name, features in grouped_features.items():
        if not features:
            continue

        # Get indices of features in this group
        indices = [feature_to_idx[f] for f in features if f in feature_to_idx]

        if not indices:
            logger.debug(f"No matching features found for group '{group_name}'")
            continue

        # Aggregate SHAP values (sum for group)
        if len(shap_values.shape) == 1:
            group_shap = np.sum(shap_values[indices])
        else:
            group_shap = np.sum(shap_values[:, indices], axis=1)

        grouped_shap[group_name] = group_shap

    logger.debug(
        f"Grouped SHAP values into {len(grouped_shap)} pedagogical categories: {list(grouped_shap.keys())}"
    )

    return grouped_shap


def calculate_impact_percentage(
    shap_values: np.ndarray,
    method: Optional[str] = None,
) -> np.ndarray:
    """Calculate impact percentage from SHAP values.

    Args:
        shap_values: SHAP values array (n_samples, n_features) or (n_features,)
        method: Calculation method - "absolute" or "relative" (default: from config)

    Returns:
        Impact percentage array with same shape as input
    """
    if method is None:
        method = IMPACT_CONFIG.get("calculation_method", "relative")

    if method == "absolute":
        # Use absolute SHAP values directly
        impact = np.abs(shap_values)
    elif method == "relative":
        # Normalize to percentage of total absolute SHAP
        abs_shap = np.abs(shap_values)
        total_abs = np.sum(abs_shap, axis=-1, keepdims=True) if len(abs_shap.shape) > 1 else np.sum(abs_shap)
        
        if total_abs > 0:
            if len(abs_shap.shape) > 1:
                impact = (abs_shap / total_abs) * 100
            else:
                impact = (abs_shap / total_abs) * 100
        else:
            impact = np.zeros_like(abs_shap)
    else:
        raise ValueError(f"Unknown impact calculation method: {method}")

    # Round to specified decimal places
    decimal_places = IMPACT_CONFIG.get("impact_decimal_places", 2)
    impact = np.round(impact, decimal_places)

    return impact


def get_top_negative_impacts(
    grouped_shap: Dict[str, np.ndarray],
    top_n: Optional[int] = None,
) -> List[Tuple[str, float, float]]:
    """Get top N groups with highest negative impact (reducing score).

    If no negative impacts found, returns top absolute impacts (both positive and negative).

    Args:
        grouped_shap: Dictionary mapping group names to SHAP values
        top_n: Number of top groups to return (default: from config)

    Returns:
        List of tuples (group_name, shap_value, impact_percentage) sorted by impact
    """
    if top_n is None:
        top_n = SHAP_CONFIG.get("top_n_reasons", 5)

    if not grouped_shap:
        logger.warning("No grouped SHAP values found")
        return []

    # Calculate impact percentages
    group_impacts = {}
    all_shap_values = []
    
    for group_name, shap_vals in grouped_shap.items():
        # Use mean if multiple samples
        if isinstance(shap_vals, np.ndarray) and len(shap_vals.shape) > 0 and len(shap_vals) > 1:
            mean_shap = np.mean(shap_vals)
        else:
            mean_shap = float(shap_vals) if isinstance(shap_vals, (int, float, np.number)) else float(np.mean(shap_vals))
        
        all_shap_values.append(abs(mean_shap))
        
        # Only consider negative impacts (reducing score)
        if mean_shap < 0:
            group_impacts[group_name] = mean_shap

    # If no negative impacts, use top absolute impacts (both positive and negative)
    if not group_impacts:
        logger.debug("No negative impacts found, using top absolute impacts")
        for group_name, shap_vals in grouped_shap.items():
            if isinstance(shap_vals, np.ndarray) and len(shap_vals.shape) > 0 and len(shap_vals) > 1:
                mean_shap = np.mean(shap_vals)
            else:
                mean_shap = float(shap_vals) if isinstance(shap_vals, (int, float, np.number)) else float(np.mean(shap_vals))
            group_impacts[group_name] = mean_shap

    # Calculate total absolute SHAP for percentage calculation
    total_abs = sum([abs(v) for v in group_impacts.values()])
    
    if total_abs == 0:
        logger.warning("Total absolute SHAP is zero, cannot calculate impact percentages")
        return []

    # Calculate impact percentages
    impacts_with_pct = {}
    for group_name, mean_shap in group_impacts.items():
        impact_pct = (abs(mean_shap) / total_abs) * 100
        impacts_with_pct[group_name] = (mean_shap, impact_pct)

    # Sort by impact percentage (descending)
    sorted_impacts = sorted(
        impacts_with_pct.items(),
        key=lambda x: x[1][1],  # Sort by impact percentage
        reverse=True,
    )

    # Return top N
    result = [
        (group_name, shap_val, impact_pct)
        for group_name, (shap_val, impact_pct) in sorted_impacts[:top_n]
    ]

    logger.debug(f"Found {len(result)} top impacts from {len(grouped_shap)} groups")
    
    return result


def aggregate_class_shap(
    shap_values_list: List[np.ndarray],
    feature_names: List[str],
) -> np.ndarray:
    """Aggregate SHAP values across multiple instances (for class-level analysis).

    Args:
        shap_values_list: List of SHAP value arrays (one per instance)
        feature_names: List of feature names (must be consistent across instances)

    Returns:
        Aggregated SHAP values array (averaged across instances)
    """
    if not shap_values_list:
        raise ValueError("shap_values_list cannot be empty")

    # Stack all SHAP values
    stacked = np.stack(shap_values_list, axis=0)

    # Average across instances
    aggregated = np.mean(stacked, axis=0)

    logger.debug(
        f"Aggregated SHAP values from {len(shap_values_list)} instances, "
        f"shape: {aggregated.shape}"
    )

    return aggregated


def process_shap_for_analysis(
    shap_values: np.ndarray,
    feature_names: List[str],
    df: Optional[pd.DataFrame] = None,
    filter_threshold: Optional[float] = None,
    top_n: Optional[int] = None,
) -> Dict[str, any]:
    """Complete SHAP processing pipeline for analysis.

    Args:
        shap_values: SHAP values array (n_samples, n_features) or (n_features,)
        feature_names: List of feature names
        df: Optional DataFrame to auto-detect TM/EM columns (default: None)
        filter_threshold: Optional threshold for filtering (default: from config)
        top_n: Optional number of top reasons (default: from config)

    Returns:
        Dictionary containing:
        - filtered_shap: Filtered SHAP values
        - filtered_features: Filtered feature names
        - grouped_shap: SHAP grouped by pedagogy
        - impact_percentages: Impact percentages per group
        - top_negative_impacts: Top N negative impacts
    """
    # Step 1: Filter by threshold
    filtered_shap, filtered_features = filter_shap_values(
        shap_values, feature_names, filter_threshold
    )

    # Step 2: Group by pedagogy
    grouped_shap = group_shap_by_pedagogy(filtered_shap, filtered_features, df)

    # Step 3: Calculate impact percentages
    impact_percentages = {}
    for group_name, group_shap_vals in grouped_shap.items():
        impact_pct = calculate_impact_percentage(group_shap_vals)
        impact_percentages[group_name] = impact_pct

    # Step 4: Get top negative impacts
    top_negative = get_top_negative_impacts(grouped_shap, top_n)

    return {
        "filtered_shap": filtered_shap,
        "filtered_features": filtered_features,
        "grouped_shap": grouped_shap,
        "impact_percentages": impact_percentages,
        "top_negative_impacts": top_negative,
    }

