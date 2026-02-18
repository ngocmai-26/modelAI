"""Feature grouping utilities.

This module provides functions to organize and group features according to
pedagogical categories for analysis and XAI interpretation.
"""

from typing import Dict, List, Optional

import pandas as pd

from ml_clo.config.feature_config import (
    FEATURE_GROUPS,
    PEDAGOGICAL_GROUPS,
    add_assessment_method_features,
    add_teaching_method_features,
    get_pedagogical_group_mapping,
)
from ml_clo.utils.logger import get_logger

logger = get_logger(__name__)


def get_feature_groups() -> Dict[str, List[str]]:
    """Get feature groups dictionary.

    Returns:
        Dictionary mapping group names to feature lists
    """
    return FEATURE_GROUPS.copy()


def get_pedagogical_groups() -> Dict[str, List[str]]:
    """Get pedagogical groups dictionary (for XAI).

    Returns:
        Dictionary mapping pedagogical group names to feature lists
    """
    return PEDAGOGICAL_GROUPS.copy()


def group_features_by_pedagogy(
    feature_names: List[str],
    df: Optional[pd.DataFrame] = None,
) -> Dict[str, List[str]]:
    """Group features by pedagogical categories.

    Args:
        feature_names: List of feature names to group
        df: Optional DataFrame to auto-detect TM/EM columns (default: None)

    Returns:
        Dictionary mapping pedagogical group names to feature lists
    """
    # Auto-detect teaching and assessment method columns if DataFrame provided
    if df is not None:
        tm_cols = [col for col in df.columns if "TM" in col or col.startswith("TM")]
        em_cols = [col for col in df.columns if "EM" in col or col.startswith("EM")]

        if tm_cols:
            add_teaching_method_features(tm_cols)
        if em_cols:
            add_assessment_method_features(em_cols)

    # Get mapping
    mapping = get_pedagogical_group_mapping()

    # Import pattern-based mapping from xai_config
    from ml_clo.config.xai_config import PEDAGOGICAL_GROUP_PATTERNS

    # Group features
    grouped = {}
    for group_name in PEDAGOGICAL_GROUPS.keys():
        grouped[group_name] = []

    # Add features to their groups
    for feature in feature_names:
        feature_lower = feature.lower()
        matched = False

        # First, try exact mapping
        if feature in mapping:
            group = mapping[feature]
            grouped[group].append(feature)
            matched = True
        else:
            # Try pattern-based matching
            for group_name, patterns in PEDAGOGICAL_GROUP_PATTERNS.items():
                for pattern in patterns:
                    if pattern.lower() in feature_lower:
                        grouped[group_name].append(feature)
                        matched = True
                        break
                if matched:
                    break

        # If still not matched, try special cases
        if not matched:
            # Check if it's a TM or EM column
            if "TM" in feature or feature.startswith("TM"):
                grouped["Giảng dạy"].append(feature)
                matched = True
            elif "EM" in feature or feature.startswith("EM"):
                grouped["Đánh giá"].append(feature)
                matched = True
            # Check for common academic features
            elif any(term in feature_lower for term in ["score", "exam", "percent", "credit", "summary"]):
                grouped["Học lực"].append(feature)
                matched = True
            # Check for library/study features
            elif any(term in feature_lower for term in ["library", "study", "tuhoc"]):
                grouped["Tự học"].append(feature)
                matched = True
            # Check for attendance features
            elif any(term in feature_lower for term in ["attendance", "diemdanh", "chuyencan"]):
                grouped["Chuyên cần"].append(feature)
                matched = True
            # Check for conduct features
            elif any(term in feature_lower for term in ["conduct", "renluyen", "diemrenluyen"]):
                grouped["Rèn luyện"].append(feature)
                matched = True
            # Check for demographic/personal features
            elif any(term in feature_lower for term in ["gender", "religion", "birth", "ethnicity", "major", "tuition", "user"]):
                grouped["Cá nhân"].append(feature)
                matched = True

        if not matched:
            logger.debug(f"Feature '{feature}' not mapped to any pedagogical group")

    # Remove empty groups
    grouped = {k: v for k, v in grouped.items() if v}

    logger.debug(f"Mapped {len(feature_names)} features into {len(grouped)} pedagogical groups")

    return grouped


def get_features_by_group(
    group_name: str,
    df: pd.DataFrame,
) -> List[str]:
    """Get list of feature names in a specific group that exist in DataFrame.

    Args:
        group_name: Name of feature group
        df: DataFrame to check for feature existence

    Returns:
        List of feature names that exist in DataFrame
    """
    if group_name not in FEATURE_GROUPS:
        logger.warning(f"Unknown feature group: {group_name}")
        return []

    group_features = FEATURE_GROUPS[group_name]
    existing_features = [f for f in group_features if f in df.columns]

    return existing_features


def get_all_existing_features(df: pd.DataFrame) -> List[str]:
    """Get all features that exist in DataFrame, organized by groups.

    Args:
        df: DataFrame to check

    Returns:
        List of all existing feature names
    """
    all_features = []
    for group_features in FEATURE_GROUPS.values():
        existing = [f for f in group_features if f in df.columns]
        all_features.extend(existing)

    # Also check for TM and EM columns
    tm_cols = [col for col in df.columns if ("TM" in col or col.startswith("TM")) and col not in all_features]
    em_cols = [col for col in df.columns if ("EM" in col or col.startswith("EM")) and col not in all_features]

    all_features.extend(tm_cols)
    all_features.extend(em_cols)

    return all_features


