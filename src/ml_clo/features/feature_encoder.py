"""Shared feature preparation/encoding logic for train, predict, and analysis pipelines.

DESIGN-02: Centralizes the feature-selection + categorical-encoding logic
that was previously duplicated across `train_pipeline.prepare_features`,
`predict_pipeline.prepare_features`, and `analysis_pipeline.prepare_features`.
Any change to feature exclusion rules or encoding strategy now lives in
exactly one place.
"""

from typing import List, Optional, Tuple

import numpy as np
import pandas as pd

from ml_clo.utils.hash_utils import stable_hash_int
from ml_clo.utils.logger import get_logger

logger = get_logger(__name__)

# Columns that are never used as model features.
EXCLUDE_COLS = ("Student_ID", "Subject_ID", "Lecturer_ID", "exam_score", "year")
# Features explicitly excluded for stability (raw min score is too noisy;
# the model uses min_exam_score_adj + academic_core_score instead).
ALWAYS_EXCLUDE_FEATURES = ("min_exam_score",)


def select_feature_columns(
    df: pd.DataFrame,
    feature_names: Optional[List[str]] = None,
) -> List[str]:
    """Select feature columns from a DataFrame, applying training-time rules.

    Args:
        df: DataFrame containing candidate columns
        feature_names: If provided (predict/analyze path), align to this list.

    Returns:
        Ordered list of feature column names.
    """
    feature_cols = [c for c in df.columns if c not in EXCLUDE_COLS]
    feature_cols = [c for c in feature_cols if c not in ALWAYS_EXCLUDE_FEATURES]
    feature_cols = [c for c in feature_cols if df[c].notna().sum() > 0]

    if feature_names:
        # Align to model's training-time feature order; fill missing with 0.
        present = [c for c in feature_names if c in feature_cols]
        missing = [c for c in feature_names if c not in feature_cols]
        for feat in missing:
            df[feat] = 0
        feature_cols = present + missing

    return feature_cols


def encode_features(X: pd.DataFrame) -> pd.DataFrame:
    """Encode a feature matrix in place using deterministic hash for categoricals.

    - Object/category columns → `stable_hash_int` per value (deterministic
      across train and predict, no fitting required).
    - Numeric columns → fill NaN with column median (or 0 if all-NaN).
    - Anything else → coerce to numeric, fall back to hash if it fails.
    - Final pass: coerce remainder to numeric and fill any leftover NaN with 0.

    Args:
        X: Feature matrix (will be copied internally).

    Returns:
        Encoded numeric DataFrame.
    """
    X = X.copy()
    for col in X.columns:
        if X[col].dtype == "object" or X[col].dtype.name == "category":
            X[col] = X[col].map(stable_hash_int)
        elif X[col].dtype.kind in "iuf":
            if X[col].notna().any():
                X[col] = X[col].fillna(X[col].median())
            else:
                X[col] = X[col].fillna(0)
        else:
            try:
                X[col] = pd.to_numeric(X[col], errors="coerce")
                if X[col].notna().any():
                    X[col] = X[col].fillna(X[col].median())
                else:
                    X[col] = X[col].fillna(0)
            except (ValueError, TypeError):
                X[col] = X[col].map(stable_hash_int)

    X = X.apply(pd.to_numeric, errors="coerce")
    nan_count = int(X.isna().sum().sum())
    if nan_count > 0:
        logger.debug(f"Filling {nan_count} residual NaN values with 0")
        X = X.fillna(0)

    return X


def prepare_features(
    df: pd.DataFrame,
    feature_names: Optional[List[str]] = None,
    target_column: str = "exam_score",
) -> Tuple[pd.DataFrame, Optional[pd.Series], List[str]]:
    """Prepare X (and y, if target column present) for train/predict/analyze.

    Args:
        df: Source DataFrame after merging + feature building.
        feature_names: Optional training-time feature order to enforce.
        target_column: Name of target column (default "exam_score").

    Returns:
        Tuple (X, y_or_None, feature_cols)
    """
    feature_cols = select_feature_columns(df, feature_names)
    X = df[feature_cols].copy()
    X = encode_features(X)
    if feature_names:
        X = X[feature_names]
        feature_cols = list(feature_names)

    y: Optional[pd.Series] = None
    if target_column in df.columns:
        y = df[target_column].loc[X.index].copy()

    return X, y, feature_cols
