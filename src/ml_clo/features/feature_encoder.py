"""Shared feature preparation/encoding logic for train, predict, and analysis pipelines.

DESIGN-02: Centralizes the feature-selection + categorical-encoding logic
that was previously duplicated across `train_pipeline.prepare_features`,
`predict_pipeline.prepare_features`, and `analysis_pipeline.prepare_features`.
Any change to feature exclusion rules or encoding strategy now lives in
exactly one place.
"""

from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from ml_clo.utils.hash_utils import stable_hash_int
from ml_clo.utils.logger import get_logger

logger = get_logger(__name__)

# Columns that are never used as model features.
DEFAULT_EXCLUDE_COLS = ("Student_ID", "Subject_ID", "Lecturer_ID", "exam_score", "year")
# IDs that may optionally be included as features (Phase 3 ablation).
ID_FEATURE_COLS = ("Subject_ID", "Lecturer_ID")
# Features explicitly excluded for stability or to prevent target leakage.
#
# Why each column is dropped:
#   - ``min_exam_score``       : raw min score is too noisy; the model uses
#                                ``min_exam_score_adj`` + ``academic_core_score``.
#   - ``summary_score``        : DiemTong column derived from the same exam
#                                grading as the target (corr ≈ 0.9 with
#                                ``exam_score``) — direct target leakage.
#   - ``letter_system``        : letter grade (A/B/C/D) is a deterministic
#                                bucketing of ``exam_score``.
#   - ``Passed_the_module``    : Pass/Fail flag computed from the exam score.
#
# These three columns are unambiguous leakage sources (they encode the
# exam outcome directly). They were previously masked by MD5 hashing —
# the hash spread values across ``[0, 10^9)`` so trees could not learn an
# effective split — but with ``LabelEncoderWrapper`` mapping into
# ``[0, N-1]`` the leakage becomes learnable and inflates Test R² above
# 0.98. Dropping them is the principled fix regardless of encoder.
#
# Other potentially noisy columns (Test, L2, Library, Birthdate,
# Subject_Name, Lecturer_Name, ...) are intentionally left in the
# feature set: their correlation with the target is weak (|r| < 0.5),
# they carry residual pedagogical or contextual signal, and excluding
# them all degrades Test R² substantially. The encoder still controls
# how they are turned into integers (LabelEncoder vs hash).
ALWAYS_EXCLUDE_FEATURES = (
    "min_exam_score",
    "summary_score",
    "letter_system",
    "Passed_the_module",
)


def select_feature_columns(
    df: pd.DataFrame,
    feature_names: Optional[List[str]] = None,
    include_id_features: bool = False,
) -> List[str]:
    """Select feature columns from a DataFrame, applying training-time rules.

    Args:
        df: DataFrame containing candidate columns
        feature_names: If provided (predict/analyze path), align to this list.
        include_id_features: When True, ``Subject_ID``/``Lecturer_ID`` are kept
            as features so a categorical encoder (target/frequency/hash) can
            replace their raw values. Default False keeps existing behaviour.

    Returns:
        Ordered list of feature column names.
    """
    exclude = set(DEFAULT_EXCLUDE_COLS)
    if include_id_features:
        exclude -= set(ID_FEATURE_COLS)

    feature_cols = [c for c in df.columns if c not in exclude]
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


def encode_features(
    X: pd.DataFrame,
    categorical_strategy: str = "label",
    fitted_encoders: Optional[Dict[str, Any]] = None,
    y: Optional[pd.Series] = None,
    fit: bool = True,
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """Encode a feature matrix using the requested categorical strategy.

    Args:
        X: Feature matrix (will be copied internally).
        categorical_strategy: Encoder for ``ID_FEATURE_COLS`` (Subject_ID,
            Lecturer_ID): one of ``"label"`` (default — deterministic
            ``[0, N-1]`` mapping via ``LabelEncoderWrapper``), ``"hash"``
            (legacy MD5 hashing), ``"frequency"``, or ``"target"``.
            Other object/category columns are also encoded using
            ``LabelEncoderWrapper`` when strategy is ``"label"`` (so a
            saved model can deterministically reproduce the encoding at
            predict time); they fall back to ``stable_hash_int`` only
            when strategy is ``"hash"``.
        fitted_encoders: When non-None and ``fit=False``, reuse these fitted
            encoders for transform-only mode (predict/analyse path).
        y: Target vector — required when fitting target encoder.
        fit: When True, fit a fresh encoder on this data; when False, reuse
            ``fitted_encoders``.

    Returns:
        Tuple of (encoded DataFrame, encoders dict to persist alongside the
        model).
    """
    from ml_clo.features.categorical_encoder import (
        FrequencyEncoder,
        LabelEncoderWrapper,
        TargetEncoder,
    )

    X = X.copy()
    encoders: Dict[str, Any] = (
        dict(fitted_encoders) if (fitted_encoders and not fit) else {}
    )

    use_label_for_other_object = categorical_strategy == "label"

    for col in X.columns:
        is_object = X[col].dtype == "object" or X[col].dtype.name == "category"
        is_id_col = col in ID_FEATURE_COLS

        if is_id_col and categorical_strategy != "hash":
            if fit:
                if categorical_strategy == "label":
                    enc = LabelEncoderWrapper()
                    X[col] = enc.fit_transform(X[col])
                elif categorical_strategy == "frequency":
                    enc = FrequencyEncoder()
                    X[col] = enc.fit_transform(X[col])
                elif categorical_strategy == "target":
                    if y is None:
                        raise ValueError(
                            "Target encoder requires y; pass it to encode_features"
                        )
                    enc = TargetEncoder()
                    X[col] = enc.fit_transform(X[col], y)
                else:
                    raise ValueError(
                        f"Unknown categorical_strategy: {categorical_strategy}"
                    )
                encoders[col] = enc
            else:
                enc = encoders.get(col)
                if enc is None:
                    # No fitted encoder available — fall back to hash for safety.
                    X[col] = X[col].map(stable_hash_int)
                else:
                    X[col] = enc.transform(X[col])
            continue

        if is_object:
            if use_label_for_other_object:
                if fit:
                    enc = LabelEncoderWrapper()
                    X[col] = enc.fit_transform(X[col])
                    encoders[col] = enc
                else:
                    enc = encoders.get(col)
                    if enc is None:
                        # Backward-compat: a model trained before label
                        # encoding may not carry an encoder for this column.
                        X[col] = X[col].map(stable_hash_int)
                    else:
                        X[col] = enc.transform(X[col])
            else:
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
                if use_label_for_other_object:
                    if fit:
                        enc = LabelEncoderWrapper()
                        X[col] = enc.fit_transform(X[col])
                        encoders[col] = enc
                    else:
                        enc = encoders.get(col)
                        if enc is None:
                            X[col] = X[col].map(stable_hash_int)
                        else:
                            X[col] = enc.transform(X[col])
                else:
                    X[col] = X[col].map(stable_hash_int)

    X = X.apply(pd.to_numeric, errors="coerce")
    nan_count = int(X.isna().sum().sum())
    if nan_count > 0:
        logger.debug(f"Filling {nan_count} residual NaN values with 0")
        X = X.fillna(0)

    return X, encoders


def prepare_features(
    df: pd.DataFrame,
    feature_names: Optional[List[str]] = None,
    target_column: str = "exam_score",
    categorical_strategy: str = "label",
    include_id_features: bool = False,
    fitted_encoders: Optional[Dict[str, Any]] = None,
    fit: bool = True,
) -> Tuple[pd.DataFrame, Optional[pd.Series], List[str], Dict[str, Any]]:
    """Prepare X (and y, if target column present) for train/predict/analyze.

    Args:
        df: Source DataFrame after merging + feature building.
        feature_names: Optional training-time feature order to enforce.
        target_column: Name of target column (default "exam_score").
        categorical_strategy: Encoder for ID columns (``"label"`` default,
            ``"hash"``, ``"frequency"``, or ``"target"``).
        include_id_features: When True, ``Subject_ID``/``Lecturer_ID`` are
            kept and encoded; required for non-hash strategies to have effect.
        fitted_encoders: Existing encoders for transform-only mode.
        fit: When True, fit fresh encoders; when False, reuse fitted ones.

    Returns:
        Tuple ``(X, y_or_None, feature_cols, encoders)``.
    """
    feature_cols = select_feature_columns(
        df, feature_names, include_id_features=include_id_features
    )
    X = df[feature_cols].copy()

    y: Optional[pd.Series] = None
    if target_column in df.columns:
        y = df[target_column].loc[X.index].copy()

    X, encoders = encode_features(
        X,
        categorical_strategy=categorical_strategy,
        fitted_encoders=fitted_encoders,
        y=y,
        fit=fit,
    )

    if feature_names:
        X = X[feature_names]
        feature_cols = list(feature_names)

    return X, y, feature_cols, encoders
