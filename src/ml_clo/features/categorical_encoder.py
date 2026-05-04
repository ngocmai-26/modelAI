"""Categorical encoders for tree models (Phase 3 — advisor feedback #3).

The legacy ``stable_hash_int`` encoder maps categorical IDs (Subject_ID,
Lecturer_ID) to a 31-bit hash. Tree splits on a hash space have no
pedagogical meaning — two semantically-similar subjects end up at
arbitrary numeric distances. This module provides three structured
alternatives:

- ``LabelEncoderWrapper``: deterministic 1-1 mapping into ``[0, N-1]``.
  Smaller value range makes tree splits faster and more stable; the
  mapping is reversible (``inverse_transform``) and safe under unseen
  inputs (returns ``unknown_label`` instead of silently hashing into
  a random bucket).
- ``FrequencyEncoder``: replace category with its train-set frequency.
  Cheap, robust, but loses meaning when frequencies collide.
- ``TargetEncoder``: replace category with K-fold mean of the target
  to avoid leakage. Adds smoothing toward the global mean for rare
  categories. Falls back to global mean for unseen categories at
  predict time, which makes the encoder safe under the existing
  ``__UNKNOWN__`` handling.

All encoders are picklable (state lives in plain dicts/floats), so
they can be saved alongside the model in ``extra_metadata``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Optional

import numpy as np
import pandas as pd
from sklearn.model_selection import KFold

from ml_clo.utils.logger import get_logger

logger = get_logger(__name__)


def _key(value: object) -> str:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return "__MISSING__"
    s = str(value).strip()
    return s if s else "__MISSING__"


@dataclass
class LabelEncoderWrapper:
    """Deterministic label encoder mapping categories to ``[0, N-1]``.

    Compared to ``stable_hash_int`` (MD5 hashing), this encoder:
      * keeps the value range tight (``[0, N-1]``) so tree splits are
        faster and the model is less prone to overfitting on hash noise;
      * is reversible via ``inverse_transform`` — useful when explaining
        a SHAP attribution back to the original Subject_ID / Lecturer_ID;
      * handles unseen IDs at predict time explicitly via
        ``unknown_label`` instead of silently mapping them into a random
        bucket.

    Args:
        unknown_label: Integer code returned for categories that were
            never seen at fit time. Defaults to ``-1`` so unseen IDs are
            distinguishable from any trained bucket.
    """

    classes_: Dict[str, int] = field(default_factory=dict)
    unknown_label: int = -1

    def fit(self, X_col: pd.Series) -> "LabelEncoderWrapper":
        # Collect unique non-null keys, sort for determinism so the same
        # input data always produces the same integer codes regardless of
        # row order.
        keys = sorted(set(X_col.map(_key).tolist()))
        self.classes_ = {k: i for i, k in enumerate(keys)}
        return self

    def transform(self, X_col: pd.Series) -> pd.Series:
        return (
            X_col.map(_key)
            .map(self.classes_)
            .fillna(self.unknown_label)
            .astype("int64")
        )

    def fit_transform(
        self, X_col: pd.Series, y: Optional[pd.Series] = None
    ) -> pd.Series:
        self.fit(X_col)
        return self.transform(X_col)

    def inverse_transform(self, codes: pd.Series) -> pd.Series:
        """Reverse mapping for debugging / Feature Importance reports."""
        inverse = {i: k for k, i in self.classes_.items()}
        return codes.map(inverse).fillna("__UNKNOWN__")


@dataclass
class FrequencyEncoder:
    """Replace category with its frequency (count) in the fit data."""

    counts: Dict[str, int] = field(default_factory=dict)
    default: int = 0

    def fit(self, X_col: pd.Series) -> "FrequencyEncoder":
        keys = X_col.map(_key)
        vc = keys.value_counts()
        self.counts = vc.to_dict()
        self.default = 0
        return self

    def transform(self, X_col: pd.Series) -> pd.Series:
        return X_col.map(_key).map(self.counts).fillna(self.default).astype("int64")

    def fit_transform(self, X_col: pd.Series, y: Optional[pd.Series] = None) -> pd.Series:
        self.fit(X_col)
        return self.transform(X_col)


@dataclass
class TargetEncoder:
    """K-fold mean target encoder with additive smoothing.

    Args:
        n_folds: Number of folds for out-of-fold mean computation (default 5).
        smoothing: Additive smoothing constant ``m`` toward the global mean.
            ``encoded = (n*mean + m*global) / (n + m)`` per category. Higher
            values shrink rare categories toward the global mean to fight
            overfitting.
        random_state: Fold split seed (default 42).
    """

    n_folds: int = 5
    smoothing: float = 10.0
    random_state: int = 42
    means: Dict[str, float] = field(default_factory=dict)
    global_mean: float = 0.0

    def _smoothed_mean(self, sums: pd.Series, counts: pd.Series) -> Dict[str, float]:
        smoothed = (sums + self.global_mean * self.smoothing) / (counts + self.smoothing)
        return smoothed.to_dict()

    def fit(self, X_col: pd.Series, y: pd.Series) -> "TargetEncoder":
        if X_col is None or y is None or len(X_col) != len(y):
            raise ValueError("X_col and y must be same length")

        keys = X_col.map(_key).reset_index(drop=True)
        target = y.reset_index(drop=True).astype(float)

        self.global_mean = float(target.mean())

        grouped = target.groupby(keys)
        self.means = self._smoothed_mean(grouped.sum(), grouped.count())
        return self

    def transform(self, X_col: pd.Series) -> pd.Series:
        keys = X_col.map(_key)
        return keys.map(self.means).fillna(self.global_mean).astype(float)

    def fit_transform(self, X_col: pd.Series, y: pd.Series) -> pd.Series:
        """Fit + transform with K-fold leakage guard.

        Out-of-fold encoding: for each row, the target mean is computed
        from the *other* folds, so the row's own target never feeds its
        own encoded value. After K-fold encoding, ``self.means`` is set
        from the *full* fit so transform-time predictions remain stable.
        """
        if len(X_col) != len(y):
            raise ValueError("X_col and y must be same length")
        if len(X_col) < self.n_folds:
            return self.fit(X_col, y).transform(X_col)

        keys = X_col.map(_key).reset_index(drop=True)
        target = y.reset_index(drop=True).astype(float)
        self.global_mean = float(target.mean())

        encoded = pd.Series(np.full(len(keys), self.global_mean), dtype=float)
        kf = KFold(n_splits=self.n_folds, shuffle=True, random_state=self.random_state)

        for train_idx, val_idx in kf.split(keys):
            train_keys = keys.iloc[train_idx]
            train_target = target.iloc[train_idx]
            grouped = train_target.groupby(train_keys)
            fold_means = self._smoothed_mean(grouped.sum(), grouped.count())
            encoded.iloc[val_idx] = (
                keys.iloc[val_idx].map(fold_means).fillna(self.global_mean).values
            )

        # Final fit on full data — used by transform() at predict time
        full_grouped = target.groupby(keys)
        self.means = self._smoothed_mean(full_grouped.sum(), full_grouped.count())

        return encoded
