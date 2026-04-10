"""
Hash utilities for deterministic categorical encoding.

Why:
- Avoid fitting LabelEncoder on single-record inputs during prediction.
- Ensure the same mapping is used in both training and prediction.
"""

from __future__ import annotations

import hashlib
from typing import Any, Optional


def stable_hash_int(value: Any, mod: int = 2**31 - 1) -> int:
    """Convert a value into a deterministic integer via sha256.

    The default modulus is max int32 (~2.15B) to keep the hash space large
    enough that collisions are rare even for high-cardinality columns
    (~65k unique values still yields <0.1% birthday-paradox collision).
    Encoded values fit in a signed 32-bit integer for downstream
    compatibility with sklearn/numpy.

    Returns:
        -1 for missing/None/NaN-like values.
    """
    if value is None:
        return -1
    # pandas.NaN / numpy.nan
    try:
        # Avoid importing pandas here; rely on duck-typing.
        if value != value:  # NaN check
            return -1
    except Exception:
        pass

    s = str(value).strip()
    if s == "" or s.lower() == "nan":
        return -1

    h = hashlib.sha256(s.encode("utf-8")).hexdigest()
    return int(h[:16], 16) % mod

