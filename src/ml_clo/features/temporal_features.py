"""Temporal feature engineering from attendance time series.

Phase 2 — advisor feedback #1.

Derives per-(student, year) trend features from raw attendance records:
- ``attendance_slope_3w``  : OLS slope of last 3 weekly rates
- ``attendance_slope_full``: OLS slope across the full semester
- ``attendance_volatility``: std of weekly rates
- ``late_streak_max``      : longest run of weeks with rate < 0.7
- ``early_dropoff_flag``   : 1 if second-half mean is > 0.3 below first-half
- ``num_weeks_observed``   : number of distinct weeks present (proxy for data density)

All output features are vectorized numpy operations — no deep learning,
no external time-series libraries — which keeps inference cost matching
the existing ensemble pipeline.

Self-study temporal features are intentionally NOT computed here: the
``tuhoc.xlsx`` source only has (year, semester) granularity — no weekly
timestamps — so weekly variance of self-study hours is data-unavailable.
We capture what is feasible (cross-semester variance) inside
``feature_builder.build_study_hours_features`` instead.
"""

from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd

from ml_clo.utils.logger import get_logger

logger = get_logger(__name__)


# Map ``Điểm danh`` text → numeric attendance score in [0, 1]
_ATTENDANCE_SCORE = {
    "Có": 1.0, "Có mặt": 1.0,
    "Sớm": 1.0,
    "Trễ": 0.5, "trễ": 0.5,
    "Phép": 0.7,
    "Vắng": 0.0,
}


def _row_score(value: object) -> float:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return np.nan
    return _ATTENDANCE_SCORE.get(str(value).strip(), np.nan)


def _max_streak_below(rates: np.ndarray, threshold: float) -> int:
    cur = best = 0
    for r in rates:
        if r < threshold:
            cur += 1
            best = max(best, cur)
        else:
            cur = 0
    return best


def _slope(weeks: np.ndarray, rates: np.ndarray) -> float:
    if len(rates) < 2 or np.all(weeks == weeks[0]):
        return 0.0
    try:
        return float(np.polyfit(weeks, rates, 1)[0])
    except (np.linalg.LinAlgError, ValueError):
        return 0.0


def build_temporal_attendance_features(
    attendance_df: pd.DataFrame,
    student_id_col: str = "MSSV",
    date_col: str = "Ngày học",
    status_col: str = "Điểm danh",
    year_col: str = "Năm học",
) -> pd.DataFrame:
    """Compute weekly trend features per (student, year).

    Args:
        attendance_df: Raw attendance frame with one row per session.
        student_id_col: Student identifier column.
        date_col: Datetime column for session date.
        status_col: Categorical attendance status column (Sớm/Có/Trễ/Vắng/...).
        year_col: Academic-year column (e.g. ``"2024-2025"``).

    Returns:
        DataFrame with columns ``Student_ID``, ``year``,
        and the temporal feature columns.
    """
    if attendance_df is None or len(attendance_df) == 0:
        return pd.DataFrame(
            columns=[
                "Student_ID", "year",
                "attendance_slope_3w", "attendance_slope_full",
                "attendance_volatility", "late_streak_max",
                "early_dropoff_flag", "num_weeks_observed",
            ]
        )

    df = attendance_df.copy()

    # Score each session
    df["_score"] = df[status_col].apply(_row_score)
    df = df.dropna(subset=["_score", date_col, student_id_col])

    # Standardise Student_ID — preserve numeric when possible, otherwise keep as-is.
    sid_numeric = pd.to_numeric(df[student_id_col], errors="coerce")
    if sid_numeric.notna().all():
        df["Student_ID"] = sid_numeric.astype("int64")
    else:
        df["Student_ID"] = df[student_id_col].astype(str)

    # Year as 4-digit int (e.g. "2024-2025" → 2024)
    df["year"] = (
        df[year_col].astype(str).str.extract(r"(\d{4})", expand=False)
    )
    df["year"] = pd.to_numeric(df["year"], errors="coerce")
    df = df.dropna(subset=["year"])
    df["year"] = df["year"].astype("int64")

    # ISO week within calendar — same week semantics across rows
    df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
    df = df.dropna(subset=[date_col])

    # Use day-relative week number to keep weeks contiguous within a year
    df["_week_idx"] = (
        (df[date_col] - df.groupby(["Student_ID", "year"])[date_col].transform("min"))
        .dt.days // 7
    ).astype("int64")

    # Aggregate to weekly attendance rate
    weekly = (
        df.groupby(["Student_ID", "year", "_week_idx"], as_index=False)["_score"]
        .mean()
        .rename(columns={"_score": "weekly_rate"})
    )

    # Compute trend features per (student, year)
    rows = []
    for (sid, year), grp in weekly.groupby(["Student_ID", "year"], sort=False):
        grp = grp.sort_values("_week_idx")
        weeks = grp["_week_idx"].to_numpy(dtype=float)
        rates = grp["weekly_rate"].to_numpy(dtype=float)
        n = len(rates)

        slope_3w = _slope(weeks[-3:], rates[-3:]) if n >= 2 else 0.0
        slope_full = _slope(weeks, rates) if n >= 2 else 0.0
        volatility = float(np.std(rates)) if n >= 2 else 0.0
        late_streak = _max_streak_below(rates, 0.7)
        if n >= 4:
            half = n // 2
            dropoff = 1 if (rates[:half].mean() - rates[half:].mean()) > 0.3 else 0
        else:
            dropoff = 0

        rows.append({
            "Student_ID": sid,
            "year": year,
            "attendance_slope_3w": slope_3w,
            "attendance_slope_full": slope_full,
            "attendance_volatility": volatility,
            "late_streak_max": int(late_streak),
            "early_dropoff_flag": int(dropoff),
            "num_weeks_observed": int(n),
        })

    out = pd.DataFrame(rows)
    if len(out) == 0:
        # Ensure consistent schema even with empty result
        return pd.DataFrame(
            columns=[
                "Student_ID", "year",
                "attendance_slope_3w", "attendance_slope_full",
                "attendance_volatility", "late_streak_max",
                "early_dropoff_flag", "num_weeks_observed",
            ]
        )
    logger.info(
        f"Built temporal attendance features: {len(out)} (student, year) groups, "
        f"{out['attendance_slope_3w'].abs().mean():.4f} mean |slope_3w|"
    )
    return out


def merge_temporal_attendance_features(
    df: pd.DataFrame,
    temporal_df: Optional[pd.DataFrame],
    year_column: str = "year",
) -> pd.DataFrame:
    """Left-join temporal features onto the main training frame."""
    if temporal_df is None or len(temporal_df) == 0:
        return df

    base = df.copy()
    temp = temporal_df.copy()

    # Match dtypes on keys
    if pd.api.types.is_numeric_dtype(base["Student_ID"]):
        temp["Student_ID"] = pd.to_numeric(temp["Student_ID"], errors="coerce")
        temp = temp.dropna(subset=["Student_ID"]).astype({"Student_ID": base["Student_ID"].dtype})

    if year_column in base.columns:
        base[year_column] = pd.to_numeric(base[year_column], errors="coerce")
        temp[year_column] = pd.to_numeric(temp[year_column], errors="coerce")

    merged = base.merge(
        temp,
        on=["Student_ID", year_column],
        how="left",
        suffixes=("", "_temporal"),
    )
    return merged
