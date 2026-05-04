"""Feature engineering and aggregation.

This module provides functions to build aggregate features from raw data,
such as average scores, trends, and historical statistics.
"""

from typing import Optional

import numpy as np
import pandas as pd

from ml_clo.utils.exceptions import DataValidationError
from ml_clo.utils.logger import get_logger

logger = get_logger(__name__)


def build_conduct_features(
    df: pd.DataFrame,
    conduct_history_df: pd.DataFrame,
    student_id_column: str = "Student_ID",
    year_column: str = "year",
) -> pd.DataFrame:
    """Build conduct score features from historical data.

    Creates:
    - avg_conduct_score: Average conduct score across all semesters
    - latest_conduct_score: Most recent conduct score
    - conduct_trend: Trend indicator (1=improving, 0=stable, -1=declining)

    Args:
        df: Main DataFrame (must have Student_ID, year)
        conduct_history_df: Historical conduct scores DataFrame
        student_id_column: Name of student ID column (default: "Student_ID")
        year_column: Name of year column (default: "year")

    Returns:
        DataFrame with conduct features added

    Raises:
        DataValidationError: If required columns are missing
    """
    logger.info("Building conduct score features")

    required_cols = [student_id_column, year_column]
    missing = [col for col in required_cols if col not in df.columns]
    if missing:
        raise DataValidationError(f"Missing columns in df: {missing}")

    df = df.copy()

    # Prepare conduct history: extract year from semester_year if needed
    conduct_df = conduct_history_df.copy()
    if "year" not in conduct_df.columns and "semester_year" in conduct_df.columns:
        conduct_df["year"] = conduct_df["semester_year"].astype(str).str.split("-").str[0]
        conduct_df["year"] = pd.to_numeric(conduct_df["year"], errors="coerce")

    # Ensure Student_ID and year are numeric
    conduct_df[student_id_column] = pd.to_numeric(conduct_df[student_id_column], errors="coerce")
    conduct_df["year"] = pd.to_numeric(conduct_df["year"], errors="coerce")

    # Calculate features for each student
    student_features = []

    for student_id in df[student_id_column].unique():
        student_conduct = conduct_df[conduct_df[student_id_column] == student_id].copy()

        if len(student_conduct) == 0:
            # No conduct data for this student
            student_features.append({
                student_id_column: student_id,
                "avg_conduct_score": np.nan,
                "latest_conduct_score": np.nan,
                "conduct_trend": np.nan,
            })
            continue

        # Sort by year and semester (if available)
        sort_cols = ["year"]
        if "semester" in student_conduct.columns:
            sort_cols.append("semester")
        student_conduct = student_conduct.sort_values(sort_cols)

        # Calculate features
        avg_score = student_conduct["conduct_score"].mean()
        latest_score = student_conduct["conduct_score"].iloc[-1]

        # Calculate trend: compare last 2 semesters if available
        if len(student_conduct) >= 2:
            last_two = student_conduct["conduct_score"].tail(2).values
            if last_two[1] > last_two[0]:
                trend = 1  # Improving
            elif last_two[1] < last_two[0]:
                trend = -1  # Declining
            else:
                trend = 0  # Stable
        else:
            trend = 0  # Not enough data

        student_features.append({
            student_id_column: student_id,
            "avg_conduct_score": avg_score,
            "latest_conduct_score": latest_score,
            "conduct_trend": trend,
        })

    features_df = pd.DataFrame(student_features)

    # Merge back to main DataFrame
    df = df.merge(
        features_df,
        on=student_id_column,
        how="left",
        suffixes=("", "_conduct_feat"),
    )

    logger.info(
        f"Built conduct features: "
        f"avg_conduct_score ({df['avg_conduct_score'].notna().sum()} non-null), "
        f"latest_conduct_score ({df['latest_conduct_score'].notna().sum()} non-null), "
        f"conduct_trend ({df['conduct_trend'].notna().sum()} non-null)"
    )

    return df


def build_academic_history_features(
    df: pd.DataFrame,
    exam_history_df: pd.DataFrame,
    student_id_column: str = "Student_ID",
    year_column: str = "year",
    current_year: Optional[int] = None,
    recent_semesters: int = 2,
) -> pd.DataFrame:
    """Build academic history features from historical exam scores.

    Creates:
    - total_subjects: Total number of subjects taken
    - passed_subjects: Number of subjects passed (score >= 3.0 in 6-point scale)
    - pass_rate: Ratio of passed to total subjects
    - avg_exam_score: Average exam score across all subjects
    - median_exam_score: Median exam score (robust to one very low/high course)
    - min_exam_score: Minimum course score (thô, dùng tham chiếu / hiển thị; không đưa vào vector train)
    - recent_avg_score: Average exam score in recent N semesters
    - recent_median_score: Median of the same recent window as recent_avg_score
    - academic_core_score: Trung bình median / recent_avg / recent_median (ít nhạy một môn lệch)
    - min_exam_score_adj: Điểm min điều chỉnh nền (≥4 môn: max(raw_min, median(nền)−1)) — dùng cho mô hình
    - improvement_trend: Trend indicator (1=improving, 0=stable, -1=declining)

    Args:
        df: Main DataFrame (must have Student_ID, year)
        exam_history_df: Historical exam scores DataFrame (must be preprocessed, scores in 0-6 scale)
        student_id_column: Name of student ID column (default: "Student_ID")
        year_column: Name of year column (default: "year")
        current_year: Current year to filter recent data. If None, use max year in df (default: None)
        recent_semesters: Number of recent semesters to consider for recent_avg_score (default: 2)

    Returns:
        DataFrame with academic history features added

    Raises:
        DataValidationError: If required columns are missing
    """
    logger.info("Building academic history features")

    required_cols = [student_id_column, year_column]
    missing = [col for col in required_cols if col not in df.columns]
    if missing:
        raise DataValidationError(f"Missing columns in df: {missing}")

    if "exam_score" not in exam_history_df.columns:
        raise DataValidationError("exam_history_df must have 'exam_score' column")

    df = df.copy()
    exam_df = exam_history_df.copy()

    # Determine current year if not provided
    if current_year is None:
        max_year = df[year_column].max()
        if pd.isna(max_year):
            # If all years are NaN, use a default or skip
            logger.warning("All year values are NaN, using default current_year=2024")
            current_year = 2024
        elif df[year_column].dtype in [np.int64, np.float64]:
            current_year = int(max_year) if not pd.isna(max_year) else 2024
        else:
            # Try to extract from string format
            current_year_str = str(max_year)
            if "-" in current_year_str:
                current_year = int(current_year_str.split("-")[0])
            else:
                try:
                    current_year = int(current_year_str)
                except (ValueError, TypeError):
                    logger.warning(f"Could not parse year '{current_year_str}', using default current_year=2024")
                    current_year = 2024

    # Ensure year is numeric
    if year_column in exam_df.columns:
        exam_df[year_column] = pd.to_numeric(exam_df[year_column], errors="coerce")

    # DESIGN-03: Vectorized academic-history aggregation. The previous
    # implementation looped per student and was O(N·S) — slow on 10k+
    # students. We now do one global sort + groupby ops.

    # Stable sort so tie-breakers (year, semester, Subject_ID, Lecturer_ID)
    # match the per-student version exactly.
    sort_cols = [student_id_column, year_column]
    if "semester" in exam_df.columns:
        sort_cols.append("semester")
    for tie_col in ("Subject_ID", "Lecturer_ID"):
        if tie_col in exam_df.columns:
            sort_cols.append(tie_col)
    exam_sorted = exam_df.sort_values(sort_cols, kind="mergesort").reset_index(drop=True)

    has_semester = "semester" in exam_sorted.columns
    recent_n = recent_semesters if has_semester else recent_semesters * 2

    grp = exam_sorted.groupby(student_id_column, sort=False)
    total = grp["exam_score"].size()
    passed = grp["exam_score"].apply(lambda s: int((s >= 3.0).sum()))
    avg_score = grp["exam_score"].mean()
    median_score = grp["exam_score"].median()
    min_score = grp["exam_score"].min()

    # Recent slice: tail(recent_n) per student.
    # Use cumcount from the END of each group to mark the last N rows.
    rev_idx = grp.cumcount(ascending=False)
    recent_mask = rev_idx < recent_n
    recent_df = exam_sorted[recent_mask]
    recent_grp = recent_df.groupby(student_id_column, sort=False)["exam_score"]
    recent_avg = recent_grp.mean()
    recent_median = recent_grp.median()

    # Older slice: everything except the last recent_n rows, but only when
    # the student has ≥4 total exams (matches the original guard).
    older_mask = (~recent_mask) & (exam_sorted[student_id_column].map(total) >= 4)
    older_df = exam_sorted[older_mask]
    older_avg = older_df.groupby(student_id_column, sort=False)["exam_score"].mean()

    features_df = pd.DataFrame({
        "total_subjects": total,
        "passed_subjects": passed,
        "avg_exam_score": avg_score,
        "median_exam_score": median_score,
        "min_exam_score": min_score,
        "recent_avg_score": recent_avg,
        "recent_median_score": recent_median,
        "_older_avg": older_avg,
    }).reset_index()

    features_df["pass_rate"] = np.where(
        features_df["total_subjects"] > 0,
        features_df["passed_subjects"] / features_df["total_subjects"].clip(lower=1),
        0.0,
    )

    # academic_core_score = mean of (median, recent_avg, recent_median),
    # ignoring NaNs. Same semantics as the per-student version.
    core_cols = features_df[["median_exam_score", "recent_avg_score", "recent_median_score"]]
    features_df["academic_core_score"] = core_cols.mean(axis=1, skipna=True)

    # min_exam_score_adj: floor the raw min by median(median, recent_median,
    # recent_avg) - 1.0, but only for students with ≥4 subjects.
    floor_cols = features_df[["median_exam_score", "recent_median_score", "recent_avg_score"]]
    floor_ref = floor_cols.median(axis=1, skipna=True)
    can_adjust = (features_df["total_subjects"] >= 4) & floor_ref.notna() & features_df["min_exam_score"].notna()
    adj_min = np.maximum(features_df["min_exam_score"], floor_ref - 1.0)
    features_df["min_exam_score_adj"] = np.where(
        can_adjust,
        adj_min,
        features_df["min_exam_score"],
    )

    # improvement_trend: requires ≥4 total subjects AND non-NaN recent + older avg.
    diff = features_df["recent_avg_score"] - features_df["_older_avg"]
    trend = np.zeros(len(features_df), dtype=int)
    eligible = (features_df["total_subjects"] >= 4) & features_df["recent_avg_score"].notna() & features_df["_older_avg"].notna()
    trend = np.where(eligible & (diff > 0.1), 1, trend)
    trend = np.where(eligible & (diff < -0.1), -1, trend)
    features_df["improvement_trend"] = trend
    features_df = features_df.drop(columns=["_older_avg"])

    # Reorder columns to match the original output exactly.
    features_df = features_df[[
        student_id_column,
        "total_subjects",
        "passed_subjects",
        "pass_rate",
        "avg_exam_score",
        "median_exam_score",
        "min_exam_score",
        "recent_avg_score",
        "recent_median_score",
        "academic_core_score",
        "min_exam_score_adj",
        "improvement_trend",
    ]]

    # Merge back to main DataFrame
    df = df.merge(
        features_df,
        on=student_id_column,
        how="left",
        suffixes=("", "_academic"),
    )

    logger.info(
        f"Built academic history features: "
        f"total_subjects ({df['total_subjects'].notna().sum()} non-null), "
        f"avg_exam_score ({df['avg_exam_score'].notna().sum()} non-null), "
        f"median_exam_score ({df['median_exam_score'].notna().sum()} non-null), "
        f"recent_avg_score ({df['recent_avg_score'].notna().sum()} non-null)"
    )

    return df


def build_study_hours_features(
    df: pd.DataFrame,
    study_hours_df: pd.DataFrame,
    student_id_column: str = "Student_ID",
    year_column: str = "year",
) -> pd.DataFrame:
    """Build study hours features.

    Creates:
    - study_hours_this_year: Study hours in current year
    - total_study_hours: Total study hours across all years

    Args:
        df: Main DataFrame (must have Student_ID, year)
        study_hours_df: Study hours DataFrame
        student_id_column: Name of student ID column (default: "Student_ID")
        year_column: Name of year column (default: "year")

    Returns:
        DataFrame with study hours features added

    Raises:
        DataValidationError: If required columns are missing
    """
    logger.info("Building study hours features")

    required_cols = [student_id_column, year_column]
    missing = [col for col in required_cols if col not in df.columns]
    if missing:
        raise DataValidationError(f"Missing columns in df: {missing}")

    df = df.copy()
    study_df = study_hours_df.copy()

    # Ensure year is numeric
    if year_column in study_df.columns:
        study_df[year_column] = pd.to_numeric(study_df[year_column], errors="coerce")
    df[year_column] = pd.to_numeric(df[year_column], errors="coerce")

    # Calculate total study hours per student
    total_hours = study_df.groupby(student_id_column)["accumulated_study_hours"].sum().reset_index()
    total_hours = total_hours.rename(columns={"accumulated_study_hours": "total_study_hours"})

    # Calculate study hours for current year (merge with df to get current year)
    # For each row in df, get study hours for that student-year
    study_this_year = study_df.groupby([student_id_column, year_column])["accumulated_study_hours"].sum().reset_index()
    study_this_year = study_this_year.rename(columns={"accumulated_study_hours": "study_hours_this_year"})

    # Merge total hours
    df = df.merge(total_hours, on=student_id_column, how="left", suffixes=("", "_total"))

    # Merge this year hours
    df = df.merge(study_this_year, on=[student_id_column, year_column], how="left", suffixes=("", "_year"))

    logger.info(
        f"Built study hours features: "
        f"total_study_hours ({df['total_study_hours'].notna().sum()} non-null), "
        f"study_hours_this_year ({df['study_hours_this_year'].notna().sum()} non-null)"
    )

    return df


def build_all_features(
    df: pd.DataFrame,
    conduct_history_df: Optional[pd.DataFrame] = None,
    exam_history_df: Optional[pd.DataFrame] = None,
    study_hours_df: Optional[pd.DataFrame] = None,
    attendance_history_df: Optional[pd.DataFrame] = None,
    student_id_column: str = "Student_ID",
    year_column: str = "year",
) -> pd.DataFrame:
    """Build all aggregate features.

    This is a convenience function that builds all feature types:
    - Conduct features
    - Academic history features
    - Study hours features
    - Temporal attendance features (slope/volatility per student-year)

    Args:
        df: Main DataFrame (merged training dataset)
        conduct_history_df: Historical conduct scores (optional)
        exam_history_df: Historical exam scores (optional, must be preprocessed)
        study_hours_df: Study hours DataFrame (optional)
        attendance_history_df: Raw attendance frame (optional). When supplied,
            temporal features (slope, volatility, late streak, early-dropoff)
            are derived per (student, year) and merged onto ``df``.
        student_id_column: Name of student ID column (default: "Student_ID")
        year_column: Name of year column (default: "year")

    Returns:
        DataFrame with all features added
    """
    logger.info("Building all aggregate features")

    df = df.copy()

    # Build conduct features
    if conduct_history_df is not None:
        df = build_conduct_features(df, conduct_history_df, student_id_column, year_column)

    # Build academic history features
    if exam_history_df is not None:
        df = build_academic_history_features(df, exam_history_df, student_id_column, year_column)

    # Build study hours features
    if study_hours_df is not None:
        df = build_study_hours_features(df, study_hours_df, student_id_column, year_column)

    # Build temporal attendance features (Phase 2 — advisor feedback #1)
    if attendance_history_df is not None:
        from ml_clo.features.temporal_features import (
            build_temporal_attendance_features,
            merge_temporal_attendance_features,
        )
        temporal = build_temporal_attendance_features(attendance_history_df)
        df = merge_temporal_attendance_features(df, temporal, year_column=year_column)

    logger.info(f"Feature engineering complete: {len(df)} records, {len(df.columns)} columns")
    return df

