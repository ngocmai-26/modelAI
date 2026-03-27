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

    # Calculate features for each student
    student_features = []

    for student_id in df[student_id_column].unique():
        student_exams = exam_df[exam_df[student_id_column] == student_id].copy()

        if len(student_exams) == 0:
            # No exam history for this student
            student_features.append({
                student_id_column: student_id,
                "total_subjects": 0,
                "passed_subjects": 0,
                "pass_rate": 0.0,
                "avg_exam_score": np.nan,
                "median_exam_score": np.nan,
                "min_exam_score": np.nan,
                "recent_avg_score": np.nan,
                "recent_median_score": np.nan,
                "academic_core_score": np.nan,
                "min_exam_score_adj": np.nan,
                "improvement_trend": 0,
            })
            continue

        # Sort by time, then stable course keys (avoids arbitrary order within one year)
        sort_cols = [year_column]
        if "semester" in student_exams.columns:
            sort_cols.append("semester")
        for tie_col in ("Subject_ID", "Lecturer_ID"):
            if tie_col in student_exams.columns:
                sort_cols.append(tie_col)
        student_exams = student_exams.sort_values(sort_cols, kind="mergesort")

        # Basic statistics
        total_subjects = len(student_exams)
        passed_subjects = (student_exams["exam_score"] >= 3.0).sum()  # Pass threshold: 3.0 (equivalent to 5.0 in 10-point)
        pass_rate = passed_subjects / total_subjects if total_subjects > 0 else 0.0
        avg_exam_score = student_exams["exam_score"].mean()
        median_exam_score = student_exams["exam_score"].median()
        min_exam_score = student_exams["exam_score"].min()

        # Recent average (last N semesters or years)
        if "semester" in student_exams.columns:
            # Use semester-based
            recent_exams = student_exams.tail(recent_semesters)
        else:
            # Use year-based (assume 2 semesters per year)
            recent_exams = student_exams.tail(recent_semesters * 2)
        recent_avg_score = recent_exams["exam_score"].mean() if len(recent_exams) > 0 else np.nan
        recent_median_score = recent_exams["exam_score"].median() if len(recent_exams) > 0 else np.nan

        core_parts = [
            x for x in (median_exam_score, recent_avg_score, recent_median_score) if pd.notna(x)
        ]
        academic_core_score = float(np.mean(core_parts)) if core_parts else np.nan

        # Một môn điểm cực thấp không nên “neo” toàn bộ dự đoán môn mới khi nền trung vị/gần đây cao hơn
        floor_refs = [x for x in (median_exam_score, recent_median_score, recent_avg_score) if pd.notna(x)]
        if total_subjects >= 4 and floor_refs and pd.notna(min_exam_score):
            floor_ref = float(np.median(floor_refs))
            min_exam_score_adj = float(max(float(min_exam_score), floor_ref - 1.0))
        else:
            min_exam_score_adj = float(min_exam_score) if pd.notna(min_exam_score) else np.nan

        recent_n = recent_semesters if "semester" in student_exams.columns else recent_semesters * 2

        # Improvement trend: compare recent vs older performance
        if len(student_exams) >= 4:  # Need at least 4 records
            older_exams = student_exams.head(max(0, len(student_exams) - recent_n))
            older_avg = older_exams["exam_score"].mean()
            if pd.notna(recent_avg_score) and pd.notna(older_avg):
                if recent_avg_score > older_avg + 0.1:  # Threshold for improvement
                    improvement_trend = 1  # Improving
                elif recent_avg_score < older_avg - 0.1:  # Threshold for decline
                    improvement_trend = -1  # Declining
                else:
                    improvement_trend = 0  # Stable
            else:
                improvement_trend = 0
        else:
            improvement_trend = 0  # Not enough data

        student_features.append({
            student_id_column: student_id,
            "total_subjects": total_subjects,
            "passed_subjects": passed_subjects,
            "pass_rate": pass_rate,
            "avg_exam_score": avg_exam_score,
            "median_exam_score": median_exam_score,
            "min_exam_score": float(min_exam_score) if pd.notna(min_exam_score) else np.nan,
            "recent_avg_score": recent_avg_score,
            "recent_median_score": recent_median_score,
            "academic_core_score": academic_core_score,
            "min_exam_score_adj": min_exam_score_adj,
            "improvement_trend": improvement_trend,
        })

    features_df = pd.DataFrame(student_features)

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
    student_id_column: str = "Student_ID",
    year_column: str = "year",
) -> pd.DataFrame:
    """Build all aggregate features.

    This is a convenience function that builds all feature types:
    - Conduct features
    - Academic history features
    - Study hours features

    Args:
        df: Main DataFrame (merged training dataset)
        conduct_history_df: Historical conduct scores (optional)
        exam_history_df: Historical exam scores (optional, must be preprocessed)
        study_hours_df: Study hours DataFrame (optional)
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

    logger.info(f"Feature engineering complete: {len(df)} records, {len(df.columns)} columns")
    return df

