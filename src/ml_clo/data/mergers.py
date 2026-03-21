"""Data merging functions.

This module provides functions to merge multiple data sources into a single
training dataset. All merges are based on Student_ID, Subject_ID, and year.

Yêu cầu mới: Hỗ trợ tạo record "ảo" từ (student_id, subject_id, lecturer_id)
khi không có trong DiemTong — dùng nhân khẩu, PPGD/PPDG làm nguồn chính.
"""

from typing import Optional

import numpy as np
import pandas as pd

from ml_clo.utils.exceptions import DataValidationError
from ml_clo.utils.logger import get_logger

logger = get_logger(__name__)

LECTURER_PLACEHOLDER = "__UNKNOWN__"


def student_has_history(exam_df: pd.DataFrame, student_id: str) -> bool:
    """Kiểm tra sinh viên có lịch sử trong DiemTong (đã học ít nhất 1 môn).

    SV năm 2+: có ít nhất 1 bản ghi trong DiemTong.
    SV năm 1: không có bản ghi nào.

    Args:
        exam_df: DataFrame điểm thi (đã preprocess, có cột Student_ID)
        student_id: Mã sinh viên (str hoặc int)

    Returns:
        True nếu sinh viên có trong exam_df, False nếu không
    """
    if exam_df is None or exam_df.empty or "Student_ID" not in exam_df.columns:
        return False
    _sid = int(student_id) if isinstance(student_id, str) and str(student_id).isdigit() else student_id
    return (exam_df["Student_ID"] == _sid).any()


def create_student_record_from_ids(
    student_id: str,
    subject_id: str,
    lecturer_id: str,
    demographics_df: Optional[pd.DataFrame] = None,
    teaching_methods_df: Optional[pd.DataFrame] = None,
    assessment_methods_df: Optional[pd.DataFrame] = None,
    year: Optional[float] = None,
) -> pd.DataFrame:
    """Tạo 1 record "ảo" cho (student_id, subject_id, lecturer_id) từ nhân khẩu, PPGD/PPDG.

    Dùng khi sinh viên/môn/GV chưa có trong DiemTong (yêu cầu mới: không bắt buộc).

    Args:
        student_id: Mã sinh viên
        subject_id: Mã môn học
        lecturer_id: Mã giảng viên (có thể mới, không có file)
        demographics_df: Nhân khẩu (nhankhau.xlsx)
        teaching_methods_df: PPGD (đã encode TM)
        assessment_methods_df: PPDG (đã encode EM)
        year: Năm học (optional, mặc định 2024)

    Returns:
        DataFrame 1 dòng với Student_ID, Subject_ID, Lecturer_ID, year, exam_score (NaN),
        và các cột từ demographics, TM, EM (nếu có).
    """
    _sid = int(student_id) if isinstance(student_id, str) and str(student_id).isdigit() else student_id
    _subj = str(subject_id).strip()
    _lec = str(lecturer_id).strip() if lecturer_id else LECTURER_PLACEHOLDER
    _year = year if year is not None else 2024

    base = pd.DataFrame([{
        "Student_ID": _sid,
        "Subject_ID": _subj,
        "Lecturer_ID": _lec,
        "year": _year,
        "exam_score": np.nan,
    }])

    if demographics_df is not None and "Student_ID" in demographics_df.columns:
        demo_copy = demographics_df.copy()
        demo_copy["Student_ID"] = pd.to_numeric(demo_copy["Student_ID"], errors="coerce")
        base = merge_demographics(base, demo_copy)

    if teaching_methods_df is not None and "Subject_ID" in teaching_methods_df.columns:
        base = merge_teaching_methods(base, teaching_methods_df)

    if assessment_methods_df is not None and "Subject_ID" in assessment_methods_df.columns:
        base = merge_assessment_methods(base, assessment_methods_df)

    # Thiếu TM/EM sẽ được prepare_features bù bằng 0 khi predict

    logger.info(
        f"Created virtual record: student_id={student_id}, subject_id={subject_id}, "
        f"lecturer_id={lecturer_id or 'placeholder'}"
    )
    return base


def merge_exam_and_conduct_scores(
    exam_df: pd.DataFrame,
    conduct_df: pd.DataFrame,
    year_column: str = "year",
) -> pd.DataFrame:
    """Merge exam scores with conduct scores.

    Merges on Student_ID and year. Conduct scores are aggregated if multiple
    records exist for the same student-year combination.

    Args:
        exam_df: Exam scores DataFrame (must have Student_ID, year)
        conduct_df: Conduct scores DataFrame (must have Student_ID, semester_year or year)
        year_column: Name of year column in exam_df (default: "year")

    Returns:
        Merged DataFrame with exam and conduct scores

    Raises:
        DataValidationError: If required columns are missing
    """
    logger.info("Merging exam scores with conduct scores")

    # Validate required columns
    required_exam_cols = ["Student_ID", year_column]
    missing = [col for col in required_exam_cols if col not in exam_df.columns]
    if missing:
        raise DataValidationError(f"Missing columns in exam_df: {missing}")

    # Prepare conduct scores: extract year from semester_year if needed
    conduct_df = conduct_df.copy()
    if "year" not in conduct_df.columns and "semester_year" in conduct_df.columns:
        # Extract start year from semester_year (e.g., "2021-2022" -> 2021)
        conduct_df["year"] = conduct_df["semester_year"].astype(str).str.split("-").str[0]
        conduct_df["year"] = pd.to_numeric(conduct_df["year"], errors="coerce")

    if "year" not in conduct_df.columns:
        raise DataValidationError("Conduct scores must have 'year' or 'semester_year' column")

    # Ensure year columns have same type before merging
    # Convert both to numeric (int) for consistency
    exam_df = exam_df.copy()
    if year_column in exam_df.columns:
        exam_df[year_column] = pd.to_numeric(exam_df[year_column], errors="coerce")
    conduct_df["year"] = pd.to_numeric(conduct_df["year"], errors="coerce")

    # Aggregate conduct scores by Student_ID and year
    # Take average conduct_score if multiple records exist
    conduct_agg = conduct_df.groupby(["Student_ID", "year"]).agg({
        "conduct_score": "mean",  # Average if multiple semesters
    }).reset_index()

    # Rename year column in conduct_agg to match exam_df if needed
    if year_column != "year":
        conduct_agg = conduct_agg.rename(columns={"year": year_column})

    # Merge
    merged = exam_df.merge(
        conduct_agg,
        on=["Student_ID", year_column],
        how="left",
        suffixes=("", "_conduct"),
    )

    logger.info(
        f"Merged exam and conduct scores: {len(merged)} records "
        f"({merged['conduct_score'].notna().sum()} with conduct scores)"
    )

    return merged


def merge_demographics(
    df: pd.DataFrame,
    demographics_df: pd.DataFrame,
) -> pd.DataFrame:
    """Merge demographics data.

    Merges on Student_ID only (demographics are student-level, not subject-level).

    Args:
        df: Main DataFrame (must have Student_ID)
        demographics_df: Demographics DataFrame (must have Student_ID)

    Returns:
        Merged DataFrame with demographics

    Raises:
        DataValidationError: If Student_ID is missing
    """
    logger.info("Merging demographics data")

    if "Student_ID" not in df.columns:
        raise DataValidationError("Main DataFrame must have 'Student_ID' column")
    if "Student_ID" not in demographics_df.columns:
        raise DataValidationError("Demographics DataFrame must have 'Student_ID' column")

    # Select only relevant demographic columns (avoid duplicates)
    demo_cols = ["Student_ID"]
    # Add common demographic columns if they exist
    for col in ["Gender", "place_of_birth", "Ethnicity", "Religion", "birth_place_region"]:
        if col in demographics_df.columns:
            demo_cols.append(col)

    demographics_subset = demographics_df[demo_cols].drop_duplicates(subset=["Student_ID"])

    # Merge
    merged = df.merge(
        demographics_subset,
        on="Student_ID",
        how="left",
        suffixes=("", "_demo"),
    )

    logger.info(
        f"Merged demographics: {len(merged)} records "
        f"({merged[demo_cols[1] if len(demo_cols) > 1 else 'Student_ID'].notna().sum() if len(demo_cols) > 1 else len(merged)} with demographics)"
    )

    return merged


def merge_teaching_methods(
    df: pd.DataFrame,
    teaching_methods_df: pd.DataFrame,
) -> pd.DataFrame:
    """Merge teaching methods data.

    Merges on Subject_ID. Teaching methods are subject-level, not student-level.

    Args:
        df: Main DataFrame (must have Subject_ID)
        teaching_methods_df: Teaching methods DataFrame (must have Subject_ID)

    Returns:
        Merged DataFrame with teaching methods

    Raises:
        DataValidationError: If Subject_ID is missing
    """
    logger.info("Merging teaching methods data")

    if "Subject_ID" not in df.columns:
        raise DataValidationError("Main DataFrame must have 'Subject_ID' column")
    if "Subject_ID" not in teaching_methods_df.columns:
        raise DataValidationError("Teaching methods DataFrame must have 'Subject_ID' column")

    # Select teaching method columns (TM*)
    tm_cols = ["Subject_ID"] + [col for col in teaching_methods_df.columns if "TM" in col or col.startswith("TM")]

    teaching_subset = teaching_methods_df[tm_cols].drop_duplicates(subset=["Subject_ID"])

    # Merge
    merged = df.merge(
        teaching_subset,
        on="Subject_ID",
        how="left",
        suffixes=("", "_tm"),
    )

    tm_cols_count = len([col for col in tm_cols if col != "Subject_ID"])
    logger.info(
        f"Merged teaching methods: {len(merged)} records "
        f"({merged[tm_cols[1] if len(tm_cols) > 1 else 'Subject_ID'].notna().sum() if len(tm_cols) > 1 else len(merged)} with teaching methods, {tm_cols_count} TM columns)"
    )

    return merged


def merge_assessment_methods(
    df: pd.DataFrame,
    assessment_methods_df: pd.DataFrame,
) -> pd.DataFrame:
    """Merge assessment methods data.

    Merges on Subject_ID. Assessment methods are subject-level, not student-level.

    Args:
        df: Main DataFrame (must have Subject_ID)
        assessment_methods_df: Assessment methods DataFrame (must have Subject_ID)

    Returns:
        Merged DataFrame with assessment methods

    Raises:
        DataValidationError: If Subject_ID is missing
    """
    logger.info("Merging assessment methods data")

    if "Subject_ID" not in df.columns:
        raise DataValidationError("Main DataFrame must have 'Subject_ID' column")
    if "Subject_ID" not in assessment_methods_df.columns:
        raise DataValidationError("Assessment methods DataFrame must have 'Subject_ID' column")

    # Select assessment method columns (EM*)
    em_cols = ["Subject_ID"] + [col for col in assessment_methods_df.columns if "EM" in col or col.startswith("EM")]

    assessment_subset = assessment_methods_df[em_cols].drop_duplicates(subset=["Subject_ID"])

    # Merge
    merged = df.merge(
        assessment_subset,
        on="Subject_ID",
        how="left",
        suffixes=("", "_em"),
    )

    em_cols_count = len([col for col in em_cols if col != "Subject_ID"])
    logger.info(
        f"Merged assessment methods: {len(merged)} records "
        f"({merged[em_cols[1] if len(em_cols) > 1 else 'Subject_ID'].notna().sum() if len(em_cols) > 1 else len(merged)} with assessment methods, {em_cols_count} EM columns)"
    )

    return merged


def merge_study_hours(
    df: pd.DataFrame,
    study_hours_df: pd.DataFrame,
    year_column: str = "year",
) -> pd.DataFrame:
    """Merge study hours data.

    Merges on Student_ID and year. Study hours are aggregated if multiple
    records exist for the same student-year combination.

    Args:
        df: Main DataFrame (must have Student_ID, year)
        study_hours_df: Study hours DataFrame (must have Student_ID, year)
        year_column: Name of year column (default: "year")

    Returns:
        Merged DataFrame with study hours

    Raises:
        DataValidationError: If required columns are missing
    """
    logger.info("Merging study hours data")

    required_cols = ["Student_ID", year_column]
    missing = [col for col in required_cols if col not in df.columns]
    if missing:
        raise DataValidationError(f"Missing columns in main DataFrame: {missing}")

    if "Student_ID" not in study_hours_df.columns or "year" not in study_hours_df.columns:
        raise DataValidationError("Study hours DataFrame must have 'Student_ID' and 'year' columns")

    # Ensure year columns have same type before merging
    df = df.copy()
    if year_column in df.columns:
        df[year_column] = pd.to_numeric(df[year_column], errors="coerce")
    study_hours_df["year"] = pd.to_numeric(study_hours_df["year"], errors="coerce")

    # Aggregate study hours by Student_ID and year
    # Sum accumulated_study_hours if multiple semesters
    study_agg = study_hours_df.groupby(["Student_ID", "year"]).agg({
        "accumulated_study_hours": "sum",  # Sum across semesters
    }).reset_index()

    # Rename to avoid conflicts and match column name
    study_agg = study_agg.rename(columns={
        "accumulated_study_hours": "total_study_hours",
        "year": year_column if year_column != "year" else "year"
    })

    # Merge
    merged = df.merge(
        study_agg,
        on=["Student_ID", year_column],
        how="left",
        suffixes=("", "_study"),
    )

    logger.info(
        f"Merged study hours: {len(merged)} records "
        f"({merged['total_study_hours'].notna().sum()} with study hours)"
    )

    return merged


def merge_attendance(
    df: pd.DataFrame,
    attendance_df: pd.DataFrame,
    year_column: str = "year",
) -> pd.DataFrame:
    """Merge attendance data.

    Merges on Student_ID, Subject_ID, and year. Calculates attendance rate
    from attendance records.

    Args:
        df: Main DataFrame (must have Student_ID, Subject_ID, year)
        attendance_df: Attendance DataFrame (must have MSSV, Mã môn học, Niên khoá)
        year_column: Name of year column in main DataFrame (default: "year")

    Returns:
        Merged DataFrame with attendance rate

    Raises:
        DataValidationError: If required columns are missing
    """
    logger.info("Merging attendance data")

    required_cols = ["Student_ID", "Subject_ID", year_column]
    missing = [col for col in required_cols if col not in df.columns]
    if missing:
        raise DataValidationError(f"Missing columns in main DataFrame: {missing}")

    # Map Vietnamese column names to standard names
    attendance_df = attendance_df.copy()
    # Primary mapping; fallbacks for files that use different column names
    column_mapping = {
        "MSSV": "Student_ID",
        "Mã môn học": "Subject_ID",
        "Niên khoá": "year",
    }
    for vn_col, std_col in column_mapping.items():
        if vn_col in attendance_df.columns and std_col not in attendance_df.columns:
            attendance_df[std_col] = attendance_df[vn_col]

    # Fallback: Năm học -> year (một số file dùng Năm học thay Niên khoá)
    if "year" not in attendance_df.columns and "Năm học" in attendance_df.columns:
        attendance_df["year"] = attendance_df["Năm học"]
    # Fallback: Tên môn học -> Subject_ID khi thiếu Mã môn học
    # Lưu ý: Tên môn học (tên) không khớp Subject_ID (mã) trong exam → merge theo Student_ID+year
    use_subject_in_merge = "Mã môn học" in attendance_df.columns
    if "Subject_ID" not in attendance_df.columns:
        if "Mã nhóm" in attendance_df.columns:
            attendance_df["Subject_ID"] = attendance_df["Mã nhóm"].astype(str).str.strip()
        elif "Tên môn học" in attendance_df.columns:
            attendance_df["Subject_ID"] = attendance_df["Tên môn học"].astype(str).str.strip()
            use_subject_in_merge = False  # Tên môn ≠ mã môn trong exam

    # Extract year from Niên khoá/Năm học if needed (e.g., "2024-2025" -> 2024)
    if "year" in attendance_df.columns:
        attendance_df["year"] = attendance_df["year"].astype(str).str.split("-").str[0]
        attendance_df["year"] = pd.to_numeric(attendance_df["year"], errors="coerce")

    # Kiểm tra cột bắt buộc sau mapping; nếu thiếu thì bỏ qua merge, trả về df với attendance_rate=NaN
    required_attendance_cols = ["Student_ID", "Subject_ID", "year"]
    missing_att = [c for c in required_attendance_cols if c not in attendance_df.columns]
    if missing_att:
        logger.warning(
            f"Attendance file thiếu cột sau mapping: {missing_att}. Bỏ qua merge attendance. "
            f"(Cần MSSV, Mã môn học hoặc Tên môn học, Niên khoá hoặc Năm học)"
        )
        df = df.copy()
        if year_column in df.columns:
            df[year_column] = pd.to_numeric(df[year_column], errors="coerce")
        df["attendance_rate"] = np.nan
        return df

    # Ensure Student_ID and Subject_ID are correct types
    if "Student_ID" in attendance_df.columns:
        attendance_df["Student_ID"] = pd.to_numeric(attendance_df["Student_ID"], errors="coerce")

    # Ensure year columns have same type before merging
    df = df.copy()
    if year_column in df.columns:
        df[year_column] = pd.to_numeric(df[year_column], errors="coerce")

    # Calculate attendance rate
    # Map attendance status to numeric: Sớm/Có = 1, Trễ = 0.5, Vắng/Phép = 0
    if "Điểm danh" in attendance_df.columns:
        attendance_df["attendance_status"] = attendance_df["Điểm danh"].map({
            "Sớm": 1.0,
            "Có": 1.0,
            "Trễ": 0.5,
            "Vắng": 0.0,
            "Phép": 0.0,
        }).fillna(0.0)

        if use_subject_in_merge:
            # Aggregate by Student_ID, Subject_ID, year (có Mã môn học)
            attendance_agg = attendance_df.groupby(["Student_ID", "Subject_ID", "year"]).agg({
                "attendance_status": "mean",
            }).reset_index()
            merge_on = ["Student_ID", "Subject_ID", year_column]
        else:
            # Chỉ có Tên môn học: aggregate theo Student_ID + year (tỷ lệ điểm danh chung)
            attendance_agg = attendance_df.groupby(["Student_ID", "year"]).agg({
                "attendance_status": "mean",
            }).reset_index()
            merge_on = ["Student_ID", year_column]
            logger.info(
                "Attendance file dùng Tên môn học, merge theo Student_ID+year (tỷ lệ điểm danh chung/năm)"
            )

        attendance_agg = attendance_agg.rename(columns={
            "attendance_status": "attendance_rate",
            "year": year_column if year_column != "year" else "year"
        })
    else:
        logger.warning("'Điểm danh' column not found in attendance data, skipping attendance rate calculation")
        if use_subject_in_merge:
            attendance_agg = attendance_df[["Student_ID", "Subject_ID", "year"]].drop_duplicates()
        else:
            attendance_agg = attendance_df[["Student_ID", "year"]].drop_duplicates()
        attendance_agg["attendance_rate"] = None
        if year_column != "year":
            attendance_agg = attendance_agg.rename(columns={"year": year_column})
        merge_on = ["Student_ID", "Subject_ID", year_column] if use_subject_in_merge else ["Student_ID", year_column]

    # Merge
    merged = df.merge(
        attendance_agg,
        on=merge_on,
        how="left",
        suffixes=("", "_attendance"),
    )

    logger.info(
        f"Merged attendance: {len(merged)} records "
        f"({merged['attendance_rate'].notna().sum() if 'attendance_rate' in merged.columns else 0} with attendance data)"
    )

    return merged


def merge_all_data_sources(
    exam_df: pd.DataFrame,
    conduct_df: Optional[pd.DataFrame] = None,
    demographics_df: Optional[pd.DataFrame] = None,
    teaching_methods_df: Optional[pd.DataFrame] = None,
    assessment_methods_df: Optional[pd.DataFrame] = None,
    study_hours_df: Optional[pd.DataFrame] = None,
    attendance_df: Optional[pd.DataFrame] = None,
    year_column: str = "year",
) -> pd.DataFrame:
    """Merge all data sources into a single training dataset.

    This is the main function to create the complete training dataset by merging:
    1. Exam scores (base table)
    2. Conduct scores
    3. Demographics
    4. Teaching methods
    5. Assessment methods
    6. Study hours
    7. Attendance

    All merges are left joins to preserve all exam score records.

    Args:
        exam_df: Exam scores DataFrame (base table, must be preprocessed)
        conduct_df: Conduct scores DataFrame (optional)
        demographics_df: Demographics DataFrame (optional)
        teaching_methods_df: Teaching methods DataFrame (optional)
        assessment_methods_df: Assessment methods DataFrame (optional)
        study_hours_df: Study hours DataFrame (optional)
        attendance_df: Attendance DataFrame (optional)
        year_column: Name of year column (default: "year")

    Returns:
        Complete merged DataFrame ready for feature engineering

    Raises:
        DataValidationError: If exam_df is missing required columns
    """
    logger.info("Starting complete data merging pipeline")

    # Validate base table
    required_cols = ["Student_ID", "Subject_ID", year_column]
    missing = [col for col in required_cols if col not in exam_df.columns]
    if missing:
        raise DataValidationError(f"Missing required columns in exam_df: {missing}")

    merged = exam_df.copy()
    original_count = len(merged)

    # Merge conduct scores
    if conduct_df is not None:
        merged = merge_exam_and_conduct_scores(merged, conduct_df, year_column=year_column)

    # Merge demographics
    if demographics_df is not None:
        merged = merge_demographics(merged, demographics_df)

    # Merge teaching methods
    if teaching_methods_df is not None:
        merged = merge_teaching_methods(merged, teaching_methods_df)

    # Merge assessment methods
    if assessment_methods_df is not None:
        merged = merge_assessment_methods(merged, assessment_methods_df)

    # Merge study hours
    if study_hours_df is not None:
        merged = merge_study_hours(merged, study_hours_df, year_column=year_column)

    # Merge attendance
    if attendance_df is not None:
        merged = merge_attendance(merged, attendance_df, year_column=year_column)

    logger.info(
        f"Complete merging pipeline finished: {len(merged)} records "
        f"(started with {original_count} exam score records)"
    )
    logger.info(f"Final dataset has {len(merged.columns)} columns")

    return merged


def create_training_dataset(
    exam_df: pd.DataFrame,
    conduct_df: Optional[pd.DataFrame] = None,
    demographics_df: Optional[pd.DataFrame] = None,
    teaching_methods_df: Optional[pd.DataFrame] = None,
    assessment_methods_df: Optional[pd.DataFrame] = None,
    study_hours_df: Optional[pd.DataFrame] = None,
    attendance_df: Optional[pd.DataFrame] = None,
    year_column: str = "year",
    target_column: str = "exam_score",
    drop_missing_target: bool = True,
) -> pd.DataFrame:
    """Create final training dataset with all features merged.

    This function:
    1. Merges all data sources
    2. Handles missing values
    3. Ensures target column exists
    4. Drops rows with missing target if requested

    Args:
        exam_df: Preprocessed exam scores DataFrame
        conduct_df: Conduct scores DataFrame (optional)
        demographics_df: Demographics DataFrame (optional)
        teaching_methods_df: Teaching methods DataFrame (optional)
        assessment_methods_df: Assessment methods DataFrame (optional)
        study_hours_df: Study hours DataFrame (optional)
        attendance_df: Attendance DataFrame (optional)
        year_column: Name of year column (default: "year")
        target_column: Name of target variable (default: "exam_score")
        drop_missing_target: If True, drop rows with missing target (default: True)

    Returns:
        Final training dataset ready for model training

    Raises:
        DataValidationError: If target column is missing after merging
    """
    logger.info("Creating final training dataset")

    # Merge all data sources
    training_df = merge_all_data_sources(
        exam_df=exam_df,
        conduct_df=conduct_df,
        demographics_df=demographics_df,
        teaching_methods_df=teaching_methods_df,
        assessment_methods_df=assessment_methods_df,
        study_hours_df=study_hours_df,
        attendance_df=attendance_df,
        year_column=year_column,
    )

    # Validate target column
    if target_column not in training_df.columns:
        raise DataValidationError(
            f"Target column '{target_column}' not found after merging. "
            f"Available columns: {list(training_df.columns)[:20]}..."
        )

    # Drop rows with missing target if requested
    if drop_missing_target:
        before_count = len(training_df)
        training_df = training_df.dropna(subset=[target_column])
        dropped_count = before_count - len(training_df)
        if dropped_count > 0:
            logger.warning(f"Dropped {dropped_count} rows with missing target ({target_column})")

    logger.info(
        f"Final training dataset created: {len(training_df)} records, "
        f"{len(training_df.columns)} columns, target: {target_column}"
    )

    return training_df

