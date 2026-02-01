"""Data loaders for Excel files.

This module provides functions to load various Excel data files used in the
CLO prediction system. All loaders handle file reading, basic validation,
and return standardized pandas DataFrames.
"""

import os
from pathlib import Path
from typing import Optional

import pandas as pd

from ml_clo.utils.exceptions import DataLoadError
from ml_clo.utils.logger import get_logger

logger = get_logger(__name__)


def _load_excel_file(
    file_path: str,
    sheet_name: Optional[str] = None,
    required_columns: Optional[list[str]] = None,
) -> pd.DataFrame:
    """Internal helper to load Excel file with error handling.

    Args:
        file_path: Path to Excel file
        sheet_name: Sheet name to read (None for first sheet)
        required_columns: List of required column names (None to skip check)

    Returns:
        DataFrame with loaded data

    Raises:
        DataLoadError: If file cannot be loaded or required columns missing
    """
    file_path_obj = Path(file_path)

    if not file_path_obj.exists():
        raise DataLoadError(f"File not found: {file_path}")

    if not file_path_obj.is_file():
        raise DataLoadError(f"Path is not a file: {file_path}")

    try:
        logger.info(f"Loading Excel file: {file_path}")
        result = pd.read_excel(file_path, sheet_name=sheet_name, engine="openpyxl")

        # Handle case where read_excel returns a dict (multiple sheets)
        if isinstance(result, dict):
            if sheet_name is None:
                # If no sheet specified, use the first sheet
                first_sheet = list(result.keys())[0]
                logger.info(f"Multiple sheets found, using first sheet: {first_sheet}")
                df = result[first_sheet]
            else:
                # Sheet name was specified but not found
                available_sheets = list(result.keys())
                raise DataLoadError(
                    f"Sheet '{sheet_name}' not found. Available sheets: {available_sheets}"
                )
        else:
            df = result

        if df.empty:
            logger.warning(f"Loaded file is empty: {file_path}")
            return df

        logger.info(f"Loaded {len(df)} rows and {len(df.columns)} columns")

        # Check required columns if specified
        if required_columns:
            missing_cols = [col for col in required_columns if col not in df.columns]
            if missing_cols:
                raise DataLoadError(
                    f"Missing required columns in {file_path}: {missing_cols}"
                )

        return df

    except Exception as e:
        raise DataLoadError(f"Error loading file {file_path}: {str(e)}") from e


def load_conduct_scores(file_path: str) -> pd.DataFrame:
    """Load conduct scores (điểm rèn luyện) data.

    Expected columns:
        - Student_ID (int64): Mã số sinh viên
        - name (object): Họ và tên sinh viên
        - Class_ID (object): Mã lớp
        - semester_year (object): Năm học (e.g., "2021-2022")
        - semester (int64): Học kỳ (1, 2, 3)
        - conduct_score (int64): Điểm rèn luyện (0-100)
        - student_conduct_classification (object): Xếp loại rèn luyện

    Args:
        file_path: Path to diemrenluyen.xlsx file

    Returns:
        DataFrame with conduct scores data

    Raises:
        DataLoadError: If file cannot be loaded or required columns missing
    """
    required_columns = [
        "Student_ID",
        "name",
        "Class_ID",
        "semester_year",
        "semester",
        "conduct_score",
        "student_conduct_classification",
    ]

    df = _load_excel_file(file_path, required_columns=required_columns)

    # Validate data types
    if "Student_ID" in df.columns:
        df["Student_ID"] = pd.to_numeric(df["Student_ID"], errors="coerce")
    if "semester" in df.columns:
        df["semester"] = pd.to_numeric(df["semester"], errors="coerce")
    if "conduct_score" in df.columns:
        df["conduct_score"] = pd.to_numeric(df["conduct_score"], errors="coerce")

    logger.info(f"Successfully loaded conduct scores: {len(df)} records")
    return df


def load_exam_scores(file_path: str) -> pd.DataFrame:
    """Load exam scores (điểm tổng) data.

    Expected columns include:
        - Student_ID: Mã số sinh viên
        - Subject_ID: Mã môn học
        - Lecturer_ID: Mã giảng viên
        - exam_score: Điểm thi (string, needs conversion)
        - summary_score: Điểm tổng kết
        - year: Năm học
        - Passed_the_module: Đậu (1) / Rớt (0)

    Args:
        file_path: Path to DiemTong.xlsx file

    Returns:
        DataFrame with exam scores data

    Raises:
        DataLoadError: If file cannot be loaded
    """
    # Key columns that should exist (some may be optional)
    key_columns = ["Student_ID", "Subject_ID", "Lecturer_ID"]

    df = _load_excel_file(file_path)

    # Check for key columns
    missing_key_cols = [col for col in key_columns if col not in df.columns]
    if missing_key_cols:
        logger.warning(
            f"Missing key columns in exam scores file: {missing_key_cols}. "
            f"Available columns: {list(df.columns)}"
        )

    # Ensure Student_ID is numeric
    if "Student_ID" in df.columns:
        df["Student_ID"] = pd.to_numeric(df["Student_ID"], errors="coerce")

    logger.info(f"Successfully loaded exam scores: {len(df)} records")
    return df


def load_demographics(file_path: str) -> pd.DataFrame:
    """Load demographics (nhân khẩu học) data.

    Expected columns include:
        - Student_ID: Mã số sinh viên
        - Gender: Giới tính
        - place_of_birth: Nơi sinh
        - Ethnicity: Dân tộc
        - Religion: Tôn giáo

    Args:
        file_path: Path to nhankhau.xlsx file

    Returns:
        DataFrame with demographics data

    Raises:
        DataLoadError: If file cannot be loaded
    """
    df = _load_excel_file(file_path)

    # Ensure Student_ID is numeric if it exists
    if "Student_ID" in df.columns:
        df["Student_ID"] = pd.to_numeric(df["Student_ID"], errors="coerce")

    logger.info(f"Successfully loaded demographics: {len(df)} records")
    return df


def load_teaching_methods(file_path: str) -> pd.DataFrame:
    """Load teaching methods (phương pháp giảng dạy) data.

    Expected format: File with teaching method mappings (TM1, TM2, etc.)
    Typically contains Subject_ID and columns for each teaching method.

    Args:
        file_path: Path to PPGD.xlsx or PPGDfull.xlsx file

    Returns:
        DataFrame with teaching methods data

    Raises:
        DataLoadError: If file cannot be loaded
    """
    df = _load_excel_file(file_path)

    # Check for Subject_ID
    if "Subject_ID" not in df.columns:
        logger.warning(
            f"Subject_ID not found in teaching methods file. "
            f"Available columns: {list(df.columns)}"
        )

    logger.info(f"Successfully loaded teaching methods: {len(df)} records")
    return df


def load_assessment_methods(file_path: str) -> pd.DataFrame:
    """Load assessment methods (phương pháp đánh giá) data.

    Expected format: File with assessment method mappings (EM1, EM2, etc.)
    Typically contains Subject_ID and columns for each assessment method.

    Args:
        file_path: Path to PPDG.xlsx or PPDGfull.xlsx file

    Returns:
        DataFrame with assessment methods data

    Raises:
        DataLoadError: If file cannot be loaded
    """
    df = _load_excel_file(file_path)

    # Check for Subject_ID
    if "Subject_ID" not in df.columns:
        logger.warning(
            f"Subject_ID not found in assessment methods file. "
            f"Available columns: {list(df.columns)}"
        )

    logger.info(f"Successfully loaded assessment methods: {len(df)} records")
    return df


def load_study_hours(file_path: str) -> pd.DataFrame:
    """Load study hours (giờ tự học) data.

    Expected columns:
        - Student_ID: Mã số sinh viên
        - name: Họ tên sinh viên
        - Class_Id: Mã lớp
        - year: Năm học
        - semester: Học kỳ
        - accumulated_study_hours: Tổng giờ tự học
        - accumulated_study_minutes: Tổng phút tự học
        - time: Thời gian tự học dạng HH:MM

    Args:
        file_path: Path to tuhoc.xlsx file

    Returns:
        DataFrame with study hours data

    Raises:
        DataLoadError: If file cannot be loaded or required columns missing
    """
    # Key columns (some may be optional)
    key_columns = ["Student_ID", "year", "semester"]

    df = _load_excel_file(file_path)

    # Check for key columns
    missing_key_cols = [col for col in key_columns if col not in df.columns]
    if missing_key_cols:
        logger.warning(
            f"Missing key columns in study hours file: {missing_key_cols}. "
            f"Available columns: {list(df.columns)}"
        )

    # Ensure Student_ID is numeric
    if "Student_ID" in df.columns:
        df["Student_ID"] = pd.to_numeric(df["Student_ID"], errors="coerce")
    if "semester" in df.columns:
        df["semester"] = pd.to_numeric(df["semester"], errors="coerce")

    # Ensure accumulated_study_hours is numeric if exists
    if "accumulated_study_hours" in df.columns:
        df["accumulated_study_hours"] = pd.to_numeric(
            df["accumulated_study_hours"], errors="coerce"
        )

    logger.info(f"Successfully loaded study hours: {len(df)} records")
    return df


def load_attendance(file_path: str) -> pd.DataFrame:
    """Load attendance (điểm danh) data.

    Expected columns:
        - MSSV: Mã số sinh viên (maps to Student_ID)
        - Họ Tên: Tên sinh viên
        - Mã môn học: Mã học phần (maps to Subject_ID)
        - Tên môn học: Tên học phần
        - Mã giảng viên: Mã GV (maps to Lecturer_ID)
        - Ngày: Ngày học
        - Buổi: Thứ tự buổi học
        - Niên khoá: Năm học
        - Học kì: Học kỳ
        - Điểm danh: Trạng thái đi học (Sớm, Trễ, Vắng, Có mặt)

    Args:
        file_path: Path to attendance Excel file

    Returns:
        DataFrame with attendance data

    Raises:
        DataLoadError: If file cannot be loaded
    """
    df = _load_excel_file(file_path)

    # Map Vietnamese column names to standard names if needed
    # Keep original columns for now, mapping will be done in preprocessor

    # Check for key columns (Vietnamese names)
    key_columns_vn = ["MSSV", "Mã môn học", "Mã giảng viên"]
    missing_key_cols = [
        col for col in key_columns_vn if col not in df.columns
    ]
    if missing_key_cols:
        logger.warning(
            f"Missing key columns in attendance file: {missing_key_cols}. "
            f"Available columns: {list(df.columns)}"
        )

    # Ensure MSSV is numeric if it exists
    if "MSSV" in df.columns:
        df["MSSV"] = pd.to_numeric(df["MSSV"], errors="coerce")

    logger.info(f"Successfully loaded attendance: {len(df)} records")
    return df


def load_all_data_files(data_dir: str) -> dict[str, pd.DataFrame]:
    """Load all data files from a directory.

    This is a convenience function to load all expected data files at once.
    File names are expected to match standard naming conventions.

    Args:
        data_dir: Directory containing data files

    Returns:
        Dictionary mapping data type names to DataFrames:
        - 'conduct_scores': Conduct scores data
        - 'exam_scores': Exam scores data
        - 'demographics': Demographics data
        - 'teaching_methods': Teaching methods data
        - 'assessment_methods': Assessment methods data
        - 'study_hours': Study hours data
        - 'attendance': Attendance data

    Raises:
        DataLoadError: If data directory doesn't exist or files cannot be loaded
    """
    data_path = Path(data_dir)

    if not data_path.exists():
        raise DataLoadError(f"Data directory not found: {data_dir}")

    if not data_path.is_dir():
        raise DataLoadError(f"Path is not a directory: {data_dir}")

    results: dict[str, pd.DataFrame] = {}
    file_mappings = {
        "conduct_scores": ["diemrenluyen.xlsx"],
        "exam_scores": ["DiemTong.xlsx"],
        "demographics": ["nhankhau.xlsx"],
        "teaching_methods": ["PPGDfull.xlsx", "PPGD.xlsx"],
        "assessment_methods": ["PPDGfull.xlsx", "PPDG.xlsx"],
        "study_hours": ["tuhoc.xlsx"],
        "attendance": ["Dữ liệu điểm danh Khoa FIRA.xlsx"],
    }

    loader_functions = {
        "conduct_scores": load_conduct_scores,
        "exam_scores": load_exam_scores,
        "demographics": load_demographics,
        "teaching_methods": load_teaching_methods,
        "assessment_methods": load_assessment_methods,
        "study_hours": load_study_hours,
        "attendance": load_attendance,
    }

    for data_type, file_names in file_mappings.items():
        file_found = False
        for file_name in file_names:
            file_path = data_path / file_name
            if file_path.exists():
                try:
                    loader_func = loader_functions[data_type]
                    results[data_type] = loader_func(str(file_path))
                    file_found = True
                    logger.info(f"Loaded {data_type} from {file_name}")
                    break
                except DataLoadError as e:
                    logger.warning(f"Failed to load {file_name}: {e}")
                    continue

        if not file_found:
            logger.warning(
                f"Could not find or load {data_type} from any of: {file_names}"
            )

    logger.info(f"Successfully loaded {len(results)} data files")
    return results

