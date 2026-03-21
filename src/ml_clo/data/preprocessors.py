"""Data preprocessing functions.

This module provides functions to clean, normalize, and preprocess data
before feature engineering and model training. Key operations include:
- ID standardization
- Score conversion (10-point scale → 6-point scale)
- Data cleaning and validation
- Missing value handling
"""

from typing import Optional

import numpy as np
import pandas as pd

from ml_clo.utils.exceptions import DataValidationError
from ml_clo.utils.logger import get_logger

logger = get_logger(__name__)


def standardize_student_id(df: pd.DataFrame, id_column: str = "Student_ID") -> pd.DataFrame:
    """Standardize Student_ID column.

    Converts Student_ID to numeric, removes invalid values, and ensures consistency.

    Args:
        df: DataFrame with Student_ID column
        id_column: Name of the student ID column (default: "Student_ID")

    Returns:
        DataFrame with standardized Student_ID

    Raises:
        DataValidationError: If Student_ID column is missing or cannot be standardized
    """
    if id_column not in df.columns:
        raise DataValidationError(f"Column '{id_column}' not found in DataFrame")

    df = df.copy()
    original_count = len(df)

    # Convert to numeric, invalid values become NaN
    df[id_column] = pd.to_numeric(df[id_column], errors="coerce")

    # Remove rows with invalid Student_ID
    invalid_count = df[id_column].isna().sum()
    if invalid_count > 0:
        logger.warning(
            f"Removed {invalid_count} rows with invalid {id_column} "
            f"({invalid_count}/{original_count} = {invalid_count/original_count*100:.1f}%)"
        )
        df = df.dropna(subset=[id_column])

    # Convert to int64 (remove decimal if any)
    df[id_column] = df[id_column].astype("int64")

    logger.info(f"Standardized {id_column}: {len(df)} valid records")
    return df


def standardize_subject_id(df: pd.DataFrame, id_column: str = "Subject_ID") -> pd.DataFrame:
    """Standardize Subject_ID column.

    Ensures Subject_ID is string type and removes leading/trailing whitespace.

    Args:
        df: DataFrame with Subject_ID column
        id_column: Name of the subject ID column (default: "Subject_ID")

    Returns:
        DataFrame with standardized Subject_ID

    Raises:
        DataValidationError: If Subject_ID column is missing
    """
    if id_column not in df.columns:
        raise DataValidationError(f"Column '{id_column}' not found in DataFrame")

    df = df.copy()
    original_count = len(df)

    # Convert to string and strip whitespace
    df[id_column] = df[id_column].astype(str).str.strip()

    # Remove rows with invalid Subject_ID (empty, "nan", "None")
    invalid_mask = (
        df[id_column].isin(["", "nan", "None", "NaN"]) | df[id_column].isna()
    )
    invalid_count = invalid_mask.sum()
    if invalid_count > 0:
        logger.warning(
            f"Removed {invalid_count} rows with invalid {id_column} "
            f"({invalid_count}/{original_count} = {invalid_count/original_count*100:.1f}%)"
        )
        df = df[~invalid_mask]

    logger.info(f"Standardized {id_column}: {len(df)} valid records")
    return df


def standardize_lecturer_id(df: pd.DataFrame, id_column: str = "Lecturer_ID") -> pd.DataFrame:
    """Standardize Lecturer_ID column.

    Converts Lecturer_ID to string and removes invalid values.

    Args:
        df: DataFrame with Lecturer_ID column
        id_column: Name of the lecturer ID column (default: "Lecturer_ID")

    Returns:
        DataFrame with standardized Lecturer_ID

    Raises:
        DataValidationError: If Lecturer_ID column is missing
    """
    if id_column not in df.columns:
        raise DataValidationError(f"Column '{id_column}' not found in DataFrame")

    df = df.copy()
    original_count = len(df)

    # Convert to string and strip whitespace
    df[id_column] = df[id_column].astype(str).str.strip()

    # Remove rows with invalid Lecturer_ID
    invalid_mask = (
        df[id_column].isin(["", "nan", "None", "NaN"]) | df[id_column].isna()
    )
    invalid_count = invalid_mask.sum()
    if invalid_count > 0:
        logger.warning(
            f"Removed {invalid_count} rows with invalid {id_column} "
            f"({invalid_count}/{original_count} = {invalid_count/original_count*100:.1f}%)"
        )
        df = df[~invalid_mask]

    logger.info(f"Standardized {id_column}: {len(df)} valid records")
    return df


def clean_exam_score(
    df: pd.DataFrame,
    score_column: str = "exam_score",
    remove_invalid: bool = True,
) -> pd.DataFrame:
    """Clean exam_score column.

    Removes invalid values (non-numeric strings like "VT", "Vắng", etc.)
    and converts to float.

    Args:
        df: DataFrame with exam_score column
        score_column: Name of the exam score column (default: "exam_score")
        remove_invalid: If True, remove rows with invalid scores (default: True)

    Returns:
        DataFrame with cleaned exam_score

    Raises:
        DataValidationError: If score_column is missing
    """
    if score_column not in df.columns:
        raise DataValidationError(f"Column '{score_column}' not found in DataFrame")

    df = df.copy()
    original_count = len(df)

    # Convert to numeric, invalid values become NaN
    df[score_column] = pd.to_numeric(df[score_column], errors="coerce")

    # Count invalid values
    invalid_count = df[score_column].isna().sum()
    if invalid_count > 0:
        logger.info(
            f"Found {invalid_count} invalid {score_column} values "
            f"({invalid_count}/{original_count} = {invalid_count/original_count*100:.1f}%)"
        )

        if remove_invalid:
            df = df.dropna(subset=[score_column])
            logger.info(f"Removed {invalid_count} rows with invalid {score_column}")
        else:
            logger.warning(f"Keeping {invalid_count} rows with NaN {score_column}")

    # Ensure float type
    df[score_column] = df[score_column].astype("float64")

    logger.info(f"Cleaned {score_column}: {len(df)} valid records")
    return df


def convert_score_10_to_6(
    df: pd.DataFrame,
    score_column: str = "exam_score",
    output_column: Optional[str] = None,
    inplace: bool = False,
) -> pd.DataFrame:
    """Convert exam scores from 10-point scale to 6-point scale (CLO scale).

    **CRITICAL**: This conversion is mandatory for all historical exam scores.
    Formula: CLO_6 = Score_10 / 10 × 6

    Args:
        df: DataFrame with exam scores in 10-point scale
        score_column: Name of the score column to convert (default: "exam_score")
        output_column: Name of output column. If None, overwrites score_column (default: None)
        inplace: If False, returns new DataFrame (default: False)

    Returns:
        DataFrame with converted scores in 6-point scale

    Raises:
        DataValidationError: If score_column is missing or contains invalid values
    """
    if score_column not in df.columns:
        raise DataValidationError(f"Column '{score_column}' not found in DataFrame")

    if not inplace:
        df = df.copy()

    # Check if scores are in valid range (0-10)
    valid_scores = df[score_column].dropna()
    if len(valid_scores) > 0:
        max_score = valid_scores.max()

        # Warn if scores seem to be already in 6-point scale
        if max_score <= 6.0:
            logger.warning(
                f"Scores in {score_column} appear to be already in 6-point scale "
                f"(max={max_score}). Conversion may not be needed."
            )
        elif max_score > 10.0:
            logger.warning(
                f"Scores in {score_column} exceed 10.0 (max={max_score}). "
                f"Some values may be invalid."
            )

    # Convert: CLO_6 = Score_10 / 10 × 6
    output_col = output_column if output_column else score_column
    df[output_col] = df[score_column] / 10.0 * 6.0

    # Ensure values are in valid range [0, 6]
    df[output_col] = df[output_col].clip(lower=0.0, upper=6.0)

    logger.info(
        f"Converted {score_column} from 10-point to 6-point scale "
        f"(stored in {output_col})"
    )

    return df


def create_result_column(
    df: pd.DataFrame,
    score_column: str = "exam_score",
    result_column: str = "Result",
    pass_threshold: float = 6.0,
) -> pd.DataFrame:
    """Create binary Result column (Pass/Fail).

    Result = 1 if score >= pass_threshold, else 0.
    **Note**: This column is for analysis only, NOT for model training.

    Args:
        df: DataFrame with exam scores
        score_column: Name of the score column (default: "exam_score")
        result_column: Name of the result column to create (default: "Result")
        pass_threshold: Minimum score to pass (default: 6.0)

    Returns:
        DataFrame with Result column added

    Raises:
        DataValidationError: If score_column is missing
    """
    if score_column not in df.columns:
        raise DataValidationError(f"Column '{score_column}' not found in DataFrame")

    df = df.copy()

    # Create binary result: 1 if Pass (>= threshold), 0 if Fail (< threshold)
    df[result_column] = (df[score_column] >= pass_threshold).astype(int)

    # Set NaN for rows with missing scores
    df.loc[df[score_column].isna(), result_column] = np.nan

    pass_count = (df[result_column] == 1).sum()
    fail_count = (df[result_column] == 0).sum()
    logger.info(
        f"Created {result_column} column: {pass_count} Pass, {fail_count} Fail "
        f"(threshold: {pass_threshold})"
    )

    return df


def handle_missing_values(
    df: pd.DataFrame,
    strategy: str = "drop",
    columns: Optional[list[str]] = None,
    threshold: float = 0.5,
) -> pd.DataFrame:
    """Handle missing values in DataFrame.

    Args:
        df: DataFrame with potential missing values
        strategy: Strategy to handle missing values:
            - "drop": Drop rows with any missing values in specified columns
            - "drop_threshold": Drop columns with missing rate > threshold
            - "fill_zero": Fill missing values with 0
            - "fill_mean": Fill missing values with column mean
            - "fill_median": Fill missing values with column median
        columns: List of columns to process. If None, process all columns (default: None)
        threshold: Threshold for drop_threshold strategy (default: 0.5)

    Returns:
        DataFrame with missing values handled

    Raises:
        DataValidationError: If strategy is invalid
    """
    if strategy not in ["drop", "drop_threshold", "fill_zero", "fill_mean", "fill_median"]:
        raise DataValidationError(f"Invalid strategy: {strategy}")

    df = df.copy()
    original_count = len(df)

    if columns is None:
        columns = df.columns.tolist()

    # Count missing values
    missing_counts = df[columns].isnull().sum()
    total_missing = missing_counts.sum()

    if total_missing == 0:
        logger.info("No missing values found")
        return df

    logger.info(f"Found {total_missing} missing values across {len(columns)} columns")

    if strategy == "drop":
        # Drop rows with any missing values in specified columns
        df = df.dropna(subset=columns)
        removed_count = original_count - len(df)
        logger.info(f"Dropped {removed_count} rows with missing values")

    elif strategy == "drop_threshold":
        # Drop columns with missing rate > threshold
        missing_rates = missing_counts / len(df)
        cols_to_drop = missing_rates[missing_rates > threshold].index.tolist()
        if cols_to_drop:
            df = df.drop(columns=cols_to_drop)
            logger.info(f"Dropped {len(cols_to_drop)} columns with missing rate > {threshold}")

    elif strategy == "fill_zero":
        df[columns] = df[columns].fillna(0)
        logger.info(f"Filled missing values with 0 in {len(columns)} columns")

    elif strategy == "fill_mean":
        numeric_cols = df[columns].select_dtypes(include=[np.number]).columns
        df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].mean())
        logger.info(f"Filled missing values with mean in {len(numeric_cols)} columns")

    elif strategy == "fill_median":
        numeric_cols = df[columns].select_dtypes(include=[np.number]).columns
        df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].median())
        logger.info(f"Filled missing values with median in {len(numeric_cols)} columns")

    return df


def ensure_year_column(df: pd.DataFrame, year_column: str = "year") -> pd.DataFrame:
    """Ensure year column exists and has valid numeric values.

    Derives year from semester_year, Niên khoá, Năm học, or registration_date if
    year is missing or all NaN. Required for merging with conduct, attendance, study_hours.

    Args:
        df: DataFrame (exam scores or similar)
        year_column: Name of year column (default: "year")

    Returns:
        DataFrame with year column populated
    """
    df = df.copy()
    if year_column not in df.columns:
        df[year_column] = np.nan

    year_vals = pd.to_numeric(df[year_column], errors="coerce")
    has_valid_year = year_vals.notna().any()

    if not has_valid_year:
        # Try alternative columns (Vietnamese or English)
        for src_col, fmt in [
            ("semester_year", "range"),  # "2021-2022" -> 2021
            ("Niên khoá", "range"),
            ("Năm học", "range"),
            ("registration_date", "date"),
        ]:
            if src_col not in df.columns:
                continue
            if fmt == "range":
                extracted = df[src_col].astype(str).str.split("-").str[0]
                extracted = pd.to_numeric(extracted, errors="coerce")
            else:  # date
                try:
                    # dayfirst=True cho format dd/mm/yyyy (phổ biến tại VN)
                    dates = pd.to_datetime(df[src_col], errors="coerce", dayfirst=True)
                    extracted = dates.dt.year
                except Exception:
                    extracted = pd.Series([np.nan] * len(df), dtype=float)
            if extracted.notna().any():
                df[year_column] = extracted
                logger.info(
                    f"Derived year from column '{src_col}': "
                    f"{int(extracted.notna().sum())} valid values"
                )
                break

    # Final fallback: fill remaining NaN with default
    if df[year_column].isna().all() or df[year_column].isna().any():
        default_year = 2024
        fill_count = int(df[year_column].isna().sum())
        if fill_count > 0:
            df.loc[df[year_column].isna(), year_column] = default_year
            logger.warning(
                f"Filled {fill_count} rows with default year={default_year} "
                "(year was missing or invalid; merge with conduct/attendance may be limited)"
            )
    df[year_column] = pd.to_numeric(df[year_column], errors="coerce")
    return df


def preprocess_exam_scores(
    df: pd.DataFrame,
    convert_to_clo: bool = True,
    create_result: bool = True,
) -> pd.DataFrame:
    """Complete preprocessing pipeline for exam scores data.

    This function applies all necessary preprocessing steps:
    1. Standardize IDs (Student_ID, Subject_ID, Lecturer_ID)
    2. Clean exam_score
    3. Convert to 6-point scale (if convert_to_clo=True)
    4. Create Result column (if create_result=True)
    5. Handle missing values

    Args:
        df: Raw exam scores DataFrame
        convert_to_clo: If True, convert scores from 10-point to 6-point scale (default: True)
        create_result: If True, create Result column (default: True)

    Returns:
        Preprocessed DataFrame ready for feature engineering
    """
    logger.info("Starting exam scores preprocessing pipeline")

    # Step 1: Standardize IDs
    df = standardize_student_id(df, id_column="Student_ID")
    df = standardize_subject_id(df, id_column="Subject_ID")
    df = standardize_lecturer_id(df, id_column="Lecturer_ID")

    # Step 2: Clean exam_score
    df = clean_exam_score(df, score_column="exam_score", remove_invalid=True)

    # Step 3: Convert to 6-point scale (CRITICAL)
    if convert_to_clo:
        df = convert_score_10_to_6(df, score_column="exam_score", inplace=True)

    # Step 4: Create Result column (for analysis only)
    if create_result:
        df = create_result_column(df, score_column="exam_score", result_column="Result")

    # Step 5: Handle missing values in key columns
    key_columns = ["Student_ID", "Subject_ID", "Lecturer_ID", "exam_score"]
    df = handle_missing_values(df, strategy="drop", columns=key_columns)

    # Step 6: Ensure year column (required for merging with conduct, attendance, study_hours)
    df = ensure_year_column(df, year_column="year")

    logger.info(f"Preprocessing complete: {len(df)} records ready for feature engineering")
    return df

