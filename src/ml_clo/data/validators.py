"""Data validation functions.

This module provides functions to validate data quality, types, ranges,
and consistency before processing and model training.
"""

from typing import Optional

import numpy as np
import pandas as pd

from ml_clo.utils.exceptions import DataValidationError
from ml_clo.utils.logger import get_logger

logger = get_logger(__name__)


def validate_data_types(
    df: pd.DataFrame,
    expected_types: dict[str, type],
    strict: bool = False,
) -> tuple[bool, list[str]]:
    """Validate data types of columns.

    Args:
        df: DataFrame to validate
        expected_types: Dictionary mapping column names to expected types
        strict: If True, raise exception on mismatch. If False, return warnings (default: False)

    Returns:
        Tuple of (is_valid, list of error messages)

    Raises:
        DataValidationError: If strict=True and validation fails
    """
    errors = []
    is_valid = True

    for col, expected_type in expected_types.items():
        if col not in df.columns:
            error_msg = f"Column '{col}' not found in DataFrame"
            errors.append(error_msg)
            is_valid = False
            if strict:
                raise DataValidationError(error_msg)
            continue

        actual_type = df[col].dtype
        if not pd.api.types.is_dtype_equal(actual_type, expected_type):
            # Check if types are compatible (e.g., int64 vs int32)
            if not _are_types_compatible(actual_type, expected_type):
                error_msg = (
                    f"Column '{col}' has type {actual_type}, expected {expected_type}"
                )
                errors.append(error_msg)
                is_valid = False
                if strict:
                    raise DataValidationError(error_msg)

    if errors:
        logger.warning(f"Data type validation found {len(errors)} issues")
        for error in errors:
            logger.warning(f"  - {error}")
    else:
        logger.info("Data type validation passed")

    return is_valid, errors


def _are_types_compatible(type1, type2) -> bool:
    """Check if two pandas dtypes are compatible."""
    # Both numeric
    if pd.api.types.is_numeric_dtype(type1) and pd.api.types.is_numeric_dtype(type2):
        return True
    # Both string/object
    if pd.api.types.is_string_dtype(type1) and pd.api.types.is_string_dtype(type2):
        return True
    # Exact match
    return type1 == type2


def validate_ranges(
    df: pd.DataFrame,
    column_ranges: dict[str, tuple[float, float]],
    strict: bool = False,
) -> tuple[bool, list[str]]:
    """Validate that numeric columns are within expected ranges.

    Args:
        df: DataFrame to validate
        column_ranges: Dictionary mapping column names to (min, max) tuples
        strict: If True, raise exception on violation. If False, return warnings (default: False)

    Returns:
        Tuple of (is_valid, list of error messages)

    Raises:
        DataValidationError: If strict=True and validation fails
    """
    errors = []
    is_valid = True

    for col, (min_val, max_val) in column_ranges.items():
        if col not in df.columns:
            error_msg = f"Column '{col}' not found in DataFrame"
            errors.append(error_msg)
            is_valid = False
            if strict:
                raise DataValidationError(error_msg)
            continue

        # Only validate numeric columns
        if not pd.api.types.is_numeric_dtype(df[col]):
            logger.debug(f"Column '{col}' is not numeric, skipping range validation")
            continue

        # Check range
        col_min = df[col].min()
        col_max = df[col].max()

        if col_min < min_val or col_max > max_val:
            error_msg = (
                f"Column '{col}' has values outside range [{min_val}, {max_val}]: "
                f"actual range [{col_min:.2f}, {col_max:.2f}]"
            )
            errors.append(error_msg)
            is_valid = False
            if strict:
                raise DataValidationError(error_msg)

    if errors:
        logger.warning(f"Range validation found {len(errors)} issues")
        for error in errors:
            logger.warning(f"  - {error}")
    else:
        logger.info("Range validation passed")

    return is_valid, errors


def validate_required_fields(
    df: pd.DataFrame,
    required_columns: list[str],
    strict: bool = False,
) -> tuple[bool, list[str]]:
    """Validate that required columns exist in DataFrame.

    Args:
        df: DataFrame to validate
        required_columns: List of required column names
        strict: If True, raise exception on missing column. If False, return warnings (default: False)

    Returns:
        Tuple of (is_valid, list of error messages)

    Raises:
        DataValidationError: If strict=True and validation fails
    """
    errors = []
    is_valid = True

    missing_columns = [col for col in required_columns if col not in df.columns]

    if missing_columns:
        error_msg = f"Missing required columns: {missing_columns}"
        errors.append(error_msg)
        is_valid = False
        if strict:
            raise DataValidationError(error_msg)

    if errors:
        logger.warning(f"Required fields validation found {len(errors)} issues")
        for error in errors:
            logger.warning(f"  - {error}")
    else:
        logger.info(f"Required fields validation passed: all {len(required_columns)} columns present")

    return is_valid, errors


def validate_no_missing_values(
    df: pd.DataFrame,
    columns: Optional[list[str]] = None,
    strict: bool = False,
) -> tuple[bool, list[str]]:
    """Validate that specified columns have no missing values.

    Args:
        df: DataFrame to validate
        columns: List of columns to check. If None, check all columns (default: None)
        strict: If True, raise exception on missing values. If False, return warnings (default: False)

    Returns:
        Tuple of (is_valid, list of error messages)

    Raises:
        DataValidationError: If strict=True and validation fails
    """
    errors = []
    is_valid = True

    if columns is None:
        columns = df.columns.tolist()

    for col in columns:
        if col not in df.columns:
            continue

        missing_count = df[col].isna().sum()
        if missing_count > 0:
            missing_pct = missing_count / len(df) * 100
            error_msg = (
                f"Column '{col}' has {missing_count} missing values "
                f"({missing_pct:.1f}%)"
            )
            errors.append(error_msg)
            is_valid = False
            if strict:
                raise DataValidationError(error_msg)

    if errors:
        logger.warning(f"Missing values validation found {len(errors)} issues")
        for error in errors:
            logger.warning(f"  - {error}")
    else:
        logger.info("Missing values validation passed: no missing values in checked columns")

    return is_valid, errors


def validate_clo_score_range(
    df: pd.DataFrame,
    score_column: str = "exam_score",
    strict: bool = False,
) -> tuple[bool, list[str]]:
    """Validate that CLO scores are in valid range [0, 6].

    Args:
        df: DataFrame with CLO scores
        score_column: Name of score column (default: "exam_score")
        strict: If True, raise exception on violation. If False, return warnings (default: False)

    Returns:
        Tuple of (is_valid, list of error messages)

    Raises:
        DataValidationError: If strict=True and validation fails
    """
    return validate_ranges(
        df, {score_column: (0.0, 6.0)}, strict=strict
    )


def validate_conduct_score_range(
    df: pd.DataFrame,
    score_column: str = "conduct_score",
    strict: bool = False,
) -> tuple[bool, list[str]]:
    """Validate that conduct scores are in valid range [0, 100].

    Args:
        df: DataFrame with conduct scores
        score_column: Name of score column (default: "conduct_score")
        strict: If True, raise exception on violation. If False, return warnings (default: False)

    Returns:
        Tuple of (is_valid, list of error messages)

    Raises:
        DataValidationError: If strict=True and validation fails
    """
    return validate_ranges(
        df, {score_column: (0.0, 100.0)}, strict=strict
    )


def validate_data_consistency(
    df: pd.DataFrame,
    id_columns: list[str],
    strict: bool = False,
) -> tuple[bool, list[str]]:
    """Validate data consistency (e.g., no duplicate key combinations).

    Args:
        df: DataFrame to validate
        id_columns: List of ID columns that should form unique combinations
        strict: If True, raise exception on inconsistency. If False, return warnings (default: False)

    Returns:
        Tuple of (is_valid, list of error messages)

    Raises:
        DataValidationError: If strict=True and validation fails
    """
    errors = []
    is_valid = True

    # Check if all ID columns exist
    missing_cols = [col for col in id_columns if col not in df.columns]
    if missing_cols:
        error_msg = f"Cannot validate consistency: missing columns {missing_cols}"
        errors.append(error_msg)
        is_valid = False
        if strict:
            raise DataValidationError(error_msg)
        return is_valid, errors

    # Check for duplicates
    duplicates = df.duplicated(subset=id_columns, keep=False)
    duplicate_count = duplicates.sum()

    if duplicate_count > 0:
        error_msg = (
            f"Found {duplicate_count} duplicate rows based on columns {id_columns}"
        )
        errors.append(error_msg)
        is_valid = False
        if strict:
            raise DataValidationError(error_msg)

    if errors:
        logger.warning(f"Data consistency validation found {len(errors)} issues")
        for error in errors:
            logger.warning(f"  - {error}")
    else:
        logger.info(
            f"Data consistency validation passed: no duplicates in {id_columns}"
        )

    return is_valid, errors


def validate_exam_scores_data(
    df: pd.DataFrame,
    strict: bool = False,
) -> tuple[bool, list[str]]:
    """Complete validation for exam scores data.

    Validates:
    - Required columns (Student_ID, Subject_ID, Lecturer_ID, exam_score)
    - CLO score range [0, 6]
    - No missing values in key columns
    - Data consistency

    Args:
        df: Exam scores DataFrame
        strict: If True, raise exception on validation failure (default: False)

    Returns:
        Tuple of (is_valid, list of all error messages)
    """
    logger.info("Starting comprehensive validation for exam scores data")

    all_errors = []
    is_valid = True

    # 1. Required fields
    required_cols = ["Student_ID", "Subject_ID", "Lecturer_ID", "exam_score"]
    valid, errors = validate_required_fields(df, required_cols, strict=strict)
    is_valid = is_valid and valid
    all_errors.extend(errors)

    # 2. CLO score range
    if "exam_score" in df.columns:
        valid, errors = validate_clo_score_range(df, "exam_score", strict=strict)
        is_valid = is_valid and valid
        all_errors.extend(errors)

    # 3. No missing values in key columns
    key_columns = ["Student_ID", "Subject_ID", "Lecturer_ID", "exam_score"]
    valid, errors = validate_no_missing_values(df, key_columns, strict=strict)
    is_valid = is_valid and valid
    all_errors.extend(errors)

    # 4. Data consistency (optional - may have legitimate duplicates)
    # Skip for now as exam scores may have multiple records per student-subject

    if is_valid:
        logger.info("Comprehensive validation passed for exam scores data")
    else:
        logger.warning(
            f"Comprehensive validation found {len(all_errors)} issues "
            f"for exam scores data"
        )

    return is_valid, all_errors


def validate_conduct_scores_data(
    df: pd.DataFrame,
    strict: bool = False,
) -> tuple[bool, list[str]]:
    """Complete validation for conduct scores data.

    Validates:
    - Required columns (Student_ID, conduct_score)
    - Conduct score range [0, 100]
    - No missing values in key columns

    Args:
        df: Conduct scores DataFrame
        strict: If True, raise exception on validation failure (default: False)

    Returns:
        Tuple of (is_valid, list of all error messages)
    """
    logger.info("Starting comprehensive validation for conduct scores data")

    all_errors = []
    is_valid = True

    # 1. Required fields
    required_cols = ["Student_ID", "conduct_score"]
    valid, errors = validate_required_fields(df, required_cols, strict=strict)
    is_valid = is_valid and valid
    all_errors.extend(errors)

    # 2. Conduct score range
    if "conduct_score" in df.columns:
        valid, errors = validate_conduct_score_range(df, "conduct_score", strict=strict)
        is_valid = is_valid and valid
        all_errors.extend(errors)

    # 3. No missing values in key columns
    key_columns = ["Student_ID", "conduct_score"]
    valid, errors = validate_no_missing_values(df, key_columns, strict=strict)
    is_valid = is_valid and valid
    all_errors.extend(errors)

    if is_valid:
        logger.info("Comprehensive validation passed for conduct scores data")
    else:
        logger.warning(
            f"Comprehensive validation found {len(all_errors)} issues "
            f"for conduct scores data"
        )

    return is_valid, all_errors

