"""Data encoding functions.

This module provides functions to encode categorical data for machine learning:
- Teaching method encoding (X → 1, others/NaN → 0)
- Demographics encoding (Gender, Birth place, Ethnicity, Religion)
- ID encoding (Student_ID, Lecturer_ID, Subject_ID)
"""

from typing import Optional

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder

from ml_clo.utils.exceptions import DataValidationError
from ml_clo.utils.logger import get_logger

logger = get_logger(__name__)


def encode_teaching_methods(
    df: pd.DataFrame,
    method_columns: Optional[list[str]] = None,
    prefix: str = "TM",
) -> pd.DataFrame:
    """Encode teaching methods: X → 1, others/NaN → 0.

    Teaching method columns typically contain "X" to indicate the method is used,
    or NaN/empty for not used.

    Args:
        df: DataFrame with teaching method columns
        method_columns: List of column names to encode. If None, auto-detect columns
                       starting with prefix (default: None)
        prefix: Prefix for teaching method columns (default: "TM")

    Returns:
        DataFrame with encoded teaching method columns (0 or 1)

    Raises:
        DataValidationError: If no teaching method columns found
    """
    df = df.copy()

    # Auto-detect teaching method columns if not specified
    if method_columns is None:
        method_columns = [col for col in df.columns if col.startswith(prefix) or "TM" in col]

    if not method_columns:
        raise DataValidationError(
            f"No teaching method columns found with prefix '{prefix}'. "
            f"Available columns: {list(df.columns)}"
        )

    logger.info(f"Encoding {len(method_columns)} teaching method columns")

    # Encode each column: X → 1, others/NaN → 0
    for col in method_columns:
        if col not in df.columns:
            logger.warning(f"Column '{col}' not found, skipping")
            continue

        # Convert to string, then check for "X"
        df[col] = df[col].astype(str).str.strip().str.upper()
        df[col] = (df[col] == "X").astype(int)

        # Count encoded values
        ones = (df[col] == 1).sum()
        logger.debug(f"  {col}: {ones} methods used (out of {len(df)} records)")

    logger.info(f"Successfully encoded teaching methods")
    return df


def encode_assessment_methods(
    df: pd.DataFrame,
    method_columns: Optional[list[str]] = None,
    prefix: str = "EM",
) -> pd.DataFrame:
    """Encode assessment methods: X → 1, others/NaN → 0.

    Assessment method columns typically contain "X" to indicate the method is used,
    or NaN/empty for not used.

    Args:
        df: DataFrame with assessment method columns
        method_columns: List of column names to encode. If None, auto-detect columns
                       starting with prefix (default: None)
        prefix: Prefix for assessment method columns (default: "EM")

    Returns:
        DataFrame with encoded assessment method columns (0 or 1)

    Raises:
        DataValidationError: If no assessment method columns found
    """
    df = df.copy()

    # Auto-detect assessment method columns if not specified
    if method_columns is None:
        method_columns = [col for col in df.columns if col.startswith(prefix) or "EM" in col]

    if not method_columns:
        raise DataValidationError(
            f"No assessment method columns found with prefix '{prefix}'. "
            f"Available columns: {list(df.columns)}"
        )

    logger.info(f"Encoding {len(method_columns)} assessment method columns")

    # Encode each column: X → 1, others/NaN → 0
    for col in method_columns:
        if col not in df.columns:
            logger.warning(f"Column '{col}' not found, skipping")
            continue

        # Convert to string, then check for "X"
        df[col] = df[col].astype(str).str.strip().str.upper()
        df[col] = (df[col] == "X").astype(int)

        # Count encoded values
        ones = (df[col] == 1).sum()
        logger.debug(f"  {col}: {ones} methods used (out of {len(df)} records)")

    logger.info(f"Successfully encoded assessment methods")
    return df


def encode_gender(df: pd.DataFrame, gender_column: str = "Gender") -> pd.DataFrame:
    """Encode gender as binary (0/1).

    Args:
        df: DataFrame with gender column
        gender_column: Name of gender column (default: "Gender")

    Returns:
        DataFrame with encoded gender column

    Raises:
        DataValidationError: If gender column is missing
    """
    if gender_column not in df.columns:
        raise DataValidationError(f"Column '{gender_column}' not found in DataFrame")

    df = df.copy()

    # Convert to numeric if not already
    df[gender_column] = pd.to_numeric(df[gender_column], errors="coerce")

    # Ensure binary (0 or 1)
    # Typically: 0 = Female, 1 = Male (or vice versa, but keep consistent)
    df[gender_column] = df[gender_column].clip(lower=0, upper=1).astype(int)

    logger.info(f"Encoded {gender_column}: {df[gender_column].value_counts().to_dict()}")
    return df


def encode_birth_place_to_region(
    df: pd.DataFrame,
    birth_place_column: str = "place_of_birth",
    output_column: str = "birth_place_region",
) -> pd.DataFrame:
    """Encode birth place to region (Bắc/Trung/Nam).

    Maps Vietnamese provinces/cities to regions:
    - Bắc (North): Hà Nội, Hải Phòng, và các tỉnh phía Bắc
    - Trung (Central): Đà Nẵng, Huế, và các tỉnh miền Trung
    - Nam (South): TP.HCM, Cần Thơ, và các tỉnh phía Nam

    Args:
        df: DataFrame with birth place column
        birth_place_column: Name of birth place column (default: "place_of_birth")
        output_column: Name of output region column (default: "birth_place_region")

    Returns:
        DataFrame with region column added

    Raises:
        DataValidationError: If birth place column is missing
    """
    if birth_place_column not in df.columns:
        raise DataValidationError(f"Column '{birth_place_column}' not found in DataFrame")

    df = df.copy()

    # Define region mapping (common Vietnamese provinces/cities)
    # This is a simplified mapping - may need to be expanded based on actual data
    region_mapping = {
        # Bắc (North)
        "hà nội": "Bắc",
        "hanoi": "Bắc",
        "hải phòng": "Bắc",
        "haiphong": "Bắc",
        "quảng ninh": "Bắc",
        "hải dương": "Bắc",
        "hưng yên": "Bắc",
        "thái bình": "Bắc",
        "nam định": "Bắc",
        "ninh bình": "Bắc",
        "vĩnh phúc": "Bắc",
        "bắc ninh": "Bắc",
        "bắc giang": "Bắc",
        "phú thọ": "Bắc",
        "thái nguyên": "Bắc",
        "lào cai": "Bắc",
        "yên bái": "Bắc",
        "tuyên quang": "Bắc",
        "cao bằng": "Bắc",
        "lạng sơn": "Bắc",
        "bắc kạn": "Bắc",
        "hà giang": "Bắc",
        "điện biên": "Bắc",
        "sơn la": "Bắc",
        "hoà bình": "Bắc",
        # Trung (Central)
        "đà nẵng": "Trung",
        "danang": "Trung",
        "huế": "Trung",
        "hue": "Trung",
        "quảng nam": "Trung",
        "quảng ngãi": "Trung",
        "bình định": "Trung",
        "phú yên": "Trung",
        "khánh hòa": "Trung",
        "nha trang": "Trung",
        "ninh thuận": "Trung",
        "bình thuận": "Trung",
        "quảng bình": "Trung",
        "quảng trị": "Trung",
        "thừa thiên huế": "Trung",
        "kon tum": "Trung",
        "gia lai": "Trung",
        "đắk lắk": "Trung",
        "đắk nông": "Trung",
        "lâm đồng": "Trung",
        "đà lạt": "Trung",
        # Nam (South)
        "tp.hcm": "Nam",
        "hồ chí minh": "Nam",
        "ho chi minh": "Nam",
        "cần thơ": "Nam",
        "can tho": "Nam",
        "bình dương": "Nam",
        "đồng nai": "Nam",
        "bà rịa - vũng tàu": "Nam",
        "vũng tàu": "Nam",
        "tây ninh": "Nam",
        "bình phước": "Nam",
        "long an": "Nam",
        "tiền giang": "Nam",
        "bến tre": "Nam",
        "trà vinh": "Nam",
        "vĩnh long": "Nam",
        "đồng tháp": "Nam",
        "an giang": "Nam",
        "kiên giang": "Nam",
        "cà mau": "Nam",
        "bạc liêu": "Nam",
        "sóc trăng": "Nam",
        "hậu giang": "Nam",
    }

    # Normalize birth place values (lowercase, remove accents if needed)
    df[output_column] = df[birth_place_column].astype(str).str.lower().str.strip()

    # Map to regions
    df[output_column] = df[output_column].map(region_mapping)

    # For unmapped values, try to infer from common patterns
    unmapped = df[output_column].isna()
    if unmapped.sum() > 0:
        logger.warning(
            f"Could not map {unmapped.sum()} birth places to regions. "
            f"Setting to 'Unknown'"
        )
        df.loc[unmapped, output_column] = "Unknown"

    # Encode regions as integers: Bắc=0, Trung=1, Nam=2, Unknown=3
    region_encoding = {"Bắc": 0, "Trung": 1, "Nam": 2, "Unknown": 3}
    df[output_column] = df[output_column].map(region_encoding).astype(int)

    logger.info(
        f"Encoded {birth_place_column} to {output_column}: "
        f"{df[output_column].value_counts().to_dict()}"
    )

    return df


def encode_ethnicity(
    df: pd.DataFrame,
    ethnicity_column: str = "Ethnicity",
    output_column: Optional[str] = None,
) -> pd.DataFrame:
    """Encode ethnicity as integer.

    Uses LabelEncoder to assign unique integer codes to each ethnicity.

    Args:
        df: DataFrame with ethnicity column
        ethnicity_column: Name of ethnicity column (default: "Ethnicity")
        output_column: Name of output column. If None, overwrites ethnicity_column (default: None)

    Returns:
        DataFrame with encoded ethnicity

    Raises:
        DataValidationError: If ethnicity column is missing
    """
    if ethnicity_column not in df.columns:
        raise DataValidationError(f"Column '{ethnicity_column}' not found in DataFrame")

    df = df.copy()
    output_col = output_column if output_column else ethnicity_column

    # Use LabelEncoder
    le = LabelEncoder()
    df[output_col] = le.fit_transform(df[ethnicity_column].astype(str).fillna("Unknown"))

    logger.info(
        f"Encoded {ethnicity_column} to {output_col}: "
        f"{len(le.classes_)} unique values"
    )

    return df


def encode_religion(
    df: pd.DataFrame,
    religion_column: str = "Religion",
    output_column: Optional[str] = None,
) -> pd.DataFrame:
    """Encode religion as integer.

    Uses LabelEncoder to assign unique integer codes to each religion.

    Args:
        df: DataFrame with religion column
        religion_column: Name of religion column (default: "Religion")
        output_column: Name of output column. If None, overwrites religion_column (default: None)

    Returns:
        DataFrame with encoded religion

    Raises:
        DataValidationError: If religion column is missing
    """
    if religion_column not in df.columns:
        raise DataValidationError(f"Column '{religion_column}' not found in DataFrame")

    df = df.copy()
    output_col = output_column if output_column else religion_column

    # Use LabelEncoder
    le = LabelEncoder()
    df[output_col] = le.fit_transform(df[religion_column].astype(str).fillna("Unknown"))

    logger.info(
        f"Encoded {religion_column} to {output_col}: "
        f"{len(le.classes_)} unique values"
    )

    return df


def encode_student_id(
    df: pd.DataFrame,
    id_column: str = "Student_ID",
    output_column: Optional[str] = None,
    method: str = "label",
) -> pd.DataFrame:
    """Encode Student_ID for machine learning.

    Args:
        df: DataFrame with Student_ID column
        id_column: Name of student ID column (default: "Student_ID")
        output_column: Name of output column. If None, creates "{id_column}_encoded" (default: None)
        method: Encoding method: "label" (LabelEncoder) or "hash" (hash-based) (default: "label")

    Returns:
        DataFrame with encoded Student_ID

    Raises:
        DataValidationError: If id_column is missing or method is invalid
    """
    if id_column not in df.columns:
        raise DataValidationError(f"Column '{id_column}' not found in DataFrame")

    if method not in ["label", "hash"]:
        raise DataValidationError(f"Invalid encoding method: {method}. Use 'label' or 'hash'")

    df = df.copy()
    output_col = output_column if output_column else f"{id_column}_encoded"

    if method == "label":
        le = LabelEncoder()
        df[output_col] = le.fit_transform(df[id_column])
        logger.info(
            f"Encoded {id_column} to {output_col} using LabelEncoder: "
            f"{len(le.classes_)} unique students"
        )
    else:  # hash
        # Hash-based encoding (for very large datasets)
        df[output_col] = df[id_column].apply(lambda x: hash(str(x)) % (10**9))
        logger.info(f"Encoded {id_column} to {output_col} using hash-based encoding")

    return df


def encode_lecturer_id(
    df: pd.DataFrame,
    id_column: str = "Lecturer_ID",
    output_column: Optional[str] = None,
    method: str = "label",
) -> pd.DataFrame:
    """Encode Lecturer_ID for machine learning.

    Args:
        df: DataFrame with Lecturer_ID column
        id_column: Name of lecturer ID column (default: "Lecturer_ID")
        output_column: Name of output column. If None, creates "{id_column}_encoded" (default: None)
        method: Encoding method: "label" (LabelEncoder) or "hash" (hash-based) (default: "label")

    Returns:
        DataFrame with encoded Lecturer_ID

    Raises:
        DataValidationError: If id_column is missing or method is invalid
    """
    if id_column not in df.columns:
        raise DataValidationError(f"Column '{id_column}' not found in DataFrame")

    if method not in ["label", "hash"]:
        raise DataValidationError(f"Invalid encoding method: {method}. Use 'label' or 'hash'")

    df = df.copy()
    output_col = output_column if output_column else f"{id_column}_encoded"

    if method == "label":
        le = LabelEncoder()
        df[output_col] = le.fit_transform(df[id_column].astype(str))
        logger.info(
            f"Encoded {id_column} to {output_col} using LabelEncoder: "
            f"{len(le.classes_)} unique lecturers"
        )
    else:  # hash
        df[output_col] = df[id_column].astype(str).apply(lambda x: hash(x) % (10**9))
        logger.info(f"Encoded {id_column} to {output_col} using hash-based encoding")

    return df


def encode_subject_id(
    df: pd.DataFrame,
    id_column: str = "Subject_ID",
    output_column: Optional[str] = None,
    method: str = "label",
) -> pd.DataFrame:
    """Encode Subject_ID for machine learning.

    Args:
        df: DataFrame with Subject_ID column
        id_column: Name of subject ID column (default: "Subject_ID")
        output_column: Name of output column. If None, creates "{id_column}_encoded" (default: None)
        method: Encoding method: "label" (LabelEncoder) or "hash" (hash-based) (default: "label")

    Returns:
        DataFrame with encoded Subject_ID

    Raises:
        DataValidationError: If id_column is missing or method is invalid
    """
    if id_column not in df.columns:
        raise DataValidationError(f"Column '{id_column}' not found in DataFrame")

    if method not in ["label", "hash"]:
        raise DataValidationError(f"Invalid encoding method: {method}. Use 'label' or 'hash'")

    df = df.copy()
    output_col = output_column if output_column else f"{id_column}_encoded"

    if method == "label":
        le = LabelEncoder()
        df[output_col] = le.fit_transform(df[id_column].astype(str))
        logger.info(
            f"Encoded {id_column} to {output_col} using LabelEncoder: "
            f"{len(le.classes_)} unique subjects"
        )
    else:  # hash
        df[output_col] = df[id_column].astype(str).apply(lambda x: hash(x) % (10**9))
        logger.info(f"Encoded {id_column} to {output_col} using hash-based encoding")

    return df

