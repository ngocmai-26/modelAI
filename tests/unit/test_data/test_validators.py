"""Unit tests for data validators."""

import pandas as pd
import pytest

from ml_clo.data.validators import (
    validate_clo_score_range,
    validate_conduct_score_range,
    validate_data_consistency,
    validate_data_types,
    validate_no_missing_values,
    validate_required_fields,
    validate_ranges,
)
from ml_clo.utils.exceptions import DataValidationError


class TestValidateDataTypes:
    """Test validate_data_types function."""

    def test_validate_data_types_success(self):
        """Test successful data type validation."""
        df = pd.DataFrame({
            "col1": [1, 2, 3],
            "col2": [1.0, 2.0, 3.0],
            "col3": ["a", "b", "c"],
        })

        schema = {
            "col1": "int64",
            "col2": "float64",
            "col3": "object",
        }

        # Should not raise
        validate_data_types(df, schema)

    def test_validate_data_types_failure(self):
        """Test data type validation failure."""
        df = pd.DataFrame({
            "col1": ["a", "b", "c"],  # Should be int
        })

        schema = {
            "col1": "int64",
        }

        with pytest.raises(DataValidationError):
            validate_data_types(df, schema)


class TestValidateRanges:
    """Test validate_ranges function."""

    def test_validate_ranges_success(self):
        """Test successful range validation."""
        df = pd.DataFrame({
            "score": [1.0, 2.0, 3.0, 4.0, 5.0],
        })

        ranges = {
            "score": (0.0, 6.0),
        }

        # Should not raise
        validate_ranges(df, ranges)

    def test_validate_ranges_failure(self):
        """Test range validation failure."""
        df = pd.DataFrame({
            "score": [7.0, 8.0],  # Out of range
        })

        ranges = {
            "score": (0.0, 6.0),
        }

        with pytest.raises(DataValidationError):
            validate_ranges(df, ranges)


class TestValidateRequiredFields:
    """Test validate_required_fields function."""

    def test_validate_required_fields_success(self):
        """Test successful required fields validation."""
        df = pd.DataFrame({
            "Student_ID": ["SV001", "SV002"],
            "Subject_ID": ["SUB001", "SUB002"],
            "exam_score": [5.0, 6.0],
        })

        required = ["Student_ID", "Subject_ID", "exam_score"]

        # Should not raise
        validate_required_fields(df, required)

    def test_validate_required_fields_failure(self):
        """Test required fields validation failure."""
        df = pd.DataFrame({
            "Student_ID": ["SV001", "SV002"],
        })

        required = ["Student_ID", "Subject_ID"]

        with pytest.raises(DataValidationError):
            validate_required_fields(df, required)


class TestValidateNoMissingValues:
    """Test validate_no_missing_values function."""

    def test_validate_no_missing_values_success(self):
        """Test successful missing values validation."""
        df = pd.DataFrame({
            "col1": [1.0, 2.0, 3.0],
        })

        # Should not raise
        validate_no_missing_values(df, ["col1"])

    def test_validate_no_missing_values_failure(self):
        """Test missing values validation failure."""
        df = pd.DataFrame({
            "col1": [1.0, None, 3.0],
        })

        with pytest.raises(DataValidationError):
            validate_no_missing_values(df, ["col1"])


class TestValidateCloScoreRange:
    """Test validate_clo_score_range function."""

    def test_validate_clo_score_range_success(self):
        """Test successful CLO score range validation."""
        df = pd.DataFrame({
            "exam_score": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
        })

        # Should not raise
        validate_clo_score_range(df)

    def test_validate_clo_score_range_failure(self):
        """Test CLO score range validation failure."""
        df = pd.DataFrame({
            "exam_score": [7.0, 8.0],  # Out of range
        })

        with pytest.raises(DataValidationError):
            validate_clo_score_range(df)


class TestValidateConductScoreRange:
    """Test validate_conduct_score_range function."""

    def test_validate_conduct_score_range_success(self):
        """Test successful conduct score range validation."""
        df = pd.DataFrame({
            "conduct_score": [70.0, 80.0, 90.0, 100.0],
        })

        # Should not raise
        validate_conduct_score_range(df)

    def test_validate_conduct_score_range_failure(self):
        """Test conduct score range validation failure."""
        df = pd.DataFrame({
            "conduct_score": [150.0],  # Out of range
        })

        with pytest.raises(DataValidationError):
            validate_conduct_score_range(df)


class TestValidateDataConsistency:
    """Test validate_data_consistency function."""

    def test_validate_data_consistency_success(self):
        """Test successful data consistency validation."""
        df1 = pd.DataFrame({
            "Student_ID": ["SV001", "SV002"],
            "exam_score": [5.0, 6.0],
        })

        df2 = pd.DataFrame({
            "Student_ID": ["SV001", "SV002"],
            "conduct_score": [85.0, 90.0],
        })

        # Should not raise
        validate_data_consistency(df1, df2, key_column="Student_ID")

