"""Unit tests for data encoders."""

import pandas as pd
import pytest

from ml_clo.data.encoders import (
    encode_assessment_methods,
    encode_ethnicity,
    encode_gender,
    encode_teaching_methods,
)


class TestEncodeGender:
    """Test encode_gender function."""

    def test_encode_gender_success(self):
        """Test successful gender encoding."""
        df = pd.DataFrame({
            "Gender": ["M", "F", "Male", "Female", "Nam", "Nữ"],
        })

        result = encode_gender(df)

        assert "Gender_encoded" in result.columns
        assert result["Gender_encoded"].dtype in [int, "int64"]

    def test_encode_gender_binary(self):
        """Test binary encoding."""
        df = pd.DataFrame({
            "Gender": ["M", "F"],
        })

        result = encode_gender(df)

        # Should have binary values (0 or 1)
        assert set(result["Gender_encoded"].unique()).issubset({0, 1})


class TestEncodeTeachingMethods:
    """Test encode_teaching_methods function."""

    def test_encode_teaching_methods_success(self):
        """Test successful teaching methods encoding."""
        df = pd.DataFrame({
            "Subject_ID": ["SUB001", "SUB002"],
            "TM1": ["X", None],
            "TM2": [None, "X"],
        })

        result = encode_teaching_methods(df)

        # Should have encoded columns
        assert "TM1_encoded" in result.columns or "TM1" in result.columns
        assert result.select_dtypes(include=["int", "float64"]).shape[1] > 0


class TestEncodeAssessmentMethods:
    """Test encode_assessment_methods function."""

    def test_encode_assessment_methods_success(self):
        """Test successful assessment methods encoding."""
        df = pd.DataFrame({
            "Subject_ID": ["SUB001", "SUB002"],
            "AM1": ["X", None],
            "AM2": [None, "X"],
        })

        result = encode_assessment_methods(df)

        # Should have encoded columns
        assert result.select_dtypes(include=["int", "float64"]).shape[1] > 0


class TestEncodeEthnicity:
    """Test encode_ethnicity function."""

    def test_encode_ethnicity_success(self):
        """Test successful ethnicity encoding."""
        df = pd.DataFrame({
            "Ethnicity": ["Kinh", "Tày", "Mường"],
        })

        result = encode_ethnicity(df)

        assert "Ethnicity_encoded" in result.columns or "Ethnicity" in result.columns
        assert result.select_dtypes(include=["int", "float64"]).shape[1] > 0

