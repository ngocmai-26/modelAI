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
    """Test encode_gender function (encoder expects numeric 0/1 input)."""

    def test_encode_gender_success(self):
        """Test successful gender encoding."""
        df = pd.DataFrame({"Gender": [1, 0, 1, 0, 1, 0]})
        result = encode_gender(df)
        assert "Gender" in result.columns
        assert result["Gender"].dtype in [int, "int64", "int32"]

    def test_encode_gender_binary(self):
        """Test binary encoding."""
        df = pd.DataFrame({"Gender": [1, 0]})
        result = encode_gender(df)
        assert set(result["Gender"].unique()).issubset({0, 1})


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
    """Test encode_assessment_methods function (encoder expects EM prefix)."""

    def test_encode_assessment_methods_success(self):
        """Test successful assessment methods encoding."""
        df = pd.DataFrame({
            "Subject_ID": ["SUB001", "SUB002"],
            "EM1": ["X", None],
            "EM2": [None, "X"],
        })
        result = encode_assessment_methods(df)
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

