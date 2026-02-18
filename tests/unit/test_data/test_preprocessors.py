"""Unit tests for data preprocessors."""

import numpy as np
import pandas as pd
import pytest

from ml_clo.data.preprocessors import (
    clean_exam_score,
    convert_score_10_to_6,
    create_result_column,
    handle_missing_values,
    preprocess_exam_scores,
    standardize_lecturer_id,
    standardize_student_id,
    standardize_subject_id,
)
from ml_clo.utils.exceptions import DataValidationError


class TestStandardizeStudentID:
    """Test standardize_student_id function."""

    def test_standardize_student_id_success(self):
        """Test successful standardization."""
        df = pd.DataFrame({
            "Student_ID": ["SV001", "SV002", "SV003"],
            "exam_score": [5.0, 6.0, 4.5],
        })

        result = standardize_student_id(df)

        assert len(result) == 3
        assert "Student_ID" in result.columns

    def test_standardize_student_id_with_missing(self):
        """Test standardization with missing values."""
        df = pd.DataFrame({
            "Student_ID": ["SV001", None, "SV003"],
            "exam_score": [5.0, 6.0, 4.5],
        })

        result = standardize_student_id(df, remove_invalid=True)

        # Should remove rows with missing Student_ID
        assert len(result) <= 3


class TestStandardizeSubjectID:
    """Test standardize_subject_id function."""

    def test_standardize_subject_id_success(self):
        """Test successful standardization."""
        df = pd.DataFrame({
            "Subject_ID": ["SUB001", "SUB002", "SUB003"],
            "exam_score": [5.0, 6.0, 4.5],
        })

        result = standardize_subject_id(df)

        assert len(result) == 3
        assert "Subject_ID" in result.columns


class TestStandardizeLecturerID:
    """Test standardize_lecturer_id function."""

    def test_standardize_lecturer_id_success(self):
        """Test successful standardization."""
        df = pd.DataFrame({
            "Lecturer_ID": ["LEC001", "LEC002", "LEC003"],
            "exam_score": [5.0, 6.0, 4.5],
        })

        result = standardize_lecturer_id(df)

        assert len(result) == 3
        assert "Lecturer_ID" in result.columns


class TestCleanExamScore:
    """Test clean_exam_score function."""

    def test_clean_exam_score_success(self):
        """Test successful cleaning."""
        df = pd.DataFrame({
            "exam_score": [5.0, 6.0, 4.5, 10.0, 0.0],
        })

        result = clean_exam_score(df)

        assert len(result) == 5
        assert result["exam_score"].dtype in [np.float64, float]

    def test_clean_exam_score_remove_invalid(self):
        """Test cleaning with invalid values."""
        df = pd.DataFrame({
            "exam_score": [5.0, -1.0, 15.0, None, 6.0],
        })

        result = clean_exam_score(df, remove_invalid=True)

        # Should remove invalid values
        assert len(result) <= 5
        assert (result["exam_score"] >= 0).all()
        assert (result["exam_score"] <= 10).all()


class TestConvertScore10To6:
    """Test convert_score_10_to_6 function."""

    def test_convert_score_10_to_6_success(self):
        """Test successful conversion."""
        df = pd.DataFrame({
            "exam_score": [10.0, 5.0, 0.0],
        })

        result = convert_score_10_to_6(df, score_column="exam_score")

        assert len(result) == 3
        assert result["exam_score"].max() <= 6.0
        assert result["exam_score"].min() >= 0.0
        # Verify conversion formula: CLO_6 = Score_10 / 10 × 6
        assert abs(result["exam_score"].iloc[0] - 6.0) < 0.01
        assert abs(result["exam_score"].iloc[1] - 3.0) < 0.01
        assert abs(result["exam_score"].iloc[2] - 0.0) < 0.01

    def test_convert_score_10_to_6_missing_column(self):
        """Test conversion with missing column."""
        df = pd.DataFrame({
            "other_column": [10.0, 5.0, 0.0],
        })

        with pytest.raises(DataValidationError):
            convert_score_10_to_6(df, score_column="exam_score")


class TestCreateResultColumn:
    """Test create_result_column function."""

    def test_create_result_column_success(self):
        """Test successful result column creation."""
        df = pd.DataFrame({
            "exam_score": [6.0, 5.0, 3.0],
        })

        result = create_result_column(df, pass_threshold=5.0)

        assert "Result" in result.columns
        assert result["Result"].iloc[0] == 1  # Pass
        assert result["Result"].iloc[1] == 1  # Pass
        assert result["Result"].iloc[2] == 0  # Fail


class TestHandleMissingValues:
    """Test handle_missing_values function."""

    def test_handle_missing_values_drop(self):
        """Test dropping missing values."""
        df = pd.DataFrame({
            "col1": [1.0, 2.0, None, 4.0],
            "col2": [1.0, None, 3.0, 4.0],
        })

        result = handle_missing_values(df, strategy="drop", columns=["col1"])

        assert len(result) <= 4
        assert result["col1"].notna().all()

    def test_handle_missing_values_fill(self):
        """Test filling missing values."""
        df = pd.DataFrame({
            "col1": [1.0, 2.0, None, 4.0],
        })

        result = handle_missing_values(df, strategy="fill", columns=["col1"])

        assert len(result) == 4
        assert result["col1"].notna().all()


class TestPreprocessExamScores:
    """Test preprocess_exam_scores function."""

    def test_preprocess_exam_scores_success(self):
        """Test successful preprocessing."""
        df = pd.DataFrame({
            "Student_ID": ["SV001", "SV002", "SV003"],
            "Subject_ID": ["SUB001", "SUB002", "SUB003"],
            "Lecturer_ID": ["LEC001", "LEC002", "LEC003"],
            "exam_score": [10.0, 5.0, 0.0],
        })

        result = preprocess_exam_scores(df, convert_to_clo=True, create_result=True)

        assert len(result) == 3
        assert "exam_score" in result.columns
        assert result["exam_score"].max() <= 6.0  # Converted to 6-point scale
        assert "Result" in result.columns  # Result column created

