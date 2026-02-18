"""Unit tests for data loaders."""

import pytest
from pathlib import Path

from ml_clo.data.loaders import (
    load_assessment_methods,
    load_conduct_scores,
    load_demographics,
    load_exam_scores,
    load_study_hours,
    load_teaching_methods,
)
from ml_clo.utils.exceptions import DataLoadError


class TestLoadExamScores:
    """Test load_exam_scores function."""

    def test_load_exam_scores_success(self):
        """Test successful loading of exam scores."""
        file_path = "data/DiemTong.xlsx"
        if not Path(file_path).exists():
            pytest.skip(f"Data file not found: {file_path}")

        df = load_exam_scores(file_path)

        assert df is not None
        assert len(df) > 0
        assert "Student_ID" in df.columns or "student_id" in df.columns.lower()
        assert "exam_score" in df.columns or "score" in df.columns.lower()

    def test_load_exam_scores_file_not_found(self):
        """Test loading with non-existent file."""
        with pytest.raises(DataLoadError):
            load_exam_scores("nonexistent_file.xlsx")


class TestLoadConductScores:
    """Test load_conduct_scores function."""

    def test_load_conduct_scores_success(self):
        """Test successful loading of conduct scores."""
        file_path = "data/diemrenluyen.xlsx"
        if not Path(file_path).exists():
            pytest.skip(f"Data file not found: {file_path}")

        df = load_conduct_scores(file_path)

        assert df is not None
        assert len(df) > 0


class TestLoadDemographics:
    """Test load_demographics function."""

    def test_load_demographics_success(self):
        """Test successful loading of demographics."""
        file_path = "data/nhankhau.xlsx"
        if not Path(file_path).exists():
            pytest.skip(f"Data file not found: {file_path}")

        df = load_demographics(file_path)

        assert df is not None
        assert len(df) > 0


class TestLoadTeachingMethods:
    """Test load_teaching_methods function."""

    def test_load_teaching_methods_success(self):
        """Test successful loading of teaching methods."""
        file_path = "data/PPGDfull.xlsx"
        if not Path(file_path).exists():
            pytest.skip(f"Data file not found: {file_path}")

        df = load_teaching_methods(file_path)

        assert df is not None
        assert len(df) > 0


class TestLoadAssessmentMethods:
    """Test load_assessment_methods function."""

    def test_load_assessment_methods_success(self):
        """Test successful loading of assessment methods."""
        file_path = "data/PPDGfull.xlsx"
        if not Path(file_path).exists():
            pytest.skip(f"Data file not found: {file_path}")

        df = load_assessment_methods(file_path)

        assert df is not None
        assert len(df) > 0


class TestLoadStudyHours:
    """Test load_study_hours function."""

    def test_load_study_hours_success(self):
        """Test successful loading of study hours."""
        file_path = "data/tuhoc.xlsx"
        if not Path(file_path).exists():
            pytest.skip(f"Data file not found: {file_path}")

        df = load_study_hours(file_path)

        assert df is not None
        assert len(df) > 0

