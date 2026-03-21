"""Unit tests for data loaders."""

import pandas as pd
import pytest
from pathlib import Path

from ml_clo.data.loaders import (
    load_assessment_methods,
    load_attendance,
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

    def test_load_demographics_mssv_maps_to_student_id(self, tmp_path):
        """Test MSSV -> Student_ID mapping khi file chỉ có MSSV (yêu cầu mới)."""
        demo_path = tmp_path / "demo_mssv.xlsx"
        df = pd.DataFrame({"MSSV": [19050006, 19050007], "Gender": [1, 0]})
        df.to_excel(demo_path, index=False)
        result = load_demographics(str(demo_path))
        assert "Student_ID" in result.columns
        assert list(result["Student_ID"]) == [19050006, 19050007]

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


class TestLoadAttendance:
    """Test load_attendance function (Bước 1, 2: điểm danh train/predict)."""

    def test_load_attendance_success(self):
        """Test successful loading of attendance."""
        att_files = list(Path("data").glob("Dữ liệu điểm danh*.xlsx")) if Path("data").exists() else []
        file_path = str(att_files[0]) if att_files else "data/Dữ liệu điểm danh Khoa FIRA.xlsx"
        if not Path(file_path).exists():
            pytest.skip(f"Attendance file not found: {file_path}")

        df = load_attendance(file_path)

        assert df is not None
        assert len(df) > 0
        # Cần có MSSV và cột điểm danh (hoặc tương đương)
        has_key = "MSSV" in df.columns or "Student_ID" in df.columns
        assert has_key, f"Expected MSSV/Student_ID; columns: {list(df.columns)}"

    def test_load_attendance_file_not_found(self):
        """Test loading with non-existent file."""
        from ml_clo.utils.exceptions import DataLoadError

        with pytest.raises(DataLoadError):
            load_attendance("nonexistent_attendance.xlsx")

