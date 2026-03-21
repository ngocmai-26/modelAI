"""Unit tests for data mergers.

Yêu cầu mới (Bước 5): create_student_record_from_ids, encoding unknown.
"""

import pandas as pd
import pytest

from ml_clo.data.mergers import (
    LECTURER_PLACEHOLDER,
    create_student_record_from_ids,
    merge_attendance,
    student_has_history,
)


class TestStudentHasHistory:
    """Unit test cho student_has_history (Bước 3: Logic SV năm 1 vs 2+)."""

    def test_has_history_true(self):
        """SV có trong DiemTong → True."""
        exam_df = pd.DataFrame({
            "Student_ID": [19050006, 19050007],
            "Subject_ID": ["INF0823", "INF0824"],
            "Lecturer_ID": ["90316", "90317"],
        })
        assert student_has_history(exam_df, "19050006") == True
        assert student_has_history(exam_df, 19050007) == True

    def test_has_history_false(self):
        """SV không có trong DiemTong → False."""
        exam_df = pd.DataFrame({
            "Student_ID": [19050006],
            "Subject_ID": ["INF0823"],
            "Lecturer_ID": ["90316"],
        })
        assert student_has_history(exam_df, "19050099") == False

    def test_empty_or_none(self):
        """exam_df rỗng hoặc None → False."""
        assert student_has_history(pd.DataFrame(), "19050006") == False
        assert student_has_history(None, "19050006") == False


class TestCreateStudentRecordFromIds:
    """Unit test cho create_student_record_from_ids (yêu cầu mới: SV/môn/GV không cần DiemTong)."""

    def test_create_record_minimal(self):
        """Tạo record khi không có demographics, PPGD, PPDG."""
        result = create_student_record_from_ids(
            student_id="19050006",
            subject_id="INF0823",
            lecturer_id="90316",
        )
        assert len(result) == 1
        assert result["Student_ID"].iloc[0] == 19050006
        assert result["Subject_ID"].iloc[0] == "INF0823"
        assert result["Lecturer_ID"].iloc[0] == "90316"
        assert "exam_score" in result.columns
        assert pd.isna(result["exam_score"].iloc[0])

    def test_create_record_with_demographics(self):
        """Tạo record có merge nhân khẩu."""
        demo = pd.DataFrame({
            "Student_ID": [19050006],
            "Gender": [1],
            "place_of_birth": ["Bình Dương"],
        })
        result = create_student_record_from_ids(
            student_id="19050006",
            subject_id="INF0823",
            lecturer_id="90316",
            demographics_df=demo,
        )
        assert len(result) == 1
        assert "Gender" in result.columns
        assert result["Gender"].iloc[0] == 1

    def test_create_record_lecturer_placeholder(self):
        """GV mới/trống → Lecturer_ID = __UNKNOWN__ (encoding unknown)."""
        result = create_student_record_from_ids(
            student_id="19050007",
            subject_id="INF0823",
            lecturer_id="",
        )
        assert result["Lecturer_ID"].iloc[0] == LECTURER_PLACEHOLDER
        assert LECTURER_PLACEHOLDER == "__UNKNOWN__"

    def test_create_record_with_ppgd_ppdg(self):
        """Tạo record có merge PPGD và PPDG."""
        ppgd = pd.DataFrame({"Subject_ID": ["INF0823"], "TM1": ["X"], "TM2": [""]})
        ppdg = pd.DataFrame({"Subject_ID": ["INF0823"], "EM1": ["X"], "EM2": [""]})
        result = create_student_record_from_ids(
            student_id="19050008",
            subject_id="INF0823",
            lecturer_id="90316",
            teaching_methods_df=ppgd,
            assessment_methods_df=ppdg,
        )
        assert len(result) == 1
        assert "TM1" in result.columns or any("TM" in c for c in result.columns)
        assert "EM1" in result.columns or any("EM" in c for c in result.columns)

    def test_create_record_numeric_student_id(self):
        """Student_id có thể là int hoặc str số."""
        result = create_student_record_from_ids(
            student_id=19050009,
            subject_id="INF0823",
            lecturer_id="90316",
        )
        assert result["Student_ID"].iloc[0] == 19050009


class TestMergeAttendance:
    """Unit test cho merge_attendance (Bước 1, 2: điểm danh train/predict)."""

    def test_merge_attendance_adds_attendance_rate(self):
        """Merge attendance thêm cột attendance_rate."""
        exam_df = pd.DataFrame({
            "Student_ID": [19050006, 19050007],
            "Subject_ID": ["INF0823", "INF0823"],
            "Lecturer_ID": ["90316", "90316"],
            "year": [2024, 2024],
        })
        att_df = pd.DataFrame({
            "MSSV": [19050006, 19050007],
            "Mã môn học": ["INF0823", "INF0823"],
            "Niên khoá": ["2024-2025", "2024-2025"],
            "Điểm danh": ["Có", "Vắng"],
        })
        merged = merge_attendance(exam_df, att_df)
        assert "attendance_rate" in merged.columns
        assert len(merged) == 2

    def test_merge_attendance_missing_main_cols_raises(self):
        """Thiếu cột trong main df → DataValidationError."""
        from ml_clo.utils.exceptions import DataValidationError

        exam_df = pd.DataFrame({"Student_ID": [1], "Subject_ID": ["X"]})  # thiếu year
        att_df = pd.DataFrame({
            "MSSV": [1],
            "Mã môn học": ["X"],
            "Niên khoá": ["2024"],
        })
        with pytest.raises(DataValidationError):
            merge_attendance(exam_df, att_df)

    def test_merge_attendance_year_column_param(self):
        """Có thể chỉ định tên cột year khác."""
        exam_df = pd.DataFrame({
            "Student_ID": [19050006],
            "Subject_ID": ["INF0823"],
            "Lecturer_ID": ["90316"],
            "semester_year": [2024],
        })
        att_df = pd.DataFrame({
            "MSSV": [19050006],
            "Mã môn học": ["INF0823"],
            "Niên khoá": ["2024"],
            "Điểm danh": ["Có"],
        })
        merged = merge_attendance(exam_df, att_df, year_column="semester_year")
        assert "attendance_rate" in merged.columns
