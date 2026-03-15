"""Unit tests for data mergers.

Yêu cầu mới (Bước 5): create_student_record_from_ids, encoding unknown.
"""

import pandas as pd
import pytest

from ml_clo.data.mergers import (
    LECTURER_PLACEHOLDER,
    create_student_record_from_ids,
)


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
