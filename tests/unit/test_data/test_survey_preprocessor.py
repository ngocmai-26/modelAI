"""Tests for ml_clo.data.survey_preprocessor (Phase 1)."""

import numpy as np
import pandas as pd
import pytest

from ml_clo.data.survey_preprocessor import preprocess_survey


def _minimal_raw_row(student_id="19050006", year="2024-2025", semester="Học kì 1"):
    """Build a single raw survey row with all required columns."""
    return {
        "Dấu thời gian": "2025-01-01 12:00:00",
        "Địa chỉ email": "x@bdu.edu.vn",
        "1.1. Mã sinh viên": student_id,
        "1.2. Năm học": year,
        "1.3. Học kỳ": semester,
        "1.3 . Sinh viên năm": "Năm 2",
        "2.1. Nơi cư trú của gia đình": "Thành thị",
        "2.2. Nơi ở hiện tại trong thời gian học": "Nhà trọ",
        "2.3. Mức độ hỗ trợ tài chính từ gia đình cho việc học": "Đủ một phần",
        "2.4. Trong học kỳ này, bạn có gặp khó khăn tài chính không?": "Ít",
        "2.5. Mức độ hỗ trợ tinh thần từ gia đình đối với việc học": "Cao",
        "2.6. Mức độ kỳ vọng của gia đình về kết quả học tập": "Cao",
        "2.7. Thu nhập gia đình hàng tháng (ước tính)": "10 – 15 triệu",
        "2.8. Bạn có phải là lao động chính trong gia đình không?": "Không",
        "2.9. Học phí của bạn trong học kỳ này ở mức nào?": "Trung bình",
        "2.10. Ai là người chi trả chính học phí cho bạn trong học kỳ này?": "Gia đình chi trả hoàn toàn",
        "2.11. Việc đóng học phí trong học kỳ này gây áp lực cho bạn ở mức nào?": "Trung bình",
        "2.12. Học phí đã từng ảnh hưởng đến các quyết định sau của bạn trong học kỳ này chưa?": "Không ảnh hưởng",
        "3.1. Trong học kỳ này, bạn có đi làm thêm không?": "Có",
        "3.2. Số giờ làm thêm trung bình mỗi tuần ": "10 – 20 giờ",
        "3.3. Mục đích chính của việc làm thêm": "Tích lũy kinh nghiệm",
        "3.4. Việc làm thêm ảnh hưởng đến việc học của bạn như thế nào?": "Tiêu cực",
        "3.5. Loại công việc làm thêm": "Gia sư / Dạy kèm",
        "3.6. Thu nhập làm thêm mỗi tháng (ước tính)": "2 – 5 triệu",
        "4.1. Mức độ áp lực học tập": "Cao",
        "4.2. Bạn gặp những vấn đề nào sau đây?": "Căng thẳng / stress, Lo âu",
        "4.3. Bạn có từng nghĩ đến việc bỏ học / bảo lưu trong học kỳ này không?": "Đã từng nghĩ",
        "4.4. Số giờ ngủ trung bình mỗi đêm": "6 – 7 giờ",
        "5.1. Bạn có không gian học tập yên tĩnh, ổn định không?": "Có",
        "5.2. Thời gian tự học trung bình mỗi ngày": "1 – 2 giờ",
        "5.3. Khi gặp khó khăn học tập, bạn thường:": "Tự tìm tài liệu, Hỏi bạn bè",
        "5.4. Phương pháp học bạn thường sử dụng ": "Đọc và ghi nhớ, Làm bài tập nhiều",
        "5.5. Kỹ năng quản lý thời gian của bạn": "Trung bình",
        "5.6. Bạn có lập kế hoạch học tập cho từng tuần/tháng không?": "Thỉnh thoảng",
        "5.7. Bạn có thường xuyên trao đổi với bạn bè về bài học không?": "Thường xuyên",
        "5.8. Bạn có tham gia nhóm học tập thường xuyên không?": "Có, thỉnh thoảng",
        "5.9. Bạn có cảm thấy áp lực từ bạn bè về việc học không?": "Trung bình",
        "6.1. Bạn có thiết bị học tập ổn định (laptop/máy tính) không?": "Có, đầy đủ",
        "6.2. Chất lượng kết nối internet của bạn": "Tốt",
        "6.3. Bạn có thường xuyên sử dụng LMS (hệ thống quản lý học tập) không?": "Thường xuyên",
        "6.4. Bạn có thường xuyên sử dụng các công cụ học tập online khác không? (Zoom, Teams, Google Meet, v.v.)": "Thỉnh thoảng",
        "6.5. Bạn có dễ dàng truy cập tài liệu học tập online (thư viện số, tài liệu điện tử) không?": "Dễ dàng",
    }


class TestPreprocessSurvey:
    def test_returns_empty_for_empty_input(self):
        result = preprocess_survey(pd.DataFrame())
        assert len(result) == 0
        assert {"Student_ID", "year", "semester"}.issubset(result.columns)

    def test_returns_none_safe(self):
        result = preprocess_survey(None)
        assert len(result) == 0

    def test_keys_present_after_preprocess(self):
        df = pd.DataFrame([_minimal_raw_row()])
        out = preprocess_survey(df)
        assert "Student_ID" in out.columns
        assert "year" in out.columns
        assert "semester" in out.columns
        assert len(out) == 1

    def test_semester_extracted_as_int(self):
        df = pd.DataFrame([_minimal_raw_row(semester="Học kì 2")])
        out = preprocess_survey(df)
        assert out.iloc[0]["semester"] == 2

    def test_likert_encoding_ordinal(self):
        # Higher Likert text → higher numeric
        rows = [
            _minimal_raw_row(student_id="A"),
            _minimal_raw_row(student_id="B"),
        ]
        rows[0]["4.1. Mức độ áp lực học tập"] = "Rất thấp"
        rows[1]["4.1. Mức độ áp lực học tập"] = "Rất cao"
        out = preprocess_survey(pd.DataFrame(rows))
        a_press = out[out["Student_ID"] == "A"]["survey_study_pressure"].iloc[0]
        b_press = out[out["Student_ID"] == "B"]["survey_study_pressure"].iloc[0]
        assert a_press < b_press

    def test_multihot_mental_issues_expanded(self):
        df = pd.DataFrame([_minimal_raw_row()])
        out = preprocess_survey(df)
        assert "survey_mh_stress" in out.columns
        assert "survey_mh_anxiety" in out.columns
        # Raw "Căng thẳng / stress, Lo âu" should set both flags
        assert out.iloc[0]["survey_mh_stress"] == 1
        assert out.iloc[0]["survey_mh_anxiety"] == 1
        assert out.iloc[0]["survey_mh_health"] == 0

    def test_income_range_midpoint_extracted(self):
        df = pd.DataFrame([_minimal_raw_row()])
        out = preprocess_survey(df)
        # "10 – 15 triệu" → midpoint 12.5
        assert out.iloc[0]["survey_family_income_trieu"] == pytest.approx(12.5)

    def test_sleep_hours_midpoint(self):
        df = pd.DataFrame([_minimal_raw_row()])
        out = preprocess_survey(df)
        # "6 – 7 giờ" → midpoint 6.5
        assert out.iloc[0]["survey_sleep_hours"] == pytest.approx(6.5)

    def test_unknown_value_yields_nan(self):
        row = _minimal_raw_row()
        row["4.1. Mức độ áp lực học tập"] = "Some new label not in mapping"
        out = preprocess_survey(pd.DataFrame([row]))
        assert pd.isna(out.iloc[0]["survey_study_pressure"])

    def test_dedup_keeps_latest_per_key(self):
        # Same (student, year, semester) but different sleep — keep last
        a = _minimal_raw_row(student_id="X")
        a["4.4. Số giờ ngủ trung bình mỗi đêm"] = "< 5 giờ"
        b = _minimal_raw_row(student_id="X")
        b["4.4. Số giờ ngủ trung bình mỗi đêm"] = "> 8 giờ"
        out = preprocess_survey(pd.DataFrame([a, b]))
        assert len(out) == 1
        # Last (8.5) should win
        assert out.iloc[0]["survey_sleep_hours"] == pytest.approx(8.5)

    def test_drops_rows_missing_keys(self):
        a = _minimal_raw_row(student_id="X")
        b = _minimal_raw_row(student_id=None)  # Missing student_id
        out = preprocess_survey(pd.DataFrame([a, b]))
        assert len(out) == 1
        assert out.iloc[0]["Student_ID"] == "X"
