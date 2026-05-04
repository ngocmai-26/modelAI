"""Survey response preprocessor (Phase 1 — advisor feedback ablation).

Encodes the 42-column raw survey file into ~30 numeric/binary features that
can be merged onto the training set keyed by (Student_ID, year, semester).

Encoding strategy:
- Likert / ordinal scales → ordinal int (higher = more / better)
- Binary yes-no → 0/1
- Multi-select (comma-separated) → multi-hot binary columns
- Numeric ranges ("< 5 triệu", "10 – 15 triệu") → midpoint numeric
- Anything unrecognised → NaN, later imputed in feature_encoder
"""

from __future__ import annotations

import re
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from ml_clo.utils.logger import get_logger

logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# Ordinal mappings (higher value = stronger / more positive)
# ---------------------------------------------------------------------------

LIKERT_5 = {
    "Rất thấp": 0, "Thấp": 1, "Trung bình": 2, "Cao": 3, "Rất cao": 4,
    "Không bao giờ": 0, "Hiếm khi": 1, "Thỉnh thoảng": 2,
    "Thường xuyên": 3, "Rất thường xuyên": 4, "Luôn luôn": 4,
    "Rất kém": 0, "Kém": 1, "Tốt": 3, "Rất tốt": 4,
    "Không": 0, "Ít": 1, "Nhiều": 3, "Rất nhiều": 4,
    "Rất khó khăn": 0, "Khó khăn": 1, "Dễ dàng": 3, "Rất dễ dàng": 4,
}

FAMILY_FINANCE_SUPPORT = {
    "Gần như không có": 0, "Rất hạn chế": 1,
    "Đủ một phần": 2, "Hoàn toàn đủ": 3,
}

INCOME_RANGE_TRIEU = {
    "< 5 triệu": 2.5, "5 – 10 triệu": 7.5, "10 – 15 triệu": 12.5,
    "15 – 20 triệu": 17.5, "> 20 triệu": 25.0,
    "Không xác định / Không muốn trả lời": np.nan,
}

PARTTIME_INCOME_TRIEU = {
    "< 2 triệu": 1.0, "2 – 5 triệu": 3.5,
    "5 – 10 triệu": 7.5, "> 10 triệu": 12.5,
}

PARTTIME_HOURS = {
    "Không làm thêm": 0.0,
    "< 10 giờ": 5.0, "10 – 20 giờ": 15.0,
    "20 – 30 giờ": 25.0, "> 30 giờ": 35.0,
}

PARTTIME_IMPACT = {
    "Rất tiêu cực": 0, "Tiêu cực": 1, "Không ảnh hưởng": 2, "Tích cực": 3,
}

DROPOUT_THOUGHT = {
    "Chưa bao giờ": 0, "Đã từng nghĩ": 1, "Nghĩ thường xuyên": 2,
}

SLEEP_HOURS = {
    "< 5 giờ": 4.5, "5 – 6 giờ": 5.5, "6 – 7 giờ": 6.5,
    "7 – 8 giờ": 7.5, "> 8 giờ": 8.5,
}

SELF_STUDY_HOURS_PER_DAY = {
    "< 1 giờ": 0.5, "1 – 2 giờ": 1.5, "2 – 4 giờ": 3.0, "> 4 giờ": 5.0,
}

LIVING_AREA = {
    "Vùng sâu/ vùng xa": 0, "Nông thôn": 1, "Thành thị": 2,
}

DEVICE_AVAILABILITY = {
    "Không có": 0, "Có, nhưng hạn chế": 1, "Có, đầy đủ": 2,
}

INTERNET_QUALITY = {
    "Kém": 0, "Trung bình": 1, "Tốt": 2, "Rất tốt": 3,
}

STUDY_GROUP = {
    "Không": 0, "Có, thỉnh thoảng": 1, "Có, thường xuyên": 2,
}

YESNO = {"Không": 0, "Có": 1}

PRIMARY_EARNER = {"Không": 0, "Một phần": 1, "Có": 2}

# ---------------------------------------------------------------------------
# Multi-select expansions
# ---------------------------------------------------------------------------

MENTAL_ISSUES = [
    ("stress", "căng thẳng"),
    ("anxiety", "lo âu"),
    ("loss_motivation", "mất động lực"),
    ("health", "vấn đề sức khỏe"),
    ("family_conflict", "xung đột gia đình"),
    ("loneliness", "cô đơn"),
    ("burnout", "kiệt sức"),
]

DIFFICULTY_HELP_SOURCES = [
    ("self_search", "tự tìm tài liệu"),
    ("ask_friends", "hỏi bạn bè"),
    ("ask_lecturer", "hỏi giảng viên"),
    ("nobody", "không biết hỏi ai"),
]

STUDY_METHODS = [
    ("memorize", "đọc và ghi nhớ"),
    ("practice", "làm bài tập nhiều"),
    ("group_discuss", "thảo luận nhóm"),
    ("mindmap", "sơ đồ tư duy"),
]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _norm(s: object) -> str:
    if s is None or (isinstance(s, float) and np.isnan(s)):
        return ""
    return str(s).strip().lower()


def _ordinal(value: object, mapping: Dict[str, float]) -> float:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return np.nan
    s = str(value).strip()
    return mapping.get(s, np.nan)


def _multihot(value: object, slots: List[tuple]) -> Dict[str, int]:
    text = _norm(value)
    return {f"survey_mh_{key}": int(needle in text) for key, needle in slots}


def _semester_to_int(value: object) -> Optional[int]:
    s = _norm(value)
    if not s:
        return None
    m = re.search(r"\d+", s)
    return int(m.group()) if m else None


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

RAW_TO_FEATURE = {
    "1.1. Mã sinh viên": "Student_ID",
    "1.2. Năm học": "year",
    "1.3. Học kỳ": "semester_raw",
}


def preprocess_survey(raw_df: pd.DataFrame) -> pd.DataFrame:
    """Convert raw survey DataFrame into numeric feature DataFrame.

    Returns a DataFrame with merge keys (Student_ID, year, semester) and
    survey-derived numeric/binary feature columns prefixed ``survey_``.
    """
    if raw_df is None or len(raw_df) == 0:
        return pd.DataFrame(columns=["Student_ID", "year", "semester"])

    df = raw_df.copy()

    # Merge keys
    out = pd.DataFrame()
    out["Student_ID"] = df["1.1. Mã sinh viên"].astype(str).str.strip()
    out["year"] = df["1.2. Năm học"].astype(str).str.strip()
    out["semester"] = df["1.3. Học kỳ"].apply(_semester_to_int)

    # Section 2 — Family / Finance
    out["survey_family_finance_support"] = df["2.3. Mức độ hỗ trợ tài chính từ gia đình cho việc học"].apply(
        lambda v: _ordinal(v, FAMILY_FINANCE_SUPPORT)
    )
    out["survey_finance_difficulty"] = df["2.4. Trong học kỳ này, bạn có gặp khó khăn tài chính không?"].apply(
        lambda v: _ordinal(v, LIKERT_5)
    )
    out["survey_family_emotional_support"] = df["2.5. Mức độ hỗ trợ tinh thần từ gia đình đối với việc học"].apply(
        lambda v: _ordinal(v, LIKERT_5)
    )
    out["survey_family_expectation"] = df["2.6. Mức độ kỳ vọng của gia đình về kết quả học tập"].apply(
        lambda v: _ordinal(v, LIKERT_5)
    )
    out["survey_family_income_trieu"] = df["2.7. Thu nhập gia đình hàng tháng (ước tính)"].apply(
        lambda v: _ordinal(v, INCOME_RANGE_TRIEU)
    )
    out["survey_is_primary_earner"] = df["2.8. Bạn có phải là lao động chính trong gia đình không?"].apply(
        lambda v: _ordinal(v, PRIMARY_EARNER)
    )
    out["survey_tuition_pressure"] = df["2.11. Việc đóng học phí trong học kỳ này gây áp lực cho bạn ở mức nào?"].apply(
        lambda v: _ordinal(v, LIKERT_5)
    )
    out["survey_living_area"] = df["2.1. Nơi cư trú của gia đình"].apply(
        lambda v: _ordinal(v, LIVING_AREA)
    )

    # Section 3 — Part-time
    out["survey_has_parttime"] = df["3.1. Trong học kỳ này, bạn có đi làm thêm không?"].apply(
        lambda v: _ordinal(v, YESNO)
    )
    out["survey_parttime_hours_per_week"] = df["3.2. Số giờ làm thêm trung bình mỗi tuần "].apply(
        lambda v: _ordinal(v, PARTTIME_HOURS)
    )
    out["survey_parttime_impact"] = df["3.4. Việc làm thêm ảnh hưởng đến việc học của bạn như thế nào?"].apply(
        lambda v: _ordinal(v, PARTTIME_IMPACT)
    )
    out["survey_parttime_income_trieu"] = df["3.6. Thu nhập làm thêm mỗi tháng (ước tính)"].apply(
        lambda v: _ordinal(v, PARTTIME_INCOME_TRIEU)
    )

    # Section 4 — Mental / Health
    out["survey_study_pressure"] = df["4.1. Mức độ áp lực học tập"].apply(
        lambda v: _ordinal(v, LIKERT_5)
    )
    out["survey_dropout_thought"] = df["4.3. Bạn có từng nghĩ đến việc bỏ học / bảo lưu trong học kỳ này không?"].apply(
        lambda v: _ordinal(v, DROPOUT_THOUGHT)
    )
    out["survey_sleep_hours"] = df["4.4. Số giờ ngủ trung bình mỗi đêm"].apply(
        lambda v: _ordinal(v, SLEEP_HOURS)
    )
    # Multi-hot mental issues
    mental_mh = df["4.2. Bạn gặp những vấn đề nào sau đây?"].apply(
        lambda v: _multihot(v, MENTAL_ISSUES)
    )
    out = pd.concat([out, pd.json_normalize(mental_mh)], axis=1)

    # Section 5 — Study habits
    out["survey_quiet_space"] = df["5.1. Bạn có không gian học tập yên tĩnh, ổn định không?"].apply(
        lambda v: _ordinal(v, YESNO)
    )
    out["survey_self_study_hours_per_day"] = df["5.2. Thời gian tự học trung bình mỗi ngày"].apply(
        lambda v: _ordinal(v, SELF_STUDY_HOURS_PER_DAY)
    )
    out["survey_time_management"] = df["5.5. Kỹ năng quản lý thời gian của bạn"].apply(
        lambda v: _ordinal(v, LIKERT_5)
    )
    out["survey_has_study_plan"] = df["5.6. Bạn có lập kế hoạch học tập cho từng tuần/tháng không?"].apply(
        lambda v: _ordinal(v, LIKERT_5)
    )
    out["survey_peer_discussion"] = df["5.7. Bạn có thường xuyên trao đổi với bạn bè về bài học không?"].apply(
        lambda v: _ordinal(v, LIKERT_5)
    )
    out["survey_study_group"] = df["5.8. Bạn có tham gia nhóm học tập thường xuyên không?"].apply(
        lambda v: _ordinal(v, STUDY_GROUP)
    )
    out["survey_peer_pressure"] = df["5.9. Bạn có cảm thấy áp lực từ bạn bè về việc học không?"].apply(
        lambda v: _ordinal(v, LIKERT_5)
    )
    # Multi-hot help sources
    help_mh = df["5.3. Khi gặp khó khăn học tập, bạn thường:"].apply(
        lambda v: _multihot(v, DIFFICULTY_HELP_SOURCES)
    )
    out = pd.concat([out, pd.json_normalize(help_mh)], axis=1)
    method_mh = df["5.4. Phương pháp học bạn thường sử dụng "].apply(
        lambda v: _multihot(v, STUDY_METHODS)
    )
    out = pd.concat([out, pd.json_normalize(method_mh)], axis=1)

    # Section 6 — Tech / Resources
    out["survey_device"] = df["6.1. Bạn có thiết bị học tập ổn định (laptop/máy tính) không?"].apply(
        lambda v: _ordinal(v, DEVICE_AVAILABILITY)
    )
    out["survey_internet"] = df["6.2. Chất lượng kết nối internet của bạn"].apply(
        lambda v: _ordinal(v, INTERNET_QUALITY)
    )
    out["survey_lms_usage"] = df["6.3. Bạn có thường xuyên sử dụng LMS (hệ thống quản lý học tập) không?"].apply(
        lambda v: _ordinal(v, LIKERT_5)
    )
    out["survey_online_tools"] = df["6.4. Bạn có thường xuyên sử dụng các công cụ học tập online khác không? (Zoom, Teams, Google Meet, v.v.)"].apply(
        lambda v: _ordinal(v, LIKERT_5)
    )
    out["survey_resource_access"] = df["6.5. Bạn có dễ dàng truy cập tài liệu học tập online (thư viện số, tài liệu điện tử) không?"].apply(
        lambda v: _ordinal(v, LIKERT_5)
    )

    # Drop rows missing merge keys
    out = out.dropna(subset=["Student_ID", "year", "semester"]).copy()
    out["semester"] = out["semester"].astype(int)
    # Dedup: keep latest response per (student, year, semester)
    out = out.drop_duplicates(subset=["Student_ID", "year", "semester"], keep="last")

    n_features = sum(1 for c in out.columns if c.startswith("survey_"))
    logger.info(
        f"Preprocessed survey: {len(out)} rows, {n_features} features "
        f"({n_features} survey columns + 3 keys)"
    )

    return out
