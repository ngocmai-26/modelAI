"""Reason templates for generating human-readable explanations.

This module contains rule-based templates for generating educational reasons
in Vietnamese language, focusing on learning behavior, participation, and study methods.
"""

from typing import Any, Dict, List, Optional, Tuple

import pandas as pd

from ml_clo.config.xai_config import REASON_VALUE_CALIBRATION

# Reason templates for individual analysis
INDIVIDUAL_REASON_TEMPLATES: Dict[str, Dict[str, str]] = {
    "Tự học": {
        "low": "Sinh viên có số giờ tự học thấp, chưa đủ thời gian để củng cố kiến thức và rèn luyện kỹ năng.",
        "medium": "Thời gian tự học của sinh viên chưa đạt mức tối ưu, ảnh hưởng đến khả năng nắm vững nội dung học tập.",
        "high": "Sinh viên thiếu đầu tư thời gian tự học, dẫn đến kết quả học tập chưa đạt yêu cầu.",
    },
    "Chuyên cần": {
        "low": "Tỷ lệ chuyên cần của sinh viên thấp, việc tham gia các buổi học không đều đặn ảnh hưởng đến quá trình tiếp thu kiến thức.",
        "medium": "Sinh viên chưa duy trì được mức độ chuyên cần ổn định, làm gián đoạn quá trình học tập liên tục.",
        "high": "Sinh viên có tỷ lệ vắng mặt cao, thiếu sự tham gia tích cực trong các hoạt động học tập trên lớp.",
        "calibrated_good": (
            "Tỷ lệ điểm danh trên dữ liệu nhập vào ở mức tốt; nhóm chuyên cần vẫn xuất hiện trong giải thích "
            "vì mô hình gán tác động tương đối làm giảm điểm so với baseline, không đồng nghĩa vắng nhiều buổi học"
        ),
    },
    "Rèn luyện": {
        "low": "Điểm rèn luyện của sinh viên ở mức thấp, phản ánh sự thiếu tích cực trong các hoạt động ngoại khóa và tuân thủ nội quy.",
        "medium": "Mức độ rèn luyện của sinh viên chưa đạt yêu cầu, cần cải thiện thái độ và hành vi học tập.",
        "high": "Sinh viên có điểm rèn luyện kém, ảnh hưởng tiêu cực đến môi trường học tập và kết quả học tập.",
        # Chỉ dùng khi hồ sơ rèn luyện tốt nhưng SHAP nhóm này vẫn âm (kéo điểm so với baseline mô hình)
        "calibrated_good": (
            "Điểm rèn luyện trên hồ sơ ở mức tốt; nhóm yếu tố này vẫn được xếp cao trong giải thích mô hình "
            "vì nó hơi kéo điểm dự đoán xuống so với mức cơ sở (baseline) của mô hình, không có nghĩa rèn luyện kém"
        ),
    },
    "Học lực": {
        "low": "Học lực hiện tại của sinh viên còn yếu, điểm số các môn học trước đó thấp, chưa đủ nền tảng để tiếp thu kiến thức mới.",
        "medium": "Nền tảng học tập của sinh viên chưa vững chắc, cần củng cố lại kiến thức cơ bản để cải thiện kết quả.",
        "high": "Sinh viên có học lực kém, thiếu nền tảng kiến thức cần thiết để đạt được kết quả tốt trong môn học này.",
        "calibrated_good": (
            "Điểm trung bình các môn trước trên hồ sơ ở mức khá; nhóm học lực vẫn được mô hình nhấn mạnh "
            "do tác động SHAP tương đối so với baseline, không mô tả toàn bộ quá trình học là yếu"
        ),
    },
    "Giảng dạy": {
        "low": "Phương pháp giảng dạy hiện tại có thể chưa phù hợp với phong cách học tập của sinh viên, ảnh hưởng đến khả năng tiếp thu.",
        "medium": "Sự tương tác giữa phương pháp giảng dạy và cách học của sinh viên chưa tối ưu, cần điều chỉnh để nâng cao hiệu quả.",
        "high": "Phương pháp giảng dạy không phù hợp với đặc điểm học tập của sinh viên, làm giảm hiệu quả tiếp thu kiến thức.",
    },
    "Đánh giá": {
        "low": "Phương pháp đánh giá hiện tại có thể chưa phản ánh đúng năng lực của sinh viên, ảnh hưởng đến động lực học tập.",
        "medium": "Cách thức đánh giá chưa khuyến khích sinh viên phát huy tối đa khả năng, cần điều chỉnh để phù hợp hơn.",
        "high": "Phương pháp đánh giá không phù hợp với đặc điểm và năng lực của sinh viên, làm giảm hiệu quả học tập.",
    },
    "Cá nhân": {
        "low": "Các yếu tố cá nhân như hoàn cảnh gia đình, điều kiện sống có thể ảnh hưởng đến khả năng tập trung và đầu tư thời gian cho việc học.",
        "medium": "Đặc điểm cá nhân của sinh viên có thể tác động đến quá trình học tập, cần được quan tâm và hỗ trợ phù hợp.",
        "high": "Các yếu tố cá nhân đang ảnh hưởng đáng kể đến kết quả học tập của sinh viên, cần có biện pháp hỗ trợ cụ thể.",
    },
}

# Reason templates for class-level analysis
CLASS_REASON_TEMPLATES: Dict[str, Dict[str, str]] = {
    "Tự học": {
        "low": "Nhiều sinh viên trong lớp có số giờ tự học thấp, cho thấy lớp học chưa có thói quen tự học tích cực.",
        "medium": "Mức độ tự học của lớp chưa đạt yêu cầu, cần khuyến khích và tạo động lực cho sinh viên đầu tư thời gian tự học.",
        "high": "Lớp học có vấn đề nghiêm trọng về thời gian tự học, đây là nguyên nhân chính ảnh hưởng đến kết quả học tập chung.",
    },
    "Chuyên cần": {
        "low": "Tỷ lệ chuyên cần của lớp thấp, nhiều sinh viên vắng mặt thường xuyên, ảnh hưởng đến không khí học tập.",
        "medium": "Lớp học chưa duy trì được mức độ chuyên cần ổn định, cần có biện pháp khuyến khích tham gia đầy đủ các buổi học.",
        "high": "Vấn đề chuyên cần là điểm yếu lớn của lớp, cần có giải pháp cụ thể để cải thiện tỷ lệ tham gia học tập.",
    },
    "Rèn luyện": {
        "low": "Điểm rèn luyện trung bình của lớp ở mức thấp, phản ánh sự thiếu tích cực trong các hoạt động ngoại khóa.",
        "medium": "Mức độ rèn luyện của lớp chưa đạt yêu cầu, cần tăng cường các hoạt động và khuyến khích sự tham gia.",
        "high": "Lớp học có vấn đề về rèn luyện, ảnh hưởng tiêu cực đến môi trường học tập và kết quả chung.",
    },
    "Học lực": {
        "low": "Học lực trung bình của lớp còn yếu, nhiều sinh viên thiếu nền tảng kiến thức cơ bản.",
        "medium": "Nền tảng học tập của lớp chưa vững chắc, cần có kế hoạch bổ trợ kiến thức cho sinh viên.",
        "high": "Lớp học có học lực kém, đây là thách thức lớn cần được giải quyết để cải thiện kết quả học tập.",
    },
    "Giảng dạy": {
        "low": "Phương pháp giảng dạy hiện tại có thể chưa phù hợp với đặc điểm của lớp học, cần điều chỉnh để nâng cao hiệu quả.",
        "medium": "Cần đa dạng hóa và cải thiện phương pháp giảng dạy để phù hợp hơn với nhu cầu học tập của lớp.",
        "high": "Phương pháp giảng dạy không phù hợp là nguyên nhân chính ảnh hưởng đến kết quả học tập của lớp.",
    },
    "Đánh giá": {
        "low": "Phương pháp đánh giá hiện tại có thể chưa phản ánh đúng năng lực của lớp học, cần xem xét điều chỉnh.",
        "medium": "Cách thức đánh giá cần được cải thiện để khuyến khích và phản ánh đúng năng lực học tập của lớp.",
        "high": "Phương pháp đánh giá không phù hợp là vấn đề lớn ảnh hưởng đến động lực và kết quả học tập của lớp.",
    },
    "Cá nhân": {
        "low": "Các yếu tố cá nhân của sinh viên trong lớp có thể ảnh hưởng đến kết quả học tập chung, cần được quan tâm.",
        "medium": "Đặc điểm cá nhân của sinh viên trong lớp cần được xem xét để có biện pháp hỗ trợ phù hợp.",
        "high": "Các yếu tố cá nhân đang ảnh hưởng đáng kể đến kết quả học tập của lớp, cần có chính sách hỗ trợ cụ thể.",
    },
}

# Impact level thresholds (percentage)
# DESIGN-07: 5 finer-grained bands so impact 10.1% and 24.9% no longer
# render with the same text. Templates still ship 3 base levels (low/
# medium/high); intermediate bands map to base + an intensity adverb.
IMPACT_LEVELS = {
    "low": (1.0, 10.0),
    "medium": (10.0, 25.0),
    "high": (25.0, 100.0),
}

# Band → (base_template_key, intensity_adverb)
# Adverb is prepended to the chosen template text in Vietnamese to nudge
# the perceived severity within the band without rewriting all templates.
IMPACT_BANDS = [
    (1.0, 7.0,   "low",    ""),                  # nhẹ
    (7.0, 13.0,  "low",    "Có dấu hiệu: "),     # đáng chú ý
    (13.0, 18.0, "medium", ""),                  # trung bình thấp
    (18.0, 25.0, "medium", "Đáng kể: "),         # trung bình cao
    (25.0, 35.0, "high",   ""),                  # cao
    (35.0, 100.0, "high",  "Rất nghiêm trọng: "),# rất cao
]


def _row_float(row: Optional[pd.Series], col: str) -> Optional[float]:
    if row is None or col not in row.index:
        return None
    v = row[col]
    if pd.isna(v):
        return None
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


def _use_calibrated_template(
    group_name: str,
    raw_feature_row: Optional[pd.Series],
) -> bool:
    """True nên dùng văn calibrated_good: chỉ số thực tế tốt, tránh nói 'kém' oan."""
    if raw_feature_row is None:
        return False
    cfg: Dict[str, Any] = REASON_VALUE_CALIBRATION

    if group_name == "Rèn luyện":
        best = None
        for col in ("avg_conduct_score", "latest_conduct_score"):
            x = _row_float(raw_feature_row, col)
            if x is not None:
                best = x if best is None else max(best, x)
        return best is not None and best >= float(cfg["conduct_score_good_min"])

    if group_name == "Chuyên cần":
        ar = _row_float(raw_feature_row, "attendance_rate")
        return ar is not None and ar >= float(cfg["attendance_rate_good_min"])

    if group_name == "Học lực":
        core = _row_float(raw_feature_row, "academic_core_score")
        if core is not None and core >= float(cfg["academic_avg_good_min"]):
            return True
        avg = _row_float(raw_feature_row, "avg_exam_score")
        recent = _row_float(raw_feature_row, "recent_avg_score")
        if avg is None:
            return False
        ok_avg = avg >= float(cfg["academic_avg_good_min"])
        ok_recent = recent is None or recent >= float(cfg["academic_recent_ok_min"])
        return ok_avg and ok_recent

    return False


def get_reason_template(
    group_name: str,
    impact_percentage: float,
    context: str = "individual",
    raw_feature_row: Optional[pd.Series] = None,
) -> Tuple[str, bool]:
    """Get reason template based on group name and impact level.

    Args:
        group_name: Pedagogical group name (e.g., "Tự học", "Chuyên cần")
        impact_percentage: Impact percentage (0-100)
        context: "individual" or "class" (default: "individual")
        raw_feature_row: Một dòng DataFrame trước khi encode (để hiệu chỉnh văn bản)

    Returns:
        (reason_text_template, calibrated): calibrated True nếu dùng calibrated_good
    """
    templates = (
        CLASS_REASON_TEMPLATES if context == "class" else INDIVIDUAL_REASON_TEMPLATES
    )

    if group_name not in templates:
        return f"Yếu tố {group_name} đang ảnh hưởng đến kết quả học tập.", False

    if context == "individual" and _use_calibrated_template(group_name, raw_feature_row):
        cal = templates[group_name].get("calibrated_good")
        if cal:
            return cal, True

    # DESIGN-07: Pick band → base template key + intensity adverb so two
    # impacts inside the same coarse level (e.g. 10.1% vs 24.9%) no longer
    # render with identical text.
    base_key = "medium"
    adverb = ""
    for lo, hi, key, adv in IMPACT_BANDS:
        if lo <= impact_percentage < hi:
            base_key, adverb = key, adv
            break
    else:
        # Above the top band (>= 100): treat as the highest band.
        if impact_percentage >= IMPACT_BANDS[-1][1]:
            base_key, adverb = IMPACT_BANDS[-1][2], IMPACT_BANDS[-1][3]

    text = templates[group_name].get(base_key, templates[group_name]["medium"])
    return (adverb + text if adverb else text), False


def format_reason_with_impact(
    reason_text: str,
    impact_percentage: float,
    data_source: Optional[str] = None,
) -> str:
    """Format reason text with impact percentage and optional data source.

    Args:
        reason_text: Base reason text
        impact_percentage: Impact percentage (0-100)
        data_source: Tên nguồn dữ liệu (e.g. "điểm danh", "nhân khẩu") — thêm "(dựa vào file X)"

    Returns:
        Formatted reason text with impact information
    """
    if data_source:
        reason_text = f"{reason_text} (dựa vào file {data_source})"
    impact_str = f"{impact_percentage:.1f}%"
    return f"{reason_text}. (Mức độ ảnh hưởng: {impact_str})"

