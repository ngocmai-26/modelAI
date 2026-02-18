"""Reason templates for generating human-readable explanations.

This module contains rule-based templates for generating educational reasons
in Vietnamese language, focusing on learning behavior, participation, and study methods.
"""

from typing import Dict, List

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
    },
    "Rèn luyện": {
        "low": "Điểm rèn luyện của sinh viên ở mức thấp, phản ánh sự thiếu tích cực trong các hoạt động ngoại khóa và tuân thủ nội quy.",
        "medium": "Mức độ rèn luyện của sinh viên chưa đạt yêu cầu, cần cải thiện thái độ và hành vi học tập.",
        "high": "Sinh viên có điểm rèn luyện kém, ảnh hưởng tiêu cực đến môi trường học tập và kết quả học tập.",
    },
    "Học lực": {
        "low": "Học lực hiện tại của sinh viên còn yếu, điểm số các môn học trước đó thấp, chưa đủ nền tảng để tiếp thu kiến thức mới.",
        "medium": "Nền tảng học tập của sinh viên chưa vững chắc, cần củng cố lại kiến thức cơ bản để cải thiện kết quả.",
        "high": "Sinh viên có học lực kém, thiếu nền tảng kiến thức cần thiết để đạt được kết quả tốt trong môn học này.",
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
IMPACT_LEVELS = {
    "low": (1.0, 10.0),      # 1-10%
    "medium": (10.0, 25.0),  # 10-25%
    "high": (25.0, 100.0),   # >25%
}


def get_reason_template(
    group_name: str,
    impact_percentage: float,
    context: str = "individual",
) -> str:
    """Get reason template based on group name and impact level.

    Args:
        group_name: Pedagogical group name (e.g., "Tự học", "Chuyên cần")
        impact_percentage: Impact percentage (0-100)
        context: "individual" or "class" (default: "individual")

    Returns:
        Reason text template
    """
    templates = (
        CLASS_REASON_TEMPLATES if context == "class" else INDIVIDUAL_REASON_TEMPLATES
    )

    if group_name not in templates:
        return f"Yếu tố {group_name} đang ảnh hưởng đến kết quả học tập."

    # Determine impact level
    if impact_percentage >= IMPACT_LEVELS["high"][0]:
        level = "high"
    elif impact_percentage >= IMPACT_LEVELS["medium"][0]:
        level = "medium"
    else:
        level = "low"

    return templates[group_name].get(level, templates[group_name]["medium"])


def format_reason_with_impact(
    reason_text: str,
    impact_percentage: float,
) -> str:
    """Format reason text with impact percentage.

    Args:
        reason_text: Base reason text
        impact_percentage: Impact percentage (0-100)

    Returns:
        Formatted reason text with impact information
    """
    impact_str = f"{impact_percentage:.1f}%"
    return f"{reason_text} (Mức độ ảnh hưởng: {impact_str})"

