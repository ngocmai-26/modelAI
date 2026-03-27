"""Solution mapping for educational interventions.

This module maps reason keys to actionable solutions for improving CLO scores.
Solutions are context-aware (individual vs class) and appropriate for university education.
"""

from typing import Dict, List

# Solution mappings for individual analysis
INDIVIDUAL_SOLUTIONS: Dict[str, List[str]] = {
    "Tự học": [
        "Tăng thời gian tự học tại thư viện tối thiểu 10-15 giờ mỗi tuần để củng cố kiến thức.",
        "Thiết lập kế hoạch học tập cá nhân với mục tiêu cụ thể và theo dõi tiến độ hàng tuần.",
        "Tham gia nhóm học tập có hướng dẫn để trao đổi và giải đáp thắc mắc.",
        "Sử dụng các tài liệu tham khảo và bài tập bổ sung để nâng cao hiểu biết.",
    ],
    "Chuyên cần": [
        "Đảm bảo tham gia đầy đủ các buổi học trên lớp, đặc biệt là các buổi quan trọng.",
        "Thông báo trước với giảng viên nếu có lý do chính đáng không thể tham gia.",
        "Chủ động bù đắp kiến thức đã bỏ lỡ bằng cách học nhóm hoặc xem lại tài liệu.",
        "Thiết lập thói quen đi học đúng giờ và chuẩn bị bài trước khi đến lớp.",
    ],
    "Rèn luyện": [
        "Tích cực tham gia các hoạt động ngoại khóa, câu lạc bộ học thuật và tình nguyện.",
        "Tuân thủ nghiêm túc nội quy học tập và quy định của nhà trường.",
        "Xây dựng thái độ học tập tích cực, tôn trọng giảng viên và bạn học.",
        "Tham gia các hoạt động phát triển kỹ năng mềm và nâng cao năng lực cá nhân.",
    ],
    "Học lực": [
        "Ôn tập lại kiến thức cơ bản của các môn học trước đó để củng cố nền tảng.",
        "Tham gia các lớp bổ trợ hoặc học kèm để cải thiện điểm số các môn yếu.",
        "Lập kế hoạch học tập dài hạn để cải thiện dần học lực theo từng học kỳ.",
        "Tìm kiếm sự hỗ trợ từ giảng viên, trợ giảng hoặc các dịch vụ tư vấn học tập.",
    ],
    "Giảng dạy": [
        "Chủ động trao đổi với giảng viên về phương pháp học phù hợp với bản thân.",
        "Yêu cầu giảng viên cung cấp thêm tài liệu hoặc ví dụ minh họa nếu cần.",
        "Tham gia tích cực các hoạt động tương tác trong lớp để tăng hiệu quả học tập.",
        "Tìm kiếm các nguồn học tập bổ sung (video, tài liệu online) phù hợp với phong cách học.",
    ],
    "Đánh giá": [
        "Nắm rõ tiêu chí đánh giá và yêu cầu của từng bài tập, bài kiểm tra.",
        "Chuẩn bị kỹ lưỡng cho các kỳ thi và bài đánh giá theo đúng format yêu cầu.",
        "Yêu cầu phản hồi từ giảng viên sau mỗi bài đánh giá để cải thiện.",
        "Thực hành các dạng bài tập tương tự để làm quen với phương pháp đánh giá.",
    ],
    "Cá nhân": [
        "Tìm kiếm sự hỗ trợ từ phòng tư vấn học tập hoặc dịch vụ hỗ trợ sinh viên.",
        "Cân bằng giữa học tập và các hoạt động cá nhân để duy trì sức khỏe tinh thần.",
        "Thiết lập môi trường học tập thuận lợi tại nhà hoặc thư viện.",
        "Xây dựng mạng lưới hỗ trợ từ gia đình, bạn bè và cộng đồng học tập.",
    ],
}

# Solution mappings for class-level analysis
CLASS_SOLUTIONS: Dict[str, List[str]] = {
    "Tự học": [
        "Tổ chức các buổi hướng dẫn phương pháp tự học hiệu quả cho toàn lớp.",
        "Khuyến khích sinh viên thành lập nhóm học tập và tổ chức các buổi học nhóm định kỳ.",
        "Tạo động lực học tập thông qua các hoạt động thi đua, trao đổi kinh nghiệm học tập.",
        "Cung cấp tài liệu và nguồn học tập bổ sung để hỗ trợ sinh viên tự học.",
    ],
    "Chuyên cần": [
        "Thực hiện điểm danh nghiêm túc và có chế độ khuyến khích cho sinh viên đi học đầy đủ.",
        "Tổ chức các hoạt động học tập hấp dẫn để tăng động lực tham gia của sinh viên.",
        "Thông báo rõ ràng về tầm quan trọng của việc tham gia đầy đủ các buổi học.",
        "Có biện pháp hỗ trợ cho sinh viên có hoàn cảnh khó khăn ảnh hưởng đến chuyên cần.",
    ],
    "Rèn luyện": [
        "Tổ chức các hoạt động ngoại khóa, câu lạc bộ học thuật để tăng cường rèn luyện.",
        "Khuyến khích và ghi nhận các đóng góp tích cực của sinh viên trong các hoạt động.",
        "Xây dựng văn hóa lớp học tích cực, tôn trọng và hỗ trợ lẫn nhau.",
        "Có chế độ khen thưởng và động viên cho sinh viên có điểm rèn luyện tốt.",
    ],
    "Học lực": [
        "Tổ chức các lớp bổ trợ kiến thức cơ bản cho sinh viên có học lực yếu.",
        "Phân nhóm học tập theo trình độ để hỗ trợ lẫn nhau hiệu quả hơn.",
        "Cung cấp tài liệu ôn tập và bài tập bổ trợ để củng cố nền tảng kiến thức.",
        "Tăng cường tương tác giữa giảng viên và sinh viên để phát hiện và hỗ trợ kịp thời.",
    ],
    "Giảng dạy": [
        "Đa dạng hóa phương pháp giảng dạy, kết hợp lý thuyết và thực hành, tăng tính tương tác.",
        "Điều chỉnh tốc độ và cách trình bày để phù hợp hơn với đặc điểm của lớp học.",
        "Sử dụng công nghệ và phương tiện hỗ trợ để nâng cao hiệu quả giảng dạy.",
        "Thu thập phản hồi từ sinh viên để cải thiện phương pháp giảng dạy liên tục.",
    ],
    "Đánh giá": [
        "Đa dạng hóa các hình thức đánh giá để phù hợp với nhiều phong cách học khác nhau.",
        "Công bố rõ ràng tiêu chí đánh giá và hướng dẫn cách đạt điểm cao.",
        "Cung cấp phản hồi chi tiết và kịp thời sau mỗi bài đánh giá để sinh viên cải thiện.",
        "Tổ chức các buổi ôn tập và hướng dẫn trước các kỳ thi quan trọng.",
    ],
    "Cá nhân": [
        "Tổ chức các buổi tư vấn học tập và hỗ trợ tâm lý cho sinh viên gặp khó khăn.",
        "Xây dựng mạng lưới hỗ trợ trong lớp học, khuyến khích sinh viên giúp đỡ lẫn nhau.",
        "Phối hợp với các dịch vụ hỗ trợ sinh viên để giải quyết các vấn đề cá nhân.",
        "Tạo môi trường học tập thân thiện, không phân biệt đối xử và hỗ trợ tất cả sinh viên.",
    ],
}

# Maximum number of solutions per reason
MAX_SOLUTIONS_PER_REASON = 3

# Gợi ý khi chỉ số thực tế tốt nhưng SHAP vẫn nêu nhóm đó (tác động tương đối)
INDIVIDUAL_SOLUTIONS_CALIBRATED: Dict[str, List[str]] = {
    "Rèn luyện": [
        "Tiếp tục duy trì điểm rèn luyện và tham gia hoạt động tích cực như hiện tại.",
        "Theo dõi định kỳ file điểm rèn luyện để đảm bảo dữ liệu nhập cho mô hình luôn cập nhật.",
    ],
    "Chuyên cần": [
        "Duy trì thói quen đi học đều như số liệu điểm danh hiện có.",
        "Nếu có buổi vắng, chủ động bù kiến thức để tránh ảnh hưởng kỳ sau.",
    ],
    "Học lực": [
        "Tiếp tục phát huy điểm mạnh ở các môn đã đạt kết quả tốt.",
        "Ưu tiên củng cố thêm các môn có điểm thấp hơn so với trung bình cá nhân.",
    ],
}


def get_solutions(
    group_name: str,
    context: str = "individual",
    max_solutions: int = None,
) -> List[str]:
    """Get solutions for a given reason group.

    Args:
        group_name: Pedagogical group name (e.g., "Tự học", "Chuyên cần")
        context: "individual" or "class" (default: "individual")
        max_solutions: Maximum number of solutions to return (default: from config)

    Returns:
        List of solution texts
    """
    if max_solutions is None:
        max_solutions = MAX_SOLUTIONS_PER_REASON

    solutions = (
        CLASS_SOLUTIONS if context == "class" else INDIVIDUAL_SOLUTIONS
    )

    if group_name not in solutions:
        return [
            f"Cần có biện pháp cụ thể để cải thiện yếu tố {group_name}."
        ]

    return solutions[group_name][:max_solutions]


def get_calibrated_solutions(
    group_name: str,
    max_solutions: int = None,
) -> List[str]:
    """Gợi ý nhẹ khi lý do đã hiệu chỉnh (hồ sơ tốt, SHAP tương đối)."""
    if max_solutions is None:
        max_solutions = MAX_SOLUTIONS_PER_REASON
    sol = INDIVIDUAL_SOLUTIONS_CALIBRATED.get(group_name)
    if not sol:
        return get_solutions(group_name, "individual", max_solutions)
    return sol[:max_solutions]


def get_solutions_for_reasons(
    reason_groups: List[tuple],
    context: str = "individual",
    max_solutions_per_reason: int = None,
) -> Dict[str, List[str]]:
    """Get solutions for multiple reason groups.

    Args:
        reason_groups: List of tuples (group_name, shap_value, impact_percentage)
        context: "individual" or "class" (default: "individual")
        max_solutions_per_reason: Maximum solutions per reason (default: from config)

    Returns:
        Dictionary mapping group names to solution lists
    """
    result = {}

    for group_name, _, _ in reason_groups:
        solutions = get_solutions(group_name, context, max_solutions_per_reason)
        result[group_name] = solutions

    return result

