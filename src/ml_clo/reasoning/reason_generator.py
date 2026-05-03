"""Rule-based reason generation from XAI outputs.

This module generates human-readable educational reasons from SHAP analysis results.
Uses rule-based templates (no ML, no LLMs) to map XAI outputs to pedagogical reasons.
"""

from typing import Dict, List, Optional, Tuple

import pandas as pd

from ml_clo.config.xai_config import DATA_SOURCE_MAPPING, REASON_VALUE_CALIBRATION
from ml_clo.reasoning.solution_mapper import (
    get_calibrated_solutions,
    get_solutions_for_reasons,
)
from ml_clo.reasoning.templates import format_reason_with_impact, get_reason_template
from ml_clo.utils.logger import get_logger

logger = get_logger(__name__)


def generate_reasons(
    top_negative_impacts: List[Tuple[str, float, float]],
    context: str = "individual",
    include_solutions: bool = True,
    raw_feature_row: Optional[pd.Series] = None,
) -> List[Dict[str, any]]:
    """Generate educational reasons from top negative impacts.

    Args:
        top_negative_impacts: List of tuples (group_name, shap_value, impact_percentage)
        context: "individual" or "class" (default: "individual")
        include_solutions: Whether to include solutions (default: True)
        raw_feature_row: Một dòng dữ liệu gốc (trước hash) để khớp lời giải với chỉ số thực tế

    Returns:
        List of reason dictionaries with:
        - group_name: Pedagogical group name
        - reason_text: Human-readable reason text
        - impact_percentage: Impact percentage
        - shap_value: Raw SHAP value
        - solutions: List of actionable solutions (if include_solutions=True)
        - calibrated: True nếu đã hiệu chỉnh theo hồ sơ tốt (optional)
    """
    reasons = []

    for group_name, shap_value, impact_percentage in top_negative_impacts:
        reason_text, calibrated = get_reason_template(
            group_name,
            impact_percentage,
            context,
            raw_feature_row=raw_feature_row,
        )

        # Get data source for this group (e.g. "điểm danh", "nhân khẩu")
        data_source = DATA_SOURCE_MAPPING.get(group_name)

        # Format with impact percentage and data source
        formatted_reason = format_reason_with_impact(
            reason_text, impact_percentage, data_source=data_source
        )

        # DESIGN-05: Emit both `group_name` (legacy) and `reason_key` (schema
        # name) so callers can use either consistently. `reason_key` is the
        # canonical name and matches the dataclass field.
        reason_dict: Dict[str, any] = {
            "group_name": group_name,
            "reason_key": group_name,
            "reason_text": formatted_reason,
            "impact_percentage": round(impact_percentage, 2),
            "shap_value": round(float(shap_value), 4),
            "calibrated": calibrated,
        }

        # Add solutions if requested
        if include_solutions:
            if context == "individual" and calibrated:
                reason_dict["solutions"] = get_calibrated_solutions(group_name)
            else:
                solutions = get_solutions_for_reasons(
                    [(group_name, shap_value, impact_percentage)],
                    context=context,
                )
                reason_dict["solutions"] = solutions.get(group_name, [])

        reasons.append(reason_dict)

    logger.debug(
        f"Generated {len(reasons)} reasons for {context} context"
    )

    return reasons


def generate_summary_reason(
    reasons: List[Dict[str, any]],
    context: str = "individual",
    predicted_score: Optional[float] = None,
) -> str:
    """Generate a summary reason text from multiple reasons.

    Args:
        reasons: List of reason dictionaries
        context: "individual" or "class" (default: "individual")
        predicted_score: Điểm dự đoán (để tránh mâu thuẫn khi điểm cao nhưng SHAP nêu nhóm âm)

    Returns:
        Summary reason text
    """
    if not reasons:
        if context == "individual":
            return "Không xác định được nguyên nhân cụ thể ảnh hưởng đến kết quả học tập."
        else:
            return "Không xác định được nguyên nhân cụ thể ảnh hưởng đến kết quả học tập của lớp."

    # Get top reason
    top_reason = reasons[0]
    top_group = top_reason["group_name"]
    top_impact = top_reason["impact_percentage"]
    high_clo = float(REASON_VALUE_CALIBRATION["high_predicted_clo_min"])

    if (
        context == "individual"
        and predicted_score is not None
        and predicted_score >= high_clo
        and top_reason.get("calibrated")
    ):
        summary = (
            f"Điểm dự đoán {predicted_score:.2f}/6 ở mức khá tốt. "
            f"Nhóm yếu tố SHAP xếp đầu là {top_group.lower()} ({top_impact:.1f}% trong phân bổ giải thích), "
            "trong khi chỉ số thực tế trên hồ sơ cho nhóm đó vẫn tốt — đây là tác động tương đối so với baseline mô hình, "
            "không có nghĩa biểu hiện thực tế kém."
        )
    elif context == "individual":
        summary = f"Nguyên nhân chính ảnh hưởng đến kết quả học tập là {top_group.lower()} "
        summary += f"(mức độ ảnh hưởng: {top_impact:.1f}%)."
    else:
        summary = f"Nguyên nhân chính ảnh hưởng đến kết quả học tập của lớp là {top_group.lower()} "
        summary += f"(mức độ ảnh hưởng: {top_impact:.1f}%)."

    # Add secondary reasons if any
    if len(reasons) > 1:
        secondary_groups = [r["group_name"] for r in reasons[1:3]]  # Top 2-3
        if secondary_groups:
            summary += f" Các yếu tố khác bao gồm: {', '.join(secondary_groups)}."

    return summary


def _append_data_source_to_reason(reason_text: str, group_name: str) -> str:
    """Append '(dựa vào file X)' to reason_text based on group."""
    data_source = DATA_SOURCE_MAPPING.get(group_name)
    if data_source:
        return f"{reason_text} (dựa vào file {data_source})"
    return reason_text


def generate_explanation_from_distribution(
    scores: List[float],
    context: str = "class",
) -> Dict[str, any]:
    """Tạo explanation từ phân phối điểm khi không có SHAP (chỉ danh sách điểm, không MSSV).

    Args:
        scores: Danh sách điểm CLO (0-6)
        context: "class" (mặc định)

    Returns:
        Dict giống generate_complete_explanation: summary, reasons, predicted_score
    """
    import numpy as np

    scores_arr = np.array([float(s) for s in scores if s is not None])
    if len(scores_arr) == 0:
        return {
            "predicted_score": 0.0,
            "summary": "Không có dữ liệu điểm để phân tích.",
            "reasons": [],
        }

    mean_score = float(np.mean(scores_arr))
    median_score = float(np.median(scores_arr))
    std_score = float(np.std(scores_arr)) if len(scores_arr) > 1 else 0.0
    low_count = int((scores_arr < 3.0).sum())
    total = len(scores_arr)
    low_pct = (low_count / total * 100) if total > 0 else 0

    reasons = []

    # Điểm thấp
    if mean_score < 3.5:
        base_text = (
            f"Điểm trung bình lớp thấp ({mean_score:.1f}/6). "
            f"Có {low_count}/{total} sinh viên ({low_pct:.0f}%) có điểm dưới 3.0."
        )
        reasons.append({
            "group_name": "Học lực",
            "reason_text": _append_data_source_to_reason(base_text, "Học lực"),
            "impact_percentage": min(100, max(50, 100 - mean_score * 20)),
            "solutions": [
                "Ôn tập lại kiến thức nền tảng",
                "Tăng cường bài tập thực hành",
                "Tổ chức buổi học bù cho các phần khó",
            ],
        })

    # Phân tán cao
    if std_score > 1.2 and total > 5:
        base_text = (
            f"Phân tán điểm cao (độ lệch chuẩn {std_score:.1f}). "
            "Trình độ sinh viên trong lớp chênh lệch nhiều."
        )
        reasons.append({
            "group_name": "Chênh lệch trình độ",
            "reason_text": _append_data_source_to_reason(base_text, "Chênh lệch trình độ"),
            "impact_percentage": min(80, int(std_score * 30)),
            "solutions": [
                "Phân nhóm học theo trình độ",
                "Hỗ trợ riêng cho nhóm yếu",
                "Bài tập phân cấp độ khó",
            ],
        })

    # Nếu không có lý do cụ thể
    if not reasons:
        base_text = (
            f"Điểm trung bình: {mean_score:.1f}, trung vị: {median_score:.1f}. "
            + ("Phân phối điểm ổn định." if std_score < 1.0 else f"Độ phân tán: {std_score:.1f}.")
        )
        reasons.append({
            "group_name": "Tổng quan",
            "reason_text": _append_data_source_to_reason(base_text, "Tổng quan"),
            "impact_percentage": 30,
            "solutions": ["Duy trì phương pháp giảng dạy hiện tại", "Theo dõi tiến độ lớp"],
        })

    summary = (
        f"Phân tích từ {total} điểm CLO: trung bình {mean_score:.1f}/6. "
        + (f"{low_count} sinh viên ({low_pct:.0f}%) có điểm thấp." if low_count > 0 else "Lớp đạt mức trung bình trở lên.")
    )

    return {
        "predicted_score": mean_score,
        "summary": summary,
        "reasons": reasons,
    }


def generate_complete_explanation(
    top_negative_impacts: List[Tuple[str, float, float]],
    predicted_score: float,
    context: str = "individual",
    include_solutions: bool = True,
    raw_feature_row: Optional[pd.Series] = None,
) -> Dict[str, any]:
    """Generate complete explanation with reasons and solutions.

    Args:
        top_negative_impacts: List of tuples (group_name, shap_value, impact_percentage)
        predicted_score: Predicted CLO score
        context: "individual" or "class" (default: "individual")
        include_solutions: Whether to include solutions (default: True)
        raw_feature_row: Dòng feature gốc (trước encode) để khớp lý do với dữ liệu thực tế

    Returns:
        Dictionary containing:
        - predicted_score: Predicted CLO score
        - summary: Summary reason text
        - reasons: List of detailed reasons
        - context: Analysis context (individual/class)
    """
    # Generate reasons
    reasons = generate_reasons(
        top_negative_impacts,
        context=context,
        include_solutions=include_solutions,
        raw_feature_row=raw_feature_row,
    )

    # Generate summary
    summary = generate_summary_reason(
        reasons,
        context=context,
        predicted_score=predicted_score,
    )

    explanation = {
        "predicted_score": round(predicted_score, 2),
        "summary": summary,
        "reasons": reasons,
        "context": context,
    }

    logger.info(
        f"Generated complete explanation for {context} context "
        f"with {len(reasons)} reasons"
    )

    return explanation

