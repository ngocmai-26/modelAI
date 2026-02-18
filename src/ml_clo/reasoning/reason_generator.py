"""Rule-based reason generation from XAI outputs.

This module generates human-readable educational reasons from SHAP analysis results.
Uses rule-based templates (no ML, no LLMs) to map XAI outputs to pedagogical reasons.
"""

from typing import Dict, List, Optional, Tuple

from ml_clo.config.xai_config import REASON_CONFIG
from ml_clo.reasoning.solution_mapper import get_solutions_for_reasons
from ml_clo.reasoning.templates import format_reason_with_impact, get_reason_template
from ml_clo.utils.logger import get_logger

logger = get_logger(__name__)


def generate_reasons(
    top_negative_impacts: List[Tuple[str, float, float]],
    context: str = "individual",
    include_solutions: bool = True,
) -> List[Dict[str, any]]:
    """Generate educational reasons from top negative impacts.

    Args:
        top_negative_impacts: List of tuples (group_name, shap_value, impact_percentage)
        context: "individual" or "class" (default: "individual")
        include_solutions: Whether to include solutions (default: True)

    Returns:
        List of reason dictionaries with:
        - group_name: Pedagogical group name
        - reason_text: Human-readable reason text
        - impact_percentage: Impact percentage
        - shap_value: Raw SHAP value
        - solutions: List of actionable solutions (if include_solutions=True)
    """
    reasons = []

    for group_name, shap_value, impact_percentage in top_negative_impacts:
        # Get reason template
        reason_text = get_reason_template(group_name, impact_percentage, context)

        # Format with impact percentage
        formatted_reason = format_reason_with_impact(reason_text, impact_percentage)

        reason_dict = {
            "group_name": group_name,
            "reason_text": formatted_reason,
            "impact_percentage": round(impact_percentage, 2),
            "shap_value": round(float(shap_value), 4),
        }

        # Add solutions if requested
        if include_solutions:
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
) -> str:
    """Generate a summary reason text from multiple reasons.

    Args:
        reasons: List of reason dictionaries
        context: "individual" or "class" (default: "individual")

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

    # Build summary
    if context == "individual":
        summary = f"Nguyên nhân chính ảnh hưởng đến kết quả học tập là {top_group.lower()} "
    else:
        summary = f"Nguyên nhân chính ảnh hưởng đến kết quả học tập của lớp là {top_group.lower()} "

    summary += f"(mức độ ảnh hưởng: {top_impact:.1f}%)."

    # Add secondary reasons if any
    if len(reasons) > 1:
        secondary_groups = [r["group_name"] for r in reasons[1:3]]  # Top 2-3
        if secondary_groups:
            summary += f" Các yếu tố khác bao gồm: {', '.join(secondary_groups)}."

    return summary


def generate_complete_explanation(
    top_negative_impacts: List[Tuple[str, float, float]],
    predicted_score: float,
    context: str = "individual",
    include_solutions: bool = True,
) -> Dict[str, any]:
    """Generate complete explanation with reasons and solutions.

    Args:
        top_negative_impacts: List of tuples (group_name, shap_value, impact_percentage)
        predicted_score: Predicted CLO score
        context: "individual" or "class" (default: "individual")
        include_solutions: Whether to include solutions (default: True)

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
    )

    # Generate summary
    summary = generate_summary_reason(reasons, context=context)

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

