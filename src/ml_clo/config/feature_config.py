"""Feature configuration and definitions.

This module defines feature names, groups, and mappings for the CLO prediction model.
"""

from typing import Dict, List

# Feature group definitions
FEATURE_GROUPS = {
    "A. Basic Information": [
        "student_id_encoded",
        "lecturer_id_encoded",
        "subject_id_encoded",
    ],
    "B. Demographics": [
        "gender_encoded",
        "religion_encoded",
        "birth_place_region",
        "ethnicity_encoded",
    ],
    "C. Conduct Scores": [
        "avg_conduct_score",
        "latest_conduct_score",
        "conduct_trend",
    ],
    "D. Self-Study": [
        "study_hours_this_year",
        "total_study_hours",
    ],
    "E. Academic History": [
        "total_subjects",
        "passed_subjects",
        "pass_rate",
        "avg_exam_score",
        "median_exam_score",
        "min_exam_score",
        "recent_avg_score",
        "recent_median_score",
        "academic_core_score",
        "min_exam_score_adj",
        "improvement_trend",
    ],
    "F. Teaching & Assessment Methods": [
        # Teaching methods (TM*) - will be added dynamically
        # Assessment methods (EM*) - will be added dynamically
    ],
}

# Pedagogical feature groups for XAI (Vietnamese names)
PEDAGOGICAL_GROUPS = {
    "Tự học": [  # Self-study
        "study_hours_this_year",
        "total_study_hours",
    ],
    "Chuyên cần": [  # Attendance
        "attendance_rate",
    ],
    "Rèn luyện": [  # Conduct
        "avg_conduct_score",
        "latest_conduct_score",
        "conduct_trend",
    ],
    "Học lực": [  # Academic
        "avg_exam_score",
        "median_exam_score",
        "min_exam_score",
        "recent_avg_score",
        "recent_median_score",
        "academic_core_score",
        "min_exam_score_adj",
        "total_subjects",
        "passed_subjects",
        "pass_rate",
        "improvement_trend",
    ],
    "Giảng dạy": [  # Teaching
        # Teaching method features (TM*) - added dynamically
    ],
    "Đánh giá": [  # Assessment
        # Assessment method features (EM*) - added dynamically
    ],
    "Cá nhân": [  # Personal/Demographics
        "gender_encoded",
        "religion_encoded",
        "birth_place_region",
        "ethnicity_encoded",
    ],
}


def get_all_feature_names() -> List[str]:
    """Get list of all feature names from all groups.

    Returns:
        List of all feature names
    """
    all_features = []
    for group_features in FEATURE_GROUPS.values():
        all_features.extend(group_features)
    return all_features


def get_feature_group_mapping() -> Dict[str, str]:
    """Get mapping from feature name to feature group.

    Returns:
        Dictionary mapping feature name to group name
    """
    mapping = {}
    for group_name, features in FEATURE_GROUPS.items():
        for feature in features:
            mapping[feature] = group_name
    return mapping


def get_pedagogical_group_mapping() -> Dict[str, str]:
    """Get mapping from feature name to pedagogical group (for XAI).

    Returns:
        Dictionary mapping feature name to pedagogical group name
    """
    mapping = {}
    for group_name, features in PEDAGOGICAL_GROUPS.items():
        for feature in features:
            mapping[feature] = group_name
    return mapping


def add_teaching_method_features(tm_columns: List[str]) -> None:
    """Add teaching method columns to feature groups.

    Args:
        tm_columns: List of teaching method column names (e.g., ["TM 1", "TM 2", ...])
    """
    FEATURE_GROUPS["F. Teaching & Assessment Methods"].extend(tm_columns)
    PEDAGOGICAL_GROUPS["Giảng dạy"].extend(tm_columns)


def add_assessment_method_features(em_columns: List[str]) -> None:
    """Add assessment method columns to feature groups.

    Args:
        em_columns: List of assessment method column names (e.g., ["EM 1", "EM 2", ...])
    """
    FEATURE_GROUPS["F. Teaching & Assessment Methods"].extend(em_columns)
    PEDAGOGICAL_GROUPS["Đánh giá"].extend(em_columns)


