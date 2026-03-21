"""XAI configuration and settings.

This module defines configuration for SHAP explainability, including thresholds,
feature grouping rules, and impact calculation methods.
"""

from typing import Dict, Any

# SHAP configuration
SHAP_CONFIG: Dict[str, Any] = {
    # Threshold for filtering SHAP values (absolute value)
    # Only features with |SHAP| > threshold will be considered
    "shap_threshold": 0.01,  # Minimum absolute SHAP value to consider
    
    # Number of top reasons to return
    "top_n_reasons": 5,  # Top 5 reasons for individual/class analysis
    
    # Method for computing SHAP values
    "explainer_type": "tree",  # Use TreeExplainer for tree-based models
    
    # Whether to cache explainer for performance
    "cache_explainer": True,
    
    # Background data size for SHAP (if using KernelExplainer)
    "background_size": 100,  # Not used for TreeExplainer, but kept for consistency
}

# Feature grouping rules for pedagogical categories
# These map feature patterns to pedagogical groups
PEDAGOGICAL_GROUP_PATTERNS: Dict[str, list] = {
    "Tự học": [  # Self-study
        "study_hours",
        "tuhoc",
        "library",
        "Library",
        "total_study_hours",
    ],
    "Chuyên cần": [  # Attendance
        "attendance",
        "diemdanh",
        "chuyencan",
    ],
    "Rèn luyện": [  # Conduct
        "conduct",
        "renluyen",
        "diemrenluyen",
    ],
    "Học lực": [  # Academic
        "exam_score",
        "avg_exam_score",
        "recent_avg_score",
        "total_subjects",
        "passed_subjects",
        "pass_rate",
        "improvement",
        "summary_score",
        "summary",
        "percent",
        "Percent",
        "Test",
        "Credit_Hours",
        "Credit",
    ],
    "Giảng dạy": [  # Teaching
        "TM",
        "teaching",
        "PPGD",
    ],
    "Đánh giá": [  # Assessment
        "EM",
        "assessment",
        "PPDG",
        "PPĐG",
    ],
    "Cá nhân": [  # Personal/Demographics
        "gender",
        "religion",
        "birth_place",
        "ethnicity",
        "demographic",
        "Major",
        "major",
        "Tuition",
        "tuition",
        "User_make",
        "LastName",
        "FirstName",
        "Lecturer_Name",
    ],
}

# Mapping: pedagogical group -> data source (for XAI reason text)
# Dùng trong reason_text: "(dựa vào file X)"
DATA_SOURCE_MAPPING: Dict[str, str] = {
    "Chuyên cần": "điểm danh",
    "Tự học": "tự học",
    "Rèn luyện": "điểm rèn luyện",
    "Học lực": "điểm tổng",
    "Giảng dạy": "PPGD",
    "Đánh giá": "PPDG",
    "Cá nhân": "nhân khẩu",
    "Chênh lệch trình độ": "điểm tổng",
    "Tổng quan": "phân phối điểm",
}

# Impact percentage calculation rules
IMPACT_CONFIG: Dict[str, Any] = {
    # Method for calculating impact percentage
    # "absolute": Use absolute SHAP values
    # "relative": Use relative to total absolute SHAP
    "calculation_method": "relative",
    
    # Minimum impact percentage to report
    "min_impact_percentage": 1.0,  # 1% minimum
    
    # Round impact percentage to N decimal places
    "impact_decimal_places": 2,
}

# Reason generation configuration
REASON_CONFIG: Dict[str, Any] = {
    # Language for reasons
    "language": "vi",  # Vietnamese
    
    # Style: "educational" or "technical"
    "style": "educational",
    
    # Focus areas for reason generation
    "focus_areas": [
        "learning_behavior",  # Hành vi học tập
        "participation_level",  # Mức độ tham gia
        "study_methods",  # Cách học
    ],
}

# Solution mapping configuration
SOLUTION_CONFIG: Dict[str, Any] = {
    # Maximum number of solutions per reason
    "max_solutions_per_reason": 3,
    
    # Solution specificity level: "general" or "specific"
    "specificity": "specific",
    
    # Context awareness: True if solutions should vary by individual vs class
    "context_aware": True,
}

