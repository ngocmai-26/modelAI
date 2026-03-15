"""Output schemas for CLO prediction and analysis.

This module defines structured output formats for individual and class analysis.
All schemas are JSON-serializable and backend-friendly.
"""

from dataclasses import dataclass, field
from typing import List, Optional, Union
import json


@dataclass
class Reason:
    """Reason schema for individual or class analysis.

    Attributes:
        reason_key: Pedagogical group name (e.g., "Tự học", "Học lực")
        reason_text: Human-readable reason text in Vietnamese
        impact_percentage: Impact percentage (0-100)
        solutions: List of actionable solutions
    """

    reason_key: str
    reason_text: str
    impact_percentage: float
    solutions: List[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        """Convert to dictionary for JSON serialization.

        Returns:
            Dictionary representation
        """
        return {
            "reason_key": self.reason_key,
            "reason_text": self.reason_text,
            "impact_percentage": round(self.impact_percentage, 2),
            "solutions": self.solutions,
        }

    def to_json(self) -> str:
        """Convert to JSON string.

        Returns:
            JSON string representation
        """
        return json.dumps(self.to_dict(), ensure_ascii=False, indent=2)


@dataclass
class IndividualAnalysisOutput:
    """Output schema for individual student analysis.

    Attributes:
        predicted_clo_score: Predicted CLO score (0-6)
        summary: Summary reason text
        reasons: List of detailed reasons with solutions
        student_id: Student ID (optional, for reference)
        subject_id: Subject ID (optional, for reference)
        lecturer_id: Lecturer ID (optional, for reference)
        actual_clo_score: Điểm CLO thực tế (môn đã học, đã đỗ) — nếu có thì output ưu tiên giá trị này
    """

    predicted_clo_score: float
    summary: str
    reasons: List[Reason] = field(default_factory=list)
    student_id: Optional[str] = None
    subject_id: Optional[str] = None
    lecturer_id: Optional[str] = None
    actual_clo_score: Optional[float] = None

    def __post_init__(self):
        """Validate data after initialization."""
        if not 0 <= self.predicted_clo_score <= 6:
            raise ValueError(
                f"predicted_clo_score must be between 0 and 6, got {self.predicted_clo_score}"
            )
        if self.actual_clo_score is not None and not 0 <= self.actual_clo_score <= 6:
            raise ValueError(
                f"actual_clo_score must be between 0 and 6, got {self.actual_clo_score}"
            )

        for reason in self.reasons:
            if not 0 <= reason.impact_percentage <= 100:
                raise ValueError(
                    f"impact_percentage must be between 0 and 100, "
                    f"got {reason.impact_percentage}"
                )

    def to_dict(self) -> dict:
        """Convert to dictionary for JSON serialization.

        Returns:
            Dictionary representation
        """
        result = {
            "predicted_clo_score": round(self.predicted_clo_score, 2),
            "summary": self.summary,
            "reasons": [reason.to_dict() for reason in self.reasons],
            "student_id": self.student_id,
            "subject_id": self.subject_id,
            "lecturer_id": self.lecturer_id,
        }
        if self.actual_clo_score is not None:
            result["actual_clo_score"] = round(self.actual_clo_score, 2)
        return result

    def to_json(self, indent: Optional[int] = 2) -> str:
        """Convert to JSON string.

        Args:
            indent: JSON indentation (default: 2)

        Returns:
            JSON string representation
        """
        return json.dumps(
            self.to_dict(), ensure_ascii=False, indent=indent
        )

    @classmethod
    def from_explanation_dict(
        cls,
        explanation: dict,
        student_id: Optional[str] = None,
        subject_id: Optional[str] = None,
        lecturer_id: Optional[str] = None,
        predicted_clo_score: Optional[float] = None,
        actual_clo_score: Optional[float] = None,
    ) -> "IndividualAnalysisOutput":
        """Create from reason generator output.

        Args:
            explanation: Dictionary from generate_complete_explanation()
            student_id: Student ID (optional)
            subject_id: Subject ID (optional)
            lecturer_id: Lecturer ID (optional)
            predicted_clo_score: Điểm dự đoán từ model (nếu None thì lấy từ explanation)
            actual_clo_score: Điểm CLO thực (môn đã đỗ) — nếu có thì output ưu tiên hiển thị

        Returns:
            IndividualAnalysisOutput instance
        """
        reasons = [
            Reason(
                reason_key=reason_dict["group_name"],
                reason_text=reason_dict["reason_text"],
                impact_percentage=reason_dict["impact_percentage"],
                solutions=reason_dict.get("solutions", []),
            )
            for reason_dict in explanation.get("reasons", [])
        ]

        score = predicted_clo_score if predicted_clo_score is not None else explanation["predicted_score"]
        return cls(
            predicted_clo_score=score,
            summary=explanation["summary"],
            reasons=reasons,
            student_id=student_id,
            subject_id=subject_id,
            lecturer_id=lecturer_id,
            actual_clo_score=actual_clo_score,
        )


@dataclass
class ClassReason:
    """Reason schema for class-level analysis.

    Attributes:
        reason_key: Pedagogical group name
        reason_text: Human-readable reason text in Vietnamese
        average_impact_percentage: Average impact percentage across class
        affected_students_count: Number of students affected by this reason
        priority_solutions: List of priority solutions for the class
    """

    reason_key: str
    reason_text: str
    average_impact_percentage: float
    affected_students_count: int
    priority_solutions: List[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        """Convert to dictionary for JSON serialization.

        Returns:
            Dictionary representation
        """
        return {
            "reason_key": self.reason_key,
            "reason_text": self.reason_text,
            "average_impact_percentage": round(self.average_impact_percentage, 2),
            "affected_students_count": self.affected_students_count,
            "priority_solutions": self.priority_solutions,
        }

    def to_json(self) -> str:
        """Convert to JSON string.

        Returns:
            JSON string representation
        """
        return json.dumps(self.to_dict(), ensure_ascii=False, indent=2)


@dataclass
class ClassAnalysisOutput:
    """Output schema for class-level analysis.

    Attributes:
        summary: Summary reason text for the class
        common_reasons: List of common reasons affecting the class
        subject_id: Subject ID
        lecturer_id: Lecturer ID
        total_students: Total number of students analyzed
        average_predicted_score: Average predicted CLO score for the class
    """

    summary: str
    common_reasons: List[ClassReason] = field(default_factory=list)
    subject_id: Optional[str] = None
    lecturer_id: Optional[str] = None
    total_students: Optional[int] = None
    average_predicted_score: Optional[float] = None

    def __post_init__(self):
        """Validate data after initialization."""
        if self.average_predicted_score is not None:
            if not 0 <= self.average_predicted_score <= 6:
                raise ValueError(
                    f"average_predicted_score must be between 0 and 6, "
                    f"got {self.average_predicted_score}"
                )

        for reason in self.common_reasons:
            if not 0 <= reason.average_impact_percentage <= 100:
                raise ValueError(
                    f"average_impact_percentage must be between 0 and 100, "
                    f"got {reason.average_impact_percentage}"
                )

    def to_dict(self) -> dict:
        """Convert to dictionary for JSON serialization.

        Returns:
            Dictionary representation
        """
        result = {
            "summary": self.summary,
            "common_reasons": [reason.to_dict() for reason in self.common_reasons],
        }

        if self.subject_id is not None:
            result["subject_id"] = self.subject_id
        if self.lecturer_id is not None:
            result["lecturer_id"] = self.lecturer_id
        if self.total_students is not None:
            result["total_students"] = self.total_students
        if self.average_predicted_score is not None:
            result["average_predicted_score"] = round(
                self.average_predicted_score, 2
            )

        return result

    def to_json(self, indent: Optional[int] = 2) -> str:
        """Convert to JSON string.

        Args:
            indent: JSON indentation (default: 2)

        Returns:
            JSON string representation
        """
        return json.dumps(
            self.to_dict(), ensure_ascii=False, indent=indent
        )

    @classmethod
    def from_explanation_dict(
        cls,
        explanation: dict,
        subject_id: Optional[str] = None,
        lecturer_id: Optional[str] = None,
        total_students: Optional[int] = None,
        average_predicted_score: Optional[float] = None,
    ) -> "ClassAnalysisOutput":
        """Create from reason generator output.

        Args:
            explanation: Dictionary from generate_complete_explanation()
            subject_id: Subject ID (optional)
            lecturer_id: Lecturer ID (optional)
            total_students: Total number of students (optional)
            average_predicted_score: Average predicted score (optional)

        Returns:
            ClassAnalysisOutput instance
        """
        common_reasons = [
            ClassReason(
                reason_key=reason_dict["group_name"],
                reason_text=reason_dict["reason_text"],
                average_impact_percentage=reason_dict["impact_percentage"],
                affected_students_count=total_students or 0,  # Default to total if not specified
                priority_solutions=reason_dict.get("solutions", []),
            )
            for reason_dict in explanation.get("reasons", [])
        ]

        return cls(
            summary=explanation["summary"],
            common_reasons=common_reasons,
            subject_id=subject_id,
            lecturer_id=lecturer_id,
            total_students=total_students,
            average_predicted_score=average_predicted_score,
        )


def validate_output(output: Union[IndividualAnalysisOutput, ClassAnalysisOutput]) -> bool:
    """Validate output schema.

    Args:
        output: Output schema instance

    Returns:
        True if valid, raises ValueError if invalid
    """
    if isinstance(output, IndividualAnalysisOutput):
        if not 0 <= output.predicted_clo_score <= 6:
            raise ValueError(
                f"Invalid predicted_clo_score: {output.predicted_clo_score}"
            )
        if output.actual_clo_score is not None and not 0 <= output.actual_clo_score <= 6:
            raise ValueError(
                f"Invalid actual_clo_score: {output.actual_clo_score}"
            )
        for reason in output.reasons:
            if not 0 <= reason.impact_percentage <= 100:
                raise ValueError(
                    f"Invalid impact_percentage: {reason.impact_percentage}"
                )
    elif isinstance(output, ClassAnalysisOutput):
        if output.average_predicted_score is not None:
            if not 0 <= output.average_predicted_score <= 6:
                raise ValueError(
                    f"Invalid average_predicted_score: {output.average_predicted_score}"
                )
        for reason in output.common_reasons:
            if not 0 <= reason.average_impact_percentage <= 100:
                raise ValueError(
                    f"Invalid average_impact_percentage: {reason.average_impact_percentage}"
                )

    return True

