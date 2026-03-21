"""Test script for output schemas.

This script tests JSON serialization and validation of output schemas.
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

import json
from ml_clo.outputs.schemas import (
    IndividualAnalysisOutput,
    ClassAnalysisOutput,
    Reason,
    ClassReason,
    validate_output,
)


def test_reason_schema():
    """Test Reason schema."""
    print("=" * 80)
    print("Testing Reason Schema")
    print("=" * 80)

    reason = Reason(
        reason_key="Học lực",
        reason_text="Sinh viên có học lực kém, thiếu nền tảng kiến thức.",
        impact_percentage=91.3,
        solutions=[
            "Ôn tập lại kiến thức cơ bản",
            "Tham gia các lớp bổ trợ",
        ],
    )

    # Test to_dict
    reason_dict = reason.to_dict()
    print(f"\nReason dict: {json.dumps(reason_dict, ensure_ascii=False, indent=2)}")

    # Test to_json
    reason_json = reason.to_json()
    print(f"\nReason JSON:\n{reason_json}")

    # Verify JSON is valid
    parsed = json.loads(reason_json)
    assert parsed["reason_key"] == "Học lực"
    assert parsed["impact_percentage"] == 91.3
    assert len(parsed["solutions"]) == 2

    print("✓ Reason schema test passed")
    return True


def test_individual_analysis_output():
    """Test IndividualAnalysisOutput schema."""
    print("\n" + "=" * 80)
    print("Testing IndividualAnalysisOutput Schema")
    print("=" * 80)

    # Create reasons
    reasons = [
        Reason(
            reason_key="Học lực",
            reason_text="Sinh viên có học lực kém, thiếu nền tảng kiến thức.",
            impact_percentage=91.3,
            solutions=[
                "Ôn tập lại kiến thức cơ bản",
                "Tham gia các lớp bổ trợ",
            ],
        ),
        Reason(
            reason_key="Tự học",
            reason_text="Sinh viên có số giờ tự học thấp.",
            impact_percentage=8.7,
            solutions=[
                "Tăng thời gian tự học tại thư viện",
                "Thiết lập kế hoạch học tập",
            ],
        ),
    ]

    # Create output
    output = IndividualAnalysisOutput(
        predicted_clo_score=4.80,
        summary="Nguyên nhân chính ảnh hưởng đến kết quả học tập là học lực (91.3%).",
        reasons=reasons,
        student_id="SV001",
        subject_id="SUB001",
        lecturer_id="LEC001",
    )

    # Test to_dict
    output_dict = output.to_dict()
    print(f"\nOutput dict keys: {list(output_dict.keys())}")
    print(f"Number of reasons: {len(output_dict['reasons'])}")

    # Test to_json
    output_json = output.to_json()
    print(f"\nOutput JSON:\n{output_json}")

    # Verify JSON is valid
    parsed = json.loads(output_json)
    assert parsed["predicted_clo_score"] == 4.80
    assert len(parsed["reasons"]) == 2
    assert parsed["student_id"] == "SV001"

    # Test from_explanation_dict
    explanation_dict = {
        "predicted_score": 4.80,
        "summary": "Nguyên nhân chính là học lực.",
        "reasons": [
            {
                "group_name": "Học lực",
                "reason_text": "Sinh viên có học lực kém.",
                "impact_percentage": 91.3,
                "solutions": ["Ôn tập lại kiến thức"],
            }
        ],
    }

    output_from_dict = IndividualAnalysisOutput.from_explanation_dict(
        explanation_dict,
        student_id="SV001",
        subject_id="SUB001",
    )

    assert output_from_dict.predicted_clo_score == 4.80
    assert len(output_from_dict.reasons) == 1
    assert output_from_dict.reasons[0].reason_key == "Học lực"

    # Test actual_clo_score (yêu cầu mới: môn đã đỗ)
    output_with_actual = IndividualAnalysisOutput.from_explanation_dict(
        explanation_dict,
        student_id="SV001",
        subject_id="SUB001",
        actual_clo_score=4.2,
    )
    assert output_with_actual.actual_clo_score == 4.2
    assert output_with_actual.predicted_clo_score == 4.80  # model prediction
    d = output_with_actual.to_dict()
    assert "actual_clo_score" in d and d["actual_clo_score"] == 4.2

    print("✓ IndividualAnalysisOutput schema test passed")
    return True


def test_class_analysis_output():
    """Test ClassAnalysisOutput schema."""
    print("\n" + "=" * 80)
    print("Testing ClassAnalysisOutput Schema")
    print("=" * 80)

    # Create class reasons
    common_reasons = [
        ClassReason(
            reason_key="Tự học",
            reason_text="Lớp học có vấn đề nghiêm trọng về thời gian tự học.",
            average_impact_percentage=100.0,
            affected_students_count=50,
            priority_solutions=[
                "Tổ chức các buổi hướng dẫn phương pháp tự học",
                "Khuyến khích sinh viên thành lập nhóm học tập",
            ],
        ),
    ]

    # Create output
    output = ClassAnalysisOutput(
        summary="Nguyên nhân chính ảnh hưởng đến kết quả học tập của lớp là tự học (100.0%).",
        common_reasons=common_reasons,
        subject_id="SUB001",
        lecturer_id="LEC001",
        total_students=50,
        average_predicted_score=3.5,
    )

    # Test to_dict (mặc định không có average_predicted_score)
    output_dict = output.to_dict()
    assert "average_predicted_score" not in output_dict
    print(f"\nOutput dict keys (default): {list(output_dict.keys())}")

    # Test to_dict với include_average_predicted=True
    output_dict_full = output.to_dict(include_average_predicted=True)
    assert output_dict_full["average_predicted_score"] == 3.5
    print(f"Output dict keys (include_average): {list(output_dict_full.keys())}")

    # Test to_json (mặc định ẩn average)
    output_json = output.to_json()
    parsed = json.loads(output_json)
    assert "average_predicted_score" not in parsed
    assert len(parsed["common_reasons"]) == 1
    assert parsed["total_students"] == 50
    assert parsed["common_reasons"][0]["affected_students_count"] == 50

    # Test to_json với include_average_predicted=True
    output_json_full = output.to_json(include_average_predicted=True)
    parsed_full = json.loads(output_json_full)
    assert parsed_full["average_predicted_score"] == 3.5

    # Test from_explanation_dict
    explanation_dict = {
        "predicted_score": 3.5,
        "summary": "Nguyên nhân chính là tự học.",
        "reasons": [
            {
                "group_name": "Tự học",
                "reason_text": "Lớp học có vấn đề về thời gian tự học.",
                "impact_percentage": 100.0,
                "solutions": ["Tổ chức các buổi hướng dẫn"],
            }
        ],
    }

    output_from_dict = ClassAnalysisOutput.from_explanation_dict(
        explanation_dict,
        subject_id="SUB001",
        lecturer_id="LEC001",
        total_students=50,
        average_predicted_score=3.5,
    )

    assert len(output_from_dict.common_reasons) == 1
    assert output_from_dict.common_reasons[0].reason_key == "Tự học"
    assert output_from_dict.total_students == 50

    print("✓ ClassAnalysisOutput schema test passed")
    return True


def test_validation():
    """Test output validation."""
    print("\n" + "=" * 80)
    print("Testing Output Validation")
    print("=" * 80)

    # Test valid individual output
    valid_individual = IndividualAnalysisOutput(
        predicted_clo_score=4.5,
        summary="Test summary",
        reasons=[],
    )
    assert validate_output(valid_individual) is True

    # Test invalid individual output (score out of range)
    try:
        invalid_individual = IndividualAnalysisOutput(
            predicted_clo_score=10.0,  # Invalid: > 6
            summary="Test summary",
            reasons=[],
        )
        assert False, "Should have raised ValueError"
    except ValueError as e:
        print(f"✓ Caught expected validation error: {e}")

    # Test valid class output
    valid_class = ClassAnalysisOutput(
        summary="Test summary",
        common_reasons=[],
        average_predicted_score=3.5,
    )
    assert validate_output(valid_class) is True

    # Test invalid class output (score out of range)
    try:
        invalid_class = ClassAnalysisOutput(
            summary="Test summary",
            common_reasons=[],
            average_predicted_score=10.0,  # Invalid: > 6
        )
        assert False, "Should have raised ValueError"
    except ValueError as e:
        print(f"✓ Caught expected validation error: {e}")

    print("✓ Validation test passed")
    return True


def main():
    """Run all tests."""
    print("=" * 80)
    print("TESTING OUTPUT SCHEMAS")
    print("=" * 80)

    try:
        test_reason_schema()
        test_individual_analysis_output()
        test_class_analysis_output()
        test_validation()

        print("\n" + "=" * 80)
        print("✓ All output schema tests passed!")
        print("=" * 80)
        return 0
    except Exception as e:
        print(f"\n✗ Error during testing: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())

