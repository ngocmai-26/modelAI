#!/usr/bin/env python3
"""CLI script for predicting CLO score for a single student.

Usage:
    python scripts/predict.py \\
        --model models/model.joblib \\
        --student-id 19050006 \\
        --subject-id INF0823 \\
        --lecturer-id 90316 \\
        --exam-scores data/DiemTong.xlsx
"""

import argparse
import json
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from ml_clo.pipelines import PredictionPipeline
from ml_clo.utils.exceptions import ModelLoadError
from ml_clo.utils.logger import get_logger

logger = get_logger(__name__)


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Predict CLO score for a single student with XAI explanation",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic prediction with only exam scores
  python scripts/predict.py \\
      --model models/model.joblib \\
      --student-id 19050006 \\
      --subject-id INF0823 \\
      --lecturer-id 90316 \\
      --exam-scores data/DiemTong.xlsx

  # Full prediction with all data sources (including attendance)
  python scripts/predict.py \\
      --model models/model.joblib \\
      --student-id 19050006 \\
      --subject-id INF0823 \\
      --lecturer-id 90316 \\
      --exam-scores data/DiemTong.xlsx \\
      --conduct-scores data/diemrenluyen.xlsx \\
      --demographics data/nhankhau.xlsx \\
      --teaching-methods data/PPGDfull.xlsx \\
      --assessment-methods data/PPDGfull.xlsx \\
      --study-hours data/tuhoc.xlsx \\
      --attendance "data/Dữ liệu điểm danh Khoa FIRA.xlsx"

  # Môn đã đỗ — truyền điểm thực (--actual-score)
  python scripts/predict.py \\
      --model models/model.joblib \\
      --student-id 19050006 \\
      --subject-id INF0823 \\
      --lecturer-id 90316 \\
      --exam-scores data/DiemTong.xlsx \\
      --actual-score 4.2

  # Không cần DiemTong — dùng nhân khẩu + PPGD/PPDG
  python scripts/predict.py \\
      --model models/model.joblib \\
      --student-id 19050006 \\
      --subject-id INF0823 \\
      --lecturer-id 90316 \\
      --demographics data/nhankhau.xlsx \\
      --teaching-methods data/PPGDfull.xlsx \\
      --assessment-methods data/PPDGfull.xlsx \\
      --output result.json
        """,
    )

    # Required arguments
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        help="Path to trained model file",
    )
    parser.add_argument(
        "--student-id",
        type=str,
        required=True,
        help="Student ID",
    )
    parser.add_argument(
        "--subject-id",
        type=str,
        required=True,
        help="Subject ID",
    )
    parser.add_argument(
        "--lecturer-id",
        type=str,
        required=True,
        help="Lecturer ID",
    )
    parser.add_argument(
        "--exam-scores",
        type=str,
        default=None,
        help="Path to DiemTong (điểm tổng). Bắt buộc cho SV năm 2+; SV năm 1 có thể dùng fallback (--demographics + --teaching-methods + --assessment-methods)",
    )
    parser.add_argument(
        "--actual-score",
        type=float,
        default=None,
        help="Điểm CLO thực tế (môn đã học, đã đỗ) — nếu có thì output ưu tiên giá trị này",
    )

    # Optional data sources
    parser.add_argument(
        "--conduct-scores",
        type=str,
        default=None,
        help="Path to conduct scores Excel file (optional)",
    )
    parser.add_argument(
        "--demographics",
        type=str,
        default=None,
        help="Path to demographics Excel file (optional)",
    )
    parser.add_argument(
        "--teaching-methods",
        type=str,
        default=None,
        help="Path to teaching methods Excel file (optional)",
    )
    parser.add_argument(
        "--assessment-methods",
        type=str,
        default=None,
        help="Path to assessment methods Excel file (optional)",
    )
    parser.add_argument(
        "--study-hours",
        type=str,
        default=None,
        help="Path to study hours Excel file (optional)",
    )
    parser.add_argument(
        "--attendance",
        type=str,
        default=None,
        help="Path to attendance (điểm danh) Excel file (optional)",
    )

    # Output options
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Path to save JSON output (optional, prints to stdout if not specified)",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose logging",
    )

    return parser.parse_args()


def validate_paths(args):
    """Validate that input files exist and data sources are sufficient."""
    errors = []

    # Check model file
    if not Path(args.model).exists():
        errors.append(f"Model file not found: {args.model}")

    # Need exam_scores OR (demographics + teaching_methods + assessment_methods)
    has_exam = args.exam_scores and Path(args.exam_scores).exists()
    has_demo_tm_am = (
        args.demographics and Path(args.demographics).exists()
        and args.teaching_methods and Path(args.teaching_methods).exists()
        and args.assessment_methods and Path(args.assessment_methods).exists()
    )
    if not has_exam and not has_demo_tm_am:
        errors.append(
            "Need --exam-scores OR (--demographics + --teaching-methods + --assessment-methods)"
        )
    if args.exam_scores and not Path(args.exam_scores).exists():
        errors.append(f"Exam scores file not found: {args.exam_scores}")

    # Check optional files
    if args.conduct_scores and not Path(args.conduct_scores).exists():
        errors.append(f"Conduct scores file not found: {args.conduct_scores}")

    if args.demographics and not Path(args.demographics).exists():
        errors.append(f"Demographics file not found: {args.demographics}")

    if args.teaching_methods and not Path(args.teaching_methods).exists():
        errors.append(f"Teaching methods file not found: {args.teaching_methods}")

    if args.assessment_methods and not Path(args.assessment_methods).exists():
        errors.append(f"Assessment methods file not found: {args.assessment_methods}")

    if args.study_hours and not Path(args.study_hours).exists():
        errors.append(f"Study hours file not found: {args.study_hours}")

    if args.attendance and not Path(args.attendance).exists():
        errors.append(f"Attendance file not found: {args.attendance}")

    if errors:
        print("ERROR: Validation failed:", file=sys.stderr)
        for error in errors:
            print(f"  - {error}", file=sys.stderr)
        sys.exit(1)


def main():
    """Main entry point for prediction script."""
    args = parse_args()

    # Validate paths
    validate_paths(args)

    # Initialize prediction pipeline
    print("=" * 80)
    print("CLO Score Prediction")
    print("=" * 80)
    print(f"Model: {args.model}")
    print(f"Student ID: {args.student_id}")
    print(f"Subject ID: {args.subject_id}")
    print(f"Lecturer ID: {args.lecturer_id}")
    print("=" * 80)

    try:
        predictor = PredictionPipeline(
            args.model,
            exam_scores_path=args.exam_scores,
            conduct_scores_path=args.conduct_scores,
            demographics_path=args.demographics,
            teaching_methods_path=args.teaching_methods,
            assessment_methods_path=args.assessment_methods,
            study_hours_path=args.study_hours,
            attendance_path=args.attendance,
        )

        # Run prediction
        result = predictor.predict(
            student_id=args.student_id,
            subject_id=args.subject_id,
            lecturer_id=args.lecturer_id,
            exam_scores_path=args.exam_scores,
            conduct_scores_path=args.conduct_scores,
            demographics_path=args.demographics,
            teaching_methods_path=args.teaching_methods,
            assessment_methods_path=args.assessment_methods,
            study_hours_path=args.study_hours,
            attendance_path=args.attendance,
            actual_clo_score=args.actual_score,
        )

        # Convert to JSON
        output_json = result.to_json(indent=2)

        # Save or print
        if args.output:
            output_path = Path(args.output)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            with open(output_path, "w", encoding="utf-8") as f:
                f.write(output_json)
            print(f"\n✓ Prediction saved to: {args.output}")
        else:
            print("\n" + "=" * 80)
            print("PREDICTION RESULT")
            print("=" * 80)
            print(output_json)

        # Print summary
        print("\n" + "=" * 80)
        print("SUMMARY")
        print("=" * 80)
        if result.actual_clo_score is not None:
            print(f"Actual CLO Score: {result.actual_clo_score:.2f}")
            print(f"Predicted CLO Score: {result.predicted_clo_score:.2f}")
        else:
            print(f"Predicted CLO Score: {result.predicted_clo_score:.2f}")
        print(f"Summary: {result.summary}")
        print(f"Number of reasons: {len(result.reasons)}")
        if result.reasons:
            print("\nTop Reasons:")
            for i, reason in enumerate(result.reasons[:3], 1):
                print(f"  {i}. {reason.reason_key}: {reason.reason_text[:80]}...")
                print(f"     Impact: {reason.impact_percentage:.1f}%")
        print("=" * 80)

        return 0

    except ModelLoadError as e:
        logger.error(f"Model loading error: {e}")
        print(f"\nERROR: Failed to load model: {e}", file=sys.stderr)
        return 1

    except ValueError as e:
        logger.error(f"Prediction error: {e}")
        print(f"\nERROR: Prediction failed: {e}", file=sys.stderr)
        return 1

    except Exception as e:
        logger.exception("Unexpected error during prediction")
        print(f"\nERROR: Prediction failed: {e}", file=sys.stderr)
        if args.verbose:
            import traceback

            traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())

