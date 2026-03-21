#!/usr/bin/env python3
"""CLI script for analyzing a class and generating class-level CLO predictions.

Chế độ chính: --scores-file (điểm CLO từ file/API, không cần DiemTong).
Chế độ cũ: --exam-scores (filter DiemTong theo môn+GV) — deprecated.

Usage:
    python scripts/analyze_class.py \\
        --model models/model.joblib \\
        --subject-id INF0823 \\
        --lecturer-id 90316 \\
        --scores-file data/clo_scores.csv \\
        --demographics data/nhankhau.xlsx \\
        --teaching-methods data/PPGDfull.xlsx \\
        --assessment-methods data/PPDGfull.xlsx
"""

import argparse
import json
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from ml_clo.pipelines import AnalysisPipeline
from ml_clo.utils.exceptions import ModelLoadError
from ml_clo.utils.logger import get_logger

logger = get_logger(__name__)


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Analyze a class and generate class-level CLO predictions with XAI",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples (chế độ chính — --scores-file):
  # Phân tích từ file điểm CLO (điểm từ API/backend)
  python scripts/analyze_class.py \\
      --model models/model.joblib \\
      --subject-id INF0823 \\
      --lecturer-id 90316 \\
      --scores-file data/clo_scores.csv \\
      --demographics data/nhankhau.xlsx \\
      --teaching-methods data/PPGDfull.xlsx \\
      --assessment-methods data/PPDGfull.xlsx \\
      --output result.json

  # Chỉ danh sách điểm (không MSSV) — phân tích phân phối
  python scripts/analyze_class.py \\
      --model models/model.joblib \\
      --subject-id INF0823 \\
      --lecturer-id 90316 \\
      --scores-file data/clo_scores_simple.csv

Examples (chế độ cũ — --exam-scores, deprecated):
  python scripts/analyze_class.py \\
      --model models/model.joblib \\
      --subject-id INF0823 \\
      --lecturer-id 90316 \\
      --exam-scores data/DiemTong.xlsx
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
        help="Path to exam scores Excel file (deprecated — dùng --scores-file)",
    )
    parser.add_argument(
        "--scores-file",
        type=str,
        default=None,
        help="Path to CSV/JSON file với danh sách điểm CLO (chế độ mới). CSV: student_id,clo_score hoặc clo_score. JSON: {\"scores\": [...]} hoặc {\"student_id\": score, ...}",
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

    # Actual scores for retraining
    parser.add_argument(
        "--actual-scores",
        type=str,
        default=None,
        help="Path to JSON file with actual CLO scores (optional, format: {\"student_id\": score, ...})",
    )
    parser.add_argument(
        "--storage-path",
        type=str,
        default=None,
        help="Path to store actual scores for retraining (optional)",
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
    parser.add_argument(
        "--include-average-predicted",
        action="store_true",
        help="Hiển thị average_predicted_score trong output (mặc định ẩn theo feedback)",
    )

    return parser.parse_args()


def validate_paths(args):
    """Validate that input files exist."""
    errors = []

    # Check model file
    if not Path(args.model).exists():
        errors.append(f"Model file not found: {args.model}")

    # Need --exam-scores OR --scores-file
    if not args.exam_scores and not args.scores_file:
        errors.append("Cần --exam-scores hoặc --scores-file")
    if args.exam_scores and not Path(args.exam_scores).exists():
        errors.append(f"Exam scores file not found: {args.exam_scores}")
    if args.scores_file and not Path(args.scores_file).exists():
        errors.append(f"Scores file not found: {args.scores_file}")

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

    # Check actual scores file if provided
    if args.actual_scores and not Path(args.actual_scores).exists():
        errors.append(f"Actual scores file not found: {args.actual_scores}")

    if errors:
        print("ERROR: Validation failed:", file=sys.stderr)
        for error in errors:
            print(f"  - {error}", file=sys.stderr)
        sys.exit(1)


def load_scores_from_file(file_path: str):
    """Load clo_scores từ CSV hoặc JSON.

    Returns:
        Union[Dict[str, float], List[float], List[Tuple[str, float]]]
    """
    path = Path(file_path)
    suffix = path.suffix.lower()

    if suffix == ".json":
        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        if isinstance(data, dict):
            if "scores" in data:
                return data["scores"]
            return data
        if isinstance(data, list):
            return data
        raise ValueError(f"JSON không hợp lệ: cần dict hoặc list")

    if suffix == ".csv":
        import csv
        with open(file_path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            rows = list(reader)
        if not rows:
            raise ValueError("CSV trống")
        keys = list(rows[0].keys())
        keys_lower = {k.lower(): k for k in keys}
        id_col = keys_lower.get("student_id") or keys_lower.get("mssv")
        score_candidates = [k for k in keys if "score" in k.lower() or "clo" in k.lower() or "diem" in k.lower()]
        score_col = score_candidates[0] if score_candidates else keys[-1]
        if id_col:
            return [(str(r.get(id_col, "")), float(r.get(score_col, 0))) for r in rows]
        return [float(r.get(score_col, 0)) for r in rows]

    raise ValueError(f"Định dạng không hỗ trợ: {suffix}. Dùng .csv hoặc .json")


def load_actual_scores(file_path: str) -> dict:
    """Load actual scores from JSON file.

    Args:
        file_path: Path to JSON file with format {"student_id": score, ...}

    Returns:
        Dictionary mapping student_id to actual CLO score
    """
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            scores = json.load(f)
        # Validate scores are numeric
        for student_id, score in scores.items():
            try:
                float(score)
            except (ValueError, TypeError):
                raise ValueError(
                    f"Invalid score for student {student_id}: {score} (must be numeric)"
                )
        return scores
    except json.JSONDecodeError as e:
        raise ValueError(f"Invalid JSON format in {file_path}: {e}")
    except Exception as e:
        raise ValueError(f"Failed to load actual scores from {file_path}: {e}")


def main():
    """Main entry point for class analysis script."""
    args = parse_args()

    # Validate paths
    validate_paths(args)

    # Load actual scores if provided
    actual_scores = None
    if args.actual_scores:
        try:
            actual_scores = load_actual_scores(args.actual_scores)
            print(f"Loaded {len(actual_scores)} actual scores from {args.actual_scores}")
        except ValueError as e:
            print(f"ERROR: {e}", file=sys.stderr)
            return 1

    # Initialize analysis pipeline
    print("=" * 80)
    print("Class-Level CLO Analysis")
    print("=" * 80)
    print(f"Model: {args.model}")
    print(f"Subject ID: {args.subject_id}")
    print(f"Lecturer ID: {args.lecturer_id}")
    if actual_scores:
        print(f"Actual scores: {len(actual_scores)} students")
    if args.storage_path:
        print(f"Storage path: {args.storage_path}")
    print("=" * 80)

    try:
        analyzer = AnalysisPipeline(args.model)

        if args.scores_file:
            clo_scores = load_scores_from_file(args.scores_file)
            result = analyzer.analyze_class_from_scores(
                subject_id=args.subject_id,
                lecturer_id=args.lecturer_id,
                clo_scores=clo_scores,
                demographics_path=args.demographics,
                conduct_scores_path=args.conduct_scores,
                teaching_methods_path=args.teaching_methods,
                assessment_methods_path=args.assessment_methods,
                study_hours_path=args.study_hours,
                attendance_path=args.attendance,
            )
        else:
            logger.warning(
                "Chế độ --exam-scores (filter DiemTong) deprecated. "
                "Nên dùng --scores-file với danh sách điểm CLO."
            )
            result = analyzer.analyze_class(
                subject_id=args.subject_id,
                lecturer_id=args.lecturer_id,
                exam_scores_path=args.exam_scores,
                conduct_scores_path=args.conduct_scores,
                demographics_path=args.demographics,
                teaching_methods_path=args.teaching_methods,
                assessment_methods_path=args.assessment_methods,
                study_hours_path=args.study_hours,
                attendance_path=args.attendance,
                actual_scores=actual_scores,
                storage_path=args.storage_path,
            )

        # Convert to JSON (mặc định ẩn average_predicted_score theo feedback)
        output_json = result.to_json(
            indent=2,
            include_average_predicted=args.include_average_predicted,
        )

        # Save or print
        if args.output:
            output_path = Path(args.output)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            with open(output_path, "w", encoding="utf-8") as f:
                f.write(output_json)
            print(f"\n✓ Analysis saved to: {args.output}")
        else:
            print("\n" + "=" * 80)
            print("ANALYSIS RESULT")
            print("=" * 80)
            print(output_json)

        # Print summary
        print("\n" + "=" * 80)
        print("SUMMARY")
        print("=" * 80)
        print(f"Total students: {result.total_students}")
        if args.include_average_predicted and result.average_predicted_score is not None:
            print(f"Average predicted CLO score: {result.average_predicted_score:.2f}")
        print(f"Summary: {result.summary}")
        print(f"Number of common reasons: {len(result.common_reasons)}")
        if result.common_reasons:
            print("\nTop Common Reasons:")
            for i, reason in enumerate(result.common_reasons[:3], 1):
                print(f"  {i}. {reason.reason_key}: {reason.reason_text[:80]}...")
                print(f"     Impact: {reason.average_impact_percentage:.1f}%")
                print(f"     Affected students: {reason.affected_students_count}")
        print("=" * 80)

        return 0

    except ModelLoadError as e:
        logger.error(f"Model loading error: {e}")
        print(f"\nERROR: Failed to load model: {e}", file=sys.stderr)
        return 1

    except ValueError as e:
        logger.error(f"Analysis error: {e}")
        print(f"\nERROR: Analysis failed: {e}", file=sys.stderr)
        return 1

    except Exception as e:
        logger.exception("Unexpected error during analysis")
        print(f"\nERROR: Analysis failed: {e}", file=sys.stderr)
        if args.verbose:
            import traceback

            traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())

