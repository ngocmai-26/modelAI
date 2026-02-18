#!/usr/bin/env python3
"""CLI script for training CLO prediction model.

Usage:
    python scripts/train.py --exam-scores data/DiemTong.xlsx --output models/model.joblib
    python scripts/train.py --exam-scores data/DiemTong.xlsx --output models/model.joblib \\
        --conduct-scores data/diemrenluyen.xlsx \\
        --demographics data/nhankhau.xlsx \\
        --teaching-methods data/PPGDfull.xlsx \\
        --assessment-methods data/PPDGfull.xlsx \\
        --study-hours data/tuhoc.xlsx
"""

import argparse
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from ml_clo.pipelines import TrainingPipeline
from ml_clo.utils.exceptions import DataLoadError, DataValidationError
from ml_clo.utils.logger import get_logger

logger = get_logger(__name__)


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Train CLO prediction model using ensemble of Random Forest and Gradient Boosting",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic training with only exam scores
  python scripts/train.py --exam-scores data/DiemTong.xlsx --output models/model.joblib

  # Full training with all data sources
  python scripts/train.py \\
      --exam-scores data/DiemTong.xlsx \\
      --output models/model.joblib \\
      --conduct-scores data/diemrenluyen.xlsx \\
      --demographics data/nhankhau.xlsx \\
      --teaching-methods data/PPGDfull.xlsx \\
      --assessment-methods data/PPDGfull.xlsx \\
      --study-hours data/tuhoc.xlsx

  # Training with custom random state and split ratios
  python scripts/train.py \\
      --exam-scores data/DiemTong.xlsx \\
      --output models/model.joblib \\
      --random-state 123 \\
      --test-size 0.15 \\
      --validation-size 0.15
        """,
    )

    # Required arguments
    parser.add_argument(
        "--exam-scores",
        type=str,
        required=True,
        help="Path to exam scores Excel file (required)",
    )
    parser.add_argument(
        "--output",
        type=str,
        required=True,
        help="Path to save trained model (e.g., models/ensemble_model.joblib)",
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

    # Training parameters
    parser.add_argument(
        "--random-state",
        type=int,
        default=42,
        help="Random seed for reproducibility (default: 42)",
    )
    parser.add_argument(
        "--test-size",
        type=float,
        default=0.2,
        help="Proportion of data for testing (default: 0.2)",
    )
    parser.add_argument(
        "--validation-size",
        type=float,
        default=0.2,
        help="Proportion of training data for validation (default: 0.2)",
    )

    # Output options
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose logging",
    )

    return parser.parse_args()


def validate_paths(args):
    """Validate that input files exist."""
    errors = []

    # Check required file
    if not Path(args.exam_scores).exists():
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

    # Check output directory exists
    output_path = Path(args.output)
    if not output_path.parent.exists():
        try:
            output_path.parent.mkdir(parents=True, exist_ok=True)
        except Exception as e:
            errors.append(f"Cannot create output directory: {e}")

    if errors:
        print("ERROR: Validation failed:", file=sys.stderr)
        for error in errors:
            print(f"  - {error}", file=sys.stderr)
        sys.exit(1)


def main():
    """Main entry point for training script."""
    args = parse_args()

    # Validate paths
    validate_paths(args)

    # Initialize training pipeline
    print("=" * 80)
    print("CLO Prediction Model Training")
    print("=" * 80)
    print(f"Exam scores: {args.exam_scores}")
    if args.conduct_scores:
        print(f"Conduct scores: {args.conduct_scores}")
    if args.demographics:
        print(f"Demographics: {args.demographics}")
    if args.teaching_methods:
        print(f"Teaching methods: {args.teaching_methods}")
    if args.assessment_methods:
        print(f"Assessment methods: {args.assessment_methods}")
    if args.study_hours:
        print(f"Study hours: {args.study_hours}")
    print(f"Output: {args.output}")
    print(f"Random state: {args.random_state}")
    print(f"Test size: {args.test_size}")
    print(f"Validation size: {args.validation_size}")
    print("=" * 80)

    try:
        trainer = TrainingPipeline(
            random_state=args.random_state,
            test_size=args.test_size,
            validation_size=args.validation_size,
        )

        # Run training pipeline
        model, metrics = trainer.run(
            exam_scores_path=args.exam_scores,
            output_path=args.output,
            conduct_scores_path=args.conduct_scores,
            demographics_path=args.demographics,
            teaching_methods_path=args.teaching_methods,
            assessment_methods_path=args.assessment_methods,
            study_hours_path=args.study_hours,
        )

        # Print summary
        print("\n" + "=" * 80)
        print("TRAINING COMPLETED SUCCESSFULLY")
        print("=" * 80)
        print(f"Model: {model.model_name} (version {model.version})")
        print(f"Model saved to: {args.output}")
        print("\nTest Metrics:")
        print(f"  MAE:  {metrics['test_mae']:.4f}")
        print(f"  RMSE: {metrics['test_rmse']:.4f}")
        print(f"  R²:   {metrics['test_r2']:.4f}")
        print("\nValidation Metrics:")
        print(f"  MAE:  {metrics['val_mae']:.4f}")
        print(f"  RMSE: {metrics['val_rmse']:.4f}")
        print(f"  R²:   {metrics['val_r2']:.4f}")
        print("\nTraining Metrics:")
        print(f"  MAE:  {metrics['train_mae']:.4f}")
        print(f"  RMSE: {metrics['train_rmse']:.4f}")
        print(f"  R²:   {metrics['train_r2']:.4f}")
        print("=" * 80)

        return 0

    except DataLoadError as e:
        logger.error(f"Data loading error: {e}")
        print(f"\nERROR: Failed to load data: {e}", file=sys.stderr)
        return 1

    except DataValidationError as e:
        logger.error(f"Data validation error: {e}")
        print(f"\nERROR: Data validation failed: {e}", file=sys.stderr)
        return 1

    except Exception as e:
        logger.exception("Unexpected error during training")
        print(f"\nERROR: Training failed: {e}", file=sys.stderr)
        if args.verbose:
            import traceback

            traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())

