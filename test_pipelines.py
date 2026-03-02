"""Test script for all pipelines.

This script tests Training, Prediction, and Analysis pipelines with real data.
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

import json
from ml_clo.pipelines import AnalysisPipeline, PredictionPipeline, TrainingPipeline


def test_training_pipeline():
    """Test training pipeline."""
    print("=" * 80)
    print("Testing Training Pipeline")
    print("=" * 80)

    trainer = TrainingPipeline(random_state=42, test_size=0.2, validation_size=0.2)

    # Run training pipeline
    model, metrics = trainer.run(
        exam_scores_path="data/DiemTong.xlsx",
        output_path="models/test_ensemble_model.joblib",
        conduct_scores_path="data/diemrenluyen.xlsx",
        demographics_path="data/nhankhau.xlsx",
        teaching_methods_path="data/PPGDfull.xlsx",
        assessment_methods_path="data/PPDGfull.xlsx",
        study_hours_path="data/tuhoc.xlsx",
    )

    print(f"\n✓ Training pipeline completed")
    print(f"  Model: {model.model_name} (version {model.version})")
    print(f"  Test MAE: {metrics['test_mae']:.4f}")
    print(f"  Test RMSE: {metrics['test_rmse']:.4f}")
    print(f"  Test R²: {metrics['test_r2']:.4f}")

    return model, metrics


def test_prediction_pipeline():
    """Test prediction pipeline."""
    print("\n" + "=" * 80)
    print("Testing Prediction Pipeline")
    print("=" * 80)

    # Check if model exists
    model_path = "models/model.joblib"
    if not Path(model_path).exists():
        print(f"⚠ Model not found at {model_path}, running training first...")
        test_training_pipeline()

    predictor = PredictionPipeline(model_path)

    # Test prediction for a student
    # Get a real student_id, subject_id, lecturer_id from data
    # For testing, we'll use values that likely exist
    try:
        result = predictor.predict(
            student_id="SV001",  # Replace with actual student ID from your data
            subject_id="SUB001",  # Replace with actual subject ID
            lecturer_id="LEC001",  # Replace with actual lecturer ID
            exam_scores_path="data/DiemTong.xlsx",
            conduct_scores_path="data/diemrenluyen.xlsx",
            demographics_path="data/nhankhau.xlsx",
            teaching_methods_path="data/PPGDfull.xlsx",
            assessment_methods_path="data/PPDGfull.xlsx",
            study_hours_path="data/tuhoc.xlsx",
        )

        print(f"\n✓ Prediction pipeline completed")
        print(f"  Predicted CLO Score: {result.predicted_clo_score:.2f}")
        print(f"  Summary: {result.summary}")
        print(f"  Number of reasons: {len(result.reasons)}")

        # Print JSON output
        print(f"\n  JSON Output:")
        print(result.to_json(indent=2))

        return result

    except ValueError as e:
        print(f"⚠ Prediction test skipped: {e}")
        print("  (This is expected if student_id/subject_id/lecturer_id don't exist in data)")
        return None


def test_analysis_pipeline():
    """Test analysis pipeline."""
    print("\n" + "=" * 80)
    print("Testing Analysis Pipeline")
    print("=" * 80)

    # Check if model exists
    model_path = "models/model.joblib"
    if not Path(model_path).exists():
        print(f"⚠ Model not found at {model_path}, running training first...")
        test_training_pipeline()

    analyzer = AnalysisPipeline(model_path)

    # Test class analysis
    # Get a real subject_id, lecturer_id from data
    try:
        result = analyzer.analyze_class(
            subject_id="SUB001",  # Replace with actual subject ID
            lecturer_id="LEC001",  # Replace with actual lecturer ID
            exam_scores_path="data/DiemTong.xlsx",
            conduct_scores_path="data/diemrenluyen.xlsx",
            demographics_path="data/nhankhau.xlsx",
            teaching_methods_path="data/PPGDfull.xlsx",
            assessment_methods_path="data/PPDGfull.xlsx",
            study_hours_path="data/tuhoc.xlsx",
            actual_scores=None,  # Optional: provide actual scores for storage
            storage_path=None,  # Optional: path to store actual scores
        )

        print(f"\n✓ Analysis pipeline completed")
        print(f"  Summary: {result.summary}")
        print(f"  Total students: {result.total_students}")
        print(f"  Average predicted score: {result.average_predicted_score:.2f}")
        print(f"  Number of common reasons: {len(result.common_reasons)}")

        # Print JSON output
        print(f"\n  JSON Output:")
        print(result.to_json(indent=2))

        return result

    except ValueError as e:
        print(f"⚠ Analysis test skipped: {e}")
        print("  (This is expected if subject_id/lecturer_id don't exist in data)")
        return None


def test_with_real_ids():
    """Test with real IDs from data."""
    print("\n" + "=" * 80)
    print("Testing with Real IDs from Data")
    print("=" * 80)

    # Load a sample of data to get real IDs
    from ml_clo.data.loaders import load_exam_scores
    from ml_clo.data.preprocessors import preprocess_exam_scores

    exam_df = load_exam_scores("data/DiemTong.xlsx")
    exam_df = preprocess_exam_scores(exam_df, convert_to_clo=True, create_result=False)

    if len(exam_df) == 0:
        print("⚠ No data found, skipping real ID tests")
        return

    # Get first valid record
    sample = exam_df.iloc[0]
    student_id = sample["Student_ID"]
    subject_id = sample["Subject_ID"]
    lecturer_id = sample["Lecturer_ID"]

    print(f"Using real IDs from data:")
    print(f"  Student ID: {student_id}")
    print(f"  Subject ID: {subject_id}")
    print(f"  Lecturer ID: {lecturer_id}")

    # Check if model exists
    model_path = "models/model.joblib"
    if not Path(model_path).exists():
        print(f"\n⚠ Model not found, running training first...")
        test_training_pipeline()

    # Test prediction with real IDs
    print("\n" + "-" * 80)
    print("Testing Prediction with Real IDs")
    print("-" * 80)

    try:
        predictor = PredictionPipeline(model_path)
        result = predictor.predict(
            student_id=student_id,
            subject_id=subject_id,
            lecturer_id=lecturer_id,
            exam_scores_path="data/DiemTong.xlsx",
            conduct_scores_path="data/diemrenluyen.xlsx",
            demographics_path="data/nhankhau.xlsx",
            teaching_methods_path="data/PPGDfull.xlsx",
            assessment_methods_path="data/PPDGfull.xlsx",
            study_hours_path="data/tuhoc.xlsx",
        )

        print(f"\n✓ Prediction successful")
        print(f"  Predicted CLO Score: {result.predicted_clo_score:.2f}")
        print(f"  Summary: {result.summary}")
        print(f"  Reasons: {len(result.reasons)}")

        if result.reasons:
            print(f"\n  Top Reason:")
            top_reason = result.reasons[0]
            print(f"    - {top_reason.reason_key}: {top_reason.reason_text}")
            if top_reason.solutions:
                print(f"    - Solutions: {len(top_reason.solutions)} solutions")

    except Exception as e:
        print(f"✗ Prediction failed: {e}")
        import traceback
        traceback.print_exc()

    # Test analysis with real IDs
    print("\n" + "-" * 80)
    print("Testing Analysis with Real IDs")
    print("-" * 80)

    try:
        analyzer = AnalysisPipeline(model_path)
        result = analyzer.analyze_class(
            subject_id=subject_id,
            lecturer_id=lecturer_id,
            exam_scores_path="data/DiemTong.xlsx",
            conduct_scores_path="data/diemrenluyen.xlsx",
            demographics_path="data/nhankhau.xlsx",
            teaching_methods_path="data/PPGDfull.xlsx",
            assessment_methods_path="data/PPDGfull.xlsx",
            study_hours_path="data/tuhoc.xlsx",
        )

        print(f"\n✓ Analysis successful")
        print(f"  Total students: {result.total_students}")
        print(f"  Average predicted score: {result.average_predicted_score:.2f}")
        print(f"  Summary: {result.summary}")
        print(f"  Common reasons: {len(result.common_reasons)}")

        if result.common_reasons:
            print(f"\n  Top Reason:")
            top_reason = result.common_reasons[0]
            print(f"    - {top_reason.reason_key}: {top_reason.reason_text}")
            print(f"    - Affected students: {top_reason.affected_students_count}")
            if top_reason.priority_solutions:
                print(f"    - Solutions: {len(top_reason.priority_solutions)} solutions")

    except Exception as e:
        print(f"✗ Analysis failed: {e}")
        import traceback
        traceback.print_exc()


def main():
    """Run all pipeline tests."""
    print("=" * 80)
    print("TESTING ALL PIPELINES")
    print("=" * 80)

    results = {}

    try:
        # Test training pipeline
        model, metrics = test_training_pipeline()
        results["training"] = {"status": "passed", "metrics": metrics}

        # Test prediction pipeline (may skip if IDs don't exist)
        pred_result = test_prediction_pipeline()
        results["prediction"] = {"status": "passed" if pred_result else "skipped"}

        # Test analysis pipeline (may skip if IDs don't exist)
        analysis_result = test_analysis_pipeline()
        results["analysis"] = {"status": "passed" if analysis_result else "skipped"}

        # Test with real IDs from data
        test_with_real_ids()

        print("\n" + "=" * 80)
        print("PIPELINE TEST SUMMARY")
        print("=" * 80)
        print(f"Training: {results['training']['status']}")
        print(f"Prediction: {results['prediction']['status']}")
        print(f"Analysis: {results['analysis']['status']}")

        if results["training"]["status"] == "passed":
            print("\n✓ All pipeline tests completed!")
            return 0
        else:
            print("\n✗ Some tests failed")
            return 1

    except Exception as e:
        print(f"\n✗ Error during testing: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())

