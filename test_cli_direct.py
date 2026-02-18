"""Direct test for CLI scripts - test functions directly."""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))


def test_train_script_functions():
    """Test train.py functions."""
    print("Testing train.py functions...")
    try:
        from scripts.train import parse_args, validate_paths

        # Test parse_args with minimal args
        test_args = [
            "--exam-scores",
            "data/DiemTong.xlsx",
            "--output",
            "models/test.joblib",
        ]

        # Save original sys.argv
        original_argv = sys.argv
        sys.argv = ["train.py"] + test_args

        try:
            args = parse_args()
            print(f"✓ parse_args() works: exam_scores={args.exam_scores}, output={args.output}")
        finally:
            sys.argv = original_argv

        # Test validate_paths (will fail if files don't exist, which is OK)
        class MockArgs:
            def __init__(self):
                self.exam_scores = "data/DiemTong.xlsx"
                self.output = "models/test_model.joblib"
                self.conduct_scores = None
                self.demographics = None
                self.teaching_methods = None
                self.assessment_methods = None
                self.study_hours = None

        args = MockArgs()
        try:
            validate_paths(args)
            print("✓ validate_paths() works (files exist)")
        except SystemExit:
            print("⚠ validate_paths() found missing files (expected if data not available)")

        return True

    except Exception as e:
        print(f"✗ Error: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_predict_script_functions():
    """Test predict.py functions."""
    print("\nTesting predict.py functions...")
    try:
        from scripts.predict import parse_args, validate_paths

        # Test parse_args with minimal args
        test_args = [
            "--model",
            "models/test.joblib",
            "--student-id",
            "19050006",
            "--subject-id",
            "INF0823",
            "--lecturer-id",
            "90316",
            "--exam-scores",
            "data/DiemTong.xlsx",
        ]

        original_argv = sys.argv
        sys.argv = ["predict.py"] + test_args

        try:
            args = parse_args()
            print(f"✓ parse_args() works: student_id={args.student_id}, subject_id={args.subject_id}")
        finally:
            sys.argv = original_argv

        return True

    except Exception as e:
        print(f"✗ Error: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_analyze_script_functions():
    """Test analyze_class.py functions."""
    print("\nTesting analyze_class.py functions...")
    try:
        from scripts.analyze_class import parse_args, load_actual_scores

        # Test parse_args with minimal args
        test_args = [
            "--model",
            "models/test.joblib",
            "--subject-id",
            "INF0823",
            "--lecturer-id",
            "90316",
            "--exam-scores",
            "data/DiemTong.xlsx",
        ]

        original_argv = sys.argv
        sys.argv = ["analyze_class.py"] + test_args

        try:
            args = parse_args()
            print(f"✓ parse_args() works: subject_id={args.subject_id}, lecturer_id={args.lecturer_id}")
        finally:
            sys.argv = original_argv

        # Test load_actual_scores with a sample JSON
        import json
        import tempfile

        sample_scores = {"19050006": 4.5, "19050007": 3.8}
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump(sample_scores, f)
            temp_path = f.name

        try:
            loaded = load_actual_scores(temp_path)
            if loaded == sample_scores:
                print("✓ load_actual_scores() works")
            else:
                print(f"✗ load_actual_scores() returned wrong data: {loaded}")
        finally:
            Path(temp_path).unlink()

        return True

    except Exception as e:
        print(f"✗ Error: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_script_imports():
    """Test that all scripts can be imported."""
    print("\nTesting script imports...")
    try:
        import scripts.train
        import scripts.predict
        import scripts.analyze_class
        print("✓ All scripts can be imported")
        return True
    except Exception as e:
        print(f"✗ Import failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Run direct CLI tests."""
    print("=" * 80)
    print("DIRECT CLI SCRIPT TESTS")
    print("=" * 80)

    results = {}

    results["imports"] = test_script_imports()
    results["train"] = test_train_script_functions()
    results["predict"] = test_predict_script_functions()
    results["analyze"] = test_analyze_script_functions()

    print("\n" + "=" * 80)
    print("TEST SUMMARY")
    print("=" * 80)
    for test_name, passed in results.items():
        status = "✓ PASSED" if passed else "✗ FAILED"
        print(f"{test_name:20s}: {status}")
    print("=" * 80)

    if all(results.values()):
        print("\n✓ All CLI function tests passed!")
        print("\nTo test with real data, you can run:")
        print("  python scripts/train.py --exam-scores data/DiemTong.xlsx --output models/test.joblib")
        print("  python scripts/predict.py --model models/test.joblib --student-id 19050006 --subject-id INF0823 --lecturer-id 90316 --exam-scores data/DiemTong.xlsx")
        print("  python scripts/analyze_class.py --model models/test.joblib --subject-id INF0823 --lecturer-id 90316 --exam-scores data/DiemTong.xlsx")
        return 0
    else:
        print("\n✗ Some CLI tests failed")
        return 1


if __name__ == "__main__":
    sys.exit(main())

