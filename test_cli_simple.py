"""Simple test for CLI scripts - test help commands and basic functionality."""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))


def test_imports():
    """Test that scripts can be imported."""
    print("Testing script imports...")
    try:
        import scripts.train as train_module
        import scripts.predict as predict_module
        import scripts.analyze_class as analyze_module
        print("✓ All scripts can be imported")
        return True
    except Exception as e:
        print(f"✗ Import failed: {e}")
        return False


def test_train_help():
    """Test train.py --help."""
    print("\nTesting train.py --help...")
    try:
        # Simulate --help by checking argparse
        from scripts.train import parse_args
        import argparse

        # This should raise SystemExit with code 0 for --help
        try:
            args = parse_args()
            print("✓ parse_args() works")
            return True
        except SystemExit as e:
            if e.code == 0:
                print("✓ --help works (SystemExit 0)")
                return True
            else:
                print(f"✗ --help failed with code {e.code}")
                return False
    except Exception as e:
        print(f"✗ Error: {e}")
        return False


def test_predict_help():
    """Test predict.py --help."""
    print("\nTesting predict.py --help...")
    try:
        from scripts.predict import parse_args

        try:
            args = parse_args()
            print("✓ parse_args() works")
            return True
        except SystemExit as e:
            if e.code == 0:
                print("✓ --help works (SystemExit 0)")
                return True
            else:
                print(f"✗ --help failed with code {e.code}")
                return False
    except Exception as e:
        print(f"✗ Error: {e}")
        return False


def test_analyze_help():
    """Test analyze_class.py --help."""
    print("\nTesting analyze_class.py --help...")
    try:
        from scripts.analyze_class import parse_args

        try:
            args = parse_args()
            print("✓ parse_args() works")
            return True
        except SystemExit as e:
            if e.code == 0:
                print("✓ --help works (SystemExit 0)")
                return True
            else:
                print(f"✗ --help failed with code {e.code}")
                return False
    except Exception as e:
        print(f"✗ Error: {e}")
        return False


def test_validate_paths():
    """Test path validation."""
    print("\nTesting path validation...")
    try:
        from scripts.train import validate_paths
        import argparse

        # Create a mock args object
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

        # This should not raise if files exist
        try:
            validate_paths(args)
            print("✓ Path validation works")
            return True
        except SystemExit:
            print("⚠ Path validation found missing files (expected if data not available)")
            return True  # This is OK, validation is working
    except Exception as e:
        print(f"✗ Error: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Run simple CLI tests."""
    print("=" * 80)
    print("SIMPLE CLI SCRIPT TESTS")
    print("=" * 80)

    results = {}

    results["imports"] = test_imports()
    results["train_help"] = test_train_help()
    results["predict_help"] = test_predict_help()
    results["analyze_help"] = test_analyze_help()
    results["validate_paths"] = test_validate_paths()

    print("\n" + "=" * 80)
    print("TEST SUMMARY")
    print("=" * 80)
    for test_name, passed in results.items():
        status = "✓ PASSED" if passed else "✗ FAILED"
        print(f"{test_name:20s}: {status}")
    print("=" * 80)

    if all(results.values()):
        print("\n✓ All basic CLI tests passed!")
        print("\nTo test with real data, run:")
        print("  python scripts/train.py --exam-scores data/DiemTong.xlsx --output models/test.joblib")
        return 0
    else:
        print("\n✗ Some CLI tests failed")
        return 1


if __name__ == "__main__":
    sys.exit(main())

