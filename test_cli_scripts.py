"""Test script for CLI scripts with real data."""

import json
import subprocess
import sys
from pathlib import Path


def test_train_script():
    """Test train.py script."""
    print("=" * 80)
    print("Testing train.py")
    print("=" * 80)

    cmd = [
        sys.executable,
        "scripts/train.py",
        "--exam-scores",
        "data/DiemTong.xlsx",
        "--output",
        "models/cli_test_model.joblib",
        "--conduct-scores",
        "data/diemrenluyen.xlsx",
        "--demographics",
        "data/nhankhau.xlsx",
        "--teaching-methods",
        "data/PPGDfull.xlsx",
        "--assessment-methods",
        "data/PPDGfull.xlsx",
        "--study-hours",
        "data/tuhoc.xlsx",
    ]

    print(f"Running: {' '.join(cmd)}")
    print()

    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=300,  # 5 minutes timeout
        )

        print("STDOUT:")
        print(result.stdout)
        if result.stderr:
            print("\nSTDERR:")
            print(result.stderr)

        if result.returncode == 0:
            print("\n✓ train.py completed successfully")
            return True
        else:
            print(f"\n✗ train.py failed with exit code {result.returncode}")
            return False

    except subprocess.TimeoutExpired:
        print("\n✗ train.py timed out after 5 minutes")
        return False
    except Exception as e:
        print(f"\n✗ Error running train.py: {e}")
        return False


def test_predict_script():
    """Test predict.py script."""
    print("\n" + "=" * 80)
    print("Testing predict.py")
    print("=" * 80)

    # Check if model exists
    model_path = "models/cli_test_model.joblib"
    if not Path(model_path).exists():
        print(f"⚠ Model not found at {model_path}, running training first...")
        if not test_train_script():
            print("⚠ Training failed, skipping prediction test")
            return False

    # Get real IDs from data
    from ml_clo.data.loaders import load_exam_scores
    from ml_clo.data.preprocessors import preprocess_exam_scores

    exam_df = load_exam_scores("data/DiemTong.xlsx")
    exam_df = preprocess_exam_scores(exam_df, convert_to_clo=True, create_result=False)

    if len(exam_df) == 0:
        print("⚠ No data found, skipping prediction test")
        return False

    sample = exam_df.iloc[0]
    student_id = sample["Student_ID"]
    subject_id = sample["Subject_ID"]
    lecturer_id = sample["Lecturer_ID"]

    print(f"Using real IDs: student={student_id}, subject={subject_id}, lecturer={lecturer_id}")

    cmd = [
        sys.executable,
        "scripts/predict.py",
        "--model",
        model_path,
        "--student-id",
        str(student_id),
        "--subject-id",
        str(subject_id),
        "--lecturer-id",
        str(lecturer_id),
        "--exam-scores",
        "data/DiemTong.xlsx",
        "--conduct-scores",
        "data/diemrenluyen.xlsx",
        "--demographics",
        "data/nhankhau.xlsx",
        "--teaching-methods",
        "data/PPGDfull.xlsx",
        "--assessment-methods",
        "data/PPDGfull.xlsx",
        "--study-hours",
        "data/tuhoc.xlsx",
        "--output",
        "test_predict_output.json",
    ]

    print(f"Running: {' '.join(cmd)}")
    print()

    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=120,  # 2 minutes timeout
        )

        print("STDOUT:")
        print(result.stdout)
        if result.stderr:
            print("\nSTDERR:")
            print(result.stderr)

        if result.returncode == 0:
            # Check if output file was created
            if Path("test_predict_output.json").exists():
                with open("test_predict_output.json", "r", encoding="utf-8") as f:
                    output = json.load(f)
                print(f"\n✓ predict.py completed successfully")
                print(f"  Predicted score: {output.get('predicted_clo_score', 'N/A')}")
                print(f"  Number of reasons: {len(output.get('reasons', []))}")
                return True
            else:
                print("\n⚠ predict.py completed but output file not found")
                return False
        else:
            print(f"\n✗ predict.py failed with exit code {result.returncode}")
            return False

    except subprocess.TimeoutExpired:
        print("\n✗ predict.py timed out after 2 minutes")
        return False
    except Exception as e:
        print(f"\n✗ Error running predict.py: {e}")
        return False


def test_analyze_class_script():
    """Test analyze_class.py script."""
    print("\n" + "=" * 80)
    print("Testing analyze_class.py")
    print("=" * 80)

    # Check if model exists
    model_path = "models/cli_test_model.joblib"
    if not Path(model_path).exists():
        print(f"⚠ Model not found at {model_path}, running training first...")
        if not test_train_script():
            print("⚠ Training failed, skipping analysis test")
            return False

    # Get real IDs from data
    from ml_clo.data.loaders import load_exam_scores
    from ml_clo.data.preprocessors import preprocess_exam_scores

    exam_df = load_exam_scores("data/DiemTong.xlsx")
    exam_df = preprocess_exam_scores(exam_df, convert_to_clo=True, create_result=False)

    if len(exam_df) == 0:
        print("⚠ No data found, skipping analysis test")
        return False

    sample = exam_df.iloc[0]
    subject_id = sample["Subject_ID"]
    lecturer_id = sample["Lecturer_ID"]

    print(f"Using real IDs: subject={subject_id}, lecturer={lecturer_id}")

    cmd = [
        sys.executable,
        "scripts/analyze_class.py",
        "--model",
        model_path,
        "--subject-id",
        str(subject_id),
        "--lecturer-id",
        str(lecturer_id),
        "--exam-scores",
        "data/DiemTong.xlsx",
        "--conduct-scores",
        "data/diemrenluyen.xlsx",
        "--demographics",
        "data/nhankhau.xlsx",
        "--teaching-methods",
        "data/PPGDfull.xlsx",
        "--assessment-methods",
        "data/PPDGfull.xlsx",
        "--study-hours",
        "data/tuhoc.xlsx",
        "--output",
        "test_analyze_output.json",
    ]

    print(f"Running: {' '.join(cmd)}")
    print()

    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=300,  # 5 minutes timeout
        )

        print("STDOUT:")
        print(result.stdout)
        if result.stderr:
            print("\nSTDERR:")
            print(result.stderr)

        if result.returncode == 0:
            # Check if output file was created
            if Path("test_analyze_output.json").exists():
                with open("test_analyze_output.json", "r", encoding="utf-8") as f:
                    output = json.load(f)
                print(f"\n✓ analyze_class.py completed successfully")
                print(f"  Total students: {output.get('total_students', 'N/A')}")
                print(f"  Average predicted score: {output.get('average_predicted_score', 'N/A')}")
                print(f"  Number of common reasons: {len(output.get('common_reasons', []))}")
                return True
            else:
                print("\n⚠ analyze_class.py completed but output file not found")
                return False
        else:
            print(f"\n✗ analyze_class.py failed with exit code {result.returncode}")
            return False

    except subprocess.TimeoutExpired:
        print("\n✗ analyze_class.py timed out after 5 minutes")
        return False
    except Exception as e:
        print(f"\n✗ Error running analyze_class.py: {e}")
        return False


def test_help_commands():
    """Test --help for all scripts."""
    print("\n" + "=" * 80)
    print("Testing --help commands")
    print("=" * 80)

    scripts = ["train.py", "predict.py", "analyze_class.py"]
    results = {}

    for script in scripts:
        cmd = [sys.executable, f"scripts/{script}", "--help"]
        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=10,
            )
            if result.returncode == 0:
                print(f"✓ {script} --help works")
                results[script] = True
            else:
                print(f"✗ {script} --help failed")
                results[script] = False
        except Exception as e:
            print(f"✗ {script} --help error: {e}")
            results[script] = False

    return all(results.values())


def main():
    """Run all CLI script tests."""
    print("=" * 80)
    print("TESTING CLI SCRIPTS WITH REAL DATA")
    print("=" * 80)

    results = {}

    # Test help commands first
    results["help"] = test_help_commands()

    # Test train script
    results["train"] = test_train_script()

    # Test predict script (only if train succeeded)
    if results["train"]:
        results["predict"] = test_predict_script()
    else:
        print("\n⚠ Skipping predict test (train failed)")
        results["predict"] = False

    # Test analyze_class script (only if train succeeded)
    if results["train"]:
        results["analyze"] = test_analyze_class_script()
    else:
        print("\n⚠ Skipping analyze_class test (train failed)")
        results["analyze"] = False

    # Print summary
    print("\n" + "=" * 80)
    print("TEST SUMMARY")
    print("=" * 80)
    for test_name, passed in results.items():
        status = "✓ PASSED" if passed else "✗ FAILED"
        print(f"{test_name:20s}: {status}")
    print("=" * 80)

    if all(results.values()):
        print("\n✓ All CLI script tests passed!")
        return 0
    else:
        print("\n✗ Some CLI script tests failed")
        return 1


if __name__ == "__main__":
    # Add src to path
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).parent / "src"))

    sys.exit(main())

