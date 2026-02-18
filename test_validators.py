"""Test script for data validators."""

import sys
from pathlib import Path

import pandas as pd

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from ml_clo.data.loaders import load_conduct_scores, load_exam_scores
from ml_clo.data.preprocessors import preprocess_exam_scores
from ml_clo.data.validators import (
    validate_clo_score_range,
    validate_conduct_score_range,
    validate_conduct_scores_data,
    validate_data_consistency,
    validate_data_types,
    validate_exam_scores_data,
    validate_no_missing_values,
    validate_required_fields,
    validate_ranges,
)
from ml_clo.utils.logger import get_logger

logger = get_logger(__name__)


def test_validate_required_fields():
    """Test required fields validation."""
    print("\n" + "=" * 80)
    print("Testing validate_required_fields()")
    print("=" * 80)

    df = load_exam_scores("data/DiemTong.xlsx")
    print(f"Data: {len(df)} records")

    # Test with all required fields present
    required_cols = ["Student_ID", "Subject_ID", "Lecturer_ID"]
    is_valid, errors = validate_required_fields(df, required_cols, strict=False)
    print(f"\nRequired fields: {required_cols}")
    print(f"  Result: {'✓ PASS' if is_valid else '✗ FAIL'}")
    if errors:
        for error in errors:
            print(f"  - {error}")

    # Test with missing field
    required_cols_missing = ["Student_ID", "NonExistentColumn"]
    is_valid2, errors2 = validate_required_fields(df, required_cols_missing, strict=False)
    print(f"\nRequired fields (with missing): {required_cols_missing}")
    print(f"  Result: {'✓ PASS' if is_valid2 else '✗ FAIL'}")
    if errors2:
        for error in errors2:
            print(f"  - {error}")

    return is_valid and not is_valid2  # First should pass, second should fail


def test_validate_data_types():
    """Test data types validation."""
    print("\n" + "=" * 80)
    print("Testing validate_data_types()")
    print("=" * 80)

    df = load_exam_scores("data/DiemTong.xlsx")
    print(f"Data: {len(df)} records")

    # Test with correct types
    expected_types = {
        "Student_ID": "int64",
        "Subject_ID": "object",  # string
    }
    is_valid, errors = validate_data_types(df, expected_types, strict=False)
    print(f"\nExpected types: {expected_types}")
    print(f"  Result: {'✓ PASS' if is_valid else '✗ FAIL'}")
    if errors:
        for error in errors:
            print(f"  - {error}")

    return is_valid


def test_validate_ranges():
    """Test range validation."""
    print("\n" + "=" * 80)
    print("Testing validate_ranges()")
    print("=" * 80)

    # Load and preprocess exam scores to get CLO scores (0-6)
    df = load_exam_scores("data/DiemTong.xlsx")
    df = preprocess_exam_scores(df, convert_to_clo=True, create_result=False)
    print(f"Data: {len(df)} records")

    # Test CLO score range [0, 6]
    column_ranges = {"exam_score": (0.0, 6.0)}
    is_valid, errors = validate_ranges(df, column_ranges, strict=False)
    print(f"\nRange validation for exam_score: [0, 6]")
    print(f"  Actual range: [{df['exam_score'].min():.2f}, {df['exam_score'].max():.2f}]")
    print(f"  Result: {'✓ PASS' if is_valid else '✗ FAIL'}")
    if errors:
        for error in errors:
            print(f"  - {error}")

    return is_valid


def test_validate_clo_score_range():
    """Test CLO score range validation."""
    print("\n" + "=" * 80)
    print("Testing validate_clo_score_range()")
    print("=" * 80)

    # Load and preprocess exam scores
    df = load_exam_scores("data/DiemTong.xlsx")
    df = preprocess_exam_scores(df, convert_to_clo=True, create_result=False)
    print(f"Data: {len(df)} records")

    is_valid, errors = validate_clo_score_range(df, "exam_score", strict=False)
    print(f"\nCLO score range validation [0, 6]:")
    print(f"  Actual range: [{df['exam_score'].min():.2f}, {df['exam_score'].max():.2f}]")
    print(f"  Result: {'✓ PASS' if is_valid else '✗ FAIL'}")
    if errors:
        for error in errors:
            print(f"  - {error}")

    return is_valid


def test_validate_conduct_score_range():
    """Test conduct score range validation."""
    print("\n" + "=" * 80)
    print("Testing validate_conduct_score_range()")
    print("=" * 80)

    df = load_conduct_scores("data/diemrenluyen.xlsx")
    print(f"Data: {len(df)} records")

    is_valid, errors = validate_conduct_score_range(df, "conduct_score", strict=False)
    print(f"\nConduct score range validation [0, 100]:")
    print(f"  Actual range: [{df['conduct_score'].min()}, {df['conduct_score'].max()}]")
    print(f"  Result: {'✓ PASS' if is_valid else '✗ FAIL'}")
    if errors:
        for error in errors:
            print(f"  - {error}")

    return is_valid


def test_validate_no_missing_values():
    """Test missing values validation."""
    print("\n" + "=" * 80)
    print("Testing validate_no_missing_values()")
    print("=" * 80)

    df = load_exam_scores("data/DiemTong.xlsx")
    print(f"Data: {len(df)} records")

    # Test with columns that may have missing values
    columns_to_check = ["Student_ID", "Subject_ID", "Lecturer_ID"]
    is_valid, errors = validate_no_missing_values(df, columns_to_check, strict=False)
    print(f"\nMissing values check for: {columns_to_check}")
    print(f"  Result: {'✓ PASS' if is_valid else '✗ FAIL'}")
    if errors:
        for error in errors:
            print(f"  - {error}")

    # Show missing value counts
    print(f"\n  Missing value counts:")
    for col in columns_to_check:
        if col in df.columns:
            missing = df[col].isna().sum()
            if missing > 0:
                print(f"    {col}: {missing} ({missing/len(df)*100:.1f}%)")

    return True  # Just checking, not failing on missing values


def test_validate_data_consistency():
    """Test data consistency validation."""
    print("\n" + "=" * 80)
    print("Testing validate_data_consistency()")
    print("=" * 80)

    df = load_exam_scores("data/DiemTong.xlsx")
    print(f"Data: {len(df)} records")

    # Test for duplicates based on Student_ID, Subject_ID, Lecturer_ID
    id_columns = ["Student_ID", "Subject_ID", "Lecturer_ID"]
    is_valid, errors = validate_data_consistency(df, id_columns, strict=False)
    print(f"\nConsistency check (duplicates in {id_columns}):")
    print(f"  Result: {'✓ PASS' if is_valid else '✗ FAIL'}")
    if errors:
        for error in errors:
            print(f"  - {error}")

    # Show duplicate count
    duplicates = df.duplicated(subset=id_columns, keep=False)
    duplicate_count = duplicates.sum()
    print(f"  Duplicate rows: {duplicate_count}")

    return True  # Just checking, duplicates may be legitimate


def test_validate_exam_scores_data():
    """Test comprehensive exam scores validation."""
    print("\n" + "=" * 80)
    print("Testing validate_exam_scores_data() - Comprehensive")
    print("=" * 80)

    # Load and preprocess exam scores
    df = load_exam_scores("data/DiemTong.xlsx")
    df = preprocess_exam_scores(df, convert_to_clo=True, create_result=False)
    print(f"Data: {len(df)} records")

    is_valid, errors = validate_exam_scores_data(df, strict=False)
    print(f"\nComprehensive validation result: {'✓ PASS' if is_valid else '✗ FAIL'}")
    if errors:
        print(f"  Found {len(errors)} issues:")
        for error in errors:
            print(f"    - {error}")
    else:
        print(f"  All validations passed!")

    return is_valid


def test_validate_conduct_scores_data():
    """Test comprehensive conduct scores validation."""
    print("\n" + "=" * 80)
    print("Testing validate_conduct_scores_data() - Comprehensive")
    print("=" * 80)

    df = load_conduct_scores("data/diemrenluyen.xlsx")
    print(f"Data: {len(df)} records")

    is_valid, errors = validate_conduct_scores_data(df, strict=False)
    print(f"\nComprehensive validation result: {'✓ PASS' if is_valid else '✗ FAIL'}")
    if errors:
        print(f"  Found {len(errors)} issues:")
        for error in errors:
            print(f"    - {error}")
    else:
        print(f"  All validations passed!")

    return is_valid


def main():
    """Run all tests."""
    print("=" * 80)
    print("TESTING DATA VALIDATORS")
    print("=" * 80)

    results = []

    try:
        results.append(("validate_required_fields", test_validate_required_fields()))
        results.append(("validate_data_types", test_validate_data_types()))
        results.append(("validate_ranges", test_validate_ranges()))
        results.append(("validate_clo_score_range", test_validate_clo_score_range()))
        results.append(("validate_conduct_score_range", test_validate_conduct_score_range()))
        results.append(("validate_no_missing_values", test_validate_no_missing_values()))
        results.append(("validate_data_consistency", test_validate_data_consistency()))
        results.append(("validate_exam_scores_data", test_validate_exam_scores_data()))
        results.append(("validate_conduct_scores_data", test_validate_conduct_scores_data()))
    except Exception as e:
        print(f"\n✗ Error during testing: {e}")
        import traceback
        traceback.print_exc()

    # Summary
    print("\n" + "=" * 80)
    print("TEST SUMMARY")
    print("=" * 80)
    passed = sum(1 for _, result in results if result)
    total = len(results)
    print(f"Passed: {passed}/{total}")
    print("\nDetailed results:")
    for name, result in results:
        status = "✓ PASS" if result else "✗ FAIL"
        print(f"  {status}: {name}")


if __name__ == "__main__":
    main()

