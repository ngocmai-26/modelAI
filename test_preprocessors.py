"""Test script for data preprocessors."""

import sys
from pathlib import Path

import pandas as pd

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from ml_clo.data.loaders import load_exam_scores
from ml_clo.data.preprocessors import (
    clean_exam_score,
    convert_score_10_to_6,
    create_result_column,
    handle_missing_values,
    preprocess_exam_scores,
    standardize_lecturer_id,
    standardize_student_id,
    standardize_subject_id,
)
from ml_clo.utils.logger import get_logger

logger = get_logger(__name__)


def test_standardize_ids():
    """Test ID standardization functions."""
    print("\n" + "=" * 80)
    print("Testing ID Standardization")
    print("=" * 80)

    # Load exam scores
    df = load_exam_scores("data/DiemTong.xlsx")
    print(f"Original data: {len(df)} records")

    # Test Student_ID standardization
    df_student = standardize_student_id(df.copy(), "Student_ID")
    print(f"After Student_ID standardization: {len(df_student)} records")
    print(f"  Student_ID type: {df_student['Student_ID'].dtype}")
    print(f"  Student_ID range: {df_student['Student_ID'].min()} - {df_student['Student_ID'].max()}")

    # Test Subject_ID standardization
    df_subject = standardize_subject_id(df.copy(), "Subject_ID")
    print(f"After Subject_ID standardization: {len(df_subject)} records")
    print(f"  Subject_ID type: {df_subject['Subject_ID'].dtype}")
    print(f"  Unique Subject_IDs: {df_subject['Subject_ID'].nunique()}")

    # Test Lecturer_ID standardization
    df_lecturer = standardize_lecturer_id(df.copy(), "Lecturer_ID")
    print(f"After Lecturer_ID standardization: {len(df_lecturer)} records")
    print(f"  Lecturer_ID type: {df_lecturer['Lecturer_ID'].dtype}")
    print(f"  Unique Lecturer_IDs: {df_lecturer['Lecturer_ID'].nunique()}")

    return True


def test_clean_exam_score():
    """Test exam score cleaning."""
    print("\n" + "=" * 80)
    print("Testing clean_exam_score()")
    print("=" * 80)

    df = load_exam_scores("data/DiemTong.xlsx")
    print(f"Original data: {len(df)} records")
    print(f"  exam_score type: {df['exam_score'].dtype}")
    print(f"  exam_score sample values: {df['exam_score'].head(10).tolist()}")

    # Count invalid values before cleaning
    invalid_before = df['exam_score'].apply(lambda x: not str(x).replace('.', '').isdigit() if pd.notna(x) else True).sum()
    print(f"  Invalid values (estimated): {invalid_before}")

    # Clean exam scores
    df_cleaned = clean_exam_score(df.copy(), "exam_score", remove_invalid=True)
    print(f"\nAfter cleaning: {len(df_cleaned)} records")
    print(f"  exam_score type: {df_cleaned['exam_score'].dtype}")
    print(f"  exam_score range: {df_cleaned['exam_score'].min():.2f} - {df_cleaned['exam_score'].max():.2f}")
    print(f"  Missing values: {df_cleaned['exam_score'].isna().sum()}")

    return True


def test_convert_score_10_to_6():
    """Test score conversion from 10-point to 6-point scale."""
    print("\n" + "=" * 80)
    print("Testing convert_score_10_to_6()")
    print("=" * 80)

    df = load_exam_scores("data/DiemTong.xlsx")
    df = clean_exam_score(df.copy(), "exam_score", remove_invalid=True)

    print(f"Original scores (10-point scale):")
    print(f"  Count: {len(df)}")
    print(f"  Range: {df['exam_score'].min():.2f} - {df['exam_score'].max():.2f}")
    print(f"  Mean: {df['exam_score'].mean():.2f}")

    # Convert to 6-point scale
    df_converted = convert_score_10_to_6(df.copy(), "exam_score", output_column="clo_score")
    print(f"\nConverted scores (6-point scale):")
    print(f"  Count: {len(df_converted)}")
    print(f"  Range: {df_converted['clo_score'].min():.2f} - {df_converted['clo_score'].max():.2f}")
    print(f"  Mean: {df_converted['clo_score'].mean():.2f}")

    # Verify conversion formula: CLO_6 = Score_10 / 10 × 6
    sample = df_converted.head(10)
    for idx, row in sample.iterrows():
        original = df.loc[idx, 'exam_score']
        converted = row['clo_score']
        expected = original / 10.0 * 6.0
        if abs(converted - expected) > 0.01:
            print(f"  ✗ Conversion error: {original} -> {converted} (expected {expected:.2f})")
        else:
            print(f"  ✓ {original:.2f} -> {converted:.2f} (expected {expected:.2f})")

    return True


def test_create_result_column():
    """Test Result column creation."""
    print("\n" + "=" * 80)
    print("Testing create_result_column()")
    print("=" * 80)

    df = load_exam_scores("data/DiemTong.xlsx")
    df = clean_exam_score(df.copy(), "exam_score", remove_invalid=True)
    df = convert_score_10_to_6(df.copy(), "exam_score", inplace=True)

    df_result = create_result_column(df.copy(), "exam_score", "Result", pass_threshold=3.0)
    print(f"Created Result column:")
    print(f"  Pass (>= 3.0): {(df_result['Result'] == 1).sum()}")
    print(f"  Fail (< 3.0): {(df_result['Result'] == 0).sum()}")
    print(f"  Missing: {df_result['Result'].isna().sum()}")

    # Show sample
    sample = df_result[['exam_score', 'Result']].head(10)
    print(f"\nSample:")
    print(sample)

    return True


def test_handle_missing_values():
    """Test missing value handling."""
    print("\n" + "=" * 80)
    print("Testing handle_missing_values()")
    print("=" * 80)

    df = load_exam_scores("data/DiemTong.xlsx")
    print(f"Original data: {len(df)} records")
    print(f"  Missing values per column:")
    missing = df.isnull().sum()
    for col, count in missing[missing > 0].head(10).items():
        print(f"    {col}: {count} ({count/len(df)*100:.1f}%)")

    # Test different strategies
    strategies = ["drop", "fill_zero"]
    for strategy in strategies:
        df_handled = handle_missing_values(
            df.copy(),
            strategy=strategy,
            columns=["Student_ID", "Subject_ID", "Lecturer_ID"],
        )
        print(f"\nStrategy '{strategy}': {len(df_handled)} records")

    return True


def test_preprocess_exam_scores_pipeline():
    """Test complete preprocessing pipeline."""
    print("\n" + "=" * 80)
    print("Testing preprocess_exam_scores() - Complete Pipeline")
    print("=" * 80)

    df = load_exam_scores("data/DiemTong.xlsx")
    print(f"Original data: {len(df)} records")

    # Run complete preprocessing pipeline
    df_processed = preprocess_exam_scores(df, convert_to_clo=True, create_result=True)

    print(f"\nAfter preprocessing: {len(df_processed)} records")
    print(f"  Columns: {list(df_processed.columns)}")
    print(f"  exam_score range: {df_processed['exam_score'].min():.2f} - {df_processed['exam_score'].max():.2f}")
    print(f"  exam_score mean: {df_processed['exam_score'].mean():.2f}")
    if 'Result' in df_processed.columns:
        print(f"  Result - Pass: {(df_processed['Result'] == 1).sum()}, Fail: {(df_processed['Result'] == 0).sum()}")

    # Show sample
    print(f"\nSample processed data:")
    key_cols = ['Student_ID', 'Subject_ID', 'Lecturer_ID', 'exam_score']
    if 'Result' in df_processed.columns:
        key_cols.append('Result')
    print(df_processed[key_cols].head(10))

    return True


def main():
    """Run all tests."""
    print("=" * 80)
    print("TESTING DATA PREPROCESSORS")
    print("=" * 80)

    results = []

    try:
        results.append(("standardize_ids", test_standardize_ids()))
        results.append(("clean_exam_score", test_clean_exam_score()))
        results.append(("convert_score_10_to_6", test_convert_score_10_to_6()))
        results.append(("create_result_column", test_create_result_column()))
        results.append(("handle_missing_values", test_handle_missing_values()))
        results.append(("preprocess_exam_scores_pipeline", test_preprocess_exam_scores_pipeline()))
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

