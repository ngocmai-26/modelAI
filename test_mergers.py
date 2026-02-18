"""Test script for data merging pipeline."""

import sys
from pathlib import Path

import pandas as pd

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from ml_clo.data.encoders import encode_teaching_methods, encode_assessment_methods
from ml_clo.data.loaders import (
    load_all_data_files,
    load_assessment_methods,
    load_attendance,
    load_conduct_scores,
    load_demographics,
    load_exam_scores,
    load_study_hours,
    load_teaching_methods,
)
from ml_clo.data.mergers import (
    create_training_dataset,
    merge_all_data_sources,
    merge_assessment_methods,
    merge_attendance,
    merge_demographics,
    merge_exam_and_conduct_scores,
    merge_study_hours,
    merge_teaching_methods,
)
from ml_clo.data.preprocessors import preprocess_exam_scores
from ml_clo.utils.logger import get_logger

logger = get_logger(__name__)


def test_merge_exam_and_conduct_scores():
    """Test merging exam and conduct scores."""
    print("\n" + "=" * 80)
    print("Testing merge_exam_and_conduct_scores()")
    print("=" * 80)

    # Load and preprocess exam scores
    exam_df = load_exam_scores("data/DiemTong.xlsx")
    exam_df = preprocess_exam_scores(exam_df, convert_to_clo=True, create_result=False)
    print(f"Exam scores: {len(exam_df)} records")

    # Load conduct scores
    conduct_df = load_conduct_scores("data/diemrenluyen.xlsx")
    print(f"Conduct scores: {len(conduct_df)} records")

    # Merge
    merged = merge_exam_and_conduct_scores(exam_df, conduct_df)
    print(f"\nAfter merging: {len(merged)} records")
    print(f"  Records with conduct_score: {merged['conduct_score'].notna().sum()}")
    print(f"  Conduct score range: {merged['conduct_score'].min():.2f} - {merged['conduct_score'].max():.2f}")

    return True


def test_merge_demographics():
    """Test merging demographics."""
    print("\n" + "=" * 80)
    print("Testing merge_demographics()")
    print("=" * 80)

    # Load and preprocess exam scores
    exam_df = load_exam_scores("data/DiemTong.xlsx")
    exam_df = preprocess_exam_scores(exam_df, convert_to_clo=True, create_result=False)
    print(f"Exam scores: {len(exam_df)} records")

    # Load demographics
    demo_df = load_demographics("data/nhankhau.xlsx")
    print(f"Demographics: {len(demo_df)} records")

    # Merge
    merged = merge_demographics(exam_df, demo_df)
    print(f"\nAfter merging: {len(merged)} records")
    if "Gender" in merged.columns:
        print(f"  Records with Gender: {merged['Gender'].notna().sum()}")
    if "place_of_birth" in merged.columns:
        print(f"  Records with place_of_birth: {merged['place_of_birth'].notna().sum()}")

    return True


def test_merge_teaching_methods():
    """Test merging teaching methods."""
    print("\n" + "=" * 80)
    print("Testing merge_teaching_methods()")
    print("=" * 80)

    # Load and preprocess exam scores
    exam_df = load_exam_scores("data/DiemTong.xlsx")
    exam_df = preprocess_exam_scores(exam_df, convert_to_clo=True, create_result=False)
    print(f"Exam scores: {len(exam_df)} records")

    # Load and encode teaching methods
    tm_df = load_teaching_methods("data/PPGDfull.xlsx")
    tm_df = encode_teaching_methods(tm_df)
    print(f"Teaching methods: {len(tm_df)} records")

    # Merge
    merged = merge_teaching_methods(exam_df, tm_df)
    print(f"\nAfter merging: {len(merged)} records")
    tm_cols = [col for col in merged.columns if "TM" in col]
    print(f"  Teaching method columns: {len(tm_cols)}")
    if tm_cols:
        print(f"  Sample TM column ({tm_cols[0]}): {merged[tm_cols[0]].notna().sum()} non-null values")

    return True


def test_merge_assessment_methods():
    """Test merging assessment methods."""
    print("\n" + "=" * 80)
    print("Testing merge_assessment_methods()")
    print("=" * 80)

    # Load and preprocess exam scores
    exam_df = load_exam_scores("data/DiemTong.xlsx")
    exam_df = preprocess_exam_scores(exam_df, convert_to_clo=True, create_result=False)
    print(f"Exam scores: {len(exam_df)} records")

    # Load and encode assessment methods
    em_df = load_assessment_methods("data/PPDGfull.xlsx")
    em_df = encode_assessment_methods(em_df)
    print(f"Assessment methods: {len(em_df)} records")

    # Merge
    merged = merge_assessment_methods(exam_df, em_df)
    print(f"\nAfter merging: {len(merged)} records")
    em_cols = [col for col in merged.columns if "EM" in col]
    print(f"  Assessment method columns: {len(em_cols)}")
    if em_cols:
        print(f"  Sample EM column ({em_cols[0]}): {merged[em_cols[0]].notna().sum()} non-null values")

    return True


def test_merge_study_hours():
    """Test merging study hours."""
    print("\n" + "=" * 80)
    print("Testing merge_study_hours()")
    print("=" * 80)

    # Load and preprocess exam scores
    exam_df = load_exam_scores("data/DiemTong.xlsx")
    exam_df = preprocess_exam_scores(exam_df, convert_to_clo=True, create_result=False)
    print(f"Exam scores: {len(exam_df)} records")

    # Load study hours
    study_df = load_study_hours("data/tuhoc.xlsx")
    print(f"Study hours: {len(study_df)} records")

    # Merge
    merged = merge_study_hours(exam_df, study_df)
    print(f"\nAfter merging: {len(merged)} records")
    if "total_study_hours" in merged.columns:
        print(f"  Records with total_study_hours: {merged['total_study_hours'].notna().sum()}")
        print(f"  Study hours range: {merged['total_study_hours'].min():.2f} - {merged['total_study_hours'].max():.2f}")

    return True


def test_merge_attendance():
    """Test merging attendance."""
    print("\n" + "=" * 80)
    print("Testing merge_attendance()")
    print("=" * 80)

    # Load and preprocess exam scores
    exam_df = load_exam_scores("data/DiemTong.xlsx")
    exam_df = preprocess_exam_scores(exam_df, convert_to_clo=True, create_result=False)
    print(f"Exam scores: {len(exam_df)} records")

    # Load attendance
    attendance_df = load_attendance("data/Dữ liệu điểm danh Khoa FIRA.xlsx")
    print(f"Attendance: {len(attendance_df)} records")

    # Merge
    merged = merge_attendance(exam_df, attendance_df)
    print(f"\nAfter merging: {len(merged)} records")
    if "attendance_rate" in merged.columns:
        print(f"  Records with attendance_rate: {merged['attendance_rate'].notna().sum()}")
        print(f"  Attendance rate range: {merged['attendance_rate'].min():.2f} - {merged['attendance_rate'].max():.2f}")

    return True


def test_merge_all_data_sources():
    """Test merging all data sources."""
    print("\n" + "=" * 80)
    print("Testing merge_all_data_sources() - Complete Pipeline")
    print("=" * 80)

    # Load all data
    print("Loading all data sources...")
    exam_df = load_exam_scores("data/DiemTong.xlsx")
    exam_df = preprocess_exam_scores(exam_df, convert_to_clo=True, create_result=False)
    print(f"  Exam scores: {len(exam_df)} records")

    conduct_df = load_conduct_scores("data/diemrenluyen.xlsx")
    print(f"  Conduct scores: {len(conduct_df)} records")

    demo_df = load_demographics("data/nhankhau.xlsx")
    print(f"  Demographics: {len(demo_df)} records")

    tm_df = load_teaching_methods("data/PPGDfull.xlsx")
    tm_df = encode_teaching_methods(tm_df)
    print(f"  Teaching methods: {len(tm_df)} records")

    em_df = load_assessment_methods("data/PPDGfull.xlsx")
    em_df = encode_assessment_methods(em_df)
    print(f"  Assessment methods: {len(em_df)} records")

    study_df = load_study_hours("data/tuhoc.xlsx")
    print(f"  Study hours: {len(study_df)} records")

    attendance_df = load_attendance("data/Dữ liệu điểm danh Khoa FIRA.xlsx")
    print(f"  Attendance: {len(attendance_df)} records")

    # Merge all
    print("\nMerging all data sources...")
    merged = merge_all_data_sources(
        exam_df=exam_df,
        conduct_df=conduct_df,
        demographics_df=demo_df,
        teaching_methods_df=tm_df,
        assessment_methods_df=em_df,
        study_hours_df=study_df,
        attendance_df=attendance_df,
    )

    print(f"\nFinal merged dataset:")
    print(f"  Records: {len(merged)}")
    print(f"  Columns: {len(merged.columns)}")
    print(f"  Key columns present:")
    key_cols = ["Student_ID", "Subject_ID", "exam_score", "conduct_score", "total_study_hours", "attendance_rate"]
    for col in key_cols:
        if col in merged.columns:
            non_null = merged[col].notna().sum()
            print(f"    ✓ {col}: {non_null} non-null values")
        else:
            print(f"    ✗ {col}: NOT FOUND")

    return True


def test_create_training_dataset():
    """Test creating final training dataset."""
    print("\n" + "=" * 80)
    print("Testing create_training_dataset() - Final Training Dataset")
    print("=" * 80)

    # Load all data
    print("Loading and preprocessing all data sources...")
    exam_df = load_exam_scores("data/DiemTong.xlsx")
    exam_df = preprocess_exam_scores(exam_df, convert_to_clo=True, create_result=False)

    conduct_df = load_conduct_scores("data/diemrenluyen.xlsx")
    demo_df = load_demographics("data/nhankhau.xlsx")
    tm_df = encode_teaching_methods(load_teaching_methods("data/PPGDfull.xlsx"))
    em_df = encode_assessment_methods(load_assessment_methods("data/PPDGfull.xlsx"))
    study_df = load_study_hours("data/tuhoc.xlsx")
    attendance_df = load_attendance("data/Dữ liệu điểm danh Khoa FIRA.xlsx")

    # Create training dataset
    print("\nCreating final training dataset...")
    training_df = create_training_dataset(
        exam_df=exam_df,
        conduct_df=conduct_df,
        demographics_df=demo_df,
        teaching_methods_df=tm_df,
        assessment_methods_df=em_df,
        study_hours_df=study_df,
        attendance_df=attendance_df,
        target_column="exam_score",
        drop_missing_target=True,
    )

    print(f"\nFinal training dataset:")
    print(f"  Records: {len(training_df)}")
    print(f"  Columns: {len(training_df.columns)}")
    print(f"  Target (exam_score):")
    print(f"    Range: {training_df['exam_score'].min():.2f} - {training_df['exam_score'].max():.2f}")
    print(f"    Mean: {training_df['exam_score'].mean():.2f}")
    print(f"    Missing: {training_df['exam_score'].isna().sum()}")

    # Show feature groups
    print(f"\n  Feature groups:")
    feature_groups = {
        "Basic": ["Student_ID", "Subject_ID", "Lecturer_ID"],
        "Conduct": ["conduct_score"],
        "Demographics": ["Gender", "place_of_birth", "Ethnicity", "Religion"],
        "Study": ["total_study_hours"],
        "Attendance": ["attendance_rate"],
        "Teaching Methods": [col for col in training_df.columns if "TM" in col],
        "Assessment Methods": [col for col in training_df.columns if "EM" in col],
    }

    for group_name, cols in feature_groups.items():
        found_cols = [col for col in cols if col in training_df.columns]
        if found_cols:
            print(f"    {group_name}: {len(found_cols)} features")

    # Show sample
    print(f"\n  Sample data (first 5 rows, key columns):")
    key_cols = ["Student_ID", "Subject_ID", "exam_score", "conduct_score", "total_study_hours"]
    key_cols = [col for col in key_cols if col in training_df.columns]
    print(training_df[key_cols].head())

    return True


def main():
    """Run all tests."""
    print("=" * 80)
    print("TESTING DATA MERGING PIPELINE")
    print("=" * 80)

    results = []

    try:
        results.append(("merge_exam_and_conduct_scores", test_merge_exam_and_conduct_scores()))
        results.append(("merge_demographics", test_merge_demographics()))
        results.append(("merge_teaching_methods", test_merge_teaching_methods()))
        results.append(("merge_assessment_methods", test_merge_assessment_methods()))
        results.append(("merge_study_hours", test_merge_study_hours()))
        results.append(("merge_attendance", test_merge_attendance()))
        results.append(("merge_all_data_sources", test_merge_all_data_sources()))
        results.append(("create_training_dataset", test_create_training_dataset()))
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

