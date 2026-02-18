"""Test script for feature engineering."""

import sys
from pathlib import Path

import pandas as pd

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from ml_clo.data.encoders import encode_assessment_methods, encode_teaching_methods
from ml_clo.data.loaders import (
    load_assessment_methods,
    load_conduct_scores,
    load_exam_scores,
    load_study_hours,
    load_teaching_methods,
)
from ml_clo.data.mergers import create_training_dataset
from ml_clo.data.preprocessors import preprocess_exam_scores
from ml_clo.features.feature_builder import (
    build_academic_history_features,
    build_all_features,
    build_conduct_features,
    build_study_hours_features,
)
from ml_clo.features.feature_groups import (
    get_all_existing_features,
    get_features_by_group,
    group_features_by_pedagogy,
)
from ml_clo.utils.logger import get_logger

logger = get_logger(__name__)


def test_feature_groups():
    """Test feature groups functionality."""
    print("\n" + "=" * 80)
    print("Testing Feature Groups")
    print("=" * 80)

    from ml_clo.config.feature_config import FEATURE_GROUPS, PEDAGOGICAL_GROUPS

    print(f"\nFeature Groups ({len(FEATURE_GROUPS)} groups):")
    for group_name, features in FEATURE_GROUPS.items():
        print(f"  {group_name}: {len(features)} features")

    print(f"\nPedagogical Groups ({len(PEDAGOGICAL_GROUPS)} groups):")
    for group_name, features in PEDAGOGICAL_GROUPS.items():
        print(f"  {group_name}: {len(features)} features")

    return True


def test_build_conduct_features():
    """Test building conduct features."""
    print("\n" + "=" * 80)
    print("Testing build_conduct_features()")
    print("=" * 80)

    # Load and preprocess exam scores
    exam_df = load_exam_scores("data/DiemTong.xlsx")
    exam_df = preprocess_exam_scores(exam_df, convert_to_clo=True, create_result=False)
    print(f"Exam scores: {len(exam_df)} records")

    # Load conduct scores
    conduct_df = load_conduct_scores("data/diemrenluyen.xlsx")
    print(f"Conduct scores: {len(conduct_df)} records")

    # Build conduct features
    df_with_features = build_conduct_features(exam_df, conduct_df)
    print(f"\nAfter building features: {len(df_with_features)} records")

    # Check features
    conduct_feat_cols = ["avg_conduct_score", "latest_conduct_score", "conduct_trend"]
    for col in conduct_feat_cols:
        if col in df_with_features.columns:
            non_null = df_with_features[col].notna().sum()
            print(f"  {col}: {non_null} non-null values")
            if non_null > 0:
                print(f"    Range: {df_with_features[col].min():.2f} - {df_with_features[col].max():.2f}")

    return True


def test_build_academic_history_features():
    """Test building academic history features."""
    print("\n" + "=" * 80)
    print("Testing build_academic_history_features()")
    print("=" * 80)

    # Load and preprocess exam scores
    exam_df = load_exam_scores("data/DiemTong.xlsx")
    exam_df = preprocess_exam_scores(exam_df, convert_to_clo=True, create_result=False)
    print(f"Exam scores: {len(exam_df)} records")

    # Use same data as history (in real scenario, this would be historical data)
    history_df = exam_df.copy()

    # Build academic history features
    df_with_features = build_academic_history_features(exam_df, history_df)
    print(f"\nAfter building features: {len(df_with_features)} records")

    # Check features
    academic_feat_cols = [
        "total_subjects",
        "passed_subjects",
        "pass_rate",
        "avg_exam_score",
        "recent_avg_score",
        "improvement_trend",
    ]
    for col in academic_feat_cols:
        if col in df_with_features.columns:
            non_null = df_with_features[col].notna().sum()
            print(f"  {col}: {non_null} non-null values")
            if non_null > 0:
                if col in ["total_subjects", "passed_subjects", "improvement_trend"]:
                    print(f"    Range: {int(df_with_features[col].min())} - {int(df_with_features[col].max())}")
                else:
                    print(f"    Range: {df_with_features[col].min():.2f} - {df_with_features[col].max():.2f}")

    return True


def test_build_study_hours_features():
    """Test building study hours features."""
    print("\n" + "=" * 80)
    print("Testing build_study_hours_features()")
    print("=" * 80)

    # Load and preprocess exam scores
    exam_df = load_exam_scores("data/DiemTong.xlsx")
    exam_df = preprocess_exam_scores(exam_df, convert_to_clo=True, create_result=False)
    print(f"Exam scores: {len(exam_df)} records")

    # Load study hours
    study_df = load_study_hours("data/tuhoc.xlsx")
    print(f"Study hours: {len(study_df)} records")

    # Build study hours features
    df_with_features = build_study_hours_features(exam_df, study_df)
    print(f"\nAfter building features: {len(df_with_features)} records")

    # Check features
    study_feat_cols = ["total_study_hours", "study_hours_this_year"]
    for col in study_feat_cols:
        if col in df_with_features.columns:
            non_null = df_with_features[col].notna().sum()
            print(f"  {col}: {non_null} non-null values")
            if non_null > 0:
                print(f"    Range: {df_with_features[col].min():.2f} - {df_with_features[col].max():.2f}")

    return True


def test_build_all_features():
    """Test building all features together."""
    print("\n" + "=" * 80)
    print("Testing build_all_features() - Complete Feature Engineering")
    print("=" * 80)

    # Load and preprocess exam scores
    exam_df = load_exam_scores("data/DiemTong.xlsx")
    exam_df = preprocess_exam_scores(exam_df, convert_to_clo=True, create_result=False)
    print(f"Exam scores: {len(exam_df)} records")

    # Load all data sources
    conduct_df = load_conduct_scores("data/diemrenluyen.xlsx")
    study_df = load_study_hours("data/tuhoc.xlsx")
    history_df = exam_df.copy()  # Use same data as history

    # Build all features
    df_with_features = build_all_features(
        exam_df,
        conduct_history_df=conduct_df,
        exam_history_df=history_df,
        study_hours_df=study_df,
    )
    print(f"\nAfter building all features: {len(df_with_features)} records, {len(df_with_features.columns)} columns")

    # Check feature groups
    print(f"\nFeature groups:")
    feature_groups = {
        "Conduct": ["avg_conduct_score", "latest_conduct_score", "conduct_trend"],
        "Academic": ["total_subjects", "passed_subjects", "pass_rate", "avg_exam_score", "recent_avg_score", "improvement_trend"],
        "Study": ["total_study_hours", "study_hours_this_year"],
    }

    for group_name, feat_cols in feature_groups.items():
        found = [col for col in feat_cols if col in df_with_features.columns]
        if found:
            print(f"  {group_name}: {len(found)}/{len(feat_cols)} features")
            for col in found:
                non_null = df_with_features[col].notna().sum()
                print(f"    - {col}: {non_null} non-null")

    return True


def test_feature_grouping():
    """Test feature grouping utilities."""
    print("\n" + "=" * 80)
    print("Testing Feature Grouping Utilities")
    print("=" * 80)

    # Create a sample training dataset
    exam_df = load_exam_scores("data/DiemTong.xlsx")
    exam_df = preprocess_exam_scores(exam_df, convert_to_clo=True, create_result=False)

    tm_df = encode_teaching_methods(load_teaching_methods("data/PPGDfull.xlsx"))
    em_df = encode_assessment_methods(load_assessment_methods("data/PPDGfull.xlsx"))

    from ml_clo.data.mergers import merge_all_data_sources

    training_df = merge_all_data_sources(
        exam_df=exam_df,
        teaching_methods_df=tm_df,
        assessment_methods_df=em_df,
    )

    # Test feature grouping
    all_features = get_all_existing_features(training_df)
    print(f"\nAll existing features: {len(all_features)} features")

    # Group by pedagogy
    grouped = group_features_by_pedagogy(all_features, df=training_df)
    print(f"\nGrouped by pedagogy ({len(grouped)} groups):")
    for group_name, features in grouped.items():
        print(f"  {group_name}: {len(features)} features")
        if len(features) <= 5:
            print(f"    {features}")
        else:
            print(f"    {features[:5]} ... ({len(features) - 5} more)")

    # Get features by group
    print(f"\nFeatures by group:")
    for group_name in ["A. Basic Information", "B. Demographics", "C. Conduct Scores"]:
        features = get_features_by_group(group_name, training_df)
        if features:
            print(f"  {group_name}: {len(features)} features")

    return True


def main():
    """Run all tests."""
    print("=" * 80)
    print("TESTING FEATURE ENGINEERING")
    print("=" * 80)

    results = []

    try:
        results.append(("feature_groups", test_feature_groups()))
        results.append(("build_conduct_features", test_build_conduct_features()))
        results.append(("build_academic_history_features", test_build_academic_history_features()))
        results.append(("build_study_hours_features", test_build_study_hours_features()))
        results.append(("build_all_features", test_build_all_features()))
        results.append(("feature_grouping", test_feature_grouping()))
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


