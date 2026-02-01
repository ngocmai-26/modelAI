"""Test script for data encoders."""

import sys
from pathlib import Path

import pandas as pd

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from ml_clo.data.encoders import (
    encode_assessment_methods,
    encode_birth_place_to_region,
    encode_ethnicity,
    encode_gender,
    encode_lecturer_id,
    encode_religion,
    encode_student_id,
    encode_subject_id,
    encode_teaching_methods,
)
from ml_clo.data.loaders import (
    load_assessment_methods,
    load_demographics,
    load_exam_scores,
    load_teaching_methods,
)
from ml_clo.utils.logger import get_logger

logger = get_logger(__name__)


def test_encode_teaching_methods():
    """Test teaching methods encoding."""
    print("\n" + "=" * 80)
    print("Testing encode_teaching_methods()")
    print("=" * 80)

    df = load_teaching_methods("data/PPGDfull.xlsx")
    print(f"Original data: {len(df)} records")
    print(f"  Columns: {list(df.columns)[:10]}...")

    # Find TM columns
    tm_columns = [col for col in df.columns if "TM" in col or col.startswith("TM")]
    print(f"  Found {len(tm_columns)} teaching method columns: {tm_columns[:5]}...")

    # Show sample before encoding
    if tm_columns:
        print(f"\n  Sample before encoding:")
        print(df[tm_columns[:3]].head())

    # Encode
    df_encoded = encode_teaching_methods(df.copy())
    print(f"\nAfter encoding: {len(df_encoded)} records")

    # Show sample after encoding
    if tm_columns:
        print(f"\n  Sample after encoding:")
        print(df_encoded[tm_columns[:3]].head())
        print(f"  Value counts (should be 0 or 1):")
        for col in tm_columns[:3]:
            print(f"    {col}: {df_encoded[col].value_counts().to_dict()}")

    return True


def test_encode_assessment_methods():
    """Test assessment methods encoding."""
    print("\n" + "=" * 80)
    print("Testing encode_assessment_methods()")
    print("=" * 80)

    df = load_assessment_methods("data/PPDGfull.xlsx")
    print(f"Original data: {len(df)} records")
    print(f"  Columns: {list(df.columns)[:10]}...")

    # Find EM columns
    em_columns = [col for col in df.columns if "EM" in col or col.startswith("EM")]
    print(f"  Found {len(em_columns)} assessment method columns: {em_columns[:5]}...")

    # Show sample before encoding
    if em_columns:
        print(f"\n  Sample before encoding:")
        print(df[em_columns[:3]].head())

    # Encode
    df_encoded = encode_assessment_methods(df.copy())
    print(f"\nAfter encoding: {len(df_encoded)} records")

    # Show sample after encoding
    if em_columns:
        print(f"\n  Sample after encoding:")
        print(df_encoded[em_columns[:3]].head())
        print(f"  Value counts (should be 0 or 1):")
        for col in em_columns[:3]:
            print(f"    {col}: {df_encoded[col].value_counts().to_dict()}")

    return True


def test_encode_gender():
    """Test gender encoding."""
    print("\n" + "=" * 80)
    print("Testing encode_gender()")
    print("=" * 80)

    df = load_demographics("data/nhankhau.xlsx")
    print(f"Original data: {len(df)} records")

    if "Gender" not in df.columns:
        print("  ✗ Gender column not found")
        return False

    print(f"  Gender before encoding:")
    print(f"    Type: {df['Gender'].dtype}")
    print(f"    Value counts: {df['Gender'].value_counts().to_dict()}")

    # Encode
    df_encoded = encode_gender(df.copy(), "Gender")
    print(f"\nAfter encoding:")
    print(f"    Type: {df_encoded['Gender'].dtype}")
    print(f"    Value counts: {df_encoded['Gender'].value_counts().to_dict()}")
    print(f"    Range: {df_encoded['Gender'].min()} - {df_encoded['Gender'].max()}")

    return True


def test_encode_birth_place_to_region():
    """Test birth place to region encoding."""
    print("\n" + "=" * 80)
    print("Testing encode_birth_place_to_region()")
    print("=" * 80)

    df = load_demographics("data/nhankhau.xlsx")
    print(f"Original data: {len(df)} records")

    # Try different possible column names
    birth_place_col = None
    for col in ["place_of_birth", "Nơi sinh", "Nơi sinh theo giấy khai sinh"]:
        if col in df.columns:
            birth_place_col = col
            break

    if birth_place_col is None:
        print(f"  ✗ Birth place column not found. Available columns: {list(df.columns)[:10]}...")
        return False

    print(f"  Using column: {birth_place_col}")
    print(f"  Unique birth places: {df[birth_place_col].nunique()}")
    print(f"  Sample values: {df[birth_place_col].head(10).tolist()}")

    # Encode
    df_encoded = encode_birth_place_to_region(df.copy(), birth_place_col, "birth_place_region")
    print(f"\nAfter encoding:")
    print(f"    birth_place_region type: {df_encoded['birth_place_region'].dtype}")
    print(f"    Value counts: {df_encoded['birth_place_region'].value_counts().to_dict()}")
    print(f"    Range: {df_encoded['birth_place_region'].min()} - {df_encoded['birth_place_region'].max()}")

    return True


def test_encode_ethnicity():
    """Test ethnicity encoding."""
    print("\n" + "=" * 80)
    print("Testing encode_ethnicity()")
    print("=" * 80)

    df = load_demographics("data/nhankhau.xlsx")
    print(f"Original data: {len(df)} records")

    # Try different possible column names
    ethnicity_col = None
    for col in ["Ethnicity", "Dân tộc", "ethnicity"]:
        if col in df.columns:
            ethnicity_col = col
            break

    if ethnicity_col is None:
        print(f"  ✗ Ethnicity column not found. Available columns: {list(df.columns)[:10]}...")
        return False

    print(f"  Using column: {ethnicity_col}")
    print(f"  Unique ethnicities: {df[ethnicity_col].nunique()}")
    print(f"  Sample values: {df[ethnicity_col].head(10).tolist()}")

    # Encode
    df_encoded = encode_ethnicity(df.copy(), ethnicity_col, "ethnicity_encoded")
    print(f"\nAfter encoding:")
    print(f"    ethnicity_encoded type: {df_encoded['ethnicity_encoded'].dtype}")
    print(f"    Value counts (first 10): {df_encoded['ethnicity_encoded'].value_counts().head(10).to_dict()}")
    print(f"    Range: {df_encoded['ethnicity_encoded'].min()} - {df_encoded['ethnicity_encoded'].max()}")

    return True


def test_encode_religion():
    """Test religion encoding."""
    print("\n" + "=" * 80)
    print("Testing encode_religion()")
    print("=" * 80)

    df = load_demographics("data/nhankhau.xlsx")
    print(f"Original data: {len(df)} records")

    # Try different possible column names
    religion_col = None
    for col in ["Religion", "Tôn giáo", "religion"]:
        if col in df.columns:
            religion_col = col
            break

    if religion_col is None:
        print(f"  ✗ Religion column not found. Available columns: {list(df.columns)[:10]}...")
        return False

    print(f"  Using column: {religion_col}")
    print(f"  Unique religions: {df[religion_col].nunique()}")
    print(f"  Sample values: {df[religion_col].head(10).tolist()}")

    # Encode
    df_encoded = encode_religion(df.copy(), religion_col, "religion_encoded")
    print(f"\nAfter encoding:")
    print(f"    religion_encoded type: {df_encoded['religion_encoded'].dtype}")
    print(f"    Value counts: {df_encoded['religion_encoded'].value_counts().to_dict()}")
    print(f"    Range: {df_encoded['religion_encoded'].min()} - {df_encoded['religion_encoded'].max()}")

    return True


def test_encode_ids():
    """Test ID encoding functions."""
    print("\n" + "=" * 80)
    print("Testing ID Encoding (Student_ID, Lecturer_ID, Subject_ID)")
    print("=" * 80)

    df = load_exam_scores("data/DiemTong.xlsx")
    print(f"Original data: {len(df)} records")

    # Test Student_ID encoding
    print("\n1. Student_ID encoding:")
    print(f"   Unique Student_IDs: {df['Student_ID'].nunique()}")
    df_student = encode_student_id(df.copy(), "Student_ID", "student_id_encoded", method="label")
    print(f"   After encoding: {df_student['student_id_encoded'].nunique()} unique codes")
    print(f"   Range: {df_student['student_id_encoded'].min()} - {df_student['student_id_encoded'].max()}")

    # Test Lecturer_ID encoding
    print("\n2. Lecturer_ID encoding:")
    df_lecturer = df.copy()
    df_lecturer = df_lecturer[df_lecturer['Lecturer_ID'].notna()]  # Remove NaN
    print(f"   Unique Lecturer_IDs: {df_lecturer['Lecturer_ID'].nunique()}")
    df_lecturer = encode_lecturer_id(df_lecturer, "Lecturer_ID", "lecturer_id_encoded", method="label")
    print(f"   After encoding: {df_lecturer['lecturer_id_encoded'].nunique()} unique codes")
    print(f"   Range: {df_lecturer['lecturer_id_encoded'].min()} - {df_lecturer['lecturer_id_encoded'].max()}")

    # Test Subject_ID encoding
    print("\n3. Subject_ID encoding:")
    print(f"   Unique Subject_IDs: {df['Subject_ID'].nunique()}")
    df_subject = encode_subject_id(df.copy(), "Subject_ID", "subject_id_encoded", method="label")
    print(f"   After encoding: {df_subject['subject_id_encoded'].nunique()} unique codes")
    print(f"   Range: {df_subject['subject_id_encoded'].min()} - {df_subject['subject_id_encoded'].max()}")

    return True


def main():
    """Run all tests."""
    print("=" * 80)
    print("TESTING DATA ENCODERS")
    print("=" * 80)

    results = []

    try:
        results.append(("encode_teaching_methods", test_encode_teaching_methods()))
        results.append(("encode_assessment_methods", test_encode_assessment_methods()))
        results.append(("encode_gender", test_encode_gender()))
        results.append(("encode_birth_place_to_region", test_encode_birth_place_to_region()))
        results.append(("encode_ethnicity", test_encode_ethnicity()))
        results.append(("encode_religion", test_encode_religion()))
        results.append(("encode_ids", test_encode_ids()))
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


