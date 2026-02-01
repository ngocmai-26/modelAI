"""Test script for data loaders with real data files."""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

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
from ml_clo.utils.logger import get_logger

logger = get_logger(__name__)


def test_load_conduct_scores():
    """Test loading conduct scores."""
    print("\n" + "=" * 80)
    print("Testing load_conduct_scores()")
    print("=" * 80)

    file_path = "data/diemrenluyen.xlsx"
    try:
        df = load_conduct_scores(file_path)
        print(f"✓ Successfully loaded {len(df)} records")
        print(f"  Columns: {list(df.columns)}")
        print(f"  Shape: {df.shape}")
        print(f"  Data types:\n{df.dtypes}")
        print(f"  First few rows:\n{df.head()}")
        print(f"  Missing values:\n{df.isnull().sum()}")
        return True
    except Exception as e:
        print(f"✗ Error: {e}")
        return False


def test_load_exam_scores():
    """Test loading exam scores."""
    print("\n" + "=" * 80)
    print("Testing load_exam_scores()")
    print("=" * 80)

    file_path = "data/DiemTong.xlsx"
    try:
        df = load_exam_scores(file_path)
        print(f"✓ Successfully loaded {len(df)} records")
        print(f"  Columns ({len(df.columns)}): {list(df.columns)[:10]}...")  # Show first 10
        print(f"  Shape: {df.shape}")
        print(f"  Key columns check:")
        for col in ["Student_ID", "Subject_ID", "Lecturer_ID", "exam_score", "year"]:
            if col in df.columns:
                print(f"    ✓ {col}: {df[col].dtype}")
            else:
                print(f"    ✗ {col}: NOT FOUND")
        print(f"  First few rows (key columns):")
        key_cols = [c for c in ["Student_ID", "Subject_ID", "Lecturer_ID", "exam_score", "year"] if c in df.columns]
        if key_cols:
            print(df[key_cols].head())
        return True
    except Exception as e:
        print(f"✗ Error: {e}")
        return False


def test_load_demographics():
    """Test loading demographics."""
    print("\n" + "=" * 80)
    print("Testing load_demographics()")
    print("=" * 80)

    file_path = "data/nhankhau.xlsx"
    try:
        df = load_demographics(file_path)
        print(f"✓ Successfully loaded {len(df)} records")
        print(f"  Columns ({len(df.columns)}): {list(df.columns)[:15]}...")  # Show first 15
        print(f"  Shape: {df.shape}")
        print(f"  Key columns check:")
        for col in ["Student_ID", "Gender", "place_of_birth"]:
            if col in df.columns:
                print(f"    ✓ {col}: {df[col].dtype}")
            else:
                print(f"    ✗ {col}: NOT FOUND")
        return True
    except Exception as e:
        print(f"✗ Error: {e}")
        return False


def test_load_teaching_methods():
    """Test loading teaching methods."""
    print("\n" + "=" * 80)
    print("Testing load_teaching_methods()")
    print("=" * 80)

    # Try both files
    for file_path in ["data/PPGDfull.xlsx", "data/PPGD.xlsx"]:
        if Path(file_path).exists():
            try:
                df = load_teaching_methods(file_path)
                print(f"✓ Successfully loaded from {file_path}")
                print(f"  Records: {len(df)}")
                print(f"  Columns: {list(df.columns)}")
                print(f"  Shape: {df.shape}")
                print(f"  First few rows:\n{df.head()}")
                return True
            except Exception as e:
                print(f"✗ Error loading {file_path}: {e}")
    print("✗ No teaching methods file found")
    return False


def test_load_assessment_methods():
    """Test loading assessment methods."""
    print("\n" + "=" * 80)
    print("Testing load_assessment_methods()")
    print("=" * 80)

    # Try both files
    for file_path in ["data/PPDGfull.xlsx", "data/PPDG.xlsx"]:
        if Path(file_path).exists():
            try:
                df = load_assessment_methods(file_path)
                print(f"✓ Successfully loaded from {file_path}")
                print(f"  Records: {len(df)}")
                print(f"  Columns: {list(df.columns)}")
                print(f"  Shape: {df.shape}")
                print(f"  First few rows:\n{df.head()}")
                return True
            except Exception as e:
                print(f"✗ Error loading {file_path}: {e}")
    print("✗ No assessment methods file found")
    return False


def test_load_study_hours():
    """Test loading study hours."""
    print("\n" + "=" * 80)
    print("Testing load_study_hours()")
    print("=" * 80)

    file_path = "data/tuhoc.xlsx"
    try:
        df = load_study_hours(file_path)
        print(f"✓ Successfully loaded {len(df)} records")
        print(f"  Columns: {list(df.columns)}")
        print(f"  Shape: {df.shape}")
        print(f"  Key columns check:")
        for col in ["Student_ID", "year", "semester", "accumulated_study_hours"]:
            if col in df.columns:
                print(f"    ✓ {col}: {df[col].dtype}")
                if col == "accumulated_study_hours":
                    print(f"      Range: {df[col].min():.2f} - {df[col].max():.2f}")
            else:
                print(f"    ✗ {col}: NOT FOUND")
        print(f"  First few rows:\n{df.head()}")
        return True
    except Exception as e:
        print(f"✗ Error: {e}")
        return False


def test_load_attendance():
    """Test loading attendance."""
    print("\n" + "=" * 80)
    print("Testing load_attendance()")
    print("=" * 80)

    file_path = "data/Dữ liệu điểm danh Khoa FIRA.xlsx"
    try:
        df = load_attendance(file_path)
        print(f"✓ Successfully loaded {len(df)} records")
        print(f"  Columns: {list(df.columns)}")
        print(f"  Shape: {df.shape}")
        print(f"  Key columns check:")
        for col in ["MSSV", "Mã môn học", "Mã giảng viên", "Điểm danh"]:
            if col in df.columns:
                print(f"    ✓ {col}: {df[col].dtype}")
                if col == "Điểm danh":
                    print(f"      Values: {df[col].value_counts().to_dict()}")
            else:
                print(f"    ✗ {col}: NOT FOUND")
        print(f"  First few rows:\n{df.head()}")
        return True
    except Exception as e:
        print(f"✗ Error: {e}")
        return False


def test_load_all_data_files():
    """Test loading all data files at once."""
    print("\n" + "=" * 80)
    print("Testing load_all_data_files()")
    print("=" * 80)

    data_dir = "data"
    try:
        all_data = load_all_data_files(data_dir)
        print(f"✓ Successfully loaded {len(all_data)} data files")
        print("\nSummary:")
        for data_type, df in all_data.items():
            print(f"  {data_type}: {len(df)} records, {len(df.columns)} columns")
        return True
    except Exception as e:
        print(f"✗ Error: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Run all tests."""
    print("=" * 80)
    print("TESTING DATA LOADERS WITH REAL DATA")
    print("=" * 80)

    results = []

    # Test individual loaders
    results.append(("load_conduct_scores", test_load_conduct_scores()))
    results.append(("load_exam_scores", test_load_exam_scores()))
    results.append(("load_demographics", test_load_demographics()))
    results.append(("load_teaching_methods", test_load_teaching_methods()))
    results.append(("load_assessment_methods", test_load_assessment_methods()))
    results.append(("load_study_hours", test_load_study_hours()))
    results.append(("load_attendance", test_load_attendance()))

    # Test batch loader
    results.append(("load_all_data_files", test_load_all_data_files()))

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

