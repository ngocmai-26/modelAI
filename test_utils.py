"""Test script for utilities modules."""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

import numpy as np
import pandas as pd

from ml_clo.utils.io_utils import (
    ensure_directory_exists,
    format_file_size,
    get_file_size,
    load_csv,
    load_json,
    save_csv,
    save_json,
)
from ml_clo.utils.math_utils import (
    calculate_percentage,
    clip_scores,
    convert_score_10_to_6,
    convert_score_6_to_10,
    normalize_to_percentage,
    round_to_precision,
    validate_score_range,
)


def test_math_utils():
    """Test math utilities."""
    print("=" * 80)
    print("Testing Math Utilities")
    print("=" * 80)

    # Test score conversion
    print("\n1. Testing score conversion:")
    score_10 = 10.0
    score_6 = convert_score_10_to_6(score_10)
    print(f"   {score_10} (10-point) -> {score_6} (6-point)")

    score_6_back = convert_score_6_to_10(score_6)
    print(f"   {score_6} (6-point) -> {score_6_back} (10-point)")

    # Test with array
    scores_10 = np.array([10.0, 5.0, 0.0])
    scores_6 = convert_score_10_to_6(scores_10)
    print(f"   Array {scores_10} -> {scores_6}")

    # Test with Series
    scores_series = pd.Series([10.0, 5.0, 0.0])
    scores_6_series = convert_score_10_to_6(scores_series)
    print(f"   Series {scores_series.tolist()} -> {scores_6_series.tolist()}")

    # Test score validation
    print("\n2. Testing score validation:")
    valid = validate_score_range(4.5, min_score=0.0, max_score=6.0)
    print(f"   validate_score_range(4.5): {valid}")

    invalid = validate_score_range(7.0, min_score=0.0, max_score=6.0)
    print(f"   validate_score_range(7.0): {invalid}")

    # Test score clipping
    print("\n3. Testing score clipping:")
    clipped = clip_scores(7.0, min_score=0.0, max_score=6.0)
    print(f"   clip_scores(7.0): {clipped}")

    clipped_neg = clip_scores(-1.0, min_score=0.0, max_score=6.0)
    print(f"   clip_scores(-1.0): {clipped_neg}")

    # Test percentage calculation
    print("\n4. Testing percentage calculation:")
    pct = calculate_percentage(3, 10)
    print(f"   calculate_percentage(3, 10): {pct}%")

    # Test normalization
    print("\n5. Testing normalization:")
    values = np.array([10, 20, 30])
    normalized = normalize_to_percentage(values)
    print(f"   normalize_to_percentage({values}): {normalized}")

    # Test rounding
    print("\n6. Testing rounding:")
    rounded = round_to_precision(3.14159, precision=2)
    print(f"   round_to_precision(3.14159, precision=2): {rounded}")

    print("\n✓ Math utilities tests passed!")
    return True


def test_io_utils():
    """Test I/O utilities."""
    print("\n" + "=" * 80)
    print("Testing I/O Utilities")
    print("=" * 80)

    # Test directory creation
    print("\n1. Testing directory creation:")
    test_path = ensure_directory_exists("test_output/test_file.txt")
    print(f"   Created directory: {test_path.parent}")
    print(f"   Directory exists: {test_path.parent.exists()}")

    # Test JSON save/load
    print("\n2. Testing JSON save/load:")
    test_data = {"key1": "value1", "key2": 123, "key3": [1, 2, 3]}
    json_path = "test_output/test_data.json"
    save_json(test_data, json_path)
    print(f"   Saved JSON to: {json_path}")

    loaded_data = load_json(json_path)
    print(f"   Loaded JSON: {loaded_data}")
    assert loaded_data == test_data, "JSON data mismatch"
    print("   ✓ JSON save/load works")

    # Test CSV save/load
    print("\n3. Testing CSV save/load:")
    test_df = pd.DataFrame({"col1": [1, 2, 3], "col2": ["a", "b", "c"]})
    csv_path = "test_output/test_data.csv"
    save_csv(test_df, csv_path, index=False)
    print(f"   Saved CSV to: {csv_path}")

    loaded_df = load_csv(csv_path)
    print(f"   Loaded CSV shape: {loaded_df.shape}")
    assert len(loaded_df) == len(test_df), "CSV row count mismatch"
    print("   ✓ CSV save/load works")

    # Test file size
    print("\n4. Testing file size utilities:")
    size = get_file_size(json_path)
    print(f"   File size: {size} bytes")
    formatted = format_file_size(size)
    print(f"   Formatted size: {formatted}")

    # Cleanup
    import shutil

    if Path("test_output").exists():
        shutil.rmtree("test_output")
        print("\n   Cleaned up test_output directory")

    print("\n✓ I/O utilities tests passed!")
    return True


def main():
    """Run all utility tests."""
    print("=" * 80)
    print("TESTING UTILITIES")
    print("=" * 80)

    results = {}

    results["math"] = test_math_utils()
    results["io"] = test_io_utils()

    print("\n" + "=" * 80)
    print("TEST SUMMARY")
    print("=" * 80)
    for test_name, passed in results.items():
        status = "✓ PASSED" if passed else "✗ FAILED"
        print(f"{test_name:20s}: {status}")
    print("=" * 80)

    if all(results.values()):
        print("\n✓ All utility tests passed!")
        return 0
    else:
        print("\n✗ Some utility tests failed")
        return 1


if __name__ == "__main__":
    sys.exit(main())

