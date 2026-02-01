#!/usr/bin/env python3
"""Simple test for loaders - checks imports and basic functionality."""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

try:
    print("Testing imports...")
    from ml_clo.data.loaders import load_conduct_scores
    from ml_clo.utils.exceptions import DataLoadError
    from ml_clo.utils.logger import get_logger
    print("✓ All imports successful")
except ImportError as e:
    print(f"✗ Import error: {e}")
    sys.exit(1)

# Test if we can at least check file existence
print("\nChecking data files...")
data_dir = Path("data")
if not data_dir.exists():
    print(f"✗ Data directory not found: {data_dir}")
    sys.exit(1)

files_to_check = {
    "diemrenluyen.xlsx": "Conduct scores",
    "DiemTong.xlsx": "Exam scores",
    "nhankhau.xlsx": "Demographics",
    "PPGDfull.xlsx": "Teaching methods",
    "PPDGfull.xlsx": "Assessment methods",
    "tuhoc.xlsx": "Study hours",
    "Dữ liệu điểm danh Khoa FIRA.xlsx": "Attendance",
}

for filename, description in files_to_check.items():
    filepath = data_dir / filename
    if filepath.exists():
        print(f"✓ {description}: {filename} exists ({filepath.stat().st_size} bytes)")
    else:
        print(f"✗ {description}: {filename} NOT FOUND")

print("\nTo run full tests, install dependencies:")
print("  pip install pandas openpyxl")
print("Then run: python test_loaders.py")

