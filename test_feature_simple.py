"""Simple test for feature engineering imports."""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

try:
    print("Testing imports...")
    from ml_clo.config.feature_config import FEATURE_GROUPS
    print("✓ feature_config imported")
    
    from ml_clo.features.feature_groups import get_feature_groups
    print("✓ feature_groups imported")
    
    from ml_clo.features.feature_builder import build_conduct_features
    print("✓ feature_builder imported")
    
    print("\nAll imports successful!")
    print(f"Feature groups: {len(FEATURE_GROUPS)} groups")
    
except Exception as e:
    print(f"✗ Error: {e}")
    import traceback
    traceback.print_exc()


