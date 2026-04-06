"""Model configuration and hyperparameters.

This module defines hyperparameters and configuration for all models.
"""

from typing import Dict, Any

# Random Forest (pipeline chuẩn: độ chính xác + SHAP)
RANDOM_FOREST_CONFIG: Dict[str, Any] = {
    "n_estimators": 1000,
    "max_depth": 22,
    "min_samples_leaf": 4,
    "random_state": 42,
    "n_jobs": -1,
    "verbose": 0,
}

# Gradient Boosting — subsample + min_samples_leaf giúp tổng quát hóa
GRADIENT_BOOSTING_CONFIG: Dict[str, Any] = {
    "n_estimators": 600,
    "max_depth": 10,
    "learning_rate": 0.025,
    "subsample": 0.9,
    "min_samples_leaf": 5,
    "random_state": 42,
    "verbose": 0,
}

# Ensemble configuration
ENSEMBLE_CONFIG: Dict[str, Any] = {
    "weights_method": "validation_performance",  # or "equal"
    "min_weight": 0.1,  # Minimum weight for any model
    "max_weight": 0.9,  # Maximum weight for any model
    # GB đôi khi ngoại suy rất thấp trên vector lạ; RF ổn định hơn → kéo GB về phía RF khi lệch rõ
    "gb_low_anomaly_max_gb": 0.75,
    "gb_low_anomaly_min_gap": 0.35,
    "gb_low_anomaly_rf_blend": 0.88,  # gb_use = blend * rf + (1-blend) * gb
}

# Training configuration
TRAINING_CONFIG: Dict[str, Any] = {
    "test_size": 0.2,  # 20% for testing
    "validation_size": 0.2,  # 20% of training data for validation
    "random_state": 42,
    "shuffle": True,
}

# Model versioning
MODEL_VERSION_FORMAT = "v{version}_{timestamp}"  # e.g., "v1.0_20240205_120000"


