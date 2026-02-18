"""Model configuration and hyperparameters.

This module defines hyperparameters and configuration for all models.
"""

from typing import Dict, Any

# Random Forest Regressor hyperparameters
RANDOM_FOREST_CONFIG: Dict[str, Any] = {
    "n_estimators": 1000,
    "max_depth": 25,
    "random_state": 42,
    "n_jobs": -1,  # Use all available cores
    "verbose": 0,
}

# Gradient Boosting Regressor hyperparameters
GRADIENT_BOOSTING_CONFIG: Dict[str, Any] = {
    "n_estimators": 500,
    "max_depth": 12,
    "learning_rate": 0.03,
    "random_state": 42,
    "verbose": 0,
}

# Ensemble configuration
ENSEMBLE_CONFIG: Dict[str, Any] = {
    "weights_method": "validation_performance",  # or "equal"
    "min_weight": 0.1,  # Minimum weight for any model
    "max_weight": 0.9,  # Maximum weight for any model
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


