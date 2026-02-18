"""Pytest configuration and shared fixtures."""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import numpy as np
import pandas as pd
import pytest

from ml_clo.models.ensemble_model import EnsembleModel


@pytest.fixture
def sample_exam_scores():
    """Create sample exam scores DataFrame."""
    return pd.DataFrame({
        "Student_ID": ["SV001", "SV002", "SV003", "SV004", "SV005"],
        "Subject_ID": ["SUB001", "SUB001", "SUB002", "SUB002", "SUB001"],
        "Lecturer_ID": ["LEC001", "LEC001", "LEC002", "LEC002", "LEC001"],
        "exam_score": [5.0, 6.0, 4.5, 3.0, 5.5],
        "year": [2023, 2023, 2023, 2023, 2023],
    })


@pytest.fixture
def sample_conduct_scores():
    """Create sample conduct scores DataFrame."""
    return pd.DataFrame({
        "Student_ID": ["SV001", "SV002", "SV003"],
        "conduct_score": [85.0, 90.0, 75.0],
        "year": [2023, 2023, 2023],
    })


@pytest.fixture
def sample_demographics():
    """Create sample demographics DataFrame."""
    return pd.DataFrame({
        "Student_ID": ["SV001", "SV002", "SV003"],
        "Gender": ["M", "F", "M"],
        "Birth_Place": ["Hà Nội", "TP.HCM", "Đà Nẵng"],
    })


@pytest.fixture
def sample_features():
    """Create sample feature DataFrame for model training."""
    np.random.seed(42)
    n_samples = 100
    n_features = 10

    X = pd.DataFrame(
        np.random.randn(n_samples, n_features),
        columns=[f"feature_{i}" for i in range(n_features)],
    )

    # Create target with some relationship to features
    y = pd.Series(
        X.iloc[:, 0] * 0.5 + X.iloc[:, 1] * 0.3 + np.random.randn(n_samples) * 0.1
    )
    # Clip to CLO scale (0-6)
    y = y.clip(lower=0.0, upper=6.0)

    return X, y


@pytest.fixture
def trained_model(sample_features):
    """Create a trained ensemble model for testing."""
    X, y = sample_features

    # Split data
    from sklearn.model_selection import train_test_split

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Train model
    model = EnsembleModel(random_state=42)
    model.train(X_train, y_train, X_test, y_test)

    return model

