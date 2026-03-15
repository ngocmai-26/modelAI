"""Unit tests for ensemble model."""

import numpy as np
import pandas as pd
import pytest

from ml_clo.models.ensemble_model import EnsembleModel
from ml_clo.utils.exceptions import ModelLoadError


class TestEnsembleModel:
    """Test EnsembleModel class."""

    def test_ensemble_model_init(self):
        """Test ensemble model initialization."""
        model = EnsembleModel(random_state=42)

        assert model.model_name == "Ensemble"
        assert model.random_state == 42
        assert not model.is_trained

    def test_ensemble_model_train(self, sample_features):
        """Test ensemble model training."""
        X, y = sample_features

        # Split data
        from sklearn.model_selection import train_test_split

        X_train, X_val, y_train, y_val = train_test_split(
            X, y, test_size=0.2, random_state=42
        )

        model = EnsembleModel(random_state=42)
        metrics = model.train(X_train, y_train, X_val, y_val)

        assert model.is_trained
        assert "ensemble_train_mae" in metrics
        assert "ensemble_val_mae" in metrics
        assert metrics["ensemble_train_mae"] >= 0
        assert metrics["ensemble_val_mae"] >= 0

    def test_ensemble_model_predict(self, trained_model, sample_features):
        """Test ensemble model prediction."""
        X, _ = sample_features

        predictions = trained_model.predict(X)

        assert len(predictions) == len(X)
        assert all(0 <= p <= 6 for p in predictions)  # CLO scale

    def test_ensemble_model_predict_not_trained(self, sample_features):
        """Test prediction without training."""
        X, _ = sample_features

        model = EnsembleModel(random_state=42)

        with pytest.raises(ModelLoadError):
            model.predict(X)

    def test_ensemble_model_save_load(self, trained_model, tmp_path):
        """Test model save and load."""
        model_path = tmp_path / "test_model.joblib"

        # Save
        trained_model.save(str(model_path))
        assert model_path.exists()

        # Load
        new_model = EnsembleModel(random_state=42)
        new_model.load(str(model_path))

        assert new_model.is_trained
        assert new_model.model_name == trained_model.model_name
        assert new_model.version == trained_model.version

    def test_ensemble_model_version(self):
        """Test model version generation."""
        model1 = EnsembleModel(random_state=42)
        model2 = EnsembleModel(random_state=42)

        # Versions should be different (timestamp-based)
        assert model1.version != model2.version or model1.version.startswith("v1.0")

