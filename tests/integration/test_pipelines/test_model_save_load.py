"""Integration tests for model save/load."""

import pytest

from ml_clo.models.ensemble_model import EnsembleModel
from ml_clo.utils.exceptions import ModelLoadError


class TestModelSaveLoad:
    """Test model save and load functionality."""

    def test_model_save_load_roundtrip(self, trained_model, tmp_path):
        """Test saving and loading model maintains functionality."""
        model_path = tmp_path / "test_model.joblib"

        # Save
        trained_model.save(str(model_path))
        assert model_path.exists()

        # Load
        new_model = EnsembleModel(random_state=42)
        new_model.load(str(model_path))

        # Verify model properties
        assert new_model.is_trained
        assert new_model.model_name == trained_model.model_name
        assert new_model.version == trained_model.version
        assert new_model.feature_names == trained_model.feature_names

    def test_model_load_nonexistent_file(self):
        """Test loading non-existent model file."""
        model = EnsembleModel(random_state=42)

        with pytest.raises(ModelLoadError):
            model.load("nonexistent_model.joblib")

    def test_model_predict_after_load(self, trained_model, sample_features, tmp_path):
        """Test prediction works after loading model."""
        X, _ = sample_features
        model_path = tmp_path / "test_model.joblib"

        # Save and load
        trained_model.save(str(model_path))
        new_model = EnsembleModel(random_state=42)
        new_model.load(str(model_path))

        # Predict with original and loaded model
        predictions_original = trained_model.predict(X.iloc[:5])
        predictions_loaded = new_model.predict(X.iloc[:5])

        # Predictions should be similar (allowing for small numerical differences)
        assert len(predictions_original) == len(predictions_loaded)
        assert all(abs(p1 - p2) < 0.01 for p1, p2 in zip(predictions_original, predictions_loaded))

