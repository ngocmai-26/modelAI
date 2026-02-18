"""Integration tests for training pipeline."""

import pytest
from pathlib import Path

from ml_clo.pipelines import TrainingPipeline


class TestTrainingPipeline:
    """Test TrainingPipeline class."""

    @pytest.mark.skipif(
        not Path("data/DiemTong.xlsx").exists(),
        reason="Data file not found",
    )
    def test_training_pipeline_full(self, tmp_path):
        """Test full training pipeline."""
        output_path = tmp_path / "test_model.joblib"

        trainer = TrainingPipeline(random_state=42, test_size=0.2, validation_size=0.2)

        model, metrics = trainer.run(
            exam_scores_path="data/DiemTong.xlsx",
            output_path=str(output_path),
            conduct_scores_path="data/diemrenluyen.xlsx" if Path("data/diemrenluyen.xlsx").exists() else None,
            demographics_path="data/nhankhau.xlsx" if Path("data/nhankhau.xlsx").exists() else None,
            teaching_methods_path="data/PPGDfull.xlsx" if Path("data/PPGDfull.xlsx").exists() else None,
            assessment_methods_path="data/PPDGfull.xlsx" if Path("data/PPDGfull.xlsx").exists() else None,
            study_hours_path="data/tuhoc.xlsx" if Path("data/tuhoc.xlsx").exists() else None,
        )

        assert model.is_trained
        assert output_path.exists()
        assert "test_mae" in metrics
        assert "test_rmse" in metrics
        assert "test_r2" in metrics

    def test_training_pipeline_with_sample_data(self, sample_features, tmp_path):
        """Test training pipeline with sample data."""
        # This is a simplified test - in real scenario, we'd need to create
        # a proper training dataset structure
        pytest.skip("Requires proper data structure setup")

