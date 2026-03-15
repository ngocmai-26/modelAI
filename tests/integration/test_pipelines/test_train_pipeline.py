"""Integration tests for training pipeline and prediction pipeline."""

import pandas as pd
import pytest
from pathlib import Path

from ml_clo.pipelines import AnalysisPipeline, PredictionPipeline, TrainingPipeline


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


class TestPredictionPipeline:
    """Test PredictionPipeline (yêu cầu mới: không cần DiemTong, fallback create_student_record_from_ids)."""

    def test_load_data_cache_without_exam_scores(self, trained_model, tmp_path):
        """Test load_data_cache hoạt động khi chỉ có demographics + PPGD + PPDG (không có DiemTong)."""
        model_path = tmp_path / "model.joblib"
        trained_model.save(str(model_path))

        demo_path = tmp_path / "demo.xlsx"
        ppgd_path = tmp_path / "ppgd.xlsx"
        ppdg_path = tmp_path / "ppdg.xlsx"
        pd.DataFrame({"Student_ID": [19050006], "Gender": [1]}).to_excel(demo_path, index=False)
        pd.DataFrame({"Subject_ID": ["INF0823"], "TM1": ["X"], "TM2": [""]}).to_excel(ppgd_path, index=False)
        pd.DataFrame({"Subject_ID": ["INF0823"], "EM1": ["X"], "EM2": [""]}).to_excel(ppdg_path, index=False)

        pipeline = PredictionPipeline(
            model_path=str(model_path),
            demographics_path=str(demo_path),
            teaching_methods_path=str(ppgd_path),
            assessment_methods_path=str(ppdg_path),
        )
        assert pipeline._data_cache is not None
        assert pipeline._data_cache.get("demographics") is not None
        assert pipeline._data_cache.get("teaching_methods") is not None
        assert pipeline._data_cache.get("assessment_methods") is not None
        assert pipeline._data_cache.get("exam_scores") is None

    def test_load_student_data_fallback_create_record_from_ids(self, trained_model, tmp_path):
        """Test load_student_data fallback sang create_student_record_from_ids khi SV không có trong DiemTong."""
        model_path = tmp_path / "model.joblib"
        trained_model.save(str(model_path))

        demo_path = tmp_path / "demo.xlsx"
        ppgd_path = tmp_path / "ppgd.xlsx"
        ppdg_path = tmp_path / "ppdg.xlsx"
        pd.DataFrame({"Student_ID": [99999999], "Gender": [1]}).to_excel(demo_path, index=False)
        pd.DataFrame({"Subject_ID": ["INF0823"], "TM1": ["X"]}).to_excel(ppgd_path, index=False)
        pd.DataFrame({"Subject_ID": ["INF0823"], "EM1": ["X"]}).to_excel(ppdg_path, index=False)

        pipeline = PredictionPipeline(
            model_path=str(model_path),
            demographics_path=str(demo_path),
            teaching_methods_path=str(ppgd_path),
            assessment_methods_path=str(ppdg_path),
        )
        pipeline.load_model()
        df = pipeline.load_student_data(
            student_id="99999999",
            subject_id="INF0823",
            lecturer_id="90316",
        )
        assert len(df) >= 1
        assert "Student_ID" in df.columns
        assert df["Student_ID"].iloc[0] == 99999999 or 99999999 in df["Student_ID"].values


class TestAnalysisPipeline:
    """Test AnalysisPipeline (yêu cầu mới: analyze_class_from_scores)."""

    def test_analyze_class_from_scores_distribution_only(self, trained_model, tmp_path):
        """Test analyze_class_from_scores với List[float] — chỉ phân phối, không SHAP."""
        model_path = tmp_path / "model.joblib"
        trained_model.save(str(model_path))
        analyzer = AnalysisPipeline(str(model_path))
        result = analyzer.analyze_class_from_scores(
            subject_id="INF0823",
            lecturer_id="90316",
            clo_scores=[4.2, 3.8, 5.1, 2.9, 4.5],
        )
        assert result.total_students == 5
        assert 0 <= result.average_predicted_score <= 6
        assert result.summary
        assert result.subject_id == "INF0823"
        assert result.lecturer_id == "90316"

    def test_analyze_class_from_scores_dict(self, trained_model, tmp_path):
        """Test analyze_class_from_scores với Dict[student_id, score] — distribution (thiếu data)."""
        model_path = tmp_path / "model.joblib"
        trained_model.save(str(model_path))
        analyzer = AnalysisPipeline(str(model_path))
        result = analyzer.analyze_class_from_scores(
            subject_id="INF0823",
            lecturer_id="90316",
            clo_scores={"19050006": 4.2, "19050007": 3.8},
        )
        assert result.total_students == 2
        assert 0 <= result.average_predicted_score <= 6

