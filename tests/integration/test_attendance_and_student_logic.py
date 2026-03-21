"""Bước 7: Integration test — điểm danh (train/predict) và logic SV năm 1.

- Train với attendance → predict với attendance
- SV năm 1 (không trong DiemTong): fallback OK
- SV có lịch sử: dùng Điểm tổng
"""

from pathlib import Path

import pandas as pd
import pytest

from ml_clo.data.loaders import load_exam_scores
from ml_clo.data.preprocessors import preprocess_exam_scores
from ml_clo.pipelines import PredictionPipeline, TrainingPipeline


def _attendance_path():
    att = list(Path("data").glob("Dữ liệu điểm danh*.xlsx"))
    return str(att[0]) if att else None


def _data_paths():
    """Paths to optional data files."""
    d = Path("data")
    return {
        "exam": "data/DiemTong.xlsx" if (d / "DiemTong.xlsx").exists() else None,
        "conduct": str(d / "diemrenluyen.xlsx") if (d / "diemrenluyen.xlsx").exists() else None,
        "demo": str(d / "nhankhau.xlsx") if (d / "nhankhau.xlsx").exists() else None,
        "ppgd": str(d / "PPGDfull.xlsx") if (d / "PPGDfull.xlsx").exists() else None,
        "ppdg": str(d / "PPDGfull.xlsx") if (d / "PPDGfull.xlsx").exists() else None,
        "study": str(d / "tuhoc.xlsx") if (d / "tuhoc.xlsx").exists() else None,
        "attendance": _attendance_path(),
    }


@pytest.mark.skipif(
    not Path("data/DiemTong.xlsx").exists(),
    reason="Cần data/DiemTong.xlsx",
)
class TestTrainAndPredictWithAttendance:
    """Integration: train với attendance → predict với attendance."""

    def test_train_with_attendance_produces_model(self, tmp_path):
        """Train với --attendance chạy thành công."""
        paths = _data_paths()
        if not paths["attendance"]:
            pytest.skip("Cần file điểm danh")

        out = tmp_path / "model_att.joblib"
        trainer = TrainingPipeline(random_state=42, test_size=0.2, validation_size=0.2)
        model, metrics = trainer.run(
            exam_scores_path=paths["exam"],
            output_path=str(out),
            conduct_scores_path=paths["conduct"],
            demographics_path=paths["demo"],
            teaching_methods_path=paths["ppgd"],
            assessment_methods_path=paths["ppdg"],
            study_hours_path=paths["study"],
            attendance_path=paths["attendance"],
        )
        assert model.is_trained
        assert out.exists()
        assert "test_mae" in metrics

    def test_predict_with_attendance_when_model_trained_with_it(self, tmp_path):
        """Predict với attendance khi model đã train cùng attendance."""
        paths = _data_paths()
        if not paths["attendance"] or not paths["exam"]:
            pytest.skip("Cần DiemTong và file điểm danh")

        # Train với attendance
        out = tmp_path / "model_att.joblib"
        trainer = TrainingPipeline(random_state=42, test_size=0.2, validation_size=0.2)
        trainer.run(
            exam_scores_path=paths["exam"],
            output_path=str(out),
            conduct_scores_path=paths["conduct"],
            demographics_path=paths["demo"],
            teaching_methods_path=paths["ppgd"],
            assessment_methods_path=paths["ppdg"],
            study_hours_path=paths["study"],
            attendance_path=paths["attendance"],
        )
        assert out.exists()

        # Predict với attendance
        pipeline = PredictionPipeline(
            model_path=str(out),
            exam_scores_path=paths["exam"],
            conduct_scores_path=paths["conduct"],
            demographics_path=paths["demo"],
            teaching_methods_path=paths["ppgd"],
            assessment_methods_path=paths["ppdg"],
            attendance_path=paths["attendance"],
        )
        pipeline.load_model()

        exam_df = load_exam_scores(paths["exam"])
        exam_df = preprocess_exam_scores(exam_df, convert_to_clo=True, create_result=False)
        class_df = exam_df[
            (exam_df["Subject_ID"] == "INF0823") & (exam_df["Lecturer_ID"] == "90316")
        ]
        students = class_df["Student_ID"].unique()[:2].tolist()
        if not students:
            pytest.skip("Không tìm thấy SV INF0823 / 90316")

        out_result = pipeline.predict(
            student_id=str(students[0]),
            subject_id="INF0823",
            lecturer_id="90316",
        )
        assert out_result.predicted_clo_score >= 0
        assert out_result.predicted_clo_score <= 6


@pytest.mark.skipif(
    not Path("models/model.joblib").exists()
    or not Path("data/nhankhau.xlsx").exists()
    or not Path("data/PPGDfull.xlsx").exists(),
    reason="Cần models/model.joblib, nhankhau.xlsx, PPGDfull.xlsx",
)
class TestStudentYear1Logic:
    """Test logic SV năm 1 vs năm 2+ (Bước 3)."""

    def test_fallback_ok_for_student_not_in_diemtong(self, tmp_path):
        """SV không có trong DiemTong + demo+ppgd+ppdg → predict OK (fallback)."""
        # MSSV giả định không có trong DiemTong; dùng demo+ppgd+ppdg
        demo_path = tmp_path / "demo.xlsx"
        ppgd_path = tmp_path / "ppgd.xlsx"
        ppdg_path = tmp_path / "ppdg.xlsx"
        pd.DataFrame({"Student_ID": [99999999], "Gender": [1]}).to_excel(demo_path, index=False)
        pd.DataFrame({"Subject_ID": ["INF0823"], "TM1": ["X"]}).to_excel(ppgd_path, index=False)
        pd.DataFrame({"Subject_ID": ["INF0823"], "EM1": ["X"]}).to_excel(ppdg_path, index=False)

        pipeline = PredictionPipeline(
            model_path="models/model.joblib",
            demographics_path=str(demo_path),
            teaching_methods_path=str(ppgd_path),
            assessment_methods_path=str(ppdg_path),
        )
        pipeline.load_model()
        result = pipeline.predict(
            student_id="99999999",
            subject_id="INF0823",
            lecturer_id="90316",
        )
        assert result.predicted_clo_score >= 0
        assert result.predicted_clo_score <= 6
