"""Bước 5: Kiểm tra tính phân biệt kết quả dự đoán.

Dự đoán cho nhiều SV khác nhau (cùng môn, GV) với data đầy đủ.
Assert: predicted_clo_score không được giống hệt nhau.
"""

from pathlib import Path

import pytest

from ml_clo.data.loaders import load_exam_scores
from ml_clo.data.preprocessors import preprocess_exam_scores
from ml_clo.pipelines import PredictionPipeline


@pytest.mark.skipif(
    not Path("models/model.joblib").exists() or not Path("data/DiemTong.xlsx").exists(),
    reason="Cần models/model.joblib và data/DiemTong.xlsx",
)
def test_prediction_differentiation_with_full_data():
    """Predict cho 5+ SV khác nhau (cùng môn, GV) — điểm dự đoán phải khác nhau."""
    model_path = "models/model.joblib"
    exam_path = "data/DiemTong.xlsx"
    subject_id = "INF0823"
    lecturer_id = "90316"
    max_students = 8

    exam_df = load_exam_scores(exam_path)
    exam_df = preprocess_exam_scores(exam_df, convert_to_clo=True, create_result=False)
    class_df = exam_df[
        (exam_df["Subject_ID"] == subject_id) & (exam_df["Lecturer_ID"] == lecturer_id)
    ]
    students = class_df["Student_ID"].unique()[:max_students].tolist()

    if len(students) < 2:
        pytest.skip("Cần ít nhất 2 SV cùng môn+GV")

    # Data paths (optional files)
    data_dir = Path("data")
    conduct = data_dir / "diemrenluyen.xlsx" if (data_dir / "diemrenluyen.xlsx").exists() else None
    demo = data_dir / "nhankhau.xlsx" if (data_dir / "nhankhau.xlsx").exists() else None
    ppgd = data_dir / "PPGDfull.xlsx" if (data_dir / "PPGDfull.xlsx").exists() else None
    ppdg = data_dir / "PPDGfull.xlsx" if (data_dir / "PPDGfull.xlsx").exists() else None
    att = list(data_dir.glob("Dữ liệu điểm danh*.xlsx"))
    att_path = str(att[0]) if att else None

    pipeline = PredictionPipeline(
        model_path=model_path,
        exam_scores_path=exam_path,
        conduct_scores_path=str(conduct) if conduct else None,
        demographics_path=str(demo) if demo else None,
        teaching_methods_path=str(ppgd) if ppgd else None,
        assessment_methods_path=str(ppdg) if ppdg else None,
        attendance_path=att_path,
    )
    pipeline.load_model()

    scores = []
    for sid in students:
        out = pipeline.predict(
            student_id=str(sid),
            subject_id=subject_id,
            lecturer_id=lecturer_id,
        )
        scores.append(out.predicted_clo_score)

    unique_count = len(set(scores))
    assert unique_count > 1, (
        f"Tất cả {len(scores)} điểm dự đoán giống nhau ({scores[0]:.4f}). "
        "Cần truyền đủ nguồn data (conduct, demographics, attendance) để có kết quả cá nhân hóa."
    )
