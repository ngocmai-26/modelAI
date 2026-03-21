#!/usr/bin/env python3
"""Bước 5: Kiểm tra tính phân biệt kết quả dự đoán.

Dự đoán cho 5-10 SV khác nhau (cùng môn, cùng GV) với data đầy đủ.
Kiểm tra: predicted_clo_score phải khác nhau khi features khác.

Chạy:
    python scripts/test_prediction_differentiation.py \\
        --model models/model.joblib \\
        --exam-scores data/DiemTong.xlsx \\
        --conduct-scores data/diemrenluyen.xlsx \\
        --demographics data/nhankhau.xlsx \\
        --teaching-methods data/PPGDfull.xlsx \\
        --assessment-methods data/PPDGfull.xlsx \\
        --attendance "data/Dữ liệu điểm danh Khoa FIRA.xlsx" \\
        --subject-id INF0823 \\
        --lecturer-id 90316 \\
        --max-students 10
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import pandas as pd

from ml_clo.data.loaders import load_exam_scores
from ml_clo.data.preprocessors import preprocess_exam_scores
from ml_clo.pipelines import PredictionPipeline


def parse_args():
    parser = argparse.ArgumentParser(
        description="Kiểm tra tính phân biệt kết quả predict (Bước 5)"
    )
    parser.add_argument("--model", required=True, help="Path to model")
    parser.add_argument("--exam-scores", required=True, help="Path to DiemTong")
    parser.add_argument("--conduct-scores", default=None)
    parser.add_argument("--demographics", default=None)
    parser.add_argument("--teaching-methods", default=None)
    parser.add_argument("--assessment-methods", default=None)
    parser.add_argument("--study-hours", default=None)
    parser.add_argument("--attendance", default=None)
    parser.add_argument("--subject-id", default="INF0823")
    parser.add_argument("--lecturer-id", default="90316")
    parser.add_argument("--max-students", type=int, default=10)
    return parser.parse_args()


def main():
    args = parse_args()

    # Load exam, filter by subject+lecturer, get unique students
    exam_df = load_exam_scores(args.exam_scores)
    exam_df = preprocess_exam_scores(exam_df, convert_to_clo=True, create_result=False)
    class_df = exam_df[
        (exam_df["Subject_ID"] == args.subject_id)
        & (exam_df["Lecturer_ID"] == args.lecturer_id)
    ]
    students = class_df["Student_ID"].unique()[: args.max_students].tolist()

    if len(students) < 2:
        print(
            f"WARN: Cần ít nhất 2 SV (cùng môn {args.subject_id}, GV {args.lecturer_id}). "
            f"Tìm thấy {len(students)}."
        )
        return 0

    # Init pipeline với data đầy đủ (cache 1 lần)
    pipeline = PredictionPipeline(
        model_path=args.model,
        exam_scores_path=args.exam_scores,
        conduct_scores_path=args.conduct_scores,
        demographics_path=args.demographics,
        teaching_methods_path=args.teaching_methods,
        assessment_methods_path=args.assessment_methods,
        study_hours_path=args.study_hours,
        attendance_path=args.attendance,
    )
    pipeline.load_model()

    # Predict từng SV (dùng cache)
    results = []
    for sid in students:
        out = pipeline.predict(
            student_id=str(sid),
            subject_id=args.subject_id,
            lecturer_id=args.lecturer_id,
        )
        results.append({"student_id": sid, "predicted_clo_score": out.predicted_clo_score})

    # Phân tích
    scores = [r["predicted_clo_score"] for r in results]
    unique_scores = len(set(scores))
    all_same = unique_scores == 1

    print("=" * 60)
    print("BƯỚC 5: Kiểm tra tính phân biệt kết quả dự đoán")
    print("=" * 60)
    print(f"Môn: {args.subject_id}, GV: {args.lecturer_id}")
    print(f"Số SV test: {len(students)}")
    print(f"Số điểm dự đoán khác nhau: {unique_scores}/{len(scores)}")
    print()
    for r in results:
        print(f"  SV {r['student_id']}: {r['predicted_clo_score']:.4f}")
    print()

    if all_same:
        print(
            "⚠️  CẢNH BÁO: Tất cả điểm dự đoán giống nhau. "
            "Có thể do: features giống nhau, fill NaN quá mạnh, hoặc thiếu nguồn data."
        )
        print(
            "   Khuyến nghị: Truyền đủ --conduct-scores, --demographics, "
            "--attendance khi predict để có kết quả cá nhân hóa."
        )
        return 1
    else:
        print("✓ Các điểm dự đoán khác nhau — tính phân biệt đạt yêu cầu.")
        return 0


if __name__ == "__main__":
    sys.exit(main())
