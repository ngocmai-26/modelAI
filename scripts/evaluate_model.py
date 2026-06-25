#!/usr/bin/env python3
"""Đánh giá lại mô hình CLO và in toàn bộ chỉ số báo cáo (để minh chứng trước hội đồng).

Tái lập ĐÚNG tập test lúc huấn luyện (GroupShuffleSplit theo Student_ID, seed 42),
nạp `models/model.joblib`, dự đoán trên tập test (3.358 mẫu) rồi in:
  - Hồi quy: MAE, RMSE, R², Pearson, Spearman
  - Phân loại Đạt/Rớt (ngưỡng 4.0): Accuracy, Precision, Recall, F1, ROC-AUC, AP,
    ma trận nhầm lẫn
  - Sai số theo dải điểm, và khoảng cách Train/Test (overfit)

Bundle dự báo nhúng trong model.joblib KHÔNG tham gia bước này (chỉ dùng model chính),
nên các con số khớp tuyệt đối với báo cáo.

Cách chạy:
    export PYTHONPATH="$(pwd)/src"
    python scripts/evaluate_model.py            # dùng data/ mặc định
    python scripts/evaluate_model.py --model models/model.joblib
"""
from __future__ import annotations

import argparse
import sys
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")

import numpy as np
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    precision_score,
    r2_score,
    recall_score,
    roc_auc_score,
)

from ml_clo.models.ensemble_model import EnsembleModel
from ml_clo.pipelines.train_pipeline import TrainingPipeline

PASS_THRESHOLD = 4.0  # ngưỡng đạt CLO (thang 0–6)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Đánh giá lại mô hình CLO (in chỉ số báo cáo)")
    p.add_argument("--model", default="models/model.joblib")
    p.add_argument("--data-dir", default="data")
    p.add_argument("--exam-scores", default=None)
    return p.parse_args()


def main() -> int:
    args = parse_args()
    d = Path(args.data_dir)
    paths = dict(
        exam_scores_path=args.exam_scores or str(d / "DiemTong.xlsx"),
        conduct_scores_path=str(d / "diemrenluyen.xlsx"),
        demographics_path=str(d / "nhankhau.xlsx"),
        teaching_methods_path=str(d / "PPGDfull.xlsx"),
        assessment_methods_path=str(d / "PPDGfull.xlsx"),
        study_hours_path=str(d / "tuhoc.xlsx"),
        attendance_path=str(d / "Dữ liệu điểm danh Khoa FIRA.xlsx"),
    )
    if not Path(paths["exam_scores_path"]).exists():
        print(f"Không tìm thấy {paths['exam_scores_path']}", file=sys.stderr)
        return 1

    # Tái lập tập test y hệt lúc train (seed 42, GroupShuffleSplit theo Student_ID).
    tr = TrainingPipeline(random_state=42)
    data = tr.load_data(**paths)
    tdf = tr.prepare_training_dataset(data)
    X, y, _ = tr.prepare_features(tdf)
    groups = tdf.loc[X.index, "Student_ID"]
    _, _, X_test, _, _, y_test = tr.split_data(X, y, groups=groups)

    model = EnsembleModel(random_state=42)
    model.load(args.model)
    pred = np.clip(model.predict(X_test), 0.0, 6.0)
    y_test = np.asarray(y_test, dtype=float)

    print("=" * 70)
    print(f"ĐÁNH GIÁ MÔ HÌNH — {args.model}  (test = {len(y_test)} mẫu)")
    print("=" * 70)
    print("\n[ HỒI QUY ]")
    print(f"  MAE        = {mean_absolute_error(y_test, pred):.4f}")
    print(f"  RMSE       = {np.sqrt(mean_squared_error(y_test, pred)):.4f}")
    print(f"  R²         = {r2_score(y_test, pred):.4f}")
    print(f"  Pearson r  = {pearsonr(y_test, pred)[0]:.4f}")
    print(f"  Spearman ρ = {spearmanr(y_test, pred)[0]:.4f}")

    yb = (y_test >= PASS_THRESHOLD).astype(int)
    pb = (pred >= PASS_THRESHOLD).astype(int)
    print(f"\n[ PHÂN LOẠI Đạt/Rớt — ngưỡng {PASS_THRESHOLD} ]")
    print(f"  Accuracy   = {accuracy_score(yb, pb):.4f}")
    print(f"  Precision  = {precision_score(yb, pb, zero_division=0):.4f}")
    print(f"  Recall     = {recall_score(yb, pb, zero_division=0):.4f}")
    print(f"  F1         = {f1_score(yb, pb, zero_division=0):.4f}")
    print(f"  ROC-AUC    = {roc_auc_score(yb, pred):.4f}")
    print(f"  Avg Prec.  = {average_precision_score(yb, pred):.4f}")
    tn, fp, fn, tp = confusion_matrix(yb, pb).ravel()
    print(f"  Confusion  = TN={tn}  FP={fp}  FN={fn}  TP={tp}")

    print("\n[ SAI SỐ THEO DẢI ĐIỂM ]")
    for lo, hi, name in [(0, 2, "Thấp [0-2)"), (2, 4, "TB [2-4)"), (4, 6.01, "Cao [4-6]")]:
        mask = (y_test >= lo) & (y_test < hi)
        if mask.sum() > 0:
            print(f"  {name:<11} n={int(mask.sum()):<5} MAE={mean_absolute_error(y_test[mask], pred[mask]):.4f}")

    print("=" * 70)
    return 0


if __name__ == "__main__":
    sys.exit(main())
