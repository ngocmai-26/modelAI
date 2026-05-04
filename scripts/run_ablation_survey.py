#!/usr/bin/env python3
"""Phase 1 — Survey Ablation Study (advisor feedback #2).

Compares two scenarios at the same random_state and split:
- baseline_lms:    LMS/SIS data only (no survey)
- extended_survey: LMS/SIS + survey responses (280 students, 76 overlap with exam)

Outputs metrics to ``experiments/results.json`` under
``ablation_survey``.
"""

import json
import sys
from pathlib import Path

# Path setup
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from ml_clo.pipelines import TrainingPipeline  # noqa: E402

DATA = ROOT / "data"
MODELS = ROOT / "models"
RESULTS_FILE = ROOT / "experiments" / "results.json"

COMMON_PATHS = {
    "exam_scores_path":         str(DATA / "DiemTong.xlsx"),
    "conduct_scores_path":      str(DATA / "diemrenluyen.xlsx"),
    "demographics_path":        str(DATA / "nhankhau.xlsx"),
    "teaching_methods_path":    str(DATA / "PPGDfull.xlsx"),
    "assessment_methods_path":  str(DATA / "PPDGfull.xlsx"),
    "study_hours_path":         str(DATA / "tuhoc.xlsx"),
    "attendance_path":          str(DATA / "Dữ liệu điểm danh Khoa FIRA.xlsx"),
}

SCENARIOS = {
    "baseline_lms": {
        "description": "LMS/SIS only (no survey)",
        "paths": COMMON_PATHS,
        "model_name": "ablation_baseline_lms.joblib",
    },
    "extended_survey": {
        "description": "LMS/SIS + survey responses",
        "paths": {
            **COMMON_PATHS,
            "survey_path": str(
                DATA / "Khảo sát các yếu tố cá nhân phục vụ phân tích và dự báo kết quả học tập sinh viên (Câu trả lời).xlsx"
            ),
        },
        "model_name": "ablation_extended_survey.joblib",
    },
}


def load_results() -> dict:
    if RESULTS_FILE.exists():
        return json.loads(RESULTS_FILE.read_text(encoding="utf-8"))
    return {}


def save_results(data: dict) -> None:
    RESULTS_FILE.parent.mkdir(parents=True, exist_ok=True)
    RESULTS_FILE.write_text(
        json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8"
    )


def run_scenario(name: str, config: dict) -> dict:
    print(f"\n{'='*80}\nRunning scenario: {name}\n{config['description']}\n{'='*80}")

    pipeline = TrainingPipeline(
        random_state=42,
        test_size=0.2,
        validation_size=0.2,
        group_split_by_student=True,
    )
    output_path = MODELS / config["model_name"]
    model, metrics = pipeline.run(
        output_path=str(output_path),
        **config["paths"],
    )

    n_features = len(pipeline.feature_names) if pipeline.feature_names else 0

    return {
        "scenario": name,
        "description": config["description"],
        "model_path": str(output_path),
        "model_version": model.version,
        "n_features": n_features,
        "metrics": {
            "test_mae": float(metrics.get("test_mae", 0)),
            "test_rmse": float(metrics.get("test_rmse", 0)),
            "test_r2": float(metrics.get("test_r2", 0)),
            "val_mae": float(metrics.get("val_mae") or 0),
            "val_rmse": float(metrics.get("val_rmse") or 0),
            "val_r2": float(metrics.get("val_r2") or 0),
            "train_mae": float(metrics.get("train_mae") or 0),
            "train_rmse": float(metrics.get("train_rmse") or 0),
            "train_r2": float(metrics.get("train_r2") or 0),
        },
        "ensemble_weights": {
            "rf": float(getattr(model, "rf_weight", 0)),
            "gb": float(getattr(model, "gb_weight", 0)),
        },
    }


def main() -> int:
    results = load_results()
    ablation_results = {}

    for name, config in SCENARIOS.items():
        try:
            ablation_results[name] = run_scenario(name, config)
        except Exception as exc:
            print(f"ERROR in scenario {name}: {exc}", file=sys.stderr)
            ablation_results[name] = {"error": str(exc)}

    # Compute deltas
    base = ablation_results.get("baseline_lms", {}).get("metrics")
    ext = ablation_results.get("extended_survey", {}).get("metrics")
    if base and ext:
        ablation_results["delta"] = {
            "mae_delta": ext["test_mae"] - base["test_mae"],
            "rmse_delta": ext["test_rmse"] - base["test_rmse"],
            "r2_delta": ext["test_r2"] - base["test_r2"],
            "interpretation": (
                "Survey IMPROVES model" if ext["test_mae"] < base["test_mae"]
                else "Survey does NOT improve model"
            ),
        }

    results["ablation_survey"] = ablation_results
    save_results(results)

    print("\n" + "=" * 80)
    print("ABLATION SUMMARY (Survey)")
    print("=" * 80)
    print(f"{'Scenario':<25} {'MAE':>8} {'RMSE':>8} {'R²':>8} {'features':>10}")
    print("-" * 80)
    for name in ("baseline_lms", "extended_survey"):
        r = ablation_results.get(name, {})
        m = r.get("metrics", {})
        print(
            f"{name:<25} "
            f"{m.get('test_mae', 0):>8.4f} "
            f"{m.get('test_rmse', 0):>8.4f} "
            f"{m.get('test_r2', 0):>8.4f} "
            f"{r.get('n_features', 0):>10}"
        )
    if "delta" in ablation_results:
        d = ablation_results["delta"]
        print("-" * 80)
        print(
            f"{'Δ (extended - baseline)':<25} "
            f"{d['mae_delta']:>+8.4f} "
            f"{d['rmse_delta']:>+8.4f} "
            f"{d['r2_delta']:>+8.4f}"
        )
        print(f"\nVerdict: {d['interpretation']}")
    print("=" * 80)
    print(f"\nResults saved to: {RESULTS_FILE}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
