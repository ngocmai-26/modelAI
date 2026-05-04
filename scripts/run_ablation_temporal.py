#!/usr/bin/env python3
"""Phase 2 — Temporal Feature Ablation (advisor feedback #1).

Compares model with vs without temporal attendance features
(slope_3w, slope_full, volatility, late_streak, early_dropoff,
num_weeks_observed) at the same random_state and split.
"""

import json
import sys
from pathlib import Path

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
    "no_temporal": {
        "description": "Aggregate attendance only (no per-week temporal features)",
        "enable_temporal_features": False,
        "model_name": "ablation_no_temporal.joblib",
    },
    "with_temporal": {
        "description": "+ slope/volatility/late_streak/early_dropoff per (student, year)",
        "enable_temporal_features": True,
        "model_name": "ablation_with_temporal.joblib",
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
        enable_temporal_features=config["enable_temporal_features"],
    )
    output_path = MODELS / config["model_name"]
    model, metrics = pipeline.run(
        output_path=str(output_path),
        **COMMON_PATHS,
    )
    n_features = len(pipeline.feature_names) if pipeline.feature_names else 0

    return {
        "scenario": name,
        "description": config["description"],
        "enable_temporal_features": config["enable_temporal_features"],
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

    base = ablation_results.get("no_temporal", {}).get("metrics")
    ext = ablation_results.get("with_temporal", {}).get("metrics")
    if base and ext:
        ablation_results["delta"] = {
            "mae_delta": ext["test_mae"] - base["test_mae"],
            "rmse_delta": ext["test_rmse"] - base["test_rmse"],
            "r2_delta": ext["test_r2"] - base["test_r2"],
            "interpretation": (
                "Temporal IMPROVES model" if ext["test_mae"] < base["test_mae"]
                else "Temporal does NOT improve model"
            ),
        }

    results["ablation_temporal"] = ablation_results
    save_results(results)

    print("\n" + "=" * 80)
    print("ABLATION SUMMARY (Temporal)")
    print("=" * 80)
    print(f"{'Scenario':<25} {'MAE':>8} {'RMSE':>8} {'R²':>8} {'features':>10}")
    print("-" * 80)
    for name in ("no_temporal", "with_temporal"):
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
            f"{'Δ (with - no)':<25} "
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
