#!/usr/bin/env python3
"""Phase 3 — Categorical Encoding Ablation (advisor feedback #3).

Compares three strategies for encoding ``Subject_ID`` and ``Lecturer_ID``:
- hash:      stable_hash_int (current default; no semantic ordering)
- frequency: train-set frequency (count of occurrences)
- target:    K-fold mean target encoding (5 folds, smoothing=10)

For ``frequency`` and ``target``, the IDs are *included* as features.
For ``hash``, IDs remain excluded (current pipeline behavior) so the
ablation isolates the marginal value of structured ID encoding.
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
    "encoding_hash": {
        "description": "Hash encoding (IDs excluded, current default)",
        "categorical_strategy": "hash",
        "model_name": "ablation_encoding_hash.joblib",
    },
    "encoding_frequency": {
        "description": "Frequency encoding (Subject_ID + Lecturer_ID as count features)",
        "categorical_strategy": "frequency",
        "model_name": "ablation_encoding_frequency.joblib",
    },
    "encoding_target": {
        "description": "5-fold target encoding (Subject_ID + Lecturer_ID as smoothed mean)",
        "categorical_strategy": "target",
        "model_name": "ablation_encoding_target.joblib",
    },
}


def load_results() -> dict:
    if RESULTS_FILE.exists():
        return json.loads(RESULTS_FILE.read_text(encoding="utf-8"))
    return {}


def save_results(data: dict) -> None:
    RESULTS_FILE.parent.mkdir(parents=True, exist_ok=True)
    RESULTS_FILE.write_text(
        json.dumps(data, ensure_ascii=False, indent=2, default=str), encoding="utf-8"
    )


def run_scenario(name: str, config: dict) -> dict:
    print(f"\n{'='*80}\nRunning scenario: {name}\n{config['description']}\n{'='*80}")

    pipeline = TrainingPipeline(
        random_state=42,
        test_size=0.2,
        validation_size=0.2,
        group_split_by_student=True,
        categorical_strategy=config["categorical_strategy"],
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
        "categorical_strategy": config["categorical_strategy"],
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

    # Pick best
    best_name = None
    best_mae = float("inf")
    for name, r in ablation_results.items():
        if name == "best":
            continue
        m = r.get("metrics", {})
        mae = m.get("test_mae")
        if mae and mae < best_mae:
            best_mae = mae
            best_name = name
    if best_name:
        ablation_results["best"] = {
            "scenario": best_name,
            "test_mae": best_mae,
        }

    results["ablation_encoding"] = ablation_results
    save_results(results)

    print("\n" + "=" * 80)
    print("ABLATION SUMMARY (Encoding)")
    print("=" * 80)
    print(f"{'Scenario':<25} {'MAE':>8} {'RMSE':>8} {'R²':>8} {'features':>10}")
    print("-" * 80)
    for name in ("encoding_hash", "encoding_frequency", "encoding_target"):
        r = ablation_results.get(name, {})
        m = r.get("metrics", {})
        print(
            f"{name:<25} "
            f"{m.get('test_mae', 0):>8.4f} "
            f"{m.get('test_rmse', 0):>8.4f} "
            f"{m.get('test_r2', 0):>8.4f} "
            f"{r.get('n_features', 0):>10}"
        )
    if best_name:
        print("-" * 80)
        print(f"Best: {best_name} (MAE={best_mae:.4f})")
    print("=" * 80)
    print(f"\nResults saved to: {RESULTS_FILE}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
