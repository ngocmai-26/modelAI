# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

`ml_clo` is a Python library for predicting CLO (Course Learning Outcome) scores (scale 0–6) and providing XAI-based explanations using SHAP. It is designed for backend integration, not as a standalone app.

**Core capabilities:**
- Ensemble regression (Random Forest + Gradient Boosting) for CLO score prediction
- SHAP TreeExplainer with anomaly-aware blending for feature attribution
- Rule-based reason/solution generation in Vietnamese (no LLM), 6 impact bands
- Three pipelines: `TrainingPipeline` (k-fold CV, data quality report), `PredictionPipeline` (audit log), `AnalysisPipeline` (per-group affected count)
- Prediction uncertainty via RF per-tree variance (`predict_with_uncertainty`)
- Three categorical encoding strategies: `hash` (default, `stable_hash_int`, hash_v2, mod 2^31-1), `frequency`, `target` (5-fold mean with smoothing) — model artifact stores `encoding_method` + `fitted_encoders`, validated on load
- Optional survey integration (`survey_path`) and temporal attendance features (`enable_temporal_features`) — disabled by default, enabled per-experiment

## Environment Setup

The project uses `.venv` (not `venv/`) as the project virtual environment.

```bash
make venv        # create .venv (first time only)
make install     # install package + dependencies (editable)
make install-dev # install dev dependencies (pytest, black, etc.)
source .venv/bin/activate
```

Manual setup:
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e .
pip install -r requirements-dev.txt
```

## Commands

```bash
# Run all tests
make test
# or: pytest tests/ -v --tb=short

# Run with coverage
make test-cov
# or: pytest --cov=src/ml_clo --cov-report=term-missing

# Run only unit tests
pytest tests/unit/ -v -m unit

# Run a specific test file
pytest tests/unit/test_data/test_loaders.py -v

# Run by marker
pytest -m "not slow"

# Build distribution package
make build   # outputs to dist/

# Clean build artifacts
make clean
```

## CLI Scripts

All scripts require `PYTHONPATH` to include `src/` or the package installed:

```bash
export PYTHONPATH="${PYTHONPATH}:$(pwd)/src"

# Train (full dataset, baseline encoding)
python scripts/train.py \
  --exam-scores data/DiemTong.xlsx \
  --output models/model.joblib \
  --conduct-scores data/diemrenluyen.xlsx \
  --demographics data/nhankhau.xlsx \
  --teaching-methods data/PPGDfull.xlsx \
  --assessment-methods data/PPDGfull.xlsx \
  --study-hours data/tuhoc.xlsx \
  --attendance "data/Dữ liệu điểm danh Khoa FIRA.xlsx"

# Optional: include survey responses (advisor feedback ablation)
#   --survey "data/Khảo sát các yếu tố cá nhân...xlsx"

# Predict (exam-scores is optional; demographics + PPGD + PPDG are sufficient)
python scripts/predict.py \
  --model models/model.joblib \
  --student-id 19050006 --subject-id INF0823 --lecturer-id 90316 \
  --demographics data/nhankhau.xlsx \
  --teaching-methods data/PPGDfull.xlsx \
  --assessment-methods data/PPDGfull.xlsx

# Analyze class (primary mode: --scores-file; --exam-scores is deprecated)
python scripts/analyze_class.py \
  --model models/model.joblib \
  --subject-id INF0823 --lecturer-id 90316 \
  --scores-file data/clo_scores.csv --output result.json

# Ablation studies (advisor feedback validation)
python scripts/run_ablation_survey.py     # LMS vs LMS+Survey
python scripts/run_ablation_temporal.py   # +slope/volatility/late_streak
python scripts/run_ablation_encoding.py   # hash vs frequency vs target
```

## Architecture

```
src/ml_clo/
├── pipelines/          # TrainingPipeline (cross_validate, report_data_quality),
│                       #   PredictionPipeline (audit log), AnalysisPipeline (affected_students_count)
├── data/               # loaders.py (load_survey_responses), preprocessors.py, encoders.py,
│                       #   validators.py, mergers.py (merge_survey_responses),
│                       #   survey_preprocessor.py (Likert/multi-hot/range encoding)
├── features/           # feature_builder.py (attendance_history_df hook), feature_groups.py,
│                       #   feature_encoder.py (categorical_strategy, include_id_features),
│                       #   temporal_features.py (slope/volatility per student-year),
│                       #   categorical_encoder.py (FrequencyEncoder, TargetEncoder)
├── models/             # base_model.py (extra_metadata), ensemble_model.py (set_weights,
│                       #   predict_with_uncertainty, gb_low_anomaly, fitted_encoders),
│                       #   model_evaluator.py
├── xai/                # shap_explainer.py (anomaly-aware, clear_cache), shap_postprocess.py
├── reasoning/          # reason_generator.py, solution_mapper.py, templates.py (IMPACT_BANDS)
├── outputs/            # schemas.py (IndividualAnalysisOutput, ClassAnalysisOutput, calibrated)
├── config/             # feature_config.py, model_config.py, xai_config.py
└── utils/              # logger.py, exceptions.py, math_utils.py, io_utils.py,
                        #   hash_utils.py (stable_hash_int), audit_log.py
```

**Data flow:**
1. `data/` modules load Excel files and normalize column names; `survey_preprocessor.py` encodes the 42-column raw survey into ~30 ordinal/multi-hot features
2. `features/feature_encoder.py` (shared) selects columns and applies the chosen `categorical_strategy` (hash by default; target/frequency for ID columns when `include_id_features=True`)
3. `features/feature_builder.py` computes aggregate features (conduct trends, pass rates, etc.) via vectorized groupby; optional `temporal_features.py` derives per-(student, year) attendance slope/volatility/late-streak
4. `models/ensemble_model.py` trains weighted RF + GB ensemble with `gb_low_anomaly` blending, saves as `.joblib` with `extra_metadata` (encoding_method, fitted_encoders, ensemble_config snapshot)
5. `xai/shap_explainer.py` computes anomaly-aware SHAP values (effective weights match prediction blending); `shap_postprocess.py` groups into 7 pedagogical categories: Tự học, Chuyên cần, Rèn luyện, Học lực, Giảng dạy, Đánh giá, Cá nhân
6. `reasoning/` maps SHAP groups → Vietnamese reason text (6 impact bands) + actionable solutions (rule-based, calibrated against raw feature values)
7. `outputs/schemas.py` serializes to `IndividualAnalysisOutput` or `ClassAnalysisOutput` (with `calibrated` flag, `affected_students_count`)

**Key design decisions:**
- All training data uses CLO scale (0–6). Exam scores in hệ 10 are converted: `CLO_6 = Score_10 / 10 × 6` (guarded: skip if max ≤ 6)
- Model is not bundled — backends receive the `.joblib` path via `model_path` parameter
- Models self-validate `encoding_method` on load (`hash_v2`, `frequency_v1`, `target_v1`) — incompatible models are rejected with clear retrain message
- `analyze_class_from_scores()` is the primary class analysis API; the older `--exam-scores` filter is deprecated
- `--exam-scores` is optional for prediction; fallback uses `create_student_record_from_ids` (with study_hours) when student/subject/lecturer not in DiemTong
- `predict_with_uncertainty()` provides RF per-tree stdev as confidence proxy
- Prediction audit trail via `utils/audit_log.py` (opt-in JSONL)
- **Empirically validated** baseline (76 features, hash encoding, no survey, no temporal) — see [docs/EXPERIMENTS_LOG.md](docs/EXPERIMENTS_LOG.md). Survey, temporal, and structured ID encoding were tested via ablation and found to NOT improve MAE on current data coverage.

## Data Files

Place Excel files in `data/`. Expected files:
- `DiemTong.xlsx` — exam scores (hệ 10, converted to hệ 6 internally)
- `nhankhau.xlsx` — student demographics
- `PPGDfull.xlsx` — teaching methods
- `PPDGfull.xlsx` — assessment methods
- `diemrenluyen.xlsx` — conduct scores
- `tuhoc.xlsx` — self-study hours (granularity: year + semester only — no weekly timestamps)
- `Dữ liệu điểm danh Khoa FIRA.xlsx` — attendance data (covers ~28% of records)
- `Khảo sát các yếu tố cá nhân...xlsx` — student survey (280 responses; only 76 MSSVs overlap exam scores; ablation found NOT to improve model)

## Testing

Test markers: `unit`, `integration`, `slow`, `requires_data`

Tests that need data files auto-skip if files are missing. Shared fixtures (synthetic DataFrames) are in `tests/conftest.py`. Integration tests cover full pipeline round-trips including model save/load.

## Code Conventions

- Type hints on all functions; Google-style docstrings on public APIs
- PEP 8, max line length 100
- No `print()` or `input()` in library code — use `src/ml_clo/utils/logger.py`
- No hard-coded paths; no LLM calls; no `.ipynb` in core code
- Fixed `random_state=42` for reproducibility
- Custom exceptions are in `src/ml_clo/utils/exceptions.py` (`DataValidationError`, `ModelLoadError`, `PredictionError`)
