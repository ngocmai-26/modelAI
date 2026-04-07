# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

`ml_clo` is a Python library for predicting CLO (Course Learning Outcome) scores (scale 0–6) and providing XAI-based explanations using SHAP. It is designed for backend integration, not as a standalone app.

**Core capabilities:**
- Ensemble regression (Random Forest + Gradient Boosting) for CLO score prediction
- SHAP TreeExplainer for feature attribution
- Rule-based reason/solution generation in Vietnamese (no LLM)
- Three pipelines: `TrainingPipeline`, `PredictionPipeline`, `AnalysisPipeline`

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

# Train
python scripts/train.py --exam-scores data/DiemTong.xlsx --output models/model.joblib

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
```

## Architecture

```
src/ml_clo/
├── pipelines/          # Entry points: TrainingPipeline, PredictionPipeline, AnalysisPipeline
├── data/               # loaders.py, preprocessors.py, encoders.py, validators.py, mergers.py
├── features/           # feature_builder.py, feature_groups.py
├── models/             # base_model.py, ensemble_model.py, model_evaluator.py
├── xai/                # shap_explainer.py, shap_postprocess.py
├── reasoning/          # reason_generator.py, solution_mapper.py, templates.py
├── outputs/            # schemas.py (IndividualAnalysisOutput, ClassAnalysisOutput)
├── config/             # feature_config.py, model_config.py, xai_config.py
└── utils/              # logger.py, exceptions.py, math_utils.py, io_utils.py
```

**Data flow:**
1. `data/` modules load Excel files and normalize column names
2. `features/feature_builder.py` computes aggregate features (conduct trends, pass rates, etc.)
3. `models/ensemble_model.py` trains weighted RF + GB ensemble, saves as `.joblib`
4. `xai/shap_explainer.py` computes SHAP values; `shap_postprocess.py` groups them into 7 pedagogical categories: Tự học, Chuyên cần, Rèn luyện, Học lực, Giảng dạy, Đánh giá, Cá nhân
5. `reasoning/` maps SHAP groups → Vietnamese reason text + actionable solutions (rule-based)
6. `outputs/schemas.py` serializes to `IndividualAnalysisOutput` or `ClassAnalysisOutput`

**Key design decisions:**
- All training data uses CLO scale (0–6). Exam scores in hệ 10 are converted: `CLO_6 = Score_10 / 10 × 6`
- Model is not bundled — backends receive the `.joblib` path via `model_path` parameter
- `analyze_class_from_scores()` is the primary class analysis API; the older `--exam-scores` filter is deprecated
- `--exam-scores` is optional for prediction; fallback uses `create_student_record_from_ids` when student/subject/lecturer not in DiemTong

## Data Files

Place Excel files in `data/`. Expected files:
- `DiemTong.xlsx` — exam scores (hệ 10, converted to hệ 6 internally)
- `nhankhau.xlsx` — student demographics
- `PPGDfull.xlsx` — teaching methods
- `PPDGfull.xlsx` — assessment methods
- `diemrenluyen.xlsx` — conduct scores
- `tuhoc.xlsx` — self-study hours
- `Dữ liệu điểm danh Khoa FIRA.xlsx` — attendance data

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
