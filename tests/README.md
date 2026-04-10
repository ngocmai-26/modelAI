# Testing Guide

This directory contains unit tests and integration tests for the ml_clo library.

## Test Structure

```text
tests/
├── __init__.py
├── conftest.py              # Shared fixtures and configuration
├── unit/                    # Unit tests
│   ├── test_data/          # Data module tests (loaders, mergers, preprocessors, encoders, validators)
│   ├── test_features/      # Features module tests (feature_builder, feature_groups)
│   ├── test_models/        # Models module tests (ensemble train/predict/save/load)
│   ├── test_xai/           # XAI module tests (SHAP explainer, batch, importance)
│   └── test_missing05.py   # Gap coverage: hash, anomaly, SHAP fallback, impact bands,
│                            #   set_weights, clear_cache, predict_with_uncertainty
├── integration/             # Integration tests
│   ├── test_pipelines/     # Pipeline integration tests (train, predict, analysis, save/load)
│   ├── test_prediction_differentiation.py
│   └── test_attendance_and_student_logic.py
└── fixtures/               # Test data fixtures
```

## Running Tests

### Run all tests
```bash
pytest
```

### Run only unit tests
```bash
pytest tests/unit/
```

### Run only integration tests
```bash
pytest tests/integration/
```

### Run specific test file
```bash
pytest tests/unit/test_data/test_loaders.py
```

### Run with coverage
```bash
pytest --cov=src/ml_clo --cov-report=html
```

### Run with verbose output
```bash
pytest -v
```

## Test Markers

Tests are marked with pytest markers:

- `@pytest.mark.unit` - Unit tests
- `@pytest.mark.integration` - Integration tests
- `@pytest.mark.slow` - Slow running tests
- `@pytest.mark.requires_data` - Tests that require data files

Run tests by marker:
```bash
pytest -m unit
pytest -m integration
pytest -m "not slow"
```

## Test Fixtures

Shared fixtures are defined in `conftest.py`:

- `sample_exam_scores` - Sample exam scores DataFrame
- `sample_conduct_scores` - Sample conduct scores DataFrame
- `sample_demographics` - Sample demographics DataFrame
- `sample_features` - Sample feature DataFrame for training
- `trained_model` - Pre-trained ensemble model

## Writing New Tests

1. Create test file in appropriate directory
2. Import necessary modules and fixtures
3. Write test classes and methods
4. Use pytest fixtures for test data
5. Add appropriate markers

Example:
```python
import pytest
from ml_clo.data.loaders import load_exam_scores

class TestLoadExamScores:
    def test_load_success(self):
        df = load_exam_scores("data/DiemTong.xlsx")
        assert df is not None
        assert len(df) > 0
```

## Test Coverage

Target coverage:
- Overall: >80%
- Critical paths: >90%

Run coverage report:
```bash
pytest --cov=src/ml_clo --cov-report=term-missing
```

## Notes

- Tests that require data files will skip if files are not found
- Integration tests may take longer to run
- Use `tmp_path` fixture for temporary files
- Use `pytest.skip()` for conditional test skipping

