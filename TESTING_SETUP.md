# Testing Setup Guide

## Installing pytest

Pytest is required to run the test suite. Install it using:

```bash
# Activate your virtual environment first
source venv/bin/activate  # On macOS/Linux
# or
venv\Scripts\activate  # On Windows

# Install pytest and coverage tools
pip install pytest pytest-cov

# Or install all development dependencies
pip install -r requirements-dev.txt
```

## Running Tests

After installing pytest, you can run tests:

```bash
# Run all tests
pytest

# Run only unit tests
pytest tests/unit/

# Run only integration tests
pytest tests/integration/

# Run with verbose output
pytest -v

# Run with coverage report
pytest --cov=src/ml_clo --cov-report=html

# Run specific test file
pytest tests/unit/test_data/test_loaders.py

# Run tests matching a pattern
pytest -k "test_load"
```

## Troubleshooting

### "command not found: pytest"

**Solution:** Install pytest:
```bash
pip install pytest pytest-cov
```

### "ModuleNotFoundError: No module named 'ml_clo'"

**Solution:** Ensure you're running tests from the project root directory, and `src/` is in Python path. The `conftest.py` automatically adds `src/` to the path.

### Tests skip with "Data file not found"

**Solution:** Some tests require data files in the `data/` directory. These tests will automatically skip if files are not found. This is expected behavior.

## Test Structure

- `tests/unit/` - Unit tests for individual modules
- `tests/integration/` - Integration tests for pipelines
- `tests/conftest.py` - Shared fixtures and configuration
- `pytest.ini` - Pytest configuration

## Next Steps

1. Install pytest: `pip install pytest pytest-cov`
2. Run tests: `pytest`
3. Check coverage: `pytest --cov=src/ml_clo --cov-report=term-missing`


