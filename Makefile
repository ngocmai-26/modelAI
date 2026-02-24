# ML CLO — dùng venv của dự án (.venv)
# Cách dùng: make venv install test | make build (để đóng gói cho dự án khác)

PYTHON := .venv/bin/python
PIP    := .venv/bin/pip
PYTEST := .venv/bin/pytest

.PHONY: venv install install-dev test build clean help

help:
	@echo "ML CLO — targets: venv, install, install-dev, test, build, clean"
	@echo "  make venv        — tạo .venv (chỉ lần đầu)"
	@echo "  make install     — cài package + dependencies"
	@echo "  make install-dev — cài thêm dev deps (pytest, black, ...)"
	@echo "  make test        — chạy pytest"
	@echo "  make build       — build sdist + wheel vào dist/ (để cài vào dự án khác)"
	@echo "  make clean       — xóa dist/, build/, *.egg-info"

venv:
	python3 -m venv .venv
	@echo "Đã tạo .venv. Chạy: source .venv/bin/activate (Linux/macOS) hoặc make install"

install: venv
	$(PIP) install -e .
	$(PIP) install -r requirements.txt

install-dev: install
	$(PIP) install -r requirements-dev.txt

test:
	$(PYTEST) tests/ -v --tb=short

test-cov:
	$(PYTEST) tests/ --cov=src/ml_clo --cov-report=term-missing -v --tb=short

# Build package để cài vào dự án khác (pip install dist/...)
# Dùng python/pip từ PATH (venv đang activate), không bắt buộc .venv
build:
	python -m pip install build --quiet
	python -m build --outdir dist
	@echo "Đã build xong. File trong dist/. Cài vào dự án khác: pip install dist/ml_clo-*.whl"

clean:
	rm -rf dist/ build/ src/*.egg-info *.egg-info
