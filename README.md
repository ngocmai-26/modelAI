# ML CLO — CLO Prediction & Explainable AI

Thư viện Python để dự đoán điểm CLO (Course Learning Outcome, thang 0–6) và cung cấp giải thích dựa trên XAI (SHAP). Dùng cho tích hợp backend, không phải ứng dụng độc lập.

## Tính năng

- **Regression**: Ensemble Random Forest + Gradient Boosting, dự đoán điểm CLO 0–6
- **XAI**: SHAP TreeExplainer, lý do ảnh hưởng theo nhóm sư phạm (Tự học, Chuyên cần, Rèn luyện, Học lực, Giảng dạy, Đánh giá, Cá nhân)
- **Lý do & giải pháp**: Rule-based (không dùng LLM), tiếng Việt, gợi ý hành động
- **Pipeline**: Train, dự đoán cá nhân, phân tích lớp (aggregate SHAP, lưu dữ liệu cho retrain)

## Yêu cầu

- Python >= 3.10
- Dữ liệu: Excel (điểm thi, điểm rèn luyện, nhân khẩu, PPGD/PPDG, tự học, điểm danh) — xem `docs/data_model.md`

## Cài đặt

**Luôn dùng virtual environment của dự án (`.venv`).**

**Cách nhanh (Linux/macOS, có `make`):**

```bash
cd modelAI
make venv       # tạo .venv (chỉ lần đầu)
make install    # cài package + dependencies
source .venv/bin/activate   # kích hoạt cho các lệnh sau
```

**Cách thủ công:**

```bash
# Clone và vào thư mục project
cd modelAI

# Tạo venv của dự án (chỉ lần đầu)
python3 -m venv .venv

# Kích hoạt venv — bắt buộc trước mọi lệnh pip/python trong project
source .venv/bin/activate   # Linux/macOS
# .venv\Scripts\activate    # Windows

# Cài đặt package (editable)
pip install -e .

# Hoặc cài dependency trực tiếp
pip install -r requirements.txt
```

Cài thêm dependency cho phát triển (test, format, type check):

```bash
# Đảm bảo đã activate .venv
pip install -e ".[dev]"
# hoặc
pip install -r requirements-dev.txt
```

## Cấu trúc project

```
modelAI/
├── .venv/                # Virtual environment của dự án (luôn dùng khi chạy pip/pytest/scripts)
├── src/ml_clo/           # Thư viện chính
│   ├── data/             # Loaders, preprocessors, encoders, validators, mergers
│   ├── features/         # Feature groups, feature builder
│   ├── config/           # feature_config, model_config, xai_config
│   ├── models/           # Base, ensemble, evaluator
│   ├── xai/              # SHAP explainer, postprocess
│   ├── reasoning/        # Templates, solution mapper, reason generator
│   ├── outputs/          # Schemas (IndividualAnalysisOutput, ClassAnalysisOutput)
│   ├── pipelines/        # Train, Predict, Analysis
│   └── utils/            # Logger, exceptions, math_utils, io_utils
├── scripts/              # CLI
│   ├── train.py
│   ├── predict.py
│   └── analyze_class.py
├── data/                 # Dữ liệu Excel (đặt file vào đây)
├── models/               # Model đã train (.joblib)
├── tests/
├── docs/
├── Makefile              # make venv, make install, make test
├── pyproject.toml
├── requirements.txt
└── README.md
```

## Sử dụng

**Mọi lệnh dưới đây chạy sau khi đã `source .venv/bin/activate` (hoặc activate `.venv` trên Windows).**

### 1. Train model

Cần ít nhất file điểm thi (hệ 10 sẽ được chuyển sang hệ 6 trong pipeline).

```bash
# Từ thư mục gốc project (đã activate .venv)
export PYTHONPATH="${PYTHONPATH}:$(pwd)/src"

# Chỉ điểm thi
python scripts/train.py --exam-scores data/DiemTong.xlsx --output models/model.joblib

# Đủ nguồn dữ liệu
python scripts/train.py \
  --exam-scores data/DiemTong.xlsx \
  --conduct-scores data/diemrenluyen.xlsx \
  --demographics data/nhankhau.xlsx \
  --teaching-methods data/PPGDfull.xlsx \
  --assessment-methods data/PPDGfull.xlsx \
  --study-hours data/tuhoc.xlsx \
  --output models/model.joblib
```

### 2. Dự đoán cá nhân (có XAI)

```bash
python scripts/predict.py \
  --model models/model.joblib \
  --student-id 19050006 \
  --subject-id INF0823 \
  --lecturer-id 90316 \
  --exam-scores data/DiemTong.xlsx
```

Output JSON: `predicted_clo`, lý do, giải pháp (theo schema `IndividualAnalysisOutput`).

### 3. Phân tích lớp

```bash
python scripts/analyze_class.py \
  --model models/model.joblib \
  --subject-id INF0823 \
  --lecturer-id 90316 \
  --exam-scores data/DiemTong.xlsx \
  --conduct-scores data/diemrenluyen.xlsx \
  --demographics data/nhankhau.xlsx \
  --teaching-methods data/PPGDfull.xlsx \
  --assessment-methods data/PPDGfull.xlsx \
  --study-hours data/tuhoc.xlsx \
```

Có thể thêm `--actual-scores` để lưu điểm thực cho retraining. Output: `ClassAnalysisOutput`.

### 4. Dùng như thư viện (backend)

**Import chính** — pipelines (train, predict, phân tích lớp):

```python
from ml_clo import TrainingPipeline, PredictionPipeline, AnalysisPipeline
# hoặc
from ml_clo.pipelines import TrainingPipeline, PredictionPipeline, AnalysisPipeline
```

**Import thêm** — schema output (typings, serialize sang JSON):

```python
from ml_clo.outputs.schemas import IndividualAnalysisOutput, ClassAnalysisOutput, Reason
```

**Ví dụ:**

```python
from ml_clo import TrainingPipeline, PredictionPipeline, AnalysisPipeline

# Train
pipeline = TrainingPipeline()
pipeline.run(
    exam_scores_path="data/DiemTong.xlsx",
    output_path="models/model.joblib",
)

# Dự đoán cá nhân — khuyến nghị: truyền data paths khi khởi tạo, sau đó chỉ cần ID
pred_pipeline = PredictionPipeline(
    model_path="models/model.joblib",
    exam_scores_path="data/DiemTong.xlsx",  # (và các path khác nếu có)
)
result = pred_pipeline.predict(
    student_id="19050006",
    subject_id="INF0823",
    lecturer_id="90316",
)
# result: IndividualAnalysisOutput — dùng result.to_dict() hoặc result.to_json() để trả API
# (Nếu không cache: có thể gọi predict(..., exam_scores_path="...") mỗi lần.)

# Phân tích lớp
analysis_pipeline = AnalysisPipeline(model_path="models/model.joblib")
class_result = analysis_pipeline.analyze_class(
    subject_id="INF0823",
    lecturer_id="90316",
    exam_scores_path="data/DiemTong.xlsx",
)
# class_result: ClassAnalysisOutput — dùng class_result.to_dict() / .to_json() cho API
```

## Test

```bash
# Đảm bảo đã activate .venv, hoặc dùng Makefile (tự dùng .venv)
make test

# Hoặc thủ công
pytest tests/ -v --tb=short

# Chỉ unit
pytest tests/unit -v -m unit

# Có coverage
make test-cov
# hoặc
pytest --cov=src/ml_clo --cov-report=term-missing
```

Chi tiết test: xem `tests/README.md` và mục Testing trong `.cursor/rules.txt`.

## Tài liệu

- `docs/model_requirements.md` — Yêu cầu model
- `docs/data_model.md` — Cấu trúc dữ liệu
- `.cursor/rules.txt` — Quy tắc project, testing, plan
- `plan.txt` — Kế hoạch phát triển từng giai đoạn

## Dùng trong dự án khác (import như module)

**Lưu ý:** Package `ml_clo` **không kèm** file model đã train (.joblib). Backend cần tự có file model (train trước bằng `TrainingPipeline` hoặc copy từ nơi khác) và truyền đường dẫn vào `PredictionPipeline(model_path=...)` / `AnalysisPipeline(model_path=...)`. Model có thể đặt trong repo backend, object storage (S3, GCS), hoặc biến môi trường (ví dụ `MODEL_PATH`).

Có hai cách để dự án khác có thể `import ml_clo`:

### Cách 1: Cài trực tiếp từ thư mục nguồn (editable, khuyến nghị khi đang phát triển)

Trong dự án đích (nơi cần dùng ml_clo), tạo/activate venv rồi:

```bash
# Đường dẫn tuyệt đối hoặc tương đối tới thư mục modelAI
pip install -e /path/to/modelAI
```

Sau đó trong code:

```python
from ml_clo.pipelines import TrainingPipeline, PredictionPipeline, AnalysisPipeline
from ml_clo import __version__
```

### Cách 2: Build rồi cài file .whl / .tar.gz (phù hợp khi gửi cho người khác hoặc CI)

Trong thư mục **modelAI** (đã có .venv và đã `make install`):

```bash
source .venv/bin/activate
make build
# Hoặc: pip install build && python -m build --outdir dist
```

Sẽ tạo thư mục `dist/` chứa:
- `ml_clo-0.1.0-py3-none-any.whl` (wheel)
- `ml_clo-0.1.0.tar.gz` (source distribution)

Trong **dự án khác**:

```bash
pip install /path/to/modelAI/dist/ml_clo-0.1.0-py3-none-any.whl
# hoặc
pip install /path/to/modelAI/dist/ml_clo-0.1.0.tar.gz
```

Rồi `import ml_clo` như trên. Có thể copy thư mục `dist/` hoặc đưa lên server/file nội bộ rồi `pip install <url-hoặc-path>`.

---

## Tiếp theo

- Đặt file Excel vào `data/`, train model: `python scripts/train.py --exam-scores data/DiemTong.xlsx --output models/model.joblib`
- Dự đoán cá nhân / phân tích lớp: dùng `scripts/predict.py`, `scripts/analyze_class.py` (xem phần Sử dụng trên).
- Tích hợp backend: import `ml_clo.pipelines` và dùng `TrainingPipeline`, `PredictionPipeline`, `AnalysisPipeline`.

## License

MIT — xem file [LICENSE](LICENSE).
