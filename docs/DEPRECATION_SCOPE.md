# Phạm vi loại bỏ / thay thế (Bước 0)

**Ngày:** 2026-03-10  
**Phiên bản:** 1.0  
**Tham chiếu:** [IMPLEMENTATION_PLAN_NEW_REQUIREMENTS.md](IMPLEMENTATION_PLAN_NEW_REQUIREMENTS.md)

---

## Mục đích

Xác định rõ API, hàm và logic nào sẽ bị **loại bỏ** hoặc **thay thế** khi triển khai yêu cầu mới.

---

## Danh sách thay đổi

### 1. PredictionPipeline (`src/ml_clo/pipelines/predict_pipeline.py`)

| Thành phần | Hành động | Ghi chú |
|------------|-----------|---------|
| `load_student_data()` — bắt buộc (Student_ID, Subject_ID, Lecturer_ID) có trong DiemTong | **Thay thế** | Thêm fallback: nếu không có trong DiemTong → build từ nhân khẩu + PPGD/PPDG |
| `load_data_cache()` — exam_scores_path bắt buộc | **Sửa** | Cho phép cache chỉ từ demographics + PPGD/PPDG (không cần exam_scores khi predict cho SV/môn mới) |
| `predict()` — raise khi `_data_cache is None and exam_scores_path is None` | **Sửa** | Cho phép demographics_path + teaching_methods_path + assessment_methods_path thay thế |

### 2. AnalysisPipeline (`src/ml_clo/pipelines/analysis_pipeline.py`)

| Thành phần | Hành động | Ghi chú |
|------------|-----------|---------|
| `load_class_data()` — filter DiemTong theo subject+lecturer | **Deprecated** (Bước 4) | Dùng `analyze_class_from_scores()` |
| `analyze_class()` — đầu vào bắt buộc exam_scores_path | **Deprecated** (Bước 4) | Dùng `analyze_class_from_scores(clo_scores=...)` |

### 3. Data layer (`src/ml_clo/data/`)

| Thành phần | Hành động | Ghi chú |
|------------|-----------|---------|
| `create_training_dataset()` — yêu cầu exam_df có dữ liệu | **Bổ sung** | Thêm overload/hỗ trợ exam_df "ảo" (1 dòng, có thể không có exam_score) |
| Mergers | **Bổ sung** | Hàm mới `create_student_record_from_ids()` |

### 4. CLI (`scripts/`)

| Thành phần | Hành động | Ghi chú |
|------------|-----------|---------|
| `predict.py` — `--exam-scores` required | **Done** | Không còn bắt buộc; thêm `--actual-score` |
| `analyze_class.py` — `--exam-scores` | **Deprecated** | Chế độ chính: `--scores-file` |

---

## Không thay đổi (giữ nguyên)

- Training pipeline (vẫn dùng DiemTong cho train)
- XAI/SHAP, reason generator, solution mapper
- Model architecture (Ensemble, feature config)
- Output schemas (IndividualAnalysisOutput, ClassAnalysisOutput) — có thể thêm field `actual_clo_score`

---

## Migration note (cho user/backend)

Khi triển khai xong:

1. **Predict:** Có thể gọi không cần `exam_scores_path` nếu có `demographics_path`, `teaching_methods_path`, `assessment_methods_path`.
2. **Analyze class:** Chuyển sang dùng `--scores-file` thay vì `--exam-scores` (filter DiemTong).
