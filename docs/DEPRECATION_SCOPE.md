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

## Thay đổi thêm (bug fix round 2026-04-10)

Sau triển khai yêu cầu mới, các thành phần sau đã được cập nhật:

| Thành phần | Hành động | Ghi chú |
| --- | --- | --- |
| `LabelEncoder` trong 3 pipeline | **Thay thế** | Dùng `stable_hash_int` (hash_v2) qua shared `feature_encoder.py` |
| `prepare_features()` — 3 bản copy | **Hợp nhất** | Shared `features/feature_encoder.py` |
| `EnsembleModel` — weights cố định | **Bổ sung** | `set_weights()`, `predict_with_uncertainty()` |
| `EnsembleSHAPExplainer` — cache vĩnh viễn | **Bổ sung** | `clear_cache()` |
| `base_model.py` — không lưu metadata | **Bổ sung** | `extra_metadata` (encoding_method, ensemble_config) |
| `templates.py` — 3 impact levels | **Thay thế** | 6 `IMPACT_BANDS` chi tiết |
| `create_student_record_from_ids()` | **Bổ sung** | Thêm `study_hours_df` parameter |
| `TrainingPipeline` | **Bổ sung** | `cross_validate()`, `report_data_quality()` |
| `PredictionPipeline` | **Bổ sung** | Audit log via `utils/audit_log.py` |
| `AnalysisPipeline` | **Bổ sung** | Per-group `affected_students_count` thực tế |
| CLI scripts | **Bổ sung** | Input validation (ranges, empty IDs, actual_score) |

---

## Không thay đổi (giữ nguyên)

- Training pipeline data flow (vẫn dùng DiemTong cho train)
- Model architecture (Ensemble RF+GB, feature config)
- Output schemas interface (IndividualAnalysisOutput, ClassAnalysisOutput) — thêm field `calibrated`, `affected_students_count` nhưng backward compatible

---

## Migration note (cho user/backend)

Khi triển khai xong:

1. **Predict:** Có thể gọi không cần `exam_scores_path` nếu có `demographics_path`, `teaching_methods_path`, `assessment_methods_path`.
2. **Analyze class:** Chuyển sang dùng `--scores-file` thay vì `--exam-scores` (filter DiemTong).
3. **Model cũ:** Model train trước hash_v2 sẽ bị reject khi load — cần retrain bằng pipeline mới.
4. **Audit log:** Opt-in — gọi `set_audit_log_path("logs/predictions.jsonl")` trước khi predict.
5. **Uncertainty:** Dùng `model.predict_with_uncertainty(X)` để lấy confidence interval.
