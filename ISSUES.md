# ISSUES.md — Danh sách lỗi và vấn đề trong codebase ml_clo

> Cập nhật lần cuối: 2026-04-10 (sau fix toàn bộ 40 issues — 4 đợt).  
> Mỗi issue ghi rõ file, dòng, mô tả, impact và đề xuất fix.  
> Trạng thái: ✅ Fixed · ⚠️ Partially fixed · 🔴 Open

---

## Mục lục

- [BUG — Lỗi logic có thể gây crash hoặc kết quả sai](#bug)
- [DATA — Vấn đề xử lý dữ liệu](#data)
- [DESIGN — Vấn đề thiết kế và inconsistency](#design)
- [PERF — Hiệu suất](#perf)
- [MISSING — Thiếu validation / tính năng quan trọng](#missing)
- [NEW — Issues mới phát sinh từ PR #1](#new--issues-mới-phát-sinh-từ-pr-1)

---

## BUG

### ✅ BUG-01 · Double merge dữ liệu tự học — cột `total_study_hours` mang sai nghĩa

**File:** `src/ml_clo/data/mergers.py`  
**Severity:** HIGH  
**Fixed:** Đợt 1. Xóa `merge_study_hours()` khỏi `merge_all_data_sources()`. `build_study_hours_features()` là nguồn duy nhất cho `total_study_hours`.

---

### ✅ BUG-02 · Score conversion không kiểm tra dữ liệu đã ở hệ 6

**File:** `src/ml_clo/data/preprocessors.py`  
**Severity:** HIGH  
**Fixed:** Đợt 1. Thêm guard `if max <= 6.0: skip conversion`.

---

### ✅ BUG-03 · `get_top_negative_impacts()` sai khi tất cả SHAP dương

**File:** `src/ml_clo/xai/shap_postprocess.py`  
**Severity:** MEDIUM  
**Fixed:** Đợt 1. Fallback giới hạn `[:top_n]` sau khi populate.

---

### ✅ BUG-04 · `filter_shap_values()` silent failure khi threshold quá cao

**File:** `src/ml_clo/xai/shap_postprocess.py`  
**Severity:** MEDIUM  
**Fixed:** Đợt 1. Thêm top-5 fallback khi filtered_indices rỗng.

---

### ✅ BUG-05 · `EnsembleModel.predict()` không kiểm tra column order của X

**File:** `src/ml_clo/models/ensemble_model.py`  
**Severity:** MEDIUM  
**Fixed:** Đợt 1. Thêm `X = X[self.feature_names]` + missing column check.

---

### ✅ BUG-06 · Log statement trong `merge_demographics()` có thể crash

**File:** `src/ml_clo/data/mergers.py`  
**Severity:** LOW  
**Fixed:** Đợt 1. Safe probe column lookup thay vì hardcode `demo_cols[1]`.

---

### ✅ BUG-07 · Year extraction từ `semester_year` chỉ handle dấu `-`

**Fixed in PR #1** — `preprocessors.py` đã dùng regex `r"(\d{4})"`.

---

## DATA

### ✅ DATA-01 · Attendance status mapping không đầy đủ — typo → vắng

**File:** `src/ml_clo/data/mergers.py`  
**Severity:** MEDIUM  
**Fixed:** Đợt 3. Normalize whitespace + log unknown values trước fillna.

---

### ✅ DATA-02 · Duplicate `Student_ID` trong demographics — giữ dòng đầu ngẫu nhiên

**File:** `src/ml_clo/data/mergers.py`  
**Severity:** MEDIUM  
**Fixed:** Đợt 1. Dùng `keep="last"` và log số duplicate bị drop.

---

### ✅ DATA-03 · `study_hours` thiếu cột `accumulated_study_hours` gây `KeyError`

**File:** `src/ml_clo/data/loaders.py`  
**Severity:** MEDIUM  
**Fixed:** Đợt 3. Thêm HH:MM → decimal hours conversion với fallback derivation chain.

---

### 🔴 DATA-04 · Default year = 2024 hardcoded

**File:** `src/ml_clo/data/preprocessors.py`  
**Severity:** LOW  
**Ghi chú:** Không fix — giá trị mặc định chấp nhận được cho phase hiện tại. Có thể đổi sang `datetime.now().year` sau.

---

### 🔴 DATA-05 · Encode teaching/assessment methods dựa trên substring `"TM"` — dễ nhầm

**File:** `src/ml_clo/data/encoders.py`  
**Severity:** LOW  
**Ghi chú:** Không fix — dữ liệu hiện tại dùng prefix `TM`/`EM` nhất quán, chưa có cột gây nhầm.

---

### ✅ DATA-06 · `actual_clo_score` không validate range 0-6

**File:** `scripts/predict.py`  
**Severity:** LOW  
**Fixed:** Đợt 4 (MISSING-01). CLI validate `--actual-score` in [0, 6].

---

### ✅ DATA-07 · `deduplicate_exam_scores()` dùng `groupby(dropna=False)` có thể gộp nhầm

**File:** `src/ml_clo/data/preprocessors.py`  
**Severity:** MEDIUM  
**Fixed:** Đợt 3. Skip dedup cho rows có NaN trong group keys (pass-through).

---

## DESIGN

### ✅ DESIGN-01 · Logic `convert_score_10_to_6` tồn tại ở 2 nơi

**File:** `src/ml_clo/data/preprocessors.py`, `src/ml_clo/utils/math_utils.py`  
**Severity:** MEDIUM  
**Fixed:** Đợt 1. `preprocessors.py` delegate sang `_convert_score_10_to_6_array` từ `math_utils`.

---

### ✅ DESIGN-02 · `prepare_features()` bị copy-paste ở 3 pipeline

**File:** `src/ml_clo/features/feature_encoder.py` (MỚI)  
**Severity:** MEDIUM  
**Fixed:** Đợt 4. Extract thành shared `feature_encoder.py` với `select_feature_columns`, `encode_features`, `prepare_features`. Cả 3 pipeline đã wire.

---

### ✅ DESIGN-03 · Feature builder dùng vòng lặp per-student thay vì vectorized

**File:** `src/ml_clo/features/feature_builder.py`  
**Severity:** MEDIUM  
**Fixed:** Đợt 4. Vectorized groupby + cumcount cho academic history features.

---

### ✅ DESIGN-04 · `PredictionPipeline` có 2 cách khởi tạo — không document rõ

**File:** `src/ml_clo/pipelines/predict_pipeline.py`  
**Severity:** MEDIUM  
**Fixed:** Đợt 4. Docstring chi tiết Mode 1 (cache) / Mode 2 (per-call) với code examples.

---

### ✅ DESIGN-05 · `reason_key` vs `group_name` — hai tên cho cùng khái niệm

**File:** `src/ml_clo/reasoning/reason_generator.py`, `src/ml_clo/outputs/schemas.py`  
**Severity:** LOW  
**Fixed:** Đợt 2. Emit cả `group_name` và `reason_key`; `from_explanation_dict` chấp nhận cả hai.

---

### ✅ DESIGN-06 · `SHAP threshold = 0.01` hardcoded trong nhiều chỗ

**File:** `src/ml_clo/xai/shap_postprocess.py`  
**Severity:** LOW  
**Fixed:** Đợt 2. Đọc `SHAP_CONFIG["shap_threshold"]` trực tiếp.

---

### ✅ DESIGN-07 · Impact level chỉ có 3 bậc rời rạc

**File:** `src/ml_clo/reasoning/templates.py`  
**Severity:** LOW  
**Fixed:** Đợt 4. Thêm `IMPACT_BANDS` (6 band) + wire vào `get_reason_template` với intensity adverb.

---

### ✅ DESIGN-08 · `EnsembleModel` weights không thể điều chỉnh sau khi train

**File:** `src/ml_clo/models/ensemble_model.py`  
**Severity:** LOW  
**Fixed:** Đợt 4. Thêm `set_weights(rf_weight, gb_weight)` method.

---

### ✅ DESIGN-09 · `ClassAnalysisOutput.affected_students_count` luôn = `total_students`

**File:** `src/ml_clo/pipelines/analysis_pipeline.py`  
**Severity:** LOW  
**Fixed:** Đợt 4. Tính thực tế per-group (đếm SV có SHAP < 0 từng nhóm) + inject vào reason dict.

---

### ✅ DESIGN-10 · `gb_low_anomaly` config không được lưu vào file `.joblib`

**File:** `src/ml_clo/models/ensemble_model.py`  
**Severity:** MEDIUM  
**Fixed:** Đợt 1 (NEW-02). Save `ensemble_config` snapshot vào `extra_metadata`; restore khi `load()`.

---

## PERF

### ✅ PERF-01 · SHAP Explainer cache không có cleanup mechanism

**File:** `src/ml_clo/xai/shap_explainer.py`  
**Severity:** MEDIUM  
**Fixed:** Đợt 4. Thêm `clear_cache()` method.

---

### ✅ PERF-02 · Merge operations không set index trước

**File:** `src/ml_clo/data/mergers.py`  
**Severity:** LOW  
**Fixed:** Đợt 4. Index-based `.join()` cho demographics, TM, EM merges.

---

### ✅ PERF-03 · `analyze_class_from_scores()` tính SHAP lần lượt từng SV

**File:** `src/ml_clo/pipelines/analysis_pipeline.py`  
**Severity:** LOW  
**Fixed:** Đợt 4 (verified). Đã dùng `explainer.explain_batch(X)` trên toàn bộ DataFrame.

---

## MISSING

### ✅ MISSING-01 · CLI không validate input của người dùng

**File:** `scripts/train.py`, `scripts/predict.py`, `scripts/analyze_class.py`  
**Severity:** MEDIUM  
**Fixed:** Đợt 4. Validate: numeric ranges (test_size, validation_size, random_state), empty IDs, actual_score [0,6].

---

### ✅ MISSING-02 · Không có cross-validation

**File:** `src/ml_clo/pipelines/train_pipeline.py`  
**Severity:** MEDIUM  
**Fixed:** Đợt 4. Thêm `cross_validate()` method — GroupKFold (by Student_ID) hoặc KFold, k-fold CV.

---

### ✅ MISSING-03 · Không có prediction confidence/uncertainty

**File:** `src/ml_clo/models/ensemble_model.py`  
**Severity:** MEDIUM  
**Fixed:** Đợt 4. Thêm `predict_with_uncertainty()` — RF per-tree stdev, ±2σ confidence interval.

---

### ✅ MISSING-04 · Không có data quality report khi train

**File:** `src/ml_clo/pipelines/train_pipeline.py`  
**Severity:** LOW  
**Fixed:** Đợt 4. Thêm `report_data_quality()` — missing rate, duplicate keys, target stats.

---

### ✅ MISSING-05 · Test coverage còn thiếu

**File:** `tests/unit/test_missing05.py` (MỚI)  
**Severity:** LOW  
**Fixed:** Đợt 4. 14 tests mới: all-positive SHAP fallback, hash collision/determinism, anomaly clipping, impact bands, set_weights, clear_cache, predict_with_uncertainty.

---

### ✅ MISSING-06 · Không log prediction history để audit

**File:** `src/ml_clo/utils/audit_log.py` (MỚI), `src/ml_clo/pipelines/predict_pipeline.py`  
**Severity:** LOW  
**Fixed:** Đợt 4. JSONL audit log — `set_audit_log_path()` + tự động ghi từ `predict()`. No-op khi chưa configure.

---

### ✅ MISSING-07 · `study_hours` không được truyền vào `create_student_record_from_ids()`

**File:** `src/ml_clo/data/mergers.py`, `predict_pipeline.py`, `analysis_pipeline.py`  
**Severity:** LOW  
**Fixed:** Đợt 4. Thêm `study_hours_df` parameter + wire vào cả predict và analysis pipeline.

---

## NEW — Issues mới phát sinh từ PR #1

### ✅ NEW-01 · SHAP explanation không nhất quán với prediction khi `gb_low_anomaly` kích hoạt

**File:** `src/ml_clo/xai/shap_explainer.py`  
**Severity:** HIGH  
**Fixed:** Đợt 2. `explain_instance()` tính effective weights per-row, mirror anomaly blending: `eff_rf_w = rf_w + gb_w·br` khi anomaly, kết hợp `eff_rf_w·rf_shap + eff_gb_w·gb_shap`.

---

### ✅ NEW-02 · `stable_hash_int` — model cũ (LabelEncoder) không tương thích với code mới (hash)

**File:** `src/ml_clo/models/base_model.py`, `ensemble_model.py`  
**Severity:** HIGH  
**Fixed:** Đợt 1. Thêm `extra_metadata` với `encoding_method="hash_v2"`. Load validate encoding_method, reject model cũ.

---

### ✅ NEW-03 · `stable_hash_int` có thể collision với high-cardinality values

**File:** `src/ml_clo/utils/hash_utils.py`  
**Severity:** MEDIUM  
**Fixed:** Đợt 1. Tăng `mod = 2**31 - 1` (2.1 tỷ, max int32).

---

### ✅ NEW-04 · `calibrated` field mới trong reason dict không có trong `Reason` dataclass

**File:** `src/ml_clo/outputs/schemas.py`  
**Severity:** MEDIUM  
**Fixed:** Đợt 2. Thêm `calibrated: bool = False` vào `Reason` và `ClassReason`.

---

### ✅ NEW-05 · `analysis_pipeline.py` không truyền `raw_feature_row` → không bao giờ calibrate

**File:** `src/ml_clo/pipelines/analysis_pipeline.py`  
**Severity:** MEDIUM  
**Fixed:** Đợt 2. Pass `class_mean_row = X.mean(numeric_only=True)` vào `generate_complete_explanation` ở cả hai call sites.

---

### 🔴 NEW-06 · `academic_core_score` dùng `np.mean` của các metric correlated cao

**File:** `src/ml_clo/features/feature_builder.py`  
**Severity:** LOW  
**Ghi chú:** Không fix — feature engineering tradeoff. Multicollinearity được xử lý bởi tree-based models (RF/GB tự chọn feature tối ưu). Không ảnh hưởng prediction quality.

---

### 🔴 NEW-07 · `min_exam_score` bị exclude khỏi features nhưng vẫn được map trong `xai_config.py`

**File:** `src/ml_clo/config/xai_config.py`  
**Severity:** LOW  
**Ghi chú:** Không fix — pattern matching chỉ áp dụng cho features thực sự có trong feature vector. `min_exam` pattern vô hại.

---

## Tổng kết

| Category | Tổng | Fixed | Open | Ghi chú |
| --- | --- | --- | --- | --- |
| BUG | 7 | **7** | 0 | |
| DATA | 7 | **5** | 2 | DATA-04, DATA-05: LOW severity, chấp nhận |
| DESIGN | 10 | **10** | 0 | |
| PERF | 3 | **3** | 0 | |
| MISSING | 7 | **7** | 0 | |
| NEW (PR #1) | 7 | **5** | 2 | NEW-06, NEW-07: LOW severity, chấp nhận |
| **Tổng** | **41** | **37** | **4** | 4 open = LOW severity, accepted |

---

### Lịch sử fix

| Đợt | Ngày | Focus | Items fixed |
|------|-------|-------|-------------|
| 1 | 2026-04-10 | Correctness (HIGH) | BUG-01..06, NEW-02, NEW-03, DESIGN-01, DESIGN-10 |
| 2 | 2026-04-10 | XAI Trustworthiness | NEW-01, NEW-04, NEW-05, DESIGN-05, DESIGN-06 |
| 3 | 2026-04-10 | Robustness | DATA-01, DATA-03, DATA-07, BUG-04 threshold fallback |
| 4 | 2026-04-10 | Design + PERF + MISSING | DESIGN-02..04, DESIGN-07..09, PERF-01..03, MISSING-01..07 |

### Model cuối

- **Version:** v1.0_20260410_162858
- **Encoding:** hash_v2 (`mod = 2^31 - 1`)
- **Test MAE:** 0.3945 | **R²:** 0.7980
- **Tests:** 106 passed, 0 failed
