# ISSUES.md — Danh sách lỗi và vấn đề trong codebase ml_clo

> Cập nhật lần cuối: 2026-04-07 (sau PR #1 từ ngocmai-26/test).  
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

### 🔴 BUG-01 · Double merge dữ liệu tự học — cột `total_study_hours` mang sai nghĩa

**File:** `src/ml_clo/data/mergers.py:356-379` và `src/ml_clo/features/feature_builder.py:322-335`  
**Severity:** HIGH

`merge_study_hours()` và `build_study_hours_features()` đều tạo cột `total_study_hours` nhưng ngữ nghĩa khác nhau:

- `merge_study_hours()` → group by `Student_ID + year` → `total_study_hours` = giờ tự học **trong 1 năm**
- `build_study_hours_features()` → group by `Student_ID` → `total_study_hours` = tổng giờ **toàn thời gian**

Khi `build_study_hours_features()` merge vào df đã có `total_study_hours`, pandas dùng suffix `_total` → cột đúng bị đổi tên thành `total_study_hours_total`, model train với cột sai nghĩa (per-year).

**Fix:** Xóa `merge_study_hours()` khỏi `create_training_dataset()` trong `mergers.py:589`.

---

### 🔴 BUG-02 · Score conversion không kiểm tra dữ liệu đã ở hệ 6

**File:** `src/ml_clo/data/preprocessors.py:239`  
**Severity:** HIGH

`convert_score_10_to_6()` áp dụng `/ 10.0 * 6.0` vô điều kiện. Nếu file điểm đã ở hệ 6 (dữ liệu tương lai từ hệ thống), điểm 5.0 → 3.0 → label sai.

```python
df[output_col] = df[score_column] / 10.0 * 6.0  # Không kiểm tra max value
```

**Fix:**

```python
if df[score_column].max() <= 6.0:
    logger.warning("Scores appear to already be on 0-6 scale, skipping conversion")
    df[output_col] = df[score_column]
else:
    df[output_col] = df[score_column] / 10.0 * 6.0
```

---

### 🔴 BUG-03 · `get_top_negative_impacts()` sai khi tất cả SHAP dương

**File:** `src/ml_clo/xai/shap_postprocess.py:161-238`  
**Severity:** MEDIUM

Khi tất cả SHAP > 0, `group_impacts` trống. Fallback lặp qua tất cả nhóm nhưng không giới hạn `top_n` → trả về tất cả nhóm thay vì top N.

```python
if not group_impacts:
    for group_name, shap_vals in grouped_shap.items():
        group_impacts[group_name] = mean_shap  # Không slice [:top_n]
```

**Fix:** Sort và slice `[:top_n]` sau khi populate fallback.

---

### 🔴 BUG-04 · `filter_shap_values()` silent failure khi threshold quá cao

**File:** `src/ml_clo/xai/shap_postprocess.py:44`  
**Severity:** MEDIUM

Nếu tất cả |SHAP| < 0.01, `filtered_indices = []` → `reasons=[]` không có cảnh báo.

**Fix:**

```python
if len(filtered_indices) == 0:
    logger.warning("No SHAP values above threshold; using top-5 by absolute value")
    filtered_indices = np.argsort(mean_abs_shap)[-5:]
```

---

### 🔴 BUG-05 · `EnsembleModel.predict()` không kiểm tra column order của X

**File:** `src/ml_clo/models/ensemble_model.py:218`  
**Severity:** MEDIUM

Model không validate columns khớp với `self.feature_names`. Nếu X có cột sai thứ tự, sklearn dùng index nên predict với feature sai vị trí mà không báo lỗi.

**Fix:** Thêm `X = X[self.feature_names]` ở đầu `predict()`.

---

### 🔴 BUG-06 · Log statement trong `merge_demographics()` có thể crash

**File:** `src/ml_clo/data/mergers.py:217`  
**Severity:** LOW

Nếu không có cột demographic nào ngoài Student_ID, `demo_cols[1]` có thể không tồn tại trong `merged` → `KeyError`.

**Fix:** Bọc trong try/except hoặc check `demo_cols[1] in merged.columns` trước.

---

### ✅ BUG-07 · Year extraction từ `semester_year` chỉ handle dấu `-`

**Fixed in PR #1** — `preprocessors.py` đã dùng regex `r"(\d{4})"` để extract 4 chữ số đầu tiên, handle cả `"2019-2020"`, `"2019/2020"`, v.v.

---

## DATA

### 🔴 DATA-01 · Attendance status mapping không đầy đủ — typo → vắng

**File:** `src/ml_clo/data/mergers.py:465-472`  
**Severity:** MEDIUM

Giá trị ngoài `["Sớm", "Có", "Trễ", "Vắng", "Phép"]` (typo, khoảng trắng thừa) bị `fillna(0.0)` = vắng → `attendance_rate` thấp hơn thực tế.

**Fix:** Log warning cho giá trị không nhận dạng được trước khi fillna.

---

### 🔴 DATA-02 · Duplicate `Student_ID` trong demographics — giữ dòng đầu ngẫu nhiên

**File:** `src/ml_clo/data/mergers.py:205`  
**Severity:** MEDIUM

`drop_duplicates(subset=["Student_ID"])` giữ dòng đầu, bỏ dòng mới nhất mà không log.

**Fix:** Dùng `keep="last"` và log số lượng duplicate bị drop.

---

### 🔴 DATA-03 · `study_hours` thiếu cột `accumulated_study_hours` gây `KeyError`

**File:** `src/ml_clo/data/loaders.py:283-308`  
**Severity:** MEDIUM

`load_study_hours()` chỉ log warning, không raise. Downstream `build_study_hours_features()` sẽ `KeyError: 'accumulated_study_hours'` nếu file chỉ có cột `time` (HH:MM).

**Fix:** Thêm conversion HH:MM → float hours trong loader.

---

### 🔴 DATA-04 · Default year = 2024 hardcoded

**File:** `src/ml_clo/data/preprocessors.py:424`  
**Severity:** LOW

Dữ liệu năm 2022-2023 có year NaN sẽ được gán 2024 → merge với conduct/attendance fail.

**Fix:** Dùng `datetime.now().year` hoặc config.

---

### 🔴 DATA-05 · Encode teaching/assessment methods dựa trên substring `"TM"` — dễ nhầm

**File:** `src/ml_clo/data/encoders.py:46-47`  
**Severity:** LOW

`"TM" in col` match nhầm cột `"TOTAL_METHOD"`, bỏ sót cột `"teaching_method_1"`.

**Fix:** Dùng regex `r'^TM\d+$'`.

---

### 🔴 DATA-06 · `actual_clo_score` không validate range 0-6

**File:** `src/ml_clo/pipelines/predict_pipeline.py:418`  
**Severity:** LOW

User truyền `--actual-score 8.5` (nhầm hệ 10) không bị chặn.

**Fix:** Validate trong `scripts/predict.py` và trong schema.

---

### 🔴 DATA-07 · `deduplicate_exam_scores()` dùng `groupby(dropna=False)` có thể gộp nhầm

**File:** `src/ml_clo/data/preprocessors.py:445-504` *(mới — PR #1)*  
**Severity:** MEDIUM

`deduplicate_exam_scores()` dùng `groupby(subset, dropna=False)`. Khi `year` là NaN cho nhiều record khác nhau, tất cả đều được gom vào một group `(sid, subj, lec, NaN)` và điểm bị average thành một dòng không liên quan. Đây là silent data corruption.

```python
out = df.groupby(subset, as_index=False, dropna=False).agg(agg)
# year=NaN của các SV/môn khác nhau có thể bị average cùng nhau
```

**Fix:** Drop các dòng `year=NaN` trước khi dedup, hoặc fill year trước khi gọi `deduplicate_exam_scores`.

---

## DESIGN

### 🔴 DESIGN-01 · Logic `convert_score_10_to_6` tồn tại ở 2 nơi

**File:** `src/ml_clo/utils/math_utils.py:19` và `src/ml_clo/data/preprocessors.py:193`  
**Severity:** MEDIUM

Cùng công thức ở hai module. Bug fix phải sửa 2 chỗ.

**Fix:** `preprocessors.py` gọi `math_utils.convert_score_10_to_6()`.

---

### ⚠️ DESIGN-02 · `prepare_features()` bị copy-paste ở 3 pipeline

**File:** `train_pipeline.py:210`, `predict_pipeline.py:353`, `analysis_pipeline.py:225`  
**Severity:** MEDIUM

PR #1 đã đồng bộ logic `min_exam_score` exclusion ở cả 3 chỗ (thêm `feature_cols = [c for c in feature_cols if c != "min_exam_score"]`), nhưng vẫn là 3 bản copy. Bất kỳ thay đổi logic nào vẫn phải sửa 3 nơi.

**Fix:** Extract thành shared `feature_encoder.py`.

---

### 🔴 DESIGN-03 · Feature builder dùng vòng lặp per-student thay vì vectorized

**File:** `src/ml_clo/features/feature_builder.py:65-106`  
**Severity:** MEDIUM

O(N) loop per student. Với 10k+ SV, rất chậm.

**Fix:** Dùng `groupby().agg()`.

---

### 🔴 DESIGN-04 · `PredictionPipeline` có 2 cách khởi tạo — không document rõ

**File:** `src/ml_clo/pipelines/predict_pipeline.py:62-111`  
**Severity:** MEDIUM

Cách 2 (per-call) không được test đầy đủ và thiếu ví dụ trong README.

---

### 🔴 DESIGN-05 · `reason_key` vs `group_name` — hai tên cho cùng khái niệm

**File:** `src/ml_clo/outputs/schemas.py` và `src/ml_clo/reasoning/reason_generator.py`  
**Severity:** LOW

`reason_generator.py` dùng `"group_name"`, `Reason` dataclass dùng `reason_key`. Map thủ công trong `from_explanation_dict()`.

---

### 🔴 DESIGN-06 · `SHAP threshold = 0.01` hardcoded trong nhiều chỗ

**File:** `src/ml_clo/xai/shap_postprocess.py`, `src/ml_clo/config/xai_config.py`  
**Severity:** LOW

Config thay đổi nhưng code không pick up.

---

### 🔴 DESIGN-07 · Impact level chỉ có 3 bậc rời rạc

**File:** `src/ml_clo/reasoning/templates.py:109`  
**Severity:** LOW

Impact 10.1% và 24.9% đều là "medium", cùng template text.

---

### 🔴 DESIGN-08 · `EnsembleModel` weights không thể điều chỉnh sau khi train

**File:** `src/ml_clo/models/ensemble_model.py`  
**Severity:** LOW

Weights cố định trong `.joblib`. Không thể thử nghiệm weights khác mà không train lại.

---

### 🔴 DESIGN-09 · `ClassAnalysisOutput.affected_students_count` luôn = `total_students`

**File:** `src/ml_clo/outputs/schemas.py`, `src/ml_clo/pipelines/analysis_pipeline.py`  
**Severity:** LOW

Không đếm thực tế số SV có SHAP âm cho từng nhóm — thông tin misleading.

---

### 🔴 DESIGN-10 · `gb_low_anomaly` config không được lưu vào file `.joblib` *(mới — PR #1)*

**File:** `src/ml_clo/models/ensemble_model.py:218-238`, `src/ml_clo/config/model_config.py:31-34`  
**Severity:** MEDIUM

Logic anomaly dùng 3 threshold (`gb_low_anomaly_max_gb`, `gb_low_anomaly_min_gap`, `gb_low_anomaly_rf_blend`) từ `ENSEMBLE_CONFIG`. Nhưng khi model được load từ `.joblib`, config này đọc từ file code lúc predict — không phải lúc train. Nếu config thay đổi giữa lần train và lần predict, behavior predict khác hẳn mà không có cảnh báo nào.

```python
# ensemble_model.py:220
max_gb = float(self.ensemble_config.get("gb_low_anomaly_max_gb", 0.75))
# self.ensemble_config là ENSEMBLE_CONFIG đọc lúc runtime, không phải lúc train
```

**Fix:** Save `ensemble_config` snapshot vào model metadata khi `save()`, restore khi `load()`.

---

## PERF

### 🔴 PERF-01 · SHAP Explainer cache không có cleanup mechanism

**File:** `src/ml_clo/xai/shap_explainer.py:66-90`  
**Severity:** MEDIUM

`_rf_explainer_cached` và `_gb_explainer_cached` không bao giờ được giải phóng → memory leak trong production.

**Fix:** Thêm `clear_cache()` method.

---

### 🔴 PERF-02 · Merge operations không set index trước

**File:** `src/ml_clo/data/mergers.py`  
**Severity:** LOW

Không set index → pandas scan toàn bộ DataFrame cho mỗi join.

---

### 🔴 PERF-03 · `analyze_class_from_scores()` tính SHAP lần lượt từng SV

**File:** `src/ml_clo/pipelines/analysis_pipeline.py`  
**Severity:** LOW

`TreeExplainer` hỗ trợ batch. Nên gom toàn bộ X rồi gọi một lần.

---

## MISSING

### 🔴 MISSING-01 · CLI không validate input của người dùng

**File:** `scripts/predict.py`, `scripts/train.py`, `scripts/analyze_class.py`  
**Severity:** MEDIUM

`--actual-score`, `--student-id`, `--exam-scores` không được validate.

---

### ⚠️ MISSING-02 · Không có cross-validation

**File:** `src/ml_clo/pipelines/train_pipeline.py`  
**Severity:** MEDIUM

PR #1 thêm `GroupShuffleSplit` (split theo student, không để SV vừa train vừa test) — cải thiện đáng kể. Nhưng vẫn là một lần split, không phải k-fold.

---

### 🔴 MISSING-03 · Không có prediction confidence/uncertainty

**File:** `src/ml_clo/models/ensemble_model.py`, `src/ml_clo/outputs/schemas.py`  
**Severity:** MEDIUM

Chỉ trả về point estimate, không có variance/confidence interval.

---

### 🔴 MISSING-04 · Không có data quality report khi train

**File:** `src/ml_clo/pipelines/train_pipeline.py`  
**Severity:** LOW

---

### ⚠️ MISSING-05 · Test coverage còn thiếu

**File:** `tests/`  
**Severity:** LOW

PR #1 thêm `test_preprocess_deduplicates_same_course_key` trong `test_preprocessors.py`. Tuy nhiên vẫn thiếu test cho:

- Tất cả SHAP dương (BUG-03)
- `stable_hash_int` collision
- `gb_low_anomaly` path
- `_use_calibrated_template` với các trường hợp biên

---

### 🔴 MISSING-06 · Không log prediction history để audit

**File:** `src/ml_clo/pipelines/predict_pipeline.py`  
**Severity:** LOW

---

### 🔴 MISSING-07 · `study_hours` không được truyền vào `create_student_record_from_ids()`

**File:** `src/ml_clo/pipelines/predict_pipeline.py:291-313`  
**Severity:** LOW

SV mới (fallback path) không có giờ tự học trong feature vector dù có file tự học.

---

## NEW — Issues mới phát sinh từ PR #1

### 🔴 NEW-01 · SHAP explanation không nhất quán với prediction khi `gb_low_anomaly` kích hoạt

**File:** `src/ml_clo/models/ensemble_model.py:218-238` và `src/ml_clo/xai/shap_explainer.py`  
**Severity:** HIGH

Khi anomaly được phát hiện (`gb_pred < 0.75` AND `rf_pred - gb_pred > 0.35`), `predict()` thay `gb_pred` bằng `gb_use = 0.88*rf_pred + 0.12*gb_pred`. Nhưng `EnsembleSHAPExplainer.explain_instance()` vẫn tính SHAP bằng công thức gốc `rf_weight * rf_shap + gb_weight * gb_shap` (không biết về `gb_use`).

Hệ quả: SHAP values giải thích một hàm khác với hàm thực sự được dùng để dự đoán → explanation không trung thực với prediction.

```python
# ensemble_model.py — dự đoán thực
gb_use = np.where(anomaly, 0.88*rf_pred + 0.12*gb_pred, gb_pred)
ensemble_pred = rf_weight * rf_pred + gb_weight * gb_use  # gb_use ≠ gb_pred

# shap_explainer.py — giải thích (không biết về gb_use)
ensemble_shap = rf_weight * rf_shap + gb_weight * gb_shap  # gb_shap từ gb_pred gốc
```

**Fix:** `EnsembleSHAPExplainer` cần nhận biết anomaly logic, hoặc tính SHAP từ prediction function tổng hợp thay vì weighted sum đơn giản. Cách dễ hơn: lưu flag `anomaly_applied` trong output và log warning khi explanation có thể không chính xác.

---

### 🔴 NEW-02 · `stable_hash_int` — model cũ (LabelEncoder) không tương thích với code mới (hash)

**File:** `src/ml_clo/utils/hash_utils.py`, `src/ml_clo/pipelines/train_pipeline.py:228`, `predict_pipeline.py:370`  
**Severity:** HIGH

PR #1 thay `LabelEncoder` bằng `stable_hash_int` cho tất cả object columns. Model `.joblib` được train trước PR #1 sẽ có feature matrix với encoding kiểu LabelEncoder, nhưng prediction pipeline bây giờ dùng hash → encoding hoàn toàn khác → predict sai.

Không có cơ chế nào trong model file để biết encoding method nào đã được dùng khi train.

**Fix:** Lưu `encoding_method: "hash_v1"` vào model metadata khi save. Khi load model cũ (không có field này), raise `ModelLoadError` với thông báo rõ ràng yêu cầu retrain.

---

### 🔴 NEW-03 · `stable_hash_int` có thể collision với high-cardinality values

**File:** `src/ml_clo/utils/hash_utils.py:14`  
**Severity:** MEDIUM

`mod = 1_000_000_000` → không gian hash 10^9. Với các trường như `place_of_birth` có thể có nhiều giá trị unique, xác suất collision tăng theo birthday paradox: với ~45k giá trị unique, xác suất ít nhất 1 collision ~0.1%.

Collision nghĩa là hai giá trị khác nhau được encode giống nhau → model nhìn nhận 2 danh mục là một.

```python
def stable_hash_int(value: Any, mod: int = 1_000_000_000) -> int:
    h = hashlib.sha256(s.encode("utf-8")).hexdigest()
    return int(h[:16], 16) % mod  # Modulo → collision space
```

**Fix:** Tăng `mod = 2**31 - 1` (max int32) hoặc log warning nếu cardinality của cột > ngưỡng.

---

### 🔴 NEW-04 · `calibrated` field mới trong reason dict không có trong `Reason` dataclass

**File:** `src/ml_clo/reasoning/reason_generator.py:64` và `src/ml_clo/outputs/schemas.py`  
**Severity:** MEDIUM

`reason_generator.py` thêm `"calibrated": calibrated` vào reason dict, nhưng `Reason` dataclass trong `schemas.py` không có field này. `from_explanation_dict()` sẽ bỏ qua field → backend không biết lý do nào đã được calibrate. Nếu backend muốn hiển thị khác cho calibrated reasons, không có cách nào.

**Fix:** Thêm `calibrated: bool = False` vào `Reason` dataclass và map trong `from_explanation_dict()`.

---

### 🔴 NEW-05 · `analysis_pipeline.py` không truyền `raw_feature_row` → không bao giờ calibrate

**File:** `src/ml_clo/pipelines/analysis_pipeline.py` và `src/ml_clo/reasoning/reason_generator.py:270`  
**Severity:** MEDIUM

`generate_complete_explanation()` nhận tham số mới `raw_feature_row` để calibrate reason text. `predict_pipeline.py` truyền đúng (`raw_feature_row=student_df.iloc[0]`). Nhưng `analysis_pipeline.py` gọi hàm này mà không truyền `raw_feature_row` → class analysis không bao giờ có calibrated reasons, ngay cả khi toàn lớp có điểm danh tốt.

**Fix:** Trong `analysis_pipeline._analyze_with_shap()`, tính mean feature row của lớp và truyền vào `generate_complete_explanation(raw_feature_row=mean_row)`.

---

### 🔴 NEW-06 · `academic_core_score` dùng `np.mean` của các metric correlated cao

**File:** `src/ml_clo/features/feature_builder.py:258-261`  
**Severity:** LOW

```python
core_parts = [x for x in (median_exam_score, recent_avg_score, recent_median_score) if pd.notna(x)]
academic_core_score = float(np.mean(core_parts)) if core_parts else np.nan
```

`median_exam_score`, `recent_avg_score`, `recent_median_score` có correlation cao (cùng đo điểm học lực). Average của chúng mang ít thông tin mới hơn là từng cái riêng lẻ và tạo ra multicollinearity với các feature gốc trong cùng nhóm "Học lực" của model.

---

### 🔴 NEW-07 · `min_exam_score` bị exclude khỏi features nhưng vẫn được map trong `xai_config.py`

**File:** `src/ml_clo/config/xai_config.py:51` và `src/ml_clo/pipelines/train_pipeline.py:210`  
**Severity:** LOW

`min_exam_score` bị loại khỏi feature vector khi train (`feature_cols = [c for c in feature_cols if c != "min_exam_score"]`), nhưng `PEDAGOGICAL_GROUP_PATTERNS["Học lực"]` trong `xai_config.py` vẫn include pattern `"min_exam"`. Nếu sau này ai bỏ dòng exclusion (e.g., nhầm tưởng là thừa), `min_exam_score` sẽ lặng lẽ xuất hiện trong SHAP grouping.

**Fix:** Xóa `"min_exam"` khỏi `xai_config.py` patterns, hoặc thêm comment giải thích tại sao excluded.

---

## Tổng kết

| Category | Tổng | Open | Fixed | Partial |
| --- | --- | --- | --- | --- |
| BUG | 6 | 6 | 1 (BUG-07) | 0 |
| DATA | 7 | 7 | 0 | 0 |
| DESIGN | 10 | 9 | 0 | 1 (DESIGN-02) |
| PERF | 3 | 3 | 0 | 0 |
| MISSING | 7 | 5 | 0 | 2 (MISSING-02, 05) |
| NEW (PR #1) | 7 | 7 | 0 | 0 |
| **Tổng** | **40** | **37** | **1** | **3** |

---

### Ưu tiên fix ngay (HIGH/CRITICAL)

1. **NEW-01** — SHAP explanation ≠ actual prediction khi `gb_low_anomaly` kích hoạt → XAI không trung thực
2. **NEW-02** — Model cũ incompatible với hash encoding mới → predict sai im lặng
3. **BUG-01** — Double merge study hours → `total_study_hours` sai nghĩa
4. **BUG-02** — Score conversion không guard → corrupt label nếu dùng file hệ 6

### Ưu tiên fix tiếp theo (MEDIUM)

1. **DATA-07** — `deduplicate_exam_scores` với `dropna=False` có thể gộp nhầm record
2. **DESIGN-10** — `gb_low_anomaly` config không được save vào `.joblib`
3. **NEW-04** — Field `calibrated` không có trong `Reason` dataclass
4. **NEW-05** — `analysis_pipeline` không truyền `raw_feature_row` → không calibrate
5. **BUG-03** — SHAP fallback không filter top N
6. **BUG-05** — EnsembleModel không reorder columns khi predict
