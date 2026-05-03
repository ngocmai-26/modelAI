# Kế hoạch triển khai — Feedback từ Giảng viên Hướng dẫn (1, 2, 3)

**Ngày:** 2026-05-03
**Phiên bản:** 1.0
**Tham chiếu:** [advisor_feedback_assessment.md](advisor_feedback_assessment.md)
**Phạm vi:** Triển khai feedback **1 (Temporal FE)**, **2 (Ablation Study)**, **3 (Encoding)**. Feedback 4 (Causal) **không thực hiện**.

---

## Tổng quan timeline

| Giai đoạn | Nội dung | Effort | Output |
|-----------|---------|--------|--------|
| **Phase 0** | Chuẩn bị baseline, đóng băng metrics | 0.5 ngày | Bảng metrics gốc |
| **Phase 1** | Ablation Study (Feedback 2) | 2–3 ngày | Bảng so sánh Baseline vs Extended |
| **Phase 2** | Temporal Features (Feedback 1) | 3–4 ngày | 6 feature mới + bảng metrics so sánh |
| **Phase 3** | Encoding Ablation (Feedback 3) | 3–4 ngày | Target encoding + bảng so sánh |
| **Phase 4** | Tổng hợp luận văn | 1 ngày | Chương 5 + appendix |

**Tổng:** ~2 tuần (10–12 ngày làm việc)

---

## Hiện trạng dữ liệu (kiểm tra thực tế)

| File | Rows | Temporal? | Ghi chú |
|------|------|-----------|---------|
| `Dữ liệu điểm danh Khoa FIRA.xlsx` | 63,325 | ✅ Có cột `Ngày học` (datetime), `Buổi thứ` 1–12 | Slope/variance theo tuần làm được |
| `tuhoc.xlsx` | 1,982 | ❌ Chỉ có `(Student_ID, year, semester)` | **Không có chi tiết tuần** — variance giờ tự học **theo tuần không khả thi** |
| `Khảo sát...xlsx` | 280 | — | 42 cột, key `MSSV + Năm học + Học kỳ`, mixed ordinal + multi-select |
| `Danh sach thong tin khao sat sinh vien k28-CNTT.xlsx` | 89 | — | Admissions data, **không phải khảo sát**, dùng làm enrichment optional |

**Điều chỉnh kế hoạch so với feedback gốc:**
- Slope chuyên cần theo tuần: ✅ Khả thi
- Variance giờ tự học theo tuần: ❌ Không khả thi → **thay thế** bằng variance theo (year, semester) hoặc xu hướng tích lũy

---

## Phase 0: Chuẩn bị baseline (0.5 ngày)

### 0.1 Mục tiêu

Đóng băng metrics hiện tại làm điểm tham chiếu cho mọi so sánh sau.

### 0.2 Công việc

- [ ] Train model với dataset hiện tại (không thay đổi gì) → ghi nhận baseline metrics
- [ ] Lưu artifact: `models/baseline_v0_pre_advisor.joblib`
- [ ] Ghi metrics vào file: `experiments/results.json`
- [ ] Confirm seed `random_state=42` cho reproducibility

### 0.3 Deliverable

```json
{
  "baseline_v0": {
    "MAE": 0.3945,
    "RMSE": 0.50,
    "R²": 0.7980,
    "n_samples": 1234,
    "features": 76,
    "encoding": "hash_v2"
  }
}
```

---

## Phase 1: Ablation Study (Feedback 2) — 2–3 ngày

### 1.1 Mục tiêu

Trả lời câu hỏi: **"Dữ liệu khảo sát có cải thiện model so với chỉ dùng dữ liệu hệ thống không?"**

### 1.2 Công việc chi tiết

#### Bước 1.2.1: Loader cho file khảo sát (0.5 ngày)

**File mới:** [src/ml_clo/data/loaders.py](../src/ml_clo/data/loaders.py) — thêm `load_survey_responses()`

- Đọc file `Khảo sát...xlsx`
- Rename các cột sang tên chuẩn (ví dụ `1.1. Mã sinh viên` → `Student_ID`, `1.2. Năm học` → `year`, `1.3. Học kỳ` → `semester`)
- Trả về DataFrame thô (chưa encode)

#### Bước 1.2.2: Preprocessor cho khảo sát (1 ngày)

**File mới:** `src/ml_clo/data/survey_preprocessor.py`

- **Ordinal encoding** cho Likert: "Không" → 0, "Hiếm khi" → 1, "Thỉnh thoảng" → 2, "Thường xuyên" → 3, "Rất thường xuyên" → 4
- **Multi-select expansion** cho fields như `4.2` (vấn đề tâm lý), `5.4` (phương pháp học) → multi-hot binary columns
- **Numeric extraction** cho fields có số: giờ ngủ, giờ tự học/ngày, giờ làm thêm/tuần
- **Missing handling**: SV không có khảo sát → imputed = median (hoặc 0)

**Output features dự kiến (~30–40 cột):**

| Group | Features (ví dụ) |
|-------|-----------------|
| Family/Finance | `family_financial_support`, `family_emotional_support`, `family_expectation`, `tuition_pressure` |
| Part-time | `has_parttime_job`, `parttime_hours_per_week`, `parttime_impact_on_study` |
| Mental/Health | `study_pressure`, `sleep_hours_avg`, `mental_issue_stress`, `mental_issue_anxiety` |
| Study habits | `self_study_hours_per_day`, `time_management_score`, `has_study_plan`, `peer_pressure` |
| Tech/Resources | `internet_quality`, `lms_usage`, `digital_tools_count` |

#### Bước 1.2.3: Merger cho khảo sát (0.25 ngày)

**File:** [src/ml_clo/data/mergers.py](../src/ml_clo/data/mergers.py) — thêm `merge_survey_responses()`

- Merge key: `(Student_ID, year, semester)` — left join (giữ tất cả SV, kể cả không có khảo sát)
- Flag `has_survey_response` (binary) để model biết khi nào survey features là imputed

#### Bước 1.2.4: Tích hợp vào TrainingPipeline (0.5 ngày)

**File:** [src/ml_clo/pipelines/train_pipeline.py](../src/ml_clo/pipelines/train_pipeline.py)

- Thêm tham số `survey_path: Optional[str] = None` vào `run()`
- Khi có `survey_path` → gọi `load_survey_responses()` + `merge_survey_responses()`
- Khi không có → bỏ qua, model chỉ dùng LMS/SIS data

#### Bước 1.2.5: Script chạy ablation (0.5 ngày)

**File mới:** `scripts/run_ablation_survey.py`

```python
scenarios = {
    "baseline_lms": {
        "exam_scores_path": "data/DiemTong.xlsx",
        "conduct_scores_path": "data/diemrenluyen.xlsx",
        "demographics_path": "data/nhankhau.xlsx",
        "teaching_methods_path": "data/PPGDfull.xlsx",
        "assessment_methods_path": "data/PPDGfull.xlsx",
        "study_hours_path": "data/tuhoc.xlsx",
        "attendance_path": "data/Dữ liệu điểm danh Khoa FIRA.xlsx",
        # KHÔNG có survey
    },
    "extended_survey": {
        # ... y như baseline ...
        "survey_path": "data/Khảo sát...xlsx",
    },
}

for name, paths in scenarios.items():
    # Train với cùng random_state, cùng test split
    pipeline = TrainingPipeline(random_state=42, test_size=0.2)
    model, metrics = pipeline.run(**paths, output_path=f"models/{name}.joblib")
    save_metrics(name, metrics)

# In bảng so sánh + delta
print_comparison_table(["baseline_lms", "extended_survey"])
```

### 1.3 Deliverable

**Bảng kết quả** (cho luận văn Chương 4 — Kết quả thực nghiệm):

| Scenario | MAE | RMSE | R² | n_features | n_samples |
|----------|-----|------|----|------------|-----------|
| Baseline (LMS/SIS) | ? | ? | ? | 76 | 1234 |
| Extended (+ Survey) | ? | ? | ? | ~110 | 1234 |
| **Δ (cải thiện)** | ? | ? | ? | +34 | 0 |

**Phân tích kèm theo:**
- Thống kê % SV có khảo sát (280/?)
- SHAP top features mới từ khảo sát (nếu có cải thiện)

### 1.4 Kết luận có thể có

**Kịch bản A (cải thiện rõ):** "Dữ liệu khảo sát giảm MAE từ X xuống Y → đáng đầu tư thu thập."

**Kịch bản B (không cải thiện):** "Dữ liệu hành vi LMS/SIS có sức mạnh dự báo tuyệt đối; dữ liệu khảo sát chứa nhiễu cao và có thể lược bỏ để tối ưu hệ thống tự động."

Cả 2 đều là **bằng chứng học thuật mạnh**.

---

## Phase 2: Temporal Features (Feedback 1) — 3–4 ngày

### 2.1 Mục tiêu

Thêm features bắt **xu hướng và độ ổn định theo thời gian** từ dữ liệu điểm danh.

### 2.2 Công việc chi tiết

#### Bước 2.2.1: Mở rộng attendance preprocessor (0.5 ngày)

**File:** [src/ml_clo/data/preprocessors.py](../src/ml_clo/data/preprocessors.py) — thêm `derive_week_from_date()`

- Từ `Ngày học` → tính `week_in_semester` (1, 2, 3, ...)
- Tính `week_global` (week number tuyệt đối) qua `(Năm học, Học kì, Ngày học)`
- Encode `Điểm danh`: `{Có mặt: 1, Sớm: 1, Trễ: 0.5, Vắng: 0}` → `attendance_score_per_session`

#### Bước 2.2.2: Module temporal features (1.5 ngày)

**File mới:** `src/ml_clo/features/temporal_features.py`

```python
def build_temporal_attendance_features(
    attendance_df: pd.DataFrame,
    student_id_col: str = "MSSV",
    subject_id_col: str = "Mã nhóm",
) -> pd.DataFrame:
    """Compute weekly aggregates per (student, subject) then derive trends."""
    # Step 1: Aggregate to weekly attendance rate per (Student, Subject, week)
    weekly = attendance_df.groupby([student_id_col, subject_id_col, "week_in_semester"])
        .agg(weekly_rate=("attendance_score_per_session", "mean"))
        .reset_index()

    # Step 2: For each (student, subject), compute trend features
    features = []
    for (sid, subj), group in weekly.groupby([student_id_col, subject_id_col]):
        rates = group.sort_values("week_in_semester")["weekly_rate"].values
        weeks = group["week_in_semester"].values

        # Slope của 3 tuần gần nhất (đề xuất gốc của giảng viên)
        slope_3w = np.polyfit(weeks[-3:], rates[-3:], 1)[0] if len(rates) >= 3 else 0.0

        # Slope toàn kỳ
        slope_full = np.polyfit(weeks, rates, 1)[0] if len(rates) >= 2 else 0.0

        # Volatility
        volatility = np.std(rates)

        # Late streak
        late_streak = max_consecutive_below_threshold(rates, threshold=0.7)

        # Early dropoff flag
        first_half = rates[:len(rates)//2].mean()
        second_half = rates[len(rates)//2:].mean()
        dropoff = 1 if (first_half - second_half) > 0.3 else 0

        features.append({
            student_id_col: sid,
            subject_id_col: subj,
            "attendance_slope_3w": slope_3w,
            "attendance_slope_full": slope_full,
            "attendance_volatility": volatility,
            "late_streak_max": late_streak,
            "early_dropoff_flag": dropoff,
        })

    return pd.DataFrame(features)
```

#### Bước 2.2.3: Mở rộng cho self-study (thay thế đề xuất gốc do hạn chế dữ liệu) (0.5 ngày)

**Vì `tuhoc.xlsx` chỉ có granularity (year, semester), không có tuần:**

- Thay thế "variance giờ tự học theo tuần" bằng:
  - `study_hours_growth_rate`: tỷ lệ tăng/giảm giờ tự học giữa các kỳ
  - `study_hours_consistency`: std giờ tự học qua các kỳ (proxy cho ổn định)
  - `study_hours_recent_trend`: trung bình 2 kỳ gần nhất so với toàn quá khứ

**Lưu ý ghi chú trong luận văn:** Trình bày rõ vì sao không tính được variance theo tuần (limitation dữ liệu) và tại sao proxy theo kỳ vẫn có giá trị.

#### Bước 2.2.4: Tích hợp vào feature_builder (0.5 ngày)

**File:** [src/ml_clo/features/feature_builder.py](../src/ml_clo/features/feature_builder.py) — thêm gọi `build_temporal_attendance_features()` trong `build_all_features()`.

#### Bước 2.2.5: Train & so sánh (0.5 ngày)

```python
# scripts/run_temporal_ablation.py
scenarios = {
    "no_temporal": {... attendance dùng aggregate hiện tại ...},
    "with_temporal": {... thêm 6 temporal features ...},
}
```

### 2.3 Deliverable

**Bảng kết quả:**

| Scenario | MAE | RMSE | R² | n_features |
|----------|-----|------|----|------------|
| No temporal | ? | ? | ? | 76 |
| + Temporal | ? | ? | ? | 82 |
| **Δ** | ? | ? | ? | +6 |

**SHAP analysis:** Top 10 features mới — temporal features có vào top không?

### 2.4 Đoạn viết cho luận văn Chương 5

> "Việc làm giàu đặc trưng (Feature Engineering) cho mô hình Ensemble giúp mô hình bắt được xu hướng thời gian mà vẫn duy trì được độ phức tạp tính toán thấp, phù hợp với ràng buộc hạ tầng của Trường Đại học Bình Dương. Cụ thể, 6 đặc trưng phái sinh từ dữ liệu điểm danh (slope chuyên cần 3 tuần gần nhất, độ biến động chuyên cần, chuỗi vắng/trễ liên tiếp, dấu hiệu sụt giảm giữa kỳ) được tính bằng các phép toán vectorized trên numpy, không yêu cầu deep learning hay GPU. Kết quả thực nghiệm (Bảng X) cho thấy MAE giảm từ {a} xuống {b}, R² tăng từ {c} lên {d}, chứng tỏ thông tin thời gian đóng góp giá trị thực cho dự đoán."

---

## Phase 3: Encoding Ablation (Feedback 3) — 3–4 ngày

### 3.1 Mục tiêu

So sánh **Hash encoding (hiện tại)** vs **Target encoding** cho `Subject_ID`, `Lecturer_ID`. Có bằng chứng số liệu cho luận văn.

### 3.2 Công việc chi tiết

#### Bước 3.2.1: Implement target encoding (1 ngày)

**File mới:** `src/ml_clo/features/categorical_encoder.py`

```python
class TargetEncoder:
    """K-fold target encoder để tránh leakage."""
    def __init__(self, n_folds: int = 5, smoothing: float = 10.0):
        self.n_folds = n_folds
        self.smoothing = smoothing
        self.global_mean: Optional[float] = None
        self.category_means: Dict[Any, float] = {}

    def fit_transform(self, X_col: pd.Series, y: pd.Series) -> pd.Series:
        """Fit + transform với K-fold để tránh leakage trong training data."""
        # Ngoài fold: tính mean per category trên K-1 folds
        # In fold: apply mean đó cho fold hiện tại
        # Smoothing: weighted avg với global_mean (chống over-fit cho category hiếm)

    def transform(self, X_col: pd.Series) -> pd.Series:
        """Apply mapping đã fit (cho test data)."""
        # Unknown category → global_mean
```

#### Bước 3.2.2: Tích hợp vào feature_encoder (0.5 ngày)

**File:** [src/ml_clo/features/feature_encoder.py](../src/ml_clo/features/feature_encoder.py)

- Thêm tham số `categorical_strategy: Literal["hash", "target", "frequency"] = "hash"`
- Khi `target`: dùng `TargetEncoder` cho `Subject_ID`, `Lecturer_ID`
- **Quan trọng:** target encoder phải được **fit trên train set** và **lưu trong model artifact** để predict dùng lại

#### Bước 3.2.3: Cập nhật model save/load (0.5 ngày)

**File:** [src/ml_clo/models/base_model.py](../src/ml_clo/models/base_model.py)

- Lưu `target_encoder` vào `extra_metadata`
- `encoding_method` = `"target_v1"` hoặc giữ `"hash_v2"` tuỳ strategy
- Validate khi load (giống như hash encoding)

#### Bước 3.2.4: Frequency encoding (optional, 0.25 ngày)

Implement đơn giản trong `categorical_encoder.py` để có 3 baseline so sánh.

#### Bước 3.2.5: Script ablation encoding (0.5 ngày)

**File mới:** `scripts/run_ablation_encoding.py`

```python
scenarios = {
    "encoding_hash":      {"categorical_strategy": "hash"},
    "encoding_frequency": {"categorical_strategy": "frequency"},
    "encoding_target":    {"categorical_strategy": "target"},
}
```

#### Bước 3.2.6: Test & validate (0.5 ngày)

- Unit test: target encoder không leak (test với artificial leakage scenario)
- Integration test: predict với entity mới (`__UNKNOWN__`) vẫn work với target encoder

### 3.3 Deliverable

**Bảng kết quả:**

| Encoding | MAE | RMSE | R² | Train time | Predict time |
|----------|-----|------|----|------------|--------------|
| Hash (hiện tại) | ? | ? | ? | ? | ? |
| Frequency | ? | ? | ? | ? | ? |
| Target (5-fold) | ? | ? | ? | ? | ? |

**Phân tích kèm:**
- Cardinality `Subject_ID` và `Lecturer_ID` thực tế
- Tỷ lệ entity mới trong test set (proxy cho production scenario)
- Trade-off: target encoding cải thiện accuracy nhưng đắt hơn lúc train

### 3.4 Quyết định cuối cho production

| Strategy | Khi nào dùng |
|----------|-------------|
| Target encoding | Default cho training |
| Hash fallback | Cho `__UNKNOWN__` (entity mới chưa có trong train) |

Combine: **Target encoding với fallback hash cho unknown** — best of both worlds.

---

## Phase 4: Tổng hợp luận văn (1 ngày)

### 4.1 Output cho luận văn

#### Chương 4 — Kết quả thực nghiệm

| Bảng | Nội dung | Nguồn |
|------|---------|-------|
| Bảng 4.X | Baseline metrics (Phase 0) | `experiments/results.json` |
| Bảng 4.Y | Ablation Survey (Phase 1) | `experiments/ablation_survey.json` |
| Bảng 4.Z | Ablation Temporal (Phase 2) | `experiments/ablation_temporal.json` |
| Bảng 4.W | Ablation Encoding (Phase 3) | `experiments/ablation_encoding.json` |

#### Chương 5 — Thảo luận

- Đoạn về Feature Engineering (Phase 2 đề xuất ở 2.4)
- Đoạn về encoding choice (kết luận từ Phase 3)
- Đoạn về giá trị dữ liệu khảo sát (kết luận từ Phase 1, dù theo hướng nào)

#### Phụ lục

- `docs/EXPERIMENTS_LOG.md` — chi tiết các kịch bản đã chạy

---

## Cấu trúc file thay đổi

### File mới
```
src/ml_clo/data/survey_preprocessor.py     # Phase 1
src/ml_clo/features/temporal_features.py   # Phase 2
src/ml_clo/features/categorical_encoder.py # Phase 3

scripts/run_ablation_survey.py             # Phase 1
scripts/run_temporal_ablation.py           # Phase 2
scripts/run_ablation_encoding.py           # Phase 3

experiments/results.json                   # Tất cả phase
docs/EXPERIMENTS_LOG.md                    # Phase 4

tests/unit/test_features/test_temporal_features.py
tests/unit/test_features/test_categorical_encoder.py
tests/unit/test_data/test_survey_preprocessor.py
```

### File sửa
```
src/ml_clo/data/loaders.py                 # +load_survey_responses (Phase 1)
src/ml_clo/data/mergers.py                 # +merge_survey_responses (Phase 1)
src/ml_clo/data/preprocessors.py           # +derive_week_from_date (Phase 2)
src/ml_clo/features/feature_builder.py     # +temporal calls (Phase 2)
src/ml_clo/features/feature_encoder.py     # +categorical_strategy param (Phase 3)
src/ml_clo/models/base_model.py            # +target_encoder in metadata (Phase 3)
src/ml_clo/pipelines/train_pipeline.py     # +survey_path (Phase 1)
scripts/train.py                           # +--survey, +--encoding (Phase 1, 3)
```

---

## Rủi ro & cách giảm thiểu

| Rủi ro | Cách giảm thiểu |
|--------|----------------|
| Khảo sát chỉ 280/1234 SV → imputed nhiều → noise | Thêm flag `has_survey_response`; train cả 2 mode (chỉ SV có khảo sát vs full) để so sánh |
| Target encoding gây overfit nếu cardinality cao + smoothing không đủ | K-fold + smoothing parameter; tune trên validation set |
| Temporal features mất ý nghĩa nếu < 3 buổi điểm danh | Fallback về 0; flag `has_temporal_data` |
| Model retrain phá vỡ backend | Đặt encoding strategy mới = `"hybrid_v3"`, model cũ vẫn load được; deploy mới qua hot-reload |
| Feedback 4 (causal) bị giảng viên hỏi lại | Phụ lục: bổ sung permutation importance + PDP + caveat trong luận văn (effort 1 ngày, có thể chèn vào Phase 4) |

---

## Checkpoint & Quyết định

Sau mỗi Phase, **review trước khi qua phase tiếp theo**:

| Checkpoint | Câu hỏi quyết định |
|-----------|--------------------|
| Sau Phase 1 | Khảo sát có cải thiện không? Nếu có → giữ trong final model. Nếu không → kết luận và bỏ. |
| Sau Phase 2 | Temporal features có vào top SHAP không? Nếu có → giữ. |
| Sau Phase 3 | Target encoding có vượt hash đáng kể không? Nếu cải thiện > 5% MAE → switch default. |
| Sau Phase 4 | Đủ bằng chứng cho 3 chương luận văn chưa? |

---

## Câu hỏi cần xác nhận trước khi bắt đầu

1. **Cấu hình ablation:** Có cần thêm scenario `"survey_only"` (chỉ khảo sát, không LMS) để có 3 điểm so sánh không? (Thầy không yêu cầu, nhưng làm thêm thì luận văn mạnh hơn.)
2. **Subject_ID encoding:** Hiện tại pipeline đang **EXCLUDE** `Subject_ID` khỏi feature ([feature_encoder.py:21](../src/ml_clo/features/feature_encoder.py#L21))? Nếu vậy, target encoding cho Subject không impact — cần check lại logic exclude.
3. **Test budget:** Có nên giới hạn budget test (giảm `n_estimators`) trong các ablation để chạy nhanh không? (Final model vẫn dùng config full.)

---

## Tóm tắt

3 phase độc lập, mỗi phase đều có **bảng số liệu cụ thể** cho luận văn. Tổng effort ~2 tuần, output là:

- **3 bảng ablation** trả lời 3 câu hỏi nghiên cứu khác nhau
- **6+ feature mới** (temporal + survey-derived) cải thiện model
- **1 phương pháp encoding mới** với bằng chứng so sánh
- **Đoạn viết sẵn cho Chương 5** luận văn

Bắt đầu khi nào? Đề xuất: **Phase 0 hôm nay → Phase 1 ngày mai** (vì Phase 1 cho ROI cao nhất và validate luôn data khảo sát).
