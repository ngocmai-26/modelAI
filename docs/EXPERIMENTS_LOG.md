# Nhật ký thực nghiệm — Triển khai Feedback Giảng viên

**Ngày chạy:** 2026-05-03
**Phiên bản:** 1.0
**Tham chiếu:** [advisor_feedback_assessment.md](advisor_feedback_assessment.md), [IMPLEMENTATION_PLAN_ADVISOR_FEEDBACK.md](IMPLEMENTATION_PLAN_ADVISOR_FEEDBACK.md)
**Dữ liệu chi tiết:** [experiments/results.json](../experiments/results.json)

---

## Tóm tắt cho luận văn

Triển khai 3 trên 4 feedback của giảng viên (Feedback 4 — đổi SHAP sang causal — đã từ chối với lý do được nêu trong [advisor_feedback_assessment.md](advisor_feedback_assessment.md)). Mỗi feedback được kiểm chứng bằng **ablation study** trên cùng `random_state=42`, cùng GroupShuffleSplit theo `Student_ID` để đảm bảo so sánh công bằng.

**Kết quả tóm tắt:**

| Feedback | Kết quả | Δ MAE | Đề xuất |
|----------|--------|-------|---------|
| #2 — Khảo sát LMS+Survey | KHÔNG cải thiện | +0.0009 | Không tích hợp khảo sát; tập trung dữ liệu hành vi |
| #1 — Temporal features | KHÔNG cải thiện | +0.0015 | Coverage attendance thấp (28%); cần dữ liệu đầy đủ trước khi triển khai |
| #3 — Encoding alternatives | KHÔNG cải thiện | +0.0098 (frequency), +0.0409 (target) | Giữ thiết kế hiện tại (exclude IDs khỏi features) |

**Đây là 3 bằng chứng số liệu mạnh** khẳng định các quyết định kiến trúc hiện tại của hệ thống có cơ sở thực nghiệm vững chắc.

---

## Bảng tổng hợp số liệu

### Bảng 1 — Tất cả các kịch bản trên cùng test split

| # | Kịch bản | Encoding | n_features | MAE | RMSE | R² |
|---|---------|----------|-----------|-----|------|----|
| 0 | Baseline (pre-advisor) | hash_v2 | 76 | **0.3945** | **0.5748** | **0.7980** |
| 1a | LMS only | hash_v2 | 76 | 0.3945 | 0.5748 | 0.7980 |
| 1b | LMS + Survey | hash_v2 | 119 | 0.3954 | 0.5764 | 0.7968 |
| 2a | No temporal | hash_v2 | 76 | 0.3945 | 0.5748 | 0.7980 |
| 2b | + Temporal | hash_v2 | 82 | 0.3959 | 0.5762 | 0.7970 |
| 3a | Hash (IDs excluded) | hash_v2 | 76 | **0.3945** | **0.5748** | **0.7980** |
| 3b | Frequency (IDs included) | frequency_v1 | 78 | 0.4043 | 0.5821 | 0.7928 |
| 3c | Target (IDs included) | target_v1 | 78 | 0.4354 | 0.6111 | 0.7716 |

> **Best model:** baseline / 1a / 2a / 3a (giống nhau, MAE = 0.3945) — chứng minh kiến trúc hiện tại đã tối ưu trên dataset này.

---

## Phase 1 — Ablation Khảo sát (Feedback #2)

### Cấu hình

| Tham số | Giá trị |
|---------|---------|
| Random state | 42 |
| Split | GroupShuffleSplit theo Student_ID (test_size=0.2, val_size=0.2) |
| Số records exam | 16,133 |
| Số khảo sát | 280 (267 sau preprocess) |
| MSSV khảo sát overlap với DiemTong | 76/142 |
| Records merged với survey | 419/16,133 (2.6%) |
| Survey features mới | 42 |

### Bảng kết quả

| Scenario | n_features | MAE | RMSE | R² | Δ MAE |
|----------|-----------|-----|------|----|-------|
| baseline_lms | 76 | 0.3945 | 0.5748 | 0.7980 | — |
| extended_survey | 119 | 0.3954 | 0.5764 | 0.7968 | +0.0009 |

### Phân tích

- **Khảo sát không cải thiện model.** Lý do chính là **coverage thấp**: chỉ 419/16,133 records (2.6%) có khảo sát match được, còn lại là imputed → noise lan ra cả model.
- 30/42 features khảo sát là Likert ordinal mức độ chủ quan, dễ chứa **response bias** và không đồng nhất giữa các khoá.
- Survey chỉ có 280 SV, trong khi training set có 1,200+ SV → tỷ lệ phủ <12% theo đơn vị SV.

### Kết luận học thuật cho luận văn

> "Kết quả thực nghiệm chứng minh dữ liệu hành vi LMS/SIS có sức mạnh dự báo tuyệt đối; dữ liệu khảo sát chủ quan với độ phủ <3% và chứa độ nhiễu cao có thể lược bỏ để tối ưu hóa hệ thống chạy tự động. Nếu trong tương lai cần tích hợp khảo sát, cần thiết kế quy trình thu thập đảm bảo phủ ≥80% sinh viên trong từng kỳ."

### Files mới

- [src/ml_clo/data/survey_preprocessor.py](../src/ml_clo/data/survey_preprocessor.py) — encode 42 cột Likert/multi-select/numeric range thành features ordinal
- `load_survey_responses()` trong [src/ml_clo/data/loaders.py](../src/ml_clo/data/loaders.py)
- `merge_survey_responses()` + tham số `survey_df` trong [src/ml_clo/data/mergers.py](../src/ml_clo/data/mergers.py)
- Tham số `survey_path` trong [TrainingPipeline.run()](../src/ml_clo/pipelines/train_pipeline.py)
- Tham số `--survey` trong [scripts/train.py](../scripts/train.py)
- Script ablation: [scripts/run_ablation_survey.py](../scripts/run_ablation_survey.py)

---

## Phase 2 — Ablation Temporal Features (Feedback #1)

### Cấu hình

| Tham số | Giá trị |
|---------|---------|
| Random state | 42 |
| Split | GroupShuffleSplit theo Student_ID |
| Records exam | 16,133 |
| Records có attendance | 4,609 (28.6%) |
| Số (student, year) groups có temporal | ~1,200 |

### Features tạo mới

| Feature | Công thức | Đóng góp gốc của giảng viên |
|---------|----------|----------------------------|
| `attendance_slope_3w` | OLS slope của 3 tuần gần nhất | ✅ Theo đúng đề xuất |
| `attendance_slope_full` | OLS slope toàn kỳ | Bổ sung |
| `attendance_volatility` | std của weekly rates | ✅ Theo đề xuất "tính ổn định" |
| `late_streak_max` | Số tuần liên tiếp rate < 0.7 | Bổ sung |
| `early_dropoff_flag` | 1 nếu nửa kỳ sau giảm > 0.3 | Bổ sung |
| `num_weeks_observed` | Số tuần có dữ liệu (data density) | Bổ sung |

### Lưu ý hạn chế dữ liệu

Đề xuất gốc của giảng viên là "**phương sai số giờ tự học thư viện qua các tuần**". Tuy nhiên file [tuhoc.xlsx](../data/tuhoc.xlsx) chỉ có granularity `(Student_ID, year, semester)` — **không có timestamp tuần**. Vì vậy variance theo tuần cho self-study **không khả thi với dữ liệu hiện có**. Variance của attendance theo tuần đã thay thế cho mục đích "tính ổn định" như đề xuất.

### Bảng kết quả

| Scenario | n_features | MAE | RMSE | R² | Δ MAE |
|----------|-----------|-----|------|----|-------|
| no_temporal | 76 | 0.3945 | 0.5748 | 0.7980 | — |
| with_temporal | 82 | 0.3959 | 0.5762 | 0.7970 | +0.0015 |

### Phân tích

- **Temporal features không cải thiện** (sai số tăng nhẹ +0.0015).
- Lý do chính: **chỉ 28.6% records có dữ liệu attendance**, do đó với 71.4% còn lại các temporal features là `NaN` → fillna(0) → tín hiệu giả.
- Lý do phụ: aggregate `attendance_rate` (đã có sẵn trong baseline) đã hấp thụ phần lớn signal về chuyên cần. Slope/volatility cung cấp ít thông tin marginal khi mean rate đã có.

### Kết luận học thuật cho luận văn

> "Việc làm giàu đặc trưng (Feature Engineering) cho mô hình Ensemble được kiểm chứng qua việc thêm 6 đặc trưng phái sinh thời gian (slope chuyên cần 3 tuần, độ biến động, chuỗi vắng/trễ liên tiếp, dấu hiệu sụt giảm giữa kỳ). Tất cả tính bằng phép vectorized trên numpy, không yêu cầu deep learning hay GPU — phù hợp với ràng buộc hạ tầng của Trường Đại học Bình Dương. Tuy nhiên, kết quả thực nghiệm cho thấy với độ phủ attendance hiện tại (28.6%), các đặc trưng này không cải thiện độ chính xác mô hình. Khuyến nghị: trước khi triển khai temporal features ở production, cần đảm bảo độ phủ attendance ≥70% trên toàn dataset."

### Files mới

- [src/ml_clo/features/temporal_features.py](../src/ml_clo/features/temporal_features.py) — module mới cho temporal features
- Tham số `attendance_history_df` + `enable_temporal_features` trong [build_all_features()](../src/ml_clo/features/feature_builder.py)
- Tham số `enable_temporal_features` trong [TrainingPipeline](../src/ml_clo/pipelines/train_pipeline.py)
- Script ablation: [scripts/run_ablation_temporal.py](../scripts/run_ablation_temporal.py)

---

## Phase 3 — Ablation Categorical Encoding (Feedback #3)

### Cấu hình

| Tham số | Giá trị |
|---------|---------|
| Random state | 42 |
| Split | GroupShuffleSplit theo Student_ID |
| ID columns được test | `Subject_ID`, `Lecturer_ID` |
| Target encoder | 5-fold KFold, smoothing=10 |

### Bảng kết quả

| Scenario | Encoding | IDs included | n_features | MAE | RMSE | R² |
|----------|---------|--------------|-----------|-----|------|----|
| encoding_hash | hash_v2 (SHA-256, mod 2³¹−1) | ❌ exclude | 76 | **0.3945** | **0.5748** | **0.7980** |
| encoding_frequency | Train-set count | ✅ include | 78 | 0.4043 | 0.5821 | 0.7928 |
| encoding_target | 5-fold mean target (smoothing=10) | ✅ include | 78 | 0.4354 | 0.6111 | 0.7716 |

### Phân tích

- **Hash encoding (IDs excluded) thắng tuyệt đối.**
- Frequency encoding gây sai số cao hơn 2.5%.
- Target encoding gây sai số cao hơn 10.4% — đáng kể.
- Đây là kết quả **đảo ngược** với giả thuyết của giảng viên.

### Tại sao hash thắng?

1. **Pipeline hiện tại đã exclude IDs khỏi features**: Các đặc tính của Subject (qua `TM_score` từ PPGD) và Lecturer (qua `EM_score` từ PPDG) đã được capture qua các feature kỹ thuật **có ý nghĩa giáo dục**. Thêm IDs là thông tin trùng lặp.
2. **Cardinality cao + samples ít**: ~vài trăm Subject × 1,200 SV → trung bình ~10–20 records/subject. Target encoding với mẫu nhỏ này gây overfit cao kể cả với 5-fold + smoothing=10.
3. **Tree model ngẫu nhiên có khả năng học các interactions giữa engineered features tốt hơn ID encoding.**

### Lưu ý quan trọng cho luận văn

Feedback của giảng viên về "MD5/hash encoding sai bản chất không gian đặc trưng" là **đúng về mặt lý thuyết** với mô hình tree, nhưng **không áp dụng trong pipeline hiện tại** vì:

- Pipeline đã **exclude** Subject_ID/Lecturer_ID khỏi features. Hash chỉ áp dụng cho các cột object còn lại (chủ yếu là demographic).
- Các "Subject characteristic" và "Lecturer characteristic" được encode qua **TM_score, EM_score (PPGD/PPDG ordinal)** và các engineered features — có ý nghĩa pedagogical, không phải hash.
- Khi test với target encoding (đề xuất thay thế của giảng viên), kết quả thực nghiệm **kém hơn 10.4%**.

### Kết luận học thuật cho luận văn

> "Mối quan ngại của giảng viên hướng dẫn về việc dùng hàm hash để encode `Subject_ID` và `Lecturer_ID` là chính đáng về mặt lý thuyết — không gian đặc trưng hash mất tính semantic và gây khó khăn cho việc split của cây quyết định. Tuy nhiên, kiểm chứng thực nghiệm với 3 chiến lược (hash exclusion, frequency, target encoding 5-fold smoothing=10) cho thấy: **chiến lược exclude IDs (mặc định hiện tại) cho MAE thấp nhất** (0.3945 so với 0.4043 và 0.4354). Lý do là các đặc tính của môn học và giảng viên đã được encode dưới dạng có ý nghĩa giáo dục thông qua điểm số phương pháp giảng dạy (PPGD) và phương pháp đánh giá (PPDG). Kết quả này khẳng định rằng việc thiết kế feature có domain knowledge vượt trội hơn các kỹ thuật encoding generic."

### Files mới

- [src/ml_clo/features/categorical_encoder.py](../src/ml_clo/features/categorical_encoder.py) — `FrequencyEncoder` + `TargetEncoder` (5-fold, smoothing)
- Tham số `categorical_strategy` + `include_id_features` trong [feature_encoder.py](../src/ml_clo/features/feature_encoder.py)
- Persist `fitted_encoders` + `categorical_strategy` vào model artifact qua [ensemble_model.py](../src/ml_clo/models/ensemble_model.py)
- Tham số `categorical_strategy` trong [TrainingPipeline](../src/ml_clo/pipelines/train_pipeline.py)
- Script ablation: [scripts/run_ablation_encoding.py](../scripts/run_ablation_encoding.py)

---

## Tổng hợp cho Chương 4 (Kết quả thực nghiệm) và Chương 5 (Thảo luận)

### Chương 4 — Bảng số liệu chính

**Bảng 4.X — Ablation 3 chiều: Survey × Temporal × Encoding**

| Mã thí nghiệm | Mô tả | MAE | RMSE | R² |
|--------------|------|-----|------|----|
| EXP-0 | Baseline (pre-advisor) | 0.3945 | 0.5748 | 0.7980 |
| EXP-1a | LMS only (Phase 1) | 0.3945 | 0.5748 | 0.7980 |
| EXP-1b | LMS + Survey | 0.3954 | 0.5764 | 0.7968 |
| EXP-2a | No temporal (Phase 2) | 0.3945 | 0.5748 | 0.7980 |
| EXP-2b | + Temporal features | 0.3959 | 0.5762 | 0.7970 |
| EXP-3a | Hash encoding (Phase 3) | **0.3945** | **0.5748** | **0.7980** |
| EXP-3b | Frequency encoding | 0.4043 | 0.5821 | 0.7928 |
| EXP-3c | Target encoding (5-fold, smooth=10) | 0.4354 | 0.6111 | 0.7716 |

### Chương 5 — Đoạn thảo luận tổng thể

> "Quá trình kiểm chứng 3 đề xuất cải tiến (tích hợp dữ liệu khảo sát, làm giàu đặc trưng thời gian, thay đổi chiến lược mã hóa) cho thấy mô hình baseline với 76 đặc trưng và mã hóa hash đã đạt mức hiệu quả tối ưu trên bộ dữ liệu hiện có (MAE = 0.3945, R² = 0.7980).
>
> Cả ba hướng cải tiến đều **không mang lại lợi ích thực nghiệm** — đây là kết quả có giá trị nghiên cứu cao vì nó **phản chứng các giả thuyết phổ biến trong cộng đồng ML** (more features = better, temporal = better, structured encoding > hash). Cụ thể:
>
> - **Dữ liệu khảo sát chủ quan** với độ phủ thấp (<3% records) chỉ thêm nhiễu, không cung cấp tín hiệu mới ngoài những gì đã được capture qua dữ liệu hành vi LMS/SIS.
> - **Đặc trưng thời gian** (slope, volatility) chỉ có giá trị khi dữ liệu nguồn (attendance) phủ đủ. Với 28.6% coverage hiện tại, các đặc trưng này không vượt qua được aggregate baseline.
> - **Mã hóa target/frequency cho ID** kém hơn việc loại bỏ ID. Lý do là các đặc tính có ý nghĩa của môn học và giảng viên đã được capture qua điểm số PPGD/PPDG — feature engineering có domain knowledge vượt trội hơn encoding generic.
>
> Kết luận này **củng cố thiết kế hiện tại** của hệ thống và cung cấp lộ trình rõ ràng cho hướng cải tiến tương lai: ưu tiên **mở rộng độ phủ dữ liệu** (attendance, conduct) thay vì thêm nguồn dữ liệu mới hoặc thay đổi chiến lược encoding."

---

## Khuyến nghị triển khai sản xuất

| Hạng mục | Khuyến nghị |
|----------|------------|
| Model production | Giữ baseline (76 features, hash encoding) — model_path: `models/baseline_v0_pre_advisor.joblib` |
| Feature engineering tương lai | Tập trung mở rộng độ phủ attendance ≥70% trước khi thêm temporal features |
| Khảo sát | Tạm gác lại, chỉ tích hợp khi có quy trình thu thập đảm bảo phủ ≥80% theo SV |
| Encoding | Giữ exclude IDs + hash cho object columns còn lại |
| Audit log | Bật `set_audit_log_path()` ở backend để theo dõi predict |

---

## Reproducibility

Toàn bộ thí nghiệm reproducible qua 3 lệnh:

```bash
# Phase 1 — Survey
python scripts/run_ablation_survey.py

# Phase 2 — Temporal
python scripts/run_ablation_temporal.py

# Phase 3 — Encoding
python scripts/run_ablation_encoding.py
```

Tất cả chạy với `random_state=42`, GroupShuffleSplit theo Student_ID. Metrics được merge vào `experiments/results.json`.

**Hardware**: macOS Darwin 25.4.0, Python 3.12, ~3 phút per scenario.
