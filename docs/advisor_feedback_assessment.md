# Đánh giá tính khả thi — Feedback từ Giảng viên Hướng dẫn

**Ngày:** 2026-05-03
**Phiên bản:** 1.0
**Phạm vi:** Đánh giá 4 đề xuất từ giảng viên về model `ml_clo`

---

## Tổng quan

| # | Feedback | Khả thi | Effort | ROI | Ưu tiên |
|---|---------|---------|--------|-----|---------|
| 1 | Temporal Feature Engineering (slope, variance theo tuần) | ✅ Cao | 2–3 ngày | Cao | **Nên làm** |
| 2 | Ablation Study (Baseline vs Extended) | ✅ Rất cao | 1–2 ngày | Rất cao | **Bắt buộc** |
| 3 | MD5 / Hash encoding cho tree model | ⚠️ Trung bình | 2–4 ngày | Cao | **Nên làm có chọn lọc** |
| 4 | Đổi SHAP sang causal inference | ❌ Thấp | 1–3 tuần | Trung bình | **Không khuyến nghị** |

---

## 1. Temporal Feature Engineering ✅ Khả thi cao

### Đề xuất gốc của giảng viên

Dùng kỹ thuật **Temporal Feature Engineering** (Trích xuất đặc trưng thời gian). Dựa trên biểu đồ biến động theo tuần học, thêm các biến phái sinh:

- **Xu hướng chuyên cần:** Độ dốc (Slope) chuyên cần của 3 tuần gần nhất.
- **Tính ổn định:** Phương sai (Variance) của số giờ tự học thư viện qua các tuần.

Thêm một đoạn vào Chương 5 luận văn khẳng định:
> "Việc làm giàu đặc trưng (Feature Engineering) cho mô hình Ensemble giúp mô hình bắt được xu hướng thời gian mà vẫn duy trì được độ phức tạp tính toán thấp, phù hợp với ràng buộc hạ tầng của Trường Đại học Bình Dương."

### Hiện trạng code

- File `Dữ liệu điểm danh Khoa FIRA.xlsx` **đã có cột `Ngày`, `Buổi`, `Niên khoá`, `Học kì`** ([loaders.py:357-402](../src/ml_clo/data/loaders.py#L357-L402)) — **đủ dữ liệu cho temporal features**.
- File `tuhoc.xlsx` có `accumulated_study_hours` theo `year` — chưa có chi tiết tuần.
- Pipeline hiện tại chỉ aggregate (sum, mean) — **chưa khai thác trục thời gian**.

### Đề xuất triển khai cụ thể

| Feature mới | Công thức | Nguồn |
|-------------|----------|-------|
| `attendance_slope_3w` | `numpy.polyfit(weeks[-3:], rates[-3:], 1)[0]` | Attendance |
| `attendance_slope_full` | Slope cả kỳ | Attendance |
| `attendance_volatility` | `std(weekly_rates)` | Attendance |
| `study_hours_variance` | `var(weekly_hours)` | tuhoc.xlsx (nếu có week column) |
| `late_streak_max` | Số tuần liên tiếp trễ ≥ 1 buổi | Attendance |
| `early_dropoff_flag` | 1 nếu attendance giảm > 30% giữa kỳ | Attendance |

**Vị trí code:** [src/ml_clo/features/feature_builder.py](../src/ml_clo/features/feature_builder.py) — thêm hàm `build_temporal_attendance_features()`.

### Rủi ro

Nếu file `tuhoc.xlsx` chỉ có cột `year` (không có tuần), thì slope/variance theo tuần cho self-study **không khả thi** — cần kiểm tra cấu trúc file thật. Slope cho attendance làm được ngay vì có cột `Ngày`.

### Lợi ích kỳ vọng

- MAE giảm ~5–10% (kinh nghiệm với features dạng này trên tree ensemble)
- Bằng chứng học thuật mạnh cho luận văn
- Computational cost không đáng kể (vectorized numpy)

---

## 2. Ablation Study ✅ Bắt buộc làm

### Đề xuất gốc của giảng viên

Thiết kế Ablation Study (Nghiên cứu cắt bỏ), chạy 2 phiên bản mô hình:

- **Baseline:** Chỉ dùng dữ liệu Hệ thống (LMS/SIS) — điểm số, rèn luyện, chuyên cần.
- **Mở rộng:** Dữ liệu Hệ thống + Dữ liệu khảo sát (giờ ngủ, hỗ trợ gia đình...).

Lập bảng so sánh **MAE, RMSE, R²** của hai kịch bản.

- Nếu dữ liệu khảo sát giảm sai số → có bằng chứng toán học vững chắc.
- Nếu không cải thiện → kết luận:
  > "Kết quả chứng minh dữ liệu hành vi LMS/kho có sức mạnh dự báo tuyệt đối, dữ liệu khảo sát chủ quan chứa độ nhiễu cao và có thể lược bỏ để tối ưu hóa hệ thống chạy tự động."

### Đánh giá

**Đây là feedback mạnh nhất, giá trị học thuật cao nhất.**

### Triển khai

```python
# scripts/ablation_study.py
scenarios = {
    "baseline_lms":    [exam, conduct, attendance, demographics, ppgd, ppdg],  # KHÔNG có khảo sát
    "extended_survey": [...baseline..., survey],                                # CÓ khảo sát
}

for name, sources in scenarios.items():
    pipeline = TrainingPipeline(...)
    metrics = pipeline.run(...)
    results[name] = {"MAE": metrics.test_mae, "RMSE": ..., "R²": ...}
```

### Lợi ích

- **Bảng số liệu so sánh cho luận văn** — dạng bằng chứng vững chắc nhất
- Trả lời được câu hỏi: "Có cần thu thập dữ liệu khảo sát không?" → quyết định triển khai thực tế
- File `Khảo sát...xlsx` (280 SV) hiện **chưa được tích hợp** — đây là cơ hội tốt

### Effort thấp

Pipeline đã tách module rõ ràng, chỉ cần wrapper script chạy 2 lần với configurations khác nhau.

### Khuyến nghị

Làm trước Feedback 1 — kết quả ablation sẽ định hướng có đáng đầu tư temporal features không.

---

## 3. MD5 / Hash encoding cho tree model ⚠️ Đúng về lý thuyết, cần xử lý có chọn lọc

### Đề xuất gốc của giảng viên

> "Xem kỹ phần MD5 nhé, nếu dùng MD5 để biến `Subject_ID` và `Lecturer_ID` thành số nguyên đưa vào mô hình Cây (Random Forest, Gradient Boosting) là sai bản chất không gian đặc trưng."

### Làm rõ

Code **không dùng MD5**, mà dùng **SHA-256** ([hash_utils.py:41](../src/ml_clo/utils/hash_utils.py#L41)). Tuy nhiên **bản chất feedback của giảng viên đúng** — bất kỳ hash nào (MD5/SHA-256/MurmurHash) đều có cùng vấn đề khi đưa vào tree model.

### Vấn đề lý thuyết (giảng viên nói đúng)

Tree model học bằng cách **split theo ngưỡng số**: `if feature_X > threshold then ...`. Khi hash:

- `Subject_ID="INF0823"` → 1,234,567,890
- `Subject_ID="INF0824"` → 87,654,321 (số rất khác)
- `Subject_ID="INF0825"` → 2,000,000,000

Hai môn liền kề về **mặt ngữ nghĩa** (cùng khoa, cùng năm) lại có khoảng cách số **hỗn loạn**. Tree split kiểu `Subject_hash > 1,500,000,000` không có ý nghĩa pedagogical → model phải tốn nhiều tree depth để ghép các giá trị hash thành nhóm có ý nghĩa.

### Mức độ tác động thực tế

Với RF/GB và cardinality thấp đến trung bình (~vài trăm subjects, ~vài chục lecturers), tree vẫn học được nhưng **kém hiệu quả** so với encoding có cấu trúc.

### Đề xuất thay thế (xếp theo độ phức tạp)

| Phương án | Khi nào dùng | Effort | Đánh đổi |
|----------|-------------|--------|----------|
| **Target encoding** (mean exam_score per Subject_ID) | Cardinality cao, có dữ liệu lịch sử | Trung bình | Cần handle leakage bằng K-fold |
| **Frequency encoding** (số lần xuất hiện) | Đơn giản, robust | Thấp | Mất ý nghĩa khi tần suất giống nhau |
| **One-hot** (Subject) + ordinal (Lecturer) | Cardinality < 100 | Thấp | Tăng dimension, OK với tree |
| **Embedding học từ data** | Có nhiều data | Cao | Chỉ đáng làm với deep model |
| **Giữ hash + hỗ trợ unknown handling** | Production stability | Đã có | Rủi ro học thuật |

### Khuyến nghị

- `Subject_ID` (~vài chục đến vài trăm): **target encoding với 5-fold CV** (chống leakage)
- `Lecturer_ID` (~vài chục): **one-hot** hoặc target encoding
- **Giữ hash làm fallback** cho `__UNKNOWN__` (entity mới chưa có trong train)
- Chạy ablation: Hash vs Target Encoding → bằng chứng cho luận văn

**Vị trí code:** [src/ml_clo/features/feature_encoder.py](../src/ml_clo/features/feature_encoder.py) — thêm tham số `categorical_strategy: Literal["hash", "target", "frequency"]`.

---

## 4. Đổi SHAP sang Causal Inference ❌ Không khuyến nghị

### Đề xuất gốc của giảng viên

> "Mô hình SHAP xem thử đổi qua nguyên nhân kết quả được."

### Phân tích

- **SHAP ≠ Causal** là điều **đúng về lý thuyết** — SHAP là attribution dựa trên correlation/marginal contribution, không phải nhân quả thực sự.
- Nhưng để có causal inference đúng nghĩa cần:
  - **DAG (Directed Acyclic Graph)** mô tả giả định nhân quả → cần kiến thức domain rất sâu
  - **Counterfactual data** hoặc **RCT** (thí nghiệm có kiểm soát) — không có
  - **Confounders identification** — risky nếu thiếu biến quan trọng
  - Methods: DoWhy, EconML, instrumental variables → phức tạp gấp nhiều lần SHAP

### Vấn đề thực tế

- Dataset 1,234 mẫu, 280 khảo sát → **không đủ statistical power** cho causal inference đáng tin cậy
- Không có intervention data (không thể "thay đổi giảng viên" rồi đo lại điểm)
- Risk cao: causal claims sai có thể **làm yếu luận văn** hơn là giữ SHAP với caveat đúng

### Đề xuất compromise (làm được ngay, giá trị học thuật cao)

1. **Giữ SHAP** + **tuyên bố rõ trong luận văn**:
   > "SHAP cung cấp attribution dựa trên correlation, không phải causal. Để khẳng định nhân quả cần thí nghiệm có kiểm soát."

2. **Thêm sensitivity analysis** (rẻ, mạnh hơn pure SHAP):
   - **Permutation importance** — đảo random một feature, đo MAE thay đổi
   - **Partial Dependence Plots (PDP)** — quan hệ feature ↔ output
   - **Individual Conditional Expectation (ICE)** — counterfactual nhẹ

3. **Causal-flavored interpretation**: Phân tích các cặp feature có khả năng gây nhiễu (confounder) — ví dụ: `attendance` vs `study_hours` thường correlate, cần thận trọng khi giải thích.

### Lý do từ chối hoàn toàn việc đổi sang causal

- Effort cao (1–3 tuần)
- Risk cao (causal claims sai)
- Không phù hợp với dataset hiện tại
- SHAP với caveat đúng là chuẩn thực hành trong học thuật ML hiện nay

---

## Lộ trình đề xuất

```
Tuần 1: Ablation Study (Feedback 2)
  └─ Tích hợp Khảo sát...xlsx → so sánh Baseline vs Extended
  └─ Báo cáo bảng MAE/RMSE/R² cho luận văn

Tuần 2: Temporal Features (Feedback 1)
  └─ Implement attendance_slope, attendance_volatility
  └─ Thêm đoạn vào Chương 5 luận văn về Feature Engineering

Tuần 3: Encoding Ablation (Feedback 3 — phiên bản gọn)
  └─ Implement target_encoding cho Subject_ID
  └─ So sánh Hash vs Target Encoding → 1 bảng số liệu

Feedback 4: Bổ sung permutation importance + PDP
  └─ Effort: 1 ngày, không thay đổi SHAP
  └─ Thêm caveat về causal trong luận văn
```

**Tổng effort:** ~2–3 tuần — khả thi trong scope luận văn, mỗi feedback đều có **bằng chứng số liệu** để báo cáo.

---

## Tóm tắt khuyến nghị

| Feedback | Quyết định |
|----------|-----------|
| 1. Temporal FE | ✅ Triển khai (sau Ablation) |
| 2. Ablation Study | ✅ Triển khai NGAY — ưu tiên cao nhất |
| 3. Hash encoding | ⚠️ Triển khai bản gọn — target encoding cho Subject_ID + giữ hash cho unknown |
| 4. Causal inference | ❌ Không thay SHAP — bổ sung permutation importance + PDP + caveat trong luận văn |

**Bước tiếp theo đề xuất:** Bắt đầu với **Ablation Study** vì cho ROI cao nhất và validate được giá trị của file khảo sát chưa dùng.
