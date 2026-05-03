# Kế hoạch triển khai — Cập nhật theo Feedback

**Phiên bản:** 1.0  
**Ngày:** 2026-03-15  
**Tham chiếu:** [FEEDBACK_REPORT.txt](FEEDBACK_REPORT.txt)

---

## Tổng quan

Bản kế hoạch này triển khai các cải tiến dựa trên feedback đã được đánh giá (Phần 4 trong FEEDBACK_REPORT.txt).

## Thứ tự ưu tiên

| Bước | Mô tả | Ước lượng | Phụ thuộc |
|------|-------|-----------|-----------|
| 1 | Bổ sung Điểm danh vào Train | 1–2 ngày | — |
| 2 | Bổ sung Điểm danh vào Predict & Analyze | 1 ngày | Bước 1 |
| 3 | Logic Điểm tổng (SV năm 1 vs khác) | 1–2 ngày | — |
| 4 | XAI: Gắn nguồn dữ liệu vào lý do | 1 ngày | Bước 1, 2 |
| 5 | Kiểm tra & khắc phục kết quả trùng lặp | 0.5–1 ngày | Bước 1–3 |
| 6 | Output lớp: Loại bỏ/ẩn Average predicted | 0.5 ngày | — |
| 7 | Test & tài liệu | 1 ngày | Tất cả |

**Tổng ước lượng:** 6–8 ngày.

---

## Bước 1: Bổ sung File Điểm danh vào Train

### 1.1 Mục tiêu

Đưa file Điểm danh vào quá trình huấn luyện để mô hình học được mối liên hệ giữa chuyên cần và kết quả học tập.

### 1.2 Thay đổi chi tiết

| File | Hành động |
|------|-----------|
| `scripts/train.py` | Thêm `--attendance` (optional), truyền vào pipeline |
| `src/ml_clo/pipelines/train_pipeline.py` | Thêm `attendance_path` vào `load_data()`, `run()`; truyền `attendance_df` vào `create_training_dataset()` |
| `src/ml_clo/data/loaders.py` | Đã có `load_attendance()` — kiểm tra mapping cột (MSSV, Mã môn học, Năm học, Điểm danh) |
| `src/ml_clo/data/mergers.py` | Đã có `merge_attendance()` — không cần sửa |

### 1.3 Logic

- Nếu `--attendance` không truyền → chạy như hiện tại (attendance_rate = NaN).
- Nếu truyền → merge attendance, tính `attendance_rate` (0–1), đưa vào features.
- Feature "Chuyên cần" (`attendance_rate`) đã có trong `PEDAGOGICAL_GROUPS`.

### 1.4 Deliverables

- [x] `scripts/train.py` có `--attendance`
- [x] `train_pipeline.load_data()` load attendance khi có path
- [x] `train_pipeline.prepare_training_dataset()` truyền attendance_df vào create_training_dataset
- [x] Test: train với `--attendance` → test_train_with_attendance_produces_model (Bước 7)

---

## Bước 2: Bổ sung Điểm danh vào Predict & Analyze

### 2.1 Mục tiêu

Tích hợp điểm danh vào dự đoán cá nhân và phân tích lớp để kết quả phản ánh chuyên cần.

### 2.2 Thay đổi chi tiết

| File | Hành động |
|------|-----------|
| `scripts/predict.py` | Thêm `--attendance` (optional) |
| `scripts/analyze_class.py` | Thêm `--attendance` (optional) |
| `src/ml_clo/pipelines/predict_pipeline.py` | Thêm `attendance_path` vào init, `load_data_cache()`, `load_student_data()`; truyền vào `create_training_dataset()` / `create_student_record_from_ids` flow |
| `src/ml_clo/pipelines/analysis_pipeline.py` | Thêm `attendance_path` vào `analyze_class_from_scores()`, `load_class_data()`, `analyze_class()`; merge attendance khi build features cho từng SV |

### 2.3 Lưu ý

- `create_student_record_from_ids()` không merge attendance trực tiếp. Cần mở rộng hoặc gọi `merge_attendance()` sau khi có base_df.
- Xem `mergers.merge_all_data_sources()` và `create_training_dataset()` để đảm bảo attendance được truyền đúng.

### 2.4 Deliverables

- [x] `scripts/predict.py` có `--attendance`
- [x] `scripts/analyze_class.py` có `--attendance`
- [x] `PredictionPipeline` load và merge attendance khi predict
- [x] `AnalysisPipeline` merge attendance khi analyze_class_from_scores (có MSSV + demographics)

---

## Bước 3: Logic sử dụng File Điểm tổng (SV năm 1 vs khác)

### 3.1 Mục tiêu

- **SV năm 1:** Chỉ khi hoàn toàn không có lịch sử mới được fallback (không dùng Điểm tổng).
- **SV năm 2+:** Bắt buộc dùng Điểm tổng nếu có; không cho fallback khi thiếu.

### 3.2 Thiết kế

| Khái niệm | Cách xác định |
|------------|---------------|
| **SV năm 1** | Năm nhập học = năm hiện tại (hoặc năm học hiện tại); không có bản ghi nào trong DiemTong |
| **SV năm 2+** | Có ít nhất 1 bản ghi trong DiemTong (đã học ít nhất 1 môn) |

### 3.3 Thay đổi chi tiết

| File | Hành động |
|------|-----------|
| `src/ml_clo/pipelines/predict_pipeline.py` | Khi không có exam_scores: kiểm tra xem SV có trong DiemTong không (theo student_id). Nếu CÓ → bắt buộc yêu cầu exam_scores_path, raise nếu thiếu. Nếu KHÔNG (SV mới/năm 1) → cho phép fallback. |
| `src/ml_clo/data/mergers.py` hoặc loaders | Hàm `student_has_history(exam_df, student_id) -> bool` |
| `docs/data_model.md` | Ghi rõ nguồn "năm học" (nhân khẩu? đăng ký?) nếu cần xác định SV năm 1 chính xác hơn |

### 3.4 Lưu ý

- Hiện tại fallback không cần exam_scores. Logic mới: fallback CHỈ khi SV chưa có trong DiemTong.
- Nếu SV có trong DiemTong nhưng user không truyền exam_scores_path → có thể load và filter thay vì fallback. Cần xác định rõ use case.

### 3.5 Deliverables

- [x] Hàm kiểm tra SV có lịch sử trong DiemTong
- [x] Predict: fallback chỉ khi SV không có trong DiemTong
- [x] Raise/error rõ ràng khi SV năm 2+ mà không có exam_scores
- [x] Cập nhật help/docstring

---

## Bước 4: XAI — Gắn nguồn dữ liệu vào lý do

### 4.1 Mục tiêu

Lời giải thích (reason_text) cần gắn rõ nguồn dữ liệu: "dựa vào file điểm danh", "dựa vào file nhân khẩu", "dựa vào file tự học", v.v.

### 4.2 Thay đổi chi tiết

| File | Hành động |
|------|-----------|
| `src/ml_clo/reasoning/reason_generator.py` | Mở rộng template: thêm tham số `data_source` (điểm danh, nhân khẩu, tự học, rèn luyện, PPGD, PPDG, điểm tổng). Sinh câu: "Dựa vào file [X], ..." |
| `src/ml_clo/config/xai_config.py` hoặc mới | Mapping: nhóm sư phạm → nguồn dữ liệu (Chuyên cần → điểm danh, Tự học → tuhoc, Rèn luyện → diemrenluyen, Cá nhân → nhân khẩu, Học lực → điểm tổng, Giảng dạy → PPGD, Đánh giá → PPDG) |
| `src/ml_clo/reasoning/templates.py` | Template cho từng nhóm có thể nhúng `{data_source}` |

### 4.3 Ví dụ output mong muốn

- "Bạn thường không tham gia đầy đủ các buổi học (dựa vào file điểm danh)."
- "Do giới tính ảnh hưởng đến kết quả của bạn (dựa vào file nhân khẩu)."

### 4.4 Deliverables

- [x] Mapping group → data_source
- [x] Template/sinh lý do có gắn nguồn dữ liệu
- [x] Output IndividualAnalysisOutput, ClassAnalysisOutput có reason_text chứa "(dựa vào file X)"

---

## Bước 5: Kiểm tra & khắc phục kết quả trùng lặp

### 5.1 Mục tiêu

Đảm bảo mỗi sinh viên với input khác nhau có kết quả dự đoán khác nhau.

### 5.2 Công việc

1. **Test tạo sẵn:** Viết script test: predict cho 5–10 SV khác nhau (cùng môn, cùng GV) với data đầy đủ (exam_scores, conduct, demographics, study_hours, attendance). So sánh predicted_clo_score → phải khác nhau nếu features khác.
2. **Phân tích:** Nếu vẫn trùng → kiểm tra:
   - Feature values có thực sự khác nhau không (logging/debug)
   - Prepare_features có fill NaN bằng giá trị chung quá mạnh không
   - Ensemble weights, model có vấn đề không
3. **Khắc phục:** Điều chỉnh logic fill NaN, hoặc đảm bảo luôn truyền đủ nguồn data (conduct, study_hours, attendance) khi có.

### 5.3 Deliverables

- [x] Test script kiểm tra tính phân biệt kết quả
- [x] Nếu có lỗi: báo cáo và sửa
- [x] Doc: khuyến nghị truyền đủ nguồn khi predict để có kết quả cá nhân hóa

---

## Bước 6: Output lớp — Loại bỏ/ẩn Average predicted score

### 6.1 Mục tiêu

Theo feedback: loại bỏ "Average predicted CLO score" khi dự đoán cấp lớp.

### 6.2 Lựa chọn

- **A. Loại bỏ:** Xóa `average_predicted_score` khỏi schema và output. Có thể break API hiện tại.
- **B. Ẩn tùy chọn:** Thêm tham số `include_average_predicted: bool = False` cho ClassAnalysisOutput; mặc định không trả về.
- **C. Đổi tên/context:** Giữ field nhưng đổi mục đích (ví dụ chỉ dùng nội bộ, không hiển thị UI).

### 6.3 Đề xuất

Dùng **B**: Giữ schema, thêm tùy chọn. Backend/CLI mặc định `include_average_predicted=False` khi gọi API lớp.

### 6.4 Thay đổi

| File | Hành động |
|------|-----------|
| `src/ml_clo/outputs/schemas.py` | `ClassAnalysisOutput` có tham số tùy chọn cho việc có serialize average_predicted_score hay không (hoặc `to_dict(include_average=False)`) |
| `src/ml_clo/pipelines/analysis_pipeline.py` | Khi tạo ClassAnalysisOutput cho "phân tích lớp" → không set average_predicted_score hoặc set None (tùy thiết kế) |
| CLI/API | Document: field này deprecated hoặc optional |

### 6.5 Deliverables

- [x] Quyết định A/B/C (chọn B: ẩn tùy chọn)
- [x] Cập nhật schema/pipeline theo quyết định
- [x] Cập nhật docs

---

## Bước 7: Test & Tài liệu

### 7.1 Test

- [x] Unit test: load_attendance (tests/unit/test_data/test_loaders.py), merge_attendance (tests/unit/test_data/test_mergers.py)
- [x] Integration test: train với attendance → predict với attendance (tests/integration/test_attendance_and_student_logic.py)
- [x] Test logic SV năm 1: predict không exam_scores cho SV mới → OK (test_fallback_ok_for_student_not_in_diemtong)

### 7.2 Tài liệu

- [x] README: thêm `--attendance` vào ví dụ train, predict, analyze_class
- [x] docs/data_model.md: đã có mô tả điểm danh (IV. Dữ liệu điểm danh Khoa FIRA.xlsx), không cần sửa
- [x] docs/FEEDBACK_REPORT.txt: đánh dấu [x] các mục đã xử lý (PHẦN 5)
- [x] docs/IMPLEMENTATION_PLAN_FEEDBACK.md: cập nhật [x] deliverables khi hoàn thành

---

## Phụ lục: Thứ tự thực hiện đề xuất

```
Bước 1 (Điểm danh Train)
    ↓
Bước 2 (Điểm danh Predict & Analyze)
    ↓
Bước 4 (XAI gắn nguồn) — song song hoặc sau Bước 2
    ↓
Bước 3 (Logic Điểm tổng) — có thể làm song song Bước 1
    ↓
Bước 5 (Kiểm tra trùng lặp)
    ↓
Bước 6 (Output lớp)
    ↓
Bước 7 (Test & docs)
```

---

## Liên hệ với FEEDBACK_REPORT

| Feedback (FEEDBACK_REPORT) | Bước trong plan |
|---------------------------|-----------------|
| 2.1 Điểm danh Train      | Bước 1          |
| 2.2 Đa nguồn Predict      | Bước 2          |
| 2.3 Logic Điểm tổng      | Bước 3          |
| 2.4 Kết quả trùng lặp    | Bước 5          |
| 2.5 XAI map nguồn        | Bước 4          |
| 2.6 Thang điểm; Average  | Bước 6 (thang 6 đã có) |

---

## Hoàn tất (đã triển khai xong Bước 1–7)

Ngày hoàn thành: 2026-03-10

| Bước | Trạng thái |
|------|------------|
| 1. Điểm danh Train | ✓ |
| 2. Điểm danh Predict & Analyze | ✓ |
| 3. Logic Điểm tổng (SV năm 1 vs 2+) | ✓ |
| 4. XAI gắn nguồn dữ liệu | ✓ |
| 5. Kiểm tra trùng lặp | ✓ |
| 6. Ẩn Average predicted | ✓ |
| 7. Test & Tài liệu | ✓ |

**Kiểm tra cuối:** `pytest tests/ -v` (unit + integration)

---

## Bug fix round (2026-04-10)

Sau triển khai feedback plan, đã phát hiện và fix 40+ issues trong 4 đợt.
Chi tiết: xem [ISSUES.md](../ISSUES.md).

Các thay đổi liên quan đến feedback plan:

- **Bước 2 (Điểm danh):** `create_student_record_from_ids` giờ nhận `study_hours_df` (MISSING-07) — virtual record đầy đủ hơn
- **Bước 4 (XAI):** SHAP anomaly-aware blending (NEW-01) — SHAP giải thích đúng khi `gb_low_anomaly` kích hoạt
- **Bước 5 (Trùng lặp):** Hash encoding v2 (NEW-02/03) thay LabelEncoder — deterministic, không cần fit
- **Bước 7 (Test):** 14 tests mới (`tests/unit/test_missing05.py`) cho gap coverage

**Model retrained:** v1.0_20260410_162858, encoding=hash_v2, MAE=0.3945, R²=0.7980, 106 tests passed.
