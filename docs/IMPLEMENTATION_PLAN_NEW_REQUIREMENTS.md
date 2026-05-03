# Kế hoạch triển khai — Yêu cầu mới

**Phiên bản:** 1.1  
**Ngày:** 2026-03-10  
**Tham chiếu:** [FEASIBILITY_REPORT_NEW_REQUIREMENTS.md](FEASIBILITY_REPORT_NEW_REQUIREMENTS.md)

---

## Lưu ý: Thay thế, không mở rộng

Yêu cầu cũ không đáp ứng nhu cầu → triển khai theo hướng **thay thế hoàn toàn**:

- **Loại bỏ** các tính năng cũ tương ứng
- **Xây dựng lại** các thành phần cần thiết
- Không duy trì song song hai phiên bản lâu dài

---

## Tổng quan các bước

| Bước | Mô tả | Ước lượng |
|------|-------|-----------|
| 0 | Xác định phạm vi loại bỏ (deprecation/removal) | 0.5 ngày |
| 1 | **Xây lại** nguồn dữ liệu & ID | 3–4 ngày |
| 2 | **Xây lại** pipeline cá nhân | 1–2 ngày |
| 3 | **Xây lại** pipeline phân tích lớp | 2–3 ngày |
| 4 | Loại bỏ code cũ, cập nhật CLI | 0.5–1 ngày |
| 5 | Test & tài liệu | 1–2 ngày |

---

## Bước 0: Xác định phạm vi loại bỏ

### 0.1 Mục tiêu

Liệt kê rõ API, hàm, logic nào sẽ bị **loại bỏ** hoặc **deprecate**.

### 0.2 Thành phần có thể loại bỏ/thay thế

| Thành phần hiện tại | Hành động đề xuất |
|--------------------|-------------------|
| `AnalysisPipeline.load_class_data()` — filter DiemTong theo subject+lecturer | **Thay thế** — không dùng DiemTong làm nguồn chính cho class analysis |
| `PredictionPipeline.load_student_data()` — bắt buộc có record trong DiemTong | **Thay thế** — dùng nhân khẩu + PPGD/PPDG làm nguồn chính |
| `analyze_class()` — đầu vào subject_id, lecturer_id, path đến DiemTong | **Thay thế** — đầu vào mới: danh sách điểm CLO |
| `predict()` — bắt buộc có student+subject+lecturer trong DiemTong | **Thay thế** — cho phép MSSV từ nhân khẩu, lecturer mặc định |
| Scripts: `--exam-scores` bắt buộc cho predict | **Thay đổi** — exam-scores không còn bắt buộc khi có nhân khẩu + PPGD/PPDG |

### 0.3 Deliverables

- [x] Danh sách chính thức các API/code bị deprecate hoặc xóa → `docs/DEPRECATION_SCOPE.md`
- [x] Migration note (nếu cần cho user/backend cũ)

---

## Bước 1: Xây lại nguồn dữ liệu & ID

### 1.1 Mục tiêu

- MSSV có thể chỉ lấy từ `nhankhau.xlsx` (không bắt buộc có trong DiemTong).
- Subject_ID có thể lấy từ PPGD/PPDG (môn mới dùng TM/EM mặc định).
- Lecturer_ID được phép không tồn tại (GV mới → encoding mặc định).

### 1.2 Thay đổi chi tiết (xây lại, không patch)

#### 1.2.1 Nguồn dữ liệu mới

**File:** `src/ml_clo/data/` — loaders, mergers, có thể thêm module mới

- **Xây lại** logic lấy entity: MSSV từ `nhankhau.xlsx`, Subject từ PPGD/PPDG (hoặc file môn riêng), Lecturer = placeholder.
- Hàm `create_student_record_from_ids()` (hoặc tương đương): tạo record từ (student_id, subject_id, lecturer_id) **không** cần DiemTong.
- **Loại bỏ** phụ thuộc: không còn bắt buộc (Student_ID, Subject_ID, Lecturer_ID) phải có trong DiemTong.

**Logic mới:**
- Student: lấy từ `nhankhau` theo Student_ID (left join).
- Subject: lấy từ PPGD/PPDG theo Subject_ID (left join); nếu thiếu → TM/EM = 0.
- Lecturer: không có nguồn → dùng placeholder (ví dụ: `__UNKNOWN__`).

#### 1.2.2 LabelEncoder cho entity mới

**File:** `src/ml_clo/pipelines/train_pipeline.py`, `predict_pipeline.py`, `feature_builder` hoặc nơi encode

- Khi `fit` encoder: thêm class `"__UNKNOWN__"` hoặc tương đương.
- Khi `transform`: nếu gặp giá trị chưa thấy → map sang `"__UNKNOWN__"`.
- Lưu encoder trong model artifact để dùng khi predict.

#### 1.2.3 Xây lại logic load trong PredictionPipeline

- **Thay thế** `load_student_data()`: nguồn chính là nhân khẩu + PPGD/PPDG, không còn bắt buộc DiemTong.
- Luồng mới: student_id + subject_id + lecturer_id → build record từ nhân khẩu, PPGD/PPDG, lecturer placeholder.
- `create_training_dataset()` (hoặc hàm tương đương): hỗ trợ input “ảo” không có exam_score.

### 1.3 Deliverables

- [x] Hàm `create_student_record_from_ids()` trong `src/ml_clo/data/mergers.py`.
- [x] Cập nhật logic encoding để xử lý giá trị mới (unknown) — `prepare_features` dùng -1 cho unseen.
- [x] PredictionPipeline chạy được với MSSV chỉ có trong nhankhau, Subject trong PPGD, Lecturer mới.

---

## Bước 2: Xây lại pipeline cá nhân

### 2.1 Mục tiêu

- **Môn đã học & đã đỗ:** Dùng điểm CLO thực tế (`actual_clo_score`), trả về nguyên nhân và cách khắc phục (SHAP vẫn chạy trên feature; output ưu tiên actual).
- **Môn chưa học:** Dự đoán điểm từ điểm tích lũy, thời gian, v.v., trả về điểm dự đoán + nguyên nhân.
- **Nguồn dữ liệu:** Không bắt buộc DiemTong; dùng nhân khẩu + PPGD/PPDG.

### 2.2 Thay đổi chi tiết (xây lại)

**File:** `src/ml_clo/pipelines/predict_pipeline.py`, `src/ml_clo/outputs/schemas.py`

- **Thay thế** logic load: dùng nguồn từ Bước 1 (nhân khẩu, PPGD/PPDG).
- **Sửa** `predict()`: thêm `actual_clo_score: Optional[float]` — khi có → output ưu tiên actual, vẫn dùng SHAP cho lý do.
- Schema: `IndividualAnalysisOutput` có `actual_clo_score: Optional[float]`.

### 2.3 Deliverables

- [x] Tham số `actual_clo_score` trong `predict()`.
- [x] Schema và JSON output hỗ trợ `actual_clo_score`.
- [x] Cập nhật `scripts/predict.py` với `--actual-score` (optional).

---

## Bước 3: Xây lại pipeline phân tích lớp

### 3.1 Mục tiêu

**Thay thế** phân tích lớp cũ (filter DiemTong theo subject+lecturer) bằng **chế độ mới**: đầu vào là môn, mã GV, danh sách điểm CLO.

### 3.2 API mới (thay thế API cũ)

**File:** `src/ml_clo/pipelines/analysis_pipeline.py`

**Thay thế** `analyze_class(subject_id, lecturer_id, exam_scores_path=...)` bằng API mới (hoặc bổ sung và deprecate cũ):

```python
def analyze_class_from_scores(
    self,
    subject_id: str,
    lecturer_id: str,
    clo_scores: Union[Dict[str, float], List[float], List[Tuple[str, float]]],
    demographics_path: Optional[str] = None,
    teaching_methods_path: Optional[str] = None,
    assessment_methods_path: Optional[str] = None,
    ...
) -> ClassAnalysisOutput
```

**Trong đó:**
- `clo_scores`: có thể là:
  - `Dict[student_id, score]`
  - `List[score]` (chỉ điểm, không MSSV)
  - `List[(student_id, score)]`

### 3.3 Logic xử lý

1. **Có MSSV:**
   - Với mỗi (student_id, score): build feature từ nhân khẩu, PPGD/PPDG, lecturer placeholder.
   - Chạy model (predict) + SHAP.
   - Gom SHAP theo nhóm sư phạm, tạo lý do và giải pháp.

2. **Không MSSV (chỉ danh sách điểm):**
   - Tính thống kê mô tả (mean, median, std, phân phối).
   - Không dùng SHAP; tạo “nguyên nhân” dựa trên phân phối (ví dụ: “Đa số sinh viên có điểm thấp”, “Phân tán điểm cao”).
   - Giải pháp dùng template chung cho lớp.

### 3.4 Deliverables

- [x] Phương thức `analyze_class_from_scores()`.
- [x] Xử lý cả 3 dạng `clo_scores`.
- [x] Cập nhật `scripts/analyze_class.py` với chế độ `--scores-file`.

---

## Bước 4: Loại bỏ code cũ, cập nhật CLI

### 4.1 Loại bỏ / thay đổi

- [x] Deprecate: `analyze_class()` cũ (filter DiemTong), `load_class_data()` — log warning khi gọi.
- [x] Giữ tương thích: vẫn hoạt động, không xóa.

### 4.2 Scripts

- [x] **predict.py:** --exam-scores không còn bắt buộc; --actual-score; cập nhật help.
- [x] **analyze_class.py:** --scores-file là chế độ chính; --exam-scores deprecated; cập nhật help.

### 4.3 Ví dụ sử dụng (sau khi thay thế)

```bash
# Cá nhân — môn đã học, truyền điểm thực
python scripts/predict.py \
  --model models/model.joblib \
  --student-id 19050006 \
  --subject-id INF0823 \
  --lecturer-id 90316 \
  --exam-scores data/DiemTong.xlsx \
  --actual-score 4.2

# Phân tích lớp từ file điểm
python scripts/analyze_class.py \
  --model models/model.joblib \
  --subject-id INF0823 \
  --lecturer-id 90316 \
  --scores-file data/clo_scores_hk1.csv
```

---

## Bước 5: Test & tài liệu

### 5.1 Test

- [x] Unit test cho `create_student_record_from_ids()` — `tests/unit/test_data/test_mergers.py`.
- [x] Unit test cho encoding giá trị mới (unknown) — Lecturer placeholder `__UNKNOWN__`.
- [x] Integration test: predict với MSSV chỉ có trong nhankhau.
- [x] Integration test: `analyze_class_from_scores()` với Dict và List.
- [x] Cập nhật `test_pipelines.py` cho các luồng mới.

### 5.2 Tài liệu

- [x] Cập nhật README: mô tả chế độ mới, ví dụ lệnh.
- [x] Cập nhật `docs/model_requirements.md`: thêm mục “Yêu cầu mới”.
- [x] Cập nhật `docs/data_model.md`: mục IX "Nguồn dữ liệu mới".

---

## Thứ tự thực hiện đề xuất

```
Bước 0 (Phạm vi loại bỏ)
    ↓
Bước 1 (Xây lại nguồn dữ liệu & ID)
    ↓
Bước 2 (Xây lại pipeline cá nhân)
    ↓
Bước 3 (Xây lại pipeline phân tích lớp)
    ↓
Bước 4 (Loại bỏ code cũ, CLI)
    ↓
Bước 5 (Test & docs)
```

---

## Phụ lục: Cấu trúc file điểm CLO gợi ý

**CSV (clo_scores.csv):**
```
student_id,clo_score
19050006,4.2
19050007,3.8
19050008,5.1
```

**JSON:**
```json
{
  "scores": [
    {"student_id": "19050006", "clo_score": 4.2},
    {"student_id": "19050007", "clo_score": 3.8}
  ]
}
```

Hoặc đơn giản (chỉ điểm):
```json
{"scores": [4.2, 3.8, 5.1]}
```

---

## Phụ lục: Bug fix round (2026-04-10)

Sau khi triển khai xong các bước 0–5, đã thực hiện 4 đợt fix toàn bộ lỗi phát sinh (40+ items, xem [ISSUES.md](../ISSUES.md)):

| Đợt | Focus | Items chính |
| --- | --- | --- |
| 1 | Correctness (HIGH) | BUG-01..06, NEW-02/03, DESIGN-01/10 |
| 2 | XAI Trustworthiness | NEW-01 (anomaly SHAP), NEW-04/05, DESIGN-05/06 |
| 3 | Robustness | DATA-01/03/07, dedup NaN-safe |
| 4 | Design + PERF + MISSING | Shared feature_encoder, vectorized features, impact bands, set_weights, predict_with_uncertainty, k-fold CV, data quality report, audit log, CLI validation, study_hours passthrough, SHAP clear_cache, index-based merges |

**Kết quả:** 37/41 fixed (4 LOW severity accepted), 106 tests passed, model MAE=0.3945 R²=0.7980.

**Files mới:**

- `src/ml_clo/features/feature_encoder.py` — shared feature preparation (thay thế 3 bản copy)
- `src/ml_clo/utils/audit_log.py` — JSONL prediction audit trail
- `tests/unit/test_missing05.py` — 14 tests cho các gap coverage
