# Báo cáo đánh giá tính khả thi — Yêu cầu mới

**Ngày:** 2026-03-10  
**Phiên bản:** 1.1

---

## 0. Bối cảnh quan trọng

**Các yêu cầu cũ không đáp ứng được nhu cầu** → Yêu cầu mới **thay thế hoàn toàn**, không mở rộng dần.

- Có thể **loại bỏ** các tính năng cũ tương ứng
- Có thể **xây dựng lại hoàn toàn** các thành phần (data loader, pipeline, API)
- Triển khai theo hướng **thay thế**, không duy trì song song hai phiên bản lâu dài

---

## 1. Tóm tắt yêu cầu mới

### 1.1 Mô hình phân tích lớp (Class Analysis)

| Thành phần | Mô tả |
|------------|-------|
| **Đầu vào** | Môn học (Subject_ID), mã giảng viên (Lecturer_ID), danh sách điểm CLO của môn đó ở học kỳ hiện tại |
| **Đầu ra** | Nguyên nhân tại sao lớp có điểm CLO như vậy, nguyên nhân chi tiết, hướng giải quyết (cách khắc phục) |

### 1.2 Mô hình dự đoán cá nhân (Individual Prediction)

| Kịch bản | Đầu vào | Đầu ra |
|----------|---------|--------|
| **Đã học & đã đỗ** | MSSV, mã môn, mã GV, điểm CLO thực tế | Nguyên nhân, cách khắc phục (không cần dự đoán điểm) |
| **Chưa học** | MSSV, mã môn, mã GV, điểm tích lũy, thời gian, … | Điểm dự đoán, nguyên nhân, định hướng khắc phục |

### 1.3 Lưu ý quan trọng

**MSSV, mã môn, mã giảng viên không bắt buộc phải có trong dữ liệu lịch sử:**

- **Sinh viên mới** → MSSV lấy từ file nhân khẩu (`nhankhau.xlsx`)
- **Môn mới** → Mã môn lấy từ PPGD/PPDG (hoặc tách file môn riêng)
- **Giảng viên mới** → Không có file lịch sử, hệ thống vẫn phải chạy được

---

## 2. Đánh giá tính khả thi

### 2.1 So sánh với hệ thống hiện tại (sẽ bị thay thế)

| Yêu cầu mới | Hệ thống hiện tại | Khoảng cách | Hành động |
|-------------|-------------------|-------------|-----------|
| Phân tích lớp từ danh sách điểm CLO | Phân tích lớp từ DiemTong (filter subject+lecturer) | **Lớn** | **Thay thế** — xây pipeline mới, xóa hoặc không dùng `load_class_data()` theo DiemTong |
| Cá nhân: dùng điểm thực + giải thích | Chỉ dự đoán điểm + SHAP | **Trung bình** | **Sửa lại** — đổi logic predict, actual_score trở thành chế độ chính khi môn đã đỗ |
| Cá nhân: chưa học → dự đoán | Đã hỗ trợ | **Nhỏ** | Giữ ý tưởng, nhưng **đổi nguồn dữ liệu** (không bắt buộc DiemTong) |
| MSSV/Subject/Lecturer không cần có trong data | Bắt buộc phải có trong DiemTong | **Lớn** | **Xây lại** loader/merger, lấy MSSV từ nhân khẩu, môn từ PPGD/PPDG |

### 2.2 Phân tích từng hạng mục

#### 2.2.1 Phân tích lớp từ danh sách điểm CLO

| Khía cạnh | Đánh giá | Ghi chú |
|-----------|----------|---------|
| **Kỹ thuật** | Khả thi | Xây chế độ `analyze_class_from_scores()` nhận dict/array điểm CLO |
| **Dữ liệu** | Khả thi | Nếu có MSSV → lấy nhân khẩu, PPGD/PPDG; nếu chỉ có điểm → dùng thống kê phân phối |
| **XAI/SHAP** | Khả thi | Cần feature vector cho từng SV; thiếu feature → dùng giá trị mặc định |

**Rủi ro:** Nếu chỉ có danh sách điểm (không có MSSV) thì không build được feature theo SV; phải dùng phân tích thống kê đơn giản thay vì SHAP chi tiết.

#### 2.2.2 Cá nhân — Môn đã học, dùng điểm thực

| Khía cạnh | Đánh giá | Ghi chú |
|-----------|----------|---------|
| **Kỹ thuật** | Khả thi | Thêm `actual_clo_score` (optional); khi có → trả về actual thay vì predicted, vẫn dùng SHAP cho lý do |
| **Logic** | Khả thi | SHAP giải thích dự đoán của model; model vẫn cần feature → logic hiện tại tái sử dụng được |

#### 2.2.3 Hỗ trợ MSSV/Subject/Lecturer mới (chưa có trong data)

| Entity | Nguồn dữ liệu | Xử lý khi thiếu | Đánh giá |
|--------|---------------|-----------------|----------|
| **MSSV** | nhankhau.xlsx | Sinh viên mới: dùng nhân khẩu, điểm tích lũy = 0, conduct = NaN | Khả thi |
| **Subject_ID** | PPGDfull.xlsx, PPDGfull.xlsx | Môn mới: dùng TM/EM mặc định (0) hoặc file môn riêng | Khả thi |
| **Lecturer_ID** | Không có file | GV mới: mã hóa "unknown" hoặc giá trị mặc định | Khả thi |

**Thách thức kỹ thuật:**

- Model dùng `LabelEncoder` cho `student_id`, `subject_id`, `lecturer_id` → ID chưa thấy khi train sẽ lỗi.
- **Giải pháp:** Lưu encoder, khi gặp giá trị mới dùng `-1` hoặc class "unknown" (cần bổ sung khi fit encoder).

---

## 3. Kết luận

### 3.1 Tính khả thi tổng thể: **KHẢ THI**

Yêu cầu mới **thay thế** các yêu cầu cũ, có thể đạt được bằng cách:

1. **Xây lại** pipeline phân tích lớp — đầu vào là danh sách điểm CLO, không filter từ DiemTong.
2. **Xây lại** pipeline cá nhân — hỗ trợ actual_score khi môn đã đỗ; nguồn dữ liệu không bắt buộc từ DiemTong.
3. **Xây lại** logic load/merge — MSSV từ nhân khẩu, Subject từ PPGD/PPDG, Lecturer mặc định.
4. **Loại bỏ hoặc deprecate** các API/code cũ không còn phù hợp.

### 3.2 Điều kiện để triển khai thành công

| Điều kiện | Mô tả |
|-----------|-------|
| **1** | MSSV mới phải có trong `nhankhau.xlsx` (hoặc tương đương) để lấy nhân khẩu |
| **2** | Môn mới cần có trong PPGD/PPDG hoặc file môn riêng để lấy TM/EM |
| **3** | Với phân tích lớp từ “chỉ danh sách điểm, không MSSV”: chất lượng giải thích sẽ thấp hơn (chủ yếu thống kê) |

### 3.3 Ước lượng effort (thay thế, không mở rộng)

| Hạng mục | Effort | Ghi chú |
|----------|--------|---------|
| **Xây lại** nguồn dữ liệu & ID (MSSV từ nhân khẩu, Subject từ PPGD/PPDG) | 3–4 ngày | Thay logic load/merge, loại bỏ phụ thuộc DiemTong cho entity lookup |
| **Xây lại** pipeline phân tích lớp (đầu vào = danh sách điểm) | 2–3 ngày | API mới thay thế, xóa/deprecate logic cũ |
| **Xây lại** pipeline cá nhân (actual_score, nguồn dữ liệu mới) | 1–2 ngày | Thay logic load, sửa output |
| Loại bỏ code cũ, deprecation | 0.5–1 ngày | Cleanup, migration path nếu cần |
| Test & tài liệu | 1–2 ngày | Test cho logic mới, cập nhật docs |
| **Tổng ước lượng** | **8–12 ngày** | Thay thế toàn diện, không mở rộng dần |

---

## 4. Khuyến nghị

1. **Chuẩn bị:** Xác định rõ phạm vi loại bỏ — API/code nào bị xóa, nào giữ (nếu có).
2. **Thứ tự:** (1) Xây lại nguồn dữ liệu & ID → (2) Pipeline cá nhân → (3) Pipeline phân tích lớp → (4) Loại bỏ code cũ.
3. **Phạm vi:** Không duy trì song song hai phiên bản cũ/mới lâu dài; chuyển sang yêu cầu mới là chính.
