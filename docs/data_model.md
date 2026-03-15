# MÔ TẢ DỮ LIỆU

## file 27TH01 - Công Nghệ Thông Tin.xlsx và file Danh sach thong tin khao sat sinh vien k28-CNTT.xlsx

### Tổng quan dữ liệu

- **Số dòng (bản ghi):** 128 (tương ứng 128 sinh viên)
- **Số cột:** 100 trường
- **Dung lượng:** ~102 KB
- **Kiểu dữ liệu chính:**
  - object (chuỗi): **82 cột**
  - float64: **13 cột**
  - int64: **5 cột**

👉 Đây là **dữ liệu hồ sơ sinh viên + khảo sát tuyển sinh + học tập + gia đình**, rất giàu thông tin cho phân tích giáo dục, tuyển sinh và học máy.

### Nhóm thông tin định danh - hành chính

| **Cột** | **Mô tả** |
| --- | --- |
| STT | Số thứ tự |
| --- | --- |
| ID  | Mã hồ sơ sinh viên (có giá trị thiếu) |
| --- | --- |
| Họ và tên | Họ tên sinh viên |
| --- | --- |
| Ngày sinh | Ngày sinh (datetime) |
| --- | --- |
| Phái | Giới tính (Nam/Nữ) |
| --- | --- |
| CCCD/mã định danh | Số căn cước công dân |
| --- | --- |
| Ngày cấp CCCD | Ngày cấp CCCD (đang lưu dạng số - cần chuẩn hóa) |
| --- | --- |
| Nơi cấp CCCD/CMND | Nơi cấp (đang bị hiểu là số → lỗi kiểu dữ liệu) |
| --- | --- |
| Quốc tịch | Quốc tịch |
| --- | --- |
| Dân tộc | Dân tộc |
| --- | --- |

### Thông tin địa lý - xuất thân

| **Cột** | **Mô tả** |
| --- | --- |
| Tỉnh thành | Tỉnh/thành phố |
| --- | --- |
| Nơi sinh theo giấy khai sinh | Nơi sinh |
| --- | --- |
| Địa chỉ thường trú | Địa chỉ hộ khẩu |
| --- | --- |

### Thông tin học tập THPT

| **Cột** | **Mô tả** |
| --- | --- |
| Trường THPT | Trường THPT |
| --- | --- |
| Tên lớp 12 | Lớp học |
| --- | --- |
| Học lực cả năm lớp 12 | Học lực (Giỏi/Khá/…) |
| --- | --- |
| Hạnh kiểm cả năm lớp 12 | Hạnh kiểm |
| --- | --- |
| Năm tốt nghiệp THPT | Năm tốt nghiệp |
| --- | --- |

### Thông tin điểm & xét tuyển

| **Cột** | **Mô tả** |
| --- | --- |
| Điểm môn 1/2/3 | Điểm các môn (chuỗi, phân cách bằng dấu phẩy 7,5) |
| --- | --- |
| Tổng điểm | Tổng điểm xét tuyển |
| --- | --- |
| Tổng điểm.1 | Có khả năng là tổng điểm theo phương thức khác |
| --- | --- |
| Phương thức xét tuyển | Hình thức tuyển sinh |
| --- | --- |

### Thông tin gia đình

| **Cột** | **Mô tả** |
| --- | --- |
| Họ và tên Bố, Nghề nghiệp của bố | Thông tin bố |
| --- | --- |
| Họ và tên mẹ, Nghề nghiệp của mẹ | Thông tin mẹ |
| --- | --- |
| Mức thu nhập của gia đình mỗi tháng | Thu nhập (danh mục) |
| --- | --- |
| Số điện thoại của bố/mẹ | Liên hệ |
| --- | --- |
| Anh/chị em ruột (1,2) | Thông tin anh chị em |
| --- | --- |

### Thông tin chỗ ở & hỗ trợ

| **Cột** | **Mô tả** |
| --- | --- |
| Thông tin chỗ ở hiện tại để đi học | Nhà riêng / KTX / Nhà trọ |
| --- | --- |
| Cần hỗ trợ tìm chỗ ở | 0/1 (dạng nhị phân) |
| --- | --- |

### Khảo sát kênh tuyển sinh (rất nhiều cột nhị phân)

Ví dụ:

- Bạn biết đến trường qua … Facebook
- … qua Zalo
- … qua Cán bộ tư vấn
- … qua tờ rơi
- … qua SMS

👉 Giá trị thường là:

- "Đúng" / "Sai"
- hoặc 0 / 1

⚠️ Cần **chuẩn hóa về Boolean** (True/False) trước phân tích.

### Khảo sát đánh giá & quyết định chọn trường

| **Cột** | **Mô tả** |
| --- | --- |
| Bạn đánh giá về cán bộ tư vấn | Ý kiến sinh viên |
| --- | --- |
| Bạn đánh giá như thế nào về phương thức xét tuyển … | Đánh giá chi tiết |
| --- | --- |
| Bạn quyết định chọn Trường ĐH Bình Dương vì … | Lý do chọn trường |
| --- | --- |
| Ý kiến cá nhân | Văn bản tự do |
| --- | --- |

### Tài liệu minh chứng (link)

| **Cột** | **Mô tả** |
| --- | --- |
| Hình ảnh đóng HỌC PHÍ và LỆ PHÍ NHẬP HỌC | Link Google Drive |
| --- | --- |
| Hình ảnh đóng BẢO HIỂM Y TẾ | Link |
| --- | --- |

### Năng khiếu - hoạt động

| **Cột** | **Mô tả** |
| --- | --- |
| Bạn có năng khiếu nghệ thuật về lĩnh vực nào | Văn bản |
| --- | --- |
| Bạn có năng khiếu về các môn thể thao nào | Văn bản |
| --- | --- |

## II. file diemrenluyen

### Tổng quan dữ liệu

| **Thuộc tính** | **Giá trị** |
| --- | --- |
| Số dòng | **2.198** |
| --- | --- |
| Số cột | **7** |
| --- | --- |
| Dung lượng | ~120 KB |
| --- | --- |
| Kiểu dữ liệu | int64: 3, object: 4 |
| --- | --- |
|     |     |
| --- | --- |

### Mô tả chi tiết từng cột (Data Dictionary)

🔹 1. Student_ID (int64)

- **Ý nghĩa:** Mã số sinh viên
- **Vai trò:** 🔑 _Primary Key_
- **Tính chất:** Duy nhất theo từng sinh viên - học kỳ
- **Dùng để:** JOIN với bảng hồ sơ sinh viên

**🔹 2. name (object)**

- **Ý nghĩa:** Họ và tên sinh viên
- **Chỉ dùng:** Đối soát, kiểm tra dữ liệu

🔹 3. Class_ID (object)

- **Ý nghĩa:** Mã lớp (VD: 24TH01)
- **Thuộc tính:** Biến phân nhóm
- **Ứng dụng:**
  - Phân tích theo lớp
  - Phân tích chất lượng đào tạo

**🔹 4. semester_year (object)**

- **Ý nghĩa:** Năm học (VD: 2021-2022)
- **Bản chất:** Biến thời gian (chuỗi)
- 👉 Có thể chuyển thành:
  - start_year, end_year
  - Hoặc index theo timeline

🔹 5. semester (int64)

- **Ý nghĩa:** Học kỳ (1, 2, 3)
- **Vai trò:** Biến thứ tự thời gian

🔹 6. conduct_score (int64)

- **Ý nghĩa:** **Điểm rèn luyện**
- **Khoảng giá trị:** 0 - 100 (thực tế ~60-90)
- 🎯 **Biến số định lượng rất mạnh**
- **Rất phù hợp cho ML & thống kê**

🔹 7. student_conduct_classification (object)

- **Ý nghĩa:** Xếp loại rèn luyện
  - Ví dụ: Xuất sắc, Tốt, Khá, TB Khá
- **Là biến phân loại**

## III. PHÂN TÍCH FILE: DiemTong.xlsx

### Tổng quan dữ liệu

| **Thuộc tính** | **Giá trị** |
| --- | --- |
| Số dòng | **25.257** |
| --- | --- |
| Số cột | **36** |
| --- | --- |
| Dung lượng | ~6.9 MB |
| --- | --- |
| Bản chất | **Bảng FACT - kết quả học tập theo học phần** |
| --- | --- |

👉 Mỗi dòng ≈ **1 sinh viên - 1 môn - 1 học kỳ**

### Nhóm định danh & liên kết (KEYS)

| **Cột** | **Ý nghĩa** | **Ghi chú** |
| --- | --- | --- |
| Student_ID | Mã số sinh viên | 🔑 khóa chính để JOIN |
| --- | --- | --- |
| Class_ID | Mã lớp | JOIN với hồ sơ |
| --- | --- | --- |
| Major_ID | Mã ngành (TH) | CNTT |
| --- | --- | --- |
| Major_Name | Tên ngành |     |
| --- | --- | --- |
| Faculty_ID | Mã khoa |     |
| --- | --- | --- |
| Faculty_Name | Tên khoa |     |
| --- | --- | --- |

👉 **Student_ID là trục xương sống để nối với file 1-2-3**

### Thông tin sinh viên (mô tả - không dùng ML)

| **Cột** |
| --- |
| LastName, FirstName |
| --- |
| Birthdate |
| --- |

### Thông tin giảng viên - môn học

| **Cột** | **Ý nghĩa** |
| --- | --- |
| Lecturer_ID, Lecturer_Name | Giảng viên |
| --- | --- |
| Subject_ID | Mã môn |
| --- | --- |
| Subject_Name | Tên môn |
| --- | --- |
| Credit_Hours | Số tín chỉ |
| --- | --- |

### Thành phần điểm (RẤT QUAN TRỌNG)

| **Cột** | **Ý nghĩa** |
| --- | --- |
| Percent_K1 | Điểm quá trình |
| --- | --- |
| Percent_B1 | Bài tập / giữa kỳ |
| --- | --- |
| Percent_Final_Exam | Thi cuối kỳ |
| --- | --- |
| exam_score | Điểm thi (chuỗi) |
| --- | --- |
| summary_score | **Điểm tổng kết (chuỗi)** |
| --- | --- |
| letter_system | Điểm chữ (A, B, C…) |
| --- | --- |

### Kết quả học phần (TARGET CỰC MẠNH)

| **Cột** | **Ý nghĩa** |
| --- | --- |
| Passed_the_module | Đậu (1) / Rớt (0) |
| --- | --- |
| GC_registration_results | Kết quả đăng ký |
| --- | --- |
| nature_of_the_course | Lần đầu / Học lại |
| --- | --- |

### Thông tin học phí (KHÔNG dùng ML)

| **Cột** |
| --- |
| unit_price |
| --- |
| Tuition_fee(not_reduced) |
| --- |
| Exemptions |
| --- |
| Receivable |
| --- |
| TCHP |
| --- |

### Thời gian - tiến trình học tập

| **Cột** |
| --- |
| registration_date |
| --- |
| year |
| --- |
| semester _(gián tiếp qua year)_ |
| --- |

## IV. PHÂN TÍCH FILE: Dữ liệu điểm danh Khoa FIRA.xlsx

### Tổng quan dữ liệu

| **Thuộc tính** | **Giá trị** |
| --- | --- |
| Số dòng | **56.207** |
| --- | --- |
| Số cột | **10** |
| --- | --- |
| Dung lượng | ~4.3 MB |
| --- | --- |
| Bản chất | **FACT TABLE - điểm danh theo buổi học** |
| --- | --- |

👉 Mỗi dòng = **1 sinh viên - 1 môn - 1 buổi học**

### Nhóm định danh & liên kết

| **Cột** | **Ý nghĩa** | **Vai trò** |
| --- | --- | --- |
| MSSV | Mã số sinh viên | 🔑 JOIN với các file khác |
| --- | --- | --- |
| Họ Tên | Tên sinh viên | Đối soát |
| --- | --- | --- |
| Mã môn học | Mã học phần | JOIN với bảng điểm |
| --- | --- | --- |
| Tên môn học | Tên học phần | Mô tả |
| --- | --- | --- |
| Mã giảng viên | Mã GV | Phân tích GV |
| --- | --- | --- |

👉 **MSSV ↔ Student_ID là trục liên kết quan trọng**

### Thời gian & tiến trình học

| **Cột** | **Ý nghĩa** |
| --- | --- |
| Ngày | Ngày học (chuỗi, cần ép date) |
| --- | --- |
| Buổi | Thứ tự buổi học |
| --- | --- |
| Niên khoá | Năm học |
| --- | --- |
| Học kì | Học kỳ |
| --- | --- |

### Trạng thái điểm danh (CỰC KỲ QUAN TRỌNG)

| **Cột** | **Ý nghĩa** |
| --- | --- |
| Điểm danh | Trạng thái đi học |
| --- | --- |

Giá trị quan sát được:

- Sớm
- (khả năng có) Trễ, Vắng, Có mặt

📌 Đây là **biến hành vi học tập trực tiếp**, rất mạnh.

## V. PHÂN TÍCH FILE **Khảo sát các yếu tố cá nhân phục vụ phân tích và dự báo kết quả học tập sinh viên (Câu trả lời).xlsx**

### Tổng quan dữ liệu

| **Thuộc tính** | **Giá trị** |
| --- | --- |
| Số dòng | **280** |
| --- | --- |
| Số cột | **42** |
| --- | --- |
| Dung lượng | ~92 KB |
| --- | --- |
| Kiểu dữ liệu | 1 datetime, 41 object |
| --- | --- |
| Bản chất | **Survey / Psychosocial Data** |
| --- | --- |

### THÔNG TIN ĐỊNH DANH & NGỮ CẢNH HỌC TẬP

🎯 Mục đích bảng

Xác định **sinh viên nào**, **trong học kỳ - năm học nào** tham gia khảo sát.  
Bảng này dùng để **JOIN dữ liệu** và **lọc bản ghi hợp lệ**, không dùng trực tiếp cho ML.

| **Cột** | **Ý nghĩa** | **Loại biến** | **Ghi chú** |
| --- | --- | --- | --- |
| Dấu thời gian | Thời điểm gửi khảo sát | Thời gian | Dùng chọn bản ghi mới nhất |
| --- | --- | --- | --- |
| Địa chỉ email | Email sinh viên | Định danh | ❌ Loại khỏi phân tích |
| --- | --- | --- | --- |
| 1.1. Mã sinh viên | Mã số sinh viên | 🔑 Key | JOIN với các bảng khác |
| --- | --- | --- | --- |
| 1.2. Năm học | Năm học khảo sát | Danh mục | VD: 2024-2025 |
| --- | --- | --- | --- |
| 1.3. Học kỳ | Học kỳ khảo sát | Thứ bậc | HK1, HK2, HK3 |
| --- | --- | --- | --- |
| 1.3 . Sinh viên năm | Năm học của SV | Thứ bậc | Năm 1, 2, 3… |
| --- | --- | --- | --- |

📌 **Lưu ý khoa học**

- Một sinh viên có thể trả lời **nhiều lần** → cần lọc
- Chỉ giữ **1 bản ghi / SV / học kỳ**

### HOÀN CẢNH GIA ĐÌNH & KINH TẾ

🎯 Mục đích bảng

Đo lường **điều kiện kinh tế - xã hội (Socioeconomic Status - SES)** của sinh viên, là nhóm yếu tố nền ảnh hưởng gián tiếp đến kết quả học tập.

| **Cột** | **Ý nghĩa** | **Loại biến** |
| --- | --- | --- |
| 2.1. Nơi cư trú của gia đình | Thành thị / Nông thôn | Nhị phân |
| --- | --- | --- |
| 2.2. Nơi ở hiện tại trong thời gian học | Ở với gia đình / Nhà trọ / KTX | Danh mục |
| --- | --- | --- |
| 2.3. Mức độ hỗ trợ tài chính từ gia đình | Mức hỗ trợ tiền học | Thứ bậc |
| --- | --- | --- |
| 2.4. Gặp khó khăn tài chính trong học kỳ | Mức độ khó khăn | Thứ bậc |
| --- | --- | --- |
| 2.5. Mức độ hỗ trợ tinh thần từ gia đình | Động viên tinh thần | Thứ bậc |
| --- | --- | --- |
| 2.6. Mức độ kỳ vọng của gia đình | Kỳ vọng kết quả học | Thứ bậc |
| --- | --- | --- |
| 2.7. Thu nhập gia đình hàng tháng | Thu nhập ước tính | Thứ bậc |
| --- | --- | --- |
| 2.8. Là lao động chính trong gia đình | Có/Không | Nhị phân |
| --- | --- | --- |
| 2.9. Học phí trong học kỳ này | Mức học phí cảm nhận | Thứ bậc |
| --- | --- | --- |
| 2.10. Người chi trả chính học phí | Gia đình / Bản thân / Khác | Danh mục |
| --- | --- | --- |
| 2.11. Áp lực đóng học phí | Mức độ áp lực | Thứ bậc |
| --- | --- | --- |
| 2.12. Học phí ảnh hưởng đến quyết định học tập | Có ảnh hưởng hay không | Nhị phân/Thứ bậc |
| --- | --- | --- |

📌 **Vai trò phân tích**

- Biến nền (background variables)
- Dùng tốt cho **hồi quy**, **SEM**, **ML**

### VIỆC LÀM THÊM & GÁNH NẶNG THỜI GIAN

🎯 Mục đích bảng

Đo **xung đột giữa học tập và lao động**, yếu tố thường ảnh hưởng trực tiếp đến kết quả học tập và nguy cơ rớt môn.

| **Cột** | **Ý nghĩa** | **Loại biến** |
| --- | --- | --- |
| 3.1. Có đi làm thêm không | Trạng thái làm thêm | Nhị phân |
| --- | --- | --- |
| 3.2. Số giờ làm thêm mỗi tuần | Cường độ làm thêm | Thứ bậc |
| --- | --- | --- |
| 3.3. Mục đích làm thêm | Kiếm tiền / Kinh nghiệm | Danh mục |
| --- | --- | --- |
| 3.4. Làm thêm ảnh hưởng đến việc học | Mức độ ảnh hưởng | Thứ bậc |
| --- | --- | --- |
| 3.5. Loại công việc làm thêm | Công việc cụ thể | Danh mục |
| --- | --- | --- |
| 3.6. Thu nhập làm thêm mỗi tháng | Thu nhập ước tính | Thứ bậc |
| --- | --- | --- |

📌 **Vai trò ML**

- Predictor mạnh cho: rớt môn, GPA thấp
- Có thể tạo biến tổng hợp: work_study_conflict  

### 5\. ÁP LỰC, TÂM LÝ & SỨC KHỎE

🎯 Mục đích bảng

Phản ánh **trạng thái tâm lý - sức khỏe**, nhóm yếu tố trung gian rất quan trọng trong giáo dục.

| **Cột** | **Ý nghĩa** | **Loại biến** |
| --- | --- | --- |
| 4.1. Mức độ áp lực học tập | Áp lực học | Thứ bậc |
| --- | --- | --- |
| 4.2. Những vấn đề gặp phải | Stress, mệt mỏi, chán học… | Văn bản / đa lựa chọn |
| --- | --- | --- |
| 4.3. Từng nghĩ đến bỏ học/bảo lưu | Ý định bỏ học | Nhị phân |
| --- | --- | --- |
| 4.4. Số giờ ngủ mỗi đêm | Thời gian ngủ | Thứ bậc |
| --- | --- | --- |

### 6\. HÀNH VI & CHIẾN LƯỢC HỌC TẬP

🎯 Mục đích bảng

Đo **self-regulated learning (SRL)** - hành vi học tập chủ động của sinh viên.

| **Cột** | **Ý nghĩa** | **Loại biến** |
| --- | --- | --- |
| 5.1. Không gian học tập yên tĩnh | Điều kiện học | Thứ bậc |
| --- | --- | --- |
| 5.2. Thời gian tự học mỗi ngày | Cường độ tự học | Thứ bậc |
| --- | --- | --- |
| 5.3. Cách xử lý khi gặp khó khăn | Tự học / hỏi GV / bạn bè | Danh mục |
| --- | --- | --- |
| 5.4. Phương pháp học thường dùng | Phương pháp | Đa lựa chọn |
| --- | --- | --- |
| 5.5. Kỹ năng quản lý thời gian | Mức kỹ năng | Thứ bậc |
| --- | --- | --- |
| 5.6. Lập kế hoạch học tập | Mức độ lập kế hoạch | Thứ bậc |
| --- | --- | --- |
| 5.7. Trao đổi với bạn bè | Tần suất trao đổi | Thứ bậc |
| --- | --- | --- |
| 5.8. Tham gia nhóm học tập | Mức độ tham gia | Thứ bậc |
| --- | --- | --- |
| 5.9. Áp lực học tập từ bạn bè | Peer pressure | Thứ bậc |
| --- | --- | --- |

📌 **Nhóm biến cực mạnh cho ML**

### 7\. ĐIỀU KIỆN HỌC TẬP SỐ (DIGITAL LEARNING)

🎯 Mục đích bảng

Đo mức độ **sẵn sàng học tập số**, rất quan trọng trong bối cảnh giáo dục hiện đại.

| **Cột** | **Ý nghĩa** | **Loại biến** |
| --- | --- | --- |
| 6.1. Thiết bị học tập | Laptop/máy tính | Thứ bậc |
| --- | --- | --- |
| 6.2. Chất lượng internet | Tốc độ/kết nối | Thứ bậc |
| --- | --- | --- |
| 6.3. Sử dụng LMS | Tần suất dùng LMS | Thứ bậc |
| --- | --- | --- |
| 6.4. Công cụ học tập online | Zoom, Teams… | Thứ bậc |
| --- | --- | --- |
| 6.5. Truy cập tài liệu online | Khả năng truy cập | Thứ bậc |
| --- | --- | --- |

## VI. PHÂN TÍCH FILE: nhankhau.xlsx

### Tổng quan dữ liệu

| **Thuộc tính** | **Giá trị** |
| --- | --- |
| Số dòng | **428 sinh viên** |
| --- | --- |
| Số cột | **139 cột** |
| --- | --- |
| Kiểu dữ liệu | số, chuỗi, ngày |
| --- | --- |
| Bản chất | **Hồ sơ nhân khẩu - hành chính - đầu vào** |
| --- | --- |
| Vai trò | **BẢNG DIMENSION TRUNG TÂM** |
| --- | --- |

### ĐỊNH DANH SINH VIÊN (CORE IDENTITY)

🎯 Mục đích

Xác định **duy nhất mỗi sinh viên**, phục vụ JOIN dữ liệu.

| **Cột** | **Ý nghĩa** | **Loại biến** |
| --- | --- | --- |
| Student_ID | Mã số sinh viên | 🔑 Key |
| --- | --- | --- |
| LastName | Họ  | Văn bản |
| --- | --- | --- |
| FirstName | Tên | Văn bản |
| --- | --- | --- |
| FullName | Họ và tên đầy đủ | Văn bản |
| --- | --- | --- |
| RowID | ID nội bộ hệ thống | Kỹ thuật |
| --- | --- | --- |

📌 **Lưu ý**

- Student_ID là **khóa chính**
- RowID không dùng cho phân tích

### ĐẶC ĐIỂM NHÂN KHẨU HỌC (DEMOGRAPHICS)

🎯 Mục đích

Mô tả **đặc điểm cá nhân cơ bản** của sinh viên.

| **Cột** | **Ý nghĩa** | **Loại biến** |
| --- | --- | --- |
| Gender | Giới tính (0/1) | Nhị phân |
| --- | --- | --- |
| Birthdate | Ngày sinh | Thời gian |
| --- | --- | --- |
| place_of_birth | Nơi sinh | Danh mục |
| --- | --- | --- |
| DOITUONGTS | Đối tượng ưu tiên tuyển sinh | Danh mục |
| --- | --- | --- |

📌 **Gợi ý**

- Có thể tạo biến Age
- Giới tính dùng được cho phân tích thống kê

### THÔNG TIN HỌC THUẬT & TỔ CHỨC

🎯 Mục đích

Xác định **bối cảnh đào tạo chính thức** của sinh viên.

| **Cột** | **Ý nghĩa** | **Loại biến** |
| --- | --- | --- |
| Faculty_ID | Mã khoa | Danh mục |
| --- | --- | --- |
| Faculty_Name | Tên khoa | Danh mục |
| --- | --- | --- |
| Major_ID | Mã ngành | Danh mục |
| --- | --- | --- |
| Major_Name | Tên ngành | Danh mục |
| --- | --- | --- |
| Class_ID _(nếu có)_ | Mã lớp | Danh mục |
| --- | --- | --- |
| HeDTVB | Hệ đào tạo | Danh mục |
| --- | --- | --- |

📌 **Vai trò**

- Dùng phân tích theo khoa/ngành/lớp
- Không dùng trực tiếp cho ML (chủ yếu group analysis)

### THÔNG TIN NHÂN THÂN & HỘ KHẨU

🎯 Mục đích

Phản ánh **xuất thân địa lý - xã hội** của sinh viên.

| **Cột** | **Ý nghĩa** | **Loại biến** |
| --- | --- | --- |
| place_of_birth | Nơi sinh | Danh mục |
| --- | --- | --- |
| NhomKVTS | Nhóm khu vực tuyển sinh | Danh mục |
| --- | --- | --- |
| DOITUONGTS | Đối tượng ưu tiên | Danh mục |
| --- | --- | --- |

📌 Có thể gom nhóm:

- Thành thị / Nông thôn
- KV1 / KV2 / KV3

### THÔNG TIN GIẤY TỜ & HÀNH CHÍNH

🎯 Mục đích

Quản lý **hồ sơ pháp lý**, **không dùng cho phân tích học tập**.

| **Cột (ví dụ)** | **Ý nghĩa** |
| --- | --- |
| SoQDCC | Số quyết định cấp CCCD |
| --- | --- |
| NgayKyCC | Ngày ký |
| --- | --- |
| TenCC | Tên cơ quan |
| --- | --- |
| GcQDKL, NHHKCC | Trường nội bộ |
| --- | --- |

### THÔNG TIN KỸ THUẬT HỆ THỐNG

🎯 Mục đích

Phục vụ vận hành hệ thống.

| **Cột** | **Ghi chú** |
| --- | --- |
| AvatarSV | Ảnh đại diện |
| --- | --- |
| Các cột float NaN | Metadata |
| --- | --- |

## VII. PHÂN TÍCH BẢNG: PHƯƠNG PHÁP ĐÁNH GIÁ (PPDG)

### Mục đích của bảng

Bảng này dùng để:

- Mô tả **các phương pháp đánh giá kết quả học tập** trong học phần
- Chuẩn hóa các **EM (Evaluation Method)** theo chuẩn OBE
- Làm cơ sở:
  - xây dựng Rubric
  - mapping CLO
  - báo cáo kiểm định (AUN-QA, OBE)

### Ý nghĩa các cột trong bảng PPDG

🔹 MaPPDG

- **Ý nghĩa**: Mã phương pháp đánh giá
- **Giá trị**: EM1, EM2, …, EM14
- **Vai trò**: 🔑 Khóa chính của bảng
- **Liên hệ DOC**: trùng 100% với các EM trong _Mẫu RUBRIC_FIRA_

📌 Ví dụ:

- EM1 = Đánh giá chuyên cần
- EM12 = Đánh giá bài tập lớn / đồ án cá nhân

🔹 TenPPDG

- **Ý nghĩa**: Tên đầy đủ của phương pháp đánh giá
- **Ví dụ**:
  - Đánh giá chuyên cần
  - Đánh giá bài tập cá nhân
  - Đánh giá thực hành phòng thí nghiệm
- **Vai trò**: Diễn giải MaPPDG cho người đọc

🔹 LoaiDanhGia

- **Ý nghĩa**: Loại hình đánh giá
- **Giá trị thường gặp**:
  - Đánh giá quá trình (Formative)
  - Đánh giá tổng kết (Summative)
- **Vai trò học thuật**:
  - Phân biệt đánh giá trong quá trình học hay cuối học phần

📌 Theo chuẩn OBE:

- EM1-EM3 → thường là **quá trình**
- EM7, EM12, EM14 → thường là **tổng kết**

🔹 MoTaPPDG

- **Ý nghĩa**: Mô tả chi tiết cách đánh giá
- **Nguồn**: trích nội dung từ _Mẫu RUBRIC_FIRA.docx_
- **Ví dụ nội dung**:  
    <br/>Sinh viên được đánh giá thông qua việc tham dự lớp học, mức độ tham gia và đóng góp…

📌 Đây là **cột cực kỳ quan trọng** để:

- giải trình kiểm định
- chứng minh minh bạch đánh giá

🔹 CongCuDanhGia

- **Ý nghĩa**: Công cụ dùng để đánh giá
- **Giá trị thường gặp**:
  - Rubric đánh giá
  - Bài kiểm tra viết
  - Vấn đáp
  - Báo cáo / sản phẩm
- **Vai trò**:
  - Liên kết PPDG ↔ Rubric

📌 Ví dụ:

- EM1 → Rubric 1
- EM12 → Rubric 10

🔹 RubricApDung _(nếu có trong bảng)_

- **Ý nghĩa**: Rubric cụ thể dùng cho EM này
- **Ví dụ**:
  - Rubric 1
  - Rubric 10
- **Vai trò**:
  - Ràng buộc 1 EM ↔ 1 hoặc nhiều Rubric

🔹 GhiChu _(nếu có)_

- **Ý nghĩa**: Ghi chú đặc biệt
- **Ví dụ**:
  - EM này chỉ áp dụng cho học phần thực hành
  - EM này dùng để đánh giá PLO

### Ý nghĩa học thuật của bảng này

Bảng PPDG trả lời trực tiếp 3 câu hỏi **cốt lõi trong OBE**:

- **Sinh viên được đánh giá bằng cách nào?  
    **→ TenPPDG, MoTaPPDG
- **Đánh giá vào thời điểm nào?  
    **→ LoaiDanhGia
- **Đánh giá dựa trên tiêu chí nào?  
    **→ CongCuDanhGia, RubricApDung

### 4\. Liên hệ trực tiếp với EM trong DOC

| **EM** | **Ý nghĩa ngắn gọn** |
| --- | --- |
| EM1 | Đánh giá chuyên cần |
| --- | --- |
| EM2 | Đánh giá bài tập cá nhân |
| --- | --- |
| EM3 | Đánh giá thuyết trình |
| --- | --- |
| EM7 | Kiểm tra trắc nghiệm |
| --- | --- |
| EM9 | Đánh giá thực tập |
| --- | --- |
| EM11 | Thực hành phòng thí nghiệm |
| --- | --- |
| EM12 | Bài tập lớn / đồ án |
| --- | --- |
| EM14 | Khóa luận tốt nghiệp |
| --- | --- |

### 5\. BẢNG TỔNG HỢP CÁC PHƯƠNG PHÁP ĐÁNH GIÁ (EM)

| **Mã EM** | **Tên phương pháp đánh giá** | **Mô tả ngắn gọn** | **Loại đánh giá** |
| --- | --- | --- | --- |
| **EM1** | Đánh giá chuyên cần | Đánh giá mức độ tham gia học tập, đi học đầy đủ, đúng giờ, thái độ học tập | Quá trình |
| --- | --- | --- | --- |
| **EM2** | Đánh giá bài tập cá nhân | Đánh giá bài tập giao trong học kỳ, khả năng vận dụng kiến thức | Quá trình |
| --- | --- | --- | --- |
| **EM3** | Đánh giá thuyết trình | Đánh giá kỹ năng trình bày, giao tiếp và trả lời câu hỏi | Quá trình |
| --- | --- | --- | --- |
| **EM4** | Kiểm tra viết (tự luận) | Đánh giá kiến thức và tư duy phân tích qua bài tự luận | Tổng kết |
| --- | --- | --- | --- |
| **EM5** | Kiểm tra viết (kết hợp) | Kết hợp tự luận và các dạng câu hỏi khác | Tổng kết |
| --- | --- | --- | --- |
| **EM6** | Kiểm tra viết cuối kỳ | Đánh giá tổng hợp kiến thức toàn học phần | Tổng kết |
| --- | --- | --- | --- |
| **EM7** | Kiểm tra trắc nghiệm | Đánh giá kiến thức diện rộng qua câu hỏi trắc nghiệm | Tổng kết |
| --- | --- | --- | --- |
| **EM8** | Báo cáo / tiểu luận | Đánh giá kỹ năng viết học thuật, tổng hợp và trình bày | Quá trình / Tổng kết |
| --- | --- | --- | --- |
| **EM9** | Đánh giá thực tập | Đánh giá năng lực nghề nghiệp trong quá trình thực tập | Thực hành |
| --- | --- | --- | --- |
| **EM10** | Báo cáo thực tập | Đánh giá báo cáo tổng kết thực tập | Thực hành |
| --- | --- | --- | --- |
| **EM11** | Thực hành phòng thí nghiệm | Đánh giá kỹ năng thao tác, quy trình và kết quả thực hành | Quá trình |
| --- | --- | --- | --- |
| **EM12** | Bài tập lớn / đồ án cá nhân | Đánh giá năng lực vận dụng, phân tích, sáng tạo qua sản phẩm lớn | Tổng hợp |
| --- | --- | --- | --- |
| **EM13** | Vấn đáp / bảo vệ | Đánh giá tư duy phản biện và lập luận trực tiếp | Tổng hợp |
| --- | --- | --- | --- |
| **EM14** | Khóa luận tốt nghiệp | Đánh giá năng lực đầu ra tổng thể của sinh viên | Tổng hợp |
| --- | --- | --- | --- |

## VIII. Phân tích file PPGD

BẢNG TỔNG HỢP PHƯƠNG PHÁP GIẢNG DẠY (TM)

| **Mã TM** | **Tên phương pháp giảng dạy** | **Mô tả ngắn gọn** | **Mục tiêu sư phạm** |
| --- | --- | --- | --- |
| **TM1** | Thuyết giảng | Giảng viên trình bày nội dung bài học theo cấu trúc logic | Truyền đạt kiến thức nền tảng |
| --- | --- | --- | --- |
| **TM2** | Hỏi - đáp | Giảng viên đặt câu hỏi, sinh viên trả lời và thảo luận | Kích thích tư duy, kiểm tra mức độ hiểu |
| --- | --- | --- | --- |
| **TM3** | Thảo luận nhóm | Sinh viên trao đổi, thảo luận theo nhóm nhỏ | Phát triển kỹ năng giao tiếp, hợp tác |
| --- | --- | --- | --- |
| **TM4** | Làm việc nhóm | Sinh viên phối hợp thực hiện nhiệm vụ học tập | Rèn kỹ năng làm việc nhóm |
| --- | --- | --- | --- |
| **TM5** | Tự học có hướng dẫn | Sinh viên tự nghiên cứu theo định hướng của giảng viên | Phát triển năng lực tự học |
| --- | --- | --- | --- |
| **TM6** | Học qua bài tập | Học thông qua việc giải quyết bài tập cụ thể | Củng cố và vận dụng kiến thức |
| --- | --- | --- | --- |
| **TM7** | Học qua ví dụ minh họa | Giảng dạy thông qua ví dụ thực tế | Tăng khả năng liên hệ thực tiễn |
| --- | --- | --- | --- |
| **TM8** | Thuyết trình của sinh viên | Sinh viên chuẩn bị và trình bày nội dung trước lớp | Rèn kỹ năng trình bày, tự tin |
| --- | --- | --- | --- |
| **TM9** | Học qua dự án | Sinh viên thực hiện dự án học tập theo chủ đề | Phát triển tư duy tổng hợp |
| --- | --- | --- | --- |
| **TM10** | Học qua thực hành | Sinh viên thực hành trực tiếp (phòng máy, phòng thí nghiệm) | Rèn kỹ năng thực hành |
| --- | --- | --- | --- |
| **TM11** | Học qua nghiên cứu | Sinh viên tìm hiểu, phân tích vấn đề mang tính nghiên cứu | Phát triển tư duy nghiên cứu |
| --- | --- | --- | --- |
| **TM12** | Học qua mô phỏng | Sử dụng mô phỏng, phần mềm, tình huống giả lập | Hiểu sâu quy trình, hệ thống |
| --- | --- | --- | --- |
| **TM13** | Đóng vai (Role Play) | Sinh viên nhập vai xử lý tình huống giả định | Phát triển kỹ năng mềm, giao tiếp |
| --- | --- | --- | --- |
| **TM14** | Giải quyết vấn đề | Sinh viên học thông qua phân tích và giải quyết vấn đề | Rèn tư duy phản biện |
| --- | --- | --- | --- |
| **TM15** | Tập kích não (Brainstorming) | Huy động ý tưởng, thảo luận mở | Kích thích sáng tạo |
| --- | --- | --- | --- |
| **TM16** | Học theo tình huống (Case Study) | Phân tích tình huống thực tế | Gắn lý thuyết với thực tiễn |
| --- | --- | --- | --- |

## IV. PHÂN TÍCH FILE: tuhoc.xlsx

### Tổng quan dữ liệu

| **Thuộc tính** | **Giá trị** |
| --- | --- |
| Số dòng | **1.982 bản ghi** |
| --- | --- |
| Số cột | **9 cột** |
| --- | --- |
| Bản chất | **FACT TABLE - thời gian tự học tích lũy** |
| --- | --- |
| Vai trò | Đo **nỗ lực học tập (learning effort)** |
| --- | --- |

### ĐỊNH DANH SINH VIÊN & NGỮ CẢNH HỌC TẬP

🎯 Mục đích

Xác định **ai**, **học kỳ nào**, **thuộc lớp nào** để liên kết với các bảng khác.

| **Cột** | **Ý nghĩa** | **Loại biến** | **Ghi chú** |
| --- | --- | --- | --- |
| Student_ID | Mã số sinh viên | 🔑 Key | JOIN toàn bộ hệ dữ liệu |
| --- | --- | --- | --- |
| name | Họ tên sinh viên | Văn bản | Chỉ đối soát |
| --- | --- | --- | --- |
| Class_Id | Mã lớp | Danh mục | VD: 24TH01 |
| --- | --- | --- | --- |
| year | Năm học | Danh mục | 2022-2023 |
| --- | --- | --- | --- |
| semester | Học kỳ | Thứ bậc | 1, 2, 3 |
| --- | --- | --- | --- |

### THỜI GIAN TỰ HỌC TÍCH LŨY (CORE TABLE)

🎯 Mục đích

Đo **mức độ đầu tư thời gian học tập** của sinh viên trong học kỳ.

| **Cột** | **Ý nghĩa** | **Loại biến** | **Ghi chú** |
| --- | --- | --- | --- |
| time | Thời gian tự học dạng HH:MM | Văn bản | Dữ liệu gốc |
| --- | --- | --- | --- |
| accumulated_study_hours | Tổng giờ tự học | Số thực | Feature chính |
| --- | --- | --- | --- |
| accumulated_study_minutes | Tổng phút tự học | Số thực | Chính xác hơn giờ |
| --- | --- | --- | --- |

### BIẾN MỤC TIÊU / CHUẨN SO SÁNH

🎯 Mục đích

Làm **ngưỡng hoặc mốc chuẩn** để so sánh nỗ lực học tập.

| **Cột** | **Ý nghĩa** | **Loại biến** | **Vai trò** |
| --- | --- | --- | --- |
| target | Mốc chuẩn thời gian tự học | Số nguyên | Chuẩn tham chiếu |
| --- | --- | --- | --- |

### ĐẶT FILE tuhoc.xlsx TRONG TOÀN BỘ HỆ DỮ LIỆU

| **File** | **Nội dung** |
| --- | --- |
| nhankhau.xlsx | Hồ sơ nền |
| --- | --- |
| DiemTong.xlsx | Kết quả học tập |
| --- | --- |
| diemrenluyen.xlsx | Hành vi - thái độ |
| --- | --- |
| Điểm danh | Chuyên cần |
| --- | --- |
| **tuhoc.xlsx** | **Nỗ lực tự học** |
| --- | --- |

---

## IX. NGUỒN DỮ LIỆU MỚI — Yêu cầu mới (2026-03)

**Bối cảnh:** Predict và phân tích lớp không bắt buộc (Student_ID, Subject_ID, Lecturer_ID) phải có trong DiemTong. Hệ thống dùng các nguồn thay thế sau.

### Nguồn thay thế cho entity

| **Entity** | **Nguồn chính** | **Ghi chú** |
|------------|-----------------|-------------|
| **MSSV / Student_ID** | `nhankhau.xlsx` | Cột MSSV hoặc Student_ID; map MSSV → Student_ID khi load. Sinh viên mới chỉ cần có trong nhân khẩu. |
| **Subject_ID** | PPGD (`PPGDfull.xlsx`) / PPDG (`PPDGfull.xlsx`) | Môn mới lấy TM/EM từ file; thiếu cột → 0 |
| **Lecturer_ID** | Không có file riêng | GV mới → placeholder `__UNKNOWN__`; khi encode dùng -1 cho unseen |

### Tạo record ảo (`create_student_record_from_ids`)

Khi predict cho sinh viên/môn/GV chưa có trong DiemTong, hệ thống tạo 1 record "ảo" từ:

- `(student_id, subject_id, lecturer_id)` + `demographics_df` (left join) + `teaching_methods_df` (Subject_ID) + `assessment_methods_df` (Subject_ID)

Record có `exam_score = NaN`; `create_training_dataset` hỗ trợ `drop_missing_target=False`.

### File điểm CLO cho phân tích lớp (`--scores-file`)

**Định dạng CSV:**
```
student_id,clo_score
19050006,4.2
19050007,3.8
```

**Định dạng JSON:** `{"scores": [4.2, 3.8, 5.1]}` hoặc `[{"student_id": "19050006", "clo_score": 4.2}, ...]`

Điểm CLO thang 0–6, có thể từ API/backend thay vì filter DiemTong.