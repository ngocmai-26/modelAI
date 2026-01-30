MÔ TẢ MODEL

- Dữ liệu hiện có

- Điểm rèn luyện
- Điểm tổng
- Nhân khẩu
- Phương pháp đánh giá
- Phương pháp giảng dạy
- Giờ thư viện
- Dữ liệu điểm danh
- Khảo sát đầu năm (chưa bổ sung)
- Khảo sát từng học kì (chưa bổ sung)

- Chức năng chính
- Phân tích tương quan
  - Phương pháp giảng dạy và kết quả thi

**Dữ liệu sử dụng**

- PPGD.xlsx / PPDG.xlsx: phương pháp giảng dạy (TM\*, EM\*)
- Diem_Tong.xlsx / testne.xlsx: điểm thi

**Xử lý dữ liệu**

- Chuẩn hóa Subject_ID
- Mã hóa phương pháp giảng dạy: X → 1, còn lại/NaN → 0
- Làm sạch exam_score, loại bỏ giá trị không hợp lệ, chuyển sang float
- Tạo biến nhị phân Result (Đạt ≥ 6, Rớt < 6)
- Gộp dữ liệu theo Subject_ID

**Phương pháp / mô hình**

- Phân tích tương quan (Pearson)
- Phân tích so sánh phân phối

**Biểu đồ đầu ra**

- Heatmap (ma trận tương quan)
- Histogram (phân phối điểm thi)
- Violin plot (điểm thi theo từng phương pháp)

**Mục tiêu**

Mục tiêu của phân tích này là **xác định mức độ và chiều hướng mối tương quan** giữa các phương pháp giảng dạy được áp dụng trong học phần và **kết quả thi của sinh viên**, bao gồm cả **điểm số thực tế** và **kết quả Đạt/Không đạt**. Qua đó, phân tích nhằm đánh giá **hiệu quả tương đối của từng phương pháp giảng dạy** đối với thành tích học tập.

- 1. Thời gian học, điểm rèn luyện và kết quả thi

**Dữ liệu sử dụng**

- Diem_Tong.xlsx: điểm thi
- Gio_TV_22-24.xlsx: giờ học tích lũy
- DRL_2021_2024.xlsx: điểm rèn luyện

**Xử lý dữ liệu**

- Chuẩn hóa Student_ID, year
- Làm sạch exam_score, tạo Result
- Gộp 3 bảng theo Student_ID và year
- Loại bỏ dòng thiếu accumulated_study_hours, conduct_score

**Phương pháp / mô hình**

- Phân tích tương quan (Pearson)
- Phân tích mối quan hệ tuyến tính

**Biểu đồ đầu ra**

- Heatmap (tương quan)
- Scatter plot (giờ học/điểm rèn luyện - điểm thi)
- Boxplot (điểm thi theo Result)

**Mục tiêu**

Phân tích này hướng đến việc **khám phá mối liên hệ giữa hành vi học tập và kết quả học tập**, thông qua việc xem xét **thời gian học tích lũy** và **điểm rèn luyện** của sinh viên có ảnh hưởng như thế nào đến **điểm thi**. Mục tiêu là làm rõ vai trò của **nỗ lực cá nhân, tính kỷ luật và thái độ học tập** đối với thành tích học tập.

- 1. Nhân khẩu học và kết quả thi

**Dữ liệu sử dụng**

- Data K24 25 26 27.xlsx: giới tính, nơi sinh, dân tộc, tôn giáo
- Diem_Tong.xlsx: điểm thi

**Xử lý dữ liệu**

- Chuẩn hóa Student_ID
- Làm sạch và chuẩn hóa exam_score
- Mã hóa:
  - Giới tính: nhị phân
  - Nơi sinh → vùng miền (Bắc/Trung/Nam)
  - Dân tộc, tôn giáo → số nguyên

**Phương pháp / mô hình**

- Phân tích tương quan
- Phân tích so sánh nhóm

**Biểu đồ đầu ra**

- Heatmap (tương quan)
- Bar plot (điểm trung bình theo nhóm)
- Boxplot & Strip plot (phân phối điểm thi)

Mục tiêu

Mục tiêu của phần này là **phân tích mối tương quan và sự khác biệt về kết quả thi** giữa các nhóm sinh viên dựa trên **đặc điểm nhân khẩu học**, bao gồm giới tính, nơi sinh (vùng miền), dân tộc và tôn giáo. Phân tích nhằm phát hiện các **xu hướng hoặc chênh lệch tiềm ẩn** trong kết quả học tập gắn với bối cảnh cá nhân và xã hội của sinh viên.

- 1. Phân tích tổng hợp và kiểm định thống kê

**Dữ liệu sử dụng**

- Diem_Tong.xlsx
- DRL_2021_2024.xlsx
- Gio_TV_22-24.xlsx
- Data K24 25 26 27.xlsx

**Xử lý dữ liệu**

- Gộp toàn bộ theo Student_ID, year
- Chuẩn hóa và làm sạch tất cả biến chính
- Mã hóa biến phân loại

**Phương pháp / mô hình**

- Thống kê mô tả
- T-test (giới tính)
- ANOVA (vùng miền)
- Phân tích tương quan tổng hợp

**Biểu đồ đầu ra**

- Histogram (điểm thi)
- Boxplot (biến định lượng)
- Heatmap (tương quan tổng hợp)
- Bar plot & Pie chart (biến phân loại)

Mục tiêu

Phần phân tích này nhằm **cung cấp cái nhìn tổng quan và hệ thống về toàn bộ dữ liệu nghiên cứu** thông qua thống kê mô tả các biến chính. Đồng thời, nghiên cứu tiến hành **các kiểm định thống kê (T-test, ANOVA)** và xây dựng **ma trận tương quan tổng hợp** để đánh giá **tác động tổng hợp của các yếu tố học tập, hành vi và nhân khẩu học** đến kết quả thi của sinh viên.

- Dự đoán điểm CLO (Dành cho cá nhân)
  - Mục tiêu

Mô hình được xây dựng nhằm **dự đoán trực tiếp điểm CLO của sinh viên trên thang điểm từ 0 đến 6**, dựa trên thông tin về sinh viên, giảng viên, môn học và lịch sử học tập.  
Kết quả dự đoán phản ánh **mức độ đạt được chuẩn đầu ra học phần (CLO)** của sinh viên và được sử dụng làm đầu vào cho các phân tích nguyên nhân và đề xuất giải pháp cải thiện kết quả học tập.

- 1. Dữ liệu training
- Điểm rèn luyện
- Điểm tổng
- Nhân khẩu
- Phương pháp đánh giá
- Phương pháp giảng dạy
- Giờ thư viện
- Dữ liệu điểm danh
- Khảo sát đầu năm (chưa bổ sung)
- Khảo sát từng học kì (chưa bổ sung)
  - Dữ liệu đầu vào (tham khảo)

A. Thông tin cơ bản:

\- student_id_encoded (mã hóa)

\- lecturer_encoded (mã hóa)

\- subject_encoded (mã hóa)

B. Thông tin nhân khẩu:

\- gender_encoded (giới tính)

\- religion_encoded (tôn giáo)

\- birth_place_encoded (nơi sinh)

\- ethnicity_encoded (dân tộc)

C. Điểm rèn luyện:

\- avg_conduct_score (điểm rèn luyện trung bình)

\- latest_conduct_score (điểm rèn luyện gần nhất)

\- conduct_trend (xu hướng điểm rèn luyện)

D. Tự học:

\- study_hours_this_year (số giờ tự học trong năm)

\- total_study_hours (tổng giờ tự học)

E. Lịch sử học tập:

\- total_subjects (tổng số môn đã học)

\- passed_subjects (số môn đã pass)

\- pass_rate (tỉ lệ pass)

\- avg_exam_score (điểm thi trung bình)

\- recent_avg_score (điểm trung bình gần đây)

\- improvement_trend (xu hướng cải thiện)

F. Phương pháp giảng dạy & đánh giá:

\- Teaching method features (từ PPGDfull.xlsx)

\- Assessment method features (từ PPDGfull.xlsx)

- 1. Mô hình

Bài toán được tiếp cận dưới dạng **hồi quy (Regression)** và áp dụng **Ensemble Learning** để tăng độ chính xác và tính ổn định.

- **Random Forest Regressor**

Random Forest Regressor được sử dụng để mô hình hóa **các mối quan hệ phi tuyến và tương tác phức tạp** giữa các đặc trưng học tập của sinh viên.

- n_estimators = 1000
- max_depth = 25
- random_state = 42

- **Gradient Boosting Regressor**

Gradient Boosting Regressor được sử dụng nhằm học **các xu hướng tinh vi và phần sai số còn lại** mà Random Forest chưa khai thác hiệu quả.

- n_estimators = 500
- max_depth = 12
- learning_rate = 0.03
- random_state = 42

- **Ensemble Regression**

Kết quả dự đoán điểm CLO cuối cùng được tính bằng **trung bình có trọng số** của hai mô hình hồi quy, trong đó trọng số được xác định dựa trên hiệu suất của từng mô hình trên tập validation.

- 1. Quy trình
- B1: Sinh viên nhập: mssv, msgv và mã số môn học
- B2: Đưa vào mô hình: mô hình sẽ dựa vào các dữ liệu đã được trainning để dự đoán điểm cho sinh viên, vì là dự đoán điểm CLO nên là điểm từ 0 -> 6. Lưu ý do phần điểm Thi là hệ 10 : sẽ đổi về điểm hệ 6 để training
- B3: Đưa ra kết quả.

MÔ TẢ HỆ THỐNG DỰ ĐOÁN & PHÂN TÍCH NGUYÊN NHÂN - GIẢI PHÁP (XAI-BASED)

**1\. Mục tiêu tổng thể**

Hệ thống được xây dựng nhằm:

- **Dự đoán điểm CLO (thang điểm 6)** của sinh viên
- **Phân tích nguyên nhân và đề xuất giải pháp** dựa trên dữ liệu học tập và hành vi
- **Giải thích mức độ ảnh hưởng của từng yếu tố** bằng XAI
- Hỗ trợ phân tích ở **2 mức độ**:
  - Cá nhân sinh viên
  - Toàn bộ lớp học
- **Lưu trữ dữ liệu kết quả để tiếp tục huấn luyện mô hình trong tương lai**

**2\. Phân loại chế độ phân tích**

**2.1 Phân tích cá nhân (Individual Analysis)**

Áp dụng cho **một sinh viên - một môn học**.

**Kết quả trả về**:

- Điểm CLO dự đoán (0-6)
- Danh sách **nguyên nhân chính** (XAI-based)
- Danh sách **giải pháp tương ứng**
- **Mức độ ảnh hưởng (%)** của từng nguyên nhân

👉 Mục tiêu: hỗ trợ **can thiệp cá nhân hóa**.

**2.2 Phân tích lớp (Class Analysis)**

Áp dụng cho **toàn bộ sinh viên trong một lớp học phần**.

**Kết quả trả về**:

- **Không dự đoán điểm cho từng cá nhân**
- Tổng hợp:
  - Các nguyên nhân phổ biến nhất của lớp
  - Mức độ ảnh hưởng trung bình của từng nguyên nhân
  - Giải pháp ưu tiên cho lớp

👉 Mục tiêu: hỗ trợ **điều chỉnh giảng dạy và quản lý lớp**.

**3\. Dữ liệu sử dụng trong hệ thống**

**3.1 Các nhóm dữ liệu đầu vào**

Hệ thống sử dụng các nguồn dữ liệu sau:

- Điểm rèn luyện
- Điểm tổng (điểm thi cuối kỳ)
- Nhân khẩu học
- Phương pháp đánh giá (PPDG)
- Phương pháp giảng dạy (PPGD)
- Giờ thư viện / tự học
- Dữ liệu điểm danh
- Khảo sát đầu năm (chưa triển khai - mở rộng sau)
- Khảo sát từng học kỳ (chưa triển khai - mở rộng sau)

**4\. Quy ước thang điểm và xử lý dữ liệu điểm (RẤT QUAN TRỌNG)**

**4.1 Giai đoạn hiện tại (dữ liệu lịch sử)**

- File **điểm tổng hiện tại**:
  - Chỉ sử dụng **điểm thi cuối kỳ**
  - Thang điểm gốc: **hệ 10**
- Trước khi đưa vào mô hình:
  - **Chuyển đổi điểm từ hệ 10 sang hệ 6**
  - Công thức:

👉 Sau bước này:

- Mô hình **chỉ nhìn thấy điểm hệ 6**
- Không còn khái niệm hệ 10 trong quá trình train

**4.2 Giai đoạn tương lai (dữ liệu từ hệ thống)**

- File nhập điểm **trực tiếp là CLO hệ 6**
- **Không cần chuyển đổi**
- Dữ liệu được lưu **nguyên trạng** để huấn luyện tiếp

👉 Quy ước cốt lõi:

**Toàn bộ dữ liệu dùng để train mô hình đều là hệ 6**

**5\. Lưu trữ dữ liệu để huấn luyện tiếp (Learning Continuity)**

**5.1 Nguyên tắc lưu dữ liệu (bắt buộc)**

Khi phân tích lớp:

- Hệ thống **phải lưu lại dữ liệu điểm CLO thực tế** của sinh viên
- Dữ liệu lưu trữ:
  - student_id
  - subject_id
  - lecturer_id
  - điểm CLO (hệ 6)
  - các feature liên quan (rèn luyện, tự học, điểm danh, …)
  - học kỳ / thời gian

👉 Mục đích:

- Dùng làm **training data cho các lần huấn luyện tiếp theo**
- Không cần xử lý lại thang điểm

**5.2 Chiến lược huấn luyện lại (Retraining)**

- Khi có dữ liệu mới:
  - Gộp dữ liệu cũ + dữ liệu mới (đều hệ 6)
  - Thực hiện **huấn luyện lại toàn bộ mô hình**
- Không dùng incremental training
- Mỗi lần retrain:
  - Tăng version mô hình
  - Ghi log số lượng dữ liệu mới

**6\. Mô hình dự đoán (Prediction Model)**

**6.1 Bài toán**

- Hồi quy (Regression)
- Biến mục tiêu: **điểm CLO (0-6)**

**6.2 Mô hình sử dụng**

- Random Forest Regressor
- Gradient Boosting / XGBoost Regressor
- Ensemble averaging

# 7\. XAI - PHÂN TÍCH NGUYÊN NHÂN VÀ MỨC ĐỘ ẢNH HƯỞNG

## 7.1 Kỹ thuật XAI sử dụng

Trong nghiên cứu này, **SHAP (SHapley Additive Explanations)** được sử dụng để giải thích kết quả dự đoán của mô hình học máy.

SHAP cho phép:

- Phân rã giá trị dự đoán thành **đóng góp của từng đặc trưng**
- Xác định rõ:
  - Đặc trưng nào **làm tăng** điểm CLO
  - Đặc trưng nào **làm giảm** điểm CLO
- Đảm bảo tính:
  - Nhất quán
  - Giải thích được
  - Có cơ sở toán học (dựa trên Shapley values)

SHAP đặc biệt phù hợp với các mô hình **tree-based** như Random Forest và Gradient Boosting, vốn được sử dụng trong nghiên cứu này.

## 7.2 Quy trình áp dụng XAI

### 7.2.1 Phân tích ở mức cá nhân

Đối với mỗi sinh viên, quy trình phân tích XAI được thực hiện như sau:

**Bước 1: Tính SHAP values**

- Áp dụng TreeExplainer để tính SHAP cho từng đặc trưng đầu vào
- Mỗi đặc trưng nhận một giá trị SHAP biểu thị mức độ ảnh hưởng đến điểm CLO dự đoán

**Bước 2: Phân loại mức ảnh hưởng**

- SHAP > 0: Đặc trưng có tác động tích cực (kéo điểm tăng)
- SHAP < 0: Đặc trưng có tác động tiêu cực (kéo điểm giảm)

**Bước 3: Lọc các đặc trưng quan trọng**

- Chỉ giữ các đặc trưng có |SHAP| lớn hơn ngưỡng xác định
- Ưu tiên các đặc trưng có SHAP âm lớn nhất để xác định nguyên nhân chính

**Bước 4: Gom nhóm đặc trưng theo nguyên nhân**  
Các đặc trưng được gom theo các nhóm sư phạm:

| **Nhóm nguyên nhân** | **Đặc trưng** |
| --- | --- |
| Tự học | Giờ thư viện |
| Chuyên cần | Tỷ lệ điểm danh |
| Rèn luyện | Điểm rèn luyện |
| Học lực | Điểm tổng |
| Giảng dạy | Phương pháp giảng dạy |
| Đánh giá | Phương pháp đánh giá |
| Cá nhân | Nhân khẩu học |

**Bước 5: Tính mức độ ảnh hưởng**

- Mức độ ảnh hưởng của từng nguyên nhân được tính bằng:
  - Tổng SHAP âm của các đặc trưng trong cùng nhóm
- Chuẩn hóa về thang phần trăm (%) để dễ diễn giải

### 7.2.2 Phân tích ở mức lớp

Đối với phân tích lớp, XAI được sử dụng theo cách tổng hợp:

**Bước 1: Tính SHAP cho toàn bộ sinh viên trong lớp**

- Áp dụng SHAP cho từng sinh viên riêng lẻ

**Bước 2: Tổng hợp SHAP**

- Lấy giá trị trung bình của SHAP theo từng đặc trưng
- Sau đó tiếp tục gom theo nhóm nguyên nhân

**Bước 3: Xác định nguyên nhân chủ đạo của lớp**

- Các nhóm có tổng SHAP âm lớn nhất được xem là nguyên nhân chính ảnh hưởng đến kết quả của lớp
- Mức độ ảnh hưởng được biểu diễn bằng tỷ lệ phần trăm

# 8\. SINH NGUYÊN NHÂN VÀ GIẢI PHÁP

## 8.1 Sinh nguyên nhân (Reason Generation)

Nguyên nhân học tập không được lấy sẵn từ dataset mà **được sinh động từ kết quả XAI**, đảm bảo tính cá nhân hóa và bám sát dữ liệu.

### Quy trình sinh nguyên nhân

**Bước 1: Xác định nhóm nguyên nhân trọng yếu**

- Dựa trên nhóm có tổng SHAP âm lớn nhất

**Bước 2: Xây dựng mô tả nguyên nhân**

- Nguyên nhân được mô tả bằng ngôn ngữ sư phạm, tập trung vào:
  - Hành vi học tập
  - Mức độ tham gia
  - Cách học và môi trường học tập

**Ví dụ:**

Sinh viên có số giờ tự học thấp và tỷ lệ chuyên cần chưa cao, dẫn đến kết quả học tập chưa đạt yêu cầu.

## 8.2 Sinh giải pháp (Solution Mapping)

Giải pháp không được sinh bằng mô hình học máy mà **được ánh xạ từ nguyên nhân thông qua luật và mẫu (rule-based templates)**.

### Nguyên tắc xây dựng giải pháp

- Dễ hiểu đối với sinh viên và giảng viên
- Có tính hành động (actionable)
- Phù hợp với bối cảnh giáo dục đại học

### Quy trình ánh xạ

**Bước 1: Gán nhãn nguyên nhân**

- Mỗi nguyên nhân được gán một reason_key (ví dụ: LOW_SELF_STUDY, LOW_ATTENDANCE)

**Bước 2: Truy xuất tập giải pháp**

- Mỗi reason_key tương ứng với một tập giải pháp được thiết kế sẵn

**Bước 3: Cá nhân hóa giải pháp**

- Điều chỉnh nội dung giải pháp theo:
  - Mức độ ảnh hưởng
  - Ngữ cảnh cá nhân hoặc lớp

**Ví dụ giải pháp:**

- Tăng thời gian tự học tại thư viện tối thiểu X giờ mỗi tuần
- Thiết lập kế hoạch học tập cá nhân và theo dõi tiến độ
- Tham gia nhóm học tập có hướng dẫn

**9\. Kết quả đầu ra**

**9.1 Cá nhân**

- Điểm CLO dự đoán
- Top nguyên nhân
- Giải pháp tương ứng
- Mức độ ảnh hưởng (%)

**9.2 Lớp**

- Danh sách nguyên nhân phổ biến
- Mức độ ảnh hưởng trung bình
- Giải pháp ưu tiên cho lớp