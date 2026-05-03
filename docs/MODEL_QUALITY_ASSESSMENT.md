# Đánh giá chất lượng mô hình sau ablation

**Ngày:** 2026-05-03
**Phiên bản:** 1.0
**Câu hỏi:** Sau khi triển khai 3 feedback của giảng viên (Survey, Temporal, Encoding), mô hình có tốt hơn không?

---

## Trả lời thẳng

**Không tốt hơn.** Mọi thay đổi đều bằng hoặc kém baseline.

---

## So sánh thực tế

| Model | MAE | RMSE | R² | Δ MAE so baseline |
|-------|-----|------|----|-------------------|
| **Baseline (76 features, hash)** | **0.3945** | **0.5748** | **0.7980** | — |
| + Survey integration | 0.3954 | 0.5764 | 0.7968 | +0.0009 (kém) |
| + Temporal features | 0.3959 | 0.5762 | 0.7970 | +0.0015 (kém) |
| Frequency encoding | 0.4043 | 0.5821 | 0.7928 | +0.0098 (kém) |
| Target encoding | 0.4354 | 0.6111 | 0.7716 | +0.0409 (kém rõ rệt) |

Không có cấu hình nào vượt baseline. Đây là sự thật được ghi nhận từ [experiments/results.json](../experiments/results.json).

---

## Kết quả này không phải là thất bại

Có 2 cách hiểu:

**Cách hiểu sai (cảm tính):** "Code mới vô dụng, không cải thiện gì."

**Cách hiểu đúng (học thuật):** "Đã chứng minh empirically rằng kiến trúc baseline là tối ưu trên dataset hiện tại. 3 hướng cải tiến phổ biến đều không hoạt động vì lý do cụ thể."

Trong nghiên cứu khoa học, **negative results có giá trị bằng positive results** — miễn là có đủ rigor để rút ra kết luận. Điểm quan trọng:

- 3 ablation rigorous (cùng `random_state=42`, cùng GroupShuffleSplit theo `Student_ID`)
- 3 nguyên nhân kỹ thuật **xác định được**:
  - Coverage attendance 28.6% (Phase 2)
  - Coverage survey 2.6% records (Phase 1)
  - IDs đã được encode qua PPGD/PPDG (Phase 3)
- 3 bảng số liệu đã sẵn sàng đưa vào luận văn

---

## Vấn đề thật của baseline (không phải do feedback gây ra)

Nhìn kỹ hơn vào **subgroup metrics** của baseline:

| Phân khúc điểm | n_samples | MAE | R² |
|---------------|-----------|-----|-----|
| Toàn test | 3,358 | 0.3945 | 0.7980 |
| Low (0–2) | 389 | **0.8234** | **−1.14** |
| Medium (2–4) | 1,549 | 0.3312 | 0.2403 |
| High (4–6) | 1,382 | 0.3297 | **−0.15** |

**Vấn đề thực sự:**

- Model **kém tệ ở predict điểm thấp** — R² = −1.14 nghĩa là tệ hơn dự đoán bằng giá trị trung bình
- R² toàn dataset = 0.798 ấn tượng, nhưng **giả tạo** — chủ yếu nhờ việc model dự đoán đúng "khu vực giữa" cho phần lớn SV
- Trong từng band, model **gần như không có skill thực sự**
- Train MAE 0.148 vs Test MAE 0.395 → **overfit khá nặng** (gap 2.6×)

---

## Đánh giá trung thực về chất lượng

| Tiêu chí | Đánh giá | Giải thích |
|---------|---------|-----------|
| Overall MAE | **Tạm chấp nhận** | 0.39 trên thang 6 ≈ 6.5% lỗi tương đối |
| Overall R² | **Đẹp trên giấy, thực tế ảo** | 0.798 nhưng do dataset bị skew về "trung bình" |
| Predict điểm thấp | **Kém** | R² âm trên 389 SV nhóm Low — đây là nhóm cần predict đúng nhất (cảnh báo nguy cơ trượt) |
| Generalization | **Trung bình** | Train/test gap lớn → overfit |
| Production-ready | **Có, với caveat** | Dùng được nhưng cần caveat khi confidence thấp |

---

## Hướng cải thiện thực sự (ngoài scope feedback giảng viên)

Nếu muốn model thật sự tốt hơn, cần làm 1 trong các hướng sau:

1. **Class-aware loss/sampling cho nhóm Low** — hiện 389/3358 (12%) → resample hoặc dùng `class_weight`
2. **Giảm overfit** — train MAE 0.15 quá thấp → giảm `max_depth`, tăng `min_samples_leaf`, hoặc thêm regularization
3. **Mở rộng coverage attendance ≥70%** — đây mới là gốc của vấn đề (tương đương kết luận từ Phase 2)
4. **Tăng dataset size** — 1,200 SV × 13 môn = 16,133 records, nhưng hiệu lực chỉ 4,609 records sau merge attendance
5. **Calibration cho nhóm Low** — hiện đã có `gb_low_anomaly` blending, nhưng kết quả chưa đủ

---

## Kết luận

| Hạng mục | Kết luận |
|---------|---------|
| **Model mới vs cũ** | Bằng nhau về performance. Không nên dùng artifact ablation cho production. |
| **Best model** | `models/baseline_v0_pre_advisor.joblib` vẫn tối ưu nhất. |
| **Giá trị code mới** | (1) Bằng chứng số liệu cho luận văn (3 bảng ablation); (2) Cơ sở hạ tầng experimental (chạy lại bất cứ lúc nào); (3) Khẳng định empirical rằng thiết kế baseline có cơ sở. |
| **Vấn đề thật của model** | Low-band R² âm + overfit — chưa giải quyết, **nằm ngoài scope feedback** giảng viên. |

---

## Khuyến nghị

### Cho luận văn

- Dùng baseline `MAE=0.3945, R²=0.7980` làm số chính
- Trình bày 3 ablation như **negative results** với giải thích nguyên nhân
- Thẳng thắn về subgroup weakness (Low-band) ở phần "Hạn chế"

### Cho production

- Deploy `baseline_v0_pre_advisor.joblib`
- Bật `predict_with_uncertainty()` để frontend hiển thị confidence interval
- Cảnh báo user khi prediction nằm trong band Low (model không tin được trong vùng này)

### Cho phát triển tiếp theo

- **Phase 5 (không yêu cầu từ giảng viên):** Fix overfit + Low-band — đây mới là hướng có khả năng cải thiện model thật
- Tập trung tăng coverage attendance trước khi thêm bất kỳ feature mới nào

---

## Tham chiếu

- [docs/EXPERIMENTS_LOG.md](EXPERIMENTS_LOG.md) — chi tiết 3 ablation studies
- [docs/advisor_feedback_assessment.md](advisor_feedback_assessment.md) — đánh giá feasibility 4 feedback
- [docs/IMPLEMENTATION_PLAN_ADVISOR_FEEDBACK.md](IMPLEMENTATION_PLAN_ADVISOR_FEEDBACK.md) — kế hoạch triển khai
- [experiments/results.json](../experiments/results.json) — raw metrics
