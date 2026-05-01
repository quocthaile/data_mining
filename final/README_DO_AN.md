# Đồ Án Cuối Kỳ – MOOCCubeX Student Engagement Prediction System

> **Môn Học:** DS317 – Khai Phá Dữ Liệu  
> **Trường:** Đại Học Công Nghệ Thông Tin (UIT)  
> **Người Hướng Dẫn:** ThS. Nguyễn Thị Anh Thư

---

## 📋 Tổng Quan Đồ Án

### Bài Toán

Xây dựng hệ thống **dự đoán sớm mức độ tham gia học tập** (Low / Medium / High) của sinh viên trên nền tảng MOOCCubeX, sử dụng **28 ngày đầu tiên** sau khi đăng ký khóa học. Mục đích: phát hiện sinh viên có nguy cơ không hoàn thành khóa học để **can thiệp kịp thời**.

### Phạm Vi

- **Dataset:** MOOCCubeX (129,516 sinh viên × 21 features từ logs)
- **Khoảng Thời Gian Quan Sát:** 28 ngày đầu (cố định)
- **Mục Tiêu:** 3 mức (Low / Medium / High engagement)
- **Mô Hình:** Linear SVC (Recall = 0.9147 trên test set)

---

## 🔍 Kết Quả Chính

### Mô Hình Chiến Thắng: Linear SVC

| Chỉ Tiêu | Validation | Test | Giải Thích |
|:---|---:|---:|:---|
| **Recall_Low** | 0.9229 | **0.9147** | Phát hiện 91.47% sinh viên nguy cơ ✅ |
| **Accuracy** | 0.6488 | 0.6391 | Độ chính xác tổng thể |
| **Precision_Low** | 0.7338 | 0.7285 | 72.85% dự đoán đúng |

**Tại Sao Linear SVC?**
- Recall cao nhất (giảm False Negative – bỏ sót sinh viên nguy cơ)
- Phù hợp mục tiêu Early Warning System
- Thứ tự xếp hạng: Recall_Low ↑, Accuracy ↑

---

## 🏗️ Kiến Trúc Pipeline

```
Stage 1: Ground Truth      → Sinh nhãn 129,516 sinh viên (Low/Medium/High)
Stage 2: Time Windows      → Trích 8 features từ 28 ngày đầu
Stage 3: Split + SMOTE     → Chia train/valid/test, cân bằng lớp
Stage 4: Model Training    → Huấn luyện 5 mô hình, chọn Linear SVC
Stage 5: Evaluation        → Đánh giá trên test set
```

### Dữ Liệu Đầu Vào & Đầu Ra

| Stage | Input | Output | Hàng | Cột |
|:---|:---|:---|---:|---:|
| **1** | JSON logs | `ground_truth_labels.csv` | 129K | 2 |
| **2** | Raw events | `user_features_28days.csv` | 129K | 8 |
| **3** | Features + Labels | `train/valid/test splits` | 60K/35K/35K | 8 |
| **4** | Train/Valid | `best_model_3w.pkl` + metrics | — | — |
| **5** | Test | `final_test_metrics.csv` | 1 | 5 |

---

## 📊 Dữ Liệu & Đặc Trưng

### Phân Bố Nhãn

```
Low_Engagement (Nguy Cơ Cao)     : 77,710 (59.9%)
Medium_Engagement (Trung Bình)   : 32,378 (25.0%)
High_Engagement (Tốt)            : 19,428 (15.0%)
────────────────────────────────────────────
Tổng                             : 129,516 (100%)
```

### 8 Đặc Trưng Chính (Tính từ 28 ngày đầu)

1. **attempts_3w** – Số lần làm bài
2. **is_correct_3w** – Số câu trả lời đúng
3. **score_3w** – Tổng điểm
4. **accuracy_rate_3w** – Tỷ lệ độ chính xác (%)
5. **num_courses** – Số khóa học tham gia
6. **age** – Tuổi sinh viên
7. **school_encoded** – Trường đại học (mã hóa)
8. **gender** – Giới tính

---

## 📦 Sản Phẩm Giao Nộp

### 1. Báo Cáo Đồ Án

📄 **Vị Trí:** `report/do-an/` (docx, pptx)

**Nội Dung 8 Phần:**
- Tổng quan + Định nghĩa bài toán
- Các công trình nghiên cứu liên quan
- Cơ sở lý thuyết
- Phân tích bộ dữ liệu (EDA)
- Phương pháp đề xuất + kiến trúc
- Thực nghiệm (dataset, phương pháp, độ đo, kịch bản, kết quả)
- Kết luận + đánh giá
- Hướng phát triển

### 2. Sản Phẩm Chạy (Experiment Artifacts)

📁 **Vị Trí:** `final/san-pham/` (copy từ `experiment/dataset/` + `experiment/deployment_models/`)

#### Dữ Liệu:
- `ground_truth_labels.csv` (129,516 × 2)
- `user_features_28days.csv` (129,516 × 8)
- `pre-processing_dataset.csv` (129,516 × 10)
- `train_smote.csv`, `valid_original.csv`, `test_original.csv`

#### Mô Hình & Kết Quả:
- `best_model_3w.pkl` (Linear SVC đã huấn luyện)
- `evaluation_metrics.csv` (So sánh 5 mô hình)
- `final_test_metrics.csv` (Linear SVC test results)

### 3. Source Code

💾 **Vị Trí:** `final/source-code/` (hoặc GitHub link)

```
experiment/
├── config.py                      (Cấu hình tập trung)
├── stage_1_generate_ground_truth.py
├── stage_2_time_window_features.py
├── stage_3_split_and_smote.py
├── stage_4_model_training_eval.py
├── stage_5_explain_model_xai.py (nếu có)
├── run_pipeline.py               (Orchestrator)
└── deployment_models/            (Mô hình + metrics)

README.md (tài liệu chính với kết quả)
experiment/README_RUN_PIPELINE.md (hướng dẫn chi tiết)
```

### 4. Video Thuyết Trình

🎥 **Vị Trí:** `final/video/` (hoặc YouTube link)

- **Nội Dung:** Trình bày toàn bộ báo cáo (~10 phút)
- **Yêu Cầu:** Tất cả thành viên tham gia, bật camera
- **Cấu Trúc:**
  - Giới thiệu bài toán (1 phút)
  - Dữ liệu & phương pháp (2 phút)
  - Kết quả & mô hình (3 phút)
  - Đánh giá & định hướng (2 phút)
  - QA (2 phút)

---

## ✅ Danh Sách Kiểm Tra (Checklist)

### Dữ Liệu

- [ ] Ground truth: 129,516 dòng, 3 mức độ cân bằng (59.9% / 25% / 15%)
- [ ] Features 28 ngày: 129,516 dòng, 8 cột đặc trưng
- [ ] Pre-processing dataset: 129,516 dòng, 10 cột + target
- [ ] Train set: Cân bằng 1:1:1 (SMOTE)
- [ ] Valid/Test set: Phân bố tự nhiên

### Mô Hình

- [ ] Linear SVC: Recall_Low ≥ 0.91 ✅
- [ ] Linear SVC: Accuracy ≈ 0.64 ✅
- [ ] Model bundle có scalers + encoders ✅

### Báo Cáo

- [ ] Phần "Kết Quả Thực Nghiệm" đầy đủ ✅
- [ ] Bảng so sánh 5 mô hình ✅
- [ ] Giải thích tại sao chọn Linear SVC ✅
- [ ] Phân tích kết quả (độ chính xác, recall) ✅

### Code

- [ ] Tất cả `.py` files pass syntax check
- [ ] Config: TIME_WINDOW_MODE = "fixed" (bắt buộc)
- [ ] Không có file relative window dư thừa
- [ ] Có file `__init__.py` trong package (nếu cần)

---

## 🚀 Cách Chạy

### 1. Chạy Toàn Bộ Pipeline (5 Stage)

```bash
cd /path/to/project
python experiment/run_pipeline.py --phase all
```

### 2. Chạy Riêng Từng Stage

```bash
python experiment/run_pipeline.py --phase 1  # Ground truth
python experiment/run_pipeline.py --phase 2  # Features
python experiment/run_pipeline.py --phase 3  # Split + SMOTE
python experiment/run_pipeline.py --phase 4  # Model training
python experiment/run_pipeline.py --phase 5  # Evaluation
```

### 3. Test Nhanh Với Dữ Liệu Mẫu

```bash
python experiment/run_pipeline.py --phase all --max-rows 1000
```

---

## 📝 Ghi Chú Quan Trọng

### Lý Do Chọn FIXED 28 Days

✅ **Ưu Điểm:**
- Đơn giản, dễ tái tạo, dễ giải thích
- Hiệu năng tương đương Relative modes
- Tiêu chí chuẩn (4 tuần quan sát)
- Cho phép can thiệp sớm

### Lý Do Chọn Linear SVC

✅ **Ưu Điểm:**
- Recall cao nhất: 0.9147 (phát hiện 91.47% sinh viên nguy cơ)
- Giảm False Negative (chi phí cao)
- Phù hợp Early Warning System

### Độ Chính Xác Không Quá Cao (63.91%)

📌 **Lý Do:**
- Mất cân bằng lớp (60% Low vs 15% High)
- Features 28 ngày có thể chưa đủ
- Cần feature engineering thêm

---

## 🔮 Hướng Phát Triển

1. **Feature Engineering:** Interaction features, sliding windows
2. **Ensemble Methods:** Kết hợp Linear SVC + Logistic Regression
3. **Threshold Tuning:** Tối ưu Precision-Recall trade-off
4. **Real-time System:** Dự đoán hằng ngày/tuần
5. **Intervention Tracking:** Theo dõi hiệu quả can thiệp

---

## 📞 Liên Hệ & Hỗ Trợ

**Nếu có vấn đề:**
- Kiểm tra `config.py` → `TIME_WINDOW_MODE = "fixed"`
- Xem `experiment/README_RUN_PIPELINE.md` → Chi tiết từng stage
- Kiểm tra `README.md` → Kết quả so sánh
- Xem log output → Chi tiết lỗi

---

**Hoàn Thành:** 2025-01-30  
**Chế Độ:** FIXED 28-DAY (Chỉ FIXED, loại bỏ Relative)  
**Mô Hình Chiến Thắng:** Linear SVC  
**Test Recall:** 0.9147 ✅ | **Test Accuracy:** 0.6391 ✅
