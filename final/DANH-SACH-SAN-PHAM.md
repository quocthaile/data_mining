# Danh Sách Sản Phẩm Đồ Án – MOOCCubeX Student Engagement Prediction

## 1. Báo Cáo & Trình Bày

### Thực Hành (10đ)

- [ ] **Thuyết minh đề tài** (`.docx`, `.pptx`)
  - Mô tả bài toán, dữ liệu, đặc trưng
  - Lý do chọn phương pháp, phân tích EDA
  - Quy trình xử lý, kết quả thực nghiệm
  - **Vị trí nộp:** `report/thuc-hanh/`

- [ ] **Báo cáo phân tích bộ dữ liệu** (`.docx`)
  - Phân tích EDA chi tiết
  - Làm sạch dữ liệu, xử lý missing values
  - Biến đổi đặc trưng
  - **Vị trí nộp:** `report/thuc-hanh/`

- [ ] **Video thuyết trình Thực hành**
  - Tất cả thành viên tham gia, bật camera
  - Thuyết trình tổng quan về bài toán
  - Giới thiệu dữ liệu, pipeline
  - **Vị trí lưu:** `final/video/`

### Đồ Án Cuối Kỳ (10đ)

- [ ] **Báo cáo đồ án** (`.docx`, `.pptx`)
  - Tổng quan + bài toán
  - Các công trình nghiên cứu liên quan
  - Cơ sở lý thuyết
  - Phân tích bộ dữ liệu
  - Phương pháp đề xuất + kiến trúc
  - Thực nghiệm: mô tả dataset, phương pháp, độ đo, kịch bản, kết quả
  - Kết luận + hướng phát triển
  - **Vị trí nộp:** `report/do-an/`

- [ ] **Video thuyết trình Đồ Án**
  - Tất cả thành viên tham gia, bật camera
  - Trình bày toàn bộ báo cáo (8-10 phút)
  - **Vị trí lưu:** `final/video/`

---

## 2. Sản Phẩm Chạy – Thực Nghiệm (Experiment Artifacts)

**Vị Trí Gốc:** `experiment/dataset/` + `experiment/deployment_models/`  
**Vị Trí Nộp:** `final/san-pham/` (hoặc symlink/copy)

### Dữ Liệu Thực Nghiệm

- [ ] **Ground Truth Labels** 
  - File: `ground_truth_labels.csv`
  - Mô tả: 129,516 sinh viên + nhãn mục tiêu (Low/Medium/High)
  - **Nguồn:** `experiment/dataset/ground_truth_labels.csv`

- [ ] **Đặc Trưng 28 Ngày** 
  - File: `user_features_28days.csv`
  - Mô tả: Features trích từ 28 ngày đầu (8 cột đặc trưng)
  - **Nguồn:** `experiment/dataset/user_features_28days.csv`

- [ ] **Dữ Liệu Tiền Xử Lý** 
  - File: `pre-processing_dataset.csv`
  - Mô tả: Dataset hoàn chỉnh 10 cột (school, attempts, score, age, gender, target)
  - **Nguồn:** `experiment/dataset/pre-processing_dataset.csv`

- [ ] **Tập Train/Valid/Test**
  - Files: `train_smote.csv`, `valid_original.csv`, `test_original.csv`
  - Mô tả: Tập dữ liệu chia sẵn, train đã cân bằng SMOTE
  - **Nguồn:** `experiment/dataset/model_data/`

### Mô Hình & Kết Quả

- [ ] **Mô Hình Huấn Luyện** 
  - File: `best_model_3w.pkl`
  - Mô tả: Linear SVC đã huấn luyện + scalers + encoders
  - **Nguồn:** `experiment/deployment_models/best_model_3w.pkl`

- [ ] **Kết Quả Đánh Giá Validation** 
  - File: `evaluation_metrics.csv`
  - Mô tả: Metrics 5 mô hình (Accuracy, Recall, Precision, AUC)
  - Nội dung: Linear SVC đứng đầu với Recall_Low = 0.9229
  - **Nguồn:** `experiment/deployment_models/evaluation_metrics.csv`

- [ ] **Kết Quả Đánh Giá Test** 
  - File: `final_test_metrics.csv`
  - Mô tả: Metrics Linear SVC trên test set (Accuracy=0.6391, Recall=0.9147)
  - **Nguồn:** `experiment/deployment_models/final_test_metrics.csv`

---

## 3. Source Code

**Vị Trí:** `final/source-code/`

### Core Pipeline (Experiment)

- [ ] `experiment/config.py` – Cấu hình tập trung, FIXED 28-day mode
- [ ] `experiment/stage_1_generate_ground_truth.py` – Sinh nhãn ground truth
- [ ] `experiment/stage_2_time_window_features.py` – Trích đặc trưng 28 ngày
- [ ] `experiment/stage_3_split_and_smote.py` – Chia tập, SMOTE
- [ ] `experiment/stage_4_model_training_eval.py` – Huấn luyện 5 mô hình, chọn best
- [ ] `experiment/stage_5_explain_model_xai.py` – XAI (nếu có)
- [ ] `experiment/run_pipeline.py` – Orchestrator với menu tương tác

### Hỗ Trợ & Demo

- [ ] `final/source-code/check_environment.py` – Kiểm tra môi trường Python
- [ ] `final/source-code/demo_app.py` – Demo ứng dụng (nếu có UI)

### Tài Liệu

- [ ] `README.md` – Tài liệu chính, bao gồm kết quả thực nghiệm
- [ ] `experiment/README_RUN_PIPELINE.md` – Hướng dẫn chạy pipeline

---

## 4. Chất Lượng & Kiểm Tra

### Kiểm Tra Chung

- [ ] Tất cả file `.py` pass syntax check (`python -m py_compile`)
- [ ] Tất cả file `.csv` có đầy đủ dòng + cột theo mô tả
- [ ] README.md cập nhật kết quả thực nghiệm (Benchmark table)
- [ ] Không có file tạm/debug: xóa `user_features_relative_*.csv`, `time_window_comparison.csv`
- [ ] Không có `.pyc` files không cần thiết (dọn dẹp `__pycache__`)

### Kiểm Tra Dữ Liệu

- [ ] Ground truth: **129,516 dòng**, 3 mức độ (Low/Medium/High)
  - Low: 77,710 (59.9%)
  - Medium: 32,378 (25.0%)
  - High: 19,428 (15.0%)
- [ ] Features 28 ngày: **129,516 dòng**, 8 cột đặc trưng
- [ ] Pre-processing dataset: **129,516 dòng**, 10 cột + target
- [ ] Train set sau SMOTE: **Cân bằng 1:1:1** (20K mẫu mỗi lớp)
- [ ] Valid/test set: **Phân bố tự nhiên** (62% Low, 25% Medium, 13% High)

### Kiểm Tra Mô Hình

- [ ] Linear SVC: **Recall_Low ≥ 0.91** trên test set ✅ (0.9147)
- [ ] Linear SVC: **Accuracy ≈ 0.64** ✅ (0.6391)
- [ ] Model bundle có: scalers + label encoders ✅
- [ ] Không có model khác được lưu (chỉ Linear SVC)

### Kiểm Tra Báo Cáo

- [ ] Báo cáo có phần **"Kết Quả Thực Nghiệm"** bao gồm:
  - Kịch bản thực nghiệm (28-day FIXED mode)
  - Bảng kết quả validation (5 mô hình)
  - Bảng kết quả test (Linear SVC)
  - Giải thích tại sao chọn Linear SVC
  - Phân tích độ chính xác & Recall

---

## 5. Cấu Hình Chính

### Chế Độ FIXED 28 Days (BẮTBUỘC)

```python
# experiment/config.py
TIME_WINDOW_MODE = "fixed"     # Bắt buộc, không dùng "relative"
DEFAULT_OBSERVATION_DAYS = 28  # Chính xác 28 ngày
```

**Lý Do:**
- ✅ Đơn giản, dễ giải thích, tái tạo
- ✅ Hiệu năng tương đương Relative modes (xem COMPARISON_TABLE.md)
- ✅ Tiêu chí thống nhất (28 ngày = 4 tuần quan sát hành vi)
- ✅ Phù hợp để can thiệp sớm trước khi khóa học kết thúc

### Mô Hình Chiến Thắng

```
Linear SVC
├─ Validation: Recall_Low = 0.9229 (cao nhất)
├─ Test: Recall_Low = 0.9147 (cao nhất)
└─ Lý do: Ưu tiên phát hiện sinh viên có nguy cơ
```

**Tiêu Chí Xếp Hạng (stage_4_model_training_eval.py dòng 151):**
1. **Recall_Low_Engagement** (ưu tiên 1) → Phát hiện sớm
2. **Accuracy** (ưu tiên 2) → Độ chính xác tổng thể

---

## 6. Tệp Cần Loại Bỏ (Đã Xóa)

✅ **Đã Xóa:**
- `dataset/user_features_relative_25.csv` (không dùng Relative 25%)
- `dataset/user_features_relative_50.csv` (không dùng Relative 50%)
- `dataset/time_window_comparison.csv` (chỉ dùng cho so sánh, không submit)
- `benchmark_runner.py`, `show_benchmark_results.py` (tạm thời)

---

## 7. Hướng Dẫn Nộp Bài

### Trước Hạn Chót

1. **Hoàn Thành Báo Cáo**
   - Thực hành: Thuyết minh + báo cáo EDA
   - Đồ án: Báo cáo 8 phần (overview, related work, theory, data, method, experiment, conclusion)

2. **Chuẩn Bị Sản Phẩm**
   - Copy dữ liệu vào `final/san-pham/`
   - Verifydata integrity (dòng, cột, phân bố)
   - Kiểm tra model metrics

3. **Quay Video**
   - Thực hành: ~5 phút giới thiệu bài toán + pipeline
   - Đồ án: ~10 phút trình bày toàn bộ báo cáo
   - Tất cả thành viên, bật camera

4. **Nộp Bài**
   - `report/thuc-hanh/` – Báo cáo TH
   - `report/do-an/` – Báo cáo ĐA
   - `final/video/` – Video (hoặc link YouTube)
   - `final/san-pham/` – Dữ liệu + mô hình (hoặc link Kaggle)

---

## 8. Timeline & Trao Đổi

| Giai Đoạn | Deadline | Nội Dung | Status |
|-----------|----------|---------|--------|
| **Chuẩn Bị** | — | Hoàn thành báo cáo draft | [ ] |
| **Sản Phẩm** | — | Chuẩn bị dữ liệu + mô hình | [ ] |
| **Video** | — | Quay video thuyết trình | [ ] |
| **Kiểm Tra** | — | Review tất cả sản phẩm | [ ] |
| **Nộp Bài** | — | Nộp lên LMS/Email | [ ] |

---

**Cập nhật:** 2025-01-30  
**Chế Độ:** FIXED 28-DAY (loại bỏ Relative)  
**Mô Hình:** Linear SVC  
**Recall Test:** 0.9147 ✅  
**Accuracy Test:** 0.6391 ✅
