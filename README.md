# Student Engagement Prediction on MOOCCubeX

> **Môn học:** DS317 – Khai phá Dữ liệu | Trường Đại học Công nghệ Thông tin (UIT)

Dự án khai phá dữ liệu dự đoán **mức độ tham gia học tập** (Low / Medium / High) của học sinh trên nền tảng [MOOCCubeX](https://github.com/THU-KEG/MOOCCube), sử dụng đặc trưng hành vi trích xuất từ log tương tác quy mô lớn và mô hình phân loại có giám sát.

---

## 📁 Phân chia thư mục theo hạng mục

```
project/
├── experiment/      ← THỰC HÀNH: Source code pipeline thực nghiệm
├── final/           ← ĐỒ ÁN: Source code đồ án
├── reports/         ← SẢN PHẨM CHẠY: Artifacts sinh ra từ pipeline
│   ├── thuc-hanh/   ← Sản phẩm thực hành
│   └── do-an/       ← Sản phẩm đồ án
└── report/          ← NỘP BÀI: Báo cáo, thuyết minh, slide (.docx, .pptx)
    ├── thuc-hanh/   ← Tài liệu nộp cho phần Thực hành
    └── do-an/       ← Tài liệu nộp cho phần Đồ án môn học
```

---

## 🎓 Yêu cầu môn học & Vị trí file

### Thực hành

> 📂 Code: **`experiment/`** | 📦 Sản phẩm chạy từ pipeline: **`reports/do-an/`** | 📄 Báo cáo & slide: **`report/thuc-hanh/`**

| STT | Hạng mục nộp | Vị trí |
|-----|-------------|--------|
| 1 | Thuyết minh đề tài (.docx, .pptx) | `report/thuc-hanh/` |
| 2 | Báo cáo phân tích bộ dữ liệu (.docx) | `report/thuc-hanh/` |
| 3 | Bộ dữ liệu sau tiền xử lý | `reports/do-an/phase1/combined_user_metrics.csv` |
| 4 | Video thuyết trình (bật camera) | — |

**Thang điểm thực hành (10đ):**

| Tiêu chí | Điểm | Nội dung kiểm tra |
|----------|------|-------------------|
| What | 3đ | Mô tả bài toán, dữ liệu, đặc trưng |
| Why | 3đ | Lý do chọn phương pháp, phân tích EDA |
| How | 4đ | Quy trình xử lý, kết quả thực nghiệm |

---

### Đồ án môn học

> 📂 Code đồ án: **`final/`** | 📦 Sản phẩm chạy: **`reports/do-an/`** | 📄 Báo cáo & slide: **`report/do-an/`**

| STT | Hạng mục nộp | Vị trí |
|-----|-------------|--------|
| 1 | Báo cáo đồ án (.docx, .pptx) | `report/do-an/` |
| 2 | Bộ dữ liệu thực nghiệm (sinh từ pipeline experiment) | `reports/do-an/` |
| 3 | Toàn bộ source code | `final/` (hoặc link Kaggle) |
| 4 | Video thuyết trình (bật camera) | — |

**Thang điểm đồ án cuối kỳ (10đ):**

| Tiêu chí | Điểm | Nội dung kiểm tra |
|----------|------|-------------------|
| Data Quality | 4đ | Chất lượng dữ liệu, làm sạch, feature engineering |
| Machine Learning & Framework | 3đ | Lựa chọn mô hình, huấn luyện, đánh giá |
| Experiment & Demo | 3đ | Kết quả thực nghiệm, demo, trình bày |

---

## 🔬 Thực hành – Pipeline (`experiment/`)

Thư mục `experiment/` được thiết kế lại theo bài toán thực tế:

- Dự đoán **cảnh báo sớm theo từng user-course**.
- Chỉ dùng dữ liệu đến thời điểm dự đoán (chống leakage).
- Mục tiêu là cảnh báo trước khi khóa học kết thúc để can thiệp.

Tài liệu thiết kế đầy đủ: `report/thuc-hanh/thiet-ke-he-thong-canh-bao-som-user-course.md`

Pipeline 8 phase (hướng vận hành):

```
Phase 1: Data Preparation  →  Chuẩn hóa dữ liệu + timeline course + EDA an toàn bộ nhớ
Phase 2: Data Cleaning     →  Làm sạch missing/noise/outlier theo user-course
Phase 3: Transformation    →  Biến đổi đặc trưng có xét mốc thời gian dự đoán
Phase 4: Data Labeling     →  Nhãn risk + cảnh báo sớm theo tiến độ khóa học
Phase 5: Data Splitting    →  Chia tập theo thời gian/group để tránh leakage
Phase 6: Model Training    →  Huấn luyện và hiệu chỉnh risk score
Phase 7: Model Evaluation  →  Đánh giá theo metric cảnh báo sớm (ưu tiên recall nhóm nguy cơ)
Phase 8: Interpretability  →  Giải thích yếu tố rủi ro để hỗ trợ can thiệp
```

Pipeline `experiment` mặc định ghi sản phẩm vào `reports/do-an/`.

Nếu bạn muốn chạy bản v2 ngắn gọn theo yêu cầu mới, dùng:

```bash
python -m experiment.pipeline_v2.run_pipeline --config experiment/pipeline_v2/config.json
```

Tài liệu hướng dẫn riêng của v2 nằm tại `experiment/pipeline_v2/README.md`.

**Chạy pipeline:**
```bash
# Từ thư mục gốc project
python experiment/run_experiment_stages.py --phase all

# Chạy từng phase
python experiment/run_experiment_stages.py --phase 1

# Test nhanh với ít dữ liệu
python experiment/run_experiment_stages.py --phase 1 --max-rows 1000

# Nếu cần ghi sang thư mục khác
python experiment/run_experiment_stages.py --phase all --results-dir reports/thuc-hanh
```

**Kết quả được lưu mặc định tại `reports/do-an/`:**

| File | Mô tả |
|------|-------|
| `phase1/combined_user_metrics.csv` | Đặc trưng hành vi theo user-course |
| `phase1/phase1_eda_report.txt` | Báo cáo thống kê EDA + biểu đồ |
| `phase2/combined_user_metrics_clean.csv` | Dữ liệu sau làm sạch |
| `phase4/phase4_2_standard_labels_kmeans.csv` | Nhãn risk + cảnh báo sớm theo giai đoạn |
| `phase6/phase6_best_model.pkl` | Model tốt nhất |
| `phase7/final_summary_report.txt` | Báo cáo tổng hợp toàn pipeline |

---

## 📦 Đồ án cuối kỳ – (`final/`)

Thư mục `final/` chứa source code đồ án hoàn chỉnh để nộp.  
Sản phẩm chạy (model, báo cáo đánh giá, artifacts) đặt tại `reports/do-an/`.

---

## Dataset

| File | Mô tả |
|------|-------|
| `user.json` | Hồ sơ học sinh (id, trường, giới tính, năm sinh, khóa học) |
| `user-problem.json` | Lịch sử làm bài tập (đúng/sai, điểm, thời gian) |
| `user-video.json` | Phiên xem video (đoạn đã xem, tốc độ, thời lượng) |
| `reply.json` | Hoạt động trả lời forum |
| `comment.json` | Hoạt động bình luận forum |

**Tải về:** https://github.com/THU-KEG/MOOCCube  
**Kaggle:** https://www.kaggle.com/datasets/thiuyn/mooccubexdataset

Đặt dataset tại `D:/MOOCCubeX_dataset/` hoặc truyền `--dataset-dir <path>` khi chạy.

---

## Chạy trên Kaggle

Code đã được upload tại: https://www.kaggle.com/datasets/thaile2024/experiment

```python
import os, sys, subprocess

CODE_DIR    = "/kaggle/input/experiment"
DATASET_DIR = "/kaggle/input/mooccubexdataset/MOOCCubeXData/MOOCCubeXData"
RESULTS_DIR = "/kaggle/working/results"

os.makedirs(RESULTS_DIR, exist_ok=True)

result = subprocess.run([
    sys.executable, f"{CODE_DIR}/run_experiment_stages.py",
    "--phase", "all",
    "--dataset-dir", DATASET_DIR,
    "--results-dir", RESULTS_DIR,
], capture_output=True, text=True)

print(result.stdout)
if result.returncode != 0:
    print("ERROR:", result.stderr)
```

**Cập nhật code lên Kaggle (sau khi sửa local):**
```bash
kaggle datasets version -p "experiment" -m "Update: mô tả thay đổi"
```

---

## Tham số CLI

| Flag | Mặc định | Mô tả |
|------|----------|-------|
| `--phase` | `all` | Phase cần chạy: `1`–`8` hoặc `all` |
| `--dataset-dir` | `D:/MOOCCubeX_dataset` | Đường dẫn đến dataset gốc |
| `--results-dir` | `reports/do-an` | Override thư mục lưu kết quả |
| `--max-rows` | *(none)* | Giới hạn số dòng để test nhanh |
| `--seed` | `42` | Random seed để tái lập kết quả |
