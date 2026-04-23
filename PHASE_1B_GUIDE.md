# Phase 1B: EDA & Data Cleaning

## Mô tả

**Phase 1B** là bước tùy chọn trong pipeline để thực hiện Phân tích Khám phá Dữ liệu (EDA) và Làm sạch dữ liệu trước khi huấn luyện mô hình.

### Chức năng chính:

1. **Exploratory Data Analysis (EDA)**
   - Thống kê mô tả (descriptive statistics)
   - Phát hiện outliers bằng phương pháp IQR
   - Phân tích phần trăm dữ liệu thiếu

2. **Data Cleaning (Làm sạch dữ liệu)**
   - Xóa hàng có quá nhiều giá trị thiếu (>30%)
   - Loại bỏ giá trị không hợp lệ (NaN, Inf)
   - Xuất dữ liệu đã làm sạch

3. **Data Normalization (Chuẩn hóa dữ liệu)**
   - Chuẩn hóa `engagement_events` vào khoảng [0, 1]
   - Lưu thông số chuẩn hóa để sử dụng sau

## Cách chạy

### Chạy Phase 1B một mình:
```bash
cd project/experiment
python run_experiment_stages.py --phase 1b
```

### Chạy toàn bộ pipeline (Phase 1 -> 1B -> 2 -> ...):
```bash
cd project/experiment
python run_experiment_stages.py --phase all
```

### Chạy Phase 1B với dữ liệu sample:
```bash
cd project/experiment
python run_experiment_stages.py --phase 1b --max-rows 5000
```

### Gọi trực tiếp script:
```bash
cd project/experiment
python phase_1b_eda.py \
  --combined-csv ../results/combined_user_metrics.csv \
  --output-dir ../results \
  --missing-threshold 0.3 \
  --outlier-iqr-multiplier 1.5
```

## Đầu vào (Input)

| Tham số | Mô tả | Giá trị mặc định |
|--------|-------|-----------------|
| `--combined-csv` | File CSV từ Phase 1 (combined_user_metrics.csv) | `results/combined_user_metrics.csv` |
| `--output-dir` | Thư mục lưu kết quả | `results` |
| `--missing-threshold` | Ngưỡng xóa hàng (tỉ lệ missing > ngưỡng) | `0.3` (30%) |
| `--outlier-iqr-multiplier` | Hệ số IQR cho phát hiện outlier | `1.5` |
| `--max-rows` | Giới hạn dòng để test | `None` (không giới hạn) |

## Đầu ra (Output)

| File | Mô tả |
|------|-------|
| `combined_user_metrics_clean.csv` | Dữ liệu đã làm sạch |
| `engagement_events_normalized.csv` | Cột `engagement_events` đã chuẩn hóa |
| `phase1b_eda_report.txt` | Báo cáo EDA chi tiết |

## Báo cáo (phase1b_eda_report.txt)

Báo cáo bao gồm 6 phần chính:

### 1. Data Overview
- Số dòng trước/sau khi làm sạch
- Số dòng bị xóa (thiếu dữ liệu, giá trị không hợp lệ)
- Tỉ lệ giữ lại (retention rate)

### 2. Descriptive Statistics (Trước làm sạch)
- Count, missing, mean, std, min, Q25, median, Q75, max
- Cho tất cả các cột số

### 3. Descriptive Statistics (Sau làm sạch)
- Cùng các chỉ số như trên

### 4. Outlier Detection (IQR Method)
- Số lượng outlier + phần trăm
- Lower bound và upper bound cho mỗi cột
- Cảnh báo nếu outlier >5%

### 5. Engagement Events Normalization
- Min/max/mean/std của giá trị raw
- Công thức: `(E - min) / (max - min)`

### 6. Data Quality Summary
- Tóm tắt các bước làm sạch
- Kiểm tra chất lượng dữ liệu

## Ví dụ

```bash
# Chạy Phase 1B sau khi Phase 1 hoàn thành
cd project/experiment

# Test với 10000 dòng đầu
python run_experiment_stages.py --phase 1b --max-rows 10000

# Chạy trên toàn bộ dữ liệu
python run_experiment_stages.py --phase 1b
```

## Kiểm tra kết quả

1. **Xem báo cáo:**
   ```bash
   cat ../results/phase1b_eda_report.txt
   ```

2. **So sánh dữ liệu:**
   - Dòng cũ: `combined_user_metrics.csv`
   - Dòng sạch: `combined_user_metrics_clean.csv`

3. **Xem chuẩn hóa engagement_events:**
   ```bash
   head -20 ../results/engagement_events_normalized.csv
   ```

## Tiếp theo

Phase 1B xuất dữ liệu sạch → dùng cho **Phase 2** (K-Means clustering).

Cập nhật Phase 2 để sử dụng `combined_user_metrics_clean.csv` thay vì `combined_user_metrics.csv` nếu muốn.
