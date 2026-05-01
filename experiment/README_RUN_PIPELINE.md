# Hướng dẫn chạy pipeline thực nghiệm

Tài liệu này mô tả cách chạy pipeline thực nghiệm trong thư mục `experiment/`.

## 1. Cấu trúc chạy

Pipeline có 5 stage chính:

1. `stage_1_generate_ground_truth.py`
2. `stage_2_time_window_features.py`
3. `stage_3_split_and_smote.py`
4. `stage_4_model_training_eval.py`
5. `stage_5_explain_model_xai.py`

Điều phối tập trung được đặt tại `run_pipeline.py`.

## 2. File đầu ra chính

Các dataset đầu ra được lưu tại thư mục `dataset/`:

- `dataset/ground_truth_labels.csv`
- `dataset/ground_truth_report.csv`
- `dataset/user_features_28days.csv`
- `dataset/user_features_and_wes.csv`
- `dataset/experimental_dataset.csv`
- `dataset/model_data/train_smote.csv`
- `dataset/model_data/valid_original.csv`
- `dataset/model_data/test_original.csv`
- `dataset/pre-processing_dataset.csv`

Các file mô hình và hình ảnh được lưu trong:

- `experiment/deployment_models/`
- `experiment/output_images_3w/`

## 3. Cách chạy nhanh

### Chạy toàn bộ pipeline

```powershell
cd "d:\drive\UIT\HK4\DS317 - Khai pha du lieu\project\experiment"
python run_pipeline.py
```

### Chạy bằng menu tương tác

```powershell
python run_pipeline.py --menu
```

Trong menu, bạn có thể:

- chạy từng stage riêng lẻ
- chạy một đoạn stage theo range
- đặt tham số override

Ví dụ:

- `TRAIN_TARGET_TOTAL_SAMPLES=60000`
- `LABEL_PERCENTILES=[0.6,0.85]`

### Chạy một khoảng stage

```powershell
python run_pipeline.py --from-step 1 --to-step 3
```

### Truyền tham số runtime

```powershell
python run_pipeline.py --param TRAIN_TARGET_TOTAL_SAMPLES=60000 --param LABEL_PERCENTILES='[0.6,0.85]'
```

## 4. Ghi chú quan trọng

- Stage 1 hiện dùng phân phối nhãn tự nhiên hơn, không còn ép 1:1:1.
- Stage 3 chỉ fit preprocessing trên train để tránh leakage.
- `experimental_dataset.csv` được xuất trước khi chia tập.
- `pre-processing_dataset.csv` là bộ dữ liệu sau tiền xử lý và được lưu trong `dataset/`.

## Time windows (định nghĩa và lựa chọn)

Time windows là cách cắt dữ liệu theo khoảng thời gian để trích xuất đặc trưng chuỗi thời gian. Pipeline hỗ trợ hai cách xác định cửa sổ thời gian:

1. **Cửa sổ thời gian cố định (fixed window)**: ví dụ 14, 21, 28 ngày từ ngày bắt đầu khóa. Đây là lựa chọn tốt khi bạn muốn so sánh các hành vi trong cùng một khoảng thời gian tuyệt đối kể từ lúc đăng ký hoặc bắt đầu khóa.

2. **Cửa sổ theo phần trăm tiến độ khóa (relative percentiles)**: ví dụ 25% và 50% của toàn bộ thời lượng khóa (course duration). Thay vì dùng số ngày cố định, bạn cắt theo tỉ lệ tiến độ của mỗi học viên trong course. Điều này hữu ích khi khóa có độ dài khác nhau giữa các học viên hoặc giữa các course.

So sánh ngắn gọn:
- Fixed window: dễ triển khai, ổn định khi course có cấu trúc thời gian giống nhau; nhưng có thể bỏ lỡ hành vi sớm/ trễ nếu course dài khác nhau.
- Relative percentiles: thích ứng với các khóa có độ dài khác nhau, phản ánh hành vi theo tiến độ; nhưng cần xác định rõ cách tính độ dài khóa (ví dụ ngày đầu và ngày cuối hành động thực tế).

Trong README này, `DEFAULT_OBSERVATION_DAYS` mặc định là 28 (fixed).

Relative windows đã được triển khai trong Stage 2. Cách chạy so sánh:

```powershell
python run_pipeline.py --from-step 2 --to-step 2 --param TIME_WINDOW_MODE="relative"
```

Kết quả so sánh sẽ được lưu:

- `dataset/user_features_relative_25.csv`
- `dataset/user_features_relative_50.csv`
- `dataset/time_window_comparison.csv`

Khi chạy relative, pipeline sẽ ghi `dataset/user_features_and_wes.csv` theo mốc 50% để có thể chạy tiếp Stage 3 nếu cần.

### Logs và lệnh đang xử lý

- Pipeline không in icon trong log; mọi thông báo đều ở dạng chữ thuần.
- `run_pipeline.py` in ra **mỗi lệnh subprocess** trước khi thực hiện, ví dụ:

```
Executing command: C:\Python39\python.exe experiment\stage_1_generate_ground_truth.py
```

- Các stage cũng ghi log từng bước chính (tải dữ liệu, trích xuất features, split, sampling, lưu file...).

## 5. Kiểm tra lỗi nhanh

Nếu cần kiểm tra cú pháp các file Python:

```powershell
python -c "import py_compile, glob; [py_compile.compile(f, doraise=True) for f in glob.glob('*.py')]"
```
