# Source Code - Đồ Án Khai Phá Dữ Liệu DS317

## Tổng quan

Thư mục này chứa toàn bộ source code cho đồ án "Dự đoán mức độ tham gia của học viên trong hệ thống MOOC sử dụng kỹ thuật khai phá dữ liệu".

## Cấu trúc

```
source-code/
├── main_experiment.py      # Script chính chạy toàn bộ pipeline thực nghiệm
├── demo_app.py            # Ứng dụng web demo dự đoán
├── check_environment.py   # Script kiểm tra môi trường và chạy test nhanh
└── README.md              # Tài liệu hướng dẫn (file này)
```

## Yêu cầu hệ thống

- Python 3.8+
- Các thư viện: pandas, scikit-learn, flask, joblib, numpy, matplotlib, seaborn
- Bộ dữ liệu MOOC từ thư mục `../du-lieu-thuc-nghiem/`

## Cài đặt

1. Cài đặt các thư viện cần thiết:
```bash
pip install pandas numpy scikit-learn matplotlib seaborn flask joblib xgboost lightgbm
```

2. Đảm bảo có bộ dữ liệu trong thư mục `../du-lieu-thuc-nghiem/`

## Sử dụng

### 0. Kiểm tra môi trường

Trước khi chạy, kiểm tra môi trường:

```bash
python check_environment.py
```

Script này sẽ:
- Kiểm tra các thư viện Python cần thiết
- Kiểm tra file dữ liệu
- Kiểm tra script cần thiết
- Chạy test nhanh với dữ liệu nhỏ

### 1. Chạy toàn bộ pipeline thực nghiệm

```bash
python main_experiment.py --phase all
```

Các tùy chọn:
- `--phase`: Chọn phase cụ thể (2, 4, 5, 6, 7, 8) hoặc "all" để chạy tất cả
- `--dataset-dir`: Đường dẫn đến thư mục dữ liệu
- `--results-dir`: Đường dẫn lưu kết quả
- `--max-rows`: Giới hạn số hàng để test (tùy chọn)

Ví dụ:
```bash
# Chạy phase 2 và 4
python main_experiment.py --phase 2 --phase 4

# Chạy với giới hạn dữ liệu
python main_experiment.py --phase all --max-rows 10000
```

### 2. Chạy ứng dụng demo

Sau khi có mô hình đã train (từ phase 6):

```bash
python demo_app.py
```

Truy cập: http://localhost:5000

Ứng dụng demo cho phép:
- Nhập các đặc trưng của học viên
- Dự đoán mức độ tham gia (Low/Medium/High)
- Xem xác suất cho từng lớp

## Kịch bản thực nghiệm

Pipeline thực nghiệm gồm 8 phases (theo thứ tự chuẩn hóa luồng dữ liệu):

1. **Phase 1**: Làm sạch và chuẩn hóa dữ liệu nguồn
2. **Phase 2**: Trích xuất đặc trưng chuỗi thời gian từ dữ liệu hoạt động
3. **Phase 3**: Khám phá và phân tích dữ liệu (EDA)
4. **Phase 4**: Khởi tạo nhãn ground-truth bằng K-Means và xác thực nhãn
5. **Phase 5**: Chia tập train/valid/test và xử lý mất cân bằng dữ liệu
6. **Phase 6**: Huấn luyện mô hình supervised với tối ưu hóa
7. **Phase 7**: Đánh giá độ đo và báo cáo kết quả
8. **Phase 8**: Phân tích khả năng giải thích mô hình

## Kết quả đầu ra

Sau khi chạy thành công, các file kết quả sẽ được lưu trong thư mục `results/`:

- `best_model.pkl`: Mô hình tốt nhất
- `feature_columns.pkl`: Danh sách cột đặc trưng
- `evaluation_report.html`: Báo cáo đánh giá
- `interpretability_report.html`: Báo cáo giải thích mô hình
- Các file CSV với dữ liệu đã xử lý

## Lưu ý

- Đảm bảo đường dẫn đến bộ dữ liệu chính xác
- Pipeline có thể mất thời gian tùy thuộc vào kích thước dữ liệu
- Sử dụng `--max-rows` để test với tập dữ liệu nhỏ
- Mô hình cần được train trước khi chạy demo app

## Troubleshooting

1. **Lỗi không tìm thấy file dữ liệu**: Kiểm tra đường dẫn `--dataset-dir`
2. **Lỗi memory**: Giảm `--max-rows` hoặc tăng RAM
3. **Demo app không load mô hình**: Đảm bảo đã chạy phase 6 thành công

## Liên hệ

Đồ án môn DS317 - Khai phá dữ liệu trong doanh nghiệp
Trường Đại học Công nghệ Thông tin - ĐHQG.HCM