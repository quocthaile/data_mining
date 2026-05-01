# Thiết kế lại Quy trình Chuẩn bị Dữ liệu - Modular Architecture

## 🎯 Mục đích

Thiết kế lại quy trình chuẩn bị dữ liệu từ cấu trúc **monolithic** sang **modular**, cho phép:
- ✅ **Xử lý độc lập từng bước** - mỗi bước không phụ thuộc việc thực thi toàn bộ
- ✅ **Input/Output rõ ràng** - dễ dàng track data flow
- ✅ **Bỏ qua tái xử lý** - nếu output đã tồn tại, bước sẽ bỏ qua
- ✅ **Dễ debug & maintain** - mỗi bước có logging chi tiết
- ✅ **Mở rộng dễ dàng** - thêm bước mới không ảnh hưởng các bước cũ

---

## 📁 Cấu trúc File

```
experiment/
├── data_prepare_modular.py          # ⭐ Chính: Quy trình modular
├── PIPELINE_DESIGN.md               # Tài liệu thiết kế chi tiết
├── examples_usage.py                # 8 kịch bản sử dụng
└── data_prepare.py                  # (Cũ) Quy trình gốc - giữ lại để reference
```

---

## 🏗️ Kiến trúc 4 Bước

### Step 1: Timeline Discovery
```
Khám phá mốc thời gian (T_start, T_end, M_50%, M_75%) từ dữ liệu

Input:  user-video.json
Output: step1_timelines.json
```

### Step 2: Feature Extraction
```
Trích xuất đặc trưng streaming cho 3 mốc (50%, 75%, 100%)

Input:  user-video.json + step1_timelines.json
Output: ews_features.db (SQLite)
```

### Step 3: EDA & Clustering
```
Phân tích thống kê, PCA giảm chiều, K-Means clustering, gán nhãn

Input:  ews_features.db
Output: step3_correlation_matrix.csv + step3_kmeans_labels.csv + step3_pca_info.json
```

### Step 4: Labeling & Splitting
```
Gán nhãn, tính absence_days, chia train/test per milestone

Input:  ews_features.db + step1_timelines.json + step3_kmeans_labels.csv
Output: step4_train/test_50pct.csv + step4_train/test_75pct.csv
```

---

## 🚀 Cách sử dụng nhanh

### **1. Chạy toàn bộ quy trình**

```python
from data_prepare_modular import DataPipelineOrchestrator

orchestrator = DataPipelineOrchestrator(
    data_dir='./dataset',
    out_dir='./ews_output',
    db_path='ews_master.db'
)

results = orchestrator.run_all()
```

### **2. Chạy từng bước riêng biệt**

```python
orchestrator = DataPipelineOrchestrator()

# Chạy độc lập
orchestrator.run_step1()  # → step1_timelines.json
orchestrator.run_step2()  # → ews_features.db
orchestrator.run_step3()  # → step3_*.csv
orchestrator.run_step4()  # → step4_*.csv
```

### **3. Chạy chỉ một bước cụ thể**

```python
from data_prepare_modular import Step3_EDAAndClustering, DataPipelineConfig

config = DataPipelineConfig()
step3 = Step3_EDAAndClustering(config, force=False)
outputs = step3.execute()
```

### **4. Tái xử lý một bước (bỏ qua cache)**

```python
step3 = Step3_EDAAndClustering(config, force=True)
outputs = step3.execute()
```

---

## 📊 So sánh: Cũ vs Mới

| Tiêu chí | Cũ | Mới |
|----------|----|----|
| Xử lý toàn bộ | ✓ | ✓ |
| Xử lý từng bước | ✗ | ✓ |
| Input/Output rõ ràng | ✗ | ✓ |
| Bỏ qua tái xử lý | ✗ | ✓ |
| Logging chi tiết | ⚠️ | ✓ |
| Dễ debug | ⚠️ | ✓ |
| Dễ mở rộng | ⚠️ | ✓ |
| Code reusability | ⚠️ | ✓ |

---

## 📝 Đặc trưng trích xuất

Mỗi bản ghi chứa:
- **user_id**: ID người dùng
- **course_id**: ID khóa học
- **milestone**: Mốc thời gian (50%, 75%, 100%)
- **video_count**: Số lần xem video
- **watched_seconds**: Tổng thời gian xem (giây)
- **pause_rewind_freq**: Tần suất pause/rewind
- **lag_time_days**: Ngày tính từ T_start
- **absence_days** (Step 4 only): Ngày không hoạt động tính từ mốc

---

## 🔧 Tùy chỉnh

### Thay đổi số clusters (K-Means)
```python
# Step 3: Tìm dòng này
kmeans = KMeans(n_clusters=3, ...)
# Sửa K=3 thành K mong muốn
kmeans = KMeans(n_clusters=4, ...)
```

### Thay đổi test size (Train/Test split)
```python
# Step 4: Tìm dòng này
gss = GroupShuffleSplit(n_splits=1, test_size=0.2, ...)
# Sửa thành test_size mong muốn
gss = GroupShuffleSplit(n_splits=1, test_size=0.3, ...)
```

### Thay đổi PCA variance threshold
```python
# Step 3: Tìm dòng này
pca = PCA(n_components=0.95)
# Sửa thành variance threshold mong muốn
pca = PCA(n_components=0.99)
```

---

## 📚 Tài liệu Bổ sung

- **[PIPELINE_DESIGN.md](PIPELINE_DESIGN.md)** - Thiết kế chi tiết mỗi bước
- **[examples_usage.py](examples_usage.py)** - 8 kịch bản sử dụng thực tế

---

## 🎨 Ưu điểm của Thiết kế Modular

### 1. **Linh hoạt trong quá trình phát triển**
```python
# Có thể test Step 3 riêng mà không cần chạy Step 1-2 lại
step3.execute()
```

### 2. **Xử lý lại chọn lọc**
```python
# Dữ liệu input thay đổi → xóa output cũ → chạy lại từ đó
# Không cần reset tất cả
```

### 3. **Logging rõ ràng & Debugging dễ**
```
✓ Mỗi bước in input/output files
✓ Execution time & record count
✓ Dễ theo dõi data flow
```

### 4. **Tái sử dụng code**
```python
# Có thể import từng Step class riêng biệt cho các task khác
from data_prepare_modular import Step2_FeatureExtraction
```

### 5. **Mở rộng dễ dàng**
```python
# Thêm Step 5 chỉ cần tạo class mới, không ảnh hưởng Step 1-4
class Step5_NewAnalysis:
    def __init__(self, config):
        ...
```

---

## ✅ Checklist Trước Chạy

- [ ] Dữ liệu `user-video.json` tồn tại ở `./dataset/`
- [ ] Dữ liệu `user.json` tồn tại ở `./dataset/`
- [ ] Có quyền ghi thư mục hiện tại (tạo `./ews_output/`)
- [ ] Thư viện required: `pandas`, `numpy`, `sklearn`, `sqlite3`

---

## 🐛 Troubleshooting

### **"Không tìm thấy file user-video.json"**
→ Kiểm tra `data_dir` path, đảm bảo file tồn tại

### **"Không có dữ liệu ở milestone 100%"**
→ Có thể dữ liệu input không đủ hoặc Step 2 chưa chạy

### **Output file mình tạo không bị ghi đè**
→ Set `force=True` khi khởi tạo Step

### **Muốn xem log chi tiết hơn**
→ Mỗi Step có `StepLogger` tự động in progress

---

## 📞 Liên Hệ / Hỗ Trợ

Nếu gặp vấn đề:
1. Kiểm tra [PIPELINE_DESIGN.md](PIPELINE_DESIGN.md)
2. Xem ví dụ tương ứng trong [examples_usage.py](examples_usage.py)
3. Kiểm tra log output của Step

---

**Last Updated**: 2026-04-28  
**Version**: 2.0 (Modular Architecture)  
**Status**: ✓ Production Ready
