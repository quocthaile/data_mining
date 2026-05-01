# 📖 Index - Thiết Kế Lại Quy Trình Chuẩn Bị Dữ Liệu Modular

## 🎯 Tóm Tắt Nhanh

Quy trình chuẩn bị dữ liệu được thiết kế lại từ **monolithic** sang **modular architecture**:

- **4 bước** độc lập, mỗi bước có input/output rõ ràng
- **Bỏ qua tái xử lý** nếu output đã tồn tại
- **Logging chi tiết** để tracking data flow
- **Dễ debug & mở rộng** cho các task mới

---

## 📁 Cấu Trúc Tệp Tài Liệu

```
experiment/
├── 📄 README_MODULAR_PIPELINE.md          [START HERE] Quick start
├── 📄 PIPELINE_DESIGN.md                  Chi tiết mỗi bước
├── 📄 ARCHITECTURE_DIAGRAMS.md            Diagrams & flows
├── 📄 COMPARISON_OLD_vs_NEW.md            So sánh cũ vs mới
├── 📄 THIS FILE (INDEX.md)                Điều hướng
│
├── 🐍 data_prepare_modular.py             [MAIN] Quy trình mới
├── 🐍 examples_usage.py                   8 kịch bản sử dụng
├── 🐍 data_prepare.py                     (Cũ) Giữ lại reference
│
└── 📊 ... other files
```

---

## 🚀 Bắt Đầu Nhanh (Quick Start)

### **1️⃣ Đọc cái gì trước?**

→ **[README_MODULAR_PIPELINE.md](README_MODULAR_PIPELINE.md)** (5 phút)

### **2️⃣ Code chính nằm ở đâu?**

→ **[data_prepare_modular.py](data_prepare_modular.py)** 

### **3️⃣ Cách sử dụng?**

```python
from data_prepare_modular import DataPipelineOrchestrator

# Chạy toàn bộ
orchestrator = DataPipelineOrchestrator()
orchestrator.run_all()

# Hoặc chạy từng bước
orchestrator.run_step1()
orchestrator.run_step2()
```

### **4️⃣ Ví dụ chi tiết?**

→ **[examples_usage.py](examples_usage.py)** (8 kịch bản)

---

## 📚 Hướng Dẫn Đọc Tài Liệu

### **Nếu bạn muốn...**

| Muốn... | Đọc tài liệu này | Thời gian |
|--------|-----------------|----------|
| 🎯 Bắt đầu nhanh | [README_MODULAR_PIPELINE.md](README_MODULAR_PIPELINE.md) | 5 phút |
| 🔍 Hiểu chi tiết 4 bước | [PIPELINE_DESIGN.md](PIPELINE_DESIGN.md) | 15 phút |
| 📊 Xem diagrams | [ARCHITECTURE_DIAGRAMS.md](ARCHITECTURE_DIAGRAMS.md) | 10 phút |
| 🔄 So sánh cũ vs mới | [COMPARISON_OLD_vs_NEW.md](COMPARISON_OLD_vs_NEW.md) | 10 phút |
| 💡 Xem ví dụ | [examples_usage.py](examples_usage.py) | 15 phút |
| 📖 Đọc code | [data_prepare_modular.py](data_prepare_modular.py) | 30 phút |

---

## 🏗️ Cấu Trúc 4 Bước

```
┌─────────────────────────────────────────────────────────┐
│ STEP 1: Timeline Discovery                             │
│ Input:  user-video.json                                │
│ Output: step1_timelines.json                           │
└─────────────────────────────────────────────────────────┘
                        ↓
┌─────────────────────────────────────────────────────────┐
│ STEP 2: Feature Extraction                             │
│ Input:  user-video.json + step1_timelines.json         │
│ Output: ews_features.db (SQLite)                       │
└─────────────────────────────────────────────────────────┘
                        ↓
┌─────────────────────────────────────────────────────────┐
│ STEP 3: EDA & Clustering                               │
│ Input:  ews_features.db                                │
│ Output: step3_*.csv (correlation, labels, pca info)   │
└─────────────────────────────────────────────────────────┘
                        ↓
┌─────────────────────────────────────────────────────────┐
│ STEP 4: Labeling & Splitting                           │
│ Input:  ews_features.db + step1_timelines.json +       │
│         step3_kmeans_labels.csv                         │
│ Output: step4_train/test_50pct.csv                     │
│         step4_train/test_75pct.csv                     │
└─────────────────────────────────────────────────────────┘
```

---

## 📖 Chi Tiết Từng Tài Liệu

### 📄 **README_MODULAR_PIPELINE.md**

**Nội dung:**
- Mục đích thiết kế
- 4 bước tóm tắt
- Cách sử dụng nhanh
- Bảng so sánh cũ vs mới
- Troubleshooting

**Khi nào đọc:** Lần đầu / Quick reference

---

### 📄 **PIPELINE_DESIGN.md**

**Nội dung:**
- Tổng quan hệ thống
- Chi tiết mỗi bước (input/output)
- Schema dữ liệu
- Ví dụ output files
- Kịch bản advanced
- Ghi chú & tuning

**Khi nào đọc:** Khi cần hiểu chi tiết từng bước

---

### 📄 **ARCHITECTURE_DIAGRAMS.md**

**Nội dung:**
- Data Flow Diagram
- Architecture (class relationships)
- Data Schema Evolution
- Dependency Graph
- Execution Flow
- Caching & Reusability
- Error Handling

**Khi nào đọc:** Khi cần visualize quy trình

---

### 📄 **COMPARISON_OLD_vs_NEW.md**

**Nội dung:**
- Bảng so sánh (20+ tiêu chí)
- Code structure comparison
- Performance analysis
- Migration guide
- Khi nào dùng cái nào

**Khi nào đọc:** Nếu từng dùng code cũ

---

### 🐍 **data_prepare_modular.py**

**Nội dung:**
- 4 Step classes (Step1-4)
- DataPipelineConfig
- StepLogger
- DataPipelineOrchestrator
- Main entry point

**Khi nào dùng:** 
- Import để sử dụng
- Đọc docstring mỗi class
- Hiểu implementation details

---

### 🐍 **examples_usage.py**

**Nội dung:**
- 8 kịch bản sử dụng
  1. Chạy toàn bộ lần đầu
  2. Chạy từng bước riêng
  3. Chạy 1 bước cụ thể
  4. Tái xử lý với force=True
  5. Xử lý lại từ Step N
  6. Reset & chạy từ đầu
  7. Chỉ extraction (Step 1-2)
  8. Kiểm tra input/output

**Khi nào dùng:** 
- Chọn kịch bản phù hợp
- Copy code & sử dụng
- Hiểu các use case

---

## 🎯 Quy Trình Học Tập

### **Bước 1: Hiểu Tổng Quan (5 phút)**
→ Đọc [README_MODULAR_PIPELINE.md](README_MODULAR_PIPELINE.md)

### **Bước 2: Xem Sơ Đồ (10 phút)**
→ Xem [ARCHITECTURE_DIAGRAMS.md](ARCHITECTURE_DIAGRAMS.md) → Data Flow Diagram

### **Bước 3: Chạy Ví Dụ (10 phút)**
```python
# Ví dụ nhanh nhất
from data_prepare_modular import DataPipelineOrchestrator

orchestrator = DataPipelineOrchestrator()
orchestrator.run_all()
```

### **Bước 4: Đọc Chi Tiết (15 phút)**
→ [PIPELINE_DESIGN.md](PIPELINE_DESIGN.md) → mục "Sử dụng" & "Advanced Usage"

### **Bước 5: Tìm Kịch Bản của Bạn (5 phút)**
→ [examples_usage.py](examples_usage.py) → chọn scenario tương ứng

### **Bước 6: Sửa & Chạy Code**
→ Modify [data_prepare_modular.py](data_prepare_modular.py) hoặc tạo wrapper

---

## 🔍 Quick Reference

### **Import**
```python
from data_prepare_modular import (
    DataPipelineOrchestrator,
    Step1_TimelineDiscovery,
    Step2_FeatureExtraction,
    Step3_EDAAndClustering,
    Step4_LabelingAndSplitting,
    DataPipelineConfig
)
```

### **Chạy Toàn Bộ**
```python
orchestrator = DataPipelineOrchestrator()
results = orchestrator.run_all()
```

### **Chạy Từng Bước**
```python
orchestrator.run_step1()  # → timelines.json
orchestrator.run_step2()  # → ews_features.db
orchestrator.run_step3()  # → *.csv files
orchestrator.run_step4()  # → train/test files
```

### **Chạy 1 Bước Cụ thể**
```python
config = DataPipelineConfig()
step3 = Step3_EDAAndClustering(config)
outputs = step3.execute()
```

### **Tái Xử Lý (Skip Cache)**
```python
orchestrator = DataPipelineOrchestrator(force=True)
orchestrator.run_step3()
```

---

## 📊 Input/Output Summary

### **Input Files Cần Có**
```
./dataset/
├── user-video.json    (Bắt buộc)
└── user.json          (Optional)
```

### **Output Files Được Tạo**
```
./ews_output/
├── step1_timelines.json
├── step3_correlation_matrix.csv
├── step3_kmeans_labels.csv
├── step3_pca_info.json
├── step4_train_50pct.csv
├── step4_test_50pct.csv
├── step4_train_75pct.csv
└── step4_test_75pct.csv

./
└── ews_master.db    (SQLite)
```

---

## ❓ FAQ & Troubleshooting

### **Q: Code cũ (data_prepare.py) còn dùng được không?**
A: Vẫn được, nhưng deprecated. Chuyển sang data_prepare_modular.py

### **Q: Muốn chỉ chạy Step 2 được không?**
A: Có, nhưng Step 1 phải chạy trước (hoặc output của Step 1 phải tồn tại)

### **Q: Làm sao để tái xử lý Step 3?**
A: 
```python
step3 = Step3_EDAAndClustering(config, force=True)
step3.execute()
```

### **Q: Performance như thế nào?**
A: 
- Lần đầu: ~6 phút (toàn bộ processing)
- Lần sau: ~0.1 giây (cache, all steps SKIP)
- Tái xử lý 1 step: ~30s + step phụ thuộc

### **Q: Nên dùng file nào làm starting point?**
A: `data_prepare_modular.py` - đây là file chính!

### **Q: Làm sao để customize K-Means clusters?**
A: Edit trong `Step3_EDAAndClustering.execute()` → `KMeans(n_clusters=...)`

---

## 🔗 Cross-References

| Nếu muốn... | Tìm ở... |
|------------|---------|
| Chạy code | [data_prepare_modular.py](data_prepare_modular.py) |
| Hiểu logic | [PIPELINE_DESIGN.md](PIPELINE_DESIGN.md) |
| Xem flow | [ARCHITECTURE_DIAGRAMS.md](ARCHITECTURE_DIAGRAMS.md) |
| Copy paste | [examples_usage.py](examples_usage.py) |
| So sánh cũ-mới | [COMPARISON_OLD_vs_NEW.md](COMPARISON_OLD_vs_NEW.md) |
| Quick start | [README_MODULAR_PIPELINE.md](README_MODULAR_PIPELINE.md) |

---

## 🎓 Learning Path Recommendation

### **Người Mới (30 phút)**
1. Đọc [README_MODULAR_PIPELINE.md](README_MODULAR_PIPELINE.md)
2. Xem Data Flow Diagram trong [ARCHITECTURE_DIAGRAMS.md](ARCHITECTURE_DIAGRAMS.md)
3. Chạy scenario 1 từ [examples_usage.py](examples_usage.py)
4. Done! 🎉

### **Người Intermediate (1 giờ)**
1. Làm Người Mới (30 phút)
2. Đọc [PIPELINE_DESIGN.md](PIPELINE_DESIGN.md) - mục "Sử dụng"
3. Thử scenarios 2-5 từ [examples_usage.py](examples_usage.py)
4. Sửa custom config trong [data_prepare_modular.py](data_prepare_modular.py)

### **Người Advanced (2 giờ)**
1. Làm Người Intermediate (1 giờ)
2. Đọc [PIPELINE_DESIGN.md](PIPELINE_DESIGN.md) - hết
3. Đọc toàn bộ [data_prepare_modular.py](data_prepare_modular.py)
4. Tạo custom Step class mới
5. Thêm vào Orchestrator

---

## 📋 Checklist Trước Chạy

- [ ] Đã đọc [README_MODULAR_PIPELINE.md](README_MODULAR_PIPELINE.md)?
- [ ] Input files (`user-video.json`) tồn tại?
- [ ] Quyền ghi directory?
- [ ] Python packages cài đặt? (pandas, numpy, sklearn)
- [ ] Chọn kịch bản từ [examples_usage.py](examples_usage.py)?
- [ ] Ready to run! 🚀

---

## 📞 Support

Gặp vấn đề? Kiểm tra:
1. [README_MODULAR_PIPELINE.md](README_MODULAR_PIPELINE.md) → Troubleshooting
2. [PIPELINE_DESIGN.md](PIPELINE_DESIGN.md) → ghi chú
3. [examples_usage.py](examples_usage.py) → kịch bản tương tự

---

**Version:** 2.0 (Modular Architecture)  
**Status:** ✓ Production Ready  
**Last Updated:** 2026-04-28

---

## 🚀 Tiếp Theo?

👉 **Start here:** [README_MODULAR_PIPELINE.md](README_MODULAR_PIPELINE.md)
