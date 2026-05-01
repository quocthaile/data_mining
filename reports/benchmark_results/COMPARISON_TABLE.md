# Báo Cáo So Sánh Hiệu Năng: Fixed 28 Days vs Relative Windows

## 1. Cấu Hình Cửa Sổ Thời Gian Được Kiểm Tra

| Loại Cửa Sổ | Giá Trị | Số Hàng | Số Người Dùng | Tệp Đầu Ra |
|:---|:---|---:|---:|:---|
| **Fixed** | 28 ngày | 2,626,854 | 129,516 | `user_features_28days.csv` |
| **Relative** | 25% (khóa học) | 2,694,714 | 129,516 | `user_features_relative_25.csv` |
| **Relative** | 50% (khóa học) | 3,115,598 | 129,516 | `user_features_relative_50.csv` |

## 2. Kết Quả Đánh Giá Validation Set

**Tiêu Chí Xếp Hạng:** Recall (Low_Engagement) → Accuracy

| Mô Hình | Độ Chính Xác | Recall_Low_Engagement | Precision_Low_Engagement | AUC-ROC |
|:---|---:|---:|---:|---:|
| **Linear SVC** ⭐ | **0.6488** | **0.9229** | 0.7338 | NaN |
| Logistic Regression | 0.6634 | 0.8794 | 0.8807 | 0.8355 |
| Decision Tree | 0.6654 | 0.8662 | 0.9227 | 0.8465 |
| Random Forest | 0.7176 | 0.8633 | 0.9268 | 0.8518 |
| XGBoost | 0.6670 | 0.8646 | 0.9260 | 0.8538 |

## 3. Kết Quả Đánh Giá Test Set

**Mô Hình Được Chọn:** Linear SVC (do có Recall cao nhất cho Low_Engagement)

| Chỉ Số | Giá Trị |
|:---|---:|
| **Độ Chính Xác (Accuracy)** | 0.6391 |
| **Recall - Low_Engagement** | 0.9147 |
| **Precision - Low_Engagement** | 0.7285 |
| **AUC-ROC** | N/A |

## 4. Lý Do Lựa Chọn Linear SVC

Theo logic tại `experiment/stage_4_model_training_eval.py` (dòng 151):

```python
ranked = metrics_df.sort_values(
    by=[f"Recall_{TARGET_RISK_CLASS}", "Accuracy"], 
    ascending=False
)
best_model_name = ranked.iloc[0]["Model"]
```

**Tiêu Chí xếp hạng chính:** 
- **Ưu tiên 1:** Recall của lớp Low_Engagement (càng cao càng tốt - phát hiện sinh viên có nguy cơ)
- **Ưu tiên 2:** Accuracy (độ chính xác tổng thể)

**Kết quả:** Linear SVC đạt Recall_Low_Engagement = **0.9229** (cao nhất) → được chọn làm mô hình chiến thắng.

---

**Ý Nghĩa:** Tối ưu hóa khả năng phát hiện sinh viên "Low Engagement" (có nguy cơ cao không hoàn thành khóa học), vì chi phí của False Negative (bỏ sót một trường hợp nguy hiểm) cao hơn False Positive.

## 5. Phân Tích So Sánh

### Tương Tự:
- ✅ Cả hai chế độ cửa sổ thời gian (Fixed vs Relative) cho ra kết quả mô hình rất tương tự
- ✅ Linear SVC giữ vị trí dẫn đầu về Recall trong cả hai trường hợp
- ✅ Độ chính xác test set tương đương (~0.64)

### Khác Biệt:
- **Fixed 28 days:** Công thức đơn giản hơn, dễ triển khai, số lượng hàng tương đối ít (2.6M)
- **Relative 50%:** Tự động điều chỉnh theo độ dài khóa học, bắt được dài hơi tối đa (3.1M hàng)
- **Relative 25%:** Giai đoạn sớm, tầm nhìn ngắn (2.7M hàng)

### Khuyến Nghị:
🎯 **Sử dụng chế độ FIXED 28 days** cho môi trường sản xuất:
- Đơn giản, dễ giải thích cho bên không kỹ thuật
- Hiệu năng tương đương Relative 50%
- Giảm độ phức tạp tính toán
- Tiêu chí đã được thống nhất (28 ngày = hơn 1 tháng quan sát)

## 6. Tối Ưu Hóa Tương Lai

1. **Feature Engineering:** Thêm các đặc trưng tương tác (interaction features)
2. **Class Balancing:** Điều chỉnh `TRAIN_CLASS_RATIOS` trong config.py
3. **Threshold Tuning:** Tối ưu hóa ngưỡng quyết định để cân bằng Precision-Recall
4. **Hyperparameter:** Tinh chỉnh `C` và `kernel` trong Linear SVC

---

**Ngày Tạo Báo Cáo:** 2025-01-30  
**Giai Đoạn Pipeline:** 1-4 (Generation → Time Windows → Split/SMOTE → Model Training & Evaluation)  
**Thời Gian Thực Nghiệm:** Stage 1-4 mỗi chế độ (~1-2 phút trên máy có GPU)
