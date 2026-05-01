# Kiến trúc Pipeline và Luồng Xử lý Thực nghiệm (MOOCCubeX)

Tài liệu này mô tả chi tiết quy trình xử lý dữ liệu, trích xuất đặc trưng, gán nhãn, và huấn luyện mô hình dự báo sinh viên có nguy cơ tương tác thấp/bỏ học (early warning system). Hệ thống được thiết kế dưới dạng pipeline 8 giai đoạn (Phases 1-8) để dễ dàng mở rộng và tối ưu.

---

## 1. Luồng xử lý hệ thống và Output từng Phase

Hệ thống được điều phối bởi `run_experiment_stages.py`, chạy tuần tự qua 8 phase:

| Phase | Tên Phase | Mô tả công việc | Dữ liệu Output (đặt trong thư mục `result/phaseX/`) |
|-------|-----------|------------------|--------------------------------------------------|
| **1** | **Data Preparation** | Dịch tên trường học, kết hợp dữ liệu log sự kiện thô (events) thành các metrics theo `user_course_key` và hoạt động theo tuần. | `user_school_en.json`, `combined_user_metrics.csv`, `step2_user_week_activity.csv` |
| **2** | **Data Cleaning** | Xử lý dữ liệu bị thiếu (missing values) và loại bỏ các giá trị ngoại lai (outliers). | `combined_user_metrics_clean.csv` |
| **3** | **Data Transformation** | Biến đổi đặc trưng (feature engineering), trích xuất các features theo thời gian (time-series/temporal dynamics). | `combined_user_metrics_transformed.csv` |
| **4** | **Data Labeling** | Dùng thuật toán học không giám sát (K-Means) kết hợp với các ngưỡng phân vị để tạo nhãn tương tác (Low/Medium/High) và cảnh báo sớm. | `phase4_2_standard_labels_kmeans.csv` và các báo cáo phân bố cụm. |
| **5** | **Data Splitting** | Chia tập dữ liệu thành Train/Validation/Test an toàn để tránh rò rỉ dữ liệu (data leakage) và xử lý mất cân bằng lớp. | `phase5_train_modeling.csv`, `phase5_valid.csv`, `phase5_test.csv` |
| **6** | **Model Training** | Huấn luyện nhiều mô hình phân lớp khác nhau và chọn ra mô hình tốt nhất dựa trên Validation set. | `phase6_best_model.pkl`, các file so sánh, độ quan trọng đặc trưng, confusion matrix. |
| **7** | **Model Evaluation** | Đánh giá lại mô hình tốt nhất dựa trên tập Test, so sánh các metrics chuẩn mực. | `final_summary_report.txt`, biểu đồ metrics phân lớp. |
| **8** | **Interpretability** | Phân tích và giải thích quyết định của mô hình trên phạm vi toàn cục (Global) và cục bộ (Local/từng sinh viên). | Các file giải thích mô hình (global/class/local contributions). |

---

## 2. Đặc trưng dữ liệu và Cách tính toán

### 2.1. Đối tượng (Granularity)
Dữ liệu được xử lý ở mức **user-course** (cặp khóa gồm sinh viên và khóa học - `user_course_key`). Điều này cho phép phân tích hành vi của một sinh viên cụ thể trong một khóa học cụ thể.

### 2.2. Dữ liệu Đầu ra Chính
1. **`combined_user_metrics.csv` (Aggregated Features):**
   - **Đặc trưng:** Tổng số lần xem video, thời gian xem, số lần tương tác bài tập (problem), forum, v.v.
   - **Cách tính:** Gộp (Group By) từ log dữ liệu kiện thô dựa trên `user_course_key`. Lấy tổng (`sum`), trung bình (`mean`), và độ lệch chuẩn của các hành động tương tác. Tích hợp thêm các chỉ số về thời gian (first_activity_time, last_activity_time).
2. **`step2_user_week_activity.csv` (Time-Series Features):**
   - **Đặc trưng:** Chuỗi sự kiện tương tác tuần tự của sinh viên.
   - **Cách tính:** Map thời gian thực hiện sự kiện thành các Tuần chuẩn (Week ISO) của khóa học, từ đó đếm số lượng sự kiện mỗi tuần nhằm phát hiện xu hướng tương tác giảm dần hoặc đột biến.

---

## 3. Cách chia Nhãn (Data Labeling)

Việc gán nhãn được thực hiện ở **Phase 4**, vì dữ liệu gốc không có sẵn nhãn rớt môn hay bỏ học. Cách tiếp cận như sau:
1. **Chấm điểm tương tác (Engagement Scoring):** Dựa trên mức độ tham gia vào video, problem, forum.
2. **K-Means Clustering:** Phân cụm những sinh viên trong cùng một khóa học thành các nhóm (thường là 3 nhóm: Kém, Trung Bình, Tốt).
3. **Gán nhãn chuẩn (Standard Labels):** Phân chia thành các nhãn `Low`, `Medium`, `High` sử dụng phân vị (`q_low`, `q_high`) trên điểm số tương tác.
4. **Cảnh báo sớm (Early-Warning Stages):** Xây dựng các nhãn độc lập đại diện cho việc sinh viên có nguy cơ thấp ở thời điểm hoàn thành 50%, 75% quá trình diễn ra khóa học. 

---

## 4. Cách phân chia tập Train / Test (Data Splitting)

Việc chia tập dữ liệu ở **Phase 5** được tính toán kỹ lưỡng để tránh hiện tượng Data Leakage (mô hình học lỏm dữ liệu). 
- **Chiến lược chia (Split Strategy):**
  - **Group-based Split:** Bắt buộc gom toàn bộ record của cùng một `user_id` (hoặc `user_course_key`) vào cùng một tập (Train, Valid, hoặc Test). Việc này đảm bảo mô hình không dự đoán trên những người nó đã từng học một phần dữ liệu ở tập Train.
  - **Temporal Split (Theo thời gian):** Cắt một mốc thời gian cố định dựa trên biến `last_activity_time` (vd: các tương tác trước 2020-10 là Train, sau đó là Test). Việc này sát với thực tế mô hình dự đoán tương lai nhất.
- **Tỉ lệ (Splits):** Train/Valid/Test thường được chia với tham số `valid_size=0.1` và `test_size=0.1`.
- **Xử lý mất cân bằng lớp (Class Imbalance):**
  - Sau khi chia tập, nhãn thiểu số (ví dụ: nhóm Low engagement) trên tập Train sẽ được áp dụng các phương pháp upsampling như Random Oversample hoặc SMOTE nhằm cân bằng với nhãn đa số.

---

## 5. Dữ liệu đưa vào mô hình & Cách hoạt động của Model

### 5.1 Đầu vào của mô hình
- Dữ liệu đưa vào Phase 6 là các file CSV (`phase5_train_modeling.csv`) trong đó:
  - **Features (X):** Các biến thuộc tính định lượng và phân loại (số lượt xem, thời gian làm bài, xu hướng tương tác tuần, ...).
  - **Target (Y):** Cột nhãn như `StandardLabelKMeans` (High/Medium/Low).

### 5.2 Quá trình Training & Dự đoán
- **Mô hình huấn luyện:** 
  Hệ thống sử dụng các thuật toán dựa trên cây quyết định hiệu năng cao như `Random Forest`, `HistGradientBoosting`, và `Logistic Regression` làm mốc so sánh.
- **Cross-Validation:** 
  Áp dụng K-fold CV với tập Validation trên các chỉ số `macro_f1`, `weighted_f1` để tối ưu siêu tham số. 
- **Cơ chế hoạt động chung (Luồng suy diễn):**
  1. Mô hình phân tích các hành vi tần suất và thời gian học của sinh viên cho đến một thời điểm quan sát cụ thể (ví dụ sau nửa khóa học).
  2. Bằng cách so sánh bộ hành vi này với những tập mẫu người dùng rớt/đậu trong quá khứ đã được học trên nhánh cây phân loại (RF/GBDT).
  3. Mô hình sinh ra các giá trị xác suất (Probabilities). Nếu xác suất dự báo người dùng rơi vào lớp `Low` vượt qua ngưỡng nhất định, hệ thống sẽ chốt dự đoán sinh viên là "Nguy cơ rớt".

### 5.3 Cách mở rộng tối ưu
Để mở rộng hệ thống, có thể tiếp cận theo hướng sau:
- **Phase 3:** Bổ sung thêm feature về đồ thị quan hệ sinh viên (Graph) hoặc chuỗi thời gian sâu hơn (RNN/LSTM embeddings).
- **Phase 4:** Đưa trực tiếp điểm khóa học thực tế vào để tạo nhãn có giám sát (nếu data tương lai thu thập được).
- **Phase 6:** Thêm các kiến trúc deep learning, XGBoost, LightGBM vào tham số truyền vào model training.
- **Thực thi:** Sử dụng `--phase 6 --phase6-models lightgbm,xgboost` thông qua command line để linh hoạt mà không cần sửa code cốt lõi.
