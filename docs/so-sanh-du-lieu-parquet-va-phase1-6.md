# So sánh dữ liệu Parquet cũ và pipeline Phase 1-6

## 1) Mục tiêu tài liệu
Tài liệu này mô tả:
- Khác biệt bản chất dữ liệu giữa Parquet cũ và dữ liệu chuẩn của pipeline hiện tại.
- Độ lớn dữ liệu và mức hạt dữ liệu (event-level hay user-level).
- Cột đặc trưng chính theo từng phase.
- Các vấn đề có thể phát sinh nếu dùng trực tiếp Parquet cũ cho Phase 1-6.

## 2) Ảnh chụp nhanh dữ liệu Parquet cũ
Nguồn kiểm tra: examples/bản cũ/combined_user_problem.parquet

Thông số thực tế:
- Số dòng: 202,905
- Số cột: 14
- Số user duy nhất: 129,516
- Danh sách cột:
  - user_id
  - name
  - gender
  - school
  - year_of_birth
  - course_order
  - enroll_time
  - num_courses
  - log_id
  - problem_id
  - is_correct
  - attempts
  - score
  - submit_time

Nhận xét:
- Đây là dữ liệu mức sự kiện (event-level), một user có thể xuất hiện nhiều dòng.
- Dữ liệu thiên về tương tác problem; chưa có cấu trúc weekly activity chuẩn cho video/reply/comment.

## 3) Khác biệt cốt lõi: Parquet cũ vs dữ liệu chuẩn pipeline mới

### Parquet cũ (bản notebook trước đây)
- Dữ liệu tạo bằng cách merge nhiều bảng theo user_id và lưu trung gian parquet.
- Có thể chứa nhiều giá trị rỗng do outer merge.
- Chưa phải bộ dữ liệu chuẩn để đưa thẳng sang huấn luyện supervised.

### Pipeline mới (phase scripts hiện tại)
- Chuẩn hóa ngay từ đầu về 2 bộ dữ liệu sau combine:
  - combined_user_metrics.csv (mức user)
  - step2_user_week_activity.csv (mức user-week)
- Các phase sau dựa trên 2 bộ chuẩn này để tính engagement, gán nhãn, chia tập, train, evaluate, interpret.
- Thiết kế theo hướng ổn định schema và giảm lệch nghĩa đặc trưng giữa các phase.

## 4) Cột đặc trưng và vai trò theo từng phase

## 4.1) Khác biệt cách tính đặc trưng: bản cũ vs hiện tại

### A. Nhóm lọc user đầu vào
- Bản cũ:
  - num_courses = len(course_order)
  - Giữ user nếu num_courses > 5
- Hiện tại:
  - Vẫn giữ logic num_courses > 5 khi chuẩn hóa từ parquet sang combined_user_metrics.csv.

### B. Nhóm đặc trưng weekly activity nhị phân
- Bản cũ (trên combined_all_data.parquet):
  - week = isocalendar(submit_time)
  - video = 1 nếu có seq theo (user_id, week), ngược lại 0
  - problem = 1 nếu có problem_id theo (user_id, week), ngược lại 0
  - reply = 1 nếu có id_x theo (user_id, week), ngược lại 0
  - comment = 1 nếu có id_y theo (user_id, week), ngược lại 0
- Hiện tại:
  - Chuẩn weekly lưu trực tiếp trong step2_user_week_activity.csv với cột:
    - user_id, week, video, problem, reply, comment
  - Cùng ý nghĩa nhị phân, nhưng được chuẩn hóa schema sớm để phase 2 dùng trực tiếp.

### C. Nhóm trọng số hoạt động và điểm engagement
- Bản cũ:
  - Trọng số hoạt động:
    - w_a = sum(x_{i,a,w}) / (S * N)
  - Điểm theo tuần:
    - E_{i,w} = sum_a (w_a * x_{i,a,w})
  - Điểm theo user:
    - E_i = sum_w E_{i,w}
  - Chuẩn hóa:
    - E_norm = (E - E_min) / (E_max - E_min)
  - Gán nhãn theo phân vị:
    - Low: E_norm <= q33
    - Medium: q33 < E_norm <= q66
    - High: E_norm > q66
- Hiện tại (phase 2):
  - Giữ cùng logic công thức trọng số, cộng điểm theo tuần, tổng theo user và chuẩn hóa.
  - Khác biệt chính là dữ liệu đầu vào đã ở dạng chuẩn (combined + weekly), giảm phụ thuộc vào merge ad-hoc trong notebook.

### D. Nhóm đặc trưng user-level cho supervised
- Bản cũ:
  - Chủ yếu xây từ các bảng merge thô, sau đó tự tính trong notebook; đặc trưng có thể thay đổi theo từng cell.
  - Dễ phát sinh nhiều null do outer merge.
- Hiện tại (phase 1):
  - Đầu ra cố định schema trong combined_user_metrics.csv, ví dụ:
    - problem_total, problem_correct, problem_accuracy
    - attempts_sum, avg_attempts
    - score_sum, score_count, avg_score
    - video_sessions, video_count, watched_seconds, watched_hours
    - reply_count, comment_count, forum_total
    - engagement_events, first_activity_time, last_activity_time

### E. Mapping nhanh đặc trưng tương ứng
- num_courses (cũ) -> num_courses (mới)
- problem_id notna count (cũ) -> problem_total (mới)
- is_correct tổng hợp (cũ) -> problem_correct (mới)
- attempts trung bình/tổng (cũ) -> attempts_sum, avg_attempts (mới)
- score trung bình/tổng (cũ) -> score_sum, score_count, avg_score (mới)
- video/reply/comment binary theo tuần (cũ) -> video/problem/reply/comment trong step2_user_week_activity.csv (mới)
- E, E_norm, EngagementLabel (cũ) -> nhóm cột/nhãn engagement do phase 2 sinh ra (mới)

### F. Ý nghĩa thực tế của khác biệt
- Bản cũ mạnh về khám phá nhanh trong notebook, nhưng công thức và đặc trưng dễ phụ thuộc trạng thái từng cell.
- Bản hiện tại mạnh về tính tái lập:
  - Công thức được đóng gói trong script.
  - Schema đặc trưng ổn định xuyên phase 1-6.
  - Ít rủi ro sai lệch đặc trưng khi train/evaluate/interpret.

### G. Bảng làm rõ khác biệt theo từng cột đặc trưng

| Nhóm | Cột nguồn bản cũ | Cách tính bản cũ | Cột chuẩn hiện tại | Cách tính hiện tại | Khác biệt chính |
|---|---|---|---|---|---|
| Lọc user | course_order | num_courses = len(course_order), giữ num_courses > 5 | num_courses | Giữ num_courses > 5 trong bước chuẩn hóa | Tương đương logic |
| Problem volume | problem_id | count problem_id theo user | problem_total | đếm problem_id khác rỗng theo user | Tương đương ý nghĩa |
| Problem correctness | is_correct | mean/sum tùy cell notebook | problem_correct, problem_accuracy | problem_correct = số bản ghi đúng; problem_accuracy = problem_correct/problem_total | Hiện tại cố định công thức, không phụ thuộc cell |
| Attempts | attempts | mean attempts theo user (thường trong notebook) | attempts_sum, avg_attempts | attempts_sum = tổng attempts; avg_attempts = attempts_sum/problem_total | Hiện tại giữ cả tổng và trung bình |
| Score | score | mean/sum tùy cell, nhiều NaN | score_sum, score_count, avg_score | avg_score = score_sum/score_count (bỏ NaN) | Hiện tại tách rõ mẫu số score hợp lệ |
| Video weekly | seq | video=1 nếu seq notna theo (user,week) | weekly.video | cờ nhị phân video theo user-week | Tương đương ý nghĩa, khác ở nơi lưu chuẩn |
| Problem weekly | problem_id | problem=1 nếu problem_id notna theo (user,week) | weekly.problem | cờ nhị phân problem theo user-week | Tương đương ý nghĩa |
| Reply weekly | id_x | reply=1 nếu id_x notna theo (user,week) | weekly.reply | cờ nhị phân reply theo user-week | Cột nguồn cũ là hậu quả merge (id_x), hiện tại đặt tên rõ nghĩa |
| Comment weekly | id_y | comment=1 nếu id_y notna theo (user,week) | weekly.comment | cờ nhị phân comment theo user-week | Cột nguồn cũ là hậu quả merge (id_y), hiện tại đặt tên rõ nghĩa |
| Engagement weekly | video/problem/reply/comment | E_i,w = sum_a(w_a*x_i,a,w) | E theo user (phase 2) | cộng điểm theo tuần từ weekly weights | Tương đương công thức cốt lõi |
| Engagement normalize | E | E_norm min-max | E_norm | normalize_score min-max | Tương đương |
| Label | E_norm | q33, q66 -> Low/Medium/High | EngagementLabel | quantile theo q_low, q_high (mặc định 0.33/0.66) | Hiện tại có tham số hóa ngưỡng |
| Time boundary | submit_time | dùng trực tiếp khi xử lý tuần | first_activity_time, last_activity_time | lấy min/max thời gian hoạt động | Hiện tại có thêm đặc trưng biên thời gian |
| Forum tổng hợp | reply/comment merge thô | không luôn có cột tổng hợp ổn định | reply_count, comment_count, forum_total | forum_total = reply_count + comment_count | Hiện tại có schema rõ cho diễn đàn |

Ghi chú:
- weekly.* ở bảng là các cột trong step2_user_week_activity.csv.
- Với nhánh dùng combined parquet cũ làm đầu vào phase 1, các cột video/reply/comment ở user-level có thể bằng 0 nếu parquet không chứa đủ tín hiệu tương ứng.

## Phase 1: Time-series feature extraction
Đầu vào:
- Luồng mặc định: JSONL dataset gốc.
- Luồng thay thế: combined parquet qua tham số combined-parquet.

Đầu ra chính:
- combined_user_metrics.csv
- step2_user_week_activity.csv

Cột quan trọng thường gặp trong combined_user_metrics.csv:
- user_id
- num_courses
- problem_total
- problem_correct
- attempts_sum
- problem_accuracy
- avg_attempts
- video_sessions
- reply_count
- comment_count
- engagement_events

Cột weekly:
- user_id
- week
- video
- problem
- reply
- comment

Lưu ý khi chuyển từ Parquet cũ:
- Có lọc user theo num_courses > 5.
- Nếu parquet không có video/reply/comment thì các cột tương ứng có thể về 0.

## Phase 2: KMeans label validation (step_3_engagement_report)
Đầu vào:
- combined_user_metrics.csv
- step2_user_week_activity.csv

Ý nghĩa:
- Tính trọng số hoạt động tuần.
- Tính engagement score và chuẩn hóa.
- Phân cụm MiniBatchKMeans để tạo/kiểm tra nhãn mức độ tương tác.

Đầu ra điển hình:
- step3_activity_weights.csv
- Báo cáo/nhãn engagement cho phase sau.

## Phase 3: Train-valid-test split
Đầu vào:
- File nhãn từ phase trước.
- Có thể enrich thêm cột còn thiếu từ combined_user_metrics.csv theo user_id.

Ý nghĩa:
- Tạo tập train, valid, test tránh leakage theo cấu hình.
- Xử lý mất cân bằng lớp chỉ trên tập train-modeling.

Đầu ra điển hình:
- stage3_train_modeling.csv
- stage3_valid.csv
- stage3_test.csv
- stage3_split_report.txt

## Phase 4: Supervised model training
Đầu vào mặc định:
- stage3_train_modeling.csv
- stage3_valid.csv
- stage3_test.csv

Ý nghĩa:
- Huấn luyện nhiều model supervised.
- Chọn model tốt nhất theo metric validation.
- Đánh giá trên test và xuất artifact đầy đủ.

Đầu ra điển hình:
- phase4_model_comparison.csv
- phase4_classification_metrics.csv
- phase4_confusion_matrix.csv
- phase4_best_model_predictions.csv
- phase4_feature_importance.csv
- phase4_best_model.pkl
- phase4_training_report.txt

## Phase 5: Model evaluation metrics
Đầu vào mặc định:
- phase4_model_comparison.csv
- phase4_classification_metrics.csv
- phase4_confusion_matrix.csv
- phase4_feature_importance.csv

Ý nghĩa:
- Tổng hợp metric theo model.
- Chọn best model theo tiêu chí định sẵn.
- Xuất báo cáo cuối phục vụ thuyết minh.

Đầu ra điển hình:
- phase5_model_selection_summary.csv
- phase5_best_model_class_metrics.csv
- phase5_best_model_confusion_matrix.csv
- phase5_top_features.csv
- phase5_metric_checks.csv
- phase5_evaluation_report.txt
- final_summary_report.txt

## Phase 6: Model interpretability
Đầu vào mặc định:
- phase4_best_model.pkl
- phase4_best_model_predictions.csv
- stage3_test.csv
- final_summary_report.txt

Ý nghĩa:
- Giải thích mô hình đã chọn ở phase 4.
- Xuất global importance và local explanation.
- Ưu tiên SHAP cho CatBoost, fallback theo loại mô hình.

Đầu ra điển hình:
- phase6_global_importance.csv
- phase6_classwise_importance.csv
- phase6_local_explanations.csv
- phase6_interpretability_report.txt
- final_summary_report.txt (được cập nhật bổ sung phần interpretability)

## 5) Các vấn đề liên quan khi dùng Parquet cũ xuyên pipeline

## 5.1 Sai mức hạt dữ liệu
- Parquet cũ là event-level, trong khi phase 3-6 kỳ vọng dữ liệu đã chuẩn hóa theo user/feature.
- Nếu dùng trực tiếp dễ gây trùng user và làm méo phân phối nhãn.

## 5.2 Thiếu cột hoạt động đa kênh
- Parquet cũ đang thiên về problem.
- Video/reply/comment có thể không đủ để tính engagement cân bằng.
- Kết quả: trọng số hoạt động lệch, điểm engagement thiếu ổn định.

## 5.3 Rủi ro mismatch schema giữa phase
- Phase 4 và 6 phụ thuộc mạnh vào danh sách feature_columns thống nhất từ phase trước.
- Nếu schema thay đổi đột ngột do đầu vào parquet thô, có thể phát sinh lỗi:
  - Không suy luận được feature columns.
  - Thiếu cột bắt buộc cho huấn luyện hoặc giải thích.

## 5.4 Rủi ro leakage và split quality
- Dữ liệu event-level có thể khiến cùng user xuất hiện quá nhiều và chồng chéo qua các split nếu xử lý không chuẩn.
- Làm cho metric đánh giá cao giả tạo.

## 5.5 Rủi ro diễn giải sai ý nghĩa đặc trưng
- Feature importance ở phase 4-6 chỉ đáng tin khi feature đã được chuẩn hóa đúng ngữ nghĩa.
- Nếu input thô, đóng góp đặc trưng có thể phản ánh nhiễu merge thay vì hành vi học tập thật.

## 6) Khuyến nghị sử dụng dữ liệu đúng cách
- Không dùng trực tiếp parquet cũ để chạy thẳng phase 2-6.
- Luôn chuẩn hóa qua phase 1 để tạo:
  - combined_user_metrics.csv
  - step2_user_week_activity.csv
- Sau đó chạy tuần tự phase 2 -> phase 3 -> phase 4 -> phase 5 -> phase 6.
- Nếu buộc dùng parquet cũ làm đầu vào, cần kiểm tra tối thiểu:
  - Tính duy nhất user_id sau tổng hợp.
  - Tỷ lệ thiếu của các cột hoạt động video/reply/comment.
  - Tính nhất quán cột feature giữa phase 3, phase 4 và phase 6.

## 7) Kết luận ngắn
- Parquet cũ phù hợp vai trò dữ liệu trung gian lịch sử, không phải bộ dữ liệu chuẩn huấn luyện hiện tại.
- Bộ dữ liệu chuẩn sau combine cho pipeline mới là cặp CSV user-level và user-week.
- Độ tin cậy kết quả phase 4-6 phụ thuộc trực tiếp vào bước chuẩn hóa schema và đặc trưng ở phase 1-2.

## 8) Bổ sung phân tích từ bài báo (electronics-14-03018)

Nguồn tham chiếu:
- docs/electronics-14-03018.pdf

### 8.1 Bài báo mô tả đặc trưng như thế nào
- Quy mô dữ liệu nghiên cứu trong bài báo:
  - 127 sinh viên, 13 tuần học.
  - 44 đặc trưng hoạt động theo tuần cho mỗi sinh viên.
- Nhóm đặc trưng chính được nêu:
  - Demographics: giới tính, khoa.
  - Interaction: mức độ tương tác với e-book, thảo luận, bài giảng, assignment, quiz.
  - Effort/Investment: thời gian làm bài và các tín hiệu nỗ lực.
- Bài báo nhấn mạnh đặc trưng mang tính hành vi theo thời gian (week-level), không xem engagement là tĩnh.

### 8.2 Công thức đặc trưng và nhãn trong bài báo
- Dữ liệu hoạt động theo tuần được mã hóa nhị phân có/không theo từng loại hoạt động.
- Trọng số hoạt động trong bài báo theo dạng:
  - w_a = sum(x_{i,a,w}) / (S * N)
- Điểm engagement của người học được cộng dồn từ hoạt động theo tuần:
  - E_i = sum_w sum_a (w_a * x_{i,a,w})
- Gán nhãn engagement theo phân vị điểm:
  - Low, Moderate, High (xấp xỉ theo các ngưỡng percentile thấp/trung/cao).

### 8.3 Đối chiếu trực tiếp với pipeline hiện tại
- Tương đồng cốt lõi:
  - Cùng tư tưởng weekly behavioral profiling.
  - Cùng logic dùng trọng số hoạt động và tổng hợp điểm engagement từ nhiều hoạt động.
  - Cùng gán nhãn theo phân vị (thấp, trung bình, cao).
- Khác biệt triển khai:
  - Bài báo mô tả tập đặc trưng rộng (44 đặc trưng/tuần) và mang tính nghiên cứu.
  - Pipeline hiện tại chuẩn hóa thành schema vận hành rõ ràng hơn:
    - combined_user_metrics.csv (user-level)
    - step2_user_week_activity.csv (user-week binary)
  - Pipeline hiện tại ưu tiên tính tái lập giữa các phase 1-6, giảm phụ thuộc thao tác ad-hoc của notebook.

### 8.4 Hàm ý cho tài liệu so sánh dữ liệu
- Khi viết thuyết minh, có thể trình bày pipeline hiện tại là phiên bản triển khai hóa từ ý tưởng bài báo:
  - Giữ nguyên nguyên lý đặc trưng hành vi theo tuần.
  - Chuẩn hóa lại đầu vào/đầu ra để phù hợp chuỗi ML production (split, train, evaluate, interpret).
- Nếu cần bám sát bài báo hơn, hướng mở rộng tự nhiên là tăng số lượng đặc trưng weekly từ bộ hiện tại lên gần cấu hình giàu đặc trưng hơn (theo nhóm interaction + effort).
