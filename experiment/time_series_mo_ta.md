# Mô tả `time series` và cách chia dữ liệu trong đề tài

## 1) `time series` của bài toán là gì?

Trong đề tài này, `time series` không phải là chuỗi dự đoán theo từng phút/giây, mà là chuỗi hành vi theo **tuần** của từng người học.

Mỗi bản ghi mức tuần thể hiện hoạt động của một user trong một tuần, gồm:

- `user_id`: mã người học
- `week`: tuần ISO được suy ra từ thời gian phát sinh hoạt động
- `video`: có/không hoạt động xem video trong tuần
- `problem`: có/không hoạt động làm bài trong tuần
- `reply`: có/không hoạt động trả lời trong tuần
- `comment`: có/không hoạt động bình luận trong tuần

Vì vậy, `time series` ở đây là dữ liệu hành vi được sắp theo thời gian tuần, dùng để mô tả mức độ tương tác và phục vụ dự đoán nhãn sớm.

## 2) Cách chia test trong pipeline

Pipeline có 4 kiểu chia tập, nhưng nếu xét theo hướng dự đoán sớm thì cách quan trọng nhất là:

- `temporal`: chia theo thời gian, lấy các bản ghi cũ hơn làm train/valid và phần cuối theo thời gian làm test.
- `hybrid`: test được lấy theo phần cuối thời gian, còn train/valid thì chia tiếp theo group hoặc stratified để hạn chế leakage.

Trong thực tế, test chỉ là **một tập duy nhất** cho đánh giá cuối, không chia thành nhiều cấp độ test riêng biệt.

Mặc định phase split dùng:

- `valid_size = 0.10`
- `test_size = 0.10`

Nghĩa là dữ liệu được chia thành 3 phần: `train` / `valid` / `test`.

## 3) Ý nghĩa cảnh báo sớm Low / Medium / High

Nhãn của bài toán có 3 mức:

- `Low`
- `Medium`
- `High`

Các nhãn này được gán theo phân vị của điểm engagement chuẩn hoá:

- dưới ngưỡng thấp -> `Low`
- giữa hai ngưỡng -> `Medium`
- trên ngưỡng cao -> `High`

Ý nghĩa trong cảnh báo sớm:

- `Low`: nhóm cần cảnh báo sớm nhất, vì mức tương tác thấp có thể là tín hiệu rủi ro.
- `Medium`: nhóm trung gian, cần theo dõi thêm.
- `High`: nhóm tương tác tốt, ít rủi ro hơn.

Trong đánh giá mô hình, pipeline ưu tiên `recall` của lớp `Low`, vì bắt được càng nhiều người thuộc nhóm rủi ro càng tốt.

## 4) Trong dữ liệu train có những cột gì?

Nếu nói về file huấn luyện sau khi split và xử lý mất cân bằng, file `stage3_train_modeling.csv` có cấu trúc:

- các cột đặc trưng số được chọn tự động từ dữ liệu gốc
- một cột nhãn: `StandardLabelKMeans`
- một cột phụ trợ: `sample_origin`

Nhóm cột đặc trưng thường lấy từ dữ liệu user-level sau khi đã merge, ví dụ:

- `num_courses`
- `problem_total`
- `problem_correct`
- `attempts_sum`
- `problem_accuracy`
- `avg_attempts`
- `video_sessions`
- `reply_count`
- `comment_count`
- `engagement_events`

Nếu nói theo file split gốc `stage3_train.csv`, file này giữ schema của bản dữ liệu đã tách, gồm:

- toàn bộ cột từ dữ liệu gán nhãn và các cột được enrich từ combined CSV
- `stage3_row_id`
- `SplitSet`

## 5) Dữ liệu test time series có những cột gì?

Nếu nói đúng phần `time series` ở mức user-week, dữ liệu tương ứng là `step2_user_week_activity.csv`, gồm:

- `user_id`
- `week`
- `video`
- `problem`
- `reply`
- `comment`

Nếu nói về file test sau khi split để đánh giá mô hình, file `stage3_test.csv` có cùng schema với tập split gốc, tức là:

- các cột gốc từ dữ liệu đã gán nhãn
- các cột được merge từ `combined_user_metrics.csv` nếu có
- `stage3_row_id`
- `SplitSet`

## 6) Kết luận ngắn

- `time series` của đề tài là chuỗi hành vi theo tuần của từng user.
- Test không chia thành nhiều cấp độ riêng, mà chỉ chia thành một tập test theo thời gian hoặc theo chiến lược hybrid.
- Dự đoán sớm trong pipeline tập trung vào phát hiện lớp `Low` để cảnh báo rủi ro sớm.
- Dữ liệu train-modeling là tập feature số + nhãn + nguồn mẫu; dữ liệu time-series mức tuần là bộ cột `user_id, week, video, problem, reply, comment`.