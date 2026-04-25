# Mô tả time series của bài toán dự báo sớm

## 1. Bản chất dữ liệu time series

Trong đề tài này, time series không được hiểu là chuỗi số liệu liên tục theo giây, phút hay ngày để dự báo giá trị tương lai theo kiểu forecasting thuần túy. Ở đây, time series là chuỗi hành vi học tập theo tuần của từng người học, được biểu diễn bằng dữ liệu user-week. Mỗi bản ghi tương ứng với một user trong một tuần, phản ánh người học có tham gia hay không tham gia vào từng loại hoạt động học tập trong khoảng thời gian đó.

Lớp dữ liệu này được lưu trong file `step2_user_week_activity.csv`. Đây là lớp dữ liệu mô tả động thái học tập theo thời gian và là nền tảng để hình thành đặc trưng phục vụ dự báo sớm. Các cột cốt lõi của file này gồm `user_id`, `week`, `video`, `problem`, `reply`, `comment`. Trong đó, `week` là tuần ISO suy ra từ dấu thời gian hoạt động, còn bốn cột còn lại là các cờ nhị phân thể hiện sự xuất hiện của hoạt động tương ứng trong tuần đó.

## 2. Cách time series được sử dụng trong bài toán

Pipeline của đề tài không đưa toàn bộ chuỗi tuần thô trực tiếp vào mô hình tuần tự như RNN hoặc Transformer. Thay vào đó, dữ liệu theo tuần được tổng hợp và chuẩn hóa thành đặc trưng ở mức user-level để phục vụ phân loại nhãn mức độ tham gia. Cách tổ chức này phù hợp với mục tiêu của bài toán dự báo sớm, vì mô hình cần xác định sớm người học đang ở mức tương tác thấp, trung bình hay cao, chứ không phải dự đoán chính xác giá trị của từng tuần kế tiếp.

Từ góc nhìn pipeline, `step2_user_week_activity.csv` là nguồn time series đầu vào ở mức tuần, còn các file `phase5_*` là các tập đã được chia để huấn luyện và đánh giá mô hình dự báo sớm. Nhãn mục tiêu của bài toán gồm 3 mức: `Low`, `Medium`, `High`. Ba nhãn này được xây dựng từ điểm engagement chuẩn hóa theo các ngưỡng phân vị, qua đó chuyển bài toán từ mô tả hành vi thành bài toán phân loại có ý nghĩa cảnh báo sớm.

## 2.1. Mô phỏng kịch bản dự báo sớm với `cutoff-week`

Để đảm bảo tính hợp lệ của bài toán "dự báo sớm", pipeline phải ngăn chặn rò rỉ thông tin từ tương lai (data leakage). Nếu các đặc trưng được tính trên toàn bộ lịch sử hoạt động của học viên, mô hình sẽ "biết trước" kết quả, và bài toán trở thành "phân loại sau khi sự đã rồi" thay vì "dự báo".

Để giải quyết vấn đề này, pipeline triển khai một cơ chế "tuần cắt" (`cutoff-week`).

-   **Trích xuất đặc trưng có yếu tố thời gian**: Khi tham số `--cutoff-week` (ví dụ: `202004` cho tuần 4 năm 2020) được chỉ định, tất cả các đặc trưng ở mức user-level (như `problem_total`, `avg_attempts`, `video_sessions`) sẽ chỉ được tính toán dựa trên dữ liệu từ đầu khóa học cho đến tuần cắt đó.
-   **Nhãn mục tiêu (Ground-truth)**: Ngược lại, nhãn mục tiêu (ví dụ `EngagementLabel`) vẫn được tính trên **toàn bộ** lịch sử hoạt động của khóa học. Nhãn này đại diện cho kết quả cuối cùng mà chúng ta muốn dự báo.

Bằng cách này, mô hình được huấn luyện để trả lời câu hỏi: "Dựa trên hành vi của học viên trong N tuần đầu tiên, mức độ tham gia cuối cùng của họ sẽ là gì?". Đây là cách mô phỏng chính xác kịch bản thực tế, nơi chúng ta muốn can thiệp sớm dựa trên những dữ liệu ban đầu có được.

## 3. Ý nghĩa của dự báo sớm (Early Prediction)

Mục tiêu của dự báo sớm trong đề tài là phát hiện nhóm người học có mức tham gia thấp càng sớm càng tốt để hỗ trợ cảnh báo và can thiệp. Vì vậy, `Low` là lớp quan trọng nhất về mặt nghiệp vụ. `Medium` là nhóm trung gian, cần theo dõi thêm trước khi đưa ra kết luận. `High` đại diện cho nhóm tham gia tốt, có mức độ tương tác tích cực và rủi ro thấp hơn.

Trong đánh giá mô hình, pipeline ưu tiên khả năng nhận diện lớp `Low`, vì đây là nhóm cần cảnh báo sớm nhất. Điều này giải thích vì sao độ nhớ lại của lớp `Low` được xem là chỉ số rất quan trọng trong giai đoạn đánh giá.

## 4. Cách chia tập dữ liệu (Data Splitting)

Phase chia dữ liệu chỉ sử dụng ba tập chuẩn là train, valid và test. Tuy nhiên, để tránh leakage theo thời gian, pipeline hỗ trợ chiến lược chia `temporal` và `hybrid`. Với `temporal`, test được lấy từ phần dữ liệu xuất hiện muộn hơn theo trật tự thời gian, còn train và valid là phần sớm hơn. Với `hybrid`, test vẫn là phần cuối theo thời gian, trong khi train/valid được chia tiếp bằng group hoặc stratified để cân bằng giữa tính đúng thứ tự thời gian và độ ổn định phân bố nhãn.

Điểm quan trọng là test không bị chia thành nhiều cấp độ riêng. Đây là một tập đánh giá duy nhất, đại diện cho dữ liệu chưa nhìn thấy, và được dùng để kiểm tra khả năng tổng quát hóa của mô hình trên bối cảnh dự báo sớm.

## 5. Dữ liệu huấn luyện thực tế (Training Data)

File thực tế mà phase huấn luyện mô hình đọc là `phase5_train_modeling.csv`. Đây không còn là dữ liệu tuần thô mà là dữ liệu đã flatten ở mức feature để đưa vào bộ phân loại. File này gồm các cột đặc trưng số được suy ra từ train split, một cột nhãn mục tiêu `StandardLabelKMeans`, và một cột phụ trợ `sample_origin` dùng để ghi nhận nguồn gốc mẫu sau xử lý mất cân bằng.

Các cột đặc trưng thường gặp trong pipeline gồm `num_courses`, `problem_total`, `problem_correct`, `attempts_sum`, `problem_accuracy`, `avg_attempts`, `video_sessions`, `reply_count`, `comment_count`, `engagement_events`. Khi tạo feature, phase huấn luyện sẽ tự loại bỏ các cột metadata như `SplitSet`, `phase5_row_id`, `user_id`, `school`, `EngagementLabel`, `cluster`, `first_activity_time`, `last_activity_time` để tránh làm nhiễu mô hình.

## 6. Dữ liệu kiểm thử thực tế (Test Data)

File test thực tế mà phase đánh giá đọc là `phase5_test.csv`. File này giữ schema của dữ liệu split ở phase trước, nghĩa là vẫn có các cột gốc từ dữ liệu gán nhãn và các cột được enrich từ `combined_user_metrics.csv` nếu có. Ngoài ra, file còn có `phase5_row_id` và `SplitSet` để phục vụ truy vết tập dữ liệu.

Điểm khác biệt quan trọng là test không trải qua oversampling hoặc SMOTE, vì mục đích của nó là phản ánh đúng phân phối thật khi mô hình gặp dữ liệu mới. Do đó, test là thước đo thực tế hơn cho khả năng dự báo sớm của hệ thống.

## 7. So sánh dữ liệu Train và Test

| Tiêu chí | Train thực tế | Test thực tế |
|---|---|---|
| File | `phase5_train_modeling.csv` | `phase5_test.csv` |
| Vai trò | Huấn luyện classifier | Đánh giá trên dữ liệu chưa nhìn thấy |
| Dạng dữ liệu | Bảng đặc trưng đã flatten | Bảng split để đánh giá mô hình |
| Nhãn mục tiêu | Có, `StandardLabelKMeans` | Có, `StandardLabelKMeans` |
| `sample_origin` | Có | Không |
| `SplitSet` | Không | Có |
| `phase5_row_id` | Không | Có |
| `user_id`, `school` | Không dùng làm feature | Có trong dữ liệu gốc |
| Cột time series theo tuần | Không trực tiếp | Không trực tiếp; chuỗi tuần gốc nằm ở `step2_user_week_activity.csv` |

Từ bảng trên có thể thấy rằng train và test khác nhau chủ yếu ở vai trò và ở việc train có thể đã được cân bằng lại bằng oversampling hoặc SMOTE, trong khi test luôn giữ nguyên phân phối thật. Tuy cùng thuộc cùng một bài toán, train là đầu vào cho quá trình học mô hình, còn test là đầu vào cho giai đoạn kiểm tra khả năng tổng quát hóa cuối cùng.

## 8. Tổng quan pipeline theo từng phase

| Phase | Input chính | Phase làm gì | Output chính |
|---|---|---|---|
| Phase 1 | Dữ liệu nguồn và tên trường/tổ chức | Làm sạch, chuẩn hóa tên trường, thống nhất định dạng | Dữ liệu đã chuẩn hóa và báo cáo làm sạch |
| Phase 2 | Dữ liệu sạch từ Phase 1 | Tổng hợp chuỗi hành vi theo tuần, tạo user-week time series và đặc trưng user-level có kiểm soát `cutoff-week` | `combined_user_metrics.csv`, `step2_user_week_activity.csv`, báo cáo chuyển đổi |
| Phase 3 | Dữ liệu tổng hợp | Khảo sát phân phối, kiểm tra thiếu dữ liệu, quan sát xu hướng ban đầu | Báo cáo EDA và nhận xét chất lượng dữ liệu |
| Phase 4 | Dữ liệu đặc trưng và engagement score | Gán nhãn `Low`/`Medium`/`High`, kiểm tra chất lượng nhãn | Bảng nhãn chuẩn hóa, ma trận cluster, báo cáo labeling |
| Phase 5 | Dữ liệu đã gán nhãn | Chia `train`/`valid`/`test` theo chiến lược temporal hoặc hybrid, xử lý mất cân bằng | `phase5_train_modeling.csv`, `phase5_valid.csv`, `phase5_test.csv`, báo cáo split |
| Phase 6 | `phase5_train_modeling.csv`, `phase5_valid.csv`, `phase5_test.csv` | Huấn luyện nhiều mô hình, so sánh metric trên valid/test, chọn mô hình tốt nhất | CSV so sánh mô hình, confusion matrix, predictions, feature importance, model pickle, biểu đồ PNG |
| Phase 7 | Kết quả Phase 6 | Đánh giá mô hình tốt nhất, kiểm tra ngưỡng chất lượng, tổng hợp số liệu cuối | CSV đánh giá, báo cáo cuối, biểu đồ metric và class-wise |
| Phase 8 | Model pickle, predictions và test set | Diễn giải mô hình bằng global importance và local explanation cho từng mẫu | CSV interpretability, báo cáo cuối, biểu đồ global/class-wise/local |

Các output dạng PNG ở các phase cuối được bổ sung để báo cáo không chỉ có số liệu bảng mà còn có hình minh họa trực quan cho mô hình, metric và diễn giải hành vi.

## 9. Kết luận

Tóm lại, time series của đề tài là chuỗi hành vi theo tuần của từng user, được biểu diễn bởi các hoạt động `video`, `problem`, `reply`, `comment` trên nền `week`. Bài toán dự báo sớm được triển khai dưới dạng phân loại ba lớp `Low`, `Medium`, `High`, trong đó `Low` là lớp cần ưu tiên cảnh báo sớm nhất. Dữ liệu train thực tế cho mô hình là `phase5_train_modeling.csv`, còn dữ liệu test thực tế là `phase5_test.csv`. Chuỗi thời gian mức tuần được lưu tách riêng trong `step2_user_week_activity.csv` để làm cơ sở cho quá trình trích xuất và diễn giải hành vi.