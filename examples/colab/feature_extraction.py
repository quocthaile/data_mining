import pandas as pd

# =============================================================================
# CẢNH BÁO: SCRIPT KHÁM PHÁ
# File này chỉ dành cho mục đích khám phá dữ liệu nhanh (EDA) trong notebook.
# Phương pháp lấy mẫu (sample 5k -> top 5k) có thể gây thiên lệch (bias)
# và không đại diện cho toàn bộ tập dữ liệu.
# Logic tính toán trong pipeline chính (các file phase_*.py) là phiên bản chuẩn.
# =============================================================================

# 1. Load dữ liệu và lấy sample 5k
df = pd.read_parquet("/kaggle/input/datasets/kaling92/combined-all/combined_all_data.parquet")
df_sample = df.sample(n=5000, random_state=42)

# 2. Lấy top 5000 học viên có nhiều khóa học nhất
top_students = df_sample.groupby('user_id')['num_courses'].max().reset_index()
top_students = top_students.sort_values(by='num_courses', ascending=False).head(5000)
top_ids = top_students['user_id'].tolist()

# 3. Lọc dữ liệu chỉ cho nhóm top học viên
df_top = df_sample[df_sample['user_id'].isin(top_ids)]

# 4. Chuyển submit_time sang datetime và tính tuần
df_top['submit_time'] = pd.to_datetime(df_top['submit_time'], errors='coerce')
df_top['week'] = df_top['submit_time'].dt.isocalendar().week

# 5. Biểu diễn hoạt động nhị phân
activities = {
    'video': df_top.groupby(['user_id','week'])['seq'].apply(lambda x: 1 if x.notna().any() else 0),
    'problem': df_top.groupby(['user_id','week'])['problem_id'].apply(lambda x: 1 if x.notna().any() else 0),
    'reply': df_top.groupby(['user_id','week'])['id_x'].apply(lambda x: 1 if x.notna().any() else 0),
    'comment': df_top.groupby(['user_id','week'])['id_y'].apply(lambda x: 1 if x.notna().any() else 0)
}
activity_df = pd.concat(activities, axis=1).reset_index()

# 6. Tính trọng số và chuẩn hóa
S = activity_df['user_id'].nunique()
N = activity_df['week'].nunique()
weights = {a: activity_df[a].sum() / (S * N) for a in activities.keys()}
total_w = sum(weights.values())
weights = {k: v/total_w for k,v in weights.items()}
print("Trọng số chuẩn hóa (top học viên):", weights)

# 7. Tính điểm E cho từng học viên
activity_df['E'] = (weights['video']*activity_df['video'] +
                    weights['problem']*activity_df['problem'] +
                    weights['reply']*activity_df['reply'] +
                    weights['comment']*activity_df['comment'])
student_scores = activity_df.groupby('user_id')['E'].sum().reset_index()

# 8. Chuẩn hóa điểm và gán nhãn
min_e, max_e = student_scores['E'].min(), student_scores['E'].max()
student_scores['E_norm'] = (student_scores['E'] - min_e) / (max_e - min_e) if max_e > min_e else 0.0
low_th = student_scores['E_norm'].quantile(0.33)
high_th = student_scores['E_norm'].quantile(0.67)

def label(score):
    if score <= low_th: return 'Low'
    elif score <= high_th: return 'Medium'
    else: return 'High'

student_scores['EngagementLabel'] = student_scores['E_norm'].apply(label)

# 9. Tính tỷ lệ nhãn cho nhóm top học viên
label_counts = student_scores['EngagementLabel'].value_counts(normalize=True) * 100
print("Tỷ lệ nhãn (%):")
print(label_counts)