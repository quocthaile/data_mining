import json
import pandas as pd
import numpy as np
from collections import defaultdict
import os
import datetime

def normalize_user_id(uid):
    """Chuẩn hóa ID người dùng về định dạng chuẩn 'U_xxxx'."""
    uid_str = str(uid)
    if not uid_str.startswith('U_'):
        return f"U_{uid_str}"
    return uid_str

def load_school_mapping(school_file_path):
    """Tải bộ từ điển ánh xạ tên trường học (O(N) Complexity)."""
    school_map = {}
    if not os.path.exists(school_file_path):
        print(f"[CẢNH BÁO] Không tìm thấy tệp {school_file_path}. Bỏ qua bước mapping tiếng Anh.")
        return school_map

    with open(school_file_path, 'r', encoding='utf-8') as f:
        for line in f:
            data = json.loads(line.strip())
            name_vn_cn = data.get('name')
            name_en = data.get('name_en')
            if name_vn_cn and name_en:
                school_map[name_vn_cn] = name_en
                
    return school_map

def extract_features(
    user_file='user.json',
    problem_file='user-problem.json',
    video_file='user-video.json',
    comment_file='comment.json',
    reply_file='reply.json',
    school_file='school.json',
    output_file='compiled_feature_vector.csv'
):
    school_map = load_school_mapping(school_file)

    # In-memory Database với các trường trung gian phục vụ tính toán
    user_features = defaultdict(lambda: {
        'gender': np.nan, 'school': '', 'year_of_birth': np.nan, 'num_courses': 0,
        'is_correct': 0, 'attempts': 0, 'score': 0.0, 
        'seq': 0.0, 'speed_sum': 0.0, 'speed_count': 0,
        'rep_counts': 0, 'cmt_counts': 0
    })

    # 1. Trích xuất: user_id, gender, school, year_of_birth, num_courses
    print(f"Đang xử lý: {user_file}")
    with open(user_file, 'r', encoding='utf-8') as f:
        for line in f:
            data = json.loads(line.strip())
            uid = normalize_user_id(data.get('id'))
            user_features[uid]['gender'] = data.get('gender')
            
            original_school = data.get('school', '')
            user_features[uid]['school'] = school_map.get(original_school, original_school)
            user_features[uid]['year_of_birth'] = data.get('year_of_birth')
            user_features[uid]['num_courses'] = len(data.get('course_order', []))

    # 2. Trích xuất: is_correct, attempts, score
    print(f"Đang xử lý: {problem_file}")
    with open(problem_file, 'r', encoding='utf-8') as f:
        for line in f:
            data = json.loads(line.strip())
            uid = normalize_user_id(data.get('user_id'))
            user_features[uid]['is_correct'] += int(data.get('is_correct', 0))
            user_features[uid]['attempts'] += int(data.get('attempts', 0))
            score = data.get('score')
            if score is not None:
                user_features[uid]['score'] += float(score)

    # 3. Trích xuất: seq (tổng thời gian), speed (phục vụ tính trung bình)
    print(f"Đang xử lý: {video_file}")
    with open(video_file, 'r', encoding='utf-8') as f:
        for line in f:
            data = json.loads(line.strip())
            uid = normalize_user_id(data.get('user_id'))
            for sequence in data.get('seq', []):
                for segment in sequence.get('segment', []):
                    duration = segment.get('end_point', 0) - segment.get('start_point', 0)
                    if duration > 0:
                        user_features[uid]['seq'] += duration
                    user_features[uid]['speed_sum'] += segment.get('speed', 1.0)
                    user_features[uid]['speed_count'] += 1

    # 4. Trích xuất: cmt_counts
    print(f"Đang xử lý: {comment_file}")
    with open(comment_file, 'r', encoding='utf-8') as f:
        for line in f:
            data = json.loads(line.strip())
            uid = normalize_user_id(data.get('user_id'))
            user_features[uid]['cmt_counts'] += 1

    # 5. Trích xuất: rep_counts
    print(f"Đang xử lý: {reply_file}")
    with open(reply_file, 'r', encoding='utf-8') as f:
        for line in f:
            data = json.loads(line.strip())
            uid = normalize_user_id(data.get('user_id'))
            user_features[uid]['rep_counts'] += 1

    # Khóa cấu trúc cột (Strict Column Ordering) theo feature_vector.txt
    print("Đang biên dịch Feature Vector Matrix...")
    target_columns = [
        'user_id', 'gender', 'school', 'year_of_birth', 'num_courses',
        'is_correct', 'attempts', 'score', 'seq', 'speed',
        'rep_counts', 'cmt_counts'
    ]
    
    final_data = []
    for uid, feats in user_features.items():
        # Tính tốc độ trung bình, nếu không xem video thì mặc định tốc độ là 1.0 (chuẩn)
        avg_speed = (feats['speed_sum'] / feats['speed_count']) if feats['speed_count'] > 0 else 1.0
        
        row_data = {
            'user_id': uid,
            'gender': feats['gender'],
            'school': feats['school'],
            'year_of_birth': feats['year_of_birth'],
            'num_courses': feats['num_courses'],
            'is_correct': feats['is_correct'],
            'attempts': feats['attempts'],
            'score': feats['score'],
            'seq': feats['seq'],
            'speed': avg_speed,
            'rep_counts': feats['rep_counts'],
            'cmt_counts': feats['cmt_counts']
        }
        final_data.append(row_data)

    # Chuyển đổi thành DataFrame và ép thứ tự cột (Enforce Column Order)
    df_features = pd.DataFrame(final_data)
    df_features = df_features[target_columns]
    
    # Cơ chế xử lý ngoại lệ I/O (File Lock Bypass)
    try:
        df_features.to_csv(output_file, index=False, encoding='utf-8-sig')
        print(f"[HOÀN TẤT] Pipeline thực thi thành công. Dữ liệu lưu tại: {output_file}")
    except PermissionError:
        print(f"\n[LỖI I/O] Tệp '{output_file}' đang bị khóa. Đang kích hoạt Fallback mode...")
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        base_name, ext = os.path.splitext(output_file)
        fallback_file = f"{base_name}_{timestamp}{ext}"
        
        df_features.to_csv(fallback_file, index=False, encoding='utf-8-sig')
        print(f"[KHẮC PHỤC] Đã ghi dữ liệu vào tệp dự phòng: {fallback_file}")
        
    return df_features

if __name__ == "__main__":
    df = extract_features()