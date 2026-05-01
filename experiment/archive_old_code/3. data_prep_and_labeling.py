import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler

# Cấu hình đường dẫn
INPUT_FILE = "user_centric_features.csv" # Bộ dữ liệu 15 tuần tổng hợp của bạn
GROUND_TRUTH_FILE = "ground_truth_labels.csv" # File đáp án cần xuất ra

# BỘ TRỌNG SỐ TÍNH ĐIỂM TƯƠNG TÁC (WES)
WEIGHTS = {
    'total_study_time': 0.35,  
    'avg_score': 0.30,         
    'accuracy_rate': 0.20,     
    'attempts': 0.10,          
    'total_forum_activity': 0.05 
}

def main():
    print("="*60)
    print("TẠO NHÃN CHUNG CUỘC (GROUND TRUTH) TỪ DỮ LIỆU 15 TUẦN")
    print("="*60)

    # 1. Đọc dữ liệu
    print("-> Đang tải dữ liệu tổng hợp 15 tuần...")
    df = pd.read_csv(INPUT_FILE, low_memory=False)
    
    # Kiểm tra user_id
    if 'user_id' not in df.columns:
        print("❌ LỖI NGHIÊM TRỌNG: Không tìm thấy cột 'user_id' trong file. Vui lòng kiểm tra lại file CSV gốc.")
        return

    df.rename(columns={'seq': 'total_study_time', 'speed': 'avg_speed', 'score': 'avg_score'}, inplace=True)
    
    # 2. Xử lý thiếu dữ liệu cơ bản
    safe_attempts = df['attempts'].replace(0, np.nan)
    df['accuracy_rate'] = (df['is_correct'] / safe_attempts).fillna(0)
    df['total_forum_activity'] = df['cmt_counts'] + df['rep_counts']
    
    for col in WEIGHTS.keys():
        df[col] = df[col].fillna(0)

    # 3. Tính WES
    print("-> Đang tính toán điểm WES...")
    scaler = MinMaxScaler()
    scaled_wes_features = scaler.fit_transform(df[list(WEIGHTS.keys())])
    df_scaled_wes = pd.DataFrame(scaled_wes_features, columns=WEIGHTS.keys())
    
    df['weighted_score'] = 0
    for feature, weight in WEIGHTS.items():
        df['weighted_score'] += df_scaled_wes[feature] * weight

    # 4. Cắt phân vị P33, P66 và Gán nhãn
    print("-> Đang gán nhãn dựa trên phân vị...")
    p33 = df['weighted_score'].quantile(0.33)
    p66 = df['weighted_score'].quantile(0.66)
    
    def assign_label(score):
        if score <= p33: return 'Low_Engagement'
        elif score <= p66: return 'Medium_Engagement'
        else: return 'High_Engagement'
            
    df['target_label'] = df['weighted_score'].apply(assign_label)
    
    # 5. XUẤT FILE ĐÁP ÁN (Chỉ lấy user_id và Nhãn)
    print("-> Đang xuất file đáp án...")
    ground_truth_df = df[['user_id', 'target_label']]
    
    # Xóa các user_id bị trùng lặp (nếu có) để khi merge không bị lỗi
    ground_truth_df = ground_truth_df.drop_duplicates(subset=['user_id'])
    
    ground_truth_df.to_csv(GROUND_TRUTH_FILE, index=False)
    
    print("="*60)
    print(f"✅ HOÀN TẤT! Đã xuất file: {GROUND_TRUTH_FILE}")
    print(ground_truth_df['target_label'].value_counts())
    print("="*60)

if __name__ == "__main__":
    main()