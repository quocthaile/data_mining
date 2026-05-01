import gc
import logging
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from pathlib import Path

from config import (
    DEFAULT_OBSERVATION_DAYS,
    GROUND_TRUTH_FILE,
    PRIMARY_KEY,
    RAW_DATA_PARQUET,
)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - [%(levelname)s] - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

FEATURES_WINDOW_FILE = Path(f"user_features_{DEFAULT_OBSERVATION_DAYS}days.csv")
FEATURES_COMPAT_FILE = Path("user_features_and_wes.csv")
OBSERVATION_DAYS = DEFAULT_OBSERVATION_DAYS

RAW_REQUIRED_COLUMNS_STEP2 = [
    'user_id', 'enroll_time', 'submit_time', 'create_time_x', 'create_time_y',
    'school', 'year_of_birth', 'gender', 'num_courses', 'attempts', 'is_correct', 'score'
]

def parse_datetime(series: pd.Series) -> pd.Series:
    extracted = series.astype(str).str.extract(r"(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2})")[0]
    return pd.to_datetime(extracted, errors="coerce")

def load_data() -> pd.DataFrame:
    if not RAW_DATA_PARQUET.exists():
        logger.error(f"Không tìm thấy file dữ liệu gốc: {RAW_DATA_PARQUET}")
        raise FileNotFoundError(f"Input parquet not found: {RAW_DATA_PARQUET}")
        
    if not GROUND_TRUTH_FILE.exists():
        logger.error(f"Không tìm thấy file Nhãn. Bạn phải chạy Bước 1 trước: {GROUND_TRUTH_FILE}")
        raise FileNotFoundError(f"Run step 1 first: {GROUND_TRUTH_FILE}")

    logger.info("Đang kiểm tra schema (cấu trúc) của file Parquet...")
    available_columns = set(pq.ParquetFile(RAW_DATA_PARQUET).schema.names)
    selected_columns = [col for col in RAW_REQUIRED_COLUMNS_STEP2 if col in available_columns]
    
    missing_columns = sorted(set(RAW_REQUIRED_COLUMNS_STEP2) - available_columns)
    if missing_columns:
        logger.warning(f"File Parquet bị thiếu các cột: {', '.join(missing_columns)}")

    logger.info(f"Đang tải {len(selected_columns)} cột Profile và Hành vi...")
    df = pd.read_parquet(RAW_DATA_PARQUET, columns=selected_columns)
    logger.info(f"-> Đã tải thành công: {len(df):,} dòng dữ liệu thô.")
    return df

def build_time_window(df: pd.DataFrame) -> pd.DataFrame:
    logger.info("Đang xây dựng bộ đếm thời gian hành động (Action Time)...")
    for col in ["submit_time", "create_time_x", "create_time_y"]:
        if col not in df.columns:
            df[col] = pd.NaT

    df["action_time"] = df["submit_time"].combine_first(df["create_time_x"]).combine_first(df["create_time_y"])
    df["action_time"] = parse_datetime(df["action_time"])

    logger.info("Đang xác định thời điểm bắt đầu khóa học (Enroll Time)...")
    if "enroll_time" in df.columns:
        df["enroll_time"] = parse_datetime(df["enroll_time"])
    else:
        df["enroll_time"] = df.groupby("user_id")["action_time"].transform("min")

    logger.info("Đang tính toán số ngày kể từ lúc bắt đầu học...")
    df["days_since_enroll"] = (df["action_time"] - df["enroll_time"]).dt.days

    logger.info(f"CHỐT CHẶN LEAK DATA: Cắt bỏ mọi hành vi sau ngày thứ {OBSERVATION_DAYS}...")
    within_window = (df["days_since_enroll"] <= OBSERVATION_DAYS) | df["days_since_enroll"].isna()
    df_window = df.loc[within_window].copy()
    
    logger.info(f"-> Dữ liệu sau khi cắt thời gian: {len(df_window):,} dòng.")
    return df_window

def extract_features(df_window: pd.DataFrame) -> pd.DataFrame:
    logger.info("Đang tổng hợp Đặc trưng khởi đầu (Early Features) cho từng sinh viên...")
    
    for col in ["attempts", "is_correct", "score", "num_courses"]:
        if col not in df_window.columns:
            df_window[col] = 0
        df_window[col] = pd.to_numeric(df_window[col], errors="coerce").fillna(0)
        
    for col in ["school", "year_of_birth", "gender"]:
        if col not in df_window.columns:
            df_window[col] = np.nan

    features = (
        df_window.groupby(PRIMARY_KEY)
        .agg(
            school=("school", "first"),
            year_of_birth=("year_of_birth", "first"),
            gender=("gender", "first"),
            num_courses=("num_courses", "first"),
            attempts_3w=("attempts", "sum"),
            is_correct_3w=("is_correct", "sum"),
            score_3w=("score", "mean"),
        )
        .reset_index()
    )
    
    features["accuracy_rate_3w"] = (
        features["is_correct_3w"] / features["attempts_3w"].replace(0, np.nan)
    ).fillna(0)

    logger.info(f"-> Đã tổng hợp đặc trưng cho {len(features):,} sinh viên.")
    return features

def main():
    print("=" * 80)
    print(f"STEP 2: EXTRACT EARLY TIME-WINDOW FEATURES ({OBSERVATION_DAYS} DAYS)")
    print("=" * 80)

    try:
        df = load_data()
        df_window = build_time_window(df)
        logger.info("Đang dọn dẹp bộ nhớ RAM...")
        del df
        gc.collect()
        features = extract_features(df_window)
        logger.info("Đang nối (Merge) Đặc trưng cửa sổ thời gian với Nhãn toàn khóa...")
        labels = pd.read_csv(GROUND_TRUTH_FILE)
        final_df = features.merge(labels, on="user_id", how="inner")
        logger.info("Đang lưu file kết quả...")
        final_df.to_csv(FEATURES_WINDOW_FILE, index=False, encoding="utf-8-sig")
        final_df.to_csv(FEATURES_COMPAT_FILE, index=False, encoding="utf-8-sig")

        print("=" * 80)
        logger.info(f"✅ HOÀN TẤT GIAI ĐOẠN 2! Đã lưu file Đặc trưng: {FEATURES_WINDOW_FILE}")
        logger.info(f"✅ Đã lưu file tương thích: {FEATURES_COMPAT_FILE}")
        logger.info(f"   Tổng số sinh viên có trong tập huấn luyện: {len(final_df):,}")
        print("=" * 80)

    except Exception as e:
        logger.exception("Đã xảy ra lỗi nghiêm trọng trong quá trình xử lý:")

if __name__ == "__main__":
    main()
