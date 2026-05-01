import gc
import logging
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from pathlib import Path

from config import (
    DEFAULT_OBSERVATION_DAYS,
    FEATURES_COMPAT_FILE,
    FEATURES_WINDOW_FILE,
    GROUND_TRUTH_FILE,
    PRIMARY_KEY,
    RAW_DATA_PARQUET,
    TIME_WINDOW_COMPARE_SUMMARY_FILE,
    TIME_WINDOW_MODE,
)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - [%(levelname)s] - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

OBSERVATION_DAYS = DEFAULT_OBSERVATION_DAYS
WINDOW_MODE = str(TIME_WINDOW_MODE).strip().lower()

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

def build_action_timeline(df: pd.DataFrame) -> pd.DataFrame:
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
    df["days_since_enroll"] = pd.to_numeric(df["days_since_enroll"], errors="coerce")
    df["days_since_enroll"] = df["days_since_enroll"].clip(lower=0)

    logger.info("Đang tính độ dài hành vi tối đa theo từng user để hỗ trợ relative windows...")
    max_days_per_user = df.groupby(PRIMARY_KEY)["days_since_enroll"].max().fillna(0)
    max_days_per_user = max_days_per_user.clip(lower=1)
    df["max_days_per_user"] = df[PRIMARY_KEY].map(max_days_per_user)
    return df


def build_fixed_window(df: pd.DataFrame) -> pd.DataFrame:
    logger.info(f"CHỐT CHẶN LEAK DATA (fixed): Cắt bỏ mọi hành vi sau ngày thứ {OBSERVATION_DAYS}...")
    within_window = (df["days_since_enroll"] <= OBSERVATION_DAYS) | df["days_since_enroll"].isna()
    df_window = df.loc[within_window].copy()
    df_window["window_type"] = "fixed"
    df_window["window_value"] = OBSERVATION_DAYS

    logger.info(f"-> Dữ liệu sau khi cắt thời gian: {len(df_window):,} dòng.")
    return df_window


def build_relative_window(df: pd.DataFrame, fraction: float) -> pd.DataFrame:
    pct_text = int(round(float(fraction) * 100))
    logger.info(f"CHỐT CHẶN LEAK DATA (relative): Cắt theo {pct_text}% timeline của từng user...")

    per_row_limit = (df["max_days_per_user"] * float(fraction)).round().astype(int).clip(lower=1)
    within_window = (df["days_since_enroll"] <= per_row_limit) | df["days_since_enroll"].isna()
    df_window = df.loc[within_window].copy()
    df_window["window_type"] = "relative"
    df_window["window_value"] = pct_text

    logger.info(f"-> Dữ liệu sau khi cắt relative {pct_text}%: {len(df_window):,} dòng.")
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


def finalize_with_labels(features: pd.DataFrame) -> pd.DataFrame:
    labels = pd.read_csv(GROUND_TRUTH_FILE)
    return features.merge(labels, on="user_id", how="inner")


def export_window_dataset(final_df: pd.DataFrame, mode: str, window_value: int) -> str:
    if mode == "fixed":
        output_path = FEATURES_WINDOW_FILE
        final_df.to_csv(output_path, index=False, encoding="utf-8-sig")
        final_df.to_csv(FEATURES_COMPAT_FILE, index=False, encoding="utf-8-sig")
        return str(output_path)

    output_path = Path(str(RELATIVE_FEATURES_OUTPUT_PATTERN).format(pct=window_value))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    final_df.to_csv(output_path, index=False, encoding="utf-8-sig")

    # Relative branch: mặc định chọn 50% để làm file tương thích cho Stage 3 nếu user muốn chạy tiếp.
    if int(window_value) == 50:
        final_df.to_csv(FEATURES_COMPAT_FILE, index=False, encoding="utf-8-sig")

    return str(output_path)


def summarize_outputs(items: list[dict]) -> None:
    summary = pd.DataFrame(items)
    summary.to_csv(TIME_WINDOW_COMPARE_SUMMARY_FILE, index=False, encoding="utf-8-sig")
    logger.info(f"Đã lưu bảng so sánh time-window tại: {TIME_WINDOW_COMPARE_SUMMARY_FILE}")

def main():
    print("=" * 80)
    print(f"STEP 2: EXTRACT EARLY TIME-WINDOW FEATURES ({WINDOW_MODE.upper()})")
    print("=" * 80)

    try:
        df = load_data()
        df = build_action_timeline(df)
        logger.info("Đang dọn dẹp bộ nhớ RAM...")
        summary_rows = []

        if WINDOW_MODE == "relative":
            for frac in RELATIVE_WINDOW_FRACTIONS:
                pct_value = int(round(float(frac) * 100))
                df_window = build_relative_window(df, float(frac))
                features = extract_features(df_window)
                final_df = finalize_with_labels(features)
                output_path = export_window_dataset(final_df, mode="relative", window_value=pct_value)
                summary_rows.append(
                    {
                        "window_mode": "relative",
                        "window_value": pct_value,
                        "num_rows": int(len(df_window)),
                        "num_users": int(final_df[PRIMARY_KEY].nunique()),
                        "output_file": output_path,
                    }
                )
        else:
            df_window = build_fixed_window(df)
            features = extract_features(df_window)
            final_df = finalize_with_labels(features)
            output_path = export_window_dataset(final_df, mode="fixed", window_value=OBSERVATION_DAYS)
            summary_rows.append(
                {
                    "window_mode": "fixed",
                    "window_value": int(OBSERVATION_DAYS),
                    "num_rows": int(len(df_window)),
                    "num_users": int(final_df[PRIMARY_KEY].nunique()),
                    "output_file": output_path,
                }
            )

        summarize_outputs(summary_rows)
        del df
        gc.collect()

        print("=" * 80)
        logger.info("HOÀN TẤT GIAI ĐOẠN 2.")
        logger.info(f"File tương thích cho Stage 3: {FEATURES_COMPAT_FILE}")
        logger.info(f"Số cấu hình time-window đã xuất: {len(summary_rows)}")
        print("=" * 80)

    except Exception as e:
        logger.exception("Đã xảy ra lỗi nghiêm trọng trong quá trình xử lý:")

if __name__ == "__main__":
    main()
