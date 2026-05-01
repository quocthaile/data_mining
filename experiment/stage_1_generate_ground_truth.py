import gc
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

from config import (
    DEFAULT_OBSERVATION_DAYS,
    GROUND_TRUTH_FILE,
    GROUND_TRUTH_REPORT_FILE,
    GROUND_TRUTH_WEIGHTS,
    PRIMARY_KEY,
    RANDOM_STATE,
    RAW_DATA_PARQUET,
    RAW_REQUIRED_COLUMNS_STEP1,
    LABELING_STRATEGY,
)

def main():
    print("=" * 80)
    print("STEP 1: GENERATE GROUND TRUTH LABELS (FULL COURSE BEHAVIOR)")
    print("=" * 80)

    if not RAW_DATA_PARQUET.exists():
        raise FileNotFoundError(f"Input parquet not found: {RAW_DATA_PARQUET}")

    print("[1/4] Đang tải dữ liệu hành vi thô (Toàn khóa)...")
    df = pd.read_parquet(RAW_DATA_PARQUET, columns=RAW_REQUIRED_COLUMNS_STEP1)
    raw_row_count = len(df)
    raw_user_count = df[PRIMARY_KEY].nunique()

    if PRIMARY_KEY not in df.columns:
        raise KeyError(f"Missing primary key column: {PRIMARY_KEY}")

    for col in ["attempts", "is_correct", "score"]:
        df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)

    df["video_clicks"] = df["create_time_x"].notna().astype(int)
    df["forum_posts"] = df["create_time_y"].notna().astype(int)

    print("[2/4] Đang tổng hợp hành vi của từng sinh viên...")
    user_features = (
        df.groupby("user_id")
        .agg(
            attempts=("attempts", "sum"),
            is_correct=("is_correct", "sum"),
            avg_score=("score", "mean"),
            total_study_time=("video_clicks", "sum"),
            total_forum_activity=("forum_posts", "sum"),
        )
        .reset_index()
    )

    del df
    gc.collect()

    user_features["accuracy_rate"] = (
        user_features["is_correct"] / user_features["attempts"].replace(0, np.nan)
    ).fillna(0)
    
    model_columns = list(GROUND_TRUTH_WEIGHTS.keys())
    user_features[model_columns] = (
        user_features[model_columns].replace([np.inf, -np.inf], np.nan).fillna(0)
    )

    print("[3/4] Đang tính toán điểm tương tác trọng số (WES)...")
    scaler = MinMaxScaler()
    scaled = pd.DataFrame(
        scaler.fit_transform(user_features[model_columns]),
        columns=model_columns,
        index=user_features.index,
    )
    user_features["weighted_score"] = sum(
        scaled[col] * weight for col, weight in GROUND_TRUTH_WEIGHTS.items()
    )

    print("[4/4] Đang gán nhãn theo phân vị để cân bằng lớp...")
    label_order = ["Low_Engagement", "Medium_Engagement", "High_Engagement"]
    rank_scores = user_features["weighted_score"].rank(method="first", ascending=True)

    if LABELING_STRATEGY == "quantile_rank":
        user_features["target_label"] = pd.qcut(
            rank_scores,
            q=3,
            labels=label_order,
        )
    else:
        user_features["target_label"] = pd.qcut(
            rank_scores,
            q=3,
            labels=label_order,
        )
    user_features["target_label"] = user_features["target_label"].astype(str)

    report = pd.DataFrame(
        {
            "metric": [
                "labeling_strategy",
                "raw_rows",
                "raw_users",
                "observation_days_hint",
                "labeled_users",
                "low_engagement_users",
                "medium_engagement_users",
                "high_engagement_users",
            ],
            "value": [
                LABELING_STRATEGY,
                raw_row_count,
                raw_user_count,
                DEFAULT_OBSERVATION_DAYS,
                len(user_features),
                int((user_features["target_label"] == "Low_Engagement").sum()),
                int((user_features["target_label"] == "Medium_Engagement").sum()),
                int((user_features["target_label"] == "High_Engagement").sum()),
            ],
        }
    )

    user_features[["user_id", "target_label"]].to_csv(
        GROUND_TRUTH_FILE, index=False, encoding="utf-8-sig"
    )
    report.to_csv(GROUND_TRUTH_REPORT_FILE, index=False, encoding="utf-8-sig")
    print(f"\nDa luu file nhan muc tieu: {GROUND_TRUTH_FILE}")
    print(f"Da luu report stage 1: {GROUND_TRUTH_REPORT_FILE}")
    
    print("\nPhân phối nhãn thực tế:")
    distribution = (
        user_features["target_label"]
        .value_counts(normalize=True)
        .reindex(label_order)
        .mul(100)
        .round(2)
    )
    print(distribution.to_string() + " %")

if __name__ == "__main__":
    main()
