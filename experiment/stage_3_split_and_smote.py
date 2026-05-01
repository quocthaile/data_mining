import logging
import joblib
import numpy as np
import pandas as pd
from collections import Counter
from pathlib import Path
from imblearn.over_sampling import SMOTE
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, OrdinalEncoder, StandardScaler

from config import (
    GROUND_TRUTH_FILE,
    PRIMARY_KEY,
    RANDOM_STATE,
    MAX_TRAIN_SAMPLES_PER_CLASS,
    TRAIN_CLASS_RATIOS,
    TRAIN_TARGET_TOTAL_SAMPLES,
)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - [%(levelname)s] - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

FEATURES_AND_LABELS_FILE = Path("user_features_and_wes.csv")

MODEL_DATA_DIR = Path("model_data_3w")
MODEL_OUT_DIR = Path("deployment_models")

TRAIN_FILE = MODEL_DATA_DIR / "train_smote.csv"
VALID_FILE = MODEL_DATA_DIR / "valid_original.csv"
TEST_FILE = MODEL_DATA_DIR / "test_original.csv"
FULL_PREPROCESSED_FILE = MODEL_DATA_DIR / "full_preprocessed.csv"

LABEL_ENCODER_FILE = MODEL_OUT_DIR / "label_encoder.pkl"
SCHOOL_ENCODER_FILE = MODEL_OUT_DIR / "school_encoder.pkl"
IMPUTER_FILE = MODEL_OUT_DIR / "imputer.pkl"
SCALER_FILE = MODEL_OUT_DIR / "scaler.pkl"


TARGET_LABELS_ORDER = ["Low_Engagement", "Medium_Engagement", "High_Engagement"]
NUMERIC_FEATURES = ["attempts_3w", "is_correct_3w", "score_3w", "accuracy_rate_3w", "num_courses", "age"]
FINAL_FEATURES = ["school_encoded"] + NUMERIC_FEATURES

def main():
    print("=" * 80)
    print("STEP 3: ADVANCED SPLITTING (8:1:1), K-MEANS LABELING & SMOTE")
    print("=" * 80)

    try:
        MODEL_DATA_DIR.mkdir(parents=True, exist_ok=True)
        MODEL_OUT_DIR.mkdir(parents=True, exist_ok=True)

        logger.info("[1/6] Đang tải dữ liệu đã ghép nhãn từ stage 2...")
        if not FEATURES_AND_LABELS_FILE.exists():
            raise FileNotFoundError(f"Missing features file: {FEATURES_AND_LABELS_FILE}")
        if not GROUND_TRUTH_FILE.exists():
            raise FileNotFoundError(f"Missing ground truth file: {GROUND_TRUTH_FILE}")

        df = pd.read_csv(FEATURES_AND_LABELS_FILE)
        if "target_label" not in df.columns:
            labels = pd.read_csv(GROUND_TRUTH_FILE)
            df = df.merge(labels, on=PRIMARY_KEY, how="inner")

        required_columns = [PRIMARY_KEY, "school", "year_of_birth", "gender", "num_courses", "attempts_3w", "is_correct_3w", "score_3w", "accuracy_rate_3w", "target_label"]
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            raise KeyError(f"Missing required columns: {missing_columns}")

        df["year_of_birth"] = pd.to_numeric(df["year_of_birth"], errors="coerce")
        current_year = pd.Timestamp.now().year
        if "age" not in df.columns:
            df["age"] = (current_year - df["year_of_birth"]).clip(lower=10, upper=100)
        else:
            df["age"] = pd.to_numeric(df["age"], errors="coerce")
        for col in NUMERIC_FEATURES:
            if col not in df.columns:
                df[col] = np.nan
            df[col] = pd.to_numeric(df[col], errors="coerce")
        df[NUMERIC_FEATURES] = df[NUMERIC_FEATURES].replace([np.inf, -np.inf], np.nan)
        df["school"] = df["school"].fillna("Unknown").astype(str).str.strip()
        df["gender"] = df["gender"].fillna("Unknown").astype(str).str.strip()
        df = df.dropna(subset=["target_label"]).reset_index(drop=True)

        logger.info("[2/6] Đang chia train/valid/test theo nhãn thật (stratified trên target_label)...")
        df_train, df_temp = train_test_split(
            df,
            test_size=0.2,
            random_state=RANDOM_STATE,
            stratify=df["target_label"],
        )
        df_valid, df_test = train_test_split(
            df_temp,
            test_size=0.5,
            random_state=RANDOM_STATE,
            stratify=df_temp["target_label"],
        )

        logger.info(f"   -> Size: Train ({len(df_train)}), Valid ({len(df_valid)}), Test ({len(df_test)})")

        logger.info("[3/6] Đang fit label encoder và preprocessing CHỈ trên train...")
        label_encoder = LabelEncoder()
        label_encoder.fit(TARGET_LABELS_ORDER)

        y_train = label_encoder.transform(df_train["target_label"])
        y_valid = label_encoder.transform(df_valid["target_label"])
        y_test = label_encoder.transform(df_test["target_label"])

        school_encoder = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)
        school_encoder.fit(df_train[["school"]])

        for frame in [df_train, df_valid, df_test]:
            frame["school_encoded"] = school_encoder.transform(frame[["school"]])

        X_train = df_train[FINAL_FEATURES].copy()
        X_valid = df_valid[FINAL_FEATURES].copy()
        X_test = df_test[FINAL_FEATURES].copy()

        imputer = SimpleImputer(strategy="median")
        scaler = StandardScaler()

        X_train_imputed = imputer.fit_transform(X_train)
        X_valid_imputed = imputer.transform(X_valid)
        X_test_imputed = imputer.transform(X_test)

        X_train_processed = scaler.fit_transform(X_train_imputed)
        X_valid_processed = scaler.transform(X_valid_imputed)
        X_test_processed = scaler.transform(X_test_imputed)

        logger.info("[4/6] Đang điều chỉnh phân phối train theo tỷ lệ mục tiêu 4.8:1.8:3.5...")
        logger.info(f"   -> Train trước điều chỉnh: {len(df_train):,} mẫu")
        logger.info(f"   -> Phân phối train trước điều chỉnh: {df_train['target_label'].value_counts().to_dict()}")

        target_total = TRAIN_TARGET_TOTAL_SAMPLES if TRAIN_TARGET_TOTAL_SAMPLES else len(df_train)
        ratio_sum = float(sum(TRAIN_CLASS_RATIOS.values()))
        target_counts = {
            label: max(1, int(round(target_total * weight / ratio_sum)))
            for label, weight in TRAIN_CLASS_RATIOS.items()
        }
        diff = target_total - sum(target_counts.values())
        if diff != 0:
            first_label = next(iter(TRAIN_CLASS_RATIOS))
            target_counts[first_label] += diff

        sampled_frames = []
        for label, target_count in target_counts.items():
            class_frame = df_train[df_train["target_label"] == label]
            if class_frame.empty:
                raise ValueError(f"Class '{label}' has no samples in train split.")
            replace = len(class_frame) < target_count
            sampled_frames.append(
                class_frame.sample(n=target_count, replace=replace, random_state=RANDOM_STATE)
            )

        df_train = (
            pd.concat(sampled_frames, axis=0)
            .sample(frac=1.0, random_state=RANDOM_STATE)
            .reset_index(drop=True)
        )
        logger.info(f"   -> Train sau điều chỉnh: {len(df_train):,} mẫu")
        logger.info(f"   -> Target counts: {target_counts}")
        logger.info(f"   -> Phân phối train sau điều chỉnh: {df_train['target_label'].value_counts().to_dict()}")

        y_train = label_encoder.transform(df_train["target_label"])
        X_train = df_train[FINAL_FEATURES].copy()
        X_train_imputed = imputer.fit_transform(X_train)
        X_train_processed = scaler.fit_transform(X_train_imputed)
        X_valid_processed = scaler.transform(imputer.transform(X_valid))
        X_test_processed = scaler.transform(imputer.transform(X_test))

        X_train_smote = X_train_processed
        y_train_smote = y_train

        logger.info("[5/6] Đang lưu các tập dữ liệu và artifacts tiền xử lý...")
        train_final = pd.DataFrame(X_train_smote, columns=FINAL_FEATURES)
        train_final["target_label"] = y_train_smote
        train_final.to_csv(TRAIN_FILE, index=False, encoding="utf-8-sig")

        valid_final = pd.DataFrame(X_valid_processed, columns=FINAL_FEATURES)
        valid_final["target_label"] = y_valid
        valid_final.to_csv(VALID_FILE, index=False, encoding="utf-8-sig")

        test_final = pd.DataFrame(X_test_processed, columns=FINAL_FEATURES)
        test_final["target_label"] = y_test
        test_final.to_csv(TEST_FILE, index=False, encoding="utf-8-sig")

        df_full = df.copy()
        df_full["school_encoded"] = school_encoder.transform(df_full[["school"]])
        full_preprocessed = pd.concat(
            [
                pd.DataFrame(scaler.transform(imputer.transform(df_full[FINAL_FEATURES])), columns=FINAL_FEATURES),
                df_full[[PRIMARY_KEY, "gender", "target_label"]].reset_index(drop=True),
            ],
            axis=1,
        )
        full_preprocessed.to_csv(FULL_PREPROCESSED_FILE, index=False, encoding="utf-8-sig")

        joblib.dump(label_encoder, LABEL_ENCODER_FILE)
        joblib.dump(school_encoder, SCHOOL_ENCODER_FILE)
        joblib.dump(imputer, IMPUTER_FILE)
        joblib.dump(scaler, SCALER_FILE)

        print("=" * 80)
        logger.info("✅ HOÀN TẤT! Stage 3 đã split trước, fit trên train và xuất đủ artifacts.")
        print("=" * 80)

    except Exception as e:
        logger.exception("Đã xảy ra lỗi:")

if __name__ == "__main__":
    main()
