import json
import logging
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, 
    roc_auc_score, confusion_matrix
)

from config import IMAGE_OUT_DIR, MODEL_BUNDLE_FILE, MODEL_OUT_DIR, TEST_FILE

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - [%(levelname)s] - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

MODEL_DATA_DIR = Path("model_data_3w")

TRAIN_FILE = MODEL_DATA_DIR / "train_smote.csv"
VALID_FILE = MODEL_DATA_DIR / "valid_original.csv"

LABEL_ENCODER_FILE = MODEL_OUT_DIR / "label_encoder.pkl"
SCHOOL_ENCODER_FILE = MODEL_OUT_DIR / "school_encoder.pkl"
IMPUTER_FILE = MODEL_OUT_DIR / "imputer.pkl"
SCALER_FILE = MODEL_OUT_DIR / "scaler.pkl"

BEST_MODEL_FILE = MODEL_OUT_DIR / "best_model_3w.pkl"
METRICS_FILE = MODEL_OUT_DIR / "evaluation_metrics.csv"

RANDOM_STATE = 42
TARGET_RISK_CLASS = "Low_Engagement"

def load_data_and_artifacts() -> tuple:
    if not TRAIN_FILE.exists() or not VALID_FILE.exists() or not TEST_FILE.exists():
        logger.error("Không tìm thấy dữ liệu. Vui lòng chạy Giai đoạn 3 trước.")
        raise FileNotFoundError("Run step 3 first.")

    logger.info("Đang nạp dữ liệu và LabelEncoder...")
    train_df = pd.read_csv(TRAIN_FILE)
    valid_df = pd.read_csv(VALID_FILE)
    test_df = pd.read_csv(TEST_FILE)
    label_encoder = joblib.load(LABEL_ENCODER_FILE)

    X_train = train_df.drop(columns=["target_label"])
    y_train = train_df["target_label"].astype(int)
    X_valid = valid_df.drop(columns=["target_label"])
    y_valid = valid_df["target_label"].astype(int)
    X_test = test_df.drop(columns=["target_label"])
    y_test = test_df["target_label"].astype(int)

    return X_train, y_train, X_valid, y_valid, X_test, y_test, label_encoder

def safe_name(name: str) -> str:
    return name.replace(" ", "_").replace("/", "_")

def evaluate_model(model, X_test: pd.DataFrame, y_test: pd.Series, label_encoder) -> tuple:
    y_pred = model.predict(X_test)
    labels_list = list(range(len(label_encoder.classes_)))
    try:
        low_idx = int(np.where(label_encoder.classes_ == TARGET_RISK_CLASS)[0][0])
    except IndexError:
        logger.warning(f"Không tìm thấy nhãn {TARGET_RISK_CLASS}, gán mặc định Index=0")
        low_idx = 0

    auc_roc = np.nan
    if hasattr(model, "predict_proba"):
        y_prob = model.predict_proba(X_test)
        try:
            auc_roc = roc_auc_score(y_test, y_prob, multi_class="ovr", labels=labels_list)
        except ValueError:
            auc_roc = np.nan

    precisions = precision_score(y_test, y_pred, labels=labels_list, average=None, zero_division=0)
    recalls = recall_score(y_test, y_pred, labels=labels_list, average=None, zero_division=0)

    metrics = {
        "Accuracy": round(accuracy_score(y_test, y_pred), 4),
        f"Precision_{TARGET_RISK_CLASS}": round(precisions[low_idx], 4),
        f"Recall_{TARGET_RISK_CLASS}": round(recalls[low_idx], 4),
        "AUC_ROC_OVR": round(float(auc_roc), 4) if not np.isnan(auc_roc) else "N/A",
    }
    return metrics, y_pred

def main():
    print("=" * 80)
    print("STEP 4: MODEL TRAINING, BENCHMARKING AND DEPLOYMENT EXPORT")
    print("=" * 80)

    try:
        IMAGE_OUT_DIR.mkdir(parents=True, exist_ok=True)
        MODEL_OUT_DIR.mkdir(parents=True, exist_ok=True)

        X_train, y_train, X_valid, y_valid, X_test, y_test, label_encoder = load_data_and_artifacts()

        logger.info("[1/4] Đang khởi tạo các thuật toán Học máy...")
        models = {
            "Logistic Regression": LogisticRegression(max_iter=1000, random_state=RANDOM_STATE),
            "Linear SVC": LinearSVC(dual=False, random_state=RANDOM_STATE),
            "Decision Tree": DecisionTreeClassifier(max_depth=5, random_state=RANDOM_STATE),
            "Random Forest": RandomForestClassifier(
                n_estimators=200, max_depth=10, class_weight="balanced", 
                random_state=RANDOM_STATE, n_jobs=-1
            ),
            "XGBoost": XGBClassifier(
                objective="multi:softprob", eval_metric="mlogloss",
                max_depth=5, n_estimators=200, learning_rate=0.08,
                random_state=RANDOM_STATE, n_jobs=-1
            ),
        }

        logger.info("[2/4] Bắt đầu quá trình Huấn luyện và Benchmark...")
        results = []
        fitted_models = {}
        
        for name, model in models.items():
            logger.info(f"   -> Đang huấn luyện: {name} ...")
            model.fit(X_train, y_train)
            fitted_models[name] = model
            
            valid_metrics, y_valid_pred = evaluate_model(model, X_valid, y_valid, label_encoder)
            valid_metrics["Model"] = name
            results.append(valid_metrics)

        metrics_df = pd.DataFrame(results)
        cols = ["Model", "Accuracy", f"Recall_{TARGET_RISK_CLASS}", f"Precision_{TARGET_RISK_CLASS}", "AUC_ROC_OVR"]
        metrics_df = metrics_df[cols]
        metrics_df.to_csv(METRICS_FILE, index=False, encoding="utf-8-sig")
        
        print("\n" + "=" * 90)
        print("BẢNG XẾP HẠNG THUẬT TOÁN (Tối ưu cho Cảnh báo sớm)")
        print("=" * 90)
        print(metrics_df.to_string(index=False))
        print("=" * 90 + "\n")

        ranked = metrics_df.sort_values(
            by=[f"Recall_{TARGET_RISK_CLASS}", "Accuracy"], ascending=False
        )
        best_model_name = ranked.iloc[0]["Model"]
        best_model = fitted_models[best_model_name]
        logger.info(f"Thuật toán chiến thắng: {best_model_name}")

        logger.info("Đang chấm điểm final trên test cho mô hình đã chọn...")
        test_metrics, test_pred = evaluate_model(best_model, X_test, y_test, label_encoder)
        test_metrics["Model"] = best_model_name
        test_metrics_df = pd.DataFrame([test_metrics])
        test_metrics_df.to_csv(MODEL_OUT_DIR / "final_test_metrics.csv", index=False, encoding="utf-8-sig")

        cm = confusion_matrix(y_test, test_pred, labels=list(range(len(label_encoder.classes_))))
        plt.figure(figsize=(7, 5))
        sns.heatmap(
            cm, annot=True, fmt="d", cmap="Greens",
            xticklabels=label_encoder.classes_,
            yticklabels=label_encoder.classes_
        )
        plt.title(f"Ma trận Nhầm lẫn Test - {best_model_name}")
        plt.xlabel("Dự đoán")
        plt.ylabel("Thực tế")
        plt.tight_layout()
        plt.savefig(IMAGE_OUT_DIR / f"CM_TEST_{safe_name(best_model_name)}.png", dpi=200)
        plt.close()

        logger.info("[4/4] Đang đóng gói toàn bộ Pipeline để chuẩn bị Triển khai...")
        deployment_bundle = {
            "model": best_model,
            "model_name": best_model_name,
            "label_encoder": label_encoder,
            "school_encoder": joblib.load(SCHOOL_ENCODER_FILE),
            "imputer": joblib.load(IMPUTER_FILE),
            "scaler": joblib.load(SCALER_FILE),
            "feature_columns": list(X_train.columns),
            "target_labels": label_encoder.classes_.tolist()
        }

        joblib.dump(best_model, BEST_MODEL_FILE)
        joblib.dump(deployment_bundle, MODEL_BUNDLE_FILE)

        logger.info(f"✅ HOÀN TẤT GIAI ĐOẠN 4!")
        logger.info(f"   Bundle triển khai lưu tại: {MODEL_BUNDLE_FILE}")

    except Exception as e:
        logger.exception("Đã xảy ra lỗi nghiêm trọng ở Giai đoạn 4:")

if __name__ == "__main__":
    main()
