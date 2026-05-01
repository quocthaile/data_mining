from pathlib import Path
import json


BASE_DIR = Path(__file__).resolve().parent
PROJECT_DIR = BASE_DIR.parent
DATASET_DIR = PROJECT_DIR / "dataset"

# Input
# Use combined_all_data.parquet located inside the project's `dataset/` folder
RAW_DATA_PARQUET = DATASET_DIR / "combined_all_data.parquet"
RANDOM_STATE = 42
PRIMARY_KEY = "user_id"

# Time windows suggested for early warning / time series modeling
TIME_WINDOWS_DAYS = [14, 21, 28]
DEFAULT_OBSERVATION_DAYS = 28
# FIXED MODE: Use 28-day fixed observation window (no relative/percentage-based windows)
TIME_WINDOW_MODE = "fixed"

MODEL_DATA_DIR = DATASET_DIR / "model_data"
FEATURES_WINDOW_FILE = DATASET_DIR / f"user_features_{DEFAULT_OBSERVATION_DAYS}days.csv"
FEATURES_COMPAT_FILE = DATASET_DIR / "user_features_and_wes.csv"
RELATIVE_FEATURES_OUTPUT_PATTERN = DATASET_DIR / "user_features_relative_{pct}.csv"
TIME_WINDOW_COMPARE_SUMMARY_FILE = DATASET_DIR / "time_window_comparison.csv"
EXPERIMENTAL_DATASET_FILE = DATASET_DIR / "experimental_dataset.csv"
TRAIN_FILE = MODEL_DATA_DIR / "train_smote.csv"
VALID_FILE = MODEL_DATA_DIR / "valid_original.csv"
# Preprocessed dataset placed at top-level dataset folder per user request
PREPROCESSING_DATASET_FILE = DATASET_DIR / "pre-processing_dataset.csv"
TEST_FILE = MODEL_DATA_DIR / "test_original.csv"
MODEL_OUT_DIR = BASE_DIR / "deployment_models"
IMAGE_OUT_DIR = BASE_DIR / "output_images_3w"
MODEL_BUNDLE_FILE = MODEL_OUT_DIR / "deployment_bundle.pkl"

# Backward compatibility alias
FULL_PREPROCESSED_FILE = PREPROCESSING_DATASET_FILE

# Stage outputs
GROUND_TRUTH_FILE = DATASET_DIR / "ground_truth_labels.csv"
GROUND_TRUTH_REPORT_FILE = DATASET_DIR / "ground_truth_report.csv"

# Labeling / sampling controls
LABELING_STRATEGY = "quantile_rank"
LABEL_PERCENTILES = (0.60, 0.85)
MAX_TRAIN_SAMPLES_PER_CLASS = None
TRAIN_TARGET_TOTAL_SAMPLES = 60000
TRAIN_CLASS_RATIOS = {
    "Low_Engagement": 4.8,
    "Medium_Engagement": 1.8,
    "High_Engagement": 3.5,
}


# Runtime overrides file (written by run_pipeline for interactive runs)
RUNTIME_OVERRIDES_FILE = BASE_DIR / "runtime_overrides.json"


def _apply_runtime_overrides():
    if not RUNTIME_OVERRIDES_FILE.exists():
        return
    try:
        with open(RUNTIME_OVERRIDES_FILE, "r", encoding="utf-8") as f:
            overrides = json.load(f)
    except Exception:
        return

    # Apply simple overrides: if key exists in module globals, replace it
    gl = globals()
    for k, v in overrides.items():
        if k in gl:
            gl[k] = v


_apply_runtime_overrides()

# Weighted Engagement Score (WES)
GROUND_TRUTH_WEIGHTS = {
    "total_study_time": 0.35,
    "avg_score": 0.30,
    "accuracy_rate": 0.20,
    "attempts": 0.10,
    "total_forum_activity": 0.05,
}

# Required raw columns for stage 1
RAW_REQUIRED_COLUMNS_STEP1 = [
    "user_id",
    "attempts",
    "is_correct",
    "score",
    "create_time_x",
    "create_time_y",
]
