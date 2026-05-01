#!/usr/bin/env python3
"""
Script chính cho đồ án Khai phá Dữ liệu - DS317
Dự đoán mức độ tham gia của học viên trong hệ thống MOOC

Kịch bản thực nghiệm:
1. Làm sạch và chuẩn hóa dữ liệu nguồn
2. Trích xuất đặc trưng chuỗi thời gian từ dữ liệu hoạt động học viên
3. Khám phá và phân tích dữ liệu (EDA)
4. Khởi tạo nhãn ground-truth bằng K-Means và xác thực nhãn
5. Chia tập train/valid/test và xử lý mất cân bằng dữ liệu
6. Huấn luyện mô hình supervised với tối ưu hóa
7. Đánh giá độ đo và báo cáo kết quả
8. Phân tích khả năng giải thích mô hình

Sử dụng: python main_experiment.py --phase all
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional


@dataclass
class ExperimentConfig:
    """Cấu hình cho toàn bộ thực nghiệm đồ án"""
    project_root: Path
    scripts_dir: Path
    dataset_dir: Path
    results_dir: Path
    user_input: Path
    translated_user: Path
    translate_summary: Path
    combined_file: Path
    weekly_file: Path
    db_file: Path
    combined_parquet: Optional[Path]
    phase: str
    skip_translate: bool
    clusters: int
    q_low: float
    q_high: float
    split_strategy: str
    imbalance_method: str
    label_column: str
    group_column: str
    time_column: str
    time_fallback_column: str
    valid_size: float
    test_size: float
    seed: int
    seed_trials: int
    smote_k_neighbors: int
    phase4_models: str
    phase4_primary_metric: str
    phase4_cv_folds: int
    phase4_n_jobs: int
    phase4_feature_columns: Optional[str]
    phase5_selection_metric: str
    phase5_top_features: int
    phase5_auc_threshold: float
    phase5_recall_low_threshold: float
    phase6_local_error_samples: int # Renamed from phase8_ to phase6_ for consistency with original code
    phase6_local_correct_samples: int # Renamed from phase8_ to phase6_ for consistency with original code
    phase6_top_features: int # Renamed from phase8_ to phase6_ for consistency with original code
    silhouette_sample_size: int
    top_users: int
    min_school_size: int
    top_schools: int
    log_every: int
    max_rows: Optional[int]


def now_text() -> str:
    """Trả về thời gian hiện tại dưới dạng chuỗi"""
    return time.strftime("%Y-%m-%d %H:%M:%S")


def log(message: str) -> None:
    """Ghi log với timestamp"""
    print(f"[{now_text()}] {message}")


def resolve_path_arg(path_value: Path, project_root: Path, default_base: Path) -> Path:
    """Giải quyết đường dẫn tương đối"""
    if path_value.is_absolute():
        return path_value.resolve()
    if path_value.parent == Path("."):
        return (default_base / path_value).resolve()
    return (project_root / path_value).resolve()


def run_command(command: List[str], cwd: Path, label: str) -> None:
    """Chạy lệnh con và kiểm tra lỗi"""
    log(f"Đang chạy {label}")
    proc = subprocess.run(command, cwd=str(cwd), check=False)
    if proc.returncode != 0:
        raise RuntimeError(f"{label} thất bại với mã thoát {proc.returncode}")


def run_phase_1(cfg: ExperimentConfig) -> None:
    """Phase 1: Làm sạch dữ liệu"""
    phase1_dir = cfg.results_dir / "phase1"
    phase_1 = cfg.scripts_dir / "phase_1_data_cleaning.py"

    cmd1 = [
        sys.executable,
        str(phase_1),
        "--dataset-dir",
        str(cfg.dataset_dir),
        "--output-dir",
        str(phase1_dir),
        "--user-input",
        str(cfg.user_input),
        "--translated-user",
        str(cfg.translated_user),
        "--translate-summary",
        str(cfg.translate_summary),
        "--log-every",
        str(cfg.log_every),
    ]
    if cfg.skip_translate:
        cmd1.append("--skip-translate")
    if cfg.max_rows is not None:
        cmd1.extend(["--max-rows", str(cfg.max_rows)])
    run_command(cmd1, cfg.project_root, "Phase 1 - Làm sạch dữ liệu")

def run_phase_2(cfg: ExperimentConfig) -> None:
    """Phase 2: Trích xuất đặc trưng chuỗi thời gian"""
    phase2_dir = cfg.results_dir / "phase2"
    phase_2 = cfg.scripts_dir / "phase_2_data_transformation.py"

    cmd2 = [
        sys.executable,
        str(phase_2),
        "--dataset-dir",
        str(cfg.dataset_dir),
        "--output-dir",
        str(phase2_dir),
        "--translated-user",
        str(cfg.user_input),
        "--combined-file",
        str(cfg.combined_file),
        "--weekly-file",
        str(cfg.weekly_file),
        "--db-file",
        str(cfg.db_file),
        "--log-every",
        str(cfg.log_every),
    ]
    if cfg.combined_parquet is not None:
        cmd2.extend(["--combined-parquet", str(cfg.combined_parquet)])
    if cfg.max_rows is not None:
        cmd2.extend(["--max-rows", str(cfg.max_rows)])
    run_command(cmd2, cfg.project_root, "Phase 2 - Trích xuất đặc trưng chuỗi thời gian")

def run_phase_3(cfg: ExperimentConfig) -> None:
    """Phase 3: Khám phá và phân tích dữ liệu (EDA)"""
    phase3_dir = cfg.results_dir / "phase3"
    phase_3 = cfg.scripts_dir / "phase_3_eda.py"

    cmd3 = [
        sys.executable,
        str(phase_3),
        "--results-dir",
        str(phase3_dir),
        "--combined-input",
        str(cfg.combined_file),
        "--missing-threshold",
        str(cfg.missing_threshold),
        "--outlier-iqr-multiplier",
        str(cfg.outlier_iqr_multiplier),
        "--log-every",
        str(cfg.log_every),
    ]
    if cfg.max_rows is not None:
        cmd3.extend(["--max-rows", str(cfg.max_rows)])
    run_command(cmd3, cfg.project_root, "Phase 3 - Khám phá và phân tích dữ liệu (EDA)")


def run_phase_4(cfg: ExperimentConfig) -> None:
    """Phase 4: Khởi tạo nhãn bằng K-Means và xác thực"""
    phase4_dir = cfg.results_dir / "phase4"
    phase_4 = cfg.scripts_dir / "phase_4_data_labeling.py"

    cmd2 = [
        sys.executable,
        str(phase_4),
        "--results-dir",
        str(phase4_dir),
        "--combined-input",
        str(cfg.results_dir / "phase2" / "combined_user_metrics.csv"),
        "--weekly-input",
        str(cfg.results_dir / "phase2" / "step2_user_week_activity.csv"),
        "--clusters",
        str(cfg.clusters),
        "--q-low",
        str(cfg.q_low),
        "--q-high",
        str(cfg.q_high),
        "--silhouette-sample-size",
        str(cfg.silhouette_sample_size),
        "--top-users",
        str(cfg.top_users),
        "--min-school-size",
        str(cfg.min_school_size),
        "--top-schools",
        str(cfg.top_schools),
        "--batch-size",
        str(cfg.batch_size),
        "--log-every",
        str(cfg.log_every),
    ]
    if cfg.max_rows is not None:
        cmd2.extend(["--max-rows", str(cfg.max_rows)])
    run_command(cmd2, cfg.project_root, "Phase 4 - K-Means và xác thực nhãn")


def run_phase_5(cfg: ExperimentConfig) -> None:
    """Phase 5: Chia tập dữ liệu và xử lý mất cân bằng"""
    phase5_dir = cfg.results_dir / "phase5"
    phase_5 = cfg.scripts_dir / "phase_5_data_splitting.py"
    step5_file = (cfg.results_dir / "phase4" / "step5_standard_labels_kmeans.csv").resolve()

    cmd3 = [
        sys.executable,
        str(phase_5),
        "--results-dir",
        str(phase5_dir),
        "--output-dir",
        str(phase5_dir),
        "--input",
        str(step5_file),
        "--combined-input",
        str(cfg.results_dir / "phase2" / "combined_user_metrics.csv"),
        "--label-column",
        str(cfg.label_column),
        "--split-strategy",
        str(cfg.split_strategy),
        "--group-column",
        str(cfg.group_column),
        "--time-column",
        str(cfg.time_column),
        "--time-fallback-column",
        str(cfg.time_fallback_column),
        "--valid-size",
        str(cfg.valid_size),
        "--test-size",
        str(cfg.test_size),
        "--seed",
        str(cfg.seed),
        "--seed-trials",
        str(cfg.seed_trials),
        "--imbalance-method",
        str(cfg.imbalance_method),
        "--smote-k-neighbors",
        str(cfg.smote_k_neighbors),
        "--log-every",
        str(cfg.log_every),
    ]
    if cfg.max_rows is not None:
        cmd3.extend(["--max-rows", str(cfg.max_rows)])

    run_command(cmd3, cfg.project_root, "Phase 5 - Chia tập train/valid/test và xử lý mất cân bằng")


def run_phase_6(cfg: ExperimentConfig) -> None:
    """Phase 6: Huấn luyện mô hình supervised"""
    phase6_dir = cfg.results_dir / "phase6"
    phase5_dir = cfg.results_dir / "phase5" # Input from Phase 5
    phase_6 = cfg.scripts_dir / "phase_6_model_training.py" # Script for Phase 6
    phase5_train_modeling = (phase5_dir / "phase5_train_modeling.csv").resolve()
    phase5_valid = (phase5_dir / "phase5_valid.csv").resolve()
    phase5_test = (phase5_dir / "phase5_test.csv").resolve()

    cmd4 = [
        sys.executable,
        str(phase_6),
        "--results-dir",
        str(phase6_dir),
        "--output-dir",
        str(phase6_dir),
        "--train-input",
        str(phase5_train_modeling),
        "--valid-input",
        str(phase5_valid),
        "--test-input",
        str(phase5_test),
        "--label-column",
        str(cfg.label_column),
        "--models",
        str(cfg.phase4_models),
        "--primary-metric", # Renamed to phase6_primary_metric
        str(cfg.phase6_primary_metric), # Renamed to phase6_primary_metric
        "--cv-folds", # Renamed to phase6_cv_folds
        str(cfg.phase6_cv_folds), # Renamed to phase6_cv_folds
        "--seed",
        str(cfg.seed),
        "--n-jobs",
        str(cfg.phase6_n_jobs), # Renamed to phase6_n_jobs
        "--log-every",
        str(cfg.log_every),
    ]
    if cfg.phase6_feature_columns is not None: # Renamed to phase6_feature_columns
        cmd4.extend(["--feature-columns", str(cfg.phase6_feature_columns)]) # Renamed to phase6_feature_columns
    if cfg.max_rows is not None:
        cmd4.extend(["--max-rows", str(cfg.max_rows)])

    run_command(cmd4, cfg.project_root, "Phase 6 - Huấn luyện mô hình supervised và tối ưu hóa")


def run_phase_7(cfg: ExperimentConfig) -> None:
    """Phase 7: Đánh giá độ đo và báo cáo"""
    phase7_dir = cfg.results_dir / "phase7" # Output directory for Phase 7
    phase6_dir = cfg.results_dir / "phase6" # Input from Phase 6
    phase5_dir = cfg.results_dir / "phase5"
    phase4_dir = cfg.results_dir / "phase4"
    phase_7 = cfg.scripts_dir / "phase_7_model_evaluation.py"

    cmd5b = [
        sys.executable,
        str(phase_7),
        "--results-dir",
        str(phase7_dir),
        "--output-dir",
        str(phase7_dir),
        "--model-comparison-input",
        str(phase6_dir / "phase6_model_comparison.csv"),
        "--class-metrics-input",
        str(phase6_dir / "phase6_classification_metrics.csv"),
        "--confusion-input",
        str(phase6_dir / "phase6_confusion_matrix.csv"),
        "--feature-importance-input",
        str(phase6_dir / "phase6_feature_importance.csv"),
        "--phase2-report-input",
        str(phase4_dir / "phase4_labeling_report.txt"),
        "--phase3-report-input", # Input from Phase 5
        str(phase5_dir / "phase5_split_report.txt"), # Input from Phase 5
        "--phase4-report-input", # Input from Phase 6
        str(phase6_dir / "phase6_training_report.txt"), # Input from Phase 6
        "--final-summary-output",
        str(phase7_dir / "final_summary_report.txt"),
        "--selection-metric",
        str(cfg.phase7_selection_metric), # Renamed to phase7_selection_metric
        "--top-features",
        str(cfg.phase7_top_features), # Renamed to phase7_top_features
        "--auc-threshold",
        str(cfg.phase7_auc_threshold), # Renamed to phase7_auc_threshold
        "--recall-low-threshold",
        str(cfg.phase7_recall_low_threshold), # Renamed to phase7_recall_low_threshold
        "--log-every",
        str(cfg.log_every),
    ]
    if cfg.max_rows is not None:
        cmd5b.extend(["--max-rows", str(cfg.max_rows)])
    run_command(cmd5b, cfg.project_root, "Phase 7 - Báo cáo độ đo đánh giá mô hình")


def run_phase_8(cfg: ExperimentConfig) -> None:
    """Phase 8: Phân tích khả năng giải thích mô hình"""
    phase8_dir = cfg.results_dir / "phase8" # Output directory for Phase 8
    phase6_dir = cfg.results_dir / "phase6" # Input from Phase 6
    phase5_dir = cfg.results_dir / "phase5"
    phase7_dir = cfg.results_dir / "phase7"
    phase_8 = cfg.scripts_dir / "phase_8_model_interpretability.py"

    cmd6 = [
        sys.executable,
        str(phase_8),
        "--results-dir",
        str(phase8_dir),
        "--output-dir",
        str(phase8_dir),
        "--model-bundle-input",
        str(phase6_dir / "phase6_best_model.pkl"),
        "--predictions-input",
        str(phase6_dir / "phase6_best_model_predictions.csv"),
        "--test-input",
        str(phase5_dir / "phase5_test.csv"),
        "--final-summary-input",
        str(phase7_dir / "final_summary_report.txt"),
        "--final-summary-output",
        str(phase8_dir / "final_summary_report.txt"),
        "--local-error-samples",
        str(cfg.phase8_local_error_samples), # Renamed to phase8_local_error_samples
        "--local-correct-samples",
        str(cfg.phase8_local_correct_samples), # Renamed to phase8_local_correct_samples
        "--top-features",
        str(cfg.phase8_top_features), # Renamed to phase8_top_features
        "--log-every",
        str(cfg.log_every),
    ]
    if cfg.max_rows is not None:
        cmd6.extend(["--max-rows", str(cfg.max_rows)])
    run_command(cmd6, cfg.project_root, "Phase 8 - Phân tích khả năng giải thích mô hình")


def build_parser() -> argparse.ArgumentParser:
    """Tạo parser cho command line arguments"""
    parser = argparse.ArgumentParser(
        description="Script chính chạy thực nghiệm đồ án Khai phá Dữ liệu DS317"
    )
    parser.add_argument(
        "--phase",
        type=str,
        choices=["1", "2", "3", "4", "5", "6", "7", "8", "all"], # Corrected phase choices
        default="all",
        help=(
            "Phase cần chạy: 1 (làm sạch), 2 (trích xuất đặc trưng), 3 (EDA), " # Updated help text
            "4 (nhãn K-Means), 5 (chia tập + cân bằng), 6 (huấn luyện supervised), " # Updated help text
            "7 (đánh giá + báo cáo), 8 (giải thích mô hình), " # Updated help text
            "hoặc all (1->2->3->4->5->6->7->8)." # Updated help text
        ),
    )
    parser.add_argument(
        "--dataset-dir",
        type=Path,
        default=Path(r"D:\MOOCCubeX_dataset"),
        help="Thư mục dataset (mặc định: D:\\MOOCCubeX_dataset)",
    )
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=Path("results"),
        help="Thư mục kết quả (mặc định: results)",
    )
    parser.add_argument(
        "--user-input",
        type=Path,
        default=Path("user.json"),
        help="File user gốc để dịch (mặc định: user.json trong dataset)",
    )
    parser.add_argument(
        "--translated-user",
        type=Path,
        default=Path("user_school_en.json"),
        help="File user đã dịch (mặc định: user_school_en.json trong dataset)",
    )
    parser.add_argument(
        "--translate-summary",
        type=Path,
        default=Path("school_translate_summary.txt"),
        help="File tóm tắt dịch (mặc định: school_translate_summary.txt trong results)",
    )
    parser.add_argument(
        "--skip-translate",
        action="store_true",
        help="Bỏ qua dịch và sử dụng --translated-user trực tiếp",
    )
    parser.add_argument(
        "--combined-file",
        type=Path,
        default=Path("combined_user_metrics.csv"),
        help="File metrics kết hợp (mặc định: combined_user_metrics.csv trong results)",
    )
    parser.add_argument(
        "--weekly-file",
        type=Path,
        default=Path("step2_user_week_activity.csv"),
        help="File hoạt động hàng tuần (mặc định: step2_user_week_activity.csv trong results)",
    )
    parser.add_argument(
        "--db-file",
        type=Path,
        default=Path("combined_streaming.sqlite3"),
        help="File SQLite tạm cho giai đoạn kết hợp (mặc định: combined_streaming.sqlite3 trong results)",
    )
    parser.add_argument(
        "--combined-parquet",
        type=Path,
        default=None,
        help="Sử dụng file Parquet làm nguồn dữ liệu cho phase 1 (chuyển thành CSV kết hợp)",
    )
    parser.add_argument(
        "--clusters",
        type=int,
        default=3,
        help="Số cluster cho K-Means (mặc định: 3)",
    )
    parser.add_argument(
        "--q-low",
        type=float,
        default=0.33,
        help="Ngưỡng percentile thấp cho gán nhãn engagement trước (mặc định: 0.33)",
    )
    parser.add_argument(
        "--q-high",
        type=float,
        default=0.66,
        help="Ngưỡng percentile cao cho gán nhãn engagement trước (mặc định: 0.66)",
    )
    parser.add_argument(
        "--split-strategy",
        type=str,
        choices=["stratified", "group", "temporal", "hybrid"],
        default="stratified",
        help="Chiến lược chia cho phase 3 (mặc định: stratified)",
    )
    parser.add_argument(
        "--imbalance-method",
        type=str,
        choices=["none", "random_oversample", "smote"],
        default="random_oversample",
        help="Phương pháp xử lý mất cân bằng cho phase 3",
    )
    parser.add_argument(
        "--label-column",
        type=str,
        default="StandardLabelKMeans",
        help="Cột nhãn được sử dụng bởi script phase 3",
    )
    parser.add_argument(
        "--group-column",
        type=str,
        default="user_id",
        help="Cột nhóm được sử dụng trong chiến lược group/hybrid của phase 3",
    )
    parser.add_argument(
        "--time-column",
        type=str,
        default="last_activity_time",
        help="Cột thời gian được sử dụng trong chiến lược temporal/hybrid của phase 3",
    )
    parser.add_argument(
        "--time-fallback-column",
        type=str,
        default="first_activity_time",
        help="Cột thời gian dự phòng khi time-column bị thiếu",
    )
    parser.add_argument(
        "--valid-size",
        type=float,
        default=0.10,
        help="Tỷ lệ chia validation cho phase 3 (mặc định: 0.10)",
    )
    parser.add_argument(
        "--test-size",
        type=float,
        default=0.10,
        help="Tỷ lệ chia test cho phase 3 (mặc định: 0.10)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Seed ngẫu nhiên cho tìm kiếm chia phase 3",
    )
    parser.add_argument(
        "--seed-trials",
        type=int,
        default=30,
        help="Số lần thử seed cho tìm kiếm chia phase 3",
    )
    parser.add_argument(
        "--smote-k-neighbors",
        type=int,
        default=5,
        help="k-neighbors cho SMOTE phase 3",
    )
    parser.add_argument(
        "--phase4-models",
        type=str,
        default="logistic,random_forest,hist_gb", # Renamed to phase6_models
        help=(
            "Danh sách mô hình cho phase 6 " # Updated help text
            "(logistic,random_forest,hist_gb,catboost,xgboost)"
        ),
    )
    parser.add_argument(
        "--phase4-primary-metric",
        type=str,
        choices=["macro_f1", "weighted_f1", "accuracy"], # Renamed to phase6_primary_metric
        default="macro_f1", # Renamed to phase6_primary_metric
        help="Metric chọn mô hình trong phase 6", # Updated help text
    )
    parser.add_argument(
        "--phase4-cv-folds",
        type=int,
        default=3, # Renamed to phase6_cv_folds
        help="Số folds cross-validation cho tìm kiếm hyperparameter phase 6", # Updated help text
    )
    parser.add_argument(
        "--phase4-n-jobs",
        type=int,
        default=-1, # Renamed to phase6_n_jobs
        help="Jobs song song cho tìm kiếm mô hình phase 6 (GridSearchCV)", # Updated help text
    )
    parser.add_argument(
        "--phase4-feature-columns",
        type=str,
        default=None,
        help="Tùy chọn cột đặc trưng cho phase 4, cách nhau bởi dấu phẩy",
    )
    parser.add_argument(
        "--phase5-selection-metric",
        type=str,
        choices=["macro_f1", "weighted_f1", "accuracy"], # Renamed to phase7_selection_metric
        default="macro_f1", # Renamed to phase7_selection_metric
        help="Metric lựa chọn được sử dụng trong tóm tắt mô hình phase 7", # Updated help text
    )
    parser.add_argument(
        "--phase5-top-features",
        type=int,
        default=10, # Renamed to phase7_top_features
        help="Top N đặc trưng được xuất bởi phase 7", # Updated help text
    )
    parser.add_argument(
        "--phase5-auc-threshold",
        type=float,
        default=0.85, # Renamed to phase7_auc_threshold
        help="Ngưỡng AUC được sử dụng trong kiểm tra phase 7", # Updated help text
    )
    parser.add_argument(
        "--phase5-recall-low-threshold",
        type=float,
        default=0.80, # Renamed to phase7_recall_low_threshold
        help="Ngưỡng recall cho lớp Low trong kiểm tra phase 7", # Updated help text
    )
    parser.add_argument(
        "--phase6-local-error-samples",
        type=int,
        default=3, # Renamed to phase8_local_error_samples
        help="Số giải thích cục bộ phân loại sai được xuất trong phase 8", # Updated help text
    )
    parser.add_argument(
        "--phase6-local-correct-samples",
        type=int,
        default=3, # Renamed to phase8_local_correct_samples
        help="Số giải thích cục bộ phân loại đúng được xuất trong phase 8", # Updated help text
    )
    parser.add_argument(
        "--phase6-top-features",
        type=int,
        default=10, # Renamed to phase8_top_features
        help="Top N đặc trưng toàn cục được xuất trong phase 8", # Updated help text
    )
    parser.add_argument(
        "--silhouette-sample-size",
        type=int,
        default=20000,
        help="Kích thước mẫu được sử dụng cho điểm silhouette trong xác thực phase 2",
    )
    parser.add_argument(
        "--top-users",
        type=int,
        default=100, # Used in Phase 4
        help="Top users được xuất trong báo cáo phase 4 (mặc định: 100)", # Updated help text
    )
    parser.add_argument(
        "--min-school-size",
        type=int,
        default=20, # Used in Phase 4
        help="Kích thước trường tối thiểu trong báo cáo phase 4 (mặc định: 20)", # Updated help text
    )
    parser.add_argument(
        "--top-schools",
        type=int,
        default=30, # Used in Phase 4
        help="Số trường tối đa trong báo cáo phase 4 (mặc định: 30)", # Updated help text
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=5000, # Used in Phase 4
        help="Kích thước batch cho K-Means (mặc định: 5000)", # Updated help text
    )
    parser.add_argument(
        "--log-every",
        type=int,
        default=200000,
        help="Khoảng log tiến trình được truyền xuống các script con",
    )
    parser.add_argument(
        "--max-rows",
        type=int,
        default=None,
        help="Giới hạn hàng tùy chọn cho chạy thử nghiệm được truyền xuống các script con",
    )
    return parser


def main() -> int:
    """Hàm chính"""
    args = build_parser().parse_args()

    project_root = Path(__file__).resolve().parents[2]
    scripts_dir = (project_root / "experiment").resolve() # Corrected scripts_dir path
    
    dataset_dir = resolve_path_arg(args.dataset_dir, project_root, project_root)
    results_dir = resolve_path_arg(args.results_dir, project_root, project_root)

    user_input = resolve_path_arg(args.user_input, project_root, dataset_dir)
    translated_user = resolve_path_arg(args.translated_user, project_root, results_dir / "phase1") # Corrected path
    translate_summary = resolve_path_arg(args.translate_summary, project_root, results_dir / "phase1") # Corrected path

    combined_file = resolve_path_arg(args.combined_file, project_root, results_dir / "phase2") # Corrected path
    weekly_file = resolve_path_arg(args.weekly_file, project_root, results_dir / "phase2") # Corrected path
    db_file = resolve_path_arg(args.db_file, project_root, results_dir / "phase2") # Corrected path
    combined_parquet = resolve_path_arg(args.combined_parquet, project_root, project_root) if args.combined_parquet is not None else None

    cfg = ExperimentConfig(
        project_root=project_root,
        scripts_dir=scripts_dir,
        dataset_dir=dataset_dir,
        results_dir=results_dir,
        user_input=user_input,
        translated_user=translated_user,
        translate_summary=translate_summary,
        combined_file=combined_file,
        weekly_file=weekly_file,
        db_file=db_file,
        combined_parquet=combined_parquet,
        phase=args.phase,
        skip_translate=args.skip_translate,
        clusters=max(1, args.clusters),
        q_low=max(0.0, min(1.0, args.q_low)),
        q_high=max(0.0, min(1.0, args.q_high)),
        split_strategy=args.split_strategy,
        imbalance_method=args.imbalance_method,
        label_column=args.label_column,
        group_column=args.group_column,
        time_column=args.time_column,
        time_fallback_column=args.time_fallback_column,
        valid_size=max(1e-6, min(0.99, args.valid_size)),
        test_size=max(1e-6, min(0.99, args.test_size)),
        seed=args.seed,
        seed_trials=max(1, args.seed_trials),
        smote_k_neighbors=max(1, args.smote_k_neighbors),
        phase4_models=args.phase4_models,
        phase4_primary_metric=args.phase4_primary_metric,
        phase4_cv_folds=max(2, args.phase4_cv_folds),
        phase4_n_jobs=args.phase4_n_jobs,
        phase4_feature_columns=args.phase4_feature_columns,
        phase5_selection_metric=args.phase5_selection_metric,
        phase5_top_features=max(1, args.phase5_top_features),
        phase5_auc_threshold=float(args.phase5_auc_threshold),
        phase5_recall_low_threshold=float(args.phase5_recall_low_threshold),
        phase6_local_error_samples=max(0, args.phase6_local_error_samples),
        phase6_local_correct_samples=max(0, args.phase6_local_correct_samples),
        phase6_top_features=max(1, args.phase6_top_features),
        silhouette_sample_size=max(100, args.silhouette_sample_size),
        top_users=max(0, args.top_users),
        min_school_size=max(1, args.min_school_size),
        top_schools=max(1, args.top_schools),
        log_every=max(1, args.log_every),
        max_rows=args.max_rows,
    )

    try:
        started = time.time()
        log("Bắt đầu pipeline thực nghiệm theo kịch bản")

        if cfg.q_low >= cfg.q_high:
            raise RuntimeError("q-low phải nhỏ hơn q-high")
        if cfg.valid_size + cfg.test_size >= 1:
            raise RuntimeError("valid-size + test-size phải < 1")

        if cfg.phase in ("1", "all"):
            run_phase_1(cfg)
        if cfg.phase in ("2", "all"):
            run_phase_2(cfg)
        if cfg.phase in ("3", "all"):
            run_phase_3(cfg)
        if cfg.phase in ("4", "all"):
            run_phase_4(cfg)
        if cfg.phase in ("5", "all"):
            run_phase_5(cfg)
        if cfg.phase in ("6", "all"):
            run_phase_6(cfg)
        if cfg.phase in ("7", "all"):
            run_phase_7(cfg)
        if cfg.phase in ("8", "all"):
            run_phase_8(cfg)

        log(f"Pipeline hoàn thành trong {time.time() - started:.2f}s")
        return 0

    except Exception as e:
        log(f"Lỗi: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())