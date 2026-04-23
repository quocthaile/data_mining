#!/usr/bin/env python3
"""
Script chính cho đồ án Khai phá Dữ liệu - DS317
Dự đoán mức độ tham gia của học viên trong hệ thống MOOC

Kịch bản thực nghiệm:
1. Trích xuất đặc trưng chuỗi thời gian từ dữ liệu hoạt động học viên
2. Khởi tạo nhãn ground-truth bằng K-Means và xác thực nhãn
3. Chia tập train/valid/test và xử lý mất cân bằng dữ liệu
4. Huấn luyện mô hình supervised với tối ưu hóa
5. Đánh giá độ đo và báo cáo kết quả
6. Phân tích khả năng giải thích mô hình

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
    phase5_skip_step4_report: bool
    phase6_local_error_samples: int
    phase6_local_correct_samples: int
    phase6_top_features: int
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
    """Phase 1: Trích xuất đặc trưng chuỗi thời gian"""
    phase_1 = cfg.scripts_dir / "phase_1_time_series_feature_extraction.py"

    cmd1 = [
        sys.executable,
        str(phase_1),
        "--dataset-dir",
        str(cfg.dataset_dir),
        "--output-dir",
        str(cfg.results_dir),
        "--user-input",
        str(cfg.user_input),
        "--translated-user",
        str(cfg.translated_user),
        "--translate-summary",
        str(cfg.translate_summary),
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
        cmd1.extend(["--combined-parquet", str(cfg.combined_parquet)])
    if cfg.skip_translate:
        cmd1.append("--skip-translate")
    if cfg.max_rows is not None:
        cmd1.extend(["--max-rows", str(cfg.max_rows)])
    run_command(cmd1, cfg.project_root, "Phase 1 - Trích xuất đặc trưng chuỗi thời gian")


def run_phase_2(cfg: ExperimentConfig) -> None:
    """Phase 2: Khởi tạo nhãn bằng K-Means và xác thực"""
    phase_2 = cfg.scripts_dir / "phase_2_kmeans_label_validation.py"

    cmd2 = [
        sys.executable,
        str(phase_2),
        "--results-dir",
        str(cfg.results_dir),
        "--combined-input",
        str(cfg.combined_file),
        "--weekly-input",
        str(cfg.weekly_file),
        "--clusters",
        str(cfg.clusters),
        "--q-low",
        str(cfg.q_low),
        "--q-high",
        str(cfg.q_high),
        "--silhouette-sample-size",
        str(cfg.silhouette_sample_size),
        "--log-every",
        str(cfg.log_every),
    ]
    if cfg.max_rows is not None:
        cmd2.extend(["--max-rows", str(cfg.max_rows)])
    run_command(cmd2, cfg.project_root, "Phase 2 - K-Means và xác thực nhãn")


def run_phase_3(cfg: ExperimentConfig) -> None:
    """Phase 3: Chia tập dữ liệu và xử lý mất cân bằng"""
    phase_3 = cfg.scripts_dir / "phase_3_train_valid_test_split.py"
    step5_file = (cfg.results_dir / "step5_standard_labels_kmeans.csv").resolve()

    cmd3 = [
        sys.executable,
        str(phase_3),
        "--results-dir",
        str(cfg.results_dir),
        "--output-dir",
        str(cfg.results_dir),
        "--input",
        str(step5_file),
        "--combined-input",
        str(cfg.combined_file),
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

    run_command(cmd3, cfg.project_root, "Phase 3 - Chia tập train/valid/test và xử lý mất cân bằng")


def run_phase_4(cfg: ExperimentConfig) -> None:
    """Phase 4: Huấn luyện mô hình supervised"""
    phase_4 = cfg.scripts_dir / "phase_4_supervised_model_training.py"
    stage3_train_modeling = (cfg.results_dir / "stage3_train_modeling.csv").resolve()
    stage3_valid = (cfg.results_dir / "stage3_valid.csv").resolve()
    stage3_test = (cfg.results_dir / "stage3_test.csv").resolve()

    cmd4 = [
        sys.executable,
        str(phase_4),
        "--results-dir",
        str(cfg.results_dir),
        "--output-dir",
        str(cfg.results_dir),
        "--train-input",
        str(stage3_train_modeling),
        "--valid-input",
        str(stage3_valid),
        "--test-input",
        str(stage3_test),
        "--label-column",
        str(cfg.label_column),
        "--models",
        str(cfg.phase4_models),
        "--primary-metric",
        str(cfg.phase4_primary_metric),
        "--cv-folds",
        str(cfg.phase4_cv_folds),
        "--seed",
        str(cfg.seed),
        "--n-jobs",
        str(cfg.phase4_n_jobs),
        "--log-every",
        str(cfg.log_every),
    ]
    if cfg.phase4_feature_columns is not None:
        cmd4.extend(["--feature-columns", str(cfg.phase4_feature_columns)])
    if cfg.max_rows is not None:
        cmd4.extend(["--max-rows", str(cfg.max_rows)])

    run_command(cmd4, cfg.project_root, "Phase 4 - Huấn luyện mô hình supervised và tối ưu hóa")


def run_phase_5(cfg: ExperimentConfig) -> None:
    """Phase 5: Đánh giá độ đo và báo cáo"""
    phase_5 = cfg.scripts_dir / "phase_5_model_evaluation_metrics.py"
    step_4 = cfg.scripts_dir / "step_4_detailed_report.py"
    step3_results = (cfg.results_dir / "step3_student_engagement_results.csv").resolve()

    if not cfg.phase5_skip_step4_report:
        cmd5a = [
            sys.executable,
            str(step_4),
            "--input",
            str(step3_results),
            "--output-dir",
            str(cfg.results_dir),
            "--top-users",
            str(cfg.top_users),
            "--min-school-size",
            str(cfg.min_school_size),
            "--top-schools",
            str(cfg.top_schools),
            "--log-every",
            str(cfg.log_every),
        ]
        if cfg.max_rows is not None:
            cmd5a.extend(["--max-rows", str(cfg.max_rows)])
        run_command(cmd5a, cfg.project_root, "Phase 5A - Báo cáo chi tiết dữ liệu")

    cmd5b = [
        sys.executable,
        str(phase_5),
        "--results-dir",
        str(cfg.results_dir),
        "--output-dir",
        str(cfg.results_dir),
        "--selection-metric",
        str(cfg.phase5_selection_metric),
        "--top-features",
        str(cfg.phase5_top_features),
        "--auc-threshold",
        str(cfg.phase5_auc_threshold),
        "--recall-low-threshold",
        str(cfg.phase5_recall_low_threshold),
        "--log-every",
        str(cfg.log_every),
    ]
    if cfg.max_rows is not None:
        cmd5b.extend(["--max-rows", str(cfg.max_rows)])
    run_command(cmd5b, cfg.project_root, "Phase 5B - Báo cáo độ đo đánh giá mô hình")


def run_phase_6(cfg: ExperimentConfig) -> None:
    """Phase 6: Phân tích khả năng giải thích mô hình"""
    phase_6 = cfg.scripts_dir / "phase_6_model_interpretability.py"

    cmd6 = [
        sys.executable,
        str(phase_6),
        "--results-dir",
        str(cfg.results_dir),
        "--output-dir",
        str(cfg.results_dir),
        "--local-error-samples",
        str(cfg.phase6_local_error_samples),
        "--local-correct-samples",
        str(cfg.phase6_local_correct_samples),
        "--top-features",
        str(cfg.phase6_top_features),
        "--log-every",
        str(cfg.log_every),
    ]
    if cfg.max_rows is not None:
        cmd6.extend(["--max-rows", str(cfg.max_rows)])
    run_command(cmd6, cfg.project_root, "Phase 6 - Phân tích khả năng giải thích mô hình")


def build_parser() -> argparse.ArgumentParser:
    """Tạo parser cho command line arguments"""
    parser = argparse.ArgumentParser(
        description="Script chính chạy thực nghiệm đồ án Khai phá Dữ liệu DS317"
    )
    parser.add_argument(
        "--phase",
        type=str,
        choices=["1", "2", "3", "4", "5", "6", "all"],
        default="all",
        help=(
            "Phase cần chạy: 1 (trích xuất đặc trưng), 2 (nhãn K-Means), "
            "3 (chia tập + cân bằng), 4 (huấn luyện supervised), "
            "5 (đánh giá + báo cáo), 6 (giải thích mô hình), "
            "hoặc all (1->2->3->4->5->6)."
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
        default="logistic,random_forest,hist_gb",
        help=(
            "Danh sách mô hình cho phase 4 "
            "(logistic,random_forest,hist_gb,catboost,xgboost)"
        ),
    )
    parser.add_argument(
        "--phase4-primary-metric",
        type=str,
        choices=["macro_f1", "weighted_f1", "accuracy"],
        default="macro_f1",
        help="Metric chọn mô hình trong phase 4",
    )
    parser.add_argument(
        "--phase4-cv-folds",
        type=int,
        default=3,
        help="Số folds cross-validation cho tìm kiếm hyperparameter phase 4",
    )
    parser.add_argument(
        "--phase4-n-jobs",
        type=int,
        default=-1,
        help="Jobs song song cho tìm kiếm mô hình phase 4 (GridSearchCV)",
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
        choices=["macro_f1", "weighted_f1", "accuracy"],
        default="macro_f1",
        help="Metric lựa chọn được sử dụng trong tóm tắt mô hình phase 5",
    )
    parser.add_argument(
        "--phase5-top-features",
        type=int,
        default=10,
        help="Top N đặc trưng được xuất bởi phase 5",
    )
    parser.add_argument(
        "--phase5-auc-threshold",
        type=float,
        default=0.85,
        help="Ngưỡng AUC được sử dụng trong kiểm tra phase 5",
    )
    parser.add_argument(
        "--phase5-recall-low-threshold",
        type=float,
        default=0.80,
        help="Ngưỡng recall cho lớp Low trong kiểm tra phase 5",
    )
    parser.add_argument(
        "--phase5-skip-step4-report",
        action="store_true",
        help="Bỏ qua báo cáo chi tiết dữ liệu step 4 cũ khi chạy phase 5",
    )
    parser.add_argument(
        "--phase6-local-error-samples",
        type=int,
        default=3,
        help="Số giải thích cục bộ phân loại sai được xuất trong phase 6",
    )
    parser.add_argument(
        "--phase6-local-correct-samples",
        type=int,
        default=3,
        help="Số giải thích cục bộ phân loại đúng được xuất trong phase 6",
    )
    parser.add_argument(
        "--phase6-top-features",
        type=int,
        default=10,
        help="Top N đặc trưng toàn cục được xuất trong phase 6",
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
        default=100,
        help="Top users được xuất trong báo cáo phase 5 (mặc định: 100)",
    )
    parser.add_argument(
        "--min-school-size",
        type=int,
        default=20,
        help="Kích thước trường tối thiểu trong báo cáo phase 5 (mặc định: 20)",
    )
    parser.add_argument(
        "--top-schools",
        type=int,
        default=30,
        help="Số trường tối đa trong báo cáo phase 5 (mặc định: 30)",
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

    project_root = Path(__file__).resolve().parents[1]
    scripts_dir = (project_root / "scripts").resolve()

    dataset_dir = resolve_path_arg(args.dataset_dir, project_root, project_root)
    results_dir = resolve_path_arg(args.results_dir, project_root, project_root)

    user_input = resolve_path_arg(args.user_input, project_root, dataset_dir)
    translated_user = resolve_path_arg(args.translated_user, project_root, dataset_dir)
    translate_summary = resolve_path_arg(args.translate_summary, project_root, results_dir)

    combined_file = resolve_path_arg(args.combined_file, project_root, results_dir)
    weekly_file = resolve_path_arg(args.weekly_file, project_root, results_dir)
    db_file = resolve_path_arg(args.db_file, project_root, results_dir)
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
        phase5_skip_step4_report=args.phase5_skip_step4_report,
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

        log(f"Pipeline hoàn thành trong {time.time() - started:.2f}s")
        return 0

    except Exception as e:
        log(f"Lỗi: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())