#!/usr/bin/env python3
"""
Run the current pipeline grouped by experimental phases.

Scenario-aligned mapping based on course description:
- Phase 1: Exploratory Data Analysis (EDA) -> phase_1_eda.py
- Phase 2: Data Cleaning -> phase_2_data_cleaning.py
- Phase 3: Data Transformation -> phase_3_data_transformation.py
- Phase 4: Data Labeling -> phase_4_data_labeling.py
- Phase 5: Data Splitting -> phase_5_data_splitting.py
- Phase 6: Model Training -> phase_6_model_training.py
- Phase 7: Model Evaluation -> phase_7_model_evaluation.py
- Phase 8: Model Interpretability -> phase_8_model_interpretability.py

Use --phase all to run 1 -> 2 -> 3 -> 4 -> 5 -> 6 -> 7 -> 8 sequentially.
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
class StageConfig:
    project_root: Path
    experiment_dir: Path
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
    phase6_models: str
    phase6_primary_metric: str
    phase6_cv_folds: int
    phase6_n_jobs: int
    phase6_feature_columns: Optional[str]
    phase7_selection_metric: str
    phase7_top_features: int
    phase7_auc_threshold: float
    phase7_recall_low_threshold: float
    phase7_skip_step4_report: bool
    phase8_local_error_samples: int
    phase8_local_correct_samples: int
    phase8_top_features: int
    silhouette_sample_size: int
    top_users: int
    min_school_size: int
    top_schools: int
    log_every: int
    max_rows: Optional[int]
    missing_threshold: float
    outlier_iqr_multiplier: float
    batch_size: int


def now_text() -> str:
    return time.strftime("%Y-%m-%d %H:%M:%S")


def log(message: str) -> None:
    print(f"[{now_text()}] {message}")


def resolve_path_arg(path_value: Path, project_root: Path, default_base: Path) -> Path:
    if path_value.is_absolute():
        return path_value.resolve()
    if path_value.parent == Path("."):
        return (default_base / path_value).resolve()
    return (project_root / path_value).resolve()


def run_command(command: List[str], cwd: Path, label: str) -> None:
    log(f"Running {label}")
    proc = subprocess.run(command, cwd=str(cwd), check=False)
    if proc.returncode != 0:
        raise RuntimeError(f"{label} failed with exit code {proc.returncode}")


def run_phase_1(cfg: StageConfig) -> None:
    phase_1 = cfg.experiment_dir / "phase_1_eda.py"
    cmd = [
        sys.executable, str(phase_1),
        "--results-dir", str(cfg.results_dir),
        "--combined-input", str(cfg.combined_file),
        "--missing-threshold", str(cfg.missing_threshold),
        "--outlier-iqr-multiplier", str(cfg.outlier_iqr_multiplier),
        "--log-every", str(cfg.log_every),
    ]
    if cfg.max_rows is not None:
        cmd.extend(["--max-rows", str(cfg.max_rows)])
    run_command(cmd, cfg.project_root, "Phase 1 - Exploratory Data Analysis")


def run_phase_2(cfg: StageConfig) -> None:
    phase_2 = cfg.experiment_dir / "phase_2_data_cleaning.py"
    cmd = [
        sys.executable, str(phase_2),
        "--dataset-dir", str(cfg.dataset_dir),
        "--output-dir", str(cfg.results_dir),
        "--user-input", str(cfg.user_input),
        "--translated-user", str(cfg.translated_user),
        "--translate-summary", str(cfg.translate_summary),
        "--log-every", str(cfg.log_every),
    ]
    if cfg.max_rows is not None:
        cmd.extend(["--max-rows", str(cfg.max_rows)])
    run_command(cmd, cfg.project_root, "Phase 2 - Data Cleaning")


def run_phase_3(cfg: StageConfig) -> None:
    phase_3 = cfg.experiment_dir / "phase_3_data_transformation.py"
    cmd = [
        sys.executable, str(phase_3),
        "--dataset-dir", str(cfg.dataset_dir),
        "--output-dir", str(cfg.results_dir),
        "--translated-user", str(cfg.translated_user),
        "--combined-file", str(cfg.combined_file),
        "--weekly-file", str(cfg.weekly_file),
        "--db-file", str(cfg.db_file),
        "--log-every", str(cfg.log_every),
    ]
    if cfg.combined_parquet is not None:
        cmd.extend(["--combined-parquet", str(cfg.combined_parquet)])
    if cfg.max_rows is not None:
        cmd.extend(["--max-rows", str(cfg.max_rows)])
    run_command(cmd, cfg.project_root, "Phase 3 - Data Transformation")


def run_phase_4(cfg: StageConfig) -> None:
    phase_4 = cfg.experiment_dir / "phase_4_data_labeling.py"
    cmd = [
        sys.executable, str(phase_4),
        "--results-dir", str(cfg.results_dir),
        "--combined-input", str(cfg.combined_file),
        "--weekly-input", str(cfg.weekly_file),
        "--clusters", str(cfg.clusters),
        "--batch-size", str(cfg.batch_size),
        "--q-low", str(cfg.q_low),
        "--q-high", str(cfg.q_high),
        "--silhouette-sample-size", str(cfg.silhouette_sample_size),
        "--top-users", str(cfg.top_users),
        "--min-school-size", str(cfg.min_school_size),
        "--top-schools", str(cfg.top_schools),
        "--log-every", str(cfg.log_every),
    ]
    if cfg.max_rows is not None:
        cmd.extend(["--max-rows", str(cfg.max_rows)])
    run_command(cmd, cfg.project_root, "Phase 4 - Data Labeling")


def run_phase_5(cfg: StageConfig) -> None:
    phase_5 = cfg.experiment_dir / "phase_5_data_splitting.py"
    step5_labeled_csv = (cfg.results_dir / "step5_standard_labels_kmeans.csv").resolve()

    cmd = [
        sys.executable, str(phase_5),
        "--results-dir", str(cfg.results_dir),
        "--output-dir", str(cfg.results_dir),
        "--labeled-csv", str(step5_labeled_csv),
        "--combined-csv", str(cfg.combined_file),
        "--split-strategy", str(cfg.split_strategy),
        "--imbalance-method", str(cfg.imbalance_method),
        "--label-column", str(cfg.label_column),
        "--group-column", str(cfg.group_column),
        "--time-column", str(cfg.time_column),
        "--time-fallback-column", str(cfg.time_fallback_column),
        "--valid-size", str(cfg.valid_size),
        "--test-size", str(cfg.test_size),
        "--seed", str(cfg.seed),
        "--seed-trials", str(cfg.seed_trials),
        "--smote-k-neighbors", str(cfg.smote_k_neighbors),
        "--log-every", str(cfg.log_every),
    ]
    if cfg.phase6_feature_columns is not None:
        cmd.extend(["--feature-columns", str(cfg.phase6_feature_columns)])
    if cfg.max_rows is not None:
        cmd.extend(["--max-rows", str(cfg.max_rows)])

    run_command(cmd, cfg.project_root, "Phase 5 - Data Splitting")


def run_phase_6(cfg: StageConfig) -> None:
    phase_6 = cfg.experiment_dir / "phase_6_model_training.py"
    stage3_train_modeling = (cfg.results_dir / "stage3_train_modeling.csv").resolve()
    stage3_valid = (cfg.results_dir / "stage3_valid.csv").resolve()
    stage3_test = (cfg.results_dir / "stage3_test.csv").resolve()

    cmd = [
        sys.executable, str(phase_6),
        "--results-dir", str(cfg.results_dir),
        "--output-dir", str(cfg.results_dir),
        "--train-input", str(stage3_train_modeling),
        "--valid-input", str(stage3_valid),
        "--test-input", str(stage3_test),
        "--label-column", str(cfg.label_column),
        "--models", str(cfg.phase6_models),
        "--primary-metric", str(cfg.phase6_primary_metric),
        "--cv-folds", str(cfg.phase6_cv_folds),
        "--seed", str(cfg.seed),
        "--n-jobs", str(cfg.phase6_n_jobs),
        "--log-every", str(cfg.log_every),
    ]
    if cfg.phase6_feature_columns is not None:
        cmd.extend(["--feature-columns", str(cfg.phase6_feature_columns)])
    if cfg.max_rows is not None:
        cmd.extend(["--max-rows", str(cfg.max_rows)])

    run_command(cmd, cfg.project_root, "Phase 6 - Model Training")


def run_phase_7(cfg: StageConfig) -> None:
    phase_7 = cfg.experiment_dir / "phase_7_model_evaluation.py"
    cmd = [
        sys.executable, str(phase_7),
        "--results-dir", str(cfg.results_dir),
        "--output-dir", str(cfg.results_dir),
        "--selection-metric", str(cfg.phase7_selection_metric),
        "--top-features", str(cfg.phase7_top_features),
        "--auc-threshold", str(cfg.phase7_auc_threshold),
        "--recall-low-threshold", str(cfg.phase7_recall_low_threshold),
        "--log-every", str(cfg.log_every),
    ]
    if cfg.max_rows is not None:
        cmd.extend(["--max-rows", str(cfg.max_rows)])
    run_command(cmd, cfg.project_root, "Phase 7 - Model Evaluation")


def run_phase_8(cfg: StageConfig) -> None:
    phase_8 = cfg.experiment_dir / "phase_8_model_interpretability.py"
    cmd = [
        sys.executable, str(phase_8),
        "--results-dir", str(cfg.results_dir),
        "--output-dir", str(cfg.results_dir),
        "--local-error-samples", str(cfg.phase8_local_error_samples),
        "--local-correct-samples", str(cfg.phase8_local_correct_samples),
        "--top-features", str(cfg.phase8_top_features),
        "--log-every", str(cfg.log_every),
    ]
    if cfg.max_rows is not None:
        cmd.extend(["--max-rows", str(cfg.max_rows)])
    run_command(cmd, cfg.project_root, "Phase 8 - Model Interpretability")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run pipeline grouped by scenario phases according to course description."
    )
    parser.add_argument(
        "--phase",
        type=str,
        choices=["1", "2", "3", "4", "5", "6", "7", "8", "all"],
        default="all",
        help="Phase to run (1 to 8) or 'all'.",
    )
    parser.add_argument("--dataset-dir", type=Path, default=Path(r"D:\MOOCCubeX_dataset"))
    # Default results dir = experiment/results/ (same folder as this script)
    parser.add_argument("--results-dir", type=Path, default=Path(__file__).resolve().parent / "results")
    parser.add_argument("--user-input", type=Path, default=Path("user.json"))
    parser.add_argument("--translated-user", type=Path, default=Path("user_school_en.json"))
    parser.add_argument("--translate-summary", type=Path, default=Path("school_translate_summary.txt"))
    parser.add_argument("--skip-translate", action="store_true")
    parser.add_argument("--combined-file", type=Path, default=Path("combined_user_metrics.csv"))
    parser.add_argument("--weekly-file", type=Path, default=Path("step2_user_week_activity.csv"))
    parser.add_argument("--db-file", type=Path, default=Path("combined_streaming.sqlite3"))
    parser.add_argument("--combined-parquet", type=Path, default=None)
    parser.add_argument("--missing-threshold", type=float, default=0.3)
    parser.add_argument("--outlier-iqr-multiplier", type=float, default=1.5)
    parser.add_argument("--batch-size", type=int, default=5000)
    parser.add_argument("--clusters", type=int, default=3)
    parser.add_argument("--q-low", type=float, default=0.33)
    parser.add_argument("--q-high", type=float, default=0.66)
    parser.add_argument("--split-strategy", type=str, choices=["stratified", "group", "temporal", "hybrid"], default="stratified")
    parser.add_argument("--imbalance-method", type=str, choices=["none", "random_oversample", "smote"], default="random_oversample")
    parser.add_argument("--label-column", type=str, default="StandardLabelKMeans")
    parser.add_argument("--group-column", type=str, default="user_id")
    parser.add_argument("--time-column", type=str, default="last_activity_time")
    parser.add_argument("--time-fallback-column", type=str, default="first_activity_time")
    parser.add_argument("--valid-size", type=float, default=0.10)
    parser.add_argument("--test-size", type=float, default=0.10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--seed-trials", type=int, default=30)
    parser.add_argument("--smote-k-neighbors", type=int, default=5)
    parser.add_argument("--phase6-models", type=str, default="logistic,random_forest,hist_gb")
    parser.add_argument("--phase6-primary-metric", type=str, choices=["macro_f1", "weighted_f1", "accuracy"], default="macro_f1")
    parser.add_argument("--phase6-cv-folds", type=int, default=3)
    parser.add_argument("--phase6-n-jobs", type=int, default=-1)
    parser.add_argument("--phase6-feature-columns", type=str, default=None)
    parser.add_argument("--phase7-selection-metric", type=str, choices=["macro_f1", "weighted_f1", "accuracy"], default="macro_f1")
    parser.add_argument("--phase7-top-features", type=int, default=10)
    parser.add_argument("--phase7-auc-threshold", type=float, default=0.85)
    parser.add_argument("--phase7-recall-low-threshold", type=float, default=0.80)
    parser.add_argument("--phase7-skip-step4-report", action="store_true")
    parser.add_argument("--phase8-local-error-samples", type=int, default=3)
    parser.add_argument("--phase8-local-correct-samples", type=int, default=3)
    parser.add_argument("--phase8-top-features", type=int, default=10)
    parser.add_argument("--silhouette-sample-size", type=int, default=20000)
    parser.add_argument("--top-users", type=int, default=100)
    parser.add_argument("--min-school-size", type=int, default=20)
    parser.add_argument("--top-schools", type=int, default=30)
    parser.add_argument("--log-every", type=int, default=200000)
    parser.add_argument("--max-rows", type=int, default=None)
    return parser


def main() -> int:
    args = build_parser().parse_args()

    project_root = Path(__file__).resolve().parents[1]
    experiment_dir = (project_root / "experiment").resolve()
    dataset_dir = resolve_path_arg(args.dataset_dir, project_root, project_root)
    results_dir = resolve_path_arg(args.results_dir, project_root, project_root)

    user_input = resolve_path_arg(args.user_input, project_root, dataset_dir)
    translated_user = resolve_path_arg(args.translated_user, project_root, dataset_dir)
    translate_summary = resolve_path_arg(args.translate_summary, project_root, results_dir)

    combined_file = resolve_path_arg(args.combined_file, project_root, results_dir)
    weekly_file = resolve_path_arg(args.weekly_file, project_root, results_dir)
    db_file = resolve_path_arg(args.db_file, project_root, results_dir)

    cfg = StageConfig(
        project_root=project_root,
        experiment_dir=experiment_dir,
        dataset_dir=dataset_dir,
        results_dir=results_dir,
        user_input=user_input,
        translated_user=translated_user,
        translate_summary=translate_summary,
        combined_file=combined_file,
        weekly_file=weekly_file,
        db_file=db_file,
        combined_parquet=args.combined_parquet,
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
        phase6_models=args.phase6_models,
        phase6_primary_metric=args.phase6_primary_metric,
        phase6_cv_folds=max(2, args.phase6_cv_folds),
        phase6_n_jobs=args.phase6_n_jobs,
        phase6_feature_columns=args.phase6_feature_columns,
        phase7_selection_metric=args.phase7_selection_metric,
        phase7_top_features=max(1, args.phase7_top_features),
        phase7_auc_threshold=float(args.phase7_auc_threshold),
        phase7_recall_low_threshold=float(args.phase7_recall_low_threshold),
        phase7_skip_step4_report=args.phase7_skip_step4_report,
        phase8_local_error_samples=max(0, args.phase8_local_error_samples),
        phase8_local_correct_samples=max(0, args.phase8_local_correct_samples),
        phase8_top_features=max(1, args.phase8_top_features),
        silhouette_sample_size=max(100, args.silhouette_sample_size),
        top_users=max(0, args.top_users),
        min_school_size=max(1, args.min_school_size),
        top_schools=max(1, args.top_schools),
        log_every=max(1, args.log_every),
        max_rows=args.max_rows,
        missing_threshold=args.missing_threshold,
        outlier_iqr_multiplier=args.outlier_iqr_multiplier,
        batch_size=args.batch_size
    )

    try:
        started = time.time()
        log("Starting scenario-phase pipeline")

        # Chronological execution order:
        # 1. Clean raw data (Phase 2)
        # 2. Data Transformation (Phase 3)
        # 3. EDA (Phase 1)
        # 4. Labeling (Phase 4)
        # 5. Split (Phase 5)
        # 6,7,8. Modeling

        if cfg.phase in ("2", "all"):
            run_phase_2(cfg)
        if cfg.phase in ("3", "all"):
            run_phase_3(cfg)
        if cfg.phase in ("1", "all"):
            run_phase_1(cfg)
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

        log(f"Pipeline completed in {time.time() - started:.2f}s")
        return 0
    except Exception as exc:
        log(f"FAILED: {exc}")
        return 1

if __name__ == "__main__":
    raise SystemExit(main())