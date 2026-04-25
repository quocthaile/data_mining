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


PHASE_HELP_TEXT = """
Phase 1 - Data Cleaning
    Description     : Standardize school names (translate to English) and ensure metadata consistency.
    Input           : dataset/user.json
    Output          : results/phase1/user_school_en.json, results/phase1/school_translate_summary.txt
    Visual Goal     : Summary table of school name mappings before and after cleaning.

Phase 2 - Data Transformation
    Description     : Aggregate data from event sources into user-level and user-week features.
    Input           : results/phase1/user_school_en.json and event files in dataset-dir
    Output          : results/phase2/combined_user_metrics.csv, results/phase2/step2_user_week_activity.csv
    Visual Goal     : Weekly activity trends and aggregated user-level variables.

Phase 3 - Exploratory Data Analysis (EDA)
    Description     : Perform descriptive statistics, detect data quality issues, and generate an EDA report.
    Input           : results/phase2/combined_user_metrics.csv
    Output          : results/phase3/phase3_eda_report.txt, results/phase3/combined_user_metrics_clean.csv
    Visual Goal     : Statistical tables, distribution plots, and anomaly indicators.

Phase 4 - Data Labeling
    Description     : Create Low/Medium/High labels based on engagement scores and clustering, and evaluate label quality.
    Input           : results/phase3/combined_user_metrics.csv, results/phase3/step2_user_week_activity.csv
    Output          : results/phase4/step5_standard_labels_kmeans.csv and validation reports
    Visual Goal     : Cluster centers, label distribution, and a cross-tabulation matrix of labels vs. clusters.

Phase 5 - Data Splitting
    Description     : Split data into train/valid/test sets using a specified strategy (e.g., stratified, temporal).
    Input           : results/phase4/step5_standard_labels_kmeans.csv, results/phase3/combined_user_metrics.csv
    Output          : results/phase5/phase5_train_modeling.csv, results/phase5/phase5_valid.csv, results/phase5_test.csv
    Visual Goal     : Comparison table of label distributions across the data splits.

Phase 6 - Model Training
    Description     : Train multiple classification models and select the best one based on validation metrics.
    Input           : results/phase5/phase5_train_modeling.csv, results/phase5/phase5_valid.csv, results/phase5/phase5_test.csv
    Output          : phase6_model_comparison.csv/.png, phase6_confusion_matrix.csv/.png,
                    phase6_feature_importance.csv/.png, phase6_best_model.pkl
    Visual Goal     : Model comparison charts, confusion matrices, and feature importance plots.

Phase 7 - Model Evaluation
    Description     : Summarize metrics for the best model, check against quality thresholds, and generate a final report.
    Input           : Artifacts from Phase 6
    Output          : phase7_model_selection_summary.csv, phase7_metric_checks.csv,
                    phase7_selected_model_metrics.png, phase7_class_metrics.png, final_summary_report.txt
    Visual Goal     : Comparison of metrics on validation/test sets and pass/fail status of quality checks.

Phase 8 - Model Interpretability
    Description     : Analyze and interpret feature contributions at global, class-wise, and individual sample levels.
    Input           : phase6_best_model.pkl, phase6_best_model_predictions.csv, phase5_test.csv
    Output          : phase8_global_importance.csv/.png, phase8_classwise_importance.csv/.png,
                    phase8_local_contributions.csv/.png, phase8_interpretability_report.txt
    Visual Goal     : Top important features and local explanations for representative samples.
""".strip()


PHASE_BRIEF = {
    "1": {
        "name": "Exploratory Data Analysis (EDA)",
        "description": "Perform descriptive statistics, detect data quality issues, and generate an EDA report.",
        "input": "results/combined_user_metrics.csv",
        "output": "phase1_eda_report.txt, combined_user_metrics_clean.csv",
        "visual_goal": "Statistical tables, distribution plots, and anomaly indicators.",
    },
    "2": {
        "name": "Data Cleaning",
        "description": "Standardize school names (translate to English) and ensure metadata consistency.",
        "input": "dataset/user.json",
        "output": "user_school_en.json, school_translate_summary.txt",
        "visual_goal": "Summary table of school name mappings before and after cleaning.",
    },
    "3": {
        "name": "Data Transformation",
        "description": "Aggregate data from event sources into user-level and user-week features.",
        "input": "dataset/user_school_en.json and event files",
        "output": "combined_user_metrics.csv, step2_user_week_activity.csv",
        "visual_goal": "Weekly activity trends and aggregated user-level variables.",
    },
    "4": {
        "name": "Data Labeling",
        "description": "Create Low/Medium/High labels based on engagement scores and clustering, and evaluate label quality.",
        "input": "combined_user_metrics.csv, step2_user_week_activity.csv",
        "output": "step5_standard_labels_kmeans.csv and validation reports",
        "visual_goal": "Cluster centers, label distribution, and a cross-tabulation matrix of labels vs. clusters.",
    },
    "5": {
        "name": "Data Splitting",
        "description": "Split data into train/valid/test sets using a specified strategy (e.g., stratified, temporal).",
        "input": "step5_standard_labels_kmeans.csv, combined_user_metrics.csv",
        "output": "phase5_train_modeling.csv, phase5_valid.csv, phase5_test.csv",
        "visual_goal": "Comparison table of label distributions across the data splits.",
    },
    "6": {
        "name": "Model Training",
        "description": "Train multiple classification models and select the best one based on validation metrics.",
        "input": "phase5_train_modeling.csv, phase5_valid.csv, phase5_test.csv",
        "output": "phase6_* (comparison, confusion matrix, feature importance, model .pkl)",
        "visual_goal": "Model comparison charts, confusion matrices, and feature importance plots.",
    },
    "7": {
        "name": "Model Evaluation",
        "description": "Summarize metrics for the best model, check against quality thresholds, and generate a final report.",
        "input": "Artifacts from Phase 6",
        "output": "phase7_* and final_summary_report.txt",
        "visual_goal": "Comparison of metrics on validation/test sets and pass/fail status of quality checks.",
    },
    "8": {
        "name": "Model Interpretability",
        "description": "Analyze and interpret feature contributions at global, class-wise, and individual sample levels.",
        "input": "phase6_best_model.pkl, phase6_best_model_predictions.csv, phase5_test.csv",
        "output": "phase8_* (global/class/local contributions and plots)",
        "visual_goal": "Top important features and local explanations for representative samples.",
    },
}


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
    cutoff_week: Optional[int]
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


def log_phase_description(phase: str) -> None:
    info = PHASE_BRIEF.get(phase)
    if info is None:
        return

    width = 90
    log("=" * width)
    title = f"PHASE {phase}: {info['name']}"
    log(title.center(width))
    log("-" * width)
    log(f"  Description : {info['description']}")
    log(f"  Input       : {info['input']}")
    log(f"  Output      : {info['output']}")
    log(f"  Goal        : {info['visual_goal']}")
    log("=" * width)


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
    log_phase_description("1")
    phase1_dir = cfg.results_dir / "phase1"
    phase_1 = cfg.experiment_dir / "phase_1_data_cleaning.py"
    cmd = [
        sys.executable, str(phase_1),
        "--dataset-dir", str(cfg.dataset_dir),
        "--output-dir", str(phase1_dir),
        "--user-input", str(cfg.user_input),
        "--translated-user", str(cfg.translated_user),
        "--translate-summary", str(cfg.translate_summary),
        "--log-every", str(cfg.log_every),
    ]
    if cfg.skip_translate or cfg.translated_user.exists():
        log("Translation will be skipped as requested or output file already exists.")
        cmd.append("--skip-translate")

    if cfg.max_rows is not None:
        cmd.extend(["--max-rows", str(cfg.max_rows)])
    run_command(cmd, cfg.project_root, "Phase 1 - Data Cleaning")


def run_phase_2(cfg: StageConfig) -> None:
    log_phase_description("2")
    phase2_dir = cfg.results_dir / "phase2"
    if cfg.combined_file.exists() and cfg.weekly_file.exists():
        log(f"Skipping Phase 2 as output files '{cfg.combined_file.name}' and '{cfg.weekly_file.name}' already exist.")
        # return # Do not return, allow to re-run if needed

    phase_2 = cfg.experiment_dir / "phase_2_data_transformation.py"
    cmd = [
        sys.executable, str(phase_2),
        "--dataset-dir", str(cfg.dataset_dir),
        "--output-dir", str(phase2_dir),
        "--translated-user", str(cfg.translated_user),
        "--combined-file", str(cfg.combined_file),
        "--weekly-file", str(cfg.weekly_file),
        "--db-file", str(cfg.db_file),
        "--log-every", str(cfg.log_every),
    ]
    if cfg.cutoff_week is not None:
        cmd.extend(["--cutoff-week", str(cfg.cutoff_week)])
    if cfg.max_rows is not None:
        cmd.extend(["--max-rows", str(cfg.max_rows)])
    run_command(cmd, cfg.project_root, "Phase 2 - Data Transformation")


def run_phase_3(cfg: StageConfig) -> None:
    log_phase_description("3")
    phase3_dir = cfg.results_dir / "phase3"
    if not cfg.combined_file.exists():
        log(
            "Phase 3 input not found: "
            f"{cfg.combined_file}. Skipping Phase 3 for now. "
            "You can run --phase 3 again after Phase 2 outputs are available."
        )
        return

    phase_3 = cfg.experiment_dir / "phase_3_eda.py"
    cmd = [
        sys.executable, str(phase_3),
        "--results-dir", str(phase3_dir),
        "--combined-input", str(cfg.combined_file),
        "--missing-threshold", str(cfg.missing_threshold),
        "--outlier-iqr-multiplier", str(cfg.outlier_iqr_multiplier),
        "--log-every", str(cfg.log_every),
    ]
    if cfg.max_rows is not None:
        cmd.extend(["--max-rows", str(cfg.max_rows)])
    run_command(cmd, cfg.project_root, "Phase 3 - Exploratory Data Analysis")


def run_phase_4(cfg: StageConfig) -> None:
    log_phase_description("4")
    phase4_dir = cfg.results_dir / "phase4"
    phase_4 = cfg.experiment_dir / "phase_4_data_labeling.py"
    cmd = [ # Corrected input paths for combined and weekly files
        sys.executable, str(phase_4),
        "--results-dir", str(phase4_dir),
        "--combined-input", str(cfg.results_dir / "phase2" / "combined_user_metrics.csv"),
        "--weekly-input", str(cfg.results_dir / "phase2" / "step2_user_week_activity.csv"),
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
    log_phase_description("5")
    phase5_dir = cfg.results_dir / "phase5"
    phase_5 = cfg.experiment_dir / "phase_5_data_splitting.py"
    step5_labeled_csv = (cfg.results_dir / "phase4" / "step5_standard_labels_kmeans.csv").resolve()

    cmd = [
        sys.executable, str(phase_5), # Corrected input path for combined file
        "--results-dir", str(phase5_dir),
        "--output-dir", str(phase5_dir),
        "--input", str(step5_labeled_csv),
        "--combined-input", str(cfg.results_dir / "phase2" / "combined_user_metrics.csv"),
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
    log_phase_description("6")
    phase6_dir = cfg.results_dir / "phase6"
    phase5_dir = cfg.results_dir / "phase5"
    phase_6 = cfg.experiment_dir / "phase_6_model_training.py"
    phase5_train_modeling = (phase5_dir / "phase5_train_modeling.csv").resolve()
    phase5_valid = (phase5_dir / "phase5_valid.csv").resolve()
    phase5_test = (phase5_dir / "phase5_test.csv").resolve()

    cmd = [
        sys.executable, str(phase_6), # Renamed phase6_models, phase6_primary_metric, etc.
        "--results-dir", str(phase6_dir),
        "--output-dir", str(phase6_dir),
        "--train-input", str(phase5_train_modeling),
        "--valid-input", str(phase5_valid),
        "--test-input", str(phase5_test),
        "--label-column", str(cfg.label_column),
        "--models", str(cfg.phase6_models), # Corrected parameter name
        "--primary-metric", str(cfg.phase6_primary_metric), # Corrected parameter name
        "--cv-folds", str(cfg.phase6_cv_folds), # Corrected parameter name
        "--seed", str(cfg.seed),
        "--n-jobs", str(cfg.phase6_n_jobs), # Corrected parameter name
        "--log-every", str(cfg.log_every),
    ]
    if cfg.phase6_feature_columns is not None:
        cmd.extend(["--feature-columns", str(cfg.phase6_feature_columns)])
    if cfg.max_rows is not None:
        cmd.extend(["--max-rows", str(cfg.max_rows)])

    run_command(cmd, cfg.project_root, "Phase 6 - Model Training")


def run_phase_7(cfg: StageConfig) -> None:
    log_phase_description("7")
    phase7_dir = cfg.results_dir / "phase7"
    phase6_dir = cfg.results_dir / "phase6"
    phase5_dir = cfg.results_dir / "phase5"
    phase4_dir = cfg.results_dir / "phase4"
    phase_7 = cfg.experiment_dir / "phase_7_model_evaluation.py"
    cmd = [
        sys.executable, str(phase_7), # Corrected input paths for reports
        "--results-dir", str(phase7_dir),
        "--output-dir", str(phase7_dir),
        "--model-comparison-input", str(phase6_dir / "phase6_model_comparison.csv"),
        "--class-metrics-input", str(phase6_dir / "phase6_classification_metrics.csv"),
        "--confusion-input", str(phase6_dir / "phase6_confusion_matrix.csv"),
        "--feature-importance-input", str(phase6_dir / "phase6_feature_importance.csv"),
        "--phase2-report-input", str(phase4_dir / "phase4_labeling_report.txt"), # Corrected path
        "--phase3-report-input", str(phase5_dir / "phase5_split_report.txt"), # Corrected path
        "--phase4-report-input", str(phase6_dir / "phase6_training_report.txt"), # Corrected path
        "--final-summary-output", str(phase7_dir / "final_summary_report.txt"),
        "--selection-metric", str(cfg.phase7_selection_metric), # Corrected parameter name
        "--top-features", str(cfg.phase7_top_features), # Corrected parameter name
        "--auc-threshold", str(cfg.phase7_auc_threshold), # Corrected parameter name
        "--recall-low-threshold", str(cfg.phase7_recall_low_threshold), # Corrected parameter name
        "--log-every", str(cfg.log_every),
    ]
    if cfg.max_rows is not None:
        cmd.extend(["--max-rows", str(cfg.max_rows)])
    run_command(cmd, cfg.project_root, "Phase 7 - Model Evaluation")


def run_phase_8(cfg: StageConfig) -> None:
    log_phase_description("8")
    phase8_dir = cfg.results_dir / "phase8"
    phase6_dir = cfg.results_dir / "phase6"
    phase5_dir = cfg.results_dir / "phase5"
    phase7_dir = cfg.results_dir / "phase7"
    phase_8 = cfg.experiment_dir / "phase_8_model_interpretability.py"
    cmd = [
        sys.executable, str(phase_8), # Corrected input paths for model bundle, predictions, test, and final summary
        "--results-dir", str(phase8_dir),
        "--output-dir", str(phase8_dir),
        "--model-bundle-input", str(phase6_dir / "phase6_best_model.pkl"), # Corrected path
        "--predictions-input", str(phase6_dir / "phase6_best_model_predictions.csv"), # Corrected path
        "--test-input", str(phase5_dir / "phase5_test.csv"), # Corrected path
        "--final-summary-input", str(phase7_dir / "final_summary_report.txt"), # Corrected path
        "--final-summary-output", str(phase8_dir / "final_summary_report.txt"), # Corrected path
        "--local-error-samples", str(cfg.phase8_local_error_samples), # Corrected parameter name
        "--local-correct-samples", str(cfg.phase8_local_correct_samples), # Corrected parameter name
        "--top-features", str(cfg.phase8_top_features), # Corrected parameter name
        "--log-every", str(cfg.log_every),
    ]
    if cfg.max_rows is not None:
        cmd.extend(["--max-rows", str(cfg.max_rows)])
    run_command(cmd, cfg.project_root, "Phase 8 - Model Interpretability")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run the full early-prediction pipeline grouped by scenario phases.",
        epilog=(
            "Examples:\n"
            "  python run_experiment_stages.py --phase all\n"
            "  python run_experiment_stages.py --phase 6 --phase6-models logistic,random_forest\n"
            "  python run_experiment_stages.py --phase 5 --split-strategy temporal --cutoff-week 202004\n\n"
            "Phase Details:\n"
            f"{PHASE_HELP_TEXT}"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--phase",
        type=str,
        choices=["1", "2", "3", "4", "5", "6", "7", "8", "all"],
        default="all",
        help="Phase to run (1 to 8) or 'all' for the full pipeline.",
    )
    parser.add_argument(
        "--describe-phases",
        action="store_true",
        help="Print detailed phase descriptions (work/input/output/visual goals) and exit.",
    )
    parser.add_argument("--dataset-dir", type=Path, default=Path(r"D:\MOOCCubeX_dataset"))
    # Default results dir = experiment/results/
    parser.add_argument("--results-dir", type=Path, default=Path("experiment/results"))
    parser.add_argument("--user-input", type=Path, default=Path("user.json"))
    parser.add_argument("--translated-user", type=Path, default=Path("phase1/user_school_en.json"))
    parser.add_argument("--translate-summary", type=Path, default=Path("phase1/school_translate_summary.txt")) # Corrected path
    parser.add_argument("--skip-translate", action="store_true")
    parser.add_argument("--combined-file", type=Path, default=Path("phase2/combined_user_metrics.csv")) # Corrected path
    parser.add_argument("--weekly-file", type=Path, default=Path("phase2/step2_user_week_activity.csv")) # Corrected path
    parser.add_argument("--db-file", type=Path, default=Path("phase2/combined_streaming.sqlite3")) # Corrected path
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
    parser.add_argument("--cutoff-week", type=int, default=None, help="Cutoff week for feature extraction (e.g., 202004 for week 4 of 2020)")
    parser.add_argument("--test-size", type=float, default=0.10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--seed-trials", type=int, default=30)
    parser.add_argument("--smote-k-neighbors", type=int, default=5)
    parser.add_argument("--phase6-models", type=str, default="logistic,random_forest,hist_gb")
    parser.add_argument("--phase6-primary-metric", type=str, choices=["macro_f1", "weighted_f1", "accuracy"], default="macro_f1") # Corrected parameter name
    parser.add_argument("--phase6-cv-folds", type=int, default=3) # Corrected parameter name
    parser.add_argument("--phase6-n-jobs", type=int, default=-1) # Corrected parameter name
    parser.add_argument("--phase6-feature-columns", type=str, default=None)
    parser.add_argument("--phase7-selection-metric", type=str, choices=["macro_f1", "weighted_f1", "accuracy"], default="macro_f1")
    parser.add_argument("--phase7-top-features", type=int, default=10)
    parser.add_argument("--phase7-auc-threshold", type=float, default=0.85)
    parser.add_argument("--phase7-recall-low-threshold", type=float, default=0.80)
    parser.add_argument("--phase8-local-error-samples", type=int, default=3)
    parser.add_argument("--phase8-local-correct-samples", type=int, default=3)
    parser.add_argument("--phase8-top-features", type=int, default=10)
    parser.add_argument("--silhouette-sample-size", type=int, default=20000)
    parser.add_argument("--top-users", type=int, default=100) # Used in Phase 4
    parser.add_argument("--min-school-size", type=int, default=20) # Used in Phase 4
    parser.add_argument("--top-schools", type=int, default=30) # Used in Phase 4
    parser.add_argument("--log-every", type=int, default=200000)
    parser.add_argument("--max-rows", type=int, default=None)
    return parser


def main() -> int:
    args = build_parser().parse_args()

    if args.describe_phases:
        print(PHASE_HELP_TEXT)
        return 0

    project_root = Path(__file__).resolve().parents[1]
    experiment_dir = (project_root / "experiment").resolve()
    dataset_dir = resolve_path_arg(args.dataset_dir, project_root, project_root)
    results_dir = resolve_path_arg(args.results_dir, project_root, project_root)

    user_input = resolve_path_arg(args.user_input, project_root, dataset_dir)
    translated_user = resolve_path_arg(args.translated_user, project_root, results_dir)
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
        cutoff_week=args.cutoff_week,
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

        log(f"Pipeline completed in {time.time() - started:.2f}s")
        return 0
    except Exception as exc:
        log(f"FAILED: {exc}")
        return 1

if __name__ == "__main__":
    raise SystemExit(main())