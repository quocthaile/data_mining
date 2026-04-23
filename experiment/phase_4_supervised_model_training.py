#!/usr/bin/env python3
"""
Phase 4: supervised model training and optimization.

Scenario alignment goals:
- Train supervised models on Phase 3 train-modeling dataset.
- Tune hyperparameters via cross-validation on the train dataset only.
- Select best model by validation metric, then evaluate on test.
- Export reproducible artifacts for metrics, confusion matrix, predictions, and model bundle.

Default inputs (from Phase 3 outputs):
- results/stage3_train_modeling.csv
- results/stage3_valid.csv
- results/stage3_test.csv

Generated outputs:
- results/phase4_model_comparison.csv
- results/phase4_classification_metrics.csv
- results/phase4_confusion_matrix.csv
- results/phase4_best_model_predictions.csv
- results/phase4_feature_importance.csv
- results/phase4_best_model.pkl
- results/phase4_training_report.txt
"""

from __future__ import annotations

import argparse
import csv
import json
import pickle
import time
import traceback
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    precision_recall_fscore_support,
    roc_auc_score,
)
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import label_binarize
from sklearn.preprocessing import StandardScaler


@dataclass
class Phase4Config:
    project_root: Path
    results_dir: Path
    output_dir: Path
    train_csv: Path
    valid_csv: Path
    test_csv: Path
    label_column: str
    sample_origin_column: str
    feature_columns_arg: Optional[str]
    models_arg: str
    primary_metric: str
    cv_folds: int
    seed: int
    n_jobs: int
    log_every: int
    max_rows: Optional[int]


@dataclass
class DatasetSplit:
    split_name: str
    X: np.ndarray
    y: np.ndarray
    user_ids: List[str]
    row_ids: List[str]


@dataclass
class ModelSpec:
    name: str
    estimator: Any
    param_grid: Dict[str, Sequence[Any]]


@dataclass
class TrainResult:
    model_name: str
    estimator: Any
    cv_best_score: float
    cv_folds_used: int
    best_params: Dict[str, Any]
    train_seconds: float


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


def safe_float(value) -> float:
    if value is None:
        return 0.0
    text = str(value).strip()
    if not text:
        return 0.0
    try:
        return float(text)
    except ValueError:
        return 0.0


def normalize_label(value: str) -> str:
    text = (value or "").strip()
    return text if text else "Unknown"


def normalize_prediction_value(value: Any) -> str:
    if isinstance(value, np.ndarray):
        flat = value.reshape(-1)
        if flat.size == 1:
            return normalize_prediction_value(flat[0])
        return str(flat.tolist())
    if isinstance(value, (list, tuple)):
        if len(value) == 1:
            return normalize_prediction_value(value[0])
        return str(list(value))
    text = str(value).strip()
    if text.startswith("[") and text.endswith("]"):
        inner = text[1:-1].strip()
        if inner.startswith("'") and inner.endswith("'") and len(inner) >= 2:
            return inner[1:-1]
        if inner.startswith('"') and inner.endswith('"') and len(inner) >= 2:
            return inner[1:-1]
        if "," not in inner:
            return inner.strip("'\"")
    return text


def normalize_prediction_array(y_pred: Any) -> np.ndarray:
    arr = np.asarray(y_pred, dtype=object)
    if arr.ndim == 0:
        return np.array([normalize_prediction_value(arr.item())], dtype=object)
    if arr.ndim == 2 and arr.shape[1] == 1:
        arr = arr[:, 0]
    if arr.ndim > 1:
        flat_values = [normalize_prediction_value(item) for item in arr.tolist()]
    else:
        flat_values = [normalize_prediction_value(item) for item in arr.tolist()]
    return np.asarray(flat_values, dtype=object)


def parse_csv_rows(
    path: Path,
    max_rows: Optional[int],
    log_every: int,
) -> Tuple[List[Dict[str, str]], List[str]]:
    rows: List[Dict[str, str]] = []
    with path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        columns = list(reader.fieldnames or [])
        for idx, row in enumerate(reader, start=1):
            if max_rows is not None and idx > max_rows:
                break
            rows.append({k: (v if v is not None else "") for k, v in row.items()})
            if idx % log_every == 0:
                log(f"Load progress ({path.name}): rows={idx:,}")

    if not rows:
        raise RuntimeError(f"No rows found in {path}")

    log(f"Loaded {len(rows):,} rows from {path.name}")
    return rows, columns


def parse_feature_columns_arg(text: Optional[str]) -> Optional[List[str]]:
    if text is None:
        return None
    values = [part.strip() for part in text.split(",")]
    values = [v for v in values if v]
    if not values:
        return None
    deduped: List[str] = []
    seen = set()
    for value in values:
        if value not in seen:
            deduped.append(value)
            seen.add(value)
    return deduped


def infer_feature_columns(
    train_columns: Sequence[str],
    label_column: str,
    sample_origin_column: str,
    feature_columns_manual: Optional[List[str]],
) -> List[str]:
    if feature_columns_manual is not None:
        return feature_columns_manual

    excluded = {
        label_column,
        sample_origin_column,
        "SplitSet",
        "stage3_row_id",
        "user_id",
        "school",
        "EngagementLabel",
        "cluster",
        "first_activity_time",
        "last_activity_time",
    }
    feature_columns = [c for c in train_columns if c not in excluded]
    if not feature_columns:
        raise RuntimeError("No feature columns detected. Provide --feature-columns explicitly.")
    return feature_columns


def build_dataset_split(
    split_name: str,
    rows: List[Dict[str, str]],
    feature_columns: Sequence[str],
    label_column: str,
) -> DatasetSplit:
    X = np.zeros((len(rows), len(feature_columns)), dtype=np.float64)
    y = np.empty(len(rows), dtype=object)
    user_ids: List[str] = []
    row_ids: List[str] = []

    for i, row in enumerate(rows):
        y[i] = normalize_label(row.get(label_column, ""))
        user_ids.append((row.get("user_id") or "").strip())
        row_id = (row.get("stage3_row_id") or "").strip()
        row_ids.append(row_id if row_id else str(i))

        for j, col in enumerate(feature_columns):
            X[i, j] = safe_float(row.get(col))

    return DatasetSplit(
        split_name=split_name,
        X=X,
        y=y,
        user_ids=user_ids,
        row_ids=row_ids,
    )


def count_labels(y: np.ndarray) -> Dict[str, int]:
    values, counts = np.unique(y, return_counts=True)
    return {str(values[i]): int(counts[i]) for i in range(values.shape[0])}


def choose_cv_folds(y: np.ndarray, requested_folds: int) -> int:
    labels, counts = np.unique(y, return_counts=True)
    if labels.shape[0] < 2:
        return 0
    min_count = int(np.min(counts))
    if min_count < 2:
        return 0
    return max(2, min(int(requested_folds), min_count))


def metric_to_sklearn_scoring(metric: str) -> str:
    mapping = {
        "macro_f1": "f1_macro",
        "weighted_f1": "f1_weighted",
        "accuracy": "accuracy",
    }
    if metric not in mapping:
        raise RuntimeError(f"Unsupported primary metric: {metric}")
    return mapping[metric]


def build_model_specs(models_arg: str, seed: int) -> List[ModelSpec]:
    requested = [p.strip().lower() for p in models_arg.split(",")]
    requested = [m for m in requested if m]
    if not requested:
        raise RuntimeError("No models selected. Provide --models with at least one model.")

    specs: List[ModelSpec] = []

    for model_name in requested:
        if model_name == "logistic":
            specs.append(
                ModelSpec(
                    name="logistic",
                    estimator=make_pipeline(
                        StandardScaler(),
                        LogisticRegression(
                            max_iter=1000,
                            solver="lbfgs",
                        ),
                    ),
                    param_grid={
                        "logisticregression__C": [0.1, 1.0, 3.0],
                    },
                )
            )
        elif model_name == "random_forest":
            specs.append(
                ModelSpec(
                    name="random_forest",
                    estimator=RandomForestClassifier(
                        random_state=seed,
                        n_estimators=300,
                        n_jobs=1,
                    ),
                    param_grid={
                        "n_estimators": [200, 300],
                        "max_depth": [None, 12, 20],
                        "min_samples_leaf": [1, 2],
                    },
                )
            )
        elif model_name == "hist_gb":
            specs.append(
                ModelSpec(
                    name="hist_gb",
                    estimator=HistGradientBoostingClassifier(
                        random_state=seed,
                    ),
                    param_grid={
                        "learning_rate": [0.05, 0.1],
                        "max_depth": [None, 8],
                        "max_iter": [200, 300],
                    },
                )
            )
        elif model_name == "catboost":
            try:
                from catboost import CatBoostClassifier  # type: ignore
            except Exception:
                log("catboost is not installed. Skipping model=catboost")
                continue

            specs.append(
                ModelSpec(
                    name="catboost",
                    estimator=CatBoostClassifier(
                        loss_function="MultiClass",
                        random_seed=seed,
                        verbose=False,
                        allow_writing_files=False,
                        thread_count=1,
                    ),
                    param_grid={
                        "iterations": [200],
                        "depth": [6, 8],
                        "learning_rate": [0.05, 0.1],
                        "l2_leaf_reg": [3.0],
                    },
                )
            )
        elif model_name == "xgboost":
            try:
                from xgboost import XGBClassifier  # type: ignore
            except Exception:
                log("xgboost is not installed. Skipping model=xgboost")
                continue

            specs.append(
                ModelSpec(
                    name="xgboost",
                    estimator=XGBClassifier(
                        objective="multi:softprob",
                        eval_metric="mlogloss",
                        random_state=seed,
                        tree_method="hist",
                        n_estimators=300,
                        learning_rate=0.05,
                        max_depth=6,
                        subsample=0.9,
                        colsample_bytree=0.9,
                        reg_lambda=1.0,
                        n_jobs=1,
                    ),
                    param_grid={
                        "n_estimators": [200, 300],
                        "max_depth": [4, 6],
                        "learning_rate": [0.05, 0.1],
                    },
                )
            )
        else:
            raise RuntimeError(
                f"Unsupported model '{model_name}'. Supported: "
                "logistic,random_forest,hist_gb,catboost,xgboost"
            )

    if not specs:
        raise RuntimeError("No trainable models available after dependency checks.")

    return specs


def train_one_model(
    spec: ModelSpec,
    cfg: Phase4Config,
    X_train: np.ndarray,
    y_train: np.ndarray,
) -> TrainResult:
    started = time.time()
    scoring = metric_to_sklearn_scoring(cfg.primary_metric)

    cv_folds_used = choose_cv_folds(y_train, cfg.cv_folds)
    if cv_folds_used >= 2:
        cv = StratifiedKFold(
            n_splits=cv_folds_used,
            shuffle=True,
            random_state=cfg.seed,
        )
        search = GridSearchCV(
            estimator=spec.estimator,
            param_grid=spec.param_grid,
            scoring=scoring,
            cv=cv,
            n_jobs=cfg.n_jobs,
            refit=True,
        )
        search.fit(X_train, y_train)
        estimator = search.best_estimator_
        cv_best_score = float(search.best_score_)
        best_params = dict(search.best_params_)
    else:
        estimator = spec.estimator
        estimator.fit(X_train, y_train)
        cv_best_score = float("nan")
        best_params = {}

    elapsed = time.time() - started
    return TrainResult(
        model_name=spec.name,
        estimator=estimator,
        cv_best_score=cv_best_score,
        cv_folds_used=cv_folds_used,
        best_params=best_params,
        train_seconds=elapsed,
    )


def align_probabilities(
    y_prob: np.ndarray,
    model_classes: Sequence[str],
    target_classes: Sequence[str],
) -> np.ndarray:
    if list(model_classes) == list(target_classes):
        return y_prob

    out = np.zeros((y_prob.shape[0], len(target_classes)), dtype=np.float64)
    model_index = {str(model_classes[i]): i for i in range(len(model_classes))}
    for j, cls in enumerate(target_classes):
        idx = model_index.get(str(cls))
        if idx is not None:
            out[:, j] = y_prob[:, idx]
    return out


def compute_split_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_prob: Optional[np.ndarray],
    classes: Sequence[str],
) -> Tuple[Dict[str, float], List[Dict[str, Any]], np.ndarray]:
    macro_p, macro_r, macro_f1, _ = precision_recall_fscore_support(
        y_true,
        y_pred,
        average="macro",
        zero_division=0,
    )
    weighted_p, weighted_r, weighted_f1, _ = precision_recall_fscore_support(
        y_true,
        y_pred,
        average="weighted",
        zero_division=0,
    )

    per_p, per_r, per_f1, per_support = precision_recall_fscore_support(
        y_true,
        y_pred,
        labels=list(classes),
        average=None,
        zero_division=0,
    )

    class_rows: List[Dict[str, Any]] = []
    recall_low = float("nan")
    for i, label in enumerate(classes):
        label_text = str(label)
        class_rows.append(
            {
                "label": label_text,
                "precision": float(per_p[i]),
                "recall": float(per_r[i]),
                "f1": float(per_f1[i]),
                "support": int(per_support[i]),
            }
        )
        if label_text.lower() == "low":
            recall_low = float(per_r[i])

    roc_auc_macro = float("nan")
    roc_auc_weighted = float("nan")
    if y_prob is not None:
        unique_true = np.unique(y_true)
        if unique_true.shape[0] >= 2 and len(classes) >= 2:
            try:
                y_true_bin = label_binarize(y_true, classes=list(classes))
                if y_true_bin.ndim == 2 and y_true_bin.shape[1] >= 2:
                    roc_auc_macro = float(
                        roc_auc_score(
                            y_true_bin,
                            y_prob,
                            multi_class="ovr",
                            average="macro",
                        )
                    )
                    roc_auc_weighted = float(
                        roc_auc_score(
                            y_true_bin,
                            y_prob,
                            multi_class="ovr",
                            average="weighted",
                        )
                    )
            except Exception:
                pass

    summary = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_precision": float(macro_p),
        "macro_recall": float(macro_r),
        "macro_f1": float(macro_f1),
        "weighted_precision": float(weighted_p),
        "weighted_recall": float(weighted_r),
        "weighted_f1": float(weighted_f1),
        "roc_auc_ovr_macro": roc_auc_macro,
        "roc_auc_ovr_weighted": roc_auc_weighted,
        "recall_low": recall_low,
    }

    cm = confusion_matrix(y_true, y_pred, labels=list(classes))
    return summary, class_rows, cm


def evaluate_model_on_split(
    model_name: str,
    estimator: Any,
    split: DatasetSplit,
    classes: Sequence[str],
) -> Tuple[Dict[str, float], List[Dict[str, Any]], np.ndarray, np.ndarray, Optional[np.ndarray]]:
    y_pred = normalize_prediction_array(estimator.predict(split.X))

    y_prob: Optional[np.ndarray] = None
    if hasattr(estimator, "predict_proba"):
        raw_prob = estimator.predict_proba(split.X)
        model_classes = [str(c) for c in estimator.classes_]
        y_prob = align_probabilities(raw_prob, model_classes, classes)

    summary, class_rows, cm = compute_split_metrics(
        y_true=split.y,
        y_pred=y_pred,
        y_prob=y_prob,
        classes=classes,
    )

    return summary, class_rows, cm, y_pred, y_prob


def metric_field_for_selection(primary_metric: str) -> str:
    mapping = {
        "macro_f1": "macro_f1",
        "weighted_f1": "weighted_f1",
        "accuracy": "accuracy",
    }
    return mapping[primary_metric]


def write_csv(path: Path, fieldnames: Sequence[str], rows: Sequence[Dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(fieldnames))
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def write_feature_importance(
    path: Path,
    estimator: Any,
    feature_columns: Sequence[str],
) -> None:
    rows: List[Dict[str, Any]] = []

    if hasattr(estimator, "feature_importances_"):
        importance = np.asarray(estimator.feature_importances_, dtype=np.float64)
        for i, feature_name in enumerate(feature_columns):
            rows.append(
                {
                    "feature": feature_name,
                    "importance": float(importance[i]),
                    "method": "feature_importances_",
                }
            )
    elif hasattr(estimator, "coef_"):
        coef = np.asarray(estimator.coef_, dtype=np.float64)
        if coef.ndim == 1:
            importance = np.abs(coef)
        else:
            importance = np.mean(np.abs(coef), axis=0)
        for i, feature_name in enumerate(feature_columns):
            rows.append(
                {
                    "feature": feature_name,
                    "importance": float(importance[i]),
                    "method": "mean_abs_coef",
                }
            )

    if rows:
        rows.sort(key=lambda x: float(x["importance"]), reverse=True)
        write_csv(
            path=path,
            fieldnames=["feature", "importance", "method"],
            rows=rows,
        )
    else:
        write_csv(
            path=path,
            fieldnames=["feature", "importance", "method"],
            rows=[{"feature": "N/A", "importance": 0.0, "method": "not_available"}],
        )


def build_report_text(
    cfg: Phase4Config,
    feature_columns: Sequence[str],
    label_counts: Dict[str, Dict[str, int]],
    comparison_rows: Sequence[Dict[str, Any]],
    best_model_name: str,
    elapsed_seconds: float,
) -> str:
    lines: List[str] = []
    lines.append("Phase 4 - Supervised Training Report")
    lines.append("")
    lines.append("Configuration:")
    lines.append(f"- Train CSV                  : {cfg.train_csv}")
    lines.append(f"- Valid CSV                  : {cfg.valid_csv}")
    lines.append(f"- Test CSV                   : {cfg.test_csv}")
    lines.append(f"- Label column               : {cfg.label_column}")
    lines.append(f"- Models requested           : {cfg.models_arg}")
    lines.append(f"- Primary metric             : {cfg.primary_metric}")
    lines.append(f"- CV folds (requested)       : {cfg.cv_folds}")
    lines.append(f"- Seed                       : {cfg.seed}")
    lines.append("")

    lines.append("Label counts:")
    for split_name in ["train_modeling", "valid", "test"]:
        counts = label_counts.get(split_name, {})
        pieces = [f"{k}={counts[k]:,}" for k in sorted(counts.keys())]
        lines.append(f"- {split_name:<24}: {', '.join(pieces)}")
    lines.append("")

    lines.append(f"Feature columns ({len(feature_columns)}):")
    for col in feature_columns:
        lines.append(f"- {col}")
    lines.append("")

    lines.append("Model comparison (validation):")
    val_rows = [r for r in comparison_rows if r.get("split") == "valid"]
    metric_name = metric_field_for_selection(cfg.primary_metric)
    val_rows_sorted = sorted(
        val_rows,
        key=lambda r: float(r.get(metric_name, float("nan")))
        if np.isfinite(float(r.get(metric_name, float("nan"))) )
        else -1e18,
        reverse=True,
    )
    for row in val_rows_sorted:
        lines.append(
            f"- {row['model']}: {metric_name}={float(row[metric_name]):.6f}, "
            f"accuracy={float(row['accuracy']):.6f}, "
            f"recall_low={float(row['recall_low']):.6f}"
        )
    lines.append("")

    lines.append(f"Selected best model: {best_model_name}")
    lines.append(f"Elapsed seconds: {elapsed_seconds:.2f}")
    lines.append("")

    lines.append("Generated files:")
    lines.append(f"- {cfg.output_dir / 'phase4_model_comparison.csv'}")
    lines.append(f"- {cfg.output_dir / 'phase4_classification_metrics.csv'}")
    lines.append(f"- {cfg.output_dir / 'phase4_confusion_matrix.csv'}")
    lines.append(f"- {cfg.output_dir / 'phase4_best_model_predictions.csv'}")
    lines.append(f"- {cfg.output_dir / 'phase4_feature_importance.csv'}")
    lines.append(f"- {cfg.output_dir / 'phase4_best_model.pkl'}")
    lines.append(f"- {cfg.output_dir / 'phase4_training_report.txt'}")

    return "\n".join(lines) + "\n"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Phase 4: supervised model training and optimization from Phase 3 splits."
    )
    parser.add_argument("--results-dir", type=Path, default=Path("results"))
    parser.add_argument("--output-dir", type=Path, default=Path("results"))

    parser.add_argument("--train-input", type=Path, default=Path("stage3_train_modeling.csv"))
    parser.add_argument("--valid-input", type=Path, default=Path("stage3_valid.csv"))
    parser.add_argument("--test-input", type=Path, default=Path("stage3_test.csv"))

    parser.add_argument("--label-column", type=str, default="StandardLabelKMeans")
    parser.add_argument("--sample-origin-column", type=str, default="sample_origin")
    parser.add_argument(
        "--feature-columns",
        type=str,
        default=None,
        help="Comma-separated feature columns. Default: infer from train-input columns.",
    )

    parser.add_argument(
        "--models",
        type=str,
        default="logistic,random_forest,hist_gb",
        help=(
            "Comma-separated model list: "
            "logistic,random_forest,hist_gb,catboost,xgboost"
        ),
    )
    parser.add_argument(
        "--primary-metric",
        type=str,
        choices=["macro_f1", "weighted_f1", "accuracy"],
        default="macro_f1",
        help="Validation metric used to select the best model",
    )
    parser.add_argument("--cv-folds", type=int, default=3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--n-jobs", type=int, default=-1)

    parser.add_argument("--log-every", type=int, default=200000)
    parser.add_argument("--max-rows", type=int, default=None)
    return parser


def main() -> int:
    args = build_parser().parse_args()

    project_root = Path(__file__).resolve().parents[1]
    results_dir = resolve_path_arg(args.results_dir, project_root, project_root)
    output_dir = resolve_path_arg(args.output_dir, project_root, project_root)

    train_csv = resolve_path_arg(args.train_input, project_root, results_dir)
    valid_csv = resolve_path_arg(args.valid_input, project_root, results_dir)
    test_csv = resolve_path_arg(args.test_input, project_root, results_dir)

    cfg = Phase4Config(
        project_root=project_root,
        results_dir=results_dir,
        output_dir=output_dir,
        train_csv=train_csv,
        valid_csv=valid_csv,
        test_csv=test_csv,
        label_column=args.label_column,
        sample_origin_column=args.sample_origin_column,
        feature_columns_arg=args.feature_columns,
        models_arg=args.models,
        primary_metric=args.primary_metric,
        cv_folds=max(2, int(args.cv_folds)),
        seed=int(args.seed),
        n_jobs=int(args.n_jobs),
        log_every=max(1, int(args.log_every)),
        max_rows=args.max_rows,
    )

    try:
        started = time.time()
        log("Starting Phase 4: supervised model training")

        for path in [cfg.train_csv, cfg.valid_csv, cfg.test_csv]:
            if not path.exists():
                raise FileNotFoundError(f"Required input not found: {path}")

        train_rows, train_columns = parse_csv_rows(cfg.train_csv, cfg.max_rows, cfg.log_every)
        valid_rows, valid_columns = parse_csv_rows(cfg.valid_csv, cfg.max_rows, cfg.log_every)
        test_rows, test_columns = parse_csv_rows(cfg.test_csv, cfg.max_rows, cfg.log_every)

        if cfg.label_column not in train_columns:
            raise RuntimeError(f"Label column not found in train-input: {cfg.label_column}")
        if cfg.label_column not in valid_columns:
            raise RuntimeError(f"Label column not found in valid-input: {cfg.label_column}")
        if cfg.label_column not in test_columns:
            raise RuntimeError(f"Label column not found in test-input: {cfg.label_column}")

        feature_columns_manual = parse_feature_columns_arg(cfg.feature_columns_arg)
        feature_columns = infer_feature_columns(
            train_columns=train_columns,
            label_column=cfg.label_column,
            sample_origin_column=cfg.sample_origin_column,
            feature_columns_manual=feature_columns_manual,
        )

        missing_valid = [c for c in feature_columns if c not in valid_columns]
        missing_test = [c for c in feature_columns if c not in test_columns]
        if missing_valid:
            log(f"Valid split missing {len(missing_valid)} feature columns. Fill missing as 0.0")
        if missing_test:
            log(f"Test split missing {len(missing_test)} feature columns. Fill missing as 0.0")

        train_split = build_dataset_split(
            split_name="train_modeling",
            rows=train_rows,
            feature_columns=feature_columns,
            label_column=cfg.label_column,
        )
        valid_split = build_dataset_split(
            split_name="valid",
            rows=valid_rows,
            feature_columns=feature_columns,
            label_column=cfg.label_column,
        )
        test_split = build_dataset_split(
            split_name="test",
            rows=test_rows,
            feature_columns=feature_columns,
            label_column=cfg.label_column,
        )

        classes = sorted({str(v) for v in train_split.y.tolist()})
        if len(classes) < 2:
            raise RuntimeError("Training data has <2 classes. Cannot train classifier.")

        for split in [valid_split, test_split]:
            split_labels = sorted({str(v) for v in split.y.tolist()})
            unknown = [label for label in split_labels if label not in classes]
            if unknown:
                raise RuntimeError(
                    f"Split {split.split_name} has labels not seen in train: {unknown}"
                )

        specs = build_model_specs(cfg.models_arg, cfg.seed)
        log(f"Training models: {', '.join([s.name for s in specs])}")

        comparison_rows: List[Dict[str, Any]] = []
        class_metric_rows: List[Dict[str, Any]] = []
        confusion_rows: List[Dict[str, Any]] = []

        trained: Dict[str, TrainResult] = {}

        for spec in specs:
            log(f"Training model={spec.name}")
            train_result = train_one_model(
                spec=spec,
                cfg=cfg,
                X_train=train_split.X,
                y_train=train_split.y,
            )
            trained[spec.name] = train_result

            for split in [valid_split, test_split]:
                summary, class_rows, cm, _, _ = evaluate_model_on_split(
                    model_name=spec.name,
                    estimator=train_result.estimator,
                    split=split,
                    classes=classes,
                )

                comparison_row = {
                    "model": spec.name,
                    "split": split.split_name,
                    "cv_best_score": train_result.cv_best_score,
                    "cv_folds_used": train_result.cv_folds_used,
                    "train_seconds": train_result.train_seconds,
                    "best_params": json.dumps(train_result.best_params, ensure_ascii=True, sort_keys=True),
                }
                comparison_row.update(summary)
                comparison_rows.append(comparison_row)

                for row in class_rows:
                    class_metric_rows.append(
                        {
                            "model": spec.name,
                            "split": split.split_name,
                            "label": row["label"],
                            "precision": row["precision"],
                            "recall": row["recall"],
                            "f1": row["f1"],
                            "support": row["support"],
                        }
                    )

                for i, true_label in enumerate(classes):
                    for j, pred_label in enumerate(classes):
                        confusion_rows.append(
                            {
                                "model": spec.name,
                                "split": split.split_name,
                                "true_label": true_label,
                                "pred_label": pred_label,
                                "count": int(cm[i, j]),
                            }
                        )

        metric_field = metric_field_for_selection(cfg.primary_metric)
        valid_rows = [r for r in comparison_rows if r["split"] == "valid"]
        if not valid_rows:
            raise RuntimeError("No validation rows found for model selection")

        valid_rows_sorted = sorted(
            valid_rows,
            key=lambda r: float(r[metric_field]) if np.isfinite(float(r[metric_field])) else -1e18,
            reverse=True,
        )
        best_model_name = str(valid_rows_sorted[0]["model"])
        best_train_result = trained[best_model_name]
        best_estimator = best_train_result.estimator

        prediction_rows: List[Dict[str, Any]] = []
        for split in [valid_split, test_split]:
            _, _, _, y_pred, y_prob = evaluate_model_on_split(
                model_name=best_model_name,
                estimator=best_estimator,
                split=split,
                classes=classes,
            )

            for i in range(split.X.shape[0]):
                pred_label = normalize_prediction_value(y_pred[i])
                row: Dict[str, Any] = {
                    "model": best_model_name,
                    "split": split.split_name,
                    "user_id": split.user_ids[i],
                    "stage3_row_id": split.row_ids[i],
                    "true_label": str(split.y[i]),
                    "pred_label": pred_label,
                    "is_correct": int(str(split.y[i]) == pred_label),
                }
                if y_prob is not None:
                    for j, cls in enumerate(classes):
                        row[f"prob_{cls}"] = float(y_prob[i, j])
                prediction_rows.append(row)

        cfg.output_dir.mkdir(parents=True, exist_ok=True)

        write_csv(
            path=cfg.output_dir / "phase4_model_comparison.csv",
            fieldnames=[
                "model",
                "split",
                "cv_best_score",
                "cv_folds_used",
                "train_seconds",
                "best_params",
                "accuracy",
                "macro_precision",
                "macro_recall",
                "macro_f1",
                "weighted_precision",
                "weighted_recall",
                "weighted_f1",
                "roc_auc_ovr_macro",
                "roc_auc_ovr_weighted",
                "recall_low",
            ],
            rows=comparison_rows,
        )

        write_csv(
            path=cfg.output_dir / "phase4_classification_metrics.csv",
            fieldnames=["model", "split", "label", "precision", "recall", "f1", "support"],
            rows=class_metric_rows,
        )

        write_csv(
            path=cfg.output_dir / "phase4_confusion_matrix.csv",
            fieldnames=["model", "split", "true_label", "pred_label", "count"],
            rows=confusion_rows,
        )

        prediction_fields = [
            "model",
            "split",
            "user_id",
            "stage3_row_id",
            "true_label",
            "pred_label",
            "is_correct",
        ] + [f"prob_{cls}" for cls in classes]
        write_csv(
            path=cfg.output_dir / "phase4_best_model_predictions.csv",
            fieldnames=prediction_fields,
            rows=prediction_rows,
        )

        write_feature_importance(
            path=cfg.output_dir / "phase4_feature_importance.csv",
            estimator=best_estimator,
            feature_columns=feature_columns,
        )

        model_bundle = {
            "model_name": best_model_name,
            "trained_at": now_text(),
            "label_column": cfg.label_column,
            "classes": classes,
            "feature_columns": list(feature_columns),
            "primary_metric": cfg.primary_metric,
            "cv_best_score": best_train_result.cv_best_score,
            "best_params": best_train_result.best_params,
            "estimator": best_estimator,
        }
        with (cfg.output_dir / "phase4_best_model.pkl").open("wb") as f:
            pickle.dump(model_bundle, f)

        label_counts = {
            "train_modeling": count_labels(train_split.y),
            "valid": count_labels(valid_split.y),
            "test": count_labels(test_split.y),
        }
        report_text = build_report_text(
            cfg=cfg,
            feature_columns=feature_columns,
            label_counts=label_counts,
            comparison_rows=comparison_rows,
            best_model_name=best_model_name,
            elapsed_seconds=time.time() - started,
        )
        with (cfg.output_dir / "phase4_training_report.txt").open("w", encoding="utf-8") as f:
            f.write(report_text)

        log(f"Selected best model: {best_model_name}")
        log(f"Phase 4 completed in {time.time() - started:.2f}s")
        return 0
    except Exception as exc:
        log(f"FAILED: {exc}")
        print(traceback.format_exc())
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
