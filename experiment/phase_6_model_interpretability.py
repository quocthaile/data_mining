#!/usr/bin/env python3
"""
Phase 6: model interpretability and explanation reporting.

Scenario alignment goals:
- Explain the selected supervised model from Phase 4.
- Provide global importance and local explanations for selected samples.
- Prefer SHAP-style explanations for CatBoost models.
- Keep the final consolidated summary up to date.

Default inputs (from Phase 4 outputs):
- results/phase4_best_model.pkl
- results/phase4_best_model_predictions.csv
- results/stage3_test.csv

Generated outputs:
- results/phase6_global_importance.csv
- results/phase6_classwise_importance.csv
- results/phase6_local_contributions.csv
- results/phase6_interpretability_report.txt
- results/final_summary_report.txt
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import pickle
import time
import traceback
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np


@dataclass
class Phase6Config:
    project_root: Path
    results_dir: Path
    output_dir: Path
    model_bundle_input: Path
    predictions_input: Path
    test_input: Path
    final_summary_input: Path
    final_summary_output: Path
    global_importance_csv: Path
    classwise_importance_csv: Path
    local_contributions_csv: Path
    interpretability_report_txt: Path
    local_error_samples: int
    local_correct_samples: int
    top_features: int
    log_every: int
    max_rows: Optional[int]


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


def safe_float(value: Any, default: float = 0.0) -> float:
    if value is None:
        return default
    text = str(value).strip()
    if not text:
        return default
    try:
        return float(text)
    except ValueError:
        return default


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


def write_csv(path: Path, fieldnames: Sequence[str], rows: Sequence[Dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(fieldnames))
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def read_text_lines(path: Path) -> List[str]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8") as f:
        return [line.rstrip("\n") for line in f]


def write_text(path: Path, text: str) -> None:
    with path.open("w", encoding="utf-8") as f:
        f.write(text)


def normalize_label(value: Any) -> str:
    text = (str(value) if value is not None else "").strip()
    return text if text else "Unknown"


def normalize_prediction_label(value: Any) -> str:
    text = (str(value) if value is not None else "").strip()
    if text.startswith("[") and text.endswith("]"):
        inner = text[1:-1].strip()
        if inner.startswith("'") and inner.endswith("'") and len(inner) >= 2:
            return inner[1:-1]
        if inner.startswith('"') and inner.endswith('"') and len(inner) >= 2:
            return inner[1:-1]
        if "," not in inner:
            return inner.strip("'\"")
    return text if text else "Unknown"


def parse_feature_columns(bundle: Dict[str, Any]) -> List[str]:
    features = bundle.get("feature_columns")
    if not isinstance(features, list) or not features:
        raise RuntimeError("Model bundle does not contain feature_columns")
    return [str(col) for col in features]


def build_feature_matrix(rows: Sequence[Dict[str, str]], feature_columns: Sequence[str]) -> np.ndarray:
    X = np.zeros((len(rows), len(feature_columns)), dtype=np.float64)
    for i, row in enumerate(rows):
        for j, col in enumerate(feature_columns):
            X[i, j] = safe_float(row.get(col), 0.0)
    return X


def sample_identifier(row: Dict[str, str], fallback_index: int) -> str:
    row_id = (row.get("stage3_row_id") or "").strip()
    if row_id:
        return row_id
    user_id = (row.get("user_id") or "").strip()
    if user_id:
        return user_id
    return str(fallback_index)


def feature_column_index(feature_columns: Sequence[str]) -> Dict[str, int]:
    return {name: idx for idx, name in enumerate(feature_columns)}


def top_feature_strings(
    feature_columns: Sequence[str],
    contributions: np.ndarray,
    feature_values: Optional[np.ndarray] = None,
    limit: int = 5,
) -> Tuple[str, str]:
    positive = []
    negative = []
    for idx, feature_name in enumerate(feature_columns):
        value = float(contributions[idx])
        feature_value = float(feature_values[idx]) if feature_values is not None else float("nan")
        item = (feature_name, value, feature_value)
        if value >= 0:
            positive.append(item)
        else:
            negative.append(item)

    positive.sort(key=lambda item: item[1], reverse=True)
    negative.sort(key=lambda item: item[1])

    def format_items(items: Sequence[Tuple[str, float, float]]) -> str:
        formatted: List[str] = []
        for name, contribution, feature_value in items[:limit]:
            if math.isfinite(feature_value):
                formatted.append(f"{name}:{contribution:+.6f}@{feature_value:.6f}")
            else:
                formatted.append(f"{name}:{contribution:+.6f}")
        return "; ".join(formatted)

    return format_items(positive), format_items(negative)


def append_or_replace_phase6_section(summary_path: Path, section_text: str) -> None:
    if summary_path.exists():
        lines = read_text_lines(summary_path)
        cutoff = None
        for idx, line in enumerate(lines):
            if line.strip().startswith("Phase 6 - Model Interpretability"):
                cutoff = idx
                break
        if cutoff is not None:
            lines = lines[:cutoff]
        base_text = "\n".join(lines).rstrip()
        if base_text:
            base_text += "\n\n"
        updated = base_text + section_text.lstrip("\n")
    else:
        updated = section_text
    write_text(summary_path, updated)


def load_predictions_map(path: Path, max_rows: Optional[int], log_every: int) -> Dict[str, Dict[str, str]]:
    rows, _ = parse_csv_rows(path, max_rows=max_rows, log_every=log_every)
    out: Dict[str, Dict[str, str]] = {}
    for idx, row in enumerate(rows):
        if (row.get("split") or "").strip() != "test":
            continue
        key = (row.get("stage3_row_id") or "").strip() or (row.get("user_id") or "").strip() or str(idx)
        normalized = dict(row)
        normalized["true_label"] = normalize_label(normalized.get("true_label"))
        normalized["pred_label"] = normalize_prediction_label(normalized.get("pred_label"))
        out[key] = normalized
    if not out:
        raise RuntimeError("No test rows found in predictions input")
    return out


def load_model_bundle(path: Path) -> Dict[str, Any]:
    with path.open("rb") as f:
        bundle = pickle.load(f)
    if not isinstance(bundle, dict):
        raise RuntimeError("Model bundle must be a dict")
    required = ["estimator", "feature_columns", "classes", "model_name"]
    for key in required:
        if key not in bundle:
            raise RuntimeError(f"Model bundle missing required key: {key}")
    return bundle


def build_catboost_shap(
    estimator: Any,
    X: np.ndarray,
    feature_columns: Sequence[str],
) -> Tuple[np.ndarray, np.ndarray, str]:
    from catboost import Pool  # type: ignore

    pool = Pool(X, feature_names=list(feature_columns))
    shap_values = np.asarray(estimator.get_feature_importance(type="ShapValues", data=pool))
    if shap_values.ndim != 3:
        raise RuntimeError(f"Unexpected SHAP shape from CatBoost: {shap_values.shape}")

    local_contribs = shap_values[:, :, :-1]
    base_values = shap_values[:, :, -1]
    return local_contribs, base_values, "catboost_shap"


def build_linear_contributions(
    estimator: Any,
    X: np.ndarray,
    feature_columns: Sequence[str],
) -> Tuple[np.ndarray, np.ndarray, str]:
    X_used = X
    final_estimator = estimator

    if hasattr(estimator, "steps"):
        try:
            X_used = estimator[:-1].transform(X)
        except Exception:
            X_used = X
        final_estimator = estimator.steps[-1][1]

    coef = np.asarray(getattr(final_estimator, "coef_", np.zeros((1, X_used.shape[1]))), dtype=np.float64)
    intercept = np.atleast_1d(
        np.asarray(getattr(final_estimator, "intercept_", np.zeros(coef.shape[0])), dtype=np.float64)
    )
    if coef.ndim == 1:
        coef = coef.reshape(1, -1)
    n_classes = coef.shape[0]
    local_contribs = X_used[:, None, :] * coef[None, :, :]
    base_values = np.tile(intercept.reshape(1, -1), (X_used.shape[0], 1))
    return local_contribs, base_values, "linear_coef"


def build_tree_proxy_contributions(
    estimator: Any,
    X: np.ndarray,
    n_classes: int,
) -> Tuple[np.ndarray, np.ndarray, str]:
    importances = np.asarray(getattr(estimator, "feature_importances_", np.zeros(X.shape[1])), dtype=np.float64)
    if importances.sum() > 0:
        importances = importances / importances.sum()
    centered = X - np.mean(X, axis=0, keepdims=True)
    local = centered[:, None, :] * importances[None, None, :]
    local_contribs = np.repeat(local, repeats=max(1, n_classes), axis=1)
    base_values = np.zeros((X.shape[0], max(1, n_classes)), dtype=np.float64)
    return local_contribs, base_values, "tree_importance_proxy"


def compute_contributions(
    bundle: Dict[str, Any],
    X: np.ndarray,
    feature_columns: Sequence[str],
) -> Tuple[np.ndarray, np.ndarray, str]:
    estimator = bundle["estimator"]
    model_name = str(bundle.get("model_name", "")).lower()
    classes = list(bundle.get("classes", []))
    n_classes = max(1, len(classes))

    if model_name == "catboost" and hasattr(estimator, "get_feature_importance"):
        try:
            return build_catboost_shap(estimator, X, feature_columns)
        except Exception as exc:
            log(f"CatBoost SHAP unavailable, falling back: {exc}")

    if hasattr(estimator, "coef_") or hasattr(estimator, "steps"):
        try:
            return build_linear_contributions(estimator, X, feature_columns)
        except Exception as exc:
            log(f"Linear contribution fallback failed, using tree proxy: {exc}")

    if hasattr(estimator, "feature_importances_"):
        return build_tree_proxy_contributions(estimator, X, n_classes)

    local_contribs = np.zeros((X.shape[0], n_classes, X.shape[1]), dtype=np.float64)
    base_values = np.zeros((X.shape[0], n_classes), dtype=np.float64)
    return local_contribs, base_values, "not_available"


def compute_global_importance(
    local_contribs: np.ndarray,
    class_indices: np.ndarray,
    feature_columns: Sequence[str],
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    n_samples, n_classes, n_features = local_contribs.shape
    classwise_rows: List[Dict[str, Any]] = []
    for class_idx in range(n_classes):
        class_contribs = local_contribs[:, class_idx, :]
        mean_abs = np.mean(np.abs(class_contribs), axis=0)
        for feature_idx, feature_name in enumerate(feature_columns):
            classwise_rows.append(
                {
                    "class_index": class_idx,
                    "feature": feature_name,
                    "mean_abs_contribution": float(mean_abs[feature_idx]),
                }
            )

    selected = local_contribs[np.arange(n_samples), class_indices, :]
    mean_abs = np.mean(np.abs(selected), axis=0)
    mean_signed = np.mean(selected, axis=0)
    global_rows = [
        {
            "feature": feature_columns[i],
            "mean_abs_contribution": float(mean_abs[i]),
            "mean_signed_contribution": float(mean_signed[i]),
        }
        for i in range(n_features)
    ]
    global_rows.sort(key=lambda row: row["mean_abs_contribution"], reverse=True)
    for rank, row in enumerate(global_rows, start=1):
        row["rank"] = rank
    return global_rows, classwise_rows


def compute_local_rows(
    rows: Sequence[Dict[str, str]],
    predictions: Dict[str, Dict[str, str]],
    local_contribs: np.ndarray,
    base_values: np.ndarray,
    feature_columns: Sequence[str],
    classes: Sequence[str],
    local_error_samples: int,
    local_correct_samples: int,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    class_to_idx = {str(cls): idx for idx, cls in enumerate(classes)}

    matched_rows: List[Dict[str, Any]] = []
    for idx, row in enumerate(rows):
        key = sample_identifier(row, idx)
        pred_row = predictions.get(key)
        if pred_row is None:
            continue
        if (pred_row.get("split") or "").strip() != "test":
            continue
        pred_label = normalize_prediction_label(pred_row.get("pred_label"))
        true_label = normalize_label(pred_row.get("true_label"))
        is_correct = str(pred_row.get("is_correct") or "0").strip() in {"1", "true", "True"}
        confidence = safe_float(pred_row.get(f"prob_{pred_label}", 0.0)) if pred_label else 0.0
        class_idx = class_to_idx.get(pred_label, 0)
        matched_rows.append(
            {
                "row_index": idx,
                "row_id": key,
                "user_id": (row.get("user_id") or "").strip(),
                "split": (pred_row.get("split") or "").strip(),
                "true_label": true_label,
                "pred_label": pred_label,
                "is_correct": is_correct,
                "confidence": confidence,
                "selected_class": pred_label if pred_label else classes[0] if classes else "Unknown",
                "class_idx": class_idx,
            }
        )

    errors = [r for r in matched_rows if not r["is_correct"]]
    corrects = [r for r in matched_rows if r["is_correct"]]
    errors.sort(key=lambda r: r["confidence"], reverse=True)
    corrects.sort(key=lambda r: r["confidence"])

    selected: List[Dict[str, Any]] = []
    selected.extend(errors[: max(0, local_error_samples)])
    selected.extend(corrects[: max(0, local_correct_samples)])
    if not selected:
        selected = matched_rows[: max(1, local_error_samples + local_correct_samples)]

    long_rows: List[Dict[str, Any]] = []
    summary_rows: List[Dict[str, Any]] = []

    for sample_rank, sample in enumerate(selected, start=1):
        row_index = int(sample["row_index"])
        class_idx = int(sample["class_idx"])
        contrib_vector = np.asarray(local_contribs[row_index, class_idx, :], dtype=np.float64)
        base_value = float(base_values[row_index, class_idx]) if base_values.ndim == 2 and base_values.shape[1] > class_idx else 0.0
        feature_values = np.asarray([safe_float(rows[row_index].get(col), 0.0) for col in feature_columns], dtype=np.float64)
        order = np.argsort(np.abs(contrib_vector))[::-1]

        top_positive, top_negative = top_feature_strings(
            feature_columns=feature_columns,
            contributions=contrib_vector,
            feature_values=feature_values,
            limit=5,
        )

        summary_rows.append(
            {
                "sample_rank": sample_rank,
                "row_id": sample["row_id"],
                "user_id": sample["user_id"],
                "split": sample["split"],
                "true_label": sample["true_label"],
                "pred_label": sample["pred_label"],
                "selected_class": sample["selected_class"],
                "sample_type": "misclassified" if not sample["is_correct"] else "correct",
                "confidence": float(sample["confidence"]),
                "base_value": base_value,
                "top_positive_contributions": top_positive,
                "top_negative_contributions": top_negative,
            }
        )

        for contrib_rank, feature_idx in enumerate(order, start=1):
            contribution = float(contrib_vector[feature_idx])
            long_rows.append(
                {
                    "sample_rank": sample_rank,
                    "sample_type": "misclassified" if not sample["is_correct"] else "correct",
                    "row_id": sample["row_id"],
                    "user_id": sample["user_id"],
                    "split": sample["split"],
                    "true_label": sample["true_label"],
                    "pred_label": sample["pred_label"],
                    "selected_class": sample["selected_class"],
                    "confidence": float(sample["confidence"]),
                    "base_value": base_value,
                    "feature": feature_columns[feature_idx],
                    "feature_value": float(feature_values[feature_idx]),
                    "contribution": contribution,
                    "abs_contribution": abs(contribution),
                    "contribution_rank": contrib_rank,
                    "sign": "positive" if contribution >= 0 else "negative",
                }
            )

    return summary_rows, long_rows


def build_report_text(
    cfg: Phase6Config,
    bundle: Dict[str, Any],
    method: str,
    global_rows: Sequence[Dict[str, Any]],
    classwise_rows: Sequence[Dict[str, Any]],
    sample_summary_rows: Sequence[Dict[str, Any]],
    elapsed_seconds: float,
) -> str:
    model_name = str(bundle.get("model_name", "N/A"))
    classes = [str(c) for c in bundle.get("classes", [])]
    feature_columns = parse_feature_columns(bundle)

    lines: List[str] = []
    lines.append("Phase 6 - Model Interpretability Report")
    lines.append("")
    lines.append("Configuration:")
    lines.append(f"- Model bundle                : {cfg.model_bundle_input}")
    lines.append(f"- Predictions CSV             : {cfg.predictions_input}")
    lines.append(f"- Test CSV                    : {cfg.test_input}")
    lines.append(f"- Final summary               : {cfg.final_summary_output}")
    lines.append(f"- Interpretability method     : {method}")
    lines.append(f"- Selected model              : {model_name}")
    lines.append(f"- Classes                     : {', '.join(classes) if classes else 'N/A'}")
    lines.append(f"- Feature count               : {len(feature_columns)}")
    lines.append(f"- Local error samples         : {cfg.local_error_samples}")
    lines.append(f"- Local correct samples       : {cfg.local_correct_samples}")
    lines.append(f"- Global top features         : {cfg.top_features}")
    lines.append("")

    lines.append("Global importance (top features):")
    for row in global_rows[: cfg.top_features]:
        lines.append(
            f"- {row['rank']}. {row['feature']}: mean_abs={float(row['mean_abs_contribution']):.6f}, "
            f"mean_signed={float(row['mean_signed_contribution']):.6f}"
        )
    lines.append("")

    lines.append("Class-wise importance summary:")
    class_names = classes if classes else ["Unknown"]
    for class_idx, class_name in enumerate(class_names):
        class_rows = [r for r in classwise_rows if int(r["class_index"]) == class_idx]
        class_rows.sort(key=lambda r: float(r["mean_abs_contribution"]), reverse=True)
        top_class_rows = class_rows[:3]
        top_text = "; ".join(
            [f"{r['feature']}={float(r['mean_abs_contribution']):.6f}" for r in top_class_rows]
        )
        lines.append(f"- {class_name}: {top_text}")
    lines.append("")

    lines.append("Local explanation samples:")
    if sample_summary_rows:
        for row in sample_summary_rows:
            lines.append(
                f"- #{row['sample_rank']} {row['sample_type']} row={row['row_id']} user={row['user_id']} "
                f"true={row['true_label']} pred={row['pred_label']} conf={float(row['confidence']):.6f}"
            )
            lines.append(f"  + Positive: {row['top_positive_contributions'] or 'N/A'}")
            lines.append(f"  + Negative: {row['top_negative_contributions'] or 'N/A'}")
    else:
        lines.append("- No local explanations available")
    lines.append("")

    lines.append("Generated files:")
    lines.append(f"- {cfg.global_importance_csv}")
    lines.append(f"- {cfg.classwise_importance_csv}")
    lines.append(f"- {cfg.local_contributions_csv}")
    lines.append(f"- {cfg.interpretability_report_txt}")
    lines.append(f"- {cfg.final_summary_output}")
    lines.append("")
    lines.append(f"Elapsed seconds: {elapsed_seconds:.2f}")

    return "\n".join(lines) + "\n"


def build_final_summary_section(
    cfg: Phase6Config,
    bundle: Dict[str, Any],
    method: str,
    global_rows: Sequence[Dict[str, Any]],
    sample_summary_rows: Sequence[Dict[str, Any]],
) -> str:
    model_name = str(bundle.get("model_name", "N/A"))
    top_features = [row["feature"] for row in global_rows[: cfg.top_features]]
    lines: List[str] = []
    lines.append("")
    lines.append("Phase 6 - Model Interpretability")
    lines.append("=" * 100)
    lines.append(f"Selected model               : {model_name}")
    lines.append(f"Interpretability method      : {method}")
    lines.append(f"Top global features           : {', '.join(top_features) if top_features else 'N/A'}")
    lines.append(f"Local explanations exported   : {len(sample_summary_rows)}")
    lines.append("Generated files:")
    lines.append(f"- {cfg.global_importance_csv}")
    lines.append(f"- {cfg.classwise_importance_csv}")
    lines.append(f"- {cfg.local_contributions_csv}")
    lines.append(f"- {cfg.interpretability_report_txt}")
    lines.append(f"- {cfg.final_summary_output}")
    lines.append("")
    return "\n".join(lines)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Phase 6: model interpretability and explanation reporting from Phase 4 outputs."
    )
    parser.add_argument("--results-dir", type=Path, default=Path("results"))
    parser.add_argument("--output-dir", type=Path, default=Path("results"))
    parser.add_argument("--model-bundle-input", type=Path, default=Path("phase4_best_model.pkl"))
    parser.add_argument(
        "--predictions-input",
        type=Path,
        default=Path("phase4_best_model_predictions.csv"),
    )
    parser.add_argument("--test-input", type=Path, default=Path("stage3_test.csv"))
    parser.add_argument("--final-summary-input", type=Path, default=Path("final_summary_report.txt"))
    parser.add_argument("--final-summary-output", type=Path, default=Path("final_summary_report.txt"))

    parser.add_argument("--local-error-samples", type=int, default=3)
    parser.add_argument("--local-correct-samples", type=int, default=3)
    parser.add_argument("--top-features", type=int, default=10)

    parser.add_argument("--log-every", type=int, default=200000)
    parser.add_argument("--max-rows", type=int, default=None)
    return parser


def main() -> int:
    args = build_parser().parse_args()

    project_root = Path(__file__).resolve().parents[1]
    results_dir = resolve_path_arg(args.results_dir, project_root, project_root)
    output_dir = resolve_path_arg(args.output_dir, project_root, project_root)

    model_bundle_input = resolve_path_arg(args.model_bundle_input, project_root, results_dir)
    predictions_input = resolve_path_arg(args.predictions_input, project_root, results_dir)
    test_input = resolve_path_arg(args.test_input, project_root, results_dir)
    final_summary_input = resolve_path_arg(args.final_summary_input, project_root, results_dir)
    final_summary_output = resolve_path_arg(args.final_summary_output, project_root, results_dir)

    cfg = Phase6Config(
        project_root=project_root,
        results_dir=results_dir,
        output_dir=output_dir,
        model_bundle_input=model_bundle_input,
        predictions_input=predictions_input,
        test_input=test_input,
        final_summary_input=final_summary_input,
        final_summary_output=final_summary_output,
        global_importance_csv=output_dir / "phase6_global_importance.csv",
        classwise_importance_csv=output_dir / "phase6_classwise_importance.csv",
        local_contributions_csv=output_dir / "phase6_local_contributions.csv",
        interpretability_report_txt=output_dir / "phase6_interpretability_report.txt",
        local_error_samples=max(0, int(args.local_error_samples)),
        local_correct_samples=max(0, int(args.local_correct_samples)),
        top_features=max(1, int(args.top_features)),
        log_every=max(1, int(args.log_every)),
        max_rows=args.max_rows,
    )

    try:
        started = time.time()
        log("Starting Phase 6: model interpretability")

        for required in [cfg.model_bundle_input, cfg.predictions_input, cfg.test_input]:
            if not required.exists():
                raise FileNotFoundError(f"Required input not found: {required}")

        bundle = load_model_bundle(cfg.model_bundle_input)
        feature_columns = parse_feature_columns(bundle)
        classes = [str(c) for c in bundle.get("classes", [])]
        estimator = bundle["estimator"]

        test_rows, test_columns = parse_csv_rows(cfg.test_input, cfg.max_rows, cfg.log_every)
        missing_features = [col for col in feature_columns if col not in test_columns]
        if missing_features:
            log(f"Test input missing {len(missing_features)} expected features. Missing columns will use 0.0")

        predictions = load_predictions_map(cfg.predictions_input, cfg.max_rows, cfg.log_every)

        X = build_feature_matrix(test_rows, feature_columns)
        local_contribs, base_values, method = compute_contributions(bundle, X, feature_columns)

        if local_contribs.ndim != 3:
            raise RuntimeError(f"Unexpected local contribution shape: {local_contribs.shape}")

        class_to_idx = {str(cls): idx for idx, cls in enumerate(classes)}
        class_indices = np.zeros(X.shape[0], dtype=int)
        for i, row in enumerate(test_rows):
            key = sample_identifier(row, i)
            pred_row = predictions.get(key)
            pred_label = normalize_prediction_label(pred_row.get("pred_label")) if pred_row else ""
            class_indices[i] = class_to_idx.get(pred_label, 0)

        global_rows, classwise_rows = compute_global_importance(
            local_contribs=local_contribs,
            class_indices=class_indices,
            feature_columns=feature_columns,
        )

        sample_summary_rows, local_rows = compute_local_rows(
            rows=test_rows,
            predictions=predictions,
            local_contribs=local_contribs,
            base_values=base_values,
            feature_columns=feature_columns,
            classes=classes,
            local_error_samples=cfg.local_error_samples,
            local_correct_samples=cfg.local_correct_samples,
        )

        cfg.output_dir.mkdir(parents=True, exist_ok=True)

        write_csv(
            path=cfg.global_importance_csv,
            fieldnames=["rank", "feature", "mean_abs_contribution", "mean_signed_contribution"],
            rows=global_rows,
        )
        write_csv(
            path=cfg.classwise_importance_csv,
            fieldnames=["class_index", "feature", "mean_abs_contribution"],
            rows=classwise_rows,
        )
        write_csv(
            path=cfg.local_contributions_csv,
            fieldnames=[
                "sample_rank",
                "sample_type",
                "row_id",
                "user_id",
                "split",
                "true_label",
                "pred_label",
                "selected_class",
                "confidence",
                "base_value",
                "feature",
                "feature_value",
                "contribution",
                "abs_contribution",
                "contribution_rank",
                "sign",
            ],
            rows=local_rows,
        )

        report_text = build_report_text(
            cfg=cfg,
            bundle=bundle,
            method=method,
            global_rows=global_rows,
            classwise_rows=classwise_rows,
            sample_summary_rows=sample_summary_rows,
            elapsed_seconds=time.time() - started,
        )
        write_text(cfg.interpretability_report_txt, report_text)

        summary_section = build_final_summary_section(
            cfg=cfg,
            bundle=bundle,
            method=method,
            global_rows=global_rows,
            sample_summary_rows=sample_summary_rows,
        )
        append_or_replace_phase6_section(cfg.final_summary_output, summary_section)

        log(f"Selected model: {bundle.get('model_name', 'N/A')}")
        log(f"Phase 6 completed in {time.time() - started:.2f}s")
        return 0
    except Exception as exc:
        log(f"FAILED: {exc}")
        print(traceback.format_exc())
        return 1


if __name__ == "__main__":
    raise SystemExit(main())