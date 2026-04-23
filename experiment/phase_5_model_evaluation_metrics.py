#!/usr/bin/env python3
"""
Phase 5: model evaluation metrics and final reporting from Phase 4 outputs.

Scenario alignment goals:
- Select best supervised model based on validation metric.
- Summarize evaluation on validation and test splits.
- Check practical thresholds (ROC-AUC, recall of Low class).
- Export model-centric artifacts for reporting/dashboard use.

Default inputs (from Phase 4 outputs):
- results/phase4_model_comparison.csv
- results/phase4_classification_metrics.csv
- results/phase4_confusion_matrix.csv
- results/phase4_feature_importance.csv

Generated outputs:
- results/phase5_model_selection_summary.csv
- results/phase5_best_model_class_metrics.csv
- results/phase5_best_model_confusion_matrix.csv
- results/phase5_top_features.csv
- results/phase5_metric_checks.csv
- results/phase5_evaluation_report.txt
- results/final_summary_report.txt
"""

from __future__ import annotations

import argparse
import csv
import math
import time
import traceback
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple


@dataclass
class Phase5Config:
    project_root: Path
    results_dir: Path
    output_dir: Path
    model_comparison_csv: Path
    class_metrics_csv: Path
    confusion_csv: Path
    feature_importance_csv: Path
    phase2_report_txt: Path
    phase3_report_txt: Path
    phase4_report_txt: Path
    final_summary_txt: Path
    selection_metric: str
    top_features: int
    auc_threshold: float
    recall_low_threshold: float
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


def safe_float(value: Any, default: float = float("nan")) -> float:
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


def metric_field(selection_metric: str) -> str:
    mapping = {
        "macro_f1": "macro_f1",
        "weighted_f1": "weighted_f1",
        "accuracy": "accuracy",
    }
    return mapping[selection_metric]


def finite_or_negative_inf(value: float) -> float:
    if math.isfinite(value):
        return value
    return -1e18


def select_best_model(
    comparison_rows: Sequence[Dict[str, str]],
    selection_metric: str,
) -> Tuple[str, Dict[str, str], Optional[Dict[str, str]]]:
    metric_key = metric_field(selection_metric)

    valid_rows = [row for row in comparison_rows if (row.get("split") or "").strip() == "valid"]
    if not valid_rows:
        raise RuntimeError("No rows with split=valid found in phase4_model_comparison.csv")

    sorted_valid = sorted(
        valid_rows,
        key=lambda row: (
            finite_or_negative_inf(safe_float(row.get(metric_key))),
            finite_or_negative_inf(safe_float(row.get("accuracy"))),
        ),
        reverse=True,
    )
    best_valid = sorted_valid[0]
    best_model = (best_valid.get("model") or "").strip()
    if not best_model:
        raise RuntimeError("Best valid row does not have model name")

    test_rows = [
        row
        for row in comparison_rows
        if (row.get("model") or "").strip() == best_model
        and (row.get("split") or "").strip() == "test"
    ]
    best_test = test_rows[0] if test_rows else None
    return best_model, best_valid, best_test


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


def collect_highlight_lines(lines: Sequence[str], keywords: Sequence[str], limit: int) -> List[str]:
    out: List[str] = []
    keyword_lower = [k.lower() for k in keywords]
    for raw in lines:
        text = raw.strip()
        if not text:
            continue
        low = text.lower()
        if any(k in low for k in keyword_lower):
            out.append(text)
        if len(out) >= limit:
            break
    return out


def format_optional_float(value: Any) -> str:
    val = safe_float(value)
    if math.isfinite(val):
        return f"{val:.6f}"
    return "N/A"


def build_final_summary_report_text(
    cfg: Phase5Config,
    summary_row: Dict[str, Any],
    checks: Sequence[Dict[str, Any]],
    top_feature_rows: Sequence[Dict[str, Any]],
    phase2_lines: Sequence[str],
    phase3_lines: Sequence[str],
    phase4_lines: Sequence[str],
    phase5_lines: Sequence[str],
    elapsed_seconds: float,
) -> str:
    selected_model = str(summary_row.get("selected_model", "N/A"))
    selection_metric = str(summary_row.get("selection_metric", cfg.selection_metric))

    pass_count = sum(1 for row in checks if str(row.get("status", "")) == "pass")
    warn_count = sum(1 for row in checks if str(row.get("status", "")) == "warn")
    missing_count = sum(
        1
        for row in checks
        if str(row.get("status", "")) in {"missing_split", "not_available"}
    )

    def pretty_line(text: str) -> str:
        out = text.strip()
        if out.startswith("- "):
            out = out[2:].strip()
        return out

    phase2_highlights = collect_highlight_lines(
        phase2_lines,
        [
            "Silhouette",
            "R2",
            "Accuracy",
            "Macro F1",
            "Weighted F1",
        ],
        limit=6,
    )
    phase3_highlights = collect_highlight_lines(
        phase3_lines,
        [
            "Split strategy",
            "Train rows",
            "Valid rows",
            "Test rows",
            "Method",
            "Train modeling counts",
        ],
        limit=6,
    )
    phase4_highlights = collect_highlight_lines(
        phase4_lines,
        [
            "Models requested",
            "Primary metric",
            "Selected best model",
            "macro_f1",
            "accuracy=",
            "recall_low",
        ],
        limit=7,
    )
    phase5_highlights = collect_highlight_lines(
        phase5_lines,
        [
            "Selected best model",
            "valid:",
            "test:",
            "[pass]",
            "[warn]",
            "Top features",
        ],
        limit=10,
    )

    lines: List[str] = []
    lines.append("Final Consolidated Experiment Report")
    lines.append("=" * 100)
    lines.append(f"Generated at                    : {now_text()}")
    lines.append(f"Selection metric                : {selection_metric}")
    lines.append(f"Selected best model             : {selected_model}")
    lines.append(f"Valid {selection_metric:<23}: {format_optional_float(summary_row.get('valid_selection_metric'))}")
    lines.append(f"Test {selection_metric:<24}: {format_optional_float(summary_row.get('test_selection_metric'))}")
    lines.append(f"Valid AUC macro                 : {format_optional_float(summary_row.get('valid_auc_macro'))}")
    lines.append(f"Test AUC macro                  : {format_optional_float(summary_row.get('test_auc_macro'))}")
    lines.append(f"Valid Recall Low                : {format_optional_float(summary_row.get('valid_recall_low'))}")
    lines.append(f"Test Recall Low                 : {format_optional_float(summary_row.get('test_recall_low'))}")
    lines.append(
        f"Threshold checks (pass/warn/na) : {pass_count}/{warn_count}/{missing_count}"
    )
    lines.append("")

    lines.append("Phase 2 highlights:")
    if phase2_highlights:
        for item in phase2_highlights:
            lines.append(f"- {pretty_line(item)}")
    else:
        lines.append("- (phase2 report not found or no highlight lines matched)")
    lines.append("")

    lines.append("Phase 3 highlights:")
    if phase3_highlights:
        for item in phase3_highlights:
            lines.append(f"- {pretty_line(item)}")
    else:
        lines.append("- (phase3 report not found or no highlight lines matched)")
    lines.append("")

    lines.append("Phase 4 highlights:")
    if phase4_highlights:
        for item in phase4_highlights:
            lines.append(f"- {pretty_line(item)}")
    else:
        lines.append("- (phase4 report not found or no highlight lines matched)")
    lines.append("")

    lines.append("Phase 5 highlights:")
    if phase5_highlights:
        for item in phase5_highlights:
            lines.append(f"- {pretty_line(item)}")
    else:
        lines.append("- (phase5 report not found or no highlight lines matched)")
    lines.append("")

    lines.append("Top features (from selected model):")
    if top_feature_rows:
        for row in top_feature_rows:
            lines.append(
                f"- {row.get('feature', 'N/A')}: importance={format_optional_float(row.get('importance'))}"
            )
    else:
        lines.append("- (no feature importance rows found)")
    lines.append("")

    lines.append("Generated files:")
    lines.append(f"- {cfg.output_dir / 'phase5_model_selection_summary.csv'}")
    lines.append(f"- {cfg.output_dir / 'phase5_best_model_class_metrics.csv'}")
    lines.append(f"- {cfg.output_dir / 'phase5_best_model_confusion_matrix.csv'}")
    lines.append(f"- {cfg.output_dir / 'phase5_top_features.csv'}")
    lines.append(f"- {cfg.output_dir / 'phase5_metric_checks.csv'}")
    lines.append(f"- {cfg.output_dir / 'phase5_evaluation_report.txt'}")
    lines.append(f"- {cfg.final_summary_txt}")
    lines.append("")
    lines.append(f"Elapsed seconds                : {elapsed_seconds:.2f}")

    return "\n".join(lines) + "\n"


def make_metric_checks(
    best_valid: Dict[str, str],
    best_test: Optional[Dict[str, str]],
    auc_threshold: float,
    recall_low_threshold: float,
) -> List[Dict[str, Any]]:
    checks: List[Dict[str, Any]] = []

    def add_check(
        split_name: str,
        metric_name: str,
        row: Optional[Dict[str, str]],
        threshold: float,
        note: str,
    ) -> None:
        if row is None:
            checks.append(
                {
                    "split": split_name,
                    "metric": metric_name,
                    "value": float("nan"),
                    "threshold": threshold,
                    "status": "missing_split",
                    "note": "Split row is missing in comparison input",
                }
            )
            return

        value = safe_float(row.get(metric_name))
        if not math.isfinite(value):
            status = "not_available"
            note_text = note + "; metric is NaN or missing"
        elif value >= threshold:
            status = "pass"
            note_text = note
        else:
            status = "warn"
            note_text = note

        checks.append(
            {
                "split": split_name,
                "metric": metric_name,
                "value": value,
                "threshold": threshold,
                "status": status,
                "note": note_text,
            }
        )

    add_check(
        split_name="valid",
        metric_name="roc_auc_ovr_macro",
        row=best_valid,
        threshold=auc_threshold,
        note="AUC target from scenario is commonly >= 0.85",
    )
    add_check(
        split_name="test",
        metric_name="roc_auc_ovr_macro",
        row=best_test,
        threshold=auc_threshold,
        note="AUC on test reflects deployment realism",
    )
    add_check(
        split_name="valid",
        metric_name="recall_low",
        row=best_valid,
        threshold=recall_low_threshold,
        note="Recall of Low class is critical for early-risk detection",
    )
    add_check(
        split_name="test",
        metric_name="recall_low",
        row=best_test,
        threshold=recall_low_threshold,
        note="Recall of Low class on test must remain acceptable",
    )

    return checks


def build_report_text(
    cfg: Phase5Config,
    selected_model: str,
    selection_metric: str,
    best_valid: Dict[str, str],
    best_test: Optional[Dict[str, str]],
    checks: Sequence[Dict[str, Any]],
    top_feature_rows: Sequence[Dict[str, Any]],
    elapsed_seconds: float,
) -> str:
    def fmt_metric(row: Optional[Dict[str, str]], key: str) -> str:
        if row is None:
            return "N/A"
        value = safe_float(row.get(key))
        if math.isfinite(value):
            return f"{value:.6f}"
        return "N/A"

    lines: List[str] = []
    lines.append("Phase 5 - Model Evaluation Metrics Report")
    lines.append("")
    lines.append("Configuration:")
    lines.append(f"- Model comparison CSV        : {cfg.model_comparison_csv}")
    lines.append(f"- Class metrics CSV           : {cfg.class_metrics_csv}")
    lines.append(f"- Confusion matrix CSV        : {cfg.confusion_csv}")
    lines.append(f"- Feature importance CSV      : {cfg.feature_importance_csv}")
    lines.append(f"- Selection metric            : {selection_metric}")
    lines.append(f"- Top features exported       : {cfg.top_features}")
    lines.append(f"- AUC threshold               : {cfg.auc_threshold:.3f}")
    lines.append(f"- Recall Low threshold        : {cfg.recall_low_threshold:.3f}")
    lines.append("")

    lines.append(f"Selected best model: {selected_model}")
    lines.append("")

    lines.append("Selected model metrics:")
    for split_name, row in [("valid", best_valid), ("test", best_test)]:
        metric_items: List[Tuple[str, str]] = [
            (selection_metric, fmt_metric(row, metric_field(selection_metric))),
            ("accuracy", fmt_metric(row, "accuracy")),
            ("macro_f1", fmt_metric(row, "macro_f1")),
            ("weighted_f1", fmt_metric(row, "weighted_f1")),
            ("auc_macro", fmt_metric(row, "roc_auc_ovr_macro")),
            ("recall_low", fmt_metric(row, "recall_low")),
        ]
        deduped_items: List[Tuple[str, str]] = []
        seen_metric_names = set()
        for name, value in metric_items:
            if name in seen_metric_names:
                continue
            deduped_items.append((name, value))
            seen_metric_names.add(name)

        metric_text = ", ".join([f"{name}={value}" for name, value in deduped_items])
        lines.append(
            f"- {split_name}: {metric_text}"
        )
    lines.append("")

    lines.append("Threshold checks:")
    for row in checks:
        value = row.get("value")
        value_text = f"{float(value):.6f}" if isinstance(value, float) and math.isfinite(value) else "N/A"
        lines.append(
            f"- [{row['status']}] split={row['split']}, metric={row['metric']}, "
            f"value={value_text}, threshold={float(row['threshold']):.3f}"
        )
    lines.append("")

    lines.append("Top features:")
    for feature_row in top_feature_rows:
        lines.append(
            f"- {feature_row.get('feature', 'N/A')}: importance={safe_float(feature_row.get('importance'), 0.0):.6f}"
        )
    lines.append("")

    lines.append("Generated files:")
    lines.append(f"- {cfg.output_dir / 'phase5_model_selection_summary.csv'}")
    lines.append(f"- {cfg.output_dir / 'phase5_best_model_class_metrics.csv'}")
    lines.append(f"- {cfg.output_dir / 'phase5_best_model_confusion_matrix.csv'}")
    lines.append(f"- {cfg.output_dir / 'phase5_top_features.csv'}")
    lines.append(f"- {cfg.output_dir / 'phase5_metric_checks.csv'}")
    lines.append(f"- {cfg.output_dir / 'phase5_evaluation_report.txt'}")
    lines.append(f"- {cfg.final_summary_txt}")
    lines.append("")
    lines.append(f"Elapsed seconds: {elapsed_seconds:.2f}")

    return "\n".join(lines) + "\n"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Phase 5: model-centric evaluation metrics and reporting from Phase 4 outputs."
    )
    parser.add_argument("--results-dir", type=Path, default=Path("results"))
    parser.add_argument("--output-dir", type=Path, default=Path("results"))

    parser.add_argument(
        "--model-comparison-input",
        type=Path,
        default=Path("phase4_model_comparison.csv"),
    )
    parser.add_argument(
        "--class-metrics-input",
        type=Path,
        default=Path("phase4_classification_metrics.csv"),
    )
    parser.add_argument(
        "--confusion-input",
        type=Path,
        default=Path("phase4_confusion_matrix.csv"),
    )
    parser.add_argument(
        "--feature-importance-input",
        type=Path,
        default=Path("phase4_feature_importance.csv"),
    )
    parser.add_argument(
        "--phase2-report-input",
        type=Path,
        default=Path("phase2_kmeans_validation_report.txt"),
    )
    parser.add_argument(
        "--phase3-report-input",
        type=Path,
        default=Path("stage3_split_report.txt"),
    )
    parser.add_argument(
        "--phase4-report-input",
        type=Path,
        default=Path("phase4_training_report.txt"),
    )
    parser.add_argument(
        "--final-summary-output",
        type=Path,
        default=Path("final_summary_report.txt"),
    )

    parser.add_argument(
        "--selection-metric",
        type=str,
        choices=["macro_f1", "weighted_f1", "accuracy"],
        default="macro_f1",
    )
    parser.add_argument("--top-features", type=int, default=10)
    parser.add_argument("--auc-threshold", type=float, default=0.85)
    parser.add_argument("--recall-low-threshold", type=float, default=0.80)

    parser.add_argument("--log-every", type=int, default=200000)
    parser.add_argument("--max-rows", type=int, default=None)
    return parser


def main() -> int:
    args = build_parser().parse_args()

    project_root = Path(__file__).resolve().parents[1]
    results_dir = resolve_path_arg(args.results_dir, project_root, project_root)
    output_dir = resolve_path_arg(args.output_dir, project_root, project_root)

    model_comparison_csv = resolve_path_arg(args.model_comparison_input, project_root, results_dir)
    class_metrics_csv = resolve_path_arg(args.class_metrics_input, project_root, results_dir)
    confusion_csv = resolve_path_arg(args.confusion_input, project_root, results_dir)
    feature_importance_csv = resolve_path_arg(args.feature_importance_input, project_root, results_dir)
    phase2_report_txt = resolve_path_arg(args.phase2_report_input, project_root, results_dir)
    phase3_report_txt = resolve_path_arg(args.phase3_report_input, project_root, results_dir)
    phase4_report_txt = resolve_path_arg(args.phase4_report_input, project_root, results_dir)
    final_summary_txt = resolve_path_arg(args.final_summary_output, project_root, output_dir)

    cfg = Phase5Config(
        project_root=project_root,
        results_dir=results_dir,
        output_dir=output_dir,
        model_comparison_csv=model_comparison_csv,
        class_metrics_csv=class_metrics_csv,
        confusion_csv=confusion_csv,
        feature_importance_csv=feature_importance_csv,
        phase2_report_txt=phase2_report_txt,
        phase3_report_txt=phase3_report_txt,
        phase4_report_txt=phase4_report_txt,
        final_summary_txt=final_summary_txt,
        selection_metric=args.selection_metric,
        top_features=max(1, int(args.top_features)),
        auc_threshold=float(args.auc_threshold),
        recall_low_threshold=float(args.recall_low_threshold),
        log_every=max(1, int(args.log_every)),
        max_rows=args.max_rows,
    )

    try:
        started = time.time()
        log("Starting Phase 5: model evaluation metrics")

        for required in [
            cfg.model_comparison_csv,
            cfg.class_metrics_csv,
            cfg.confusion_csv,
            cfg.feature_importance_csv,
        ]:
            if not required.exists():
                raise FileNotFoundError(f"Required input not found: {required}")

        comparison_rows, _ = parse_csv_rows(cfg.model_comparison_csv, cfg.max_rows, cfg.log_every)
        class_metric_rows, _ = parse_csv_rows(cfg.class_metrics_csv, cfg.max_rows, cfg.log_every)
        confusion_rows, _ = parse_csv_rows(cfg.confusion_csv, cfg.max_rows, cfg.log_every)
        feature_rows, _ = parse_csv_rows(cfg.feature_importance_csv, cfg.max_rows, cfg.log_every)

        best_model, best_valid, best_test = select_best_model(
            comparison_rows=comparison_rows,
            selection_metric=cfg.selection_metric,
        )

        filtered_class_rows = [
            row for row in class_metric_rows if (row.get("model") or "").strip() == best_model
        ]
        filtered_confusion_rows = [
            row for row in confusion_rows if (row.get("model") or "").strip() == best_model
        ]

        sorted_features = sorted(
            feature_rows,
            key=lambda row: finite_or_negative_inf(safe_float(row.get("importance"))),
            reverse=True,
        )
        top_feature_rows = sorted_features[: cfg.top_features]

        checks = make_metric_checks(
            best_valid=best_valid,
            best_test=best_test,
            auc_threshold=cfg.auc_threshold,
            recall_low_threshold=cfg.recall_low_threshold,
        )

        cfg.output_dir.mkdir(parents=True, exist_ok=True)

        summary_row = {
            "selected_model": best_model,
            "selection_metric": cfg.selection_metric,
            "valid_selection_metric": safe_float(best_valid.get(metric_field(cfg.selection_metric))),
            "test_selection_metric": safe_float(best_test.get(metric_field(cfg.selection_metric))) if best_test else float("nan"),
            "valid_accuracy": safe_float(best_valid.get("accuracy")),
            "test_accuracy": safe_float(best_test.get("accuracy")) if best_test else float("nan"),
            "valid_macro_f1": safe_float(best_valid.get("macro_f1")),
            "test_macro_f1": safe_float(best_test.get("macro_f1")) if best_test else float("nan"),
            "valid_weighted_f1": safe_float(best_valid.get("weighted_f1")),
            "test_weighted_f1": safe_float(best_test.get("weighted_f1")) if best_test else float("nan"),
            "valid_auc_macro": safe_float(best_valid.get("roc_auc_ovr_macro")),
            "test_auc_macro": safe_float(best_test.get("roc_auc_ovr_macro")) if best_test else float("nan"),
            "valid_recall_low": safe_float(best_valid.get("recall_low")),
            "test_recall_low": safe_float(best_test.get("recall_low")) if best_test else float("nan"),
            "auc_threshold": cfg.auc_threshold,
            "recall_low_threshold": cfg.recall_low_threshold,
        }

        write_csv(
            path=cfg.output_dir / "phase5_model_selection_summary.csv",
            fieldnames=[
                "selected_model",
                "selection_metric",
                "valid_selection_metric",
                "test_selection_metric",
                "valid_accuracy",
                "test_accuracy",
                "valid_macro_f1",
                "test_macro_f1",
                "valid_weighted_f1",
                "test_weighted_f1",
                "valid_auc_macro",
                "test_auc_macro",
                "valid_recall_low",
                "test_recall_low",
                "auc_threshold",
                "recall_low_threshold",
            ],
            rows=[summary_row],
        )

        write_csv(
            path=cfg.output_dir / "phase5_best_model_class_metrics.csv",
            fieldnames=["model", "split", "label", "precision", "recall", "f1", "support"],
            rows=filtered_class_rows,
        )

        write_csv(
            path=cfg.output_dir / "phase5_best_model_confusion_matrix.csv",
            fieldnames=["model", "split", "true_label", "pred_label", "count"],
            rows=filtered_confusion_rows,
        )

        write_csv(
            path=cfg.output_dir / "phase5_top_features.csv",
            fieldnames=["feature", "importance", "method"],
            rows=top_feature_rows,
        )

        write_csv(
            path=cfg.output_dir / "phase5_metric_checks.csv",
            fieldnames=["split", "metric", "value", "threshold", "status", "note"],
            rows=checks,
        )

        report_text = build_report_text(
            cfg=cfg,
            selected_model=best_model,
            selection_metric=cfg.selection_metric,
            best_valid=best_valid,
            best_test=best_test,
            checks=checks,
            top_feature_rows=top_feature_rows,
            elapsed_seconds=time.time() - started,
        )
        phase5_report_path = cfg.output_dir / "phase5_evaluation_report.txt"
        with phase5_report_path.open("w", encoding="utf-8") as f:
            f.write(report_text)

        phase2_lines = read_text_lines(cfg.phase2_report_txt)
        phase3_lines = read_text_lines(cfg.phase3_report_txt)
        phase4_lines = read_text_lines(cfg.phase4_report_txt)
        phase5_lines = read_text_lines(phase5_report_path)

        final_summary_text = build_final_summary_report_text(
            cfg=cfg,
            summary_row=summary_row,
            checks=checks,
            top_feature_rows=top_feature_rows,
            phase2_lines=phase2_lines,
            phase3_lines=phase3_lines,
            phase4_lines=phase4_lines,
            phase5_lines=phase5_lines,
            elapsed_seconds=time.time() - started,
        )
        with cfg.final_summary_txt.open("w", encoding="utf-8") as f:
            f.write(final_summary_text)

        log(f"Selected best model: {best_model}")
        log(f"Phase 5 completed in {time.time() - started:.2f}s")
        return 0
    except Exception as exc:
        log(f"FAILED: {exc}")
        print(traceback.format_exc())
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
