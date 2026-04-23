#!/usr/bin/env python3
"""
Phase 2: K-Means ground-truth initialization + label-based validation.

Scenario/paper alignment:
- Uses weighted weekly engagement features from step 2 output (via step 3).
- Uses percentile-based pre-labeling thresholds (default p33/p66).
- Maps clusters to standard labels by mean engagement level (step 5).
- Adds internal + external validation metrics inspired by the paper:
  - Internal: WSS, BSS, TSS, R2, Silhouette
  - External: Accuracy, Precision, Recall, F1 between pre-label and cluster label

This phase wraps and reuses:
- step_3_engagement_report.py
- step_5_init_standard_labels_kmeans.py
"""

from __future__ import annotations

import argparse
import csv
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, silhouette_score
from sklearn.preprocessing import StandardScaler


LABEL_ORDER = ["Low", "Medium", "High"]
FEATURE_COLUMNS = [
    "num_courses",
    "problem_total",
    "problem_accuracy",
    "avg_attempts",
    "video_count",
    "watched_hours",
    "forum_total",
    "avg_speed",
]


@dataclass
class Phase2Config:
    project_root: Path
    scripts_dir: Path
    results_dir: Path
    combined_csv: Path
    weekly_csv: Path
    step3_results_csv: Path
    step3_weights_csv: Path
    step3_centers_csv: Path
    step3_report_txt: Path
    step5_labeled_csv: Path
    step5_cluster_map_csv: Path
    step5_report_txt: Path
    output_external_csv: Path
    output_internal_csv: Path
    output_report_txt: Path
    clusters: int
    q_low: float
    q_high: float
    batch_size: int
    log_every: int
    silhouette_sample_size: int
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


def run_command(command: List[str], cwd: Path, label: str) -> None:
    log(f"Running {label}")
    proc = subprocess.run(command, cwd=str(cwd), check=False)
    if proc.returncode != 0:
        raise RuntimeError(f"{label} failed with exit code {proc.returncode}")


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


def ordered_labels(labels: Sequence[str]) -> List[str]:
    existing = list(dict.fromkeys(labels))
    out: List[str] = [x for x in LABEL_ORDER if x in existing]
    out.extend(sorted(x for x in existing if x not in LABEL_ORDER))
    return out


def load_labeled_rows(path: Path, max_rows: Optional[int]) -> Tuple[List[str], List[str], Dict[str, int]]:
    y_true: List[str] = []
    y_pred: List[str] = []
    cluster_counts: Dict[str, int] = {}

    with path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for idx, row in enumerate(reader, start=1):
            if max_rows is not None and idx > max_rows:
                break

            true_label = (row.get("EngagementLabel") or "Unknown").strip() or "Unknown"
            pred_label = (row.get("StandardLabelKMeans") or "Unknown").strip() or "Unknown"
            cluster = (row.get("cluster") or "NA").strip() or "NA"

            y_true.append(true_label)
            y_pred.append(pred_label)
            cluster_counts[cluster] = cluster_counts.get(cluster, 0) + 1

    return y_true, y_pred, cluster_counts


def compute_external_metrics(y_true: List[str], y_pred: List[str]) -> Dict[str, Dict[str, float]]:
    labels = ordered_labels(y_true + y_pred)
    if not y_true:
        raise RuntimeError("No rows found for external validation.")

    precision, recall, f1, support = precision_recall_fscore_support(
        y_true,
        y_pred,
        labels=labels,
        zero_division=0,
    )

    per_label: Dict[str, Dict[str, float]] = {}
    for i, label in enumerate(labels):
        per_label[label] = {
            "precision": float(precision[i]),
            "recall": float(recall[i]),
            "f1": float(f1[i]),
            "support": int(support[i]),
        }

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
    acc = accuracy_score(y_true, y_pred)

    summary = {
        "accuracy": float(acc),
        "macro_precision": float(macro_p),
        "macro_recall": float(macro_r),
        "macro_f1": float(macro_f1),
        "weighted_precision": float(weighted_p),
        "weighted_recall": float(weighted_r),
        "weighted_f1": float(weighted_f1),
    }

    return {"per_label": per_label, "summary": summary}


def load_features_and_clusters(path: Path, max_rows: Optional[int]) -> Tuple[np.ndarray, np.ndarray]:
    X_rows: List[List[float]] = []
    clusters: List[int] = []

    with path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for idx, row in enumerate(reader, start=1):
            if max_rows is not None and idx > max_rows:
                break

            try:
                cluster_id = int(float(row.get("cluster") or 0))
            except ValueError:
                continue

            feature_vec = [safe_float(row.get(col)) for col in FEATURE_COLUMNS]
            X_rows.append(feature_vec)
            clusters.append(cluster_id)

    if not X_rows:
        raise RuntimeError("No rows found to compute internal validation metrics.")

    return np.array(X_rows, dtype=np.float64), np.array(clusters, dtype=np.int64)


def compute_internal_metrics(
    X_raw: np.ndarray,
    clusters: np.ndarray,
    silhouette_sample_size: int,
) -> Dict[str, float]:
    scaler = StandardScaler()
    X = scaler.fit_transform(X_raw)

    unique_clusters = np.unique(clusters)
    n_samples = X.shape[0]
    n_clusters = unique_clusters.shape[0]

    global_center = X.mean(axis=0)

    cluster_centers: Dict[int, np.ndarray] = {}
    for cid in unique_clusters:
        mask = clusters == cid
        cluster_centers[int(cid)] = X[mask].mean(axis=0)

    wss = 0.0
    for cid in unique_clusters:
        mask = clusters == cid
        diff = X[mask] - cluster_centers[int(cid)]
        wss += float(np.sum(diff * diff))

    tss = float(np.sum((X - global_center) ** 2))
    bss = max(0.0, tss - wss)
    r2 = (bss / tss) if tss > 0 else 0.0

    if n_clusters < 2 or n_samples < 3:
        silhouette = float("nan")
    else:
        sample_size = min(max(100, silhouette_sample_size), n_samples)
        if sample_size < n_samples:
            rng = np.random.default_rng(42)
            idx = rng.choice(n_samples, size=sample_size, replace=False)
            X_eval = X[idx]
            clusters_eval = clusters[idx]
        else:
            X_eval = X
            clusters_eval = clusters

        silhouette = float(
            silhouette_score(
                X_eval,
                clusters_eval,
                metric="euclidean",
            )
        )

    return {
        "samples": float(n_samples),
        "clusters": float(n_clusters),
        "wss": wss,
        "bss": bss,
        "tss": tss,
        "r2": r2,
        "silhouette": silhouette,
    }


def write_external_csv(path: Path, metrics: Dict[str, Dict[str, float]]) -> None:
    per_label = metrics["per_label"]
    summary = metrics["summary"]

    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["scope", "label", "precision", "recall", "f1", "support", "accuracy"])

        for label in ordered_labels(list(per_label.keys())):
            row = per_label[label]
            writer.writerow(
                [
                    "per_label",
                    label,
                    round(row["precision"], 6),
                    round(row["recall"], 6),
                    round(row["f1"], 6),
                    int(row["support"]),
                    "",
                ]
            )

        writer.writerow(
            [
                "summary",
                "macro",
                round(summary["macro_precision"], 6),
                round(summary["macro_recall"], 6),
                round(summary["macro_f1"], 6),
                "",
                round(summary["accuracy"], 6),
            ]
        )
        writer.writerow(
            [
                "summary",
                "weighted",
                round(summary["weighted_precision"], 6),
                round(summary["weighted_recall"], 6),
                round(summary["weighted_f1"], 6),
                "",
                round(summary["accuracy"], 6),
            ]
        )


def write_internal_csv(path: Path, metrics: Dict[str, float]) -> None:
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["metric", "value"])
        for key in ["samples", "clusters", "wss", "bss", "tss", "r2", "silhouette"]:
            value = metrics[key]
            writer.writerow([key, round(value, 6) if isinstance(value, float) else value])


def write_report(
    path: Path,
    cfg: Phase2Config,
    external_metrics: Dict[str, Dict[str, float]],
    internal_metrics: Dict[str, float],
    cluster_counts: Dict[str, int],
    elapsed: float,
) -> None:
    summary = external_metrics["summary"]

    with path.open("w", encoding="utf-8") as f:
        f.write("Phase 2 - K-Means and Label-Based Validation Report\n")
        f.write("=" * 95 + "\n")
        f.write(f"Generated at                    : {now_text()}\n")
        f.write(f"Combined CSV                   : {cfg.combined_csv}\n")
        f.write(f"Weekly CSV                     : {cfg.weekly_csv}\n")
        f.write(f"Step 3 result CSV              : {cfg.step3_results_csv}\n")
        f.write(f"Step 5 labeled CSV             : {cfg.step5_labeled_csv}\n")
        f.write(f"Elapsed seconds                : {elapsed:.2f}\n")

        f.write("\nScenario equations used:\n")
        f.write("- Activity weight: w_a = sum_{i,w}(x_{i,a,w}) / (S * N)\n")
        f.write("- Weighted engagement score: E_i = sum_w sum_a (w_a * x_{i,a,w})\n")
        f.write("- Pre-label thresholds: Low <= p_low, Medium in (p_low, p_high], High > p_high\n")
        f.write(f"- p_low={cfg.q_low:.4f}, p_high={cfg.q_high:.4f}\n")

        f.write("\nInternal validation (cluster quality):\n")
        f.write(f"- Samples                       : {int(internal_metrics['samples']):,}\n")
        f.write(f"- Clusters                      : {int(internal_metrics['clusters'])}\n")
        f.write(f"- WSS                           : {internal_metrics['wss']:.6f}\n")
        f.write(f"- BSS                           : {internal_metrics['bss']:.6f}\n")
        f.write(f"- TSS                           : {internal_metrics['tss']:.6f}\n")
        f.write(f"- R2 (=BSS/TSS)                 : {internal_metrics['r2']:.6f}\n")
        f.write(f"- Silhouette (sampled)          : {internal_metrics['silhouette']:.6f}\n")

        f.write("\nExternal validation (pre-label vs K-Means label):\n")
        f.write(f"- Accuracy                      : {summary['accuracy']:.6f}\n")
        f.write(f"- Macro Precision               : {summary['macro_precision']:.6f}\n")
        f.write(f"- Macro Recall                  : {summary['macro_recall']:.6f}\n")
        f.write(f"- Macro F1                      : {summary['macro_f1']:.6f}\n")
        f.write(f"- Weighted Precision            : {summary['weighted_precision']:.6f}\n")
        f.write(f"- Weighted Recall               : {summary['weighted_recall']:.6f}\n")
        f.write(f"- Weighted F1                   : {summary['weighted_f1']:.6f}\n")

        f.write("\nCluster distribution:\n")
        for cluster in sorted(cluster_counts.keys(), key=lambda x: int(x) if x.isdigit() else x):
            f.write(f"- Cluster {cluster:>4}: {cluster_counts[cluster]:>10,}\n")

        f.write("\nGenerated validation files:\n")
        f.write(f"- {cfg.output_external_csv}\n")
        f.write(f"- {cfg.output_internal_csv}\n")
        f.write(f"- {cfg.output_report_txt}\n")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Phase 2: K-Means clustering + label-based validation."
    )
    parser.add_argument("--results-dir", type=Path, default=Path("results"))
    parser.add_argument("--combined-input", type=Path, default=Path("combined_user_metrics.csv"))
    parser.add_argument("--weekly-input", type=Path, default=Path("step2_user_week_activity.csv"))
    parser.add_argument("--clusters", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=5000)
    parser.add_argument("--q-low", type=float, default=0.33)
    parser.add_argument("--q-high", type=float, default=0.66)
    parser.add_argument("--silhouette-sample-size", type=int, default=20000)
    parser.add_argument("--log-every", type=int, default=200000)
    parser.add_argument("--max-rows", type=int, default=None)
    return parser


def main() -> int:
    args = build_parser().parse_args()

    project_root = Path(__file__).resolve().parents[1]
    scripts_dir = (project_root / "scripts").resolve()
    results_dir = resolve_path_arg(args.results_dir, project_root, project_root)

    combined_csv = resolve_path_arg(args.combined_input, project_root, results_dir)
    weekly_csv = resolve_path_arg(args.weekly_input, project_root, results_dir)

    cfg = Phase2Config(
        project_root=project_root,
        scripts_dir=scripts_dir,
        results_dir=results_dir,
        combined_csv=combined_csv,
        weekly_csv=weekly_csv,
        step3_results_csv=(results_dir / "step3_student_engagement_results.csv").resolve(),
        step3_weights_csv=(results_dir / "step3_activity_weights.csv").resolve(),
        step3_centers_csv=(results_dir / "step3_cluster_centers.csv").resolve(),
        step3_report_txt=(results_dir / "step3_analysis_report.txt").resolve(),
        step5_labeled_csv=(results_dir / "step5_standard_labels_kmeans.csv").resolve(),
        step5_cluster_map_csv=(results_dir / "step5_kmeans_cluster_label_map.csv").resolve(),
        step5_report_txt=(results_dir / "step5_kmeans_label_init_report.txt").resolve(),
        output_external_csv=(results_dir / "phase2_external_validation_metrics.csv").resolve(),
        output_internal_csv=(results_dir / "phase2_internal_validation_metrics.csv").resolve(),
        output_report_txt=(results_dir / "phase2_kmeans_validation_report.txt").resolve(),
        clusters=max(1, args.clusters),
        q_low=max(0.0, min(1.0, args.q_low)),
        q_high=max(0.0, min(1.0, args.q_high)),
        batch_size=max(100, args.batch_size),
        log_every=max(1, args.log_every),
        silhouette_sample_size=max(100, args.silhouette_sample_size),
        max_rows=args.max_rows,
    )

    if cfg.q_low >= cfg.q_high:
        log("FAILED: q-low must be smaller than q-high")
        return 1

    step_3 = cfg.scripts_dir / "step_3_engagement_report.py"
    step_5 = cfg.scripts_dir / "step_5_init_standard_labels_kmeans.py"

    try:
        started = time.time()
        log("Starting Phase 2: K-Means + Label-Based Validation")

        cmd3 = [
            sys.executable,
            str(step_3),
            "--input",
            str(cfg.combined_csv),
            "--weekly-input",
            str(cfg.weekly_csv),
            "--output-dir",
            str(cfg.results_dir),
            "--clusters",
            str(cfg.clusters),
            "--batch-size",
            str(cfg.batch_size),
            "--q-low",
            str(cfg.q_low),
            "--q-high",
            str(cfg.q_high),
            "--log-every",
            str(cfg.log_every),
        ]
        if cfg.max_rows is not None:
            cmd3.extend(["--max-rows", str(cfg.max_rows)])
        run_command(cmd3, cfg.project_root, "Phase 2.1 - K-Means clustering")

        cmd5 = [
            sys.executable,
            str(step_5),
            "--input",
            str(cfg.step3_results_csv),
            "--output-dir",
            str(cfg.results_dir),
            "--log-every",
            str(cfg.log_every),
        ]
        if cfg.max_rows is not None:
            cmd5.extend(["--max-rows", str(cfg.max_rows)])
        run_command(cmd5, cfg.project_root, "Phase 2.2 - Standard label initialization")

        log("Phase 2.3 - Computing label-based validation metrics")
        y_true, y_pred, cluster_counts = load_labeled_rows(cfg.step5_labeled_csv, cfg.max_rows)
        external_metrics = compute_external_metrics(y_true, y_pred)
        write_external_csv(cfg.output_external_csv, external_metrics)

        X_raw, clusters = load_features_and_clusters(cfg.step3_results_csv, cfg.max_rows)
        internal_metrics = compute_internal_metrics(X_raw, clusters, cfg.silhouette_sample_size)
        write_internal_csv(cfg.output_internal_csv, internal_metrics)

        write_report(
            path=cfg.output_report_txt,
            cfg=cfg,
            external_metrics=external_metrics,
            internal_metrics=internal_metrics,
            cluster_counts=cluster_counts,
            elapsed=time.time() - started,
        )

        log(f"Phase 2 completed in {time.time() - started:.2f}s")
        return 0
    except Exception as exc:
        log(f"FAILED: {exc}")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())