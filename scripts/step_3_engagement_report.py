#!/usr/bin/env python3
"""
Step 3: Engagement labeling and clustering, aligned with notebook logic.

Notebook-inspired engagement formula:
- Build user-week binary activity matrix x_{i,a,w} from Step 2 output.
- Activity weights: w_a = sum(x_{i,a,w}) / (S * N)
  where S is number of users, N is number of weeks.
- Student engagement score: E_i = sum_w sum_a (w_a * x_{i,a,w})

Inputs:
- results/combined_user_metrics.csv
- results/step2_user_week_activity.csv

Outputs:
- results/step3_student_engagement_results.csv
- results/step3_activity_weights.csv
- results/step3_cluster_centers.csv
- results/step3_analysis_report.txt
"""

from __future__ import annotations

import argparse
import csv
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
from sklearn.cluster import MiniBatchKMeans
from sklearn.preprocessing import StandardScaler


WEEKLY_ACTIVITY_COLUMNS = ["video", "problem", "reply", "comment"]
CLUSTER_FEATURES = [
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
class Step3Config:
    project_root: Path
    combined_csv: Path
    weekly_csv: Path
    output_dir: Path
    output_results_csv: Path
    output_weights_csv: Path
    output_centers_csv: Path
    output_report_txt: Path
    clusters: int = 3
    batch_size: int = 5000
    log_every: int = 100000
    random_state: int = 42
    q_low: float = 0.33
    q_high: float = 0.66
    max_rows: Optional[int] = None


@dataclass
class WeeklyStats:
    weekly_rows: int
    num_users_total: int
    num_users_active: int
    num_weeks: int
    sums: Dict[str, int]
    weights: Dict[str, float]


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


def safe_int(value) -> Optional[int]:
    if value is None:
        return None
    text = str(value).strip()
    if not text:
        return None
    try:
        return int(float(text))
    except ValueError:
        return None


def activity_flag(value) -> int:
    return 1 if safe_float(value) > 0 else 0


def forum_total_from_row(row: Dict[str, str]) -> float:
    raw = row.get("forum_total")
    if raw is None or str(raw).strip() == "":
        return safe_float(row.get("reply_count")) + safe_float(row.get("comment_count"))
    return safe_float(raw)


def watched_hours_from_row(row: Dict[str, str]) -> float:
    raw = row.get("watched_hours")
    if raw is None or str(raw).strip() == "":
        return safe_float(row.get("watched_seconds")) / 3600.0
    return safe_float(raw)


def row_to_cluster_features(row: Dict[str, str]) -> List[float]:
    return [
        safe_float(row.get("num_courses")),
        safe_float(row.get("problem_total")),
        safe_float(row.get("problem_accuracy")),
        safe_float(row.get("avg_attempts")),
        safe_float(row.get("video_count")),
        watched_hours_from_row(row),
        forum_total_from_row(row),
        safe_float(row.get("avg_speed")),
    ]


def iter_csv_rows(path: Path, max_rows: Optional[int] = None) -> Iterable[Dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for idx, row in enumerate(reader, start=1):
            if max_rows is not None and idx > max_rows:
                break
            yield row


def compute_weekly_stats(cfg: Step3Config) -> WeeklyStats:
    if not cfg.weekly_csv.exists():
        raise FileNotFoundError(f"Weekly activity file not found: {cfg.weekly_csv}")

    log("Step 3.1/7: Computing weekly activity weights from step2 output")
    unique_users_active = set()
    unique_weeks = set()
    sums = {k: 0 for k in WEEKLY_ACTIVITY_COLUMNS}

    weekly_rows = 0
    for row in iter_csv_rows(cfg.weekly_csv, cfg.max_rows):
        user_id = (row.get("user_id") or "").strip()
        week = safe_int(row.get("week"))
        if not user_id or week is None:
            continue

        weekly_rows += 1
        unique_users_active.add(user_id)
        unique_weeks.add(week)

        for col in WEEKLY_ACTIVITY_COLUMNS:
            sums[col] += activity_flag(row.get(col))

        if weekly_rows % cfg.log_every == 0:
            log(f"Weekly-stats progress: rows={weekly_rows:,}")

    total_users = 0
    for row in iter_csv_rows(cfg.combined_csv, cfg.max_rows):
        user_id = (row.get("user_id") or "").strip()
        if user_id:
            total_users += 1

    S = total_users
    N = len(unique_weeks)
    denom = S * N

    if denom <= 0:
        weights = {k: 0.0 for k in WEEKLY_ACTIVITY_COLUMNS}
    else:
        weights = {k: sums[k] / denom for k in WEEKLY_ACTIVITY_COLUMNS}

    log(
        f"Weekly stats done: rows={weekly_rows:,}, active_users={len(unique_users_active):,}, "
        f"total_users={S:,}, weeks={N:,}"
    )
    return WeeklyStats(
        weekly_rows=weekly_rows,
        num_users_total=S,
        num_users_active=len(unique_users_active),
        num_weeks=N,
        sums=sums,
        weights=weights,
    )


def compute_user_engagement_scores(cfg: Step3Config, weekly_stats: WeeklyStats) -> Dict[str, float]:
    log("Step 3.2/7: Computing user engagement scores E")

    user_scores: Dict[str, float] = {}
    scanned = 0
    for row in iter_csv_rows(cfg.weekly_csv, cfg.max_rows):
        user_id = (row.get("user_id") or "").strip()
        week = safe_int(row.get("week"))
        if not user_id or week is None:
            continue

        scanned += 1
        e_week = 0.0
        for col in WEEKLY_ACTIVITY_COLUMNS:
            e_week += weekly_stats.weights[col] * activity_flag(row.get(col))

        user_scores[user_id] = user_scores.get(user_id, 0.0) + e_week

        if scanned % cfg.log_every == 0:
            log(f"Score progress: weekly_rows={scanned:,}")

    return user_scores


def fit_scaler(cfg: Step3Config) -> Tuple[StandardScaler, int]:
    log("Step 3.3/7: Fitting StandardScaler on combined features")
    scaler = StandardScaler()

    batch: List[List[float]] = []
    total_rows = 0
    for row in iter_csv_rows(cfg.combined_csv, cfg.max_rows):
        total_rows += 1
        batch.append(row_to_cluster_features(row))

        if len(batch) >= cfg.batch_size:
            scaler.partial_fit(np.array(batch, dtype=np.float64))
            batch.clear()

        if total_rows % cfg.log_every == 0:
            log(f"Scaler progress: rows={total_rows:,}")

    if total_rows == 0:
        raise RuntimeError("Combined CSV is empty. Cannot run step 3.")

    if batch:
        scaler.partial_fit(np.array(batch, dtype=np.float64))

    return scaler, total_rows


def fit_kmeans(cfg: Step3Config, scaler: StandardScaler, total_rows: int):
    k = min(max(1, cfg.clusters), total_rows)
    if k == 1:
        log("Step 3.4/7: Only one cluster possible (k=1)")
        return None, 1

    log(f"Step 3.4/7: Fitting MiniBatchKMeans (k={k})")
    kmeans = MiniBatchKMeans(
        n_clusters=k,
        random_state=cfg.random_state,
        batch_size=cfg.batch_size,
    )

    batch: List[List[float]] = []
    scanned = 0
    for row in iter_csv_rows(cfg.combined_csv, cfg.max_rows):
        scanned += 1
        batch.append(row_to_cluster_features(row))

        if len(batch) >= cfg.batch_size:
            X = scaler.transform(np.array(batch, dtype=np.float64))
            kmeans.partial_fit(X)
            batch.clear()

        if scanned % cfg.log_every == 0:
            log(f"KMeans progress: rows={scanned:,}")

    if batch:
        X = scaler.transform(np.array(batch, dtype=np.float64))
        kmeans.partial_fit(X)

    return kmeans, k


def normalize_score(value: float, min_v: float, max_v: float) -> float:
    if max_v <= min_v:
        return 0.0
    score = (value - min_v) / (max_v - min_v)
    if score < 0:
        return 0.0
    if score > 1:
        return 1.0
    return score


def make_label(e_norm: float, low_th: float, high_th: float) -> str:
    if e_norm <= low_th:
        return "Low"
    if e_norm <= high_th:
        return "Medium"
    return "High"


def collect_score_distribution(
    cfg: Step3Config,
    user_scores: Dict[str, float],
) -> Tuple[float, float, float, float]:
    log("Step 3.5/7: Computing normalization and label thresholds")
    raw_scores: List[float] = []

    scanned = 0
    for row in iter_csv_rows(cfg.combined_csv, cfg.max_rows):
        scanned += 1
        user_id = (row.get("user_id") or "").strip()
        raw_scores.append(user_scores.get(user_id, 0.0))

        if scanned % cfg.log_every == 0:
            log(f"Threshold progress: rows={scanned:,}")

    if not raw_scores:
        raise RuntimeError("Combined CSV is empty. Cannot compute engagement thresholds.")

    arr = np.array(raw_scores, dtype=np.float64)
    min_e = float(arr.min())
    max_e = float(arr.max())

    if max_e <= min_e:
        low_th = 0.0
        high_th = 0.0
    else:
        e_norm_arr = (arr - min_e) / (max_e - min_e)
        low_th = float(np.quantile(e_norm_arr, cfg.q_low))
        high_th = float(np.quantile(e_norm_arr, cfg.q_high))

    return min_e, max_e, low_th, high_th


def write_outputs(
    cfg: Step3Config,
    weekly_stats: WeeklyStats,
    user_scores: Dict[str, float],
    min_e: float,
    max_e: float,
    low_th: float,
    high_th: float,
    scaler: StandardScaler,
    kmeans,
    k: int,
) -> Dict[str, Dict]:
    log("Step 3.6/7: Writing outputs")
    cfg.output_dir.mkdir(parents=True, exist_ok=True)

    with cfg.output_weights_csv.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["activity", "sum_x", "S", "N", "weight"])
        for activity in WEEKLY_ACTIVITY_COLUMNS:
            writer.writerow(
                [
                    activity,
                    weekly_stats.sums[activity],
                    weekly_stats.num_users_total,
                    weekly_stats.num_weeks,
                    round(weekly_stats.weights[activity], 6),
                ]
            )

    if kmeans is not None:
        centers = scaler.inverse_transform(kmeans.cluster_centers_)
        with cfg.output_centers_csv.open("w", encoding="utf-8", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["cluster"] + CLUSTER_FEATURES)
            for idx, center in enumerate(centers):
                writer.writerow([idx] + [round(float(v), 6) for v in center])
    else:
        with cfg.output_centers_csv.open("w", encoding="utf-8", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["cluster"] + CLUSTER_FEATURES)
            writer.writerow([0] + ["NA"] * len(CLUSTER_FEATURES))

    label_counts = {"Low": 0, "Medium": 0, "High": 0}
    cluster_counts = {idx: 0 for idx in range(k)}
    e_sum = 0.0
    e_norm_sum = 0.0

    headers = [
        "user_id",
        "school",
        "E",
        "E_norm",
        "EngagementLabel",
        "cluster",
    ] + CLUSTER_FEATURES

    with cfg.output_results_csv.open("w", encoding="utf-8", newline="") as out_f:
        writer = csv.writer(out_f)
        writer.writerow(headers)

        pending_meta: List[Tuple[str, str, float, float, str, List[float]]] = []
        pending_features: List[List[float]] = []

        def flush_pending() -> None:
            nonlocal e_sum, e_norm_sum
            if not pending_meta:
                return

            if kmeans is None:
                clusters = np.zeros(len(pending_meta), dtype=np.int64)
            else:
                X = scaler.transform(np.array(pending_features, dtype=np.float64))
                clusters = kmeans.predict(X)

            for idx, meta in enumerate(pending_meta):
                user_id, school, e_value, e_norm, label, feature_vec = meta
                cluster = int(clusters[idx])
                label_counts[label] += 1
                cluster_counts[cluster] += 1
                e_sum += e_value
                e_norm_sum += e_norm

                writer.writerow(
                    [
                        user_id,
                        school,
                        round(e_value, 6),
                        round(e_norm, 6),
                        label,
                        cluster,
                    ]
                    + [round(v, 6) for v in feature_vec]
                )

            pending_meta.clear()
            pending_features.clear()

        scanned = 0
        for row in iter_csv_rows(cfg.combined_csv, cfg.max_rows):
            scanned += 1
            user_id = (row.get("user_id") or "").strip()
            school = row.get("school", "")

            e_value = user_scores.get(user_id, 0.0)
            e_norm = normalize_score(e_value, min_e, max_e)
            label = make_label(e_norm, low_th, high_th)
            feature_vec = row_to_cluster_features(row)

            pending_meta.append((user_id, school, e_value, e_norm, label, feature_vec))
            pending_features.append(feature_vec)

            if len(pending_meta) >= cfg.batch_size:
                flush_pending()

            if scanned % cfg.log_every == 0:
                log(f"Output progress: rows={scanned:,}")

        flush_pending()

    total = sum(label_counts.values())
    avg_e = e_sum / total if total > 0 else 0.0
    avg_e_norm = e_norm_sum / total if total > 0 else 0.0

    return {
        "label_counts": label_counts,
        "cluster_counts": cluster_counts,
        "avg_e": avg_e,
        "avg_e_norm": avg_e_norm,
    }


def write_report(
    cfg: Step3Config,
    weekly_stats: WeeklyStats,
    thresholds: Tuple[float, float, float, float],
    output_stats: Dict[str, Dict],
    k: int,
    elapsed_seconds: float,
) -> None:
    log("Step 3.7/7: Writing analysis report")
    min_e, max_e, low_th, high_th = thresholds
    label_counts = output_stats["label_counts"]
    cluster_counts = output_stats["cluster_counts"]
    total_users = sum(label_counts.values())

    with cfg.output_report_txt.open("w", encoding="utf-8") as f:
        f.write("Step 3 - Engagement Analysis Report (Notebook-Aligned)\n")
        f.write("=" * 90 + "\n")
        f.write(f"Generated at           : {now_text()}\n")
        f.write(f"Combined CSV           : {cfg.combined_csv}\n")
        f.write(f"Weekly activity CSV    : {cfg.weekly_csv}\n")
        f.write(f"Output results CSV     : {cfg.output_results_csv}\n")
        f.write(f"Output weights CSV     : {cfg.output_weights_csv}\n")
        f.write(f"Output centers CSV     : {cfg.output_centers_csv}\n")
        f.write(f"Total users            : {total_users:,}\n")
        f.write(f"Clusters (k)           : {k}\n")
        f.write(f"Elapsed seconds        : {elapsed_seconds:.2f}\n")

        f.write("\nEngagement formula:\n")
        f.write("- w_a = sum(x_{i,a,w}) / (S * N)\n")
        f.write("- E_i = sum_w sum_a (w_a * x_{i,a,w})\n")
        f.write(
            f"- S={weekly_stats.num_users_total:,}, N={weekly_stats.num_weeks:,}, "
            f"weekly_rows={weekly_stats.weekly_rows:,}, active_users={weekly_stats.num_users_active:,}\n"
        )

        f.write("\nActivity weights:\n")
        for activity in WEEKLY_ACTIVITY_COLUMNS:
            f.write(
                f"- {activity:8} sum_x={weekly_stats.sums[activity]:>10,}, "
                f"weight={weekly_stats.weights[activity]:.6f}\n"
            )

        f.write("\nScore normalization and labels:\n")
        f.write(f"- E min={min_e:.6f}, E max={max_e:.6f}\n")
        f.write(f"- Low threshold (E_norm q{cfg.q_low * 100:.0f}): {low_th:.6f}\n")
        f.write(f"- High threshold (E_norm q{cfg.q_high * 100:.0f}): {high_th:.6f}\n")

        f.write("\nEngagement distribution:\n")
        for label in ["Low", "Medium", "High"]:
            count = label_counts.get(label, 0)
            pct = (count / total_users * 100.0) if total_users > 0 else 0.0
            f.write(f"- {label:6}: {count:>10,} users ({pct:6.2f}%)\n")

        f.write("\nCluster distribution:\n")
        for cluster_id in sorted(cluster_counts.keys()):
            count = cluster_counts[cluster_id]
            pct = (count / total_users * 100.0) if total_users > 0 else 0.0
            f.write(f"- Cluster {cluster_id}: {count:>10,} users ({pct:6.2f}%)\n")

        f.write("\nAverages:\n")
        f.write(f"- Mean E      : {output_stats['avg_e']:.6f}\n")
        f.write(f"- Mean E_norm : {output_stats['avg_e_norm']:.6f}\n")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Step 3 analytics aligned with notebook engagement formula.")
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("combined_user_metrics.csv"),
        help=(
            "Input combined CSV. Filename resolves under results folder; paths with "
            "folders resolve from project root."
        ),
    )
    parser.add_argument(
        "--weekly-input",
        type=Path,
        default=Path("step2_user_week_activity.csv"),
        help=(
            "Input user-week activity CSV from step 2. Filename resolves under results folder; "
            "paths with folders resolve from project root."
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results"),
        help="Output directory. Relative path resolves from project root.",
    )
    parser.add_argument(
        "--clusters",
        type=int,
        default=3,
        help="Number of clusters for MiniBatchKMeans (default: 3)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=5000,
        help="Batch size for scaler, kmeans, and output (default: 5000)",
    )
    parser.add_argument(
        "--log-every",
        type=int,
        default=100000,
        help="Progress log interval by scanned rows (default: 100000)",
    )
    parser.add_argument(
        "--max-rows",
        type=int,
        default=None,
        help="Optional dry-run cap on number of rows",
    )
    parser.add_argument(
        "--q-low",
        type=float,
        default=0.33,
        help="Lower percentile for engagement label split (default: 0.33)",
    )
    parser.add_argument(
        "--q-high",
        type=float,
        default=0.66,
        help="Upper percentile for engagement label split (default: 0.66)",
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()

    project_root = Path(__file__).resolve().parents[1]
    results_dir = (project_root / "results").resolve()

    combined_csv = resolve_path_arg(args.input, project_root, results_dir)
    weekly_csv = resolve_path_arg(args.weekly_input, project_root, results_dir)
    output_dir = resolve_path_arg(args.output_dir, project_root, project_root)

    cfg = Step3Config(
        project_root=project_root,
        combined_csv=combined_csv,
        weekly_csv=weekly_csv,
        output_dir=output_dir,
        output_results_csv=(output_dir / "step3_student_engagement_results.csv").resolve(),
        output_weights_csv=(output_dir / "step3_activity_weights.csv").resolve(),
        output_centers_csv=(output_dir / "step3_cluster_centers.csv").resolve(),
        output_report_txt=(output_dir / "step3_analysis_report.txt").resolve(),
        clusters=args.clusters,
        batch_size=args.batch_size,
        log_every=args.log_every,
        q_low=args.q_low,
        q_high=args.q_high,
        max_rows=args.max_rows,
    )

    started = time.time()
    try:
        log("Starting step 3 analysis pipeline")
        weekly_stats = compute_weekly_stats(cfg)
        user_scores = compute_user_engagement_scores(cfg, weekly_stats)
        min_e, max_e, low_th, high_th = collect_score_distribution(cfg, user_scores)
        scaler, total_rows = fit_scaler(cfg)
        kmeans, k = fit_kmeans(cfg, scaler, total_rows)
        output_stats = write_outputs(
            cfg=cfg,
            weekly_stats=weekly_stats,
            user_scores=user_scores,
            min_e=min_e,
            max_e=max_e,
            low_th=low_th,
            high_th=high_th,
            scaler=scaler,
            kmeans=kmeans,
            k=k,
        )
        write_report(
            cfg=cfg,
            weekly_stats=weekly_stats,
            thresholds=(min_e, max_e, low_th, high_th),
            output_stats=output_stats,
            k=k,
            elapsed_seconds=time.time() - started,
        )
        log(f"Done. Report: {cfg.output_report_txt}")
        return 0
    except Exception as exc:
        log(f"FAILED: {exc}")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())