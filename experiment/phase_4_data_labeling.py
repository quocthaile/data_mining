"""
Phase 4: Ghi nhãn dữ liệu
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import time
import csv
import json
import sqlite3
from datetime import datetime, timezone
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import *
from sklearn.cluster import MiniBatchKMeans
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, silhouette_score
from sklearn.preprocessing import StandardScaler
import heapq
import math
import numpy as np


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


def write_step3_report(
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

"""
Step 5: Initialize standard labels from K-Means clusters.

Purpose:
- Read Step 3 output that already contains K-Means cluster assignments.
- Rank clusters by mean engagement (E_norm) and map them to standard labels.
- Export row-level initialized labels and a cluster-label mapping summary.

Input:
- results/step3_student_engagement_results.csv

Outputs:
- results/step5_standard_labels_kmeans.csv
- results/step5_kmeans_cluster_label_map.csv
- results/step5_kmeans_label_init_report.txt
"""




@dataclass
class Step5Config:
    project_root: Path
    input_csv: Path
    output_dir: Path
    output_labeled_csv: Path
    output_cluster_map_csv: Path
    output_report_txt: Path
    log_every: int = 100000
    max_rows: Optional[int] = None


@dataclass
class ClusterStats:
    cluster: str
    count: int = 0
    e_sum: float = 0.0
    e_norm_sum: float = 0.0

    def add(self, e_value: float, e_norm: float) -> None:
        self.count += 1
        self.e_sum += e_value
        self.e_norm_sum += e_norm

    def mean_e(self) -> float:
        if self.count <= 0:
            return 0.0
        return self.e_sum / self.count

    def mean_e_norm(self) -> float:
        if self.count <= 0:
            return 0.0
        return self.e_norm_sum / self.count






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




def iter_csv_rows(path: Path, max_rows: Optional[int] = None) -> Iterable[Dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for idx, row in enumerate(reader, start=1):
            if max_rows is not None and idx > max_rows:
                break
            yield row


def build_label_names(k: int) -> List[str]:
    if k <= 0:
        return []
    if k == 1:
        return ["Medium"]
    if k == 2:
        return ["Low", "High"]
    if k == 3:
        return ["Low", "Medium", "High"]
    return [f"Level_{i + 1}" for i in range(k)]


class KMeansLabelInitializer:
    def __init__(self, cfg: Step5Config):
        self.cfg = cfg

    def run(self) -> None:
        if not self.cfg.input_csv.exists():
            raise FileNotFoundError(f"Input file not found: {self.cfg.input_csv}")

        started = time.time()
        self.cfg.output_dir.mkdir(parents=True, exist_ok=True)

        log("Step 5.1/4: Scanning Step 3 output and summarizing K-Means clusters")
        cluster_stats, label_counts_old, total_rows = self._scan_clusters()

        if total_rows <= 0:
            raise RuntimeError("Input Step 3 CSV is empty. Cannot initialize standard labels.")

        log("Step 5.2/4: Building cluster-to-standard-label mapping")
        cluster_label_map = self._build_cluster_label_map(cluster_stats)

        log("Step 5.3/4: Writing labeled dataset and mapping table")
        label_counts_new = self._write_labeled_output(cluster_stats, cluster_label_map)
        self._write_cluster_map(cluster_stats, cluster_label_map)

        log("Step 5.4/4: Writing summary report")
        self._write_report(
            cluster_stats=cluster_stats,
            cluster_label_map=cluster_label_map,
            label_counts_old=label_counts_old,
            label_counts_new=label_counts_new,
            total_rows=total_rows,
            elapsed=time.time() - started,
        )

        log(f"Done. Labeled data: {self.cfg.output_labeled_csv}")

    def _scan_clusters(self) -> Tuple[Dict[str, ClusterStats], Dict[str, int], int]:
        cluster_stats: Dict[str, ClusterStats] = {}
        label_counts_old: Dict[str, int] = {}
        total_rows = 0

        for row in iter_csv_rows(self.cfg.input_csv, self.cfg.max_rows):
            total_rows += 1
            cluster = (row.get("cluster") or "NA").strip() or "NA"
            e_value = safe_float(row.get("E"))
            e_norm = safe_float(row.get("E_norm"))
            old_label = (row.get("EngagementLabel") or "Unknown").strip() or "Unknown"

            stats = cluster_stats.get(cluster)
            if stats is None:
                stats = ClusterStats(cluster=cluster)
                cluster_stats[cluster] = stats
            stats.add(e_value, e_norm)

            label_counts_old[old_label] = label_counts_old.get(old_label, 0) + 1

            if total_rows % self.cfg.log_every == 0:
                log(f"Scan progress: rows={total_rows:,}")

        return cluster_stats, label_counts_old, total_rows

    def _build_cluster_label_map(self, cluster_stats: Dict[str, ClusterStats]) -> Dict[str, Tuple[int, str]]:
        ranked = sorted(
            cluster_stats.values(),
            key=lambda s: (s.mean_e_norm(), s.mean_e(), s.count),
        )
        label_names = build_label_names(len(ranked))

        cluster_label_map: Dict[str, Tuple[int, str]] = {}
        for rank, stats in enumerate(ranked, start=1):
            cluster_label_map[stats.cluster] = (rank, label_names[rank - 1])

        return cluster_label_map

    def _write_labeled_output(
        self,
        cluster_stats: Dict[str, ClusterStats],
        cluster_label_map: Dict[str, Tuple[int, str]],
    ) -> Dict[str, int]:
        label_counts_new: Dict[str, int] = {}

        with self.cfg.input_csv.open("r", encoding="utf-8", newline="") as in_f:
            reader = csv.DictReader(in_f)
            fieldnames = list(reader.fieldnames or [])

            extra_cols = [
                "kmeans_cluster_rank",
                "kmeans_cluster_mean_E",
                "kmeans_cluster_mean_E_norm",
                "StandardLabelKMeans",
            ]
            for col in extra_cols:
                if col not in fieldnames:
                    fieldnames.append(col)

            with self.cfg.output_labeled_csv.open("w", encoding="utf-8", newline="") as out_f:
                writer = csv.DictWriter(out_f, fieldnames=fieldnames)
                writer.writeheader()

                written = 0
                for idx, row in enumerate(reader, start=1):
                    if self.cfg.max_rows is not None and idx > self.cfg.max_rows:
                        break

                    cluster = (row.get("cluster") or "NA").strip() or "NA"
                    rank, label_name = cluster_label_map[cluster]
                    stats = cluster_stats[cluster]

                    row["kmeans_cluster_rank"] = rank
                    row["kmeans_cluster_mean_E"] = round(stats.mean_e(), 6)
                    row["kmeans_cluster_mean_E_norm"] = round(stats.mean_e_norm(), 6)
                    row["StandardLabelKMeans"] = label_name
                    writer.writerow(row)

                    written += 1
                    label_counts_new[label_name] = label_counts_new.get(label_name, 0) + 1

                    if written % self.cfg.log_every == 0:
                        log(f"Write progress: rows={written:,}")

        return label_counts_new

    def _write_cluster_map(
        self,
        cluster_stats: Dict[str, ClusterStats],
        cluster_label_map: Dict[str, Tuple[int, str]],
    ) -> None:
        def cluster_sort_key(value: str):
            try:
                return (0, int(value))
            except ValueError:
                return (1, value)

        with self.cfg.output_cluster_map_csv.open("w", encoding="utf-8", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["cluster", "count", "mean_E", "mean_E_norm", "rank", "StandardLabelKMeans"])

            for cluster in sorted(cluster_stats.keys(), key=cluster_sort_key):
                stats = cluster_stats[cluster]
                rank, label_name = cluster_label_map[cluster]
                writer.writerow(
                    [
                        cluster,
                        stats.count,
                        round(stats.mean_e(), 6),
                        round(stats.mean_e_norm(), 6),
                        rank,
                        label_name,
                    ]
                )

    def _write_report(
        self,
        cluster_stats: Dict[str, ClusterStats],
        cluster_label_map: Dict[str, Tuple[int, str]],
        label_counts_old: Dict[str, int],
        label_counts_new: Dict[str, int],
        total_rows: int,
        elapsed: float,
    ) -> None:
        ranked_clusters = sorted(
            cluster_stats.values(),
            key=lambda s: cluster_label_map[s.cluster][0],
        )

        with self.cfg.output_report_txt.open("w", encoding="utf-8") as f:
            f.write("Step 5 - K-Means Standard Label Initialization\n")
            f.write("=" * 90 + "\n")
            f.write(f"Generated at           : {now_text()}\n")
            f.write(f"Input Step 3 CSV       : {self.cfg.input_csv}\n")
            f.write(f"Total rows             : {total_rows:,}\n")
            f.write(f"Elapsed seconds        : {elapsed:.2f}\n")
            f.write(f"Output labeled CSV     : {self.cfg.output_labeled_csv}\n")
            f.write(f"Output cluster map CSV : {self.cfg.output_cluster_map_csv}\n")

            f.write("\nCluster ranking and standard labels:\n")
            for stats in ranked_clusters:
                rank, label_name = cluster_label_map[stats.cluster]
                pct = (stats.count / total_rows * 100.0) if total_rows > 0 else 0.0
                f.write(
                    f"- Cluster {stats.cluster:4} -> rank={rank}, label={label_name:8}, "
                    f"count={stats.count:>10,} ({pct:6.2f}%), "
                    f"mean_E={stats.mean_e():.6f}, mean_E_norm={stats.mean_e_norm():.6f}\n"
                )

            f.write("\nOriginal EngagementLabel distribution:\n")
            for label, count in sorted(label_counts_old.items(), key=lambda x: x[0]):
                pct = (count / total_rows * 100.0) if total_rows > 0 else 0.0
                f.write(f"- {label:12} {count:>10,} ({pct:6.2f}%)\n")

            f.write("\nInitialized StandardLabelKMeans distribution:\n")
            for label, count in sorted(label_counts_new.items(), key=lambda x: x[0]):
                pct = (count / total_rows * 100.0) if total_rows > 0 else 0.0
                f.write(f"- {label:12} {count:>10,} ({pct:6.2f}%)\n")

"""
Step 4: Detailed reporting for the engagement pipeline.

Input:
- results/step3_student_engagement_results.csv

Outputs:
- results/step4_global_stats.csv
- results/step4_label_summary.csv
- results/step4_cluster_summary.csv
- results/step4_label_cluster_matrix.csv
- results/step4_school_summary.csv
- results/step4_top_users.csv
- results/step4_analysis_report.txt
"""





DEFAULT_NUMERIC_FEATURES = [
    "E",
    "E_norm",
    "num_courses",
    "problem_total",
    "problem_accuracy",
    "avg_attempts",
    "video_count",
    "watched_hours",
    "forum_total",
    "avg_speed",
]
DEFAULT_LABEL_ORDER = ["Low", "Medium", "High"]


@dataclass
class Step4Config:
    project_root: Path
    input_csv: Path
    output_dir: Path
    output_global_stats_csv: Path
    output_label_summary_csv: Path
    output_cluster_summary_csv: Path
    output_label_cluster_csv: Path
    output_school_summary_csv: Path
    output_top_users_csv: Path
    output_report_txt: Path
    top_users: int = 100
    min_school_size: int = 20
    top_schools: int = 30
    log_every: int = 100000
    max_rows: Optional[int] = None


@dataclass
class GroupStats:
    count: int = 0
    e_sum: float = 0.0
    e_norm_sum: float = 0.0

    def add(self, e_value: float, e_norm: float) -> None:
        self.count += 1
        self.e_sum += e_value
        self.e_norm_sum += e_norm

    def mean_e(self) -> float:
        if self.count <= 0:
            return 0.0
        return self.e_sum / self.count

    def mean_e_norm(self) -> float:
        if self.count <= 0:
            return 0.0
        return self.e_norm_sum / self.count


@dataclass
class SchoolStats(GroupStats):
    high_count: int = 0
    medium_count: int = 0
    low_count: int = 0

    def add_label(self, label: str) -> None:
        if label == "High":
            self.high_count += 1
        elif label == "Medium":
            self.medium_count += 1
        elif label == "Low":
            self.low_count += 1


@dataclass
class NumericAccumulator:
    name: str
    count: int = 0
    mean: float = 0.0
    m2: float = 0.0
    min_value: float = math.inf
    max_value: float = -math.inf
    values: List[float] = field(default_factory=list)

    def add(self, value: float) -> None:
        self.count += 1
        delta = value - self.mean
        self.mean += delta / self.count
        delta2 = value - self.mean
        self.m2 += delta * delta2

        if value < self.min_value:
            self.min_value = value
        if value > self.max_value:
            self.max_value = value

        self.values.append(value)

    def std(self) -> float:
        if self.count <= 1:
            return 0.0
        return math.sqrt(self.m2 / (self.count - 1))

    def quantiles(self) -> Tuple[float, float, float]:
        if not self.values:
            return 0.0, 0.0, 0.0
        arr = np.array(self.values, dtype=np.float64)
        q1 = float(np.quantile(arr, 0.25))
        q2 = float(np.quantile(arr, 0.50))
        q3 = float(np.quantile(arr, 0.75))
        return q1, q2, q3


@dataclass
class Step4State:
    total_rows: int = 0
    label_stats: Dict[str, GroupStats] = field(default_factory=dict)
    cluster_stats: Dict[str, GroupStats] = field(default_factory=dict)
    school_stats: Dict[str, SchoolStats] = field(default_factory=dict)
    label_cluster_counts: Dict[Tuple[str, str], int] = field(default_factory=dict)
    numeric_stats: Dict[str, NumericAccumulator] = field(default_factory=dict)
    top_users_heap: List[Tuple[float, float, str, str, str, str, float, float, float, float]] = field(
        default_factory=list
    )








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


def normalized_school_name(value: Optional[str]) -> str:
    if value is None:
        return "Unknown"
    text = str(value).strip()
    return text if text else "Unknown"


def iter_csv_rows(path: Path, max_rows: Optional[int] = None) -> Iterable[Dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for idx, row in enumerate(reader, start=1):
            if max_rows is not None and idx > max_rows:
                break
            yield row


def make_numeric_stats() -> Dict[str, NumericAccumulator]:
    return {name: NumericAccumulator(name=name) for name in DEFAULT_NUMERIC_FEATURES}


class Step4Reporter:
    def __init__(self, cfg: Step4Config):
        self.cfg = cfg

    def run(self) -> None:
        if not self.cfg.input_csv.exists():
            raise FileNotFoundError(f"Input file not found: {self.cfg.input_csv}")

        started = time.time()
        self.cfg.output_dir.mkdir(parents=True, exist_ok=True)

        log("Step 4.1/3: Scanning Step 3 results")
        state = self._scan_input()

        if state.total_rows <= 0:
            raise RuntimeError("Input Step 3 CSV is empty. Cannot build detailed report.")

        log("Step 4.2/3: Writing detailed CSV outputs")
        self._write_global_stats(state)
        self._write_label_summary(state)
        self._write_cluster_summary(state)
        self._write_label_cluster_matrix(state)
        self._write_school_summary(state)
        self._write_top_users(state)

        log("Step 4.3/3: Writing text report")
        self._write_report(state, elapsed=time.time() - started)
        log(f"Done. Report: {self.cfg.output_report_txt}")

    def _scan_input(self) -> Step4State:
        state = Step4State(numeric_stats=make_numeric_stats())

        for row in iter_csv_rows(self.cfg.input_csv, self.cfg.max_rows):
            user_id = (row.get("user_id") or "").strip()
            school = normalized_school_name(row.get("school"))
            label = (row.get("EngagementLabel") or "Unknown").strip() or "Unknown"
            cluster = (row.get("cluster") or "NA").strip() or "NA"

            e_value = safe_float(row.get("E"))
            e_norm = safe_float(row.get("E_norm"))
            num_courses = safe_float(row.get("num_courses"))
            video_count = safe_float(row.get("video_count"))
            forum_total = safe_float(row.get("forum_total"))
            avg_speed = safe_float(row.get("avg_speed"))

            state.total_rows += 1

            label_stat = state.label_stats.get(label)
            if label_stat is None:
                label_stat = GroupStats()
                state.label_stats[label] = label_stat
            label_stat.add(e_value, e_norm)

            cluster_stat = state.cluster_stats.get(cluster)
            if cluster_stat is None:
                cluster_stat = GroupStats()
                state.cluster_stats[cluster] = cluster_stat
            cluster_stat.add(e_value, e_norm)

            school_stat = state.school_stats.get(school)
            if school_stat is None:
                school_stat = SchoolStats()
                state.school_stats[school] = school_stat
            school_stat.add(e_value, e_norm)
            school_stat.add_label(label)

            key = (label, cluster)
            state.label_cluster_counts[key] = state.label_cluster_counts.get(key, 0) + 1

            for feature, acc in state.numeric_stats.items():
                acc.add(safe_float(row.get(feature)))

            if self.cfg.top_users > 0:
                top_entry = (
                    e_norm,
                    e_value,
                    user_id,
                    school,
                    label,
                    cluster,
                    num_courses,
                    video_count,
                    forum_total,
                    avg_speed,
                )
                if len(state.top_users_heap) < self.cfg.top_users:
                    heapq.heappush(state.top_users_heap, top_entry)
                elif top_entry > state.top_users_heap[0]:
                    heapq.heapreplace(state.top_users_heap, top_entry)

            if state.total_rows % self.cfg.log_every == 0:
                log(f"Scan progress: rows={state.total_rows:,}")

        return state

    def _write_global_stats(self, state: Step4State) -> None:
        with self.cfg.output_global_stats_csv.open("w", encoding="utf-8", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["feature", "count", "mean", "std", "min", "q1", "median", "q3", "max"])

            for feature in DEFAULT_NUMERIC_FEATURES:
                acc = state.numeric_stats[feature]
                q1, q2, q3 = acc.quantiles()
                min_value = 0.0 if acc.count == 0 else acc.min_value
                max_value = 0.0 if acc.count == 0 else acc.max_value

                writer.writerow(
                    [
                        feature,
                        acc.count,
                        round(acc.mean, 6),
                        round(acc.std(), 6),
                        round(min_value, 6),
                        round(q1, 6),
                        round(q2, 6),
                        round(q3, 6),
                        round(max_value, 6),
                    ]
                )

    def _write_label_summary(self, state: Step4State) -> None:
        with self.cfg.output_label_summary_csv.open("w", encoding="utf-8", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["label", "count", "pct", "mean_E", "mean_E_norm"])

            labels = [x for x in DEFAULT_LABEL_ORDER if x in state.label_stats]
            labels += sorted(k for k in state.label_stats.keys() if k not in DEFAULT_LABEL_ORDER)

            for label in labels:
                stat = state.label_stats[label]
                pct = (stat.count / state.total_rows * 100.0) if state.total_rows > 0 else 0.0
                writer.writerow(
                    [
                        label,
                        stat.count,
                        round(pct, 4),
                        round(stat.mean_e(), 6),
                        round(stat.mean_e_norm(), 6),
                    ]
                )

    def _write_cluster_summary(self, state: Step4State) -> None:
        def cluster_sort_key(value: str):
            try:
                return (0, int(value))
            except ValueError:
                return (1, value)

        with self.cfg.output_cluster_summary_csv.open("w", encoding="utf-8", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["cluster", "count", "pct", "mean_E", "mean_E_norm"])

            for cluster in sorted(state.cluster_stats.keys(), key=cluster_sort_key):
                stat = state.cluster_stats[cluster]
                pct = (stat.count / state.total_rows * 100.0) if state.total_rows > 0 else 0.0
                writer.writerow(
                    [
                        cluster,
                        stat.count,
                        round(pct, 4),
                        round(stat.mean_e(), 6),
                        round(stat.mean_e_norm(), 6),
                    ]
                )

    def _write_label_cluster_matrix(self, state: Step4State) -> None:
        def cluster_sort_key(value: str):
            try:
                return (0, int(value))
            except ValueError:
                return (1, value)

        labels = [x for x in DEFAULT_LABEL_ORDER if x in state.label_stats]
        labels += sorted(k for k in state.label_stats.keys() if k not in DEFAULT_LABEL_ORDER)
        clusters = sorted(state.cluster_stats.keys(), key=cluster_sort_key)

        with self.cfg.output_label_cluster_csv.open("w", encoding="utf-8", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["label", "cluster", "count", "row_pct"])

            for label in labels:
                row_total = state.label_stats[label].count
                for cluster in clusters:
                    count = state.label_cluster_counts.get((label, cluster), 0)
                    row_pct = (count / row_total * 100.0) if row_total > 0 else 0.0
                    writer.writerow([label, cluster, count, round(row_pct, 4)])

    def _write_school_summary(self, state: Step4State) -> None:
        school_rows = []
        for school, stat in state.school_stats.items():
            if stat.count < self.cfg.min_school_size:
                continue
            pct = (stat.count / state.total_rows * 100.0) if state.total_rows > 0 else 0.0
            high_rate = (stat.high_count / stat.count * 100.0) if stat.count > 0 else 0.0
            school_rows.append(
                (
                    school,
                    stat.count,
                    pct,
                    stat.mean_e(),
                    stat.mean_e_norm(),
                    stat.high_count,
                    high_rate,
                )
            )

        school_rows.sort(key=lambda x: (x[4], x[1]), reverse=True)
        if self.cfg.top_schools > 0:
            school_rows = school_rows[: self.cfg.top_schools]

        with self.cfg.output_school_summary_csv.open("w", encoding="utf-8", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["school", "count", "pct", "mean_E", "mean_E_norm", "high_count", "high_rate"])
            for row in school_rows:
                writer.writerow(
                    [
                        row[0],
                        row[1],
                        round(row[2], 4),
                        round(row[3], 6),
                        round(row[4], 6),
                        row[5],
                        round(row[6], 4),
                    ]
                )

    def _write_top_users(self, state: Step4State) -> None:
        ordered = sorted(state.top_users_heap, reverse=True)

        with self.cfg.output_top_users_csv.open("w", encoding="utf-8", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(
                [
                    "rank",
                    "user_id",
                    "school",
                    "EngagementLabel",
                    "cluster",
                    "E",
                    "E_norm",
                    "num_courses",
                    "video_count",
                    "forum_total",
                    "avg_speed",
                ]
            )

            for rank, item in enumerate(ordered, start=1):
                (
                    e_norm,
                    e_value,
                    user_id,
                    school,
                    label,
                    cluster,
                    num_courses,
                    video_count,
                    forum_total,
                    avg_speed,
                ) = item

                writer.writerow(
                    [
                        rank,
                        user_id,
                        school,
                        label,
                        cluster,
                        round(e_value, 6),
                        round(e_norm, 6),
                        round(num_courses, 6),
                        round(video_count, 6),
                        round(forum_total, 6),
                        round(avg_speed, 6),
                    ]
                )

    def _write_report(self, state: Step4State, elapsed: float) -> None:
        labels_sorted = sorted(
            state.label_stats.items(),
            key=lambda x: x[1].count,
            reverse=True,
        )
        clusters_sorted = sorted(
            state.cluster_stats.items(),
            key=lambda x: x[1].count,
            reverse=True,
        )

        school_candidates = [
            (name, stat)
            for name, stat in state.school_stats.items()
            if stat.count >= self.cfg.min_school_size
        ]
        school_candidates.sort(key=lambda x: (x[1].mean_e_norm(), x[1].count), reverse=True)

        with self.cfg.output_report_txt.open("w", encoding="utf-8") as f:
            f.write("Step 4 - Detailed Engagement Report\n")
            f.write("=" * 90 + "\n")
            f.write(f"Generated at           : {now_text()}\n")
            f.write(f"Input Step 3 CSV       : {self.cfg.input_csv}\n")
            f.write(f"Total users            : {state.total_rows:,}\n")
            f.write(f"Elapsed seconds        : {elapsed:.2f}\n")
            f.write("\nGenerated files:\n")
            f.write(f"- {self.cfg.output_global_stats_csv}\n")
            f.write(f"- {self.cfg.output_label_summary_csv}\n")
            f.write(f"- {self.cfg.output_cluster_summary_csv}\n")
            f.write(f"- {self.cfg.output_label_cluster_csv}\n")
            f.write(f"- {self.cfg.output_school_summary_csv}\n")
            f.write(f"- {self.cfg.output_top_users_csv}\n")

            f.write("\nLabel distribution:\n")
            for label, stat in labels_sorted:
                pct = (stat.count / state.total_rows * 100.0) if state.total_rows > 0 else 0.0
                f.write(
                    f"- {label:8} count={stat.count:>10,} ({pct:6.2f}%), "
                    f"mean_E={stat.mean_e():.6f}, mean_E_norm={stat.mean_e_norm():.6f}\n"
                )

            f.write("\nCluster distribution:\n")
            for cluster, stat in clusters_sorted:
                pct = (stat.count / state.total_rows * 100.0) if state.total_rows > 0 else 0.0
                f.write(
                    f"- Cluster {cluster:4} count={stat.count:>10,} ({pct:6.2f}%), "
                    f"mean_E={stat.mean_e():.6f}, mean_E_norm={stat.mean_e_norm():.6f}\n"
                )

            f.write("\nTop schools by mean E_norm:\n")
            if not school_candidates:
                f.write("- No school reached min_school_size threshold.\n")
            else:
                for school, stat in school_candidates[: min(10, len(school_candidates))]:
                    high_rate = (stat.high_count / stat.count * 100.0) if stat.count > 0 else 0.0
                    f.write(
                        f"- {school}: count={stat.count:,}, mean_E_norm={stat.mean_e_norm():.6f}, "
                        f"high_rate={high_rate:.2f}%\n"
                    )


def run_engagement_report(
    input_csv: Path,
    weekly_csv: Path,
    output_dir: Path,
    clusters: int,
    batch_size: int,
    q_low: float,
    q_high: float,
    log_every: int,
    max_rows: Optional[int],
) -> None:
    started = time.time()
    cfg = Step3Config(
        project_root=output_dir.parent,
        combined_csv=input_csv,
        weekly_csv=weekly_csv,
        output_dir=output_dir,
        output_results_csv=(output_dir / "step3_student_engagement_results.csv").resolve(),
        output_weights_csv=(output_dir / "step3_activity_weights.csv").resolve(),
        output_centers_csv=(output_dir / "step3_cluster_centers.csv").resolve(),
        output_report_txt=(output_dir / "step3_analysis_report.txt").resolve(),
        clusters=max(1, clusters),
        batch_size=max(1, batch_size),
        log_every=max(1, log_every),
        q_low=max(0.0, min(1.0, q_low)),
        q_high=max(0.0, min(1.0, q_high)),
        max_rows=max_rows,
    )

    weekly_stats = compute_weekly_stats(cfg)
    user_scores = compute_user_engagement_scores(cfg, weekly_stats)
    scaler, total_rows = fit_scaler(cfg)
    kmeans, k = fit_kmeans(cfg, scaler, total_rows)
    min_e, max_e, low_th, high_th = collect_score_distribution(cfg, user_scores)
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
    write_step3_report(
        cfg=cfg,
        weekly_stats=weekly_stats,
        thresholds=(min_e, max_e, low_th, high_th),
        output_stats=output_stats,
        k=k,
        elapsed_seconds=time.time() - started,
    )


def run_init_standard_labels(
    input_csv: Path,
    output_dir: Path,
    log_every: int,
    max_rows: Optional[int],
) -> None:
    cfg = Step5Config(
        project_root=output_dir.parent,
        input_csv=input_csv,
        output_dir=output_dir,
        output_labeled_csv=(output_dir / "step5_standard_labels_kmeans.csv").resolve(),
        output_cluster_map_csv=(output_dir / "step5_kmeans_cluster_label_map.csv").resolve(),
        output_report_txt=(output_dir / "step5_kmeans_label_init_report.txt").resolve(),
        log_every=max(1, log_every),
        max_rows=max_rows,
    )
    KMeansLabelInitializer(cfg).run()


def run_detailed_report(
    input_csv: Path,
    output_dir: Path,
    top_users: int,
    min_school_size: int,
    top_schools: int,
    log_every: int,
    max_rows: Optional[int],
) -> None:
    cfg = Step4Config(
        project_root=output_dir.parent,
        input_csv=input_csv,
        output_dir=output_dir,
        output_global_stats_csv=(output_dir / "step4_global_stats.csv").resolve(),
        output_label_summary_csv=(output_dir / "step4_label_summary.csv").resolve(),
        output_cluster_summary_csv=(output_dir / "step4_cluster_summary.csv").resolve(),
        output_label_cluster_csv=(output_dir / "step4_label_cluster_matrix.csv").resolve(),
        output_school_summary_csv=(output_dir / "step4_school_summary.csv").resolve(),
        output_top_users_csv=(output_dir / "step4_top_users.csv").resolve(),
        output_report_txt=(output_dir / "step4_analysis_report.txt").resolve(),
        top_users=max(1, top_users),
        min_school_size=max(1, min_school_size),
        top_schools=max(1, top_schools),
        log_every=max(1, log_every),
        max_rows=max_rows,
    )
    Step4Reporter(cfg).run()

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
    parser = argparse.ArgumentParser(description="Phase 4: Ghi nhãn dữ liệu (Data Labeling).")
    parser.add_argument("--results-dir", type=Path, default=Path("results"))
    parser.add_argument("--combined-input", type=Path, default=Path("combined_user_metrics.csv"))
    parser.add_argument("--weekly-input", type=Path, default=Path("step2_user_week_activity.csv"))
    parser.add_argument("--clusters", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=5000)
    parser.add_argument("--q-low", type=float, default=0.33)
    parser.add_argument("--q-high", type=float, default=0.66)
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
    results_dir = resolve_path_arg(args.results_dir, project_root, project_root)
    combined_csv = resolve_path_arg(args.combined_input, project_root, results_dir)
    weekly_csv = resolve_path_arg(args.weekly_input, project_root, results_dir)

    cfg = Phase2Config(
        project_root=project_root,
        scripts_dir=project_root / "experiment",
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
        output_external_csv=(results_dir / "phase4_external_validation_metrics.csv").resolve(),
        output_internal_csv=(results_dir / "phase4_internal_validation_metrics.csv").resolve(),
        output_report_txt=(results_dir / "phase4_labeling_report.txt").resolve(),
        clusters=max(1, args.clusters),
        q_low=max(0.0, min(1.0, args.q_low)),
        q_high=max(0.0, min(1.0, args.q_high)),
        batch_size=max(1, args.batch_size),
        log_every=max(1, args.log_every),
        silhouette_sample_size=max(100, args.silhouette_sample_size),
        max_rows=args.max_rows,
    )

    try:
        started = time.time()
        log("Starting Phase 4: Ghi nhãn dữ liệu")

        log("Phase 4.1 - Engagement Report (Ghi nhãn không giám sát - K-Means)")
        run_engagement_report(
            input_csv=combined_csv,
            weekly_csv=weekly_csv,
            output_dir=results_dir,
            clusters=args.clusters,
            batch_size=args.batch_size,
            q_low=args.q_low,
            q_high=args.q_high,
            log_every=max(1, args.log_every),
            max_rows=args.max_rows
        )

        log("Phase 4.2 - Standard label initialization (Ghi nhãn có giám sát map)")
        run_init_standard_labels(
            input_csv=cfg.step3_results_csv,
            output_dir=cfg.results_dir,
            log_every=cfg.log_every,
            max_rows=cfg.max_rows
        )
        
        log("Phase 4.3 - Detailed Report (Khai phá tri thức sau ghi nhãn)")
        run_detailed_report(
            input_csv=cfg.step3_results_csv,
            output_dir=cfg.results_dir,
            top_users=args.top_users,
            min_school_size=args.min_school_size,
            top_schools=args.top_schools,
            log_every=max(1, args.log_every),
            max_rows=args.max_rows
        )

        log("Phase 4.4 - Computing label-based validation metrics")
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

        log(f"Phase 4 completed in {time.time() - started:.2f}s")
        return 0
    except Exception as exc:
        log(f"FAILED: {exc}")
        import traceback
        traceback.print_exc()
        return 1

if __name__ == "__main__":
    raise SystemExit(main())

