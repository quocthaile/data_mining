#!/usr/bin/env python3
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

from __future__ import annotations

import argparse
import csv
import heapq
import math
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np


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


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Step 4 detailed reporting for engagement outputs."
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("step3_student_engagement_results.csv"),
        help=(
            "Input Step 3 result CSV. Filename resolves under results folder; "
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
        "--top-users",
        type=int,
        default=100,
        help="Number of top users by E_norm to export (default: 100)",
    )
    parser.add_argument(
        "--min-school-size",
        type=int,
        default=20,
        help="Minimum users per school for school summary (default: 20)",
    )
    parser.add_argument(
        "--top-schools",
        type=int,
        default=30,
        help="Maximum schools to export in school summary (default: 30)",
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
    return parser


def main() -> int:
    args = build_parser().parse_args()

    project_root = Path(__file__).resolve().parents[1]
    results_dir = (project_root / "results").resolve()

    input_csv = resolve_path_arg(args.input, project_root, results_dir)
    output_dir = resolve_path_arg(args.output_dir, project_root, project_root)

    cfg = Step4Config(
        project_root=project_root,
        input_csv=input_csv,
        output_dir=output_dir,
        output_global_stats_csv=(output_dir / "step4_global_stats.csv").resolve(),
        output_label_summary_csv=(output_dir / "step4_label_summary.csv").resolve(),
        output_cluster_summary_csv=(output_dir / "step4_cluster_summary.csv").resolve(),
        output_label_cluster_csv=(output_dir / "step4_label_cluster_matrix.csv").resolve(),
        output_school_summary_csv=(output_dir / "step4_school_summary.csv").resolve(),
        output_top_users_csv=(output_dir / "step4_top_users.csv").resolve(),
        output_report_txt=(output_dir / "step4_analysis_report.txt").resolve(),
        top_users=max(0, args.top_users),
        min_school_size=max(1, args.min_school_size),
        top_schools=max(1, args.top_schools),
        log_every=max(1, args.log_every),
        max_rows=args.max_rows,
    )

    try:
        log("Starting step 4 detailed reporting")
        reporter = Step4Reporter(cfg)
        reporter.run()
        return 0
    except Exception as exc:
        log(f"FAILED: {exc}")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())