#!/usr/bin/env python3
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

from __future__ import annotations

import argparse
import csv
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple


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


def now_text() -> str:
    return time.strftime("%Y-%m-%d %H:%M:%S")


def log(message: str) -> None:
    print(f"[{now_text()}] {message}")


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


def resolve_path_arg(path_value: Path, project_root: Path, default_base: Path) -> Path:
    if path_value.is_absolute():
        return path_value.resolve()
    if path_value.parent == Path("."):
        return (default_base / path_value).resolve()
    return (project_root / path_value).resolve()


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


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Initialize standard labels from K-Means clusters in Step 3 output."
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("step3_student_engagement_results.csv"),
        help=(
            "Input Step 3 CSV. Filename resolves under results folder; "
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

    cfg = Step5Config(
        project_root=project_root,
        input_csv=input_csv,
        output_dir=output_dir,
        output_labeled_csv=(output_dir / "step5_standard_labels_kmeans.csv").resolve(),
        output_cluster_map_csv=(output_dir / "step5_kmeans_cluster_label_map.csv").resolve(),
        output_report_txt=(output_dir / "step5_kmeans_label_init_report.txt").resolve(),
        log_every=max(1, args.log_every),
        max_rows=args.max_rows,
    )

    try:
        log("Starting Step 5 K-Means label initialization")
        runner = KMeansLabelInitializer(cfg)
        runner.run()
        return 0
    except Exception as exc:
        log(f"FAILED: {exc}")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())