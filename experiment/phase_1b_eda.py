#!/usr/bin/env python3
"""
Phase 1B: EDA and Data Cleaning for Combined Metrics.

Aligned with data preparation workflow:
1. Exploratory Data Analysis (EDA)
   - Descriptive statistics
   - Data visualization
   - Detect outliers and anomalies
   - Statistical analysis

2. Data Cleaning
   - Handle missing values (missing data)
   - Handle duplicates/inconsistencies
   - Remove/fix erroneous data
   - Standardize format

3. Data Transformation
   - Normalize engagement_events
   - Create derived features
   - Feature scaling if needed

Outputs:
- results/phase1b_eda_report.txt
- results/combined_user_metrics_clean.csv
- results/engagement_events_normalized.csv (optional)
"""

from __future__ import annotations

import argparse
import csv
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np


@dataclass
class Phase1bConfig:
    combined_csv: Path
    output_dir: Path
    output_clean_csv: Path
    output_report_txt: Path
    output_normalization_csv: Path
    missing_threshold: float = 0.3
    outlier_iqr_multiplier: float = 1.5
    log_every: int = 100000
    max_rows: Optional[int] = None


def now_text() -> str:
    return time.strftime("%Y-%m-%d %H:%M:%S")


def log(message: str) -> None:
    print(f"[{now_text()}] {message}")


def safe_float(value) -> Optional[float]:
    if value is None or str(value).strip() == "":
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def safe_int(value) -> Optional[int]:
    if value is None or str(value).strip() == "":
        return None
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return None


def load_combined_csv(path: Path, max_rows: Optional[int]) -> Tuple[List[Dict[str, str]], List[str]]:
    rows: List[Dict[str, str]] = []
    with path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        columns = list(reader.fieldnames or [])
        for idx, row in enumerate(reader, start=1):
            if max_rows is not None and idx > max_rows:
                break
            rows.append(row)
            if idx % 100000 == 0:
                log(f"Load progress: {idx:,} rows")

    return rows, columns


def compute_descriptive_stats(rows: List[Dict[str, str]]) -> Dict[str, Dict[str, float]]:
    """Compute descriptive statistics for numeric columns."""
    numeric_cols = [
        "num_courses",
        "problem_total",
        "problem_accuracy",
        "avg_attempts",
        "avg_score",
        "video_sessions",
        "video_count",
        "segment_count",
        "watched_seconds",
        "watched_hours",
        "avg_speed",
        "reply_count",
        "comment_count",
        "forum_total",
        "engagement_events",
    ]

    stats: Dict[str, Dict[str, float]] = {}

    for col in numeric_cols:
        values = []
        missing_count = 0

        for row in rows:
            val = safe_float(row.get(col))
            if val is None:
                missing_count += 1
            else:
                values.append(val)

        if not values:
            stats[col] = {
                "count": 0,
                "missing": missing_count,
                "missing_pct": 100.0,
                "mean": 0.0,
                "std": 0.0,
                "min": 0.0,
                "q25": 0.0,
                "median": 0.0,
                "q75": 0.0,
                "max": 0.0,
            }
            continue

        arr = np.array(values, dtype=np.float64)
        missing_pct = (missing_count / len(rows)) * 100.0

        stats[col] = {
            "count": len(values),
            "missing": missing_count,
            "missing_pct": missing_pct,
            "mean": float(np.mean(arr)),
            "std": float(np.std(arr)),
            "min": float(np.min(arr)),
            "q25": float(np.quantile(arr, 0.25)),
            "median": float(np.quantile(arr, 0.50)),
            "q75": float(np.quantile(arr, 0.75)),
            "max": float(np.max(arr)),
        }

    return stats


def detect_outliers_iqr(
    rows: List[Dict[str, str]], col: str, multiplier: float = 1.5
) -> Tuple[int, float, float]:
    """Detect outliers using IQR method."""
    values = []
    for row in rows:
        val = safe_float(row.get(col))
        if val is not None:
            values.append(val)

    if not values:
        return 0, 0.0, 0.0

    arr = np.array(values, dtype=np.float64)
    q1 = float(np.quantile(arr, 0.25))
    q3 = float(np.quantile(arr, 0.75))
    iqr = q3 - q1
    lower_bound = q1 - multiplier * iqr
    upper_bound = q3 + multiplier * iqr

    outlier_count = 0
    for val in arr:
        if val < lower_bound or val > upper_bound:
            outlier_count += 1

    outlier_pct = (outlier_count / len(arr)) * 100.0 if len(arr) > 0 else 0.0
    return outlier_count, outlier_pct, lower_bound, upper_bound


def normalize_engagement_events(rows: List[Dict[str, str]]) -> Dict[str, float]:
    """Compute normalization parameters for engagement_events."""
    values = []
    for row in rows:
        val = safe_float(row.get("engagement_events"))
        if val is not None:
            values.append(val)

    if not values:
        return {"min": 0.0, "max": 1.0, "mean": 0.5, "std": 0.0}

    arr = np.array(values, dtype=np.float64)
    return {
        "min": float(np.min(arr)),
        "max": float(np.max(arr)),
        "mean": float(np.mean(arr)),
        "std": float(np.std(arr)),
    }


def clean_rows(
    rows: List[Dict[str, str]],
    columns: List[str],
    normalization: Dict[str, float],
    missing_threshold: float,
) -> Tuple[List[Dict[str, str]], Dict[str, int]]:
    """Clean rows by removing/fixing problematic data."""
    numeric_cols = [
        "num_courses",
        "problem_total",
        "problem_accuracy",
        "avg_attempts",
        "avg_score",
        "video_sessions",
        "video_count",
        "segment_count",
        "watched_seconds",
        "watched_hours",
        "avg_speed",
        "reply_count",
        "comment_count",
        "forum_total",
        "engagement_events",
    ]

    cleaned_rows: List[Dict[str, str]] = []
    stats = {
        "total": len(rows),
        "removed_too_many_missing": 0,
        "removed_invalid": 0,
        "kept": 0,
    }

    for row in rows:
        missing_count = 0
        for col in numeric_cols:
            if col in columns and (row.get(col) is None or str(row.get(col)).strip() == ""):
                missing_count += 1

        missing_ratio = missing_count / len(numeric_cols)
        if missing_ratio > missing_threshold:
            stats["removed_too_many_missing"] += 1
            continue

        is_valid = True
        for col in numeric_cols:
            if col in columns:
                val = safe_float(row.get(col))
                if val is not None and (np.isnan(val) or np.isinf(val)):
                    is_valid = False
                    break

        if not is_valid:
            stats["removed_invalid"] += 1
            continue

        cleaned_rows.append(row)
        stats["kept"] += 1

    return cleaned_rows, stats


def write_normalized_csv(
    rows: List[Dict[str, str]],
    output_path: Path,
    normalization: Dict[str, float],
) -> None:
    """Write normalized engagement_events to CSV."""
    min_val = normalization["min"]
    max_val = normalization["max"]

    with output_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["user_id", "engagement_events", "engagement_events_normalized"])

        for row in rows:
            user_id = row.get("user_id", "")
            e_raw = safe_float(row.get("engagement_events"))

            if e_raw is None:
                continue

            if max_val <= min_val:
                e_norm = 0.0
            else:
                e_norm = (e_raw - min_val) / (max_val - min_val)
                e_norm = max(0.0, min(1.0, e_norm))

            writer.writerow([user_id, round(e_raw, 6), round(e_norm, 6)])


def write_clean_csv(
    rows: List[Dict[str, str]],
    columns: List[str],
    output_path: Path,
) -> None:
    """Write cleaned data to CSV."""
    with output_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


def write_report(
    output_path: Path,
    original_rows: int,
    cleaned_rows: int,
    stats_before: Dict[str, Dict[str, float]],
    stats_after: Dict[str, Dict[str, float]],
    normalization: Dict[str, float],
    cleaning_stats: Dict[str, int],
    outlier_detection: Dict[str, Tuple[int, float, float, float]],
    elapsed: float,
) -> None:
    """Write comprehensive EDA report."""
    with output_path.open("w", encoding="utf-8") as f:
        f.write("Phase 1B - EDA and Data Cleaning Report\n")
        f.write("=" * 100 + "\n")
        f.write(f"Generated at                 : {now_text()}\n")
        f.write(f"Elapsed (seconds)            : {elapsed:.2f}\n")

        f.write("\n1. DATA OVERVIEW\n")
        f.write("-" * 100 + "\n")
        f.write(f"Original rows                : {original_rows:,}\n")
        f.write(f"After cleaning               : {cleaned_rows:,}\n")
        f.write(f"Rows removed (missing data)  : {cleaning_stats['removed_too_many_missing']:,}\n")
        f.write(f"Rows removed (invalid)       : {cleaning_stats['removed_invalid']:,}\n")
        f.write(f"Retention rate               : {(cleaned_rows / original_rows * 100):.2f}%\n")

        f.write("\n2. DESCRIPTIVE STATISTICS (BEFORE CLEANING)\n")
        f.write("-" * 100 + "\n")
        f.write(f"{'Column':<20} {'Count':>10} {'Missing':>10} {'Mean':>12} {'Std':>12} {'Min':>12} {'Max':>12}\n")
        for col in sorted(stats_before.keys()):
            stat = stats_before[col]
            f.write(
                f"{col:<20} {int(stat['count']):>10,} {int(stat['missing']):>10,} "
                f"{stat['mean']:>12.4f} {stat['std']:>12.4f} {stat['min']:>12.4f} {stat['max']:>12.4f}\n"
            )

        f.write("\n3. DESCRIPTIVE STATISTICS (AFTER CLEANING)\n")
        f.write("-" * 100 + "\n")
        f.write(f"{'Column':<20} {'Count':>10} {'Missing':>10} {'Mean':>12} {'Std':>12} {'Min':>12} {'Max':>12}\n")
        for col in sorted(stats_after.keys()):
            stat = stats_after[col]
            f.write(
                f"{col:<20} {int(stat['count']):>10,} {int(stat['missing']):>10,} "
                f"{stat['mean']:>12.4f} {stat['std']:>12.4f} {stat['min']:>12.4f} {stat['max']:>12.4f}\n"
            )

        f.write("\n4. OUTLIER DETECTION (IQR Method, Multiplier=1.5)\n")
        f.write("-" * 100 + "\n")
        f.write(f"{'Column':<20} {'Outliers':>10} {'%':>8} {'Lower Bound':>15} {'Upper Bound':>15}\n")
        for col in sorted(outlier_detection.keys()):
            count, pct, lower, upper = outlier_detection[col]
            f.write(
                f"{col:<20} {count:>10,} {pct:>7.2f}% {lower:>15.4f} {upper:>15.4f}\n"
            )

        f.write("\n5. ENGAGEMENT_EVENTS NORMALIZATION\n")
        f.write("-" * 100 + "\n")
        f.write(f"Raw min value                : {normalization['min']:.6f}\n")
        f.write(f"Raw max value                : {normalization['max']:.6f}\n")
        f.write(f"Raw mean                     : {normalization['mean']:.6f}\n")
        f.write(f"Raw std                      : {normalization['std']:.6f}\n")
        f.write(f"Normalized range             : [0.0, 1.0]\n")
        f.write("Formula: (E - min) / (max - min), clipped to [0, 1]\n")

        f.write("\n6. DATA QUALITY SUMMARY\n")
        f.write("-" * 100 + "\n")
        f.write("Data Cleaning Steps:\n")
        f.write(f"- Removed rows with >30% missing numeric values: {cleaning_stats['removed_too_many_missing']:,}\n")
        f.write(f"- Removed rows with invalid values (NaN/Inf): {cleaning_stats['removed_invalid']:,}\n")
        f.write(f"- Final clean dataset: {cleaned_rows:,} rows\n")

        f.write("\nData Quality Checks:\n")
        for col in sorted(outlier_detection.keys()):
            count, pct, _, _ = outlier_detection[col]
            if pct > 5.0:
                f.write(f"⚠️  {col}: {pct:.2f}% outliers detected (>5% threshold)\n")
            elif pct > 1.0:
                f.write(f"ℹ️  {col}: {pct:.2f}% outliers detected (acceptable)\n")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Phase 1B: EDA and data cleaning for combined metrics."
    )
    parser.add_argument(
        "--combined-csv",
        type=Path,
        default=Path("results/combined_user_metrics.csv"),
        help="Input combined metrics CSV",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results"),
        help="Output directory for cleaned data and report",
    )
    parser.add_argument(
        "--missing-threshold",
        type=float,
        default=0.3,
        help="Remove rows with missing ratio > threshold (default: 0.3)",
    )
    parser.add_argument(
        "--outlier-iqr-multiplier",
        type=float,
        default=1.5,
        help="IQR multiplier for outlier detection (default: 1.5)",
    )
    parser.add_argument(
        "--max-rows",
        type=int,
        default=None,
        help="Optional cap for testing",
    )
    parser.add_argument(
        "--log-every",
        type=int,
        default=100000,
        help="Progress log interval",
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()

    combined_csv = args.combined_csv.resolve()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    cfg = Phase1bConfig(
        combined_csv=combined_csv,
        output_dir=output_dir,
        output_clean_csv=output_dir / "combined_user_metrics_clean.csv",
        output_report_txt=output_dir / "phase1b_eda_report.txt",
        output_normalization_csv=output_dir / "engagement_events_normalized.csv",
        missing_threshold=args.missing_threshold,
        outlier_iqr_multiplier=args.outlier_iqr_multiplier,
        max_rows=args.max_rows,
    )

    if not cfg.combined_csv.exists():
        log(f"FAILED: Combined CSV not found: {cfg.combined_csv}")
        return 1

    started = time.time()
    try:
        log("Starting Phase 1B: EDA and Data Cleaning")

        log("Step 1/6: Loading combined metrics")
        rows, columns = load_combined_csv(cfg.combined_csv, cfg.max_rows)
        original_count = len(rows)
        log(f"Loaded {original_count:,} rows with {len(columns)} columns")

        log("Step 2/6: Computing descriptive statistics (before cleaning)")
        stats_before = compute_descriptive_stats(rows)

        log("Step 3/6: Computing normalization parameters for engagement_events")
        normalization = normalize_engagement_events(rows)
        log(
            f"Engagement events range: [{normalization['min']:.2f}, {normalization['max']:.2f}], "
            f"mean={normalization['mean']:.2f}, std={normalization['std']:.2f}"
        )

        log("Step 4/6: Detecting outliers")
        numeric_cols = [
            "num_courses",
            "problem_total",
            "problem_accuracy",
            "avg_attempts",
            "avg_score",
            "video_sessions",
            "video_count",
            "segment_count",
            "watched_seconds",
            "watched_hours",
            "avg_speed",
            "reply_count",
            "comment_count",
            "forum_total",
            "engagement_events",
        ]
        outlier_detection: Dict[str, Tuple[int, float, float, float]] = {}
        for col in numeric_cols:
            count, pct, lower, upper = detect_outliers_iqr(rows, col, cfg.outlier_iqr_multiplier)
            outlier_detection[col] = (count, pct, lower, upper)

        log("Step 5/6: Cleaning rows")
        cleaned_rows, cleaning_stats = clean_rows(rows, columns, normalization, cfg.missing_threshold)
        log(
            f"Cleaning complete: kept {cleaning_stats['kept']:,}/{original_count:,} rows "
            f"({cleaning_stats['kept']/original_count*100:.1f}%)"
        )

        log("Step 6/6: Computing statistics after cleaning and writing outputs")
        stats_after = compute_descriptive_stats(cleaned_rows)

        write_clean_csv(cleaned_rows, columns, cfg.output_clean_csv)
        log(f"Cleaned CSV: {cfg.output_clean_csv}")

        write_normalized_csv(cleaned_rows, cfg.output_normalization_csv, normalization)
        log(f"Normalized engagement events: {cfg.output_normalization_csv}")

        write_report(
            cfg.output_report_txt,
            original_count,
            len(cleaned_rows),
            stats_before,
            stats_after,
            normalization,
            cleaning_stats,
            outlier_detection,
            time.time() - started,
        )
        log(f"EDA Report: {cfg.output_report_txt}")

        log(f"Phase 1B completed in {time.time() - started:.2f}s")
        return 0
    except Exception as exc:
        log(f"FAILED: {exc}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
