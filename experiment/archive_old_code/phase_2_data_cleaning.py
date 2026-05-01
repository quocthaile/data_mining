#!/usr/bin/env python3
"""
Phase 2: Data Cleaning

Reads the raw combined CSV output from Phase 1 and removes rows that are:
  - Missing more than `missing_threshold` fraction of numeric feature values.
  - Containing invalid numeric values (NaN or Inf).

Input :
  results/phase2/combined_user_metrics.csv   (from Phase 1 Combine step)

Output:
  results/phase2/combined_user_metrics_clean.csv
"""
import argparse
import csv
import time
from pathlib import Path
from typing import Optional, List, Dict, Tuple
import numpy as np
import utils_eda as eda_lib


def safe_float(value) -> Optional[float]:
    """Convert a raw CSV string value to float, returning None on failure."""
    if value is None or str(value).strip() == "":
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def clean_rows(
    rows: List[Dict[str, str]],
    columns: List[str],
    missing_threshold: float,
) -> Tuple[List[Dict[str, str]], Dict[str, int]]:
    """
    Filter rows by missing-value ratio and numeric validity.

    Args:
        rows: List of raw CSV row dicts.
        columns: Column names from the CSV header.
        missing_threshold: Fraction of allowed missing numeric columns (0.0 - 1.0).
                           Rows above this ratio are dropped.

    Returns:
        (cleaned_rows, stats) where stats contains counts of kept / removed rows.
    """
    # These are the numeric feature columns to check for missing / invalid values
    numeric_cols = [
        "num_courses", "problem_total", "problem_accuracy", "avg_attempts",
        "avg_score", "video_sessions", "video_count", "segment_count",
        "watched_seconds", "watched_hours", "avg_speed",
        "reply_count", "comment_count", "forum_total", "engagement_events",
    ]

    cleaned_rows: List[Dict[str, str]] = []
    stats = {
        "total": len(rows),
        "removed_too_many_missing": 0,
        "removed_invalid": 0,
        "kept": 0,
    }

    for row in rows:
        # Count how many numeric columns are missing in this row
        missing_count = 0
        for col in numeric_cols:
            if col in columns and (row.get(col) is None or str(row.get(col)).strip() == ""):
                missing_count += 1

        missing_ratio = missing_count / len(numeric_cols)
        if missing_ratio > missing_threshold:
            stats["removed_too_many_missing"] += 1
            continue

        # Check for NaN or Inf in any numeric column
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


def main():
    parser = argparse.ArgumentParser(description="Phase 2: Data Cleaning")
    parser.add_argument("--combined-input", type=Path, required=True,
                        help="Path to combined_user_metrics.csv from Phase 1")
    parser.add_argument("--output-dir", type=Path, required=True,
                        help="Directory to write cleaned CSV")
    parser.add_argument("--missing-threshold", type=float, default=0.3,
                        help="Maximum allowed fraction of missing numeric columns per row (default: 0.3)")
    parser.add_argument("--max-rows", type=int, default=None,
                        help="Limit number of input rows for testing")
    args, _ = parser.parse_known_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] Loading data from {args.combined_input}")

    rows, columns = eda_lib.load_combined_csv(args.combined_input, args.max_rows)
    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] Cleaning data...")
    cleaned_rows, cleaning_stats = clean_rows(rows, columns, args.missing_threshold)

    output_clean_csv = args.output_dir / "combined_user_metrics_clean.csv"
    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] Writing clean CSV to {output_clean_csv}")

    with output_clean_csv.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=columns)
        writer.writeheader()
        writer.writerows(cleaned_rows)

    print(
        f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] Cleaned: "
        f"{cleaning_stats['kept']:,} rows kept, "
        f"{cleaning_stats['removed_too_many_missing']:,} removed (missing), "
        f"{cleaning_stats['removed_invalid']:,} removed (invalid)"
    )


if __name__ == "__main__":
    main()
