import argparse
import csv
import time
from pathlib import Path
from typing import Optional, List, Dict, Tuple
import numpy as np
import utils_eda as eda_lib

def safe_float(value) -> Optional[float]:
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

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--combined-input", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--missing-threshold", type=float, default=0.3)
    parser.add_argument("--max-rows", type=int, default=None)
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

    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] Cleaned: {cleaning_stats['kept']:,} rows kept, {cleaning_stats['removed_too_many_missing']:,} removed (missing), {cleaning_stats['removed_invalid']:,} removed (invalid)")

if __name__ == "__main__":
    main()
