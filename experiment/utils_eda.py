import csv
from pathlib import Path
from typing import List, Dict, Tuple, Optional
import numpy as np

def log(message: str) -> None:
    import time
    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {message}")

def safe_float(value) -> Optional[float]:
    if value is None or str(value).strip() == "":
        return None
    try:
        return float(value)
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
) -> Tuple[int, float, float, float]:
    values = []
    for row in rows:
        val = safe_float(row.get(col))
        if val is not None:
            values.append(val)

    if not values:
        return 0, 0.0, 0.0, 0.0

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
