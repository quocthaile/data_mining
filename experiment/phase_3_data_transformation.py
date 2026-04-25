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

def normalize_engagement_events(rows: List[Dict[str, str]]) -> Dict[str, float]:
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

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--combined-input", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--max-rows", type=int, default=None)
    args, _ = parser.parse_known_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] Loading data from {args.combined_input}")
    
    rows, columns = eda_lib.load_combined_csv(args.combined_input, args.max_rows)
    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] Computing normalization parameters...")
    normalization = normalize_engagement_events(rows)
    
    output_csv = args.output_dir / "combined_user_metrics_transformed.csv"
    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] Writing transformed CSV to {output_csv}")
    
    min_val = normalization["min"]
    max_val = normalization["max"]

    if "engagement_events_normalized" not in columns:
        columns.append("engagement_events_normalized")

    with output_csv.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=columns)
        writer.writeheader()
        
        for row in rows:
            e_raw = safe_float(row.get("engagement_events"))
            if e_raw is None:
                row["engagement_events_normalized"] = ""
            else:
                if max_val <= min_val:
                    e_norm = 0.0
                else:
                    e_norm = (e_raw - min_val) / (max_val - min_val)
                    e_norm = max(0.0, min(1.0, e_norm))
                row["engagement_events_normalized"] = str(round(e_norm, 6))
            writer.writerow(row)

    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] Phase 3 Transformation completed.")

if __name__ == "__main__":
    main()
