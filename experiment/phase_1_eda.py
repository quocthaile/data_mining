import argparse
import time
from pathlib import Path
import utils_eda as eda_lib

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--combined-input", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--max-rows", type=int, default=None)
    args, _ = parser.parse_known_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] Loading data from {args.combined_input}")
    
    rows, columns = eda_lib.load_combined_csv(args.combined_input, args.max_rows)
    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] Computing descriptive statistics...")
    stats = eda_lib.compute_descriptive_stats(rows)
    
    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] Detecting outliers...")
    outlier_detection = {}
    numeric_cols = [
        "num_courses", "problem_total", "problem_accuracy", "avg_attempts",
        "avg_score", "video_sessions", "video_count", "segment_count",
        "watched_seconds", "watched_hours", "avg_speed",
        "reply_count", "comment_count", "forum_total", "engagement_events",
    ]
    for col in numeric_cols:
        outlier_detection[col] = eda_lib.detect_outliers_iqr(rows, col, 1.5)
        
    report_path = args.output_dir / "phase1_eda_report.txt"
    with report_path.open("w", encoding="utf-8") as f:
        f.write("Phase 1 - EDA Report\n")
        f.write("====================================================================================================\n")
        f.write(f"Generated at                 : {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Total rows                   : {len(rows):,}\n\n")
        
        f.write("1. DESCRIPTIVE STATISTICS\n")
        f.write("----------------------------------------------------------------------------------------------------\n")
        f.write(f"{'Column':<20} {'Count':>10} {'Missing':>10} {'Mean':>12} {'Std':>12} {'Min':>12} {'Max':>12}\n")
        for col in sorted(stats.keys()):
            stat = stats[col]
            f.write(f"{col:<20} {int(stat['count']):>10,} {int(stat['missing']):>10,} {stat['mean']:>12.4f} {stat['std']:>12.4f} {stat['min']:>12.4f} {stat['max']:>12.4f}\n")
            
        f.write("\n2. OUTLIER DETECTION (IQR Method, Multiplier=1.5)\n")
        f.write("----------------------------------------------------------------------------------------------------\n")
        f.write(f"{'Column':<20} {'Outliers':>10} {'%':>8} {'Lower Bound':>15} {'Upper Bound':>15}\n")
        for col in sorted(outlier_detection.keys()):
            count, pct, lower, upper = outlier_detection[col]
            f.write(f"{col:<20} {count:>10,} {pct:>7.2f}% {lower:>15.4f} {upper:>15.4f}\n")
            
    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] EDA report written to {report_path}")

if __name__ == "__main__":
    main()
