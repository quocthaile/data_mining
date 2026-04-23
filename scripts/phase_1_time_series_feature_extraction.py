#!/usr/bin/env python3
"""
Phase 1: Time-series feature extraction (scenario aligned).

This phase groups legacy steps into one stage:
1) Translate school names for robust downstream joins/display.
2) Build user-level + user-week features from raw JSONL logs (streaming, RAM-safe), or convert from a Parquet source file.

Outputs:
- results/combined_user_metrics.csv
- results/step2_user_week_activity.csv
- results/combine_summary.txt
"""

from __future__ import annotations

import argparse
import csv
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional


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


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Phase 1: time-series feature extraction from MOOCCubeX logs."
    )
    parser.add_argument("--dataset-dir", type=Path, default=Path(r"D:\MOOCCubeX_dataset"))
    parser.add_argument("--output-dir", type=Path, default=Path("results"))
    parser.add_argument("--user-input", type=Path, default=Path("user.json"))
    parser.add_argument("--translated-user", type=Path, default=Path("user_school_en.json"))
    parser.add_argument(
        "--translate-summary",
        type=Path,
        default=Path("school_translate_summary.txt"),
    )
    parser.add_argument("--skip-translate", action="store_true")
    parser.add_argument(
        "--combined-parquet",
        type=Path,
        default=None,
        help="Input Parquet file to convert into combined CSV/weekly activity output",
    )
    parser.add_argument("--combined-file", type=Path, default=Path("combined_user_metrics.csv"))
    parser.add_argument("--weekly-file", type=Path, default=Path("step2_user_week_activity.csv"))
    parser.add_argument("--db-file", type=Path, default=Path("combined_streaming.sqlite3"))
    parser.add_argument("--log-every", type=int, default=200000)
    parser.add_argument("--max-rows", type=int, default=None)
    return parser


def main() -> int:
    args = build_parser().parse_args()

    project_root = Path(__file__).resolve().parents[1]
    scripts_dir = (project_root / "scripts").resolve()

    dataset_dir = resolve_path_arg(args.dataset_dir, project_root, project_root)
    output_dir = resolve_path_arg(args.output_dir, project_root, project_root)
    user_input = resolve_path_arg(args.user_input, project_root, dataset_dir)
    translated_user = resolve_path_arg(args.translated_user, project_root, dataset_dir)
    translate_summary = resolve_path_arg(args.translate_summary, project_root, output_dir)
    combined_file = resolve_path_arg(args.combined_file, project_root, output_dir)
    weekly_file = resolve_path_arg(args.weekly_file, project_root, output_dir)
    db_file = resolve_path_arg(args.db_file, project_root, output_dir)

    def convert_parquet_input(parquet_path: Path) -> None:
        try:
            import pandas as pd
        except ImportError as exc:
            raise RuntimeError(
                "Reading Parquet input requires pandas and pyarrow. "
                "Install them with pip install pandas pyarrow."
            ) from exc

        if not parquet_path.exists():
            raise FileNotFoundError(f"Missing Parquet input file: {parquet_path}")

        start_at = time.time()
        log(f"Converting Parquet input to combined CSVs: {parquet_path}")
        df = pd.read_parquet(parquet_path)

        if args.max_rows is not None:
            df = df.head(args.max_rows)

        if "user_id" not in df.columns:
            raise ValueError("Parquet input must contain a 'user_id' column.")

        df["user_id"] = df["user_id"].astype(str).str.strip()
        if "num_courses" not in df.columns and "course_order" in df.columns:
            df["num_courses"] = df["course_order"].apply(
                lambda x: len(x) if isinstance(x, (list, tuple)) else int(x) if pd.notna(x) else 0
            )

        def first_non_null(series):
            for value in series:
                if pd.notna(value):
                    return value
            return None

        def normalize_num(value):
            if pd.isna(value):
                return 0
            try:
                return int(value)
            except (TypeError, ValueError):
                return 0

        def safe_float(value):
            if pd.isna(value):
                return None
            try:
                return float(value)
            except (TypeError, ValueError):
                return None

        def parse_week_from_datetime(value):
            if pd.isna(value):
                return None
            try:
                ts = pd.to_datetime(value, errors="coerce")
                if pd.isna(ts):
                    return None
                return int(ts.isocalendar().week)
            except Exception:
                return None

        def extract_video_metrics(seq):
            if not isinstance(seq, (list, tuple)):
                return 0, 0, 0, 0.0, 0.0, 0

            sessions = 1
            video_count = 0
            segment_count = 0
            watched_seconds = 0.0
            speed_sum = 0.0
            speed_count = 0

            for item in seq:
                if not isinstance(item, dict):
                    continue

                segments = item.get("segment")
                if not isinstance(segments, list):
                    continue

                video_count += len(segments)
                for seg in segments:
                    if not isinstance(seg, dict):
                        continue

                    segment_count += 1
                    start = safe_float(seg.get("start_point"))
                    end = safe_float(seg.get("end_point"))
                    if start is not None and end is not None and end >= start:
                        watched_seconds += end - start
                    speed = safe_float(seg.get("speed"))
                    if speed is not None and speed > 0:
                        speed_sum += speed
                        speed_count += 1

            return sessions, video_count, segment_count, watched_seconds, speed_sum, speed_count

        def bool_flag(series, col_names):
            for col_name in col_names:
                if col_name in series.index and pd.notna(series[col_name]):
                    return 1
            return 0

        grouped = df.groupby("user_id")
        combined_rows: List[Dict[str, Any]] = []
        for user_id, group in grouped:
            num_courses = normalize_num(first_non_null(group.get("num_courses", pd.Series([], dtype="float64"))))
            if num_courses <= 5:
                continue

            problem_total = int(group["problem_id"].notna().sum()) if "problem_id" in group.columns else 0
            problem_correct = int(
                ((group.get("is_correct") == 1) | (group.get("is_correct") == True) | (group.get("is_correct") == "1")).sum()
            ) if "is_correct" in group.columns else 0
            attempts_sum = float(pd.to_numeric(group.get("attempts", pd.Series([], dtype="float64")), errors="coerce").fillna(0).sum())
            score_sum = float(pd.to_numeric(group.get("score", pd.Series([], dtype="float64")), errors="coerce").fillna(0).sum())
            score_count = int(pd.to_numeric(group.get("score", pd.Series([], dtype="float64")), errors="coerce").notna().sum())

            video_sessions = 0
            video_count = 0
            segment_count = 0
            watched_seconds = 0.0
            speed_sum = 0.0
            speed_count = 0
            if "seq" in group.columns:
                for seq in group["seq"]:
                    session_count, count, segments, watched, speed_total, speed_n = extract_video_metrics(seq)
                    video_sessions += session_count
                    video_count += count
                    segment_count += segments
                    watched_seconds += watched
                    speed_sum += speed_total
                    speed_count += speed_n

            avg_speed = (speed_sum / speed_count) if speed_count > 0 else 0.0
            watched_hours = watched_seconds / 3600.0

            reply_count = int(group["id_x"].notna().sum()) if "id_x" in group.columns else 0
            comment_count = int(group["id_y"].notna().sum()) if "id_y" in group.columns else 0
            forum_total = reply_count + comment_count
            engagement_events = problem_total + video_count + forum_total

            first_activity = None
            last_activity = None
            if "submit_time" in group.columns:
                times = group["submit_time"].dropna().astype(str)
                if not times.empty:
                    first_activity = times.min()
                    last_activity = times.max()

            combined_rows.append(
                {
                    "user_id": user_id,
                    "gender": first_non_null(group.get("gender", pd.Series([], dtype=object))),
                    "school": first_non_null(group.get("school", pd.Series([], dtype=object))),
                    "year_of_birth": first_non_null(group.get("year_of_birth", pd.Series([], dtype=object))),
                    "num_courses": num_courses,
                    "problem_total": problem_total,
                    "problem_correct": problem_correct,
                    "attempts_sum": round(attempts_sum, 6),
                    "score_sum": round(score_sum, 6),
                    "score_count": score_count,
                    "video_sessions": video_sessions,
                    "video_count": video_count,
                    "segment_count": segment_count,
                    "watched_seconds": watched_seconds,
                    "watched_hours": watched_hours,
                    "avg_speed": avg_speed,
                    "reply_count": reply_count,
                    "comment_count": comment_count,
                    "forum_total": forum_total,
                    "engagement_events": engagement_events,
                    "first_activity_time": first_activity,
                    "last_activity_time": last_activity,
                }
            )

        combined_file.parent.mkdir(parents=True, exist_ok=True)
        with combined_file.open("w", encoding="utf-8", newline="") as out_f:
            writer = csv.writer(out_f)
            writer.writerow(
                [
                    "user_id",
                    "gender",
                    "school",
                    "year_of_birth",
                    "num_courses",
                    "problem_total",
                    "problem_correct",
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
                    "first_activity_time",
                    "last_activity_time",
                ]
            )
            for row in combined_rows:
                problem_accuracy = (row["problem_correct"] / row["problem_total"]) if row["problem_total"] > 0 else 0.0
                avg_attempts = (row["attempts_sum"] / row["problem_total"]) if row["problem_total"] > 0 else 0.0
                avg_score = (row["score_sum"] / row["score_count"]) if row["score_count"] > 0 else 0.0
                writer.writerow(
                    [
                        row["user_id"],
                        row["gender"],
                        row["school"],
                        row["year_of_birth"],
                        row["num_courses"],
                        row["problem_total"],
                        row["problem_correct"],
                        round(problem_accuracy, 6),
                        round(avg_attempts, 6),
                        round(avg_score, 6),
                        row["video_sessions"],
                        row["video_count"],
                        row["segment_count"],
                        round(row["watched_seconds"], 3),
                        round(row["watched_hours"], 6),
                        round(row["avg_speed"], 6),
                        row["reply_count"],
                        row["comment_count"],
                        row["forum_total"],
                        row["engagement_events"],
                        row["first_activity_time"],
                        row["last_activity_time"],
                    ]
                )

        weekly_rows: List[List[Any]] = []
        if "submit_time" in df.columns:
            times = pd.to_datetime(df["submit_time"], errors="coerce")
            valid = df[~times.isna()].copy()
            if not valid.empty:
                valid["week"] = valid["submit_time"].apply(parse_week_from_datetime)
                valid = valid[valid["week"].notna()]
                if not valid.empty:
                    valid["video"] = (
                        valid["seq"].notna().astype(int) if "seq" in valid.columns else 0
                    )
                    valid["problem"] = (
                        valid["problem_id"].notna().astype(int) if "problem_id" in valid.columns else 0
                    )
                    valid["reply"] = (
                        valid["id_x"].notna().astype(int) if "id_x" in valid.columns else 0
                    )
                    valid["comment"] = (
                        valid["id_y"].notna().astype(int) if "id_y" in valid.columns else 0
                    )
                    valid["video"] = valid["video"].astype(int)
                    valid["problem"] = valid["problem"].astype(int)
                    valid["reply"] = valid["reply"].astype(int)
                    valid["comment"] = valid["comment"].astype(int)

                    agg = valid.groupby(["user_id", "week"], as_index=False)[["video", "problem", "reply", "comment"]].max()
                    agg = agg.sort_values(["user_id", "week"])
                    weekly_rows = agg.values.tolist()

        weekly_csv.parent.mkdir(parents=True, exist_ok=True)
        with weekly_csv.open("w", encoding="utf-8", newline="") as out_f:
            writer = csv.writer(out_f)
            writer.writerow(["user_id", "week", "video", "problem", "reply", "comment"])
            writer.writerows(weekly_rows)

        summary_path = output_dir / "combine_summary.txt"
        with summary_path.open("w", encoding="utf-8") as f:
            f.write("Parquet-derived Combine Summary\n")
            f.write("=" * 80 + "\n")
            f.write(f"Parquet input          : {parquet_path}\n")
            f.write(f"Combined CSV output    : {combined_file}\n")
            f.write(f"Weekly CSV output      : {weekly_csv}\n")
            f.write(f"Selected users         : {len(combined_rows):,}\n")
            f.write(f"Source rows            : {len(df):,}\n")
            f.write(f"Elapsed (seconds)      : {time.time() - start_at:.2f}\n")

    try:
        started = time.time()
        log("Starting Phase 1: Time-series feature extraction")

        if args.combined_parquet is not None:
            parquet_path = resolve_path_arg(args.combined_parquet, project_root, project_root)
            convert_parquet_input(parquet_path)
            log("Phase 1 completed using Parquet input. Skipping translation and JSONL combine.")
            return 0

        step_1 = scripts_dir / "step_1_translate_school_zh_to_en.py"
        step_2 = scripts_dir / "step_2_combine_data_streaming.py"

        if not args.skip_translate:
            cmd1 = [
                sys.executable,
                str(step_1),
                "--input",
                str(user_input),
                "--output",
                str(translated_user),
                "--summary",
                str(translate_summary),
                "--log-every",
                str(max(1, args.log_every)),
            ]
            if args.max_rows is not None:
                cmd1.extend(["--max-lines", str(args.max_rows)])
            run_command(cmd1, project_root, "Phase 1.1 - School normalization")
        else:
            log("Phase 1.1 skipped (--skip-translate)")

        cmd2 = [
            sys.executable,
            str(step_2),
            "--dataset-dir",
            str(dataset_dir),
            "--output-dir",
            str(output_dir),
            "--user-file",
            str(translated_user),
            "--output-file",
            str(combined_file),
            "--output-weekly-file",
            str(weekly_file),
            "--db-file",
            str(db_file),
            "--log-every",
            str(max(1, args.log_every)),
        ]
        if args.max_rows is not None:
            cmd2.extend(["--max-lines-per-file", str(args.max_rows)])

        run_command(cmd2, project_root, "Phase 1.2 - Weekly feature extraction")
        log(f"Phase 1 completed in {time.time() - started:.2f}s")
        return 0
    except Exception as exc:
        log(f"FAILED: {exc}")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())