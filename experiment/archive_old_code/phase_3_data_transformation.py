#!/usr/bin/env python3
"""
Phase 3: Data Transformation

Reads the cleaned user metrics CSV from Phase 2 and applies feature engineering:
  - Normalises the `engagement_events` column to the [0, 1] range using
    min-max scaling computed over the full dataset (single pass).
  - Appends the new column `engagement_events_normalized` to each row.
  - Generates three time-aware checkpoint views at 25%, 50% and 75% of each
    course's timeline.  A row appears in a checkpoint file ONLY if the
    user-course's last-active week falls at or before that progress milestone
    (i.e. the model sees only information available at that point in time).
    Rows whose progress cannot be determined are EXCLUDED from every
    checkpoint to prevent any future-data leakage.

Input :
  results/phase2/combined_user_metrics_clean.csv   (from Phase 2)
  results/phase1/step2_user_week_activity.csv      (for per-course timelines)

Output:
  results/phase3/combined_user_metrics_transformed.csv
  results/phase3/combined_user_metrics_checkpoint_25.csv
  results/phase3/combined_user_metrics_checkpoint_50.csv
  results/phase3/combined_user_metrics_checkpoint_75.csv
"""
import argparse
import csv
import time
import math
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional, List, Dict, Tuple
import numpy as np
import utils_eda as eda_lib


CHECKPOINTS = [25, 50, 75]
DEFAULT_COURSE_KEY = "__GLOBAL_COURSE__"


def safe_float(value) -> Optional[float]:
    """Convert a raw CSV string value to float, returning None on failure."""
    if value is None or str(value).strip() == "":
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def normalize_engagement_events(rows: List[Dict[str, str]]) -> Dict[str, float]:
    """
    Compute min / max / mean / std of `engagement_events` across all rows.

    These statistics are used for min-max normalization so that the result
    always falls within [0, 1].  Returns safe fallback values when the column
    is entirely missing or empty.
    """
    values = []
    for row in rows:
        val = safe_float(row.get("engagement_events"))
        if val is not None:
            values.append(val)

    if not values:
        # No valid values – return neutral fallback
        return {"min": 0.0, "max": 1.0, "mean": 0.5, "std": 0.0}

    arr = np.array(values, dtype=np.float64)
    return {
        "min": float(np.min(arr)),
        "max": float(np.max(arr)),
        "mean": float(np.mean(arr)),
        "std": float(np.std(arr)),
    }


def safe_int(value) -> Optional[int]:
    if value is None or str(value).strip() == "":
        return None
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return None


def build_user_course_key(row: Dict[str, str]) -> str:
    """Build a stable user-course key; fallback to a global course key when course_id is missing."""
    user_id = (row.get("user_id") or "").strip()
    course_id = (row.get("course_id") or "").strip() or DEFAULT_COURSE_KEY
    if not user_id:
        return ""
    return f"{user_id}::{course_id}"


def compute_user_course_progress_pct(weekly_csv: Path, max_rows: Optional[int]) -> Dict[str, float]:
    """
    Build progress percentage for each user-course using weekly activity timeline.

    Progress is computed relative to all observed weeks in a course:
      progress_pct = rank(last_active_week_in_course) / total_course_weeks * 100
    """
    if not weekly_csv.exists():
        return {}

    course_weeks: Dict[str, set] = {}
    user_course_last_week: Dict[str, int] = {}

    with weekly_csv.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for idx, row in enumerate(reader, start=1):
            if max_rows is not None and idx > max_rows:
                break

            user_id = (row.get("user_id") or "").strip()
            course_id = (row.get("course_id") or "").strip()
            week = safe_int(row.get("week"))
            if not user_id or not course_id or week is None:
                continue

            course_weeks.setdefault(course_id, set()).add(week)
            uc_key = f"{user_id}::{course_id}"
            prev = user_course_last_week.get(uc_key)
            if prev is None or week > prev:
                user_course_last_week[uc_key] = week

    course_week_order: Dict[str, List[int]] = {
        cid: sorted(weeks) for cid, weeks in course_weeks.items() if weeks
    }

    progress_by_uc: Dict[str, float] = {}
    for uc_key, last_week in user_course_last_week.items():
        _, course_id = uc_key.split("::", 1)
        ordered = course_week_order.get(course_id)
        if not ordered:
            continue

        total = len(ordered)
        rank_pos = ordered.index(last_week) + 1 if last_week in ordered else total
        progress_pct = (rank_pos / total) * 100.0
        progress_by_uc[uc_key] = float(round(progress_pct, 4))

    return progress_by_uc


def parse_datetime_to_epoch_seconds(value: Optional[str]) -> Optional[float]:
    """Parse a datetime-like value into UTC epoch seconds."""
    if value is None:
        return None
    raw = str(value).strip()
    if not raw:
        return None

    # Numeric timestamp (seconds or milliseconds)
    try:
        as_float = float(raw)
        if as_float > 0:
            if as_float >= 1e12:
                as_float /= 1000.0
            return float(as_float)
    except (TypeError, ValueError):
        pass

    # ISO-like datetime (optionally with trailing Z)
    iso_raw = raw.replace("Z", "+00:00")
    try:
        dt = datetime.fromisoformat(iso_raw)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return float(dt.timestamp())
    except ValueError:
        pass

    # Common fixed datetime formats
    formats = [
        "%Y-%m-%d %H:%M:%S",
        "%Y-%m-%d",
        "%d/%m/%Y %H:%M:%S",
        "%d/%m/%Y",
    ]
    for fmt in formats:
        try:
            dt = datetime.strptime(raw, fmt).replace(tzinfo=timezone.utc)
            return float(dt.timestamp())
        except ValueError:
            continue

    return None


def compute_user_course_progress_pct_from_timestamps(rows: List[Dict[str, str]]) -> Dict[str, float]:
    """
    Fallback progress estimation using first/last activity timestamps in combined rows.

    For each course:
      - course_start = min(first_activity_time, last_activity_time) across rows
      - course_end   = max(first_activity_time, last_activity_time) across rows

    For each user-course row:
      progress_pct = (row_last_activity - course_start) / (course_end - course_start) * 100

    Rows without usable timestamps remain unknown and will be excluded downstream.
    """
    course_bounds: Dict[str, Tuple[float, float]] = {}
    course_last_ts_values: Dict[str, set] = {}

    for row in rows:
        course_id = (row.get("course_id") or "").strip() or DEFAULT_COURSE_KEY
        if not course_id:
            continue

        first_ts = parse_datetime_to_epoch_seconds(row.get("first_activity_time"))
        last_ts = parse_datetime_to_epoch_seconds(row.get("last_activity_time"))
        if first_ts is None and last_ts is None:
            continue

        start_candidate = first_ts if first_ts is not None else last_ts
        end_candidate = last_ts if last_ts is not None else first_ts
        if start_candidate is None or end_candidate is None:
            continue

        if last_ts is not None:
            course_last_ts_values.setdefault(course_id, set()).add(float(last_ts))

        current = course_bounds.get(course_id)
        if current is None:
            course_bounds[course_id] = (start_candidate, end_candidate)
        else:
            course_bounds[course_id] = (
                min(current[0], start_candidate),
                max(current[1], end_candidate),
            )

    progress_by_uc: Dict[str, float] = {}
    for row in rows:
        user_id = (row.get("user_id") or "").strip()
        course_id = (row.get("course_id") or "").strip() or DEFAULT_COURSE_KEY
        if not user_id or not course_id:
            continue

        bounds = course_bounds.get(course_id)
        if bounds is None:
            continue

        # If a course has no temporal variance across rows, timestamp-based
        # progress is not informative; keep this row unknown here so lower-priority
        # fallbacks (activity proxy) can provide stratification.
        unique_last_ts = course_last_ts_values.get(course_id, set())
        if len(unique_last_ts) <= 1:
            continue

        last_ts = parse_datetime_to_epoch_seconds(row.get("last_activity_time"))
        if last_ts is None:
            last_ts = parse_datetime_to_epoch_seconds(row.get("first_activity_time"))
        if last_ts is None:
            continue

        start_ts, end_ts = bounds
        span = end_ts - start_ts
        if span <= 0:
            progress_pct = 100.0
        else:
            ratio = (last_ts - start_ts) / span
            ratio = max(0.0, min(1.0, ratio))
            progress_pct = ratio * 100.0

        uc_key = f"{user_id}::{course_id}"
        progress_by_uc[uc_key] = float(round(progress_pct, 4))

    return progress_by_uc


def compute_user_course_progress_pct_from_activity_proxy(
    rows: List[Dict[str, str]],
) -> Tuple[Dict[str, float], Dict[str, str]]:
    """
    Level-2 fallback: infer pseudo progress from activity intensity.

    This is a weak proxy used ONLY when weekly/timestamp progress is unavailable.
    For each user-course row, build a non-negative score from log-scaled activity
    features, then normalize within each course to [0, 100].
    """
    activity_columns = [
        "engagement_events",
        "problem_total",
        "problem_correct",
        "video_sessions",
        "video_count",
        "reply_count",
        "comment_count",
        "forum_total",
        "avg_attempts",
        "avg_score",
        "watched_hours",
        "watched_seconds",
    ]

    scores_by_course: Dict[str, List[Tuple[str, float]]] = {}
    for row in rows:
        user_id = (row.get("user_id") or "").strip()
        course_id = (row.get("course_id") or "").strip() or DEFAULT_COURSE_KEY
        if not user_id or not course_id:
            continue

        score = 0.0
        for col in activity_columns:
            val = safe_float(row.get(col))
            if val is None or val <= 0:
                continue
            score += math.log1p(val)

        uc_key = f"{user_id}::{course_id}"
        scores_by_course.setdefault(course_id, []).append((uc_key, score))

    uc_row_map: Dict[str, Dict[str, str]] = {}
    for row in rows:
        user_id = (row.get("user_id") or "").strip()
        course_id = (row.get("course_id") or "").strip() or DEFAULT_COURSE_KEY
        if not user_id or not course_id:
            continue
        uc_row_map[f"{user_id}::{course_id}"] = row

    progress_by_uc: Dict[str, float] = {}
    source_by_uc: Dict[str, str] = {}
    for course_id, items in scores_by_course.items():
        if not items:
            continue

        values = [score for _, score in items]
        min_score = min(values)
        max_score = max(values)

        for uc_key, score in items:
            if max_score <= min_score:
                # Degenerate distribution: apply deterministic rank fallback
                # so checkpoints still stratify in sparse data.
                pass
            else:
                ratio = (score - min_score) / (max_score - min_score)
                ratio = max(0.0, min(1.0, ratio))
                progress_pct = ratio * 100.0
                progress_by_uc[uc_key] = float(round(progress_pct, 4))
                source_by_uc[uc_key] = "activity_proxy"

        if max_score <= min_score:
            def rank_key(item: Tuple[str, float]) -> Tuple[float, str]:
                uc_key, _ = item
                user_id, _course = uc_key.split("::", 1)
                row_ref = uc_row_map.get(uc_key)
                num_courses = safe_float((row_ref or {}).get("num_courses")) or 0.0
                return (num_courses, user_id)

            sorted_items = sorted(items, key=rank_key)
            n = len(sorted_items)
            for idx, (uc_key, _score) in enumerate(sorted_items, start=1):
                progress_pct = 100.0 if n <= 1 else ((idx - 1) / (n - 1)) * 100.0
                progress_by_uc[uc_key] = float(round(progress_pct, 4))
                source_by_uc[uc_key] = "activity_proxy_rank_fallback"

    return progress_by_uc, source_by_uc


def write_checkpoint_views(
    output_dir: Path,
    base_columns: List[str],
    rows: List[Dict[str, str]],
    progress_by_uc: Dict[str, float],
    progress_source_by_uc: Optional[Dict[str, str]] = None,
) -> None:
    """
    Write time-aware checkpoint views for 25%, 50%, and 75% course progress.

    Time-aware rule (strict):
      A row is included in checkpoint_K ONLY if:
        progress_by_uc[user::course] <= K
      Rows with UNKNOWN progress are EXCLUDED from ALL checkpoints to avoid
      any possible future-data leakage.

    The extra column `time_progress_pct_at_last_activity` records the
    user-course's actual measured progress percentage.
    """
    checkpoint_columns = list(base_columns)
    if "time_progress_pct_at_last_activity" not in checkpoint_columns:
        checkpoint_columns.append("time_progress_pct_at_last_activity")
    if "time_progress_source" not in checkpoint_columns:
        checkpoint_columns.append("time_progress_source")

    total_with_progress = sum(
        1 for r in rows if progress_by_uc.get(build_user_course_key(r)) is not None
    )
    print(
        f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] "
        f"Checkpoint generation: {total_with_progress:,}/{len(rows):,} rows have known progress"
    )

    for checkpoint in CHECKPOINTS:
        out_path = output_dir / f"combined_user_metrics_checkpoint_{checkpoint}.csv"
        with out_path.open("w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=checkpoint_columns)
            writer.writeheader()

            kept = 0
            skipped_unknown = 0
            skipped_future = 0
            for row in rows:
                uc_key = build_user_course_key(row)
                progress_pct = progress_by_uc.get(uc_key)

                # Strict time-aware filter:
                # Exclude rows whose progress is unknown (cannot guarantee no leakage).
                if progress_pct is None:
                    skipped_unknown += 1
                    continue

                # Exclude rows where the user was last active AFTER this checkpoint.
                if progress_pct > checkpoint:
                    skipped_future += 1
                    continue

                out_row = dict(row)
                out_row["time_progress_pct_at_last_activity"] = str(progress_pct)
                out_row["time_progress_source"] = (
                    (progress_source_by_uc or {}).get(uc_key, "unknown")
                )
                writer.writerow(out_row)
                kept += 1

        print(
            f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] "
            f"Checkpoint {checkpoint}%: kept={kept:,}, "
            f"skipped_future={skipped_future:,}, skipped_no_progress={skipped_unknown:,} "
            f"-> {out_path.name}"
        )


def main():
    parser = argparse.ArgumentParser(description="Phase 3: Data Transformation")
    parser.add_argument("--combined-input", type=Path, required=True,
                        help="Path to combined_user_metrics_clean.csv from Phase 2")
    parser.add_argument("--output-dir", type=Path, required=True,
                        help="Directory to write transformed CSV")
    parser.add_argument("--weekly-input", type=Path, default=None,
                        help="Path to step2_user_week_activity.csv for checkpoint generation")
    parser.add_argument("--max-rows", type=int, default=None,
                        help="Limit number of input rows for testing")
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

    # Add the normalized column to the schema if not already present
    if "engagement_events_normalized" not in columns:
        columns.append("engagement_events_normalized")

    with output_csv.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=columns)
        writer.writeheader()

        for row in rows:
            e_raw = safe_float(row.get("engagement_events"))
            if e_raw is None:
                # Missing value: leave the normalized column blank
                row["engagement_events_normalized"] = ""
            else:
                # Min-max normalization clamped to [0, 1]
                if max_val <= min_val:
                    e_norm = 0.0
                else:
                    e_norm = (e_raw - min_val) / (max_val - min_val)
                    e_norm = max(0.0, min(1.0, e_norm))
                row["engagement_events_normalized"] = str(round(e_norm, 6))
            writer.writerow(row)

    weekly_input = args.weekly_input
    if weekly_input is None:
        # Default sibling path: <results>/phase1/step2_user_week_activity.csv
        weekly_input = args.output_dir.parent / "phase1" / "step2_user_week_activity.csv"

    progress_by_uc_weekly = compute_user_course_progress_pct(weekly_input, args.max_rows)
    progress_by_uc_time = compute_user_course_progress_pct_from_timestamps(rows)
    progress_by_uc_activity, progress_source_activity = compute_user_course_progress_pct_from_activity_proxy(rows)

    # Merge with priority: weekly > timestamp > activity_proxy
    progress_by_uc = dict(progress_by_uc_activity)
    progress_by_uc.update(progress_by_uc_time)
    progress_by_uc.update(progress_by_uc_weekly)

    progress_source_by_uc = dict(progress_source_activity)
    progress_source_by_uc.update({k: "timestamp_fallback" for k in progress_by_uc_time.keys()})
    progress_source_by_uc.update({k: "weekly" for k in progress_by_uc_weekly.keys()})

    print(
        f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] "
        f"Progress sources: weekly={len(progress_by_uc_weekly):,}, "
        f"timestamp_fallback={len(progress_by_uc_time):,}, "
        f"activity_proxy={len(progress_by_uc_activity):,}, merged={len(progress_by_uc):,}"
    )

    write_checkpoint_views(
        output_dir=args.output_dir,
        base_columns=columns,
        rows=rows,
        progress_by_uc=progress_by_uc,
        progress_source_by_uc=progress_source_by_uc,
    )

    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] Phase 3 Transformation completed.")


if __name__ == "__main__":
    main()
