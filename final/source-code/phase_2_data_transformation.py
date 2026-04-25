"""
Phase 2: Chuyển đổi dữ liệu
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import time
import csv
import json
import sqlite3
from datetime import datetime, timezone
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import *


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

"""
Streaming combiner for large MOOCCubeX JSONL datasets.

Design goals:
- Avoid loading full files into memory.
- Process line-by-line with bounded in-memory buffers.
- Persist user aggregates in SQLite to keep RAM stable.

Input files (JSONL):
- user.json or user_school_en.json
- user-problem.json
- user-video.json
- reply.json
- comment.json

Output:
- results/combined_user_metrics.csv
- results/step2_user_week_activity.csv
- results/combine_summary.txt
"""




@dataclass
class CombineConfig:
    project_root: Path
    dataset_dir: Path
    output_dir: Path
    db_path: Path
    output_csv: Path
    output_weekly_csv: Path
    user_file: Path
    min_courses: int = 5
    commit_every: int = 5000
    flush_every: int = 10000
    weekly_flush_every: int = 20000
    log_every: int = 200000
    max_lines_per_file: Optional[int] = None
    keep_db: bool = False
    cutoff_week: Optional[int] = None


@dataclass
class FileStats:
    name: str
    scanned: int = 0
    parsed_ok: int = 0
    parsed_error: int = 0
    matched_users: int = 0


@dataclass
class AggregateDelta:
    problem_total: int = 0
    problem_correct: int = 0
    attempts_sum: float = 0.0
    score_sum: float = 0.0
    score_count: int = 0
    video_sessions: int = 0
    video_count: int = 0
    segment_count: int = 0
    watched_seconds: float = 0.0
    speed_sum: float = 0.0
    speed_count: int = 0
    reply_count: int = 0
    comment_count: int = 0
    first_activity_time: Optional[str] = None
    last_activity_time: Optional[str] = None

    def update_time(self, ts: Optional[str]) -> None:
        if not ts:
            return
        if self.first_activity_time is None or ts < self.first_activity_time:
            self.first_activity_time = ts
        if self.last_activity_time is None or ts > self.last_activity_time:
            self.last_activity_time = ts


WEEKLY_ACTIVITY_COLUMNS = ("video", "problem", "reply", "comment")






def safe_float(value) -> Optional[float]:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def iso_year_week_key(dt: datetime) -> int:
    iso = dt.isocalendar()
    return int(iso.year * 100 + iso.week)


def parse_datetime_any(value: Optional[str]) -> Optional[datetime]:
    if not value:
        return None

    text = str(value).strip()
    if not text:
        return None

    try:
        return datetime.fromisoformat(text.replace("Z", "+00:00"))
    except ValueError:
        pass

    for fmt in ("%Y-%m-%d %H:%M:%S", "%Y-%m-%d"):
        try:
            return datetime.strptime(text, fmt)
        except ValueError:
            continue

    return None


def parse_datetime_from_unix(value) -> Optional[datetime]:
    if value is None:
        return None

    try:
        ts = float(value)
        if ts <= 0:
            return None
        return datetime.fromtimestamp(ts, tz=timezone.utc)
    except (TypeError, ValueError, OSError, OverflowError):
        return None


def parse_week_from_datetime(value: Optional[str]) -> Optional[int]:
    dt = parse_datetime_any(value)
    if dt is None:
        return None
    return iso_year_week_key(dt)


def parse_week_from_unix(value) -> Optional[int]:
    dt = parse_datetime_from_unix(value)
    if dt is None:
        return None
    return iso_year_week_key(dt)


def normalize_user_id(raw) -> Optional[str]:
    if raw is None:
        return None
    if isinstance(raw, int):
        return f"U_{raw}"

    text = str(raw).strip()
    if not text:
        return None
    if text.startswith("U_"):
        return text
    if text.isdigit():
        return f"U_{text}"
    return text




class StreamingCombiner:
    def __init__(self, cfg: CombineConfig):
        self.cfg = cfg
        self.conn: Optional[sqlite3.Connection] = None
        self.file_stats: Dict[str, FileStats] = {}

    def run(self) -> None:
        started_at = time.time()
        self.cfg.output_dir.mkdir(parents=True, exist_ok=True)

        self.conn = sqlite3.connect(self.cfg.db_path)
        self._configure_sqlite()
        self._create_tables()

        try:
            selected_users = self._load_selected_users()
            if selected_users == 0:
                raise RuntimeError("No users passed the filter. Nothing to combine.")

            log(f"Selected users: {selected_users:,}")

            self._process_problem_file()
            self._process_video_file()
            self._process_reply_file()
            self._process_comment_file()

            row_count = self._export_csv()
            weekly_row_count = self._export_weekly_activity_csv()
            self._write_summary(
                selected_users,
                row_count,
                weekly_row_count,
                time.time() - started_at,
            )

            log(
                f"Done. Wrote {row_count:,} rows to {self.cfg.output_csv} and "
                f"{weekly_row_count:,} rows to {self.cfg.output_weekly_csv}"
            )
        finally:
            if self.conn is not None:
                self.conn.close()
            if not self.cfg.keep_db and self.cfg.db_path.exists():
                self.cfg.db_path.unlink()

    def _configure_sqlite(self) -> None:
        assert self.conn is not None
        self.conn.execute("PRAGMA journal_mode=WAL;")
        self.conn.execute("PRAGMA synchronous=NORMAL;")
        self.conn.execute("PRAGMA temp_store=MEMORY;")
        self.conn.execute("PRAGMA cache_size=-200000;")

    def _create_tables(self) -> None:
        assert self.conn is not None
        self.conn.execute(
            """
            CREATE TABLE IF NOT EXISTS selected_users (
                user_id TEXT PRIMARY KEY,
                gender INTEGER,
                school TEXT,
                year_of_birth INTEGER,
                num_courses INTEGER
            )
            """
        )
        self.conn.execute(
            """
            CREATE TABLE IF NOT EXISTS user_agg (
                user_id TEXT PRIMARY KEY,
                problem_total INTEGER DEFAULT 0,
                problem_correct INTEGER DEFAULT 0,
                attempts_sum REAL DEFAULT 0,
                score_sum REAL DEFAULT 0,
                score_count INTEGER DEFAULT 0,
                video_sessions INTEGER DEFAULT 0,
                video_count INTEGER DEFAULT 0,
                segment_count INTEGER DEFAULT 0,
                watched_seconds REAL DEFAULT 0,
                speed_sum REAL DEFAULT 0,
                speed_count INTEGER DEFAULT 0,
                reply_count INTEGER DEFAULT 0,
                comment_count INTEGER DEFAULT 0,
                first_activity_time TEXT,
                last_activity_time TEXT
            )
            """
        )
        self.conn.execute(
            """
            CREATE TABLE IF NOT EXISTS user_week_activity (
                user_id TEXT NOT NULL,
                week INTEGER NOT NULL,
                video INTEGER DEFAULT 0,
                problem INTEGER DEFAULT 0,
                reply INTEGER DEFAULT 0,
                comment INTEGER DEFAULT 0,
                PRIMARY KEY (user_id, week)
            )
            """
        )
        self.conn.execute("CREATE INDEX IF NOT EXISTS idx_selected_users ON selected_users(user_id)")
        self.conn.execute("CREATE INDEX IF NOT EXISTS idx_user_week_activity_week ON user_week_activity(week)")
        self.conn.commit()

    def _load_selected_users(self) -> int:
        user_path = self.cfg.user_file
        if not user_path.exists():
            raise FileNotFoundError(f"Missing file: {user_path}")

        assert self.conn is not None
        stats = FileStats(name=user_path.name)
        self.file_stats[stats.name] = stats

        selected_batch = []
        agg_batch = []

        log(f"Step 1/5: Filtering users from {user_path}")

        for record in self._iter_jsonl(user_path, stats):
            user_id = normalize_user_id(record.get("id") or record.get("user_id"))
            if not user_id:
                continue

            course_order = record.get("course_order")
            num_courses = len(course_order) if isinstance(course_order, list) else 0
            if num_courses <= self.cfg.min_courses:
                continue

            selected_batch.append(
                (
                    user_id,
                    record.get("gender"),
                    record.get("school"),
                    record.get("year_of_birth"),
                    num_courses,
                )
            )
            agg_batch.append((user_id,))
            stats.matched_users += 1

            if len(selected_batch) >= self.cfg.commit_every:
                self._flush_selected_users(selected_batch, agg_batch)
                selected_batch.clear()
                agg_batch.clear()

            if stats.scanned % self.cfg.log_every == 0:
                log(
                    f"{self.cfg.user_file} progress: "
                    f"scanned={stats.scanned:,} selected={stats.matched_users:,}"
                )

        if selected_batch:
            self._flush_selected_users(selected_batch, agg_batch)

        selected_count = self.conn.execute("SELECT COUNT(*) FROM selected_users").fetchone()[0]
        self.conn.commit()
        log(
            f"{user_path.name} done: "
            f"scanned={stats.scanned:,} parsed_error={stats.parsed_error:,} selected={selected_count:,}"
        )
        return int(selected_count)

    def _flush_selected_users(self, selected_batch, agg_batch) -> None:
        assert self.conn is not None
        self.conn.executemany(
            """
            INSERT OR IGNORE INTO selected_users
            (user_id, gender, school, year_of_birth, num_courses)
            VALUES (?, ?, ?, ?, ?)
            """,
            selected_batch,
        )
        self.conn.executemany(
            "INSERT OR IGNORE INTO user_agg (user_id) VALUES (?)",
            agg_batch,
        )
        self.conn.commit()

    def _build_selected_lookup(self) -> Callable[[str], bool]:
        assert self.conn is not None

        @lru_cache(maxsize=400000)
        def exists(user_id: str) -> bool:
            row = self.conn.execute(
                "SELECT 1 FROM selected_users WHERE user_id = ? LIMIT 1", (user_id,)
            ).fetchone()
            return row is not None

        return exists

    def _process_problem_file(self) -> None:
        path = self.cfg.dataset_dir / "user-problem.json"
        if not path.exists():
            log("user-problem.json not found, skip.")
            return

        log("Step 2/5: Streaming combine from user-problem.json")

        def apply(record, delta: AggregateDelta) -> None:
            if self.cfg.cutoff_week is not None:
                week = parse_week_from_datetime(record.get("submit_time"))
                if week is not None and week > self.cfg.cutoff_week:
                    return

            delta.problem_total += 1
            is_correct = record.get("is_correct")
            if is_correct in (1, True, "1"):
                delta.problem_correct += 1

            attempts = safe_float(record.get("attempts"))
            if attempts is not None:
                delta.attempts_sum += attempts

            score = safe_float(record.get("score"))
            if score is not None:
                delta.score_sum += score
                delta.score_count += 1

            delta.update_time(record.get("submit_time"))

        def extract_weeks(record: dict) -> Iterable[int]:
            week = parse_week_from_datetime(record.get("submit_time"))
            if week is None:
                return []
            return [week]

        self._process_event_file(
            path=path,
            stats_name="user-problem.json",
            get_user_id=lambda r: normalize_user_id(r.get("user_id")),
            apply_delta=apply,
            extract_weeks=extract_weeks,
            weekly_activity_col="problem",
        )

    def _process_video_file(self) -> None:
        path = self.cfg.dataset_dir / "user-video.json"
        if not path.exists():
            log("user-video.json not found, skip.")
            return

        log("Step 3/5: Streaming combine from user-video.json")

        def apply(record, delta: AggregateDelta) -> None:
            seq = record.get("seq")
            if not isinstance(seq, list):
                return

            delta.video_sessions += 1
            delta.video_count += len(seq)

            for item in seq:
                segments = item.get("segment") if isinstance(item, dict) else None
                if not isinstance(segments, list):
                    continue
                for seg in segments:
                    if not isinstance(seg, dict): continue

                    if self.cfg.cutoff_week is not None:
                        week = parse_week_from_unix(seg.get("local_start_time"))
                        if week is not None and week > self.cfg.cutoff_week:
                            continue

                    delta.segment_count += 1

                    local_start_dt = parse_datetime_from_unix(seg.get("local_start_time"))

                    if local_start_dt is not None:
                        delta.update_time(local_start_dt.isoformat())

                    start = safe_float(seg.get("start_point"))
                    end = safe_float(seg.get("end_point"))
                    if start is not None and end is not None and end >= start:
                        delta.watched_seconds += end - start

                    speed = safe_float(seg.get("speed"))
                    if speed is not None and speed > 0:
                        delta.speed_sum += speed
                        delta.speed_count += 1

        def extract_weeks(record: dict) -> Iterable[int]:
            seq = record.get("seq")
            if not isinstance(seq, list):
                return []

            weeks: Set[int] = set()
            for item in seq:
                segments = item.get("segment") if isinstance(item, dict) else None
                if not isinstance(segments, list):
                    continue
                for seg in segments:
                    if not isinstance(seg, dict):
                        continue
                    week = parse_week_from_unix(seg.get("local_start_time"))
                    if week is not None:
                        weeks.add(week)

            return weeks

        self._process_event_file(
            path=path,
            stats_name="user-video.json",
            get_user_id=lambda r: normalize_user_id(r.get("user_id")),
            apply_delta=apply,
            extract_weeks=extract_weeks,
            weekly_activity_col="video",
        )

    def _process_reply_file(self) -> None:
        path = self.cfg.dataset_dir / "reply.json"
        if not path.exists():
            log("reply.json not found, skip.")
            return

        log("Step 4/5: Streaming combine from reply.json")

        def apply(record, delta: AggregateDelta) -> None:
            if self.cfg.cutoff_week is not None:
                week = parse_week_from_datetime(record.get("create_time"))
                if week is not None and week > self.cfg.cutoff_week:
                    return

            delta.reply_count += 1
            delta.update_time(record.get("create_time"))

        def extract_weeks(record: dict) -> Iterable[int]:
            week = parse_week_from_datetime(record.get("create_time"))
            if week is None:
                return []
            return [week]

        self._process_event_file(
            path=path,
            stats_name="reply.json",
            get_user_id=lambda r: normalize_user_id(r.get("user_id")),
            apply_delta=apply,
            extract_weeks=extract_weeks,
            weekly_activity_col="reply",
        )

    def _process_comment_file(self) -> None:
        path = self.cfg.dataset_dir / "comment.json"
        if not path.exists():
            log("comment.json not found, skip.")
            return

        log("Step 5/5: Streaming combine from comment.json")

        def apply(record, delta: AggregateDelta) -> None:
            if self.cfg.cutoff_week is not None:
                week = parse_week_from_datetime(record.get("create_time"))
                if week is not None and week > self.cfg.cutoff_week:
                    return

            delta.comment_count += 1
            delta.update_time(record.get("create_time"))

        def extract_weeks(record: dict) -> Iterable[int]:
            week = parse_week_from_datetime(record.get("create_time"))
            if week is None:
                return []
            return [week]

        self._process_event_file(
            path=path,
            stats_name="comment.json",
            get_user_id=lambda r: normalize_user_id(r.get("user_id")),
            apply_delta=apply,
            extract_weeks=extract_weeks,
            weekly_activity_col="comment",
        )

    def _process_event_file(
        self,
        path: Path,
        stats_name: str,
        get_user_id: Callable[[dict], Optional[str]],
        apply_delta: Callable[[dict, AggregateDelta], None],
        extract_weeks: Optional[Callable[[dict], Iterable[int]]] = None,
        weekly_activity_col: Optional[str] = None,
    ) -> None:
        assert self.conn is not None
        stats = FileStats(name=stats_name)
        self.file_stats[stats.name] = stats

        selected_lookup = self._build_selected_lookup()
        deltas: Dict[str, AggregateDelta] = {}
        week_flags: Dict[Tuple[str, int], int] = {}

        for record in self._iter_jsonl(path, stats):
            user_id = get_user_id(record)
            if not user_id:
                continue
            if not selected_lookup(user_id):
                continue

            stats.matched_users += 1
            delta = deltas.get(user_id)
            if delta is None:
                delta = AggregateDelta()
                deltas[user_id] = delta

            apply_delta(record, delta)

            if extract_weeks is not None and weekly_activity_col is not None:
                for week in extract_weeks(record):
                    if week is None:
                        continue
                    week_flags[(user_id, int(week))] = 1

            if len(deltas) >= self.cfg.flush_every:
                self._flush_deltas(deltas)
                deltas.clear()

            if len(week_flags) >= self.cfg.weekly_flush_every:
                self._flush_week_activity_flags(week_flags, weekly_activity_col)
                week_flags.clear()

            if stats.scanned % self.cfg.log_every == 0:
                cache_info = selected_lookup.cache_info()
                log(
                    f"{stats_name} progress: scanned={stats.scanned:,} "
                    f"matched={stats.matched_users:,} cache={cache_info.currsize:,}"
                )

        if deltas:
            self._flush_deltas(deltas)
            deltas.clear()

        if week_flags:
            self._flush_week_activity_flags(week_flags, weekly_activity_col)
            week_flags.clear()

        self.conn.commit()
        log(
            f"{stats_name} done: scanned={stats.scanned:,} "
            f"parsed_error={stats.parsed_error:,} matched={stats.matched_users:,}"
        )

    def _flush_deltas(self, deltas: Dict[str, AggregateDelta]) -> None:
        assert self.conn is not None

        rows = []
        for user_id, d in deltas.items():
            rows.append(
                (
                    user_id,
                    d.problem_total,
                    d.problem_correct,
                    d.attempts_sum,
                    d.score_sum,
                    d.score_count,
                    d.video_sessions,
                    d.video_count,
                    d.segment_count,
                    d.watched_seconds,
                    d.speed_sum,
                    d.speed_count,
                    d.reply_count,
                    d.comment_count,
                    d.first_activity_time,
                    d.last_activity_time,
                )
            )

        self.conn.executemany(
            """
            INSERT INTO user_agg (
                user_id,
                problem_total,
                problem_correct,
                attempts_sum,
                score_sum,
                score_count,
                video_sessions,
                video_count,
                segment_count,
                watched_seconds,
                speed_sum,
                speed_count,
                reply_count,
                comment_count,
                first_activity_time,
                last_activity_time
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(user_id) DO UPDATE SET
                problem_total = user_agg.problem_total + excluded.problem_total,
                problem_correct = user_agg.problem_correct + excluded.problem_correct,
                attempts_sum = user_agg.attempts_sum + excluded.attempts_sum,
                score_sum = user_agg.score_sum + excluded.score_sum,
                score_count = user_agg.score_count + excluded.score_count,
                video_sessions = user_agg.video_sessions + excluded.video_sessions,
                video_count = user_agg.video_count + excluded.video_count,
                segment_count = user_agg.segment_count + excluded.segment_count,
                watched_seconds = user_agg.watched_seconds + excluded.watched_seconds,
                speed_sum = user_agg.speed_sum + excluded.speed_sum,
                speed_count = user_agg.speed_count + excluded.speed_count,
                reply_count = user_agg.reply_count + excluded.reply_count,
                comment_count = user_agg.comment_count + excluded.comment_count,
                first_activity_time = CASE
                    WHEN user_agg.first_activity_time IS NULL THEN excluded.first_activity_time
                    WHEN excluded.first_activity_time IS NULL THEN user_agg.first_activity_time
                    WHEN excluded.first_activity_time < user_agg.first_activity_time
                        THEN excluded.first_activity_time
                    ELSE user_agg.first_activity_time
                END,
                last_activity_time = CASE
                    WHEN user_agg.last_activity_time IS NULL THEN excluded.last_activity_time
                    WHEN excluded.last_activity_time IS NULL THEN user_agg.last_activity_time
                    WHEN excluded.last_activity_time > user_agg.last_activity_time
                        THEN excluded.last_activity_time
                    ELSE user_agg.last_activity_time
                END
            """,
            rows,
        )
        self.conn.commit()

    def _flush_week_activity_flags(
        self,
        week_flags: Dict[Tuple[str, int], int],
        activity_col: Optional[str],
    ) -> None:
        assert self.conn is not None
        if not week_flags or activity_col not in WEEKLY_ACTIVITY_COLUMNS:
            return

        rows = [(user_id, week) for user_id, week in week_flags.keys()]
        sql = f"""
            INSERT INTO user_week_activity (user_id, week, {activity_col})
            VALUES (?, ?, 1)
            ON CONFLICT(user_id, week) DO UPDATE SET
                {activity_col} = 1
        """
        self.conn.executemany(sql, rows)
        self.conn.commit()

    def _iter_jsonl(self, path: Path, stats: FileStats) -> Iterable[dict]:
        with path.open("r", encoding="utf-8", errors="ignore") as f:
            for line_no, line in enumerate(f, start=1):
                if self.cfg.max_lines_per_file is not None and line_no > self.cfg.max_lines_per_file:
                    break

                stats.scanned += 1
                line = line.strip()
                if not line:
                    continue

                try:
                    record = json.loads(line)
                except json.JSONDecodeError:
                    stats.parsed_error += 1
                    continue

                stats.parsed_ok += 1
                yield record

    def _export_csv(self) -> int:
        assert self.conn is not None
        log(f"Exporting combined CSV to: {self.cfg.output_csv}")

        query = """
            SELECT
                u.user_id,
                u.gender,
                u.school,
                u.year_of_birth,
                u.num_courses,
                a.problem_total,
                a.problem_correct,
                a.attempts_sum,
                a.score_sum,
                a.score_count,
                a.video_sessions,
                a.video_count,
                a.segment_count,
                a.watched_seconds,
                a.speed_sum,
                a.speed_count,
                a.reply_count,
                a.comment_count,
                a.first_activity_time,
                a.last_activity_time
            FROM selected_users u
            JOIN user_agg a ON a.user_id = u.user_id
            ORDER BY u.user_id
        """

        headers = [
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

        row_count = 0
        with self.cfg.output_csv.open("w", encoding="utf-8", newline="") as out_f:
            writer = csv.writer(out_f)
            writer.writerow(headers)

            cur = self.conn.execute(query)
            while True:
                rows = cur.fetchmany(10000)
                if not rows:
                    break

                for row in rows:
                    (
                        user_id,
                        gender,
                        school,
                        year_of_birth,
                        num_courses,
                        problem_total,
                        problem_correct,
                        attempts_sum,
                        score_sum,
                        score_count,
                        video_sessions,
                        video_count,
                        segment_count,
                        watched_seconds,
                        speed_sum,
                        speed_count,
                        reply_count,
                        comment_count,
                        first_activity_time,
                        last_activity_time,
                    ) = row

                    problem_total = int(problem_total or 0)
                    problem_correct = int(problem_correct or 0)
                    attempts_sum = float(attempts_sum or 0.0)
                    score_sum = float(score_sum or 0.0)
                    score_count = int(score_count or 0)
                    watched_seconds = float(watched_seconds or 0.0)
                    speed_sum = float(speed_sum or 0.0)
                    speed_count = int(speed_count or 0)
                    reply_count = int(reply_count or 0)
                    comment_count = int(comment_count or 0)
                    video_count = int(video_count or 0)

                    problem_accuracy = (
                        (problem_correct / problem_total) if problem_total > 0 else 0.0
                    )
                    avg_attempts = (attempts_sum / problem_total) if problem_total > 0 else 0.0
                    avg_score = (score_sum / score_count) if score_count > 0 else 0.0
                    avg_speed = (speed_sum / speed_count) if speed_count > 0 else 0.0
                    watched_hours = watched_seconds / 3600.0
                    forum_total = reply_count + comment_count
                    engagement_events = problem_total + video_count + forum_total

                    writer.writerow(
                        [
                            user_id,
                            gender,
                            school,
                            year_of_birth,
                            num_courses,
                            problem_total,
                            problem_correct,
                            round(problem_accuracy, 6),
                            round(avg_attempts, 6),
                            round(avg_score, 6),
                            int(video_sessions or 0),
                            video_count,
                            int(segment_count or 0),
                            round(watched_seconds, 3),
                            round(watched_hours, 6),
                            round(avg_speed, 6),
                            reply_count,
                            comment_count,
                            forum_total,
                            engagement_events,
                            first_activity_time,
                            last_activity_time,
                        ]
                    )
                    row_count += 1

        return row_count

    def _export_weekly_activity_csv(self) -> int:
        assert self.conn is not None
        log(f"Exporting user-week activity CSV to: {self.cfg.output_weekly_csv}")

        query = """
            SELECT
                user_id,
                week,
                video,
                problem,
                reply,
                comment
            FROM user_week_activity
            ORDER BY user_id, week
        """

        row_count = 0
        with self.cfg.output_weekly_csv.open("w", encoding="utf-8", newline="") as out_f:
            writer = csv.writer(out_f)
            writer.writerow(["user_id", "week", "video", "problem", "reply", "comment"])

            cur = self.conn.execute(query)
            while True:
                rows = cur.fetchmany(10000)
                if not rows:
                    break

                for user_id, week, video, problem, reply, comment in rows:
                    writer.writerow(
                        [
                            user_id,
                            int(week),
                            int(video or 0),
                            int(problem or 0),
                            int(reply or 0),
                            int(comment or 0),
                        ]
                    )
                    row_count += 1

        return row_count

    def _write_summary(
        self,
        selected_users: int,
        output_rows: int,
        weekly_rows: int,
        elapsed_sec: float,
    ) -> None:
        summary_path = self.cfg.output_dir / "combine_summary.txt"
        with summary_path.open("w", encoding="utf-8") as f:
            f.write("Streaming Combine Summary\n")
            f.write("=" * 80 + "\n")
            f.write(f"Started at            : {now_text()}\n")
            f.write(f"Project root          : {self.cfg.project_root}\n")
            f.write(f"Dataset dir           : {self.cfg.dataset_dir}\n")
            f.write(f"User source file      : {self.cfg.user_file}\n")
            f.write(f"Output CSV            : {self.cfg.output_csv}\n")
            f.write(f"Output weekly CSV     : {self.cfg.output_weekly_csv}\n")
            f.write(f"Min courses filter    : > {self.cfg.min_courses}\n")
            f.write(f"Selected users        : {selected_users:,}\n")
            f.write(f"Output rows           : {output_rows:,}\n")
            f.write(f"Weekly rows           : {weekly_rows:,}\n")
            f.write(f"Elapsed (seconds)     : {elapsed_sec:.2f}\n")
            f.write("\nFile-level stats:\n")

            for name, stats in self.file_stats.items():
                f.write(
                    f"- {name}: scanned={stats.scanned:,}, parsed_ok={stats.parsed_ok:,}, "
                    f"parsed_error={stats.parsed_error:,}, matched={stats.matched_users:,}\n"
                )


def process_parquet(parquet_path: Path, output_csv: Path, output_weekly_csv: Path, cutoff_week: Optional[int]) -> None:
    import pandas as pd
    import numpy as np

    log(f"Reading parquet file: {parquet_path} with cutoff_week={cutoff_week}")
    df = pd.read_parquet(parquet_path)

    if "user_id" not in df.columns:
        raise RuntimeError("Parquet input missing required column: user_id")

    df["user_id"] = df["user_id"].map(normalize_user_id)
    df = df[df["user_id"].notna()].copy()

    if "num_courses" not in df.columns:
        df["num_courses"] = 0
    df["num_courses"] = pd.to_numeric(df["num_courses"], errors="coerce").fillna(0)
    # Keep behavior aligned with JSON streaming path: only users with >5 courses.
    df = df[df["num_courses"] > 5].copy()

    log("Preparing event time and week columns for parquet...")
    for col in ["is_correct", "attempts", "score"]:
        if col not in df.columns:
            df[col] = 0
    df["is_correct"] = pd.to_numeric(df["is_correct"], errors="coerce").fillna(0)
    df["attempts"] = pd.to_numeric(df["attempts"], errors="coerce").fillna(0)
    df["score"] = pd.to_numeric(df["score"], errors="coerce")

    video_count_col = "log_id" if "log_id" in df.columns else ("seq" if "seq" in df.columns else None)
    problem_count_col = "problem_id" if "problem_id" in df.columns else None

    if "reply_id" in df.columns:
        reply_count_col = "reply_id"
    elif "id_x" in df.columns:
        reply_count_col = "id_x"
    else:
        reply_count_col = None

    if "comment_id" in df.columns:
        comment_count_col = "comment_id"
    elif "id_y" in df.columns:
        comment_count_col = "id_y"
    else:
        comment_count_col = None

    named_aggs: Dict[str, Tuple[str, str]] = {
        "num_courses": ("num_courses", "max"),
        "problem_correct": ("is_correct", "sum"),
        "attempts_sum": ("attempts", "sum"),
        "score_sum": ("score", "sum"),
        "score_count": ("score", "count"),
    }
    if "gender" in df.columns:
        named_aggs["gender"] = ("gender", "first")
    if "school" in df.columns:
        named_aggs["school"] = ("school", "first")
    if "year_of_birth" in df.columns:
        named_aggs["year_of_birth"] = ("year_of_birth", "first")
    if problem_count_col is not None:
        named_aggs["problem_total"] = (problem_count_col, "count")
    if video_count_col is not None:
        named_aggs["video_count"] = (video_count_col, "count")
    if reply_count_col is not None:
        named_aggs["reply_count"] = (reply_count_col, "count")
    if comment_count_col is not None:
        named_aggs["comment_count"] = (comment_count_col, "count")

    user_agg = df.groupby("user_id").agg(**named_aggs).reset_index()

    for col in ["problem_total", "video_count", "reply_count", "comment_count"]:
        if col not in user_agg.columns:
            user_agg[col] = 0

    event_time_parts = []
    if "submit_time" in df.columns:
        event_time_parts.append(pd.to_datetime(df["submit_time"], errors="coerce", utc=True))
    if "create_time" in df.columns:
        event_time_parts.append(pd.to_datetime(df["create_time"], errors="coerce", utc=True))
    if "local_start_time" in df.columns:
        local_dt = pd.to_datetime(
            pd.to_numeric(df["local_start_time"], errors="coerce"),
            errors="coerce",
            utc=True,
            unit="s",
        )
        event_time_parts.append(local_dt)

    if event_time_parts:
        df["_event_time"] = pd.concat(event_time_parts, axis=1).bfill(axis=1).iloc[:, 0]
    else:
        df["_event_time"] = pd.NaT

    df_for_agg = df
    if cutoff_week is not None and "_event_time" in df.columns:
        df_with_time = df[df["_event_time"].notna()].copy()
        iso = df_with_time["_event_time"].dt.isocalendar()
        df_with_time["_week"] = (iso["year"].astype(int) * 100 + iso["week"].astype(int))
        df_for_agg = df_with_time[df_with_time["_week"] <= cutoff_week]

    log(f"Aggregating user metrics from {len(df_for_agg):,} parquet events...")

    video_count_col = "log_id" if "log_id" in df_for_agg.columns else ("seq" if "seq" in df_for_agg.columns else None)
    problem_count_col = "problem_id" if "problem_id" in df_for_agg.columns else None

    reply_count_col = "reply_id" if "reply_id" in df_for_agg.columns else ("id_x" if "id_x" in df_for_agg.columns else None)
    comment_count_col = "comment_id" if "comment_id" in df_for_agg.columns else ("id_y" if "id_y" in df_for_agg.columns else None)

    named_aggs: Dict[str, Tuple[str, str]] = {
        "num_courses": ("num_courses", "max"),
        "problem_correct": ("is_correct", "sum"),
        "attempts_sum": ("attempts", "sum"),
        "score_sum": ("score", "sum"),
        "score_count": ("score", "count"),
    }
    if "gender" in df_for_agg.columns:
        named_aggs["gender"] = ("gender", "first")
    if "school" in df_for_agg.columns:
        named_aggs["school"] = ("school", "first")
    if "year_of_birth" in df_for_agg.columns:
        named_aggs["year_of_birth"] = ("year_of_birth", "first")
    if problem_count_col is not None:
        named_aggs["problem_total"] = (problem_count_col, "count")
    if video_count_col is not None:
        named_aggs["video_count"] = (video_count_col, "count")
    if reply_count_col is not None:
        named_aggs["reply_count"] = (reply_count_col, "count")
    if comment_count_col is not None:
        named_aggs["comment_count"] = (comment_count_col, "count")

    user_agg = df_for_agg.groupby("user_id").agg(**named_aggs).reset_index()

    for col in ["problem_total", "video_count", "reply_count", "comment_count"]:
        if col not in user_agg.columns:
            user_agg[col] = 0

    if "_event_time" in df_for_agg.columns:
        time_agg = (
            df_for_agg[df_for_agg["_event_time"].notna()].groupby("user_id")["_event_time"]
            .agg(first_activity_time="min", last_activity_time="max")
            .reset_index()
        )
        user_agg = user_agg.merge(time_agg, on="user_id", how="left")
        user_agg["first_activity_time"] = user_agg["first_activity_time"].dt.strftime("%Y-%m-%d %H:%M:%S")
        user_agg["last_activity_time"] = user_agg["last_activity_time"].dt.strftime("%Y-%m-%d %H:%M:%S")
    else:
        user_agg["first_activity_time"] = ""
        user_agg["last_activity_time"] = ""
    user_agg["avg_attempts"] = np.where(
        user_agg["problem_total"] > 0,
        user_agg["attempts_sum"] / user_agg["problem_total"],
        0,
    )
    user_agg["avg_score"] = np.where(user_agg["score_count"] > 0, user_agg["score_sum"] / user_agg["score_count"], 0)

    user_agg["problem_accuracy"] = np.where(
        user_agg["problem_total"] > 0,
        user_agg["problem_correct"] / user_agg["problem_total"],
        0,
    )
    # Limited signal in compact parquet, keep conservative proxy fields for compatibility.
    user_agg["video_sessions"] = user_agg["video_count"]
    user_agg["segment_count"] = user_agg["video_count"] * 3
    user_agg["watched_seconds"] = user_agg["video_count"] * 60
    user_agg["watched_hours"] = user_agg["watched_seconds"] / 3600
    user_agg["avg_speed"] = 1.0

    user_agg["forum_total"] = user_agg["comment_count"] + user_agg["reply_count"]
    user_agg["engagement_events"] = user_agg["problem_total"] + user_agg["video_sessions"] + user_agg["forum_total"]

    headers = [
        "user_id", "gender", "school", "year_of_birth", "num_courses",
        "problem_total", "problem_correct", "problem_accuracy", "avg_attempts",
        "avg_score", "video_sessions", "video_count", "segment_count",
        "watched_seconds", "watched_hours", "avg_speed", "reply_count",
        "comment_count", "forum_total", "engagement_events",
        "first_activity_time", "last_activity_time"
    ]

    for col in headers:
        if col not in user_agg.columns:
            user_agg[col] = 0

    user_agg = user_agg[headers]

    log(f"Saving {len(user_agg)} rows to {output_csv}")
    user_agg.to_csv(output_csv, index=False)

    log("Generating weekly user activity from full parquet data...")
    weekly_cols = ["user_id", "week", "video", "problem", "reply", "comment"]
    if "_event_time" in df.columns:
        weekly_df = df[df["_event_time"].notna()][["user_id", "_event_time"]].copy()
        iso = weekly_df["_event_time"].dt.isocalendar()
        weekly_df["week"] = (iso["year"].astype(int) * 100 + iso["week"].astype(int)).astype(int)

        weekly_df["video"] = (
            df.loc[weekly_df.index, video_count_col].notna().astype(int)
            if video_count_col is not None
            else 0
        )
        weekly_df["problem"] = (
            df.loc[weekly_df.index, problem_count_col].notna().astype(int)
            if problem_count_col is not None
            else 0
        )
        weekly_df["reply"] = (
            df.loc[weekly_df.index, reply_count_col].notna().astype(int)
            if reply_count_col is not None
            else 0
        )
        weekly_df["comment"] = (
            df.loc[weekly_df.index, comment_count_col].notna().astype(int)
            if comment_count_col is not None
            else 0
        )

        weekly = (
            weekly_df.groupby(["user_id", "week"], as_index=False)[["video", "problem", "reply", "comment"]]
            .max()
            .sort_values(["user_id", "week"])
        )
    else:
        weekly = pd.DataFrame(columns=weekly_cols)

    weekly.to_csv(output_weekly_csv, index=False)
    log("Parquet processing complete.")

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Phase 2: Data Transformation.")
    parser.add_argument("--dataset-dir", type=Path, default=Path("D:/MOOCCubeX_dataset"))
    parser.add_argument("--output-dir", type=Path, default=Path("experiment/results"))
    parser.add_argument("--translated-user", type=Path, default=Path("user_school_en.json"))
    parser.add_argument("--combined-file", type=Path, default=Path("combined_user_metrics.csv"))
    parser.add_argument("--weekly-file", type=Path, default=Path("step2_user_week_activity.csv"))
    parser.add_argument("--db-file", type=Path, default=Path("combined_streaming.sqlite3"))
    parser.add_argument("--cutoff-week", type=int, default=None, help="Cutoff week for feature extraction (e.g., 202004 for week 4 of 2020)")
    parser.add_argument("--log-every", type=int, default=200000)
    parser.add_argument("--max-rows", type=int, default=None)
    return parser

def main() -> int:
    args = build_parser().parse_args()
    project_root = Path(__file__).resolve().parents[1]
    dataset_dir = resolve_path_arg(args.dataset_dir, project_root, project_root)
    output_dir = resolve_path_arg(args.output_dir, project_root, project_root)
    
    translated_user = resolve_path_arg(args.translated_user, project_root, dataset_dir)
    combined_file = resolve_path_arg(args.combined_file, project_root, output_dir)
    weekly_file = resolve_path_arg(args.weekly_file, project_root, output_dir)
    db_file = resolve_path_arg(args.db_file, project_root, output_dir)
    
    try:
        started = time.time()
        log("Starting Phase 2: Data Transformation")
        
        cfg = CombineConfig(
            project_root=project_root,
            dataset_dir=dataset_dir,
            output_dir=output_dir,
            db_path=db_file,
            output_csv=combined_file,
            output_weekly_csv=weekly_file,
            user_file=translated_user,
            min_courses=5,
            commit_every=5000,
            flush_every=10000,
            weekly_flush_every=20000,
            log_every=max(1, args.log_every),
            max_lines_per_file=args.max_rows,
            keep_db=False,
            cutoff_week=args.cutoff_week
        )
        combiner = StreamingCombiner(cfg)
        combiner.run()
        
        log(f"Phase 2 completed in {time.time() - started:.2f}s")
        return 0
    except Exception as exc:
        log(f"FAILED: {exc}")
        return 1

if __name__ == "__main__":
    raise SystemExit(main())
