"""
Phase 2: Làm sạch dữ liệu
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
import re


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
Translate school names in user.json from Chinese to English-friendly text.

This script is designed to run before step_2_combine_data_streaming.py so the final
CSV school column is ASCII-friendly and easier to display across environments.

Workflow:
1) Read JSONL user data line-by-line (streaming, memory-safe)
2) Repair common mojibake in school names when possible
3) Translate by dictionary first, then suffix rules, then pinyin fallback
4) Write JSONL output for downstream combine script
"""



try:
    from pypinyin import lazy_pinyin
except ImportError:  # Optional dependency
    lazy_pinyin = None


CJK_RE = re.compile(r"[\u3400-\u4dbf\u4e00-\u9fff]")

# Frequent mojibake markers seen when UTF-8 text is decoded using latin-1/cp1252.
MOJIBAKE_MARKERS = tuple("ÃÂÐÑæåçéêëìíîïñóôõö÷øùúûüýþÿŽ¤")

DIRECT_TRANSLATIONS = {
    "清华大学": "Tsinghua University",
    "北京大学": "Peking University",
    "中国人民大学": "Renmin University of China",
    "北京航空航天大学": "Beihang University",
    "北京师范大学": "Beijing Normal University",
    "中国科学院大学": "University of Chinese Academy of Sciences",
    "中国科学技术大学": "University of Science and Technology of China",
    "上海交通大学": "Shanghai Jiao Tong University",
    "复旦大学": "Fudan University",
    "同济大学": "Tongji University",
    "浙江大学": "Zhejiang University",
    "南京大学": "Nanjing University",
    "武汉大学": "Wuhan University",
    "华中科技大学": "Huazhong University of Science and Technology",
    "华南理工大学": "South China University of Technology",
    "中山大学": "Sun Yat-sen University",
    "西安交通大学": "Xi'an Jiaotong University",
    "哈尔滨工业大学": "Harbin Institute of Technology",
    "四川大学": "Sichuan University",
    "山东大学": "Shandong University",
    "厦门大学": "Xiamen University",
    "南开大学": "Nankai University",
    "北京理工大学": "Beijing Institute of Technology",
    "北京邮电大学": "Beijing University of Posts and Telecommunications",
    "电子科技大学": "University of Electronic Science and Technology of China",
    "北京交通大学": "Beijing Jiaotong University",
    "清华学堂": "Tsinghua Academy",
}

SUFFIX_TRANSLATIONS = [
    ("医学院", "Medical College"),
    ("研究院", "Research Institute"),
    ("大学", "University"),
    ("学院", "College"),
    ("中学", "Middle School"),
    ("小学", "Primary School"),
    ("学堂", "Academy"),
    ("学校", "School"),
]


@dataclass
class TranslateStats:
    scanned: int = 0
    parsed_ok: int = 0
    parsed_error: int = 0
    empty_school: int = 0
    translated: int = 0
    direct_map_hits: int = 0
    suffix_rule_hits: int = 0
    pinyin_hits: int = 0
    mojibake_fixed: int = 0






def has_cjk(text: str) -> bool:
    return bool(CJK_RE.search(text))


def normalize_spaces(text: str) -> str:
    return " ".join(text.split())


def normalize_symbols(text: str) -> str:
    return (
        text.replace("（", "(")
        .replace("）", ")")
        .replace("，", ",")
        .replace("、", ",")
        .strip()
    )


def looks_like_mojibake(text: str) -> bool:
    if not text:
        return False
    if has_cjk(text):
        return False
    return any(ch in text for ch in MOJIBAKE_MARKERS)


def repair_mojibake(text: str) -> Tuple[str, bool]:
    if not looks_like_mojibake(text):
        return text, False

    for src_encoding in ("latin1", "cp1252"):
        try:
            repaired = text.encode(src_encoding).decode("utf-8")
        except UnicodeError:
            continue
        if has_cjk(repaired):
            return repaired, True
    return text, False


def to_pinyin_english(text: str) -> Optional[str]:
    if not text:
        return None

    if lazy_pinyin is None:
        return None

    parts = lazy_pinyin(text, errors="ignore")
    if not parts:
        return None

    # Keep output readable and title-cased.
    return normalize_spaces(" ".join(part.capitalize() for part in parts if part))


def translate_school(school_raw: str) -> Tuple[str, str, bool]:
    """
    Returns:
        translated_name, method, mojibake_fixed
    """
    text = normalize_symbols(school_raw)
    repaired, fixed = repair_mojibake(text)
    text = normalize_symbols(repaired)

    if not text:
        return "", "empty", fixed

    if text.isascii():
        return text, "ascii", fixed

    if text in DIRECT_TRANSLATIONS:
        return DIRECT_TRANSLATIONS[text], "direct", fixed

    for suffix_zh, suffix_en in SUFFIX_TRANSLATIONS:
        if text.endswith(suffix_zh):
            base = text[: -len(suffix_zh)].strip()
            if not base:
                break

            if base in DIRECT_TRANSLATIONS:
                return f"{DIRECT_TRANSLATIONS[base]} {suffix_en}", "suffix-direct", fixed

            base_pinyin = to_pinyin_english(base)
            if base_pinyin:
                return f"{base_pinyin} {suffix_en}", "suffix-pinyin", fixed

    fallback = to_pinyin_english(text)
    if fallback:
        return fallback, "pinyin", fixed

    # If pinyin is unavailable, keep normalized text as-is.
    return text, "unchanged", fixed


def run_translation(
    input_path: Path,
    output_path: Path,
    log_every: int = 200000,
    max_lines: Optional[int] = None,
    summary_path: Optional[Path] = None,
) -> None:
    """Stream user.json, translate each school name, write translated JSONL."""
    if not input_path.exists():
        raise FileNotFoundError(f"user input not found: {input_path}")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    stats = TranslateStats()
    method_counts: Dict[str, int] = {}

    log(f"Translating school names: {input_path} -> {output_path}")

    with input_path.open("r", encoding="utf-8", errors="ignore") as fin, \
         output_path.open("w", encoding="utf-8") as fout:

        for line_no, line in enumerate(fin, start=1):
            if max_lines is not None and line_no > max_lines:
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
            school_raw = record.get("school") or ""

            if not school_raw or not str(school_raw).strip():
                stats.empty_school += 1
                fout.write(json.dumps(record, ensure_ascii=False) + "\n")
                continue

            translated, method, fixed = translate_school(str(school_raw))
            record["school"] = translated
            if fixed:
                stats.mojibake_fixed += 1
            if method == "direct":
                stats.direct_map_hits += 1
            elif method.startswith("suffix"):
                stats.suffix_rule_hits += 1
            elif method == "pinyin":
                stats.pinyin_hits += 1
            stats.translated += 1
            method_counts[method] = method_counts.get(method, 0) + 1

            fout.write(json.dumps(record, ensure_ascii=False) + "\n")

            if stats.scanned % log_every == 0:
                log(f"Progress: scanned={stats.scanned:,} translated={stats.translated:,}")

    log(
        f"Translation done: scanned={stats.scanned:,} ok={stats.parsed_ok:,} "
        f"errors={stats.parsed_error:,} translated={stats.translated:,} "
        f"mojibake_fixed={stats.mojibake_fixed:,}"
    )

    if summary_path is not None:
        summary_path.parent.mkdir(parents=True, exist_ok=True)
        with summary_path.open("w", encoding="utf-8") as sf:
            sf.write("Phase 2 - School Name Translation Summary\n")
            sf.write("=" * 60 + "\n")
            sf.write(f"Input               : {input_path}\n")
            sf.write(f"Output              : {output_path}\n")
            sf.write(f"Generated at        : {time.strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            sf.write(f"Lines scanned       : {stats.scanned:,}\n")
            sf.write(f"Parsed OK           : {stats.parsed_ok:,}\n")
            sf.write(f"Parse errors        : {stats.parsed_error:,}\n")
            sf.write(f"Empty school        : {stats.empty_school:,}\n")
            sf.write(f"Translated          : {stats.translated:,}\n")
            sf.write(f"Mojibake fixed      : {stats.mojibake_fixed:,}\n\n")
            sf.write("Method breakdown:\n")
            for method, count in sorted(method_counts.items(), key=lambda x: -x[1]):
                sf.write(f"  {method:<20}: {count:,}\n")
        log(f"Summary written to: {summary_path}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Phase 2: Data Cleaning (school name translation).")
    parser.add_argument("--dataset-dir", type=Path, default=Path("D:/MOOCCubeX_dataset"))
    parser.add_argument("--output-dir", type=Path, default=Path("results"))
    parser.add_argument("--user-input", type=Path, default=Path("user.json"))
    parser.add_argument("--translated-user", type=Path, default=Path("user_school_en.json"))
    parser.add_argument("--translate-summary", type=Path, default=Path("school_translate_summary.txt"))
    parser.add_argument("--log-every", type=int, default=200000)
    parser.add_argument("--max-rows", type=int, default=None)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    project_root = Path(__file__).resolve().parents[1]
    dataset_dir = resolve_path_arg(args.dataset_dir, project_root, project_root)
    output_dir = resolve_path_arg(args.output_dir, project_root, project_root)

    user_input = resolve_path_arg(args.user_input, project_root, dataset_dir)
    translated_user = resolve_path_arg(args.translated_user, project_root, dataset_dir)
    translate_summary = resolve_path_arg(args.translate_summary, project_root, output_dir)

    try:
        started = time.time()
        log("Starting Phase 2: Data Cleaning")

        if not user_input.exists():
            log(f"WARNING: user input not found at {user_input}. Phase 2 skipped.")
            return 0

        run_translation(
            input_path=user_input,
            output_path=translated_user,
            log_every=max(1, args.log_every),
            max_lines=args.max_rows,
            summary_path=translate_summary,
        )
        log(f"Phase 2 completed in {time.time() - started:.2f}s")
        return 0
    except Exception as exc:
        log(f"FAILED: {exc}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
