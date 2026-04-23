#!/usr/bin/env python3
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

from __future__ import annotations

import argparse
import json
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple

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


def now_text() -> str:
    return time.strftime("%Y-%m-%d %H:%M:%S")


def log(message: str) -> None:
    print(f"[{now_text()}] {message}")


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


def resolve_path_arg(path_value: Path, project_root: Path, default_base: Path) -> Path:
    """Resolve relative CLI path against project root with sensible defaults."""
    if path_value.is_absolute():
        return path_value.resolve()

    if path_value.parent == Path("."):
        return (default_base / path_value).resolve()

    return (project_root / path_value).resolve()


def build_parser() -> argparse.ArgumentParser:
    base_dir = Path(__file__).resolve().parents[1]
    default_input = base_dir / "dataset" / "user.json"
    default_output = base_dir / "dataset" / "user_school_en.json"

    parser = argparse.ArgumentParser(
        description="Translate user school names (Chinese -> English-friendly text)."
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=default_input,
        help=f"Input JSONL file (default: {default_input})",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=default_output,
        help=f"Output JSONL file (default: {default_output})",
    )
    parser.add_argument(
        "--log-every",
        type=int,
        default=200000,
        help="Progress logging interval by lines (default: 200000)",
    )
    parser.add_argument(
        "--max-lines",
        type=int,
        default=None,
        help="Dry-run helper: process only first N lines",
    )
    parser.add_argument(
        "--summary",
        type=Path,
        default=None,
        help="Optional summary output path",
    )
    return parser


def run_translation(
    input_path: Path,
    output_path: Path,
    log_every: int,
    max_lines: Optional[int],
    summary_path: Optional[Path],
) -> int:
    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    stats = TranslateStats()

    log(f"Translating school names: {input_path} -> {output_path}")
    if lazy_pinyin is None:
        log("Warning: pypinyin is not installed; unknown schools may remain in original form.")

    with input_path.open("r", encoding="utf-8", errors="ignore") as src, output_path.open(
        "w", encoding="utf-8", newline=""
    ) as dst:
        for line_no, line in enumerate(src, start=1):
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

            school = record.get("school")
            if not isinstance(school, str) or not school.strip():
                stats.empty_school += 1
            else:
                translated, method, fixed = translate_school(school)

                if fixed:
                    stats.mojibake_fixed += 1

                if method == "direct":
                    stats.direct_map_hits += 1
                elif method in ("suffix-direct", "suffix-pinyin"):
                    stats.suffix_rule_hits += 1
                elif method == "pinyin":
                    stats.pinyin_hits += 1

                if translated and translated != school:
                    record["school_zh"] = school
                    record["school"] = translated
                    record["school_en"] = translated
                    stats.translated += 1

            dst.write(json.dumps(record, ensure_ascii=False) + "\n")

            if stats.scanned % log_every == 0:
                log(
                    "progress: "
                    f"scanned={stats.scanned:,} translated={stats.translated:,} "
                    f"errors={stats.parsed_error:,}"
                )

    log(
        "done: "
        f"scanned={stats.scanned:,}, parsed_ok={stats.parsed_ok:,}, "
        f"parsed_error={stats.parsed_error:,}, translated={stats.translated:,}"
    )

    if summary_path is not None:
        summary_path.parent.mkdir(parents=True, exist_ok=True)
        with summary_path.open("w", encoding="utf-8") as f:
            f.write("School Translation Summary\n")
            f.write("=" * 80 + "\n")
            f.write(f"Input file          : {input_path}\n")
            f.write(f"Output file         : {output_path}\n")
            f.write(f"Scanned lines       : {stats.scanned:,}\n")
            f.write(f"Parsed records      : {stats.parsed_ok:,}\n")
            f.write(f"Parse errors        : {stats.parsed_error:,}\n")
            f.write(f"Empty school        : {stats.empty_school:,}\n")
            f.write(f"Translated schools  : {stats.translated:,}\n")
            f.write(f"Mojibake repaired   : {stats.mojibake_fixed:,}\n")
            f.write(f"Direct map hits     : {stats.direct_map_hits:,}\n")
            f.write(f"Suffix rule hits    : {stats.suffix_rule_hits:,}\n")
            f.write(f"Pinyin fallback hits: {stats.pinyin_hits:,}\n")

    return 0


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()

    project_root = Path(__file__).resolve().parents[1]
    dataset_dir = (project_root / "dataset").resolve()
    results_dir = (project_root / "results").resolve()

    input_path = resolve_path_arg(args.input, project_root, dataset_dir)
    output_path = resolve_path_arg(args.output, project_root, dataset_dir)
    summary_path = (
        resolve_path_arg(args.summary, project_root, results_dir)
        if args.summary is not None
        else None
    )

    started = time.time()
    try:
        exit_code = run_translation(
            input_path=input_path,
            output_path=output_path,
            log_every=args.log_every,
            max_lines=args.max_lines,
            summary_path=summary_path,
        )
    except Exception as exc:
        log(f"FAILED: {exc}")
        return 1

    log(f"Completed in {time.time() - started:.2f}s")
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
