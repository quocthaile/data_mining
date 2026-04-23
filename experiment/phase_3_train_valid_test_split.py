#!/usr/bin/env python3
"""
Phase 3: dataset split + class imbalance handling for supervised training.

Scenario alignment goals:
- Produce train/valid/test splits with leakage-aware options.
- Keep reproducible split manifests and class-distribution reports.
- Apply imbalance handling ONLY on train modeling data.

Supported split strategies:
- stratified: stratified random split by label.
- group: group-aware split (no group leakage), with seed search to improve label balance.
- temporal: chronological split by time column.
- hybrid: temporal test split + group/stratified split for train/valid.

Supported imbalance methods:
- none: keep original train set.
- random_oversample: bootstrap minority classes to majority size.
- smote: synthetic oversampling for numeric features.
"""

from __future__ import annotations

import argparse
import csv
import time
import traceback
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Set, Tuple

import numpy as np
from sklearn.model_selection import GroupShuffleSplit, train_test_split
from sklearn.neighbors import NearestNeighbors


@dataclass
class Phase3Config:
    project_root: Path
    results_dir: Path
    output_dir: Path
    labeled_csv: Path
    combined_csv: Path
    merge_combined: bool
    label_column: str
    split_strategy: str
    group_column: str
    time_column: str
    time_fallback_column: str
    time_format: str
    valid_size: float
    test_size: float
    seed: int
    seed_trials: int
    imbalance_method: str
    smote_k_neighbors: int
    feature_columns_arg: Optional[str]
    min_numeric_ratio: float
    log_every: int
    max_rows: Optional[int]


@dataclass
class SplitResult:
    train_idx: np.ndarray
    valid_idx: np.ndarray
    test_idx: np.ndarray
    selected_seed: Optional[int]
    quality_score: float
    notes: str


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


def safe_float(value) -> float:
    if value is None:
        return 0.0
    text = str(value).strip()
    if not text:
        return 0.0
    try:
        return float(text)
    except ValueError:
        return 0.0


def parse_datetime_value(text: str, time_format: str) -> Optional[datetime]:
    raw = (text or "").strip()
    if not raw:
        return None

    if time_format:
        try:
            return datetime.strptime(raw, time_format)
        except ValueError:
            pass

    try:
        return datetime.fromisoformat(raw)
    except ValueError:
        return None


def normalize_label(value: str) -> str:
    text = (value or "").strip()
    return text if text else "Unknown"


def load_csv_rows(path: Path, max_rows: Optional[int], log_every: int) -> Tuple[List[Dict[str, str]], List[str]]:
    rows: List[Dict[str, str]] = []
    with path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        columns = list(reader.fieldnames or [])
        for idx, row in enumerate(reader, start=1):
            if max_rows is not None and idx > max_rows:
                break
            rows.append({k: (v if v is not None else "") for k, v in row.items()})
            if idx % log_every == 0:
                log(f"Load progress ({path.name}): rows={idx:,}")

    if not rows:
        raise RuntimeError(f"No rows found in {path}")

    log(f"Loaded {len(rows):,} rows from {path.name}")
    return rows, columns


def merge_missing_columns_from_combined(
    rows: List[Dict[str, str]],
    current_columns: List[str],
    combined_csv: Path,
    key_column: str,
    log_every: int,
) -> List[str]:
    if not combined_csv.exists():
        log(f"Combined CSV not found, skip merge: {combined_csv}")
        return []

    key_to_idx: Dict[str, int] = {}
    for idx, row in enumerate(rows):
        key = (row.get(key_column) or "").strip()
        if key and key not in key_to_idx:
            key_to_idx[key] = idx

    if not key_to_idx:
        log(f"No non-empty {key_column} values in labeled CSV, skip merge")
        return []

    added_columns: List[str] = []
    merged_rows = 0

    with combined_csv.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        combined_columns = list(reader.fieldnames or [])
        candidate_columns = [
            c for c in combined_columns if c != key_column and c not in current_columns
        ]
        added_columns.extend(candidate_columns)

        for idx, row in enumerate(reader, start=1):
            key = (row.get(key_column) or "").strip()
            target_idx = key_to_idx.get(key)
            if target_idx is None:
                if idx % log_every == 0:
                    log(f"Merge scan progress: rows={idx:,}")
                continue

            target = rows[target_idx]
            for col in candidate_columns:
                current_value = (target.get(col) or "").strip()
                if not current_value:
                    target[col] = row.get(col) or ""
            merged_rows += 1

            if idx % log_every == 0:
                log(f"Merge scan progress: rows={idx:,}")

    log(
        f"Combined merge done: matched_rows={merged_rows:,}, "
        f"added_columns={len(added_columns)}"
    )
    return added_columns


def class_distribution(y: np.ndarray, idx: np.ndarray, labels: Sequence[str]) -> Dict[str, float]:
    if idx.size == 0:
        return {label: 0.0 for label in labels}

    subset = y[idx]
    out: Dict[str, float] = {}
    for label in labels:
        out[label] = float(np.sum(subset == label)) / float(idx.size)
    return out


def class_counts(y: np.ndarray, idx: np.ndarray, labels: Sequence[str]) -> Dict[str, int]:
    if idx.size == 0:
        return {label: 0 for label in labels}

    subset = y[idx]
    out: Dict[str, int] = {}
    for label in labels:
        out[label] = int(np.sum(subset == label))
    return out


def split_quality_score(y: np.ndarray, train_idx: np.ndarray, valid_idx: np.ndarray, test_idx: np.ndarray) -> float:
    labels = sorted({str(v) for v in y.tolist()})
    all_idx = np.arange(y.shape[0])
    all_dist = class_distribution(y, all_idx, labels)

    score = 0.0
    for subset_idx in [train_idx, valid_idx, test_idx]:
        dist = class_distribution(y, subset_idx, labels)
        counts = class_counts(y, subset_idx, labels)
        for label in labels:
            score += abs(dist[label] - all_dist[label])
            if counts[label] == 0:
                score += 5.0
            elif counts[label] < 3:
                score += 0.5 * float(3 - counts[label])

    return score


def adjust_split_counts(n: int, valid_size: float, test_size: float) -> Tuple[int, int, int]:
    if n < 3:
        raise RuntimeError("Need at least 3 rows to split into train/valid/test")

    n_test = max(1, int(round(n * test_size))) if test_size > 0 else 0
    n_valid = max(1, int(round(n * valid_size))) if valid_size > 0 else 0

    while n_test + n_valid >= n:
        if n_valid >= n_test and n_valid > 1:
            n_valid -= 1
        elif n_test > 1:
            n_test -= 1
        else:
            break

    n_train = n - n_test - n_valid
    if n_train < 1:
        raise RuntimeError("Invalid split sizes. Train set would be empty.")

    return n_train, n_valid, n_test


def split_stratified_once(
    indices: np.ndarray,
    y: np.ndarray,
    valid_size: float,
    test_size: float,
    seed: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    y_subset = y[indices]
    unique = np.unique(y_subset)
    stratify_test = y_subset if unique.shape[0] > 1 else None

    train_valid_idx, test_idx = train_test_split(
        indices,
        test_size=test_size,
        random_state=seed,
        stratify=stratify_test,
    )

    valid_share = valid_size / max(1e-12, (1.0 - test_size))
    y_train_valid = y[train_valid_idx]
    unique_tv = np.unique(y_train_valid)
    stratify_valid = y_train_valid if unique_tv.shape[0] > 1 else None

    train_idx, valid_idx = train_test_split(
        train_valid_idx,
        test_size=valid_share,
        random_state=seed + 7919,
        stratify=stratify_valid,
    )

    return train_idx, valid_idx, test_idx


def split_group_once(
    indices: np.ndarray,
    groups: np.ndarray,
    test_size: float,
    seed: int,
) -> Tuple[np.ndarray, np.ndarray]:
    gss = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=seed)
    rel_train, rel_test = next(gss.split(indices, groups=groups[indices]))
    return indices[rel_train], indices[rel_test]


def run_stratified_search(
    y: np.ndarray,
    valid_size: float,
    test_size: float,
    seed: int,
    seed_trials: int,
) -> SplitResult:
    indices = np.arange(y.shape[0])
    best: Optional[SplitResult] = None

    for trial in range(max(1, seed_trials)):
        candidate_seed = seed + trial
        train_idx, valid_idx, test_idx = split_stratified_once(
            indices=indices,
            y=y,
            valid_size=valid_size,
            test_size=test_size,
            seed=candidate_seed,
        )
        score = split_quality_score(y, train_idx, valid_idx, test_idx)

        if best is None or score < best.quality_score:
            best = SplitResult(
                train_idx=train_idx,
                valid_idx=valid_idx,
                test_idx=test_idx,
                selected_seed=candidate_seed,
                quality_score=score,
                notes="stratified seed search",
            )

    if best is None:
        raise RuntimeError("Could not produce stratified split")
    return best


def run_group_search(
    y: np.ndarray,
    groups: np.ndarray,
    valid_size: float,
    test_size: float,
    seed: int,
    seed_trials: int,
) -> SplitResult:
    indices = np.arange(y.shape[0])
    unique_groups = np.unique(groups[indices])
    if unique_groups.shape[0] < 3:
        raise RuntimeError("Not enough unique groups for group split")

    valid_share = valid_size / max(1e-12, (1.0 - test_size))

    best: Optional[SplitResult] = None
    for trial in range(max(1, seed_trials)):
        candidate_seed = seed + trial
        try:
            train_valid_idx, test_idx = split_group_once(
                indices=indices,
                groups=groups,
                test_size=test_size,
                seed=candidate_seed,
            )
            train_idx, valid_idx = split_group_once(
                indices=train_valid_idx,
                groups=groups,
                test_size=valid_share,
                seed=candidate_seed + 7919,
            )
        except Exception:
            continue

        score = split_quality_score(y, train_idx, valid_idx, test_idx)
        if best is None or score < best.quality_score:
            best = SplitResult(
                train_idx=train_idx,
                valid_idx=valid_idx,
                test_idx=test_idx,
                selected_seed=candidate_seed,
                quality_score=score,
                notes="group seed search",
            )

    if best is None:
        raise RuntimeError("Could not produce group split")
    return best


def run_temporal_split(
    y: np.ndarray,
    timestamps: np.ndarray,
    valid_size: float,
    test_size: float,
) -> SplitResult:
    indices = np.arange(y.shape[0])
    order = indices[np.argsort(timestamps, kind="mergesort")]

    n_train, n_valid, _ = adjust_split_counts(order.shape[0], valid_size, test_size)

    train_idx = order[:n_train]
    valid_idx = order[n_train : n_train + n_valid]
    test_idx = order[n_train + n_valid :]

    score = split_quality_score(y, train_idx, valid_idx, test_idx)
    return SplitResult(
        train_idx=train_idx,
        valid_idx=valid_idx,
        test_idx=test_idx,
        selected_seed=None,
        quality_score=score,
        notes="pure temporal split",
    )


def run_hybrid_split(
    y: np.ndarray,
    groups: np.ndarray,
    timestamps: np.ndarray,
    valid_size: float,
    test_size: float,
    seed: int,
    seed_trials: int,
    use_group_for_train_valid: bool,
) -> SplitResult:
    indices = np.arange(y.shape[0])
    order = indices[np.argsort(timestamps, kind="mergesort")]

    _, _, n_test = adjust_split_counts(order.shape[0], valid_size, test_size)
    if n_test <= 0 or n_test >= order.shape[0]:
        raise RuntimeError("Invalid test split size for hybrid strategy")

    test_idx = order[-n_test:]
    remain_idx = order[:-n_test]
    valid_share = valid_size / max(1e-12, (1.0 - test_size))

    best: Optional[SplitResult] = None

    for trial in range(max(1, seed_trials)):
        candidate_seed = seed + trial
        try:
            if use_group_for_train_valid:
                train_idx, valid_idx = split_group_once(
                    indices=remain_idx,
                    groups=groups,
                    test_size=valid_share,
                    seed=candidate_seed,
                )
            else:
                y_remain = y[remain_idx]
                unique = np.unique(y_remain)
                stratify = y_remain if unique.shape[0] > 1 else None
                train_idx, valid_idx = train_test_split(
                    remain_idx,
                    test_size=valid_share,
                    random_state=candidate_seed,
                    stratify=stratify,
                )
        except Exception:
            continue

        score = split_quality_score(y, train_idx, valid_idx, test_idx)
        if best is None or score < best.quality_score:
            method = "group" if use_group_for_train_valid else "stratified"
            best = SplitResult(
                train_idx=train_idx,
                valid_idx=valid_idx,
                test_idx=test_idx,
                selected_seed=candidate_seed,
                quality_score=score,
                notes=f"hybrid split (temporal test + {method} train/valid)",
            )

    if best is None:
        raise RuntimeError("Could not produce hybrid split")
    return best


def validate_split_coverage(n_rows: int, split_result: SplitResult) -> None:
    train_set = set(split_result.train_idx.tolist())
    valid_set = set(split_result.valid_idx.tolist())
    test_set = set(split_result.test_idx.tolist())

    if train_set & valid_set or train_set & test_set or valid_set & test_set:
        raise RuntimeError("Split overlap detected")

    covered = len(train_set | valid_set | test_set)
    if covered != n_rows:
        raise RuntimeError(
            f"Split coverage mismatch: covered={covered}, total_rows={n_rows}"
        )


def parse_feature_columns_arg(text: Optional[str]) -> Optional[List[str]]:
    if text is None:
        return None
    parts = [p.strip() for p in text.split(",") if p.strip()]
    return parts if parts else None


def infer_feature_columns(
    rows: List[Dict[str, str]],
    candidate_columns: Sequence[str],
    min_numeric_ratio: float,
) -> List[str]:
    selected: List[str] = []
    for col in candidate_columns:
        non_empty = 0
        parse_ok = 0

        for row in rows:
            raw = (row.get(col) or "").strip()
            if not raw:
                continue
            non_empty += 1
            try:
                float(raw)
                parse_ok += 1
            except ValueError:
                pass

        if non_empty == 0:
            continue

        ratio = float(parse_ok) / float(non_empty)
        if ratio >= min_numeric_ratio:
            selected.append(col)

    return selected


def build_feature_matrix_and_labels(
    rows: List[Dict[str, str]],
    indices: np.ndarray,
    feature_columns: Sequence[str],
    label_column: str,
) -> Tuple[np.ndarray, np.ndarray]:
    X = np.zeros((indices.shape[0], len(feature_columns)), dtype=np.float64)
    y = np.empty(indices.shape[0], dtype=object)

    for i, row_idx in enumerate(indices.tolist()):
        row = rows[row_idx]
        y[i] = normalize_label(row.get(label_column, ""))
        for j, col in enumerate(feature_columns):
            X[i, j] = safe_float(row.get(col))

    return X, y


def random_oversample(
    X: np.ndarray,
    y: np.ndarray,
    seed: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)

    classes, counts = np.unique(y, return_counts=True)
    target = int(np.max(counts))

    selected_idx = list(range(y.shape[0]))
    origins = ["original"] * y.shape[0]

    for cls, count in zip(classes.tolist(), counts.tolist()):
        need = target - int(count)
        if need <= 0:
            continue

        cls_idx = np.where(y == cls)[0]
        extra_idx = rng.choice(cls_idx, size=need, replace=True)
        selected_idx.extend(extra_idx.tolist())
        origins.extend(["oversampled_copy"] * need)

    perm = rng.permutation(len(selected_idx))
    selected_arr = np.array(selected_idx, dtype=np.int64)[perm]
    origin_arr = np.array(origins, dtype=object)[perm]

    return X[selected_arr], y[selected_arr], origin_arr


def smote_oversample(
    X: np.ndarray,
    y: np.ndarray,
    seed: int,
    k_neighbors: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)

    classes, counts = np.unique(y, return_counts=True)
    target = int(np.max(counts))

    x_blocks = [X]
    y_blocks = [y]
    origin_blocks: List[np.ndarray] = [np.array(["original"] * y.shape[0], dtype=object)]

    for cls, count in zip(classes.tolist(), counts.tolist()):
        need = target - int(count)
        if need <= 0:
            continue

        cls_idx = np.where(y == cls)[0]
        X_cls = X[cls_idx]

        if cls_idx.shape[0] < 2:
            extra_idx = rng.choice(cls_idx, size=need, replace=True)
            x_blocks.append(X[extra_idx])
            y_blocks.append(y[extra_idx])
            origin_blocks.append(np.array(["oversampled_copy"] * need, dtype=object))
            continue

        k_use = min(max(1, k_neighbors), cls_idx.shape[0] - 1)
        nn = NearestNeighbors(n_neighbors=k_use + 1)
        nn.fit(X_cls)
        neigh = nn.kneighbors(X_cls, return_distance=False)

        synthetic = np.zeros((need, X.shape[1]), dtype=np.float64)
        for i in range(need):
            src_local = int(rng.integers(0, cls_idx.shape[0]))
            src_global = cls_idx[src_local]

            neigh_local_candidates = neigh[src_local, 1:]
            if neigh_local_candidates.shape[0] == 0:
                dst_global = src_global
            else:
                dst_local = int(rng.choice(neigh_local_candidates))
                dst_global = cls_idx[dst_local]

            lam = float(rng.random())
            synthetic[i] = X[src_global] + lam * (X[dst_global] - X[src_global])

        x_blocks.append(synthetic)
        y_blocks.append(np.array([cls] * need, dtype=object))
        origin_blocks.append(np.array(["smote_synthetic"] * need, dtype=object))

    X_out = np.vstack(x_blocks)
    y_out = np.concatenate(y_blocks)
    origin_out = np.concatenate(origin_blocks)

    perm = rng.permutation(X_out.shape[0])
    return X_out[perm], y_out[perm], origin_out[perm]


def write_split_dataset_files(
    rows: List[Dict[str, str]],
    columns: Sequence[str],
    split_result: SplitResult,
    output_dir: Path,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    split_map: Dict[int, str] = {}
    for idx in split_result.train_idx.tolist():
        split_map[int(idx)] = "train"
    for idx in split_result.valid_idx.tolist():
        split_map[int(idx)] = "valid"
    for idx in split_result.test_idx.tolist():
        split_map[int(idx)] = "test"

    all_columns = list(columns) + ["SplitSet"]

    all_path = output_dir / "stage3_dataset_with_split.csv"
    with all_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=all_columns)
        writer.writeheader()
        for idx, row in enumerate(rows):
            row_out = {col: row.get(col, "") for col in columns}
            row_out["SplitSet"] = split_map.get(idx, "")
            writer.writerow(row_out)

    subset_specs = [
        ("train", split_result.train_idx, output_dir / "stage3_train.csv"),
        ("valid", split_result.valid_idx, output_dir / "stage3_valid.csv"),
        ("test", split_result.test_idx, output_dir / "stage3_test.csv"),
    ]

    for split_name, split_idx, path in subset_specs:
        with path.open("w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=all_columns)
            writer.writeheader()
            for idx in split_idx.tolist():
                row = rows[int(idx)]
                row_out = {col: row.get(col, "") for col in columns}
                row_out["SplitSet"] = split_name
                writer.writerow(row_out)


def write_modeling_train_csv(
    path: Path,
    X_model: np.ndarray,
    y_model: np.ndarray,
    sample_origin: np.ndarray,
    feature_columns: Sequence[str],
    label_column: str,
) -> None:
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(list(feature_columns) + [label_column, "sample_origin"])

        for i in range(X_model.shape[0]):
            row = [round(float(v), 6) for v in X_model[i].tolist()]
            row.extend([str(y_model[i]), str(sample_origin[i])])
            writer.writerow(row)


def write_label_distribution_csv(
    path: Path,
    y_all: np.ndarray,
    split_result: SplitResult,
    y_model: np.ndarray,
) -> None:
    labels = sorted({str(v) for v in y_all.tolist()} | {str(v) for v in y_model.tolist()})

    specs = [
        ("train", y_all[split_result.train_idx]),
        ("valid", y_all[split_result.valid_idx]),
        ("test", y_all[split_result.test_idx]),
        ("train_modeling", y_model),
    ]

    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["split", "label", "count", "ratio"])

        for split_name, split_labels in specs:
            total = max(1, split_labels.shape[0])
            for label in labels:
                count = int(np.sum(split_labels == label))
                ratio = float(count) / float(total)
                writer.writerow([split_name, label, count, round(ratio, 6)])


def write_report(
    path: Path,
    cfg: Phase3Config,
    split_result: SplitResult,
    labels: np.ndarray,
    y_model: np.ndarray,
    feature_columns: Sequence[str],
    missing_time_rows: int,
    merged_columns: Sequence[str],
    elapsed_seconds: float,
) -> None:
    unique_labels = sorted({str(v) for v in labels.tolist()})

    def format_counts(idx: np.ndarray) -> str:
        counts = class_counts(labels, idx, unique_labels)
        parts = [f"{label}={counts[label]:,}" for label in unique_labels]
        return ", ".join(parts)

    train_model_counts = {
        label: int(np.sum(y_model == label))
        for label in sorted({str(v) for v in y_model.tolist()})
    }

    with path.open("w", encoding="utf-8") as f:
        f.write("Phase 3 - Train/Valid/Test Split and Imbalance Handling Report\n")
        f.write("=" * 100 + "\n")
        f.write(f"Generated at                    : {now_text()}\n")
        f.write(f"Input labeled CSV              : {cfg.labeled_csv}\n")
        f.write(f"Input combined CSV             : {cfg.combined_csv}\n")
        f.write(f"Output directory               : {cfg.output_dir}\n")
        f.write(f"Elapsed seconds                : {elapsed_seconds:.2f}\n")

        f.write("\nConfiguration:\n")
        f.write(f"- Split strategy               : {cfg.split_strategy}\n")
        f.write(f"- Label column                 : {cfg.label_column}\n")
        f.write(f"- Group column                 : {cfg.group_column}\n")
        f.write(f"- Time column                  : {cfg.time_column}\n")
        f.write(f"- Time fallback column         : {cfg.time_fallback_column}\n")
        f.write(f"- Valid size                   : {cfg.valid_size:.4f}\n")
        f.write(f"- Test size                    : {cfg.test_size:.4f}\n")
        f.write(f"- Seed                         : {cfg.seed}\n")
        f.write(f"- Seed trials                  : {cfg.seed_trials}\n")
        f.write(f"- Selected seed                : {split_result.selected_seed}\n")
        f.write(f"- Split quality score          : {split_result.quality_score:.6f}\n")
        f.write(f"- Notes                        : {split_result.notes}\n")

        f.write("\nLeakage and enrichment checks:\n")
        f.write(f"- Missing time rows            : {missing_time_rows:,}\n")
        f.write(f"- Merged columns from combined : {len(merged_columns)}\n")

        f.write("\nSplit sizes:\n")
        f.write(f"- Train rows                   : {split_result.train_idx.shape[0]:,}\n")
        f.write(f"- Valid rows                   : {split_result.valid_idx.shape[0]:,}\n")
        f.write(f"- Test rows                    : {split_result.test_idx.shape[0]:,}\n")

        f.write("\nLabel distribution by split:\n")
        f.write(f"- Train                        : {format_counts(split_result.train_idx)}\n")
        f.write(f"- Valid                        : {format_counts(split_result.valid_idx)}\n")
        f.write(f"- Test                         : {format_counts(split_result.test_idx)}\n")

        f.write("\nTraining imbalance handling:\n")
        f.write(f"- Method                       : {cfg.imbalance_method}\n")
        f.write(f"- SMOTE k neighbors            : {cfg.smote_k_neighbors}\n")
        train_model_line = ", ".join(
            [f"{k}={train_model_counts[k]:,}" for k in sorted(train_model_counts.keys())]
        )
        f.write(f"- Train modeling counts        : {train_model_line}\n")

        f.write("\nSelected feature columns:\n")
        for col in feature_columns:
            f.write(f"- {col}\n")

        f.write("\nGenerated files:\n")
        f.write(f"- {cfg.output_dir / 'stage3_dataset_with_split.csv'}\n")
        f.write(f"- {cfg.output_dir / 'stage3_train.csv'}\n")
        f.write(f"- {cfg.output_dir / 'stage3_valid.csv'}\n")
        f.write(f"- {cfg.output_dir / 'stage3_test.csv'}\n")
        f.write(f"- {cfg.output_dir / 'stage3_train_modeling.csv'}\n")
        f.write(f"- {cfg.output_dir / 'stage3_label_distribution.csv'}\n")
        f.write(f"- {cfg.output_dir / 'stage3_split_report.txt'}\n")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Phase 3: split labeled data and prepare train modeling set with imbalance handling."
    )
    parser.add_argument("--results-dir", type=Path, default=Path("results"))
    parser.add_argument("--output-dir", type=Path, default=Path("results"))
    parser.add_argument("--input", type=Path, default=Path("step5_standard_labels_kmeans.csv"))
    parser.add_argument("--combined-input", type=Path, default=Path("combined_user_metrics.csv"))
    parser.add_argument(
        "--skip-merge-combined",
        action="store_true",
        help="Do not enrich labeled rows with extra columns from combined CSV",
    )

    parser.add_argument("--label-column", type=str, default="StandardLabelKMeans")
    parser.add_argument(
        "--split-strategy",
        type=str,
        choices=["stratified", "group", "temporal", "hybrid"],
        default="stratified",
    )
    parser.add_argument("--group-column", type=str, default="user_id")
    parser.add_argument("--time-column", type=str, default="last_activity_time")
    parser.add_argument("--time-fallback-column", type=str, default="first_activity_time")
    parser.add_argument(
        "--time-format",
        type=str,
        default="%Y-%m-%d %H:%M:%S",
        help="datetime format for time parsing",
    )

    parser.add_argument("--valid-size", type=float, default=0.10)
    parser.add_argument("--test-size", type=float, default=0.10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--seed-trials", type=int, default=30)

    parser.add_argument(
        "--imbalance-method",
        type=str,
        choices=["none", "random_oversample", "smote"],
        default="random_oversample",
    )
    parser.add_argument("--smote-k-neighbors", type=int, default=5)

    parser.add_argument(
        "--feature-columns",
        type=str,
        default=None,
        help="Comma-separated feature columns. If omitted, infer numeric columns automatically.",
    )
    parser.add_argument(
        "--min-numeric-ratio",
        type=float,
        default=0.95,
        help="Minimum parseable ratio for automatic numeric feature detection",
    )

    parser.add_argument("--log-every", type=int, default=200000)
    parser.add_argument("--max-rows", type=int, default=None)
    return parser


def main() -> int:
    args = build_parser().parse_args()

    project_root = Path(__file__).resolve().parents[1]
    results_dir = resolve_path_arg(args.results_dir, project_root, project_root)
    output_dir = resolve_path_arg(args.output_dir, project_root, project_root)

    labeled_csv = resolve_path_arg(args.input, project_root, results_dir)
    combined_csv = resolve_path_arg(args.combined_input, project_root, results_dir)

    cfg = Phase3Config(
        project_root=project_root,
        results_dir=results_dir,
        output_dir=output_dir,
        labeled_csv=labeled_csv,
        combined_csv=combined_csv,
        merge_combined=(not args.skip_merge_combined),
        label_column=args.label_column,
        split_strategy=args.split_strategy,
        group_column=args.group_column,
        time_column=args.time_column,
        time_fallback_column=args.time_fallback_column,
        time_format=args.time_format,
        valid_size=float(args.valid_size),
        test_size=float(args.test_size),
        seed=int(args.seed),
        seed_trials=max(1, int(args.seed_trials)),
        imbalance_method=args.imbalance_method,
        smote_k_neighbors=max(1, int(args.smote_k_neighbors)),
        feature_columns_arg=args.feature_columns,
        min_numeric_ratio=max(0.0, min(1.0, float(args.min_numeric_ratio))),
        log_every=max(1, int(args.log_every)),
        max_rows=args.max_rows,
    )

    if cfg.valid_size <= 0 or cfg.test_size <= 0:
        log("FAILED: valid-size and test-size must be > 0")
        return 1
    if cfg.valid_size + cfg.test_size >= 1:
        log("FAILED: valid-size + test-size must be < 1")
        return 1

    try:
        started = time.time()
        log("Starting Phase 3: split + imbalance handling")

        rows, base_columns = load_csv_rows(cfg.labeled_csv, cfg.max_rows, cfg.log_every)

        merged_columns: List[str] = []
        if cfg.merge_combined:
            merged_columns = merge_missing_columns_from_combined(
                rows=rows,
                current_columns=base_columns,
                combined_csv=cfg.combined_csv,
                key_column="user_id",
                log_every=cfg.log_every,
            )

        columns = list(base_columns)
        for col in merged_columns:
            if col not in columns:
                columns.append(col)

        for idx, row in enumerate(rows):
            row["stage3_row_id"] = str(idx)
        if "stage3_row_id" not in columns:
            columns.append("stage3_row_id")

        if cfg.label_column not in columns:
            raise RuntimeError(f"Label column not found: {cfg.label_column}")

        labels = np.array([normalize_label(row.get(cfg.label_column, "")) for row in rows], dtype=object)

        if cfg.group_column in columns:
            groups = np.array(
                [
                    (row.get(cfg.group_column) or "").strip() or row["stage3_row_id"]
                    for row in rows
                ],
                dtype=object,
            )
        else:
            log(f"Group column not found: {cfg.group_column}. Fallback to stage3_row_id")
            groups = np.array([row["stage3_row_id"] for row in rows], dtype=object)

        timestamps = np.zeros(len(rows), dtype=np.float64)
        missing_time_rows = 0
        epoch = datetime(1970, 1, 1)
        for i, row in enumerate(rows):
            primary = row.get(cfg.time_column, "")
            fallback = row.get(cfg.time_fallback_column, "")
            dt = parse_datetime_value(primary, cfg.time_format)
            if dt is None:
                dt = parse_datetime_value(fallback, cfg.time_format)
            if dt is None:
                dt = epoch
                missing_time_rows += 1
            timestamps[i] = (
                float(dt.toordinal()) * 86400.0
                + float(dt.hour * 3600 + dt.minute * 60 + dt.second)
                + float(dt.microsecond) / 1_000_000.0
            )

        use_group_for_hybrid = len(np.unique(groups)) >= 3

        if cfg.split_strategy == "stratified":
            split_result = run_stratified_search(
                y=labels,
                valid_size=cfg.valid_size,
                test_size=cfg.test_size,
                seed=cfg.seed,
                seed_trials=cfg.seed_trials,
            )
        elif cfg.split_strategy == "group":
            split_result = run_group_search(
                y=labels,
                groups=groups,
                valid_size=cfg.valid_size,
                test_size=cfg.test_size,
                seed=cfg.seed,
                seed_trials=cfg.seed_trials,
            )
        elif cfg.split_strategy == "temporal":
            split_result = run_temporal_split(
                y=labels,
                timestamps=timestamps,
                valid_size=cfg.valid_size,
                test_size=cfg.test_size,
            )
        elif cfg.split_strategy == "hybrid":
            split_result = run_hybrid_split(
                y=labels,
                groups=groups,
                timestamps=timestamps,
                valid_size=cfg.valid_size,
                test_size=cfg.test_size,
                seed=cfg.seed,
                seed_trials=cfg.seed_trials,
                use_group_for_train_valid=use_group_for_hybrid,
            )
        else:
            raise RuntimeError(f"Unsupported split strategy: {cfg.split_strategy}")

        validate_split_coverage(len(rows), split_result)

        feature_columns_manual = parse_feature_columns_arg(cfg.feature_columns_arg)
        if feature_columns_manual is not None:
            missing_manual = [col for col in feature_columns_manual if col not in columns]
            if missing_manual:
                raise RuntimeError(f"Feature columns not found: {missing_manual}")
            feature_columns = feature_columns_manual
        else:
            excluded: Set[str] = {
                "stage3_row_id",
                "SplitSet",
                "user_id",
                "school",
                cfg.label_column,
                "EngagementLabel",
                "StandardLabelKMeans",
                "cluster",
                "kmeans_cluster_rank",
                "kmeans_cluster_mean_E",
                "kmeans_cluster_mean_E_norm",
                "E",
                "E_norm",
                cfg.group_column,
                cfg.time_column,
                cfg.time_fallback_column,
            }
            candidate_columns = [c for c in columns if c not in excluded]
            feature_columns = infer_feature_columns(
                rows=rows,
                candidate_columns=candidate_columns,
                min_numeric_ratio=cfg.min_numeric_ratio,
            )

        if not feature_columns:
            raise RuntimeError(
                "No usable feature columns found. Provide --feature-columns explicitly."
            )

        X_train, y_train = build_feature_matrix_and_labels(
            rows=rows,
            indices=split_result.train_idx,
            feature_columns=feature_columns,
            label_column=cfg.label_column,
        )

        if cfg.imbalance_method == "none":
            X_model = X_train
            y_model = y_train
            sample_origin = np.array(["original"] * y_train.shape[0], dtype=object)
        elif cfg.imbalance_method == "random_oversample":
            X_model, y_model, sample_origin = random_oversample(
                X=X_train,
                y=y_train,
                seed=cfg.seed,
            )
        elif cfg.imbalance_method == "smote":
            X_model, y_model, sample_origin = smote_oversample(
                X=X_train,
                y=y_train,
                seed=cfg.seed,
                k_neighbors=cfg.smote_k_neighbors,
            )
        else:
            raise RuntimeError(f"Unsupported imbalance method: {cfg.imbalance_method}")

        cfg.output_dir.mkdir(parents=True, exist_ok=True)

        write_split_dataset_files(
            rows=rows,
            columns=columns,
            split_result=split_result,
            output_dir=cfg.output_dir,
        )

        write_modeling_train_csv(
            path=cfg.output_dir / "stage3_train_modeling.csv",
            X_model=X_model,
            y_model=y_model,
            sample_origin=sample_origin,
            feature_columns=feature_columns,
            label_column=cfg.label_column,
        )

        write_label_distribution_csv(
            path=cfg.output_dir / "stage3_label_distribution.csv",
            y_all=labels,
            split_result=split_result,
            y_model=y_model,
        )

        write_report(
            path=cfg.output_dir / "stage3_split_report.txt",
            cfg=cfg,
            split_result=split_result,
            labels=labels,
            y_model=y_model,
            feature_columns=feature_columns,
            missing_time_rows=missing_time_rows,
            merged_columns=merged_columns,
            elapsed_seconds=time.time() - started,
        )

        log(f"Phase 3 completed in {time.time() - started:.2f}s")
        return 0
    except Exception as exc:
        log(f"FAILED: {exc}")
        print(traceback.format_exc())
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
