#!/usr/bin/env python3
"""Build a deterministic 400/400 BFCL train/test split from the parquet pool.

Usage::

    python scripts/prepare_bfcl_400_split.py \
        --pool data/bfcl_train.parquet data/bfcl_test.parquet \
        --out-train data/bfcl_train_400.parquet \
        --out-test  data/bfcl_test_400.parquet \
        --train-size 400 --test-size 400 --seed 0

Stratification key is the BFCL category inferred from ``extra_info.original_id``
(multi_turn_base / multi_turn_miss_param / multi_turn_miss_func /
multi_turn_long_context). We balance categories so each split has roughly
proportional representation, which keeps per-category metrics (success
rate, progress, etc.) statistically meaningful with 400 samples.
"""
from __future__ import annotations

import argparse
import json
import random
import sys
from collections import defaultdict
from pathlib import Path

import pandas as pd


BFCL_CATEGORY_PREFIXES = (
    "multi_turn_base",
    "multi_turn_miss_func",
    "multi_turn_miss_param",
    "multi_turn_long_context",
)


def infer_category(task_id: str) -> str:
    for prefix in BFCL_CATEGORY_PREFIXES:
        if task_id.startswith(prefix + "_") or task_id == prefix:
            return prefix
    return "other"


def extract_task_id(extra_info) -> str | None:
    if not isinstance(extra_info, dict):
        return None
    return extra_info.get("original_id") or extra_info.get("index")


def build_pool(parquet_paths: list[Path]) -> pd.DataFrame:
    frames = []
    for path in parquet_paths:
        if not path.exists():
            print(f"[warn] pool file missing: {path}", file=sys.stderr)
            continue
        df = pd.read_parquet(path)
        df = df.copy()
        df["_task_id"] = df["extra_info"].apply(extract_task_id)
        df["_category"] = df["_task_id"].apply(lambda x: infer_category(str(x)))
        frames.append(df)
    if not frames:
        raise SystemExit("no input parquet found")
    pool = pd.concat(frames, axis=0, ignore_index=True)
    # de-duplicate by (_task_id) so that overlapping train/test inputs do not
    # leak across splits.
    pool = pool.drop_duplicates(subset="_task_id", keep="first").reset_index(drop=True)
    return pool


def stratified_split(pool: pd.DataFrame, train_size: int, test_size: int, seed: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    rng = random.Random(seed)
    per_category = defaultdict(list)
    for idx, row in pool.iterrows():
        per_category[row["_category"]].append(idx)
    for cat in per_category:
        rng.shuffle(per_category[cat])

    total = sum(len(v) for v in per_category.values())
    train_alloc = {}
    test_alloc = {}
    used_train = used_test = 0
    for cat, idxs in per_category.items():
        frac = len(idxs) / total
        t_share = max(1, int(round(train_size * frac)))
        e_share = max(1, int(round(test_size * frac)))
        # Keep shares feasible given the pool for this category.
        t_share = min(t_share, len(idxs))
        e_share = min(e_share, len(idxs) - t_share)
        train_alloc[cat] = idxs[:t_share]
        test_alloc[cat] = idxs[t_share:t_share + e_share]
        used_train += t_share
        used_test += e_share

    # Fix rounding errors by padding from residual pool.
    all_train = sum(train_alloc.values(), [])
    all_test = sum(test_alloc.values(), [])
    used = set(all_train) | set(all_test)
    residual = [i for i in pool.index if i not in used]
    rng.shuffle(residual)
    while used_train < train_size and residual:
        all_train.append(residual.pop())
        used_train += 1
    while used_test < test_size and residual:
        all_test.append(residual.pop())
        used_test += 1

    train_df = pool.loc[all_train].drop(columns=["_task_id", "_category"]).reset_index(drop=True)
    test_df = pool.loc[all_test].drop(columns=["_task_id", "_category"]).reset_index(drop=True)
    return train_df, test_df


def write_split(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path, index=False)
    cats = df["extra_info"].apply(lambda e: infer_category(str(extract_task_id(e) or ""))).value_counts().to_dict()
    print(f"wrote {len(df):4d} rows -> {path}  | category_dist={cats}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pool", nargs="+", required=True, type=Path)
    ap.add_argument("--out-train", required=True, type=Path)
    ap.add_argument("--out-test", required=True, type=Path)
    ap.add_argument("--train-size", type=int, default=400)
    ap.add_argument("--test-size", type=int, default=400)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--dump-split-json", type=Path, default=None)
    args = ap.parse_args()

    pool = build_pool(args.pool)
    print(f"pool size after dedup: {len(pool)}")
    train_df, test_df = stratified_split(pool, args.train_size, args.test_size, args.seed)
    write_split(train_df, args.out_train)
    write_split(test_df, args.out_test)

    if args.dump_split_json:
        blob = {
            "train": [extract_task_id(e) for e in train_df["extra_info"]],
            "test": [extract_task_id(e) for e in test_df["extra_info"]],
            "seed": args.seed,
        }
        args.dump_split_json.parent.mkdir(parents=True, exist_ok=True)
        with open(args.dump_split_json, "w", encoding="utf-8") as f:
            json.dump(blob, f, indent=2)
        print(f"wrote split ids -> {args.dump_split_json}")


if __name__ == "__main__":
    main()
