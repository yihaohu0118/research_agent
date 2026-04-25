#!/usr/bin/env python3
"""Summarize official BFCL score files on a parquet-defined id subset."""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import pandas as pd


BFCL_CATEGORY_PREFIXES = (
    "multi_turn_base",
    "multi_turn_miss_func",
    "multi_turn_miss_param",
    "multi_turn_long_context",
)


def category_from_id(task_id: str) -> str:
    for prefix in BFCL_CATEGORY_PREFIXES:
        if task_id == prefix or task_id.startswith(prefix + "_"):
            return prefix
    return task_id.rsplit("_", 1)[0] if "_" in task_id else "unknown"


def extract_extra(row: pd.Series) -> dict[str, Any]:
    extra = row.get("extra_info")
    return extra if isinstance(extra, dict) else {}


def extract_task_id(row: pd.Series) -> str:
    extra = extract_extra(row)
    return str(extra.get("original_id") or extra.get("index") or row.get("id") or "")


def load_subset_ids(parquet_path: Path) -> set[str]:
    df = pd.read_parquet(parquet_path)
    ids = {extract_task_id(row) for _, row in df.iterrows()}
    ids.discard("")
    return ids


def iter_score_entries(value: Any):
    if isinstance(value, list):
        yield from value
        return
    if isinstance(value, dict):
        for key in ("scores", "result", "results", "data", "entries"):
            nested = value.get(key)
            if isinstance(nested, list):
                yield from nested
                return


def entry_id(entry: dict[str, Any]) -> str:
    return str(
        entry.get("id")
        or entry.get("test_id")
        or entry.get("question_id")
        or entry.get("prompt_id")
        or ""
    )


def entry_passed(entry: dict[str, Any]) -> bool:
    for key in ("valid", "passed", "success", "correct"):
        if key in entry:
            return bool(entry[key])
    for key in ("accuracy", "score"):
        if key in entry:
            try:
                return float(entry[key]) >= 1.0
            except (TypeError, ValueError):
                return False
    return False


def score_files(score_dir: Path, model_name: str) -> list[Path]:
    model_dir = score_dir / model_name.replace("/", "_")
    if model_dir.is_dir():
        return sorted(model_dir.glob("*multi_turn*score*.json"))
    return sorted(score_dir.glob(f"**/{model_name.replace('/', '_')}/*multi_turn*score*.json"))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--parquet", required=True, type=Path)
    parser.add_argument("--score-dir", required=True, type=Path)
    parser.add_argument("--model-name", required=True)
    args = parser.parse_args()

    subset_ids = load_subset_ids(args.parquet)
    files = score_files(args.score_dir, args.model_name)
    if not files:
        print(f"[bfcl-subset] no score files found under {args.score_dir}")
        return 1

    matched: dict[str, bool] = {}
    unsupported: list[Path] = []
    for path in files:
        with path.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
        entries = list(iter_score_entries(payload))
        if not entries:
            unsupported.append(path)
            continue
        for item in entries:
            if not isinstance(item, dict):
                continue
            task_id = entry_id(item)
            if task_id in subset_ids:
                matched[task_id] = entry_passed(item)

    missing = sorted(subset_ids - set(matched))
    counts = Counter(category_from_id(task_id) for task_id in subset_ids)
    correct_by_cat: dict[str, int] = defaultdict(int)
    seen_by_cat: dict[str, int] = defaultdict(int)
    for task_id, passed in matched.items():
        category = category_from_id(task_id)
        seen_by_cat[category] += 1
        correct_by_cat[category] += int(passed)

    print("============================================")
    print("  BFCL Subset Score")
    print("============================================")
    print(f"Subset parquet: {args.parquet}")
    print(f"Subset ids:     {len(subset_ids)}")
    print(f"Matched scores: {len(matched)}")
    if missing:
        print(f"Missing scores: {len(missing)} (first 10: {missing[:10]})")
    if unsupported:
        print(
            "[bfcl-subset] warning: unsupported aggregate-only score files: "
            + ", ".join(str(path) for path in unsupported)
        )

    overall_correct = sum(int(value) for value in matched.values())
    overall_total = len(matched)
    for category in sorted(counts):
        total = seen_by_cat.get(category, 0)
        correct = correct_by_cat.get(category, 0)
        acc = correct / total if total else 0.0
        print(f"{category:28s} {acc:.4f} ({correct}/{total}, subset_expected={counts[category]})")

    overall = overall_correct / overall_total if overall_total else 0.0
    print("-" * 60)
    print(f"{'overall':28s} {overall:.4f} ({overall_correct}/{overall_total})")
    return 0 if not missing and overall_total else 1


if __name__ == "__main__":
    raise SystemExit(main())
