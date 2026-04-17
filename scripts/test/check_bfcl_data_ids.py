#!/usr/bin/env python3
"""Check that BFCL data JSONL contains every id referenced by a split file."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path


def category_name(task_id: str) -> str:
    return str(task_id).rsplit("_", 1)[0]


def load_required_ids(split_path: Path, split_names: list[str]) -> list[str]:
    with split_path.open("r", encoding="utf-8") as f:
        payload = json.load(f)

    required: list[str] = []
    for split_name in split_names:
        values = payload.get(split_name, [])
        if isinstance(values, dict):
            values = list(values)
        required.extend(str(value) for value in values)
    return required


def load_data_ids(data_path: Path) -> set[str]:
    ids: set[str] = set()
    with data_path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                item = json.loads(line)
            except json.JSONDecodeError as exc:
                raise SystemExit(
                    f"Invalid JSON in {data_path} at line {line_no}: {exc}"
                ) from exc
            item_id = item.get("id")
            if item_id is not None:
                ids.add(str(item_id))
    return ids


def format_counts(ids: list[str] | set[str]) -> str:
    counts = Counter(category_name(task_id) for task_id in ids)
    return ", ".join(f"{key}={counts[key]}" for key in sorted(counts))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True, type=Path)
    parser.add_argument("--split", required=True, type=Path)
    parser.add_argument(
        "--splits",
        nargs="+",
        default=["train", "val", "test"],
        help="Split names to validate. Default: train val test.",
    )
    parser.add_argument("--sample", type=int, default=20)
    args = parser.parse_args()

    if not args.data.is_file():
        print(f"[bfcl-check] data file not found: {args.data}")
        return 1
    if not args.split.is_file():
        print(f"[bfcl-check] split file not found: {args.split}")
        return 1

    required_ids = load_required_ids(args.split, args.splits)
    data_ids = load_data_ids(args.data)
    missing = sorted(set(required_ids) - data_ids)

    print(f"[bfcl-check] data ids: {len(data_ids)} ({format_counts(data_ids)})")
    print(
        f"[bfcl-check] required split ids: {len(set(required_ids))} "
        f"({format_counts(required_ids)})"
    )

    if missing:
        print(
            f"[bfcl-check] missing {len(missing)} required ids from {args.data}:"
        )
        for task_id in missing[: args.sample]:
            print(f"  - {task_id}")
        if len(missing) > args.sample:
            print(f"  ... and {len(missing) - args.sample} more")
        return 1

    print("[bfcl-check] OK: data file covers all required split ids.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
