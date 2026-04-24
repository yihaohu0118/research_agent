#!/usr/bin/env python3
"""Validate BFCL parquet split hygiene before using a file for reporting."""

from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path
from typing import Any

import pandas as pd


BFCL_CATEGORY_PREFIXES = (
    "multi_turn_base",
    "multi_turn_miss_func",
    "multi_turn_miss_param",
    "multi_turn_long_context",
)


def is_missing(value: Any) -> bool:
    if value is None:
        return True
    try:
        result = pd.isna(value)
    except (TypeError, ValueError):
        return False
    if isinstance(result, bool):
        return result
    return False


def nested_get(value: Any, *keys: str) -> Any:
    current = value
    for key in keys:
        if not isinstance(current, dict):
            return None
        current = current.get(key)
    return current


def infer_category(task_id: Any) -> str:
    text = str(task_id or "")
    for prefix in BFCL_CATEGORY_PREFIXES:
        if text == prefix or text.startswith(prefix + "_"):
            return prefix
    return "other"


def extract_extra(row: pd.Series) -> dict[str, Any]:
    extra = row.get("extra_info")
    return extra if isinstance(extra, dict) else {}


def extract_interaction(row: pd.Series, extra: dict[str, Any]) -> dict[str, Any]:
    interaction = row.get("interaction_kwargs")
    if isinstance(interaction, dict):
        return interaction
    interaction = extra.get("interaction_kwargs")
    return interaction if isinstance(interaction, dict) else {}


def extract_task_id(row: pd.Series, extra: dict[str, Any]) -> str:
    task_id = extra.get("original_id") or extra.get("index") or row.get("id")
    return str(task_id or "")


def extract_split(extra: dict[str, Any], interaction: dict[str, Any]) -> str:
    split = extra.get("split") or interaction.get("split")
    return str(split or "")


def extract_ground_truth(row: pd.Series, extra: dict[str, Any]) -> Any:
    value = row.get("ground_truth")
    if not is_missing(value):
        return value
    value = extra.get("ground_truth")
    if not is_missing(value):
        return value
    return nested_get(extra, "interaction_kwargs", "ground_truth")


def extract_initial_config(row: pd.Series, extra: dict[str, Any], interaction: dict[str, Any]) -> Any:
    value = row.get("initial_config")
    if not is_missing(value):
        return value
    value = interaction.get("initial_config")
    if not is_missing(value):
        return value
    return nested_get(extra, "interaction_kwargs", "initial_config")


def prompt_contents(value: Any) -> list[str]:
    if hasattr(value, "tolist"):
        value = value.tolist()
    if isinstance(value, dict):
        content = str(value.get("content") or "").strip()
        return [content] if content else []
    if isinstance(value, str):
        text = value.strip()
        return [text] if text else []
    if isinstance(value, (list, tuple)):
        contents: list[str] = []
        for item in value:
            contents.extend(prompt_contents(item))
        return contents
    return []


def sample(values: list[str], limit: int) -> str:
    visible = values[:limit]
    suffix = "" if len(values) <= limit else f", ... +{len(values) - limit} more"
    return ", ".join(visible) + suffix


def validate_file(path: Path, expect_split: str | None, allow_stubs: bool, sample_size: int) -> int:
    if not path.is_file():
        print(f"[bfcl-split] missing file: {path}")
        return 1

    df = pd.read_parquet(path)
    task_ids: list[str] = []
    splits: list[str] = []
    categories: list[str] = []
    null_initial_config: list[str] = []
    null_ground_truth: list[str] = []
    id_only_prompts: list[str] = []

    for _, row in df.iterrows():
        extra = extract_extra(row)
        interaction = extract_interaction(row, extra)
        task_id = extract_task_id(row, extra)
        split = extract_split(extra, interaction)
        task_ids.append(task_id)
        splits.append(split)
        categories.append(infer_category(task_id))

        if is_missing(extract_initial_config(row, extra, interaction)):
            null_initial_config.append(task_id)
        if is_missing(extract_ground_truth(row, extra)):
            null_ground_truth.append(task_id)
        contents = prompt_contents(row.get("prompt"))
        if contents and contents[-1] == task_id:
            id_only_prompts.append(task_id)

    split_counts = Counter(splits)
    category_counts = Counter(categories)
    duplicate_ids = sorted(
        task_id for task_id, count in Counter(task_ids).items() if task_id and count > 1
    )

    print(f"[bfcl-split] {path}: rows={len(df)}")
    print(f"[bfcl-split] split_counts={dict(sorted(split_counts.items()))}")
    print(f"[bfcl-split] category_counts={dict(sorted(category_counts.items()))}")

    errors: list[str] = []
    if expect_split is not None:
        bad_splits = sorted(
            {split for split in splits if split != expect_split}
        )
        if bad_splits:
            errors.append(
                f"expected split={expect_split!r}, found {dict(sorted(split_counts.items()))}"
            )
    if duplicate_ids:
        errors.append(f"duplicate task ids: {sample(duplicate_ids, sample_size)}")
    if not allow_stubs:
        if null_initial_config:
            errors.append(
                "null initial_config ids: "
                + sample(sorted(null_initial_config), sample_size)
            )
        if null_ground_truth:
            errors.append(
                "null ground_truth ids: "
                + sample(sorted(null_ground_truth), sample_size)
            )
        if id_only_prompts:
            errors.append(
                "id-only prompt ids: "
                + sample(sorted(id_only_prompts), sample_size)
            )

    if errors:
        for error in errors:
            print(f"[bfcl-split] ERROR: {error}")
        return 1

    print("[bfcl-split] OK")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("parquet", nargs="+", type=Path)
    parser.add_argument(
        "--expect-split",
        choices=("train", "test", "val"),
        default=None,
        help="Require every row's BFCL split metadata to match this value.",
    )
    parser.add_argument(
        "--allow-stubs",
        action="store_true",
        help="Allow null ground truth / initial config and id-only prompts.",
    )
    parser.add_argument("--sample", type=int, default=8)
    args = parser.parse_args()

    status = 0
    for path in args.parquet:
        status |= validate_file(path, args.expect_split, args.allow_stubs, args.sample)
    return status


if __name__ == "__main__":
    raise SystemExit(main())
