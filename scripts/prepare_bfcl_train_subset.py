#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


def extract_task_id(extra_info) -> str | None:
    if not isinstance(extra_info, dict):
        return None
    return extra_info.get("original_id") or extra_info.get("index")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Create a BFCL parquet subset from a split-id json."
    )
    parser.add_argument("--source", required=True, help="Source parquet file")
    parser.add_argument("--split-json", required=True, help="Split json path")
    parser.add_argument(
        "--split-key",
        default="train",
        help="Which split key to read from the json, default: train",
    )
    parser.add_argument("--output", required=True, help="Output parquet path")
    args = parser.parse_args()

    source = Path(args.source)
    split_json = Path(args.split_json)
    output = Path(args.output)

    df = pd.read_parquet(source)
    with open(split_json, "r", encoding="utf-8") as f:
        split = json.load(f)

    target_ids = split[args.split_key]
    id_to_rank = {task_id: idx for idx, task_id in enumerate(target_ids)}

    df = df.copy()
    df["_task_id"] = df["extra_info"].apply(extract_task_id)
    subset = df[df["_task_id"].isin(id_to_rank)].copy()
    subset["_rank"] = subset["_task_id"].map(id_to_rank)
    subset = subset.sort_values("_rank").drop(columns=["_task_id", "_rank"])

    if len(subset) != len(target_ids):
        found_ids = set(df["extra_info"].apply(extract_task_id).dropna().tolist())
        missing = [task_id for task_id in target_ids if task_id not in found_ids]
        raise ValueError(
            f"Expected {len(target_ids)} rows, got {len(subset)}. Missing ids: {missing[:10]}"
        )

    output.parent.mkdir(parents=True, exist_ok=True)
    subset.to_parquet(output, index=False)

    print(f"Wrote {len(subset)} rows -> {output}")
    if "data_source" in subset.columns:
        print(subset["data_source"].value_counts().to_dict())


if __name__ == "__main__":
    main()
