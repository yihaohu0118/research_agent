"""Build a verl-compatible multi-turn SFT parquet from BFCL data.

This script reuses the repo's BFCL demo rendering logic so the generated
messages stay aligned with the BFCL env format used at RL time:

  system -> t3rl_text tool catalogue
  user   -> turn question
  assistant -> gold tool_call block
  user   -> plain_user tool-result text
  ...

The output parquet is intended for ``verl.trainer.fsdp_sft_trainer`` with
``data.multiturn.enable=true``.

Example
-------
    python3 scripts/build_bfcl_envtuning_sft_dataset.py \
      --input-parquet /path/to/source_a.parquet /path/to/source_b.parquet \
      --id-source-parquet data/bfcl_train_400.parquet \
      --output-parquet data/bfcl_envtuning_sft_train_200.parquet \
      --categories multi_turn_base multi_turn_miss_func \
      --max-per-category 100
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

from build_bfcl_demo_pool import (
    _extract_category,
    _extract_task_id,
    _extract_tools_from_row,
    _load_bfcl_jsonl_index,
    build_demo_for_row,
)


def _normalise_tools(value):
    if value is None:
        return None
    if hasattr(value, "tolist"):
        value = value.tolist()
    if not isinstance(value, (list, tuple)):
        return None
    out = [dict(item) for item in value if isinstance(item, dict)]
    return out or None


def _resolve_tools(row: pd.Series, bfcl_index: dict[str, dict[str, object]]) -> list[dict] | None:
    tools = _normalise_tools(_extract_tools_from_row(row))
    if tools:
        return tools

    task_id = _extract_task_id(row)
    rec = bfcl_index.get(task_id)
    if not isinstance(rec, dict):
        return None
    for key in ("function", "functions", "tools", "tool_schema"):
        tools = _normalise_tools(rec.get(key))
        if tools:
            return tools
    return None


def _sample_category_rows(
    df: pd.DataFrame,
    categories: list[str],
    max_per_category: int | None,
    seed: int,
) -> pd.DataFrame:
    cat_series = df.apply(_extract_category, axis=1)
    pieces: list[pd.DataFrame] = []
    for idx, category in enumerate(categories):
        subset = df.loc[cat_series == category]
        if subset.empty:
            raise ValueError(f"category {category!r} not found in input parquet")
        if max_per_category is not None and len(subset) < max_per_category:
            raise ValueError(
                f"category {category!r} only has {len(subset)} rows, "
                f"smaller than requested {max_per_category}"
            )
        if max_per_category is not None:
            subset = subset.sample(
                n=max_per_category,
                random_state=seed + idx,
                replace=False,
            )
        pieces.append(subset.reset_index(drop=True))
    return pd.concat(pieces, ignore_index=True)


def _collect_wanted_ids(
    df: pd.DataFrame,
    categories: list[str],
    max_per_category: int | None,
    seed: int,
) -> list[str]:
    selected = _sample_category_rows(
        df=df,
        categories=categories,
        max_per_category=max_per_category,
        seed=seed,
    )
    ids = [_extract_task_id(row) for _, row in selected.iterrows()]
    return ids


def _load_and_concat(paths: list[str]) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for raw_path in paths:
        path = Path(raw_path)
        if not path.exists():
            raise FileNotFoundError(f"input parquet not found: {path}")
        frame = pd.read_parquet(path)
        frames.append(frame)
        print(f"[envtuning_sft] loaded {len(frame)} rows from {path}")
    return pd.concat(frames, ignore_index=True)


def _select_rows_by_ids(
    df: pd.DataFrame,
    wanted_ids: list[str],
) -> pd.DataFrame:
    id_to_idx: dict[str, int] = {}
    for idx, (_, row) in enumerate(df.iterrows()):
        task_id = _extract_task_id(row)
        if task_id not in id_to_idx:
            id_to_idx[task_id] = idx

    missing = [task_id for task_id in wanted_ids if task_id not in id_to_idx]
    if missing:
        sample = ", ".join(missing[:10])
        raise ValueError(
            f"{len(missing)} wanted ids were not found in the source parquet set. "
            f"sample: {sample}"
        )

    ordered_indices = [id_to_idx[task_id] for task_id in wanted_ids]
    return df.iloc[ordered_indices].reset_index(drop=True)


def _build_records(
    df: pd.DataFrame,
    bfcl_index: dict[str, dict[str, object]],
    split: str,
    include_tools: bool,
) -> list[dict[str, object]]:
    records: list[dict[str, object]] = []
    skipped = 0
    for _, row in df.iterrows():
        demo = build_demo_for_row(row, bfcl_index=bfcl_index)
        if demo is None:
            skipped += 1
            continue

        record: dict[str, object] = {
            "task_id": demo["task_id"],
            "category": demo["category"],
            "data_source": demo["category"],
            "messages": demo["messages"],
            "enable_thinking": False,
            "split": split,
            "num_turns": demo["num_turns"],
            "num_calls": demo["num_calls"],
        }
        if include_tools:
            tools = _resolve_tools(row, bfcl_index)
            if tools:
                record["tools"] = tools
        records.append(record)

    if skipped:
        print(f"[envtuning_sft] skipped {skipped} rows with empty ground_truth")
    return records


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input-parquet",
        nargs="+",
        required=True,
        help="One or more parquet files to search for the actual SFT samples.",
    )
    parser.add_argument("--output-parquet", required=True)
    parser.add_argument(
        "--id-source-parquet",
        default=None,
        help=(
            "Optional parquet used only for choosing which task_ids to keep. "
            "This lets us preserve the local split IDs while sourcing the "
            "actual supervised rows from another dataset."
        ),
    )
    parser.add_argument(
        "--categories",
        nargs="+",
        default=["multi_turn_base", "multi_turn_miss_func"],
        help="BFCL categories to keep.",
    )
    parser.add_argument(
        "--max-per-category",
        type=int,
        default=100,
        help="Rows to keep per category. Use 0 or a negative value for all rows.",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--split", default="train")
    parser.add_argument(
        "--bfcl-jsonl",
        default=None,
        help="Optional BFCL JSONL fallback for missing tools schema.",
    )
    parser.add_argument(
        "--include-tools",
        action="store_true",
        help=(
            "Also emit a tools column. Off by default because our BFCL "
            "system prompt already inlines the tool schema and passing both "
            "can duplicate the schema in chat-template rendering."
        ),
    )
    args = parser.parse_args()

    bfcl_index = _load_bfcl_jsonl_index(
        Path(args.bfcl_jsonl) if args.bfcl_jsonl else None
    )

    max_per_category = args.max_per_category
    if max_per_category is not None and max_per_category <= 0:
        max_per_category = None

    try:
        source_df = _load_and_concat(list(args.input_parquet))
    except FileNotFoundError as exc:
        print(f"[envtuning_sft] error: {exc}", file=sys.stderr)
        return 1

    if args.id_source_parquet:
        id_source_path = Path(args.id_source_parquet)
        if not id_source_path.exists():
            print(
                f"[envtuning_sft] error: id-source parquet not found at {id_source_path}",
                file=sys.stderr,
            )
            return 1
        id_source_df = pd.read_parquet(id_source_path)
        print(f"[envtuning_sft] loaded {len(id_source_df)} id-source rows from {id_source_path}")
        wanted_ids = _collect_wanted_ids(
            df=id_source_df,
            categories=list(args.categories),
            max_per_category=max_per_category,
            seed=args.seed,
        )
        selected_df = _select_rows_by_ids(source_df, wanted_ids)
    else:
        selected_df = _sample_category_rows(
            df=source_df,
            categories=list(args.categories),
            max_per_category=max_per_category,
            seed=args.seed,
        )

    print(
        "[envtuning_sft] selected rows: "
        + ", ".join(
            f"{cat}={sum(selected_df.apply(_extract_category, axis=1) == cat)}"
            for cat in args.categories
        )
    )

    records = _build_records(
        df=selected_df,
        bfcl_index=bfcl_index,
        split=args.split,
        include_tools=bool(args.include_tools),
    )
    if not records:
        print("[envtuning_sft] error: no valid records produced", file=sys.stderr)
        return 1

    out_df = pd.DataFrame.from_records(records)
    output_path = Path(args.output_parquet)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_parquet(output_path, index=False)

    print(
        f"[envtuning_sft] wrote {len(out_df)} rows to {output_path} "
        f"(columns={out_df.columns.tolist()})"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
