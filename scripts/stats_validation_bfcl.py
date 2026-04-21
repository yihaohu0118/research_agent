#!/usr/bin/env python3
"""
Per-category validation stats for BFCL multi-turn.

Category identification strategy:
  - If the JSONL record has 'data_source' (from trainer fix): use it directly.
  - Otherwise, match by function count in the system prompt:
    * miss_func tasks have fewer tool definitions than other categories.
    * base / miss_param / long_context share identical prompts and CANNOT
      be separated from the input text alone. They are grouped as
      'base+param+lc' until data_source is available.

For full per-category stats, re-run validation after applying the trainer
fix that writes data_source into the JSONL, or use TOCF stats which
track per-category metrics during training.

Usage:
  python scripts/stats_validation_bfcl.py \
      --val-dir experiments/tech_synthetic/bfcl_tocf_tpatch/validation_log \
      --parquet data/bfcl_test.parquet \
      --bfcl-jsonl env_service/environments/bfcl/bfcl_data/multi_turn_processed.jsonl
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, FrozenSet, List, Tuple

import pandas as pd


BFCL_CATEGORY_PREFIXES = [
    "multi_turn_base",
    "multi_turn_miss_func",
    "multi_turn_miss_param",
    "multi_turn_long_context",
]

FUNC_NAME_RE = re.compile(r'"name"\s*:\s*"([^"]+)"')


def task_id_to_category(tid: str) -> str:
    for prefix in BFCL_CATEGORY_PREFIXES:
        if tid.startswith(prefix + "_") or tid == prefix:
            return prefix
    return tid.rsplit("_", 1)[0] if "_" in tid else tid


def extract_tool_names_from_input(input_text: str) -> frozenset:
    system_part = input_text.split("\nuser\n")[0] if "\nuser\n" in input_text else input_text
    return frozenset(FUNC_NAME_RE.findall(system_part))


def extract_user_query_from_input(input_text: str) -> str:
    parts = input_text.split("\nuser\n")
    if len(parts) >= 2:
        user_part = parts[1]
        for marker in ["\nassistant\n", "\nsystem\n", "\ntool\n"]:
            idx = user_part.find(marker)
            if idx != -1:
                user_part = user_part[:idx]
        return user_part.strip()[:300]
    return ""


# ─────────────────────────────────────────────────────────────────────
# Build category lookup from BFCL source data
# ─────────────────────────────────────────────────────────────────────

def build_lookup(parquet_path: str, bfcl_jsonl_path: str):
    """
    Build lookup tables for category identification.

    Returns:
        query_funccount_to_cats: (query_snippet, func_count) -> set of possible categories
        miss_func_signatures: set of (query_snippet, frozenset(tool_names)) for miss_func tasks
    """
    bfcl_data = {}
    with open(bfcl_jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            rec = json.loads(line)
            bfcl_data[rec["id"]] = rec

    df = pd.read_parquet(parquet_path)
    test_ids = []
    for _, row in df.iterrows():
        ei = row.get("extra_info", {}) or {}
        tid = str(ei.get("index", "") or ei.get("original_id", "") or "")
        test_ids.append(tid)

    cat_counts = defaultdict(int)
    # (query_snippet, frozenset_of_func_names) -> category
    # Only miss_func has different func set, so this uniquely identifies miss_func
    miss_func_tool_sets = {}  # (query_snippet, frozenset) -> True
    base_tool_sets = {}       # (query_snippet, frozenset) -> True

    for tid in test_ids:
        cat = task_id_to_category(tid)
        cat_counts[cat] += 1
        rec = bfcl_data.get(tid)
        if not rec:
            continue

        questions = rec.get("question", [[]])
        first_msg = questions[0][0] if questions and questions[0] and isinstance(questions[0], list) else {}
        query = first_msg.get("content", "")[:300] if isinstance(first_msg, dict) else ""

        funcs = rec.get("function", [])
        func_names = frozenset(
            f.get("name", "") or f.get("function", {}).get("name", "")
            for f in funcs
        )

        if cat == "multi_turn_miss_func":
            miss_func_tool_sets[(query, func_names)] = True
        else:
            base_tool_sets[(query, func_names)] = True

    return dict(cat_counts), miss_func_tool_sets, base_tool_sets


def classify_record(
    input_text: str,
    miss_func_tool_sets: dict,
    base_tool_sets: dict,
) -> str:
    """Classify a validation record into miss_func or base+param+lc."""
    query = extract_user_query_from_input(input_text)
    tool_names = extract_tool_names_from_input(input_text)

    # Check if this matches a known miss_func signature
    for (mf_query, mf_tools) in miss_func_tool_sets:
        if mf_tools == tool_names and mf_query and mf_query[:100] in input_text:
            return "multi_turn_miss_func"

    # Check if it matches a known base/param/lc signature
    for (b_query, b_tools) in base_tool_sets:
        if b_tools == tool_names and b_query and b_query[:100] in input_text:
            return "base+param+lc"

    # Fallback: if tool count is less than any base signature with the same query,
    # it's likely miss_func
    return "unknown"


# ─────────────────────────────────────────────────────────────────────
# Parse validation JSONL
# ─────────────────────────────────────────────────────────────────────

def parse_val_dir(
    val_dir: Path,
    miss_func_tool_sets: dict,
    base_tool_sets: dict,
) -> Dict[int, Dict[str, List[float]]]:
    step_data = {}

    for f in sorted(val_dir.glob("*.jsonl")):
        try:
            step = int(f.stem)
        except ValueError:
            continue

        cat_scores = defaultdict(list)
        n_matched = 0
        n_total = 0
        has_data_source = False

        with open(f, "r", encoding="utf-8") as fh:
            for line in fh:
                try:
                    rec = json.loads(line)
                except json.JSONDecodeError:
                    continue
                score = rec.get("score", rec.get("reward", None))
                if score is None:
                    continue
                score = float(score)
                n_total += 1

                ds = rec.get("data_source", None)
                if ds is not None:
                    has_data_source = True
                    cat_scores[ds].append(score)
                    n_matched += 1
                else:
                    cat = classify_record(
                        rec.get("input", ""),
                        miss_func_tool_sets,
                        base_tool_sets,
                    )
                    if cat != "unknown":
                        n_matched += 1
                    cat_scores[cat].append(score)

        if step == 0:
            mode = "data_source" if has_data_source else "tool-set matching"
            print(f"  Step {step}: {n_matched}/{n_total} matched via {mode}")

        if cat_scores:
            step_data[step] = dict(cat_scores)

    return step_data


# ─────────────────────────────────────────────────────────────────────
# Display
# ─────────────────────────────────────────────────────────────────────

def print_table(step_data, cat_counts):
    if not step_data:
        print("No validation data found.")
        return

    all_cats = sorted(
        {c for cats in step_data.values() for c in cats if c != "unknown"}
    )
    if not all_cats:
        all_cats = ["unknown"]

    # Check if we have full category breakdown or grouped
    has_grouped = "base+param+lc" in all_cats
    if has_grouped:
        print("NOTE: base / miss_param / long_context cannot be separated from existing")
        print("      validation data. They are grouped as 'base+param+lc'.")
        print("      Re-run with val_only=true after applying the trainer fix for full breakdown.\n")

    col_w = max(28, max((len(c) for c in all_cats), default=20) + 4)
    header = f"{'step':>6}"
    for cat in all_cats:
        header += f"  {cat:>{col_w}}"
    header += f"  {'overall':>{col_w}}"
    print(header)
    print("-" * len(header))

    for step in sorted(step_data.keys()):
        cats = step_data[step]
        row = f"{step:>6}"
        all_scores = []
        for cat in all_cats:
            scores = cats.get(cat, [])
            all_scores.extend(scores)
            if scores:
                n_pass = sum(1 for s in scores if s >= 1.0)
                cell = f"{n_pass/len(scores):.4f} ({n_pass}/{len(scores)})"
            else:
                cell = "—"
            row += f"  {cell:>{col_w}}"

        unknown = cats.get("unknown", [])
        all_scores.extend(unknown)

        if all_scores:
            n_pass_all = sum(1 for s in all_scores if s >= 1.0)
            cell = f"{n_pass_all/len(all_scores):.4f} ({n_pass_all}/{len(all_scores)})"
        else:
            cell = "—"
        row += f"  {cell:>{col_w}}"
        print(row)

    print()
    print(f"Expected test set: {cat_counts}")


def main():
    ap = argparse.ArgumentParser(description="BFCL per-category validation stats")
    ap.add_argument("--val-dir", required=True)
    ap.add_argument("--parquet", required=True, help="Test parquet file")
    ap.add_argument("--bfcl-jsonl", required=True, help="multi_turn_processed.jsonl")
    ap.add_argument("--csv", default="")
    args = ap.parse_args()

    val_dir = Path(args.val_dir)
    if not val_dir.exists():
        print(f"ERROR: {val_dir} not found")
        sys.exit(1)

    print("Building category lookup...")
    cat_counts, miss_func_sets, base_sets = build_lookup(args.parquet, args.bfcl_jsonl)
    print(f"  Expected categories: {cat_counts}")
    print(f"  miss_func signatures: {len(miss_func_sets)}")
    print(f"  base+param+lc signatures: {len(base_sets)}")
    print()

    print("Parsing validation JSONL files...")
    step_data = parse_val_dir(val_dir, miss_func_sets, base_sets)
    print(f"  {len(step_data)} steps found\n")

    print_table(step_data, cat_counts)


if __name__ == "__main__":
    main()
