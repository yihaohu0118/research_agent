#!/usr/bin/env python3
"""Build a GCCE teacher cache from AgentEvolver validation JSONL logs.

The intended flow is:

  1. Run a strong teacher model in ``trainer.val_only=true`` mode on the BFCL
     training split.
  2. Convert the resulting validation JSONL file(s) into the JSON cache consumed
     by ``agentevolver.module.gcce.teacher_cache.TeacherCache``.

Each validation record must contain ``task_id``. Newer trainer code writes this
field automatically. If it is missing, rerun validation with the current code.
"""
from __future__ import annotations

import argparse
import json
import time
from collections import defaultdict
from pathlib import Path
from typing import Any


BFCL_CATEGORY_PREFIXES = (
    "multi_turn_base",
    "multi_turn_miss_func",
    "multi_turn_miss_param",
    "multi_turn_long_context",
)


def task_id_to_category(task_id: str) -> str:
    for prefix in BFCL_CATEGORY_PREFIXES:
        if task_id == prefix or task_id.startswith(prefix + "_"):
            return prefix
    return task_id.rsplit("_", 1)[0] if "_" in task_id else "other"


def iter_validation_records(val_dir: Path):
    files = sorted(val_dir.glob("*.jsonl"))
    if not files:
        raise SystemExit(f"No validation JSONL files found under {val_dir}")

    for path in files:
        with path.open("r", encoding="utf-8") as f:
            for line_no, line in enumerate(f, start=1):
                line = line.strip()
                if not line:
                    continue
                try:
                    record = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise SystemExit(f"Bad JSON in {path}:{line_no}: {exc}") from exc
                yield path, line_no, record


def score_from_record(record: dict[str, Any]) -> float:
    value = record.get("score", record.get("reward", None))
    if value is None:
        raise ValueError("record has neither 'score' nor 'reward'")
    return float(value)


def main() -> None:
    parser = argparse.ArgumentParser(description="Build GCCE teacher cache from validation logs")
    parser.add_argument("--val-dir", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--teacher-model", required=True)
    parser.add_argument("--env-type", default="bfcl")
    parser.add_argument("--env-mode", default="baseline")
    parser.add_argument("--score-threshold", type=float, default=1.0)
    parser.add_argument(
        "--source-split",
        default="train",
        help="Free-form split label stored in cache metadata.",
    )
    args = parser.parse_args()

    by_task: dict[str, list[float]] = defaultdict(list)
    missing_task_id = 0
    skipped_no_score = 0

    for path, line_no, record in iter_validation_records(args.val_dir):
        task_id = record.get("task_id")
        if not task_id:
            missing_task_id += 1
            continue
        try:
            score = score_from_record(record)
        except (TypeError, ValueError):
            skipped_no_score += 1
            continue
        by_task[str(task_id)].append(score)

    if missing_task_id:
        raise SystemExit(
            f"{missing_task_id} validation records are missing task_id. "
            "Rerun validation with the current trainer before building a teacher cache."
        )
    if not by_task:
        raise SystemExit(
            f"No scored validation records found under {args.val_dir} "
            f"(skipped_no_score={skipped_no_score})."
        )

    scores = {}
    category_values: dict[str, list[float]] = defaultdict(list)
    for task_id, values in sorted(by_task.items()):
        reward = sum(values) / len(values)
        success = sum(1.0 for v in values if v >= args.score_threshold) / len(values)
        scores[task_id] = {
            "success": float(success),
            "reward": float(reward),
        }
        category_values[task_id_to_category(task_id)].append(success)

    blob = {
        "meta": {
            "teacher_model": args.teacher_model,
            "env_type": args.env_type,
            "env_mode": args.env_mode,
            "source_split": args.source_split,
            "source_val_dir": str(args.val_dir),
            "num_tasks": len(scores),
            "score_threshold": args.score_threshold,
            "created_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        },
        "scores": scores,
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as f:
        json.dump(blob, f, ensure_ascii=False, indent=2)

    print(f"loaded tasks: {len(scores)}")
    print("category success means:")
    for category in sorted(category_values):
        values = category_values[category]
        passed = sum(values)
        print(f"  {category:<28s} mean={sum(values) / len(values):.4f}  pass={passed:.1f}/{len(values)}")
    if skipped_no_score:
        print(f"skipped records without score/reward: {skipped_no_score}")
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
