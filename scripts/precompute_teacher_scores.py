#!/usr/bin/env python3
"""Precompute a teacher policy's per-task success on the training split.

GCCE's Delta_pi(c) estimator needs teacher-policy success at the category
level under the *baseline* environment E_0. Running the teacher online for
every training step is expensive and non-deterministic, so we precompute
once and cache the result as JSON. The trainer then consumes the cache via
:class:`agentevolver.module.gcce.teacher_cache.TeacherCache`.

This script is deliberately decoupled from the training stack: it drives
the env service directly through ``agentevolver.client.env_client`` and an
OpenAI-compatible LLM endpoint (Qwen-Max, GPT, Claude, DeepSeek ...). It
supports three runtime modes:

    1. ``--mode teacher``   : call the remote LLM and score every task.
    2. ``--mode fake``      : emit a deterministic synthetic score so the
                              training pipeline can be smoke-tested without
                              network access.
    3. ``--mode fill-zero`` : emit 0.0 for every task (i.e. Delta_pi=failure-rate,
                              GCCE degrades to PACE).

The default ``--mode fake`` is intended for offline development; switch to
``teacher`` once an LLM endpoint is available.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

import pandas as pd


BFCL_CATEGORY_PREFIXES = (
    "multi_turn_base",
    "multi_turn_miss_func",
    "multi_turn_miss_param",
    "multi_turn_long_context",
)


def _extract_task_id(extra_info) -> Optional[str]:
    if not isinstance(extra_info, dict):
        return None
    return extra_info.get("original_id") or extra_info.get("index")


def _category(task_id: str) -> str:
    for prefix in BFCL_CATEGORY_PREFIXES:
        if task_id.startswith(prefix + "_") or task_id == prefix:
            return prefix
    return "other"


def _fake_score(task_id: str, per_category_mean: dict[str, float]) -> float:
    cat = _category(task_id)
    base = per_category_mean.get(cat, 0.5)
    seed = int(hashlib.md5(task_id.encode("utf-8")).hexdigest()[:8], 16) / 0xFFFFFFFF
    # push around the category mean so per-task variance is non-degenerate
    jitter = (seed - 0.5) * 0.2
    return float(min(1.0, max(0.0, base + jitter)))


def run_fake_mode(task_ids: list[str]) -> dict[str, dict[str, float]]:
    # A reasonable rough prior for a strong teacher on BFCL-v3 multi-turn.
    prior = {
        "multi_turn_base": 0.82,
        "multi_turn_miss_func": 0.55,
        "multi_turn_miss_param": 0.60,
        "multi_turn_long_context": 0.70,
        "other": 0.65,
    }
    out = {}
    for tid in task_ids:
        success = _fake_score(tid, prior)
        out[tid] = {"success": success, "reward": success}
    return out


def run_zero_mode(task_ids: list[str]) -> dict[str, dict[str, float]]:
    return {tid: {"success": 0.0, "reward": 0.0} for tid in task_ids}


def run_teacher_mode(task_ids: list[str], args: argparse.Namespace) -> dict[str, dict[str, float]]:
    """Call an OpenAI-compatible endpoint for each task.

    The teacher is asked to solve the task via the BFCL env, end-to-end.
    We use the env service as the authoritative grader, so results are
    comparable to the training-time reward signal. Implementation kept
    intentionally small; expand with parallelism / retries as needed.
    """
    try:
        from agentevolver.client.env_client import EnvClient  # local import to avoid hard dep at module import
    except Exception as exc:  # pragma: no cover - guarded import
        raise SystemExit(f"teacher mode requires the training stack on PYTHONPATH: {exc}")
    try:
        from openai import OpenAI
    except Exception as exc:  # pragma: no cover
        raise SystemExit(f"pip install openai to use teacher mode: {exc}")

    client = OpenAI(base_url=args.teacher_base_url, api_key=args.teacher_api_key or "sk-unused")
    env_client = EnvClient(args.env_url)

    out: dict[str, dict[str, float]] = {}
    for idx, tid in enumerate(task_ids):
        try:
            messages = env_client.reset(tid)["messages"]
            for _ in range(args.max_turns):
                resp = client.chat.completions.create(
                    model=args.teacher_model,
                    messages=messages,
                    temperature=0.0,
                    max_tokens=args.teacher_max_tokens,
                )
                assistant_msg = resp.choices[0].message
                messages.append({"role": "assistant", "content": assistant_msg.content or ""})
                step = env_client.step(tid, messages)
                messages = step["messages"]
                if step.get("done"):
                    break
            reward = float(env_client.score(tid).get("outcome", 0.0))
            success = 1.0 if reward >= 1.0 else 0.0
        except Exception as exc:
            print(f"[warn] teacher failed on {tid}: {exc}", file=sys.stderr)
            reward = 0.0
            success = 0.0
        out[tid] = {"success": success, "reward": reward}
        if (idx + 1) % 20 == 0:
            print(f"  teacher progress: {idx + 1}/{len(task_ids)}")
    return out


@dataclass
class Args:
    train_parquet: Path
    output: Path
    mode: str
    env_type: str
    teacher_model: str
    teacher_base_url: str
    teacher_api_key: str
    teacher_max_tokens: int
    env_url: str
    max_turns: int


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--train-parquet", required=True, type=Path)
    ap.add_argument("--output", required=True, type=Path)
    ap.add_argument("--mode", choices=["teacher", "fake", "fill-zero"], default="fake")
    ap.add_argument("--env-type", default="bfcl")
    ap.add_argument("--teacher-model", default="qwen-max")
    ap.add_argument("--teacher-base-url", default=os.environ.get("OPENAI_BASE_URL", "https://dashscope.aliyuncs.com/compatible-mode/v1"))
    ap.add_argument("--teacher-api-key", default=os.environ.get("OPENAI_API_KEY", ""))
    ap.add_argument("--teacher-max-tokens", type=int, default=2048)
    ap.add_argument("--env-url", default=os.environ.get("ENV_URL", "http://127.0.0.1:8082"))
    ap.add_argument("--max-turns", type=int, default=20)
    args = ap.parse_args()

    df = pd.read_parquet(args.train_parquet)
    task_ids = [
        str(_extract_task_id(e)) for e in df["extra_info"] if _extract_task_id(e) is not None
    ]
    print(f"loaded {len(task_ids)} task ids from {args.train_parquet}")

    if args.mode == "fake":
        scores = run_fake_mode(task_ids)
    elif args.mode == "fill-zero":
        scores = run_zero_mode(task_ids)
    else:
        scores = run_teacher_mode(task_ids, args)

    out_blob = {
        "meta": {
            "teacher_model": args.teacher_model if args.mode == "teacher" else args.mode,
            "env_type": args.env_type,
            "env_mode": "baseline",
            "num_tasks": len(task_ids),
            "created_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        },
        "scores": scores,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(out_blob, f, ensure_ascii=False, indent=2)

    cat_totals: dict[str, list[float]] = {}
    for tid, sc in scores.items():
        cat_totals.setdefault(_category(tid), []).append(sc["success"])
    print("category success means:")
    for cat, values in cat_totals.items():
        print(f"  {cat:<28s} mean={sum(values) / len(values):.3f}  n={len(values)}")
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
