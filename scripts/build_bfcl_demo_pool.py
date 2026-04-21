"""Build a per-task demonstration pool from BFCL ground-truth tool calls.

This is the offline data-preparation step for **D-Patch** (demonstration
patch), the new environment-side primitive used by the CoEvo-D
(Demonstration-gated Co-Evolution) algorithm.

For every task in a BFCL train parquet file we render its
``reward_model.ground_truth`` (a list of per-turn gold tool-call strings)
into a full chat-formatted trajectory:

  system (tool catalogue)
  user    (turn-1 question)
  assistant (gold tool_calls for turn-1, wrapped in <thinking>/<tool_call>)
  user    (tool results, stubbed with a minimal placeholder)
  user    (turn-2 question)
  assistant (gold tool_calls for turn-2)
  ...

The output is a JSON file keyed by task_id, each value containing

  {
    "task_id":        str,
    "category":       str,   # multi_turn_base / miss_param / miss_func / long_context
    "messages":       [ {role, content}, ... ],   # full trajectory text
    "assistant_spans":[ (start_msg_idx, end_msg_idx), ... ],  # demo loss-mask hints
    "reward":         1.0,
  }

Why offline?
------------
Building demos at train time would require either (a) a synchronous call
into the env service to obtain real tool_results for the gold actions, or
(b) tokenising gold trajectories on the driver process every step. Both
add latency to the tightest loop in the trainer. A one-off offline dump
costs at most a few minutes and is reused for the whole training run.

Why placeholder tool_results?
-----------------------------
The policy side only needs loss on **assistant** tokens. Tool results are
masked out by the context manager's loss_mask regardless of content, so a
stable ``{"status": "ok"}`` placeholder is sufficient for gradient quality.
For correctness of downstream *message rendering* we still include
plausible JSON strings so tokenisation does not blow up on empty payloads.

Usage
-----
    python scripts/build_bfcl_demo_pool.py \\
      --train-parquet data/bfcl_train_400.parquet \\
      --output        data/bfcl_demo_pool_400.json

    # Optional: sub-sample a smaller pool for smoke tests
    python scripts/build_bfcl_demo_pool.py \\
      --train-parquet data/bfcl_train_400.parquet \\
      --output        data/bfcl_demo_pool_smoke.json \\
      --max-tasks     32
"""
from __future__ import annotations

import argparse
import json
import sys
import uuid
from pathlib import Path
from typing import Any, Iterable

import pandas as pd


PLACEHOLDER_TOOL_RESULT = {"status": "ok"}


# ---------------------------------------------------------------------------
# Ground-truth flattening
# ---------------------------------------------------------------------------
def _as_list(value: Any) -> list:
    """Normalise numpy/pandas arrays and singletons to a plain Python list."""
    if value is None:
        return []
    if isinstance(value, str):
        return [value]
    if hasattr(value, "tolist"):
        return list(value.tolist())
    if isinstance(value, (list, tuple)):
        return list(value)
    return [value]


def _extract_ground_truth_per_turn(row: pd.Series) -> list[list[str]]:
    """Pull ground_truth out of either ``reward_model`` or ``extra_info``.

    BFCL-style parquet files stored by this repo put the gold action
    sequence under two keys redundantly. The canonical shape is a list of
    *turn-level* lists, each containing Python-source strings of the form
    ``"tool_name(arg=value)"``.
    """
    gt_from_reward = None
    if isinstance(row.get("reward_model"), dict):
        gt_from_reward = row["reward_model"].get("ground_truth")

    gt_from_extra = None
    if isinstance(row.get("extra_info"), dict):
        interaction = row["extra_info"].get("interaction_kwargs")
        if isinstance(interaction, dict):
            gt_from_extra = interaction.get("ground_truth")

    raw = gt_from_reward if gt_from_reward is not None else gt_from_extra
    if raw is None:
        return []

    outer = _as_list(raw)
    turns: list[list[str]] = []
    for item in outer:
        inner = _as_list(item)
        if not inner:
            continue
        turns.append([str(step) for step in inner])
    return turns


def _extract_turn_questions(row: pd.Series) -> list[str]:
    """Pull per-turn user messages. Falls back to prompt's first user msg."""
    questions: list[str] = []
    if isinstance(row.get("extra_info"), dict):
        raw_q = row["extra_info"].get("question")
        for item in _as_list(raw_q):
            inner = _as_list(item)
            for msg in inner:
                if isinstance(msg, dict) and msg.get("role") == "user":
                    content = msg.get("content", "")
                    if content:
                        questions.append(str(content))

    if questions:
        return questions

    prompt = _as_list(row.get("prompt"))
    for msg in prompt:
        if isinstance(msg, dict) and msg.get("role") == "user":
            questions.append(str(msg.get("content", "")))
    return questions


def _extract_system_prompt(row: pd.Series) -> str:
    prompt = _as_list(row.get("prompt"))
    for msg in prompt:
        if isinstance(msg, dict) and msg.get("role") == "system":
            return str(msg.get("content", ""))
    return ""


def _extract_category(row: pd.Series) -> str:
    if isinstance(row.get("extra_info"), dict):
        cat = row["extra_info"].get("dataset_type")
        if cat:
            return str(cat)
    data_source = row.get("data_source")
    if data_source:
        return str(data_source)
    return "unknown"


def _extract_task_id(row: pd.Series) -> str:
    if isinstance(row.get("extra_info"), dict):
        for key in ("index", "original_id"):
            v = row["extra_info"].get(key)
            if v:
                return str(v)
    if "task_id" in row and row["task_id"]:
        return str(row["task_id"])
    return f"task_{uuid.uuid4().hex[:8]}"


# ---------------------------------------------------------------------------
# Tool-call string -> chat-completion payload
# ---------------------------------------------------------------------------
def _parse_tool_call_source(expr: str) -> dict:
    """Parse a BFCL ground-truth expression like ``foo(x=1, y='abc')``.

    This mirrors what the BFCL env handler does for parsing. For robustness
    we fall back to a best-effort split so that we never drop a gold step;
    even a malformed parse gets serialised as a raw string so the assistant
    learns the *form* of the call.
    """
    expr = expr.strip()
    try:
        import ast

        tree = ast.parse(expr, mode="eval")
        if not isinstance(tree.body, ast.Call):
            raise ValueError(f"Not a call expression: {expr!r}")
        call = tree.body

        if isinstance(call.func, ast.Name):
            name = call.func.id
        elif isinstance(call.func, ast.Attribute):
            name = call.func.attr
        else:
            name = ast.unparse(call.func)

        args: dict[str, Any] = {}
        for kw in call.keywords:
            try:
                args[kw.arg] = ast.literal_eval(kw.value)
            except Exception:
                args[kw.arg] = ast.unparse(kw.value)
        return {"name": name, "arguments": args}
    except Exception:
        return {"name": "_raw", "arguments": {"expr": expr}}


def _render_assistant_turn(gold_calls: list[str]) -> str:
    """Render one assistant turn in the ``llama3_json`` / t3rl_text format.

    Matches the format the env service expects, so that the tokeniser's
    special-token boundaries line up with real rollouts.
    """
    thinking = (
        "<thinking>I will invoke the required tool(s) to satisfy the "
        "user's request deterministically.</thinking>"
    )
    tool_blocks = []
    for expr in gold_calls:
        payload = _parse_tool_call_source(expr)
        tool_blocks.append(
            "<tool_call>\n" + json.dumps(payload, ensure_ascii=False) + "\n</tool_call>"
        )
    return thinking + "\n" + "\n".join(tool_blocks)


def _render_tool_result_turn(gold_calls: list[str]) -> str:
    """Render a synthetic user-role tool-result block.

    BFCL multi-turn replays *one* user message per turn that contains the
    concatenated stringified tool results. For a demonstration trajectory
    the exact result doesn't matter because loss is masked out on these
    tokens; we just need tokenisation to succeed and the shape to look
    plausible so that multi-turn structure is preserved.
    """
    bodies = []
    for expr in gold_calls:
        payload = _parse_tool_call_source(expr)
        bodies.append(
            json.dumps(
                {"tool": payload["name"], "result": PLACEHOLDER_TOOL_RESULT},
                ensure_ascii=False,
            )
        )
    return "[TOOL_RESULTS]\n" + "\n".join(bodies)


# ---------------------------------------------------------------------------
# Main builder
# ---------------------------------------------------------------------------
def build_demo_for_row(row: pd.Series) -> dict[str, Any] | None:
    gold_turns = _extract_ground_truth_per_turn(row)
    if not gold_turns:
        return None

    user_questions = _extract_turn_questions(row)
    system_prompt = _extract_system_prompt(row)
    category = _extract_category(row)
    task_id = _extract_task_id(row)

    while len(user_questions) < len(gold_turns):
        user_questions.append("(continued)")

    messages: list[dict[str, Any]] = []
    assistant_spans: list[tuple[int, int]] = []

    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})

    for turn_idx, gold_calls in enumerate(gold_turns):
        if not gold_calls:
            continue
        messages.append(
            {"role": "user", "content": user_questions[turn_idx]}
        )
        asst_start = len(messages)
        messages.append(
            {
                "role": "assistant",
                "content": _render_assistant_turn(gold_calls),
            }
        )
        asst_end = len(messages)
        assistant_spans.append((asst_start, asst_end))

        if turn_idx != len(gold_turns) - 1:
            messages.append(
                {
                    "role": "user",
                    "content": _render_tool_result_turn(gold_calls),
                }
            )

    return {
        "task_id": task_id,
        "category": category,
        "messages": messages,
        "assistant_spans": assistant_spans,
        "reward": 1.0,
        "num_turns": len(gold_turns),
        "num_calls": sum(len(t) for t in gold_turns),
    }


def build_pool(
    rows: Iterable[pd.Series], max_tasks: int | None = None
) -> dict[str, dict[str, Any]]:
    pool: dict[str, dict[str, Any]] = {}
    n_total = 0
    n_skipped = 0
    for row in rows:
        n_total += 1
        demo = build_demo_for_row(row)
        if demo is None:
            n_skipped += 1
            continue
        pool[demo["task_id"]] = demo
        if max_tasks is not None and len(pool) >= max_tasks:
            break

    print(
        f"[demo_pool] processed {n_total} rows, built {len(pool)} demos, "
        f"skipped {n_skipped} (no ground_truth)."
    )
    by_cat: dict[str, int] = {}
    for v in pool.values():
        by_cat[v["category"]] = by_cat.get(v["category"], 0) + 1
    for cat, cnt in sorted(by_cat.items()):
        print(f"  category={cat:30s}  demos={cnt}")
    return pool


def _iter_rows(df: pd.DataFrame) -> Iterable[pd.Series]:
    for _, row in df.iterrows():
        yield row


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--train-parquet",
        required=True,
        help="Path to BFCL train parquet (e.g. data/bfcl_train_400.parquet).",
    )
    parser.add_argument(
        "--output",
        required=True,
        help="Output JSON file mapping task_id -> demo spec.",
    )
    parser.add_argument(
        "--max-tasks",
        type=int,
        default=None,
        help="Cap the number of demos for smoke tests.",
    )
    args = parser.parse_args()

    train_path = Path(args.train_parquet)
    if not train_path.exists():
        print(f"[demo_pool] error: train parquet not found at {train_path}", file=sys.stderr)
        return 1

    df = pd.read_parquet(train_path)
    print(f"[demo_pool] loaded {len(df)} rows from {train_path}")

    pool = build_pool(_iter_rows(df), max_tasks=args.max_tasks)

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(pool, f, ensure_ascii=False, indent=2)

    print(f"[demo_pool] wrote {len(pool)} demos to {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
