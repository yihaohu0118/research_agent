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
# Env-format alignment (CRITICAL)
# ---------------------------------------------------------------------------
# The SFT warm-start has to see the **same** system prompt + tool-result
# formatting that the BFCL env streams to the model at rollout/eval
# time. Otherwise the policy learns a format that does not exist at
# deployment and validation score collapses to 0.
#
# Both pieces below are a **frozen copy** of:
#   env_service/environments/bfcl/bfcl_env.py
#     - ``T3RL_BFCL_SYSTEM_PROMPT`` (module top)
#     - ``tools_schema_to_qwen_prompt(..., prompt_mode="t3rl_text")``
#     - ``tool_message_to_qwen_text(..., result_mode="plain_user")``
#
# If that file changes, mirror the change here or the SFT pool will
# start drifting again. We intentionally do *not* import from
# env_service because this script is a purely offline data prep tool
# and env_service pulls in the whole BFCL eval stack.
_T3RL_BFCL_SYSTEM_PROMPT = """You are an expert in composing functions. You are given a question and a set of possible functions. Based on the question, you will need to make one or more function/tool calls to achieve the purpose.

Your response must always start with your step-by-step reasoning process enclosed in `<thinking></thinking>` XML tags. This is for you to outline your plan and justify your chosen actions.

After the thinking block, you will perform one of the following actions:

1. If tool calls are necessary: For each function call, return a json object with function name and arguments within <tool_call></tool_call> XML tags.
Examples:
<tool_call>\n{"name":"func1", "arguments":{...}}\n</tool_call>\n<tool_call>\n{"name":"func2", "arguments":{...}}\n</tool_call>

2. If no tool calls are necessary or possible: Directly provide a user-facing response in plain text. This applies if none of the functions can be used, or if the given question lacks the parameters required by the function.

At each turn, you should try your best to complete the tasks requested by the user within the current turn. Continue to output functions to call until you have fulfilled the user's request to the best of your ability. Once you have no more functions to call, the system will consider the current turn complete and proceed to the next turn or task."""


def _tools_schema_to_qwen_prompt_t3rl(tools_schema: list[dict[str, Any]]) -> str:
    """Inlined mirror of ``tools_schema_to_qwen_prompt(..., "t3rl_text")``.

    Produces the exact text the BFCL env puts in the ``system`` message
    for every rollout with ``tool_prompt_mode=t3rl_text``.
    """
    if not tools_schema:
        return _T3RL_BFCL_SYSTEM_PROMPT.strip()
    lines: list[str] = [_T3RL_BFCL_SYSTEM_PROMPT.strip()]
    lines.append("\n\n# Tools\n")
    lines.append(
        "You are provided with function signatures within <tools></tools> XML tags:"
    )
    lines.append("<tools>")
    for tool in tools_schema:
        lines.append(
            json.dumps(tool, ensure_ascii=False, separators=(",", ":"))
        )
    lines.append("</tools>\n")
    return "\n".join(lines)


def _tool_result_to_plain_user_text(
    tool_name: str, result_payload: Any
) -> str:
    """Mirror of env's ``tool_message_to_qwen_text(..., "plain_user")``.

    Returns the ``content`` body the env hands back as a ``user`` role
    message after the assistant's tool_call block. We keep the trailing
    newline behaviour identical (env joins entries with ``\\n`` and
    appends ``\\n``) for token-level parity with real rollouts.
    """
    if isinstance(result_payload, str):
        content_text = result_payload
    else:
        content_text = json.dumps(result_payload, ensure_ascii=False)
    return f"Tool result from {tool_name}:\n{content_text}\n"


# ---------------------------------------------------------------------------
# Tool schema extraction: try parquet, fall back to env's JSONL
# ---------------------------------------------------------------------------
def _extract_tools_from_row(row: pd.Series) -> list[dict[str, Any]]:
    """Best-effort tools/function schema extraction from a parquet row.

    Tries the most likely keys in order. Returns [] if nothing is found;
    the caller can then use the BFCL-JSONL fallback.
    """
    def _normalise(value: Any) -> list[dict[str, Any]] | None:
        if value is None:
            return None
        if hasattr(value, "tolist"):
            value = value.tolist()
        if isinstance(value, (list, tuple)):
            out = [dict(x) for x in value if isinstance(x, dict)]
            return out if out else None
        return None

    for container_key in ("extra_info", "reward_model"):
        container = row.get(container_key)
        if isinstance(container, dict):
            for key in ("function", "functions", "tools", "tool_schema"):
                v = _normalise(container.get(key))
                if v:
                    return v
            inter = container.get("interaction_kwargs")
            if isinstance(inter, dict):
                for key in ("function", "functions", "tools", "tool_schema"):
                    v = _normalise(inter.get(key))
                    if v:
                        return v

    for key in ("function", "functions", "tools", "tool_schema"):
        v = _normalise(row.get(key))
        if v:
            return v

    return []


def _load_bfcl_jsonl_index(path: Path | None) -> dict[str, dict[str, Any]]:
    """Build an ``id -> record`` map from the env's BFCL JSONL.

    Matches the env's own loader: one JSON object per line keyed by
    ``id``. Used as the canonical fallback for the tools schema when
    parquet rows do not carry it.
    """
    if path is None:
        return {}
    if not path.exists():
        print(f"[demo_pool] warn: --bfcl-jsonl path does not exist: {path}", file=sys.stderr)
        return {}
    index: dict[str, dict[str, Any]] = {}
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except Exception:
                continue
            tid = rec.get("id") or rec.get("task_id")
            if tid:
                index[str(tid)] = rec
    print(f"[demo_pool] loaded {len(index)} BFCL JSONL records from {path}")
    return index


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

    Matches the **env's** ``tool_message_to_qwen_text(result_mode="plain_user")``
    format exactly so that at SFT time the model sees the same boundary
    tokens between its own assistant output and the subsequent user
    message as it does at rollout time. Divergence here is what killed
    the previous SFT run: the env emits
    ``"Tool result from <name>:\\n<content>\\n"`` between turns, and any
    other shape (``[TOOL_RESULTS]\\n...``, JSON-wrapped, etc.) makes the
    policy learn a distribution that simply does not exist at
    deployment.

    BFCL multi-turn replays *one* user message per turn that
    concatenates all tool results, so we concatenate all gold tool
    calls' stub results here as well.
    """
    bodies: list[str] = []
    for expr in gold_calls:
        payload = _parse_tool_call_source(expr)
        bodies.append(
            _tool_result_to_plain_user_text(payload["name"], PLACEHOLDER_TOOL_RESULT)
        )
    return "".join(bodies).rstrip("\n")


# ---------------------------------------------------------------------------
# Main builder
# ---------------------------------------------------------------------------
def build_demo_for_row(
    row: pd.Series,
    bfcl_index: dict[str, dict[str, Any]] | None = None,
) -> dict[str, Any] | None:
    gold_turns = _extract_ground_truth_per_turn(row)
    if not gold_turns:
        return None

    user_questions = _extract_turn_questions(row)
    category = _extract_category(row)
    task_id = _extract_task_id(row)

    # Tools schema: try parquet first, then the env's JSONL index, then
    # give up and use a tool-less system prompt (still coherent --
    # matches env's own behaviour when tools_schema is empty).
    tools = _extract_tools_from_row(row)
    if not tools and bfcl_index is not None:
        rec = bfcl_index.get(task_id)
        if rec is not None:
            tools = rec.get("function") or rec.get("functions") or []
            if hasattr(tools, "tolist"):
                tools = tools.tolist()
            tools = [dict(t) for t in tools if isinstance(t, dict)]

    # Compose the system prompt via the env's own recipe. This is the
    # text the BFCL env feeds the policy at rollout/eval time via
    # ``BfclEnv.get_init_state`` -> ``tools_schema_to_qwen_prompt(...,
    # prompt_mode="t3rl_text")``. SFT MUST see the same string or the
    # policy goes out of distribution and val score collapses to 0.
    system_prompt = _tools_schema_to_qwen_prompt_t3rl(tools)

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
        "num_tools": len(tools),
    }


def build_pool(
    rows: Iterable[pd.Series],
    max_tasks: int | None = None,
    bfcl_index: dict[str, dict[str, Any]] | None = None,
) -> dict[str, dict[str, Any]]:
    pool: dict[str, dict[str, Any]] = {}
    n_total = 0
    n_skipped = 0
    n_no_tools = 0
    for row in rows:
        n_total += 1
        demo = build_demo_for_row(row, bfcl_index=bfcl_index)
        if demo is None:
            n_skipped += 1
            continue
        if demo.get("num_tools", 0) == 0:
            n_no_tools += 1
        pool[demo["task_id"]] = demo
        if max_tasks is not None and len(pool) >= max_tasks:
            break

    print(
        f"[demo_pool] processed {n_total} rows, built {len(pool)} demos, "
        f"skipped {n_skipped} (no ground_truth), "
        f"{n_no_tools} without tools schema."
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
    parser.add_argument(
        "--bfcl-jsonl",
        default=None,
        help=(
            "Optional path to the BFCL env's canonical multi-turn JSONL "
            "(e.g. ./bfcl_data/multiturn_data.jsonl). Used as a fallback "
            "when the parquet rows do not carry a tools/function schema. "
            "This is the same file the env loads at runtime via "
            "BFCL_DATA_PATH, so the tools schema matches bit-for-bit."
        ),
    )
    args = parser.parse_args()

    train_path = Path(args.train_parquet)
    if not train_path.exists():
        print(f"[demo_pool] error: train parquet not found at {train_path}", file=sys.stderr)
        return 1

    df = pd.read_parquet(train_path)
    print(f"[demo_pool] loaded {len(df)} rows from {train_path}")

    bfcl_index = _load_bfcl_jsonl_index(
        Path(args.bfcl_jsonl) if args.bfcl_jsonl else None
    )

    pool = build_pool(_iter_rows(df), max_tasks=args.max_tasks, bfcl_index=bfcl_index)

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(pool, f, ensure_ascii=False, indent=2)

    print(f"[demo_pool] wrote {len(pool)} demos to {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
