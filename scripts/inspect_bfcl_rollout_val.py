#!/usr/bin/env python3
"""Inspect BFCL validation and rollout logs for collapse modes.

The trainer writes two useful views:

* validation_log/<step>.jsonl and rollout_log/<step>.jsonl contain decoded
  prompt/output/score rows.
* rollout_log/traj_<step>.jsonl contains parsed environment trajectories with
  reward metadata, including BFCL failure tags when the dense diagnostic grader
  is enabled.

This script is intentionally dependency-free so it can be run directly on the
training box.
"""
from __future__ import annotations

import argparse
import json
import re
from collections import Counter, defaultdict
from pathlib import Path
from statistics import mean
from typing import Any, Iterable


BFCL_CATEGORY_PREFIXES = (
    "multi_turn_base",
    "multi_turn_miss_func",
    "multi_turn_miss_param",
    "multi_turn_long_context",
)


def read_jsonl(path: Path) -> Iterable[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                value = json.loads(line)
            except json.JSONDecodeError as exc:
                print(f"  WARN: {path}:{line_no} JSON decode failed: {exc}")
                continue
            if isinstance(value, dict):
                yield value


def short(text: Any, limit: int) -> str:
    value = "" if text is None else str(text)
    value = value.replace("\n", "\\n")
    if len(value) <= limit:
        return value
    return value[:limit] + "..."


def score_of(record: dict[str, Any]) -> float | None:
    for key in ("score", "reward"):
        value = record.get(key)
        if isinstance(value, dict):
            value = value.get("outcome", value.get("reward_value"))
        if value is None:
            continue
        try:
            return float(value)
        except (TypeError, ValueError):
            continue
    reward = record.get("reward")
    if isinstance(reward, dict):
        try:
            return float(reward.get("outcome"))
        except (TypeError, ValueError):
            return None
    return None


def task_id_to_category(task_id: Any) -> str:
    tid = str(task_id or "")
    for prefix in BFCL_CATEGORY_PREFIXES:
        if tid == prefix or tid.startswith(prefix + "_"):
            return prefix
    return tid.rsplit("_", 1)[0] if "_" in tid else "unknown"


def category_of(record: dict[str, Any]) -> str:
    for key in ("data_source", "category"):
        value = record.get(key)
        if value:
            return str(value)
    task_id = record.get("task_id")
    if not task_id:
        metadata = record.get("metadata")
        if isinstance(metadata, dict):
            task_id = metadata.get("task_id") or metadata.get("id")
            if metadata.get("category"):
                return str(metadata["category"])
    return task_id_to_category(task_id)


def strip_code_fence(text: str) -> str:
    stripped = text.strip()
    if not stripped.startswith("```"):
        return stripped
    lines = stripped.splitlines()
    if lines and lines[0].strip().startswith("```"):
        lines = lines[1:]
    if lines and lines[-1].strip().startswith("```"):
        lines = lines[:-1]
    return "\n".join(lines).strip()


def output_kind(text: Any) -> str:
    raw = "" if text is None else str(text)
    stripped = raw.strip()
    if not stripped:
        return "empty"
    if "[ERROR]" in stripped:
        return "error_text"
    if "<tool_call" in stripped or "</tool_call>" in stripped:
        return "xml_tool_call"
    if "<|python_tag|>" in stripped:
        return "llama_python_tag"
    cleaned = strip_code_fence(stripped.replace("<|python_tag|>", ""))
    if cleaned.startswith("{") or cleaned.startswith("["):
        try:
            payload = json.loads(cleaned)
            if isinstance(payload, dict):
                return "raw_json_object"
            if isinstance(payload, list):
                if any(isinstance(x, dict) for x in payload):
                    return "raw_json_list"
                return "json_list_other"
        except json.JSONDecodeError:
            pass
        if re.search(r"[A-Za-z_][\w.]*\s*\(", cleaned):
            return "toolace_call_list"
        return "jsonish_invalid"
    if re.match(r"^[A-Za-z_][\w.]*\s*\(", cleaned):
        return "toolace_call"
    if cleaned.startswith("<think") or cleaned.startswith("<thinking"):
        return "thinking_text"
    return "plain_text"


def summarize_scores(records: list[dict[str, Any]]) -> tuple[int, int, float, float]:
    scores = [score for score in (score_of(record) for record in records) if score is not None]
    if not scores:
        return len(records), 0, 0.0, 0.0
    passed = sum(1 for score in scores if score >= 1.0)
    return len(scores), passed, passed / len(scores), mean(scores)


def format_counter(counter: Counter[str], top_k: int = 6) -> str:
    if not counter:
        return "-"
    return ", ".join(f"{key}={value}" for key, value in counter.most_common(top_k))


def step_files(directory: Path, prefix: str = "") -> list[tuple[int, Path]]:
    files: list[tuple[int, Path]] = []
    if not directory.exists():
        return files
    for path in directory.glob(f"{prefix}*.jsonl"):
        stem = path.stem
        if prefix and stem.startswith(prefix):
            stem = stem[len(prefix) :]
        try:
            step = int(stem)
        except ValueError:
            continue
        files.append((step, path))
    return sorted(files)


def choose_steps(files: list[tuple[int, Path]], selector: str) -> list[tuple[int, Path]]:
    if selector == "all":
        return files
    if selector == "last":
        return files[-1:] if files else []
    wanted = {int(part) for part in selector.split(",") if part.strip()}
    return [(step, path) for step, path in files if step in wanted]


def inspect_generation_dir(
    title: str,
    directory: Path,
    *,
    selector: str,
    examples: int,
    chars: int,
) -> None:
    files = choose_steps(step_files(directory), selector)
    print(f"\n{title}: {directory}")
    if not files:
        print("  NOT FOUND or no <step>.jsonl files")
        return

    for step, path in files:
        records = list(read_jsonl(path))
        n, passed, pass_rate, score_mean = summarize_scores(records)
        by_kind = Counter(output_kind(record.get("output", "")) for record in records)
        by_category: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for record in records:
            by_category[category_of(record)].append(record)
        cat_text = []
        for cat, cat_records in sorted(by_category.items()):
            cat_n, cat_passed, cat_rate, _ = summarize_scores(cat_records)
            cat_text.append(f"{cat}:{cat_rate:.3f}({cat_passed}/{cat_n})")
        print(
            f"  step {step:>5}: pass={pass_rate:.4f} ({passed}/{n}) "
            f"mean_score={score_mean:.4f}"
        )
        print(f"           output_kind: {format_counter(by_kind)}")
        print(f"           category: {'; '.join(cat_text) if cat_text else '-'}")

        failed = [
            record
            for record in records
            if (score_of(record) is not None and float(score_of(record) or 0.0) < 1.0)
        ]
        for idx, record in enumerate(failed[:examples], start=1):
            print(
                f"           bad#{idx}: score={score_of(record)} "
                f"cat={category_of(record)} kind={output_kind(record.get('output', ''))}"
            )
            print(f"             input : {short(record.get('input', ''), chars)}")
            print(f"             output: {short(record.get('output', ''), chars)}")


def iter_messages(traj: dict[str, Any]) -> Iterable[dict[str, Any]]:
    steps = traj.get("steps") or []
    if not isinstance(steps, list):
        return []
    return (step for step in steps if isinstance(step, dict))


def reward_metadata(traj: dict[str, Any]) -> dict[str, Any]:
    reward = traj.get("reward") or {}
    if not isinstance(reward, dict):
        return {}
    metadata = reward.get("metadata") or {}
    return metadata if isinstance(metadata, dict) else {}


def failure_tags(traj: dict[str, Any]) -> list[str]:
    metadata = reward_metadata(traj)
    progress_info = metadata.get("bfcl_dense_progress_info") or {}
    if isinstance(progress_info, dict):
        tags = progress_info.get("failure_tags") or []
        if isinstance(tags, list):
            return [str(tag) for tag in tags if tag]
    return []


def inspect_traj_dir(
    directory: Path,
    *,
    selector: str,
    examples: int,
    chars: int,
) -> None:
    files = choose_steps(step_files(directory, prefix="traj_"), selector)
    print(f"\nRollout trajectories: {directory}")
    if not files:
        print("  NOT FOUND or no traj_<step>.jsonl files")
        return

    for step, path in files:
        trajectories = list(read_jsonl(path))
        n, passed, pass_rate, score_mean = summarize_scores(trajectories)
        tag_counter: Counter[str] = Counter()
        role_counter: Counter[str] = Counter()
        assistant_kind: Counter[str] = Counter()
        rejected_count = 0
        parse_error_count = 0
        tool_error_count = 0
        terminated_count = 0
        assistant_turns = 0
        tool_call_count = 0

        for traj in trajectories:
            if bool(traj.get("is_terminated") or traj.get("done")):
                terminated_count += 1
            tag_counter.update(failure_tags(traj))
            for msg in iter_messages(traj):
                role = str(msg.get("role", "unknown"))
                role_counter[role] += 1
                content = msg.get("content", "")
                if role == "assistant":
                    assistant_turns += 1
                    assistant_kind[output_kind(content)] += 1
                calls = msg.get("tool_calls") or []
                if isinstance(calls, list):
                    tool_call_count += len(calls)
                if msg.get("_bfcl_rejected_tool_calls"):
                    rejected_count += 1
                text = str(content or "")
                if "Invalid tool call format" in text or "parse_error" in text:
                    parse_error_count += 1
                if "[ERROR]" in text or '"error"' in text or "'error'" in text:
                    tool_error_count += 1

        print(
            f"  step {step:>5}: success={pass_rate:.4f} ({passed}/{n}) "
            f"mean_score={score_mean:.4f} terminated={terminated_count}/{n}"
        )
        print(
            f"           assistant_turns={assistant_turns} tool_calls={tool_call_count} "
            f"rejected_msgs={rejected_count} parse_err_msgs={parse_error_count} "
            f"tool_err_msgs={tool_error_count}"
        )
        print(f"           roles: {format_counter(role_counter)}")
        print(f"           assistant_content_kind: {format_counter(assistant_kind)}")
        print(f"           failure_tags: {format_counter(tag_counter, top_k=10)}")

        bad = [
            traj
            for traj in trajectories
            if (score_of(traj) is not None and float(score_of(traj) or 0.0) < 1.0)
        ]
        for idx, traj in enumerate(bad[:examples], start=1):
            print(
                f"           bad_traj#{idx}: score={score_of(traj)} "
                f"task={traj.get('data_id') or (traj.get('metadata') or {}).get('task_id')} "
                f"tags={failure_tags(traj)}"
            )
            for msg in list(iter_messages(traj))[-6:]:
                role = msg.get("role", "unknown")
                calls = msg.get("tool_calls") or []
                calls_text = f" calls={len(calls)}" if isinstance(calls, list) and calls else ""
                print(f"             {role}{calls_text}: {short(msg.get('content', ''), chars)}")


def inspect_experiment(root: Path, exp: str, args: argparse.Namespace) -> None:
    exp_dir = root / exp
    print(f"\n================ {exp} ================")
    inspect_generation_dir(
        "Validation generations",
        exp_dir / "validation_log",
        selector=args.step,
        examples=args.examples,
        chars=args.chars,
    )
    inspect_generation_dir(
        "Train rollout generations",
        exp_dir / "rollout_log",
        selector=args.step,
        examples=args.examples,
        chars=args.chars,
    )
    inspect_traj_dir(
        exp_dir / "rollout_log",
        selector=args.step,
        examples=args.examples,
        chars=args.chars,
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Inspect BFCL validation/rollout logs by step."
    )
    parser.add_argument(
        "--root",
        default="experiments/tech_synthetic",
        help="Experiment root containing <experiment_name>/validation_log.",
    )
    parser.add_argument(
        "--exp",
        action="append",
        default=[],
        help="Experiment name under --root. Can be passed multiple times.",
    )
    parser.add_argument(
        "--step",
        default="all",
        help="'all', 'last', or comma-separated step numbers, e.g. 20,50.",
    )
    parser.add_argument("--examples", type=int, default=2)
    parser.add_argument("--chars", type=int, default=360)
    args = parser.parse_args()

    if not args.exp:
        raise SystemExit("Pass at least one --exp <experiment_name>.")

    root = Path(args.root)
    for exp in args.exp:
        inspect_experiment(root, exp, args)


if __name__ == "__main__":
    main()
