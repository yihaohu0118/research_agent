"""E-Patch: Model Self-Evolution via Online Experience Replay + Self-Critique.

Two complementary self-evolution mechanisms in a single module:

**Success Replay** — the model's own successful rollout trajectories are
stored in a per-category ring buffer and injected as behavioural demos
in future rollout prompts.  Creates a positive feedback loop:

    step 1-20 : model bad, bank empty → pure GRPO
    step 20-50: model starts passing, bank fills
    step 50+  : demos from step 20-50 help model pass more → better demos

**Self-Critique** — failed trajectories are distilled into short
natural-language critiques ("I called tool X when I should have
abstained") and also injected alongside success demos, giving a
contrastive signal.  Unlike A-Patch (which scales advantages), this
operates in the *prompt* space, so the model sees both what worked
and what didn't *before* generating its next attempt.

The bank lives in-process memory — no ReMe or external service.

Config path (all under ``tocf.self_evolution``):
    enable: bool (default false)
    max_per_category: int (default 5)
    inject_prob: float (default 0.5) — probability of injecting a demo
    critique_prob: float (default 0.5) — probability of also adding a
        failure critique alongside the success demo
    apply_to_validation: bool (default false)
    template: str — wrapping template for success demo
    critique_template: str — wrapping template for failure critique
"""
from __future__ import annotations

import json
import os
import random
from collections import defaultdict
from typing import Any, Dict, List, Optional

from agentevolver.module.tocf.state import TOCFCapabilityState, dominant_failure_tag
from agentevolver.schema.task import Task


def _cfg_get(config: Any, path: str, default=None):
    cur = config
    for part in path.split("."):
        if cur is None:
            return default
        if isinstance(cur, dict):
            cur = cur.get(part, default)
        else:
            cur = getattr(cur, part, default)
    return cur


_DEFAULT_TEMPLATE = (
    "\n\n[Self-Evolved Experience] A successful strategy on a similar task:\n"
    "{demo}\n"
    "Use the above as reference, but adapt to the current request.\n"
)

_DEFAULT_CRITIQUE_TEMPLATE = (
    "\n[Self-Critique] A mistake I made on a similar task:\n"
    "{critique}\n"
    "Avoid repeating this mistake.\n"
)

# ── Lightweight failure-pattern detectors ─────────────────────────────────────
# These operate on raw trajectory messages, so they work regardless of
# which grader is active (no dependency on A-Patch failure_tags).

import re as _re

_ERROR_RE = _re.compile(
    r"\[ERROR\]|Error during execution|not available in the current tool list|"
    r"Invalid tool call format|Invalid character|unexpected keyword argument",
    _re.IGNORECASE,
)


def _extract_critique_from_messages(messages: List[dict]) -> str:
    """Distill a failed trajectory into a one-line self-critique.

    Heuristic priority:
      1. Tool call rejected by env (tool not available / parser error)
      2. Tool execution returned an error
      3. Model made no tool calls at all on a multi-turn task
      4. Generic "this trajectory failed"
    """
    rejected_tools: List[str] = []
    error_snippets: List[str] = []
    tool_call_count = 0
    user_turn_count = 0

    for msg in messages:
        role = msg.get("role", "")
        if role == "user":
            user_turn_count += 1
        elif role == "assistant" and msg.get("tool_calls"):
            tool_call_count += len(msg["tool_calls"])
        elif role == "assistant" and msg.get("_bfcl_rejected_tool_calls"):
            for tc in (msg.get("_bfcl_rejected_tool_calls") or []):
                name = (tc.get("function") or {}).get("name", "unknown")
                rejected_tools.append(name)
        elif role == "env":
            content = str(msg.get("content", ""))
            if _ERROR_RE.search(content):
                error_snippets.append(content[:100])
        elif role == "tool":
            content = str(msg.get("content", ""))
            if _ERROR_RE.search(content):
                error_snippets.append(content[:100])

    if rejected_tools:
        names = ", ".join(rejected_tools[:3])
        return (
            f"Called unavailable/malformed tool(s): {names}. "
            "Check which tools are actually listed before calling them."
        )
    if error_snippets:
        snippet = error_snippets[0]
        return f"Tool execution error: {snippet}. Verify arguments and tool availability."
    if tool_call_count == 0 and user_turn_count >= 2:
        return (
            "Made no tool calls despite the task requiring tool use. "
            "When tools are available and the task needs them, call the appropriate tool."
        )
    return "This attempt failed. Review the task requirements carefully before acting."


class ExperienceBank:
    """Ranked, persistent per-category and per-tag memory bank."""

    def __init__(self, max_per_category: int = 5, path: str | None = None):
        self._success: Dict[str, list[dict[str, Any]]] = defaultdict(list)
        self._critique: Dict[str, list[dict[str, Any]]] = defaultdict(list)
        self._max = max_per_category
        self.path = path
        self._ingest_count = 0
        self._critique_count = 0
        if path:
            self.load(path)

    def _key(self, category: str, tag: str | None = None) -> str:
        return f"{category}::{tag or 'unknown'}"

    def _entry(
        self,
        category: str,
        text: str,
        *,
        tag: str | None = None,
        reward: float = 0.0,
        task_id: str | None = None,
        step: int | None = None,
        quality: float | None = None,
    ) -> dict[str, Any]:
        score = float(quality if quality is not None else reward)
        return {
            "category": str(category),
            "tag": str(tag or "unknown"),
            "text": str(text),
            "reward": float(reward),
            "quality": score,
            "task_id": str(task_id or ""),
            "created_step": step,
        }

    def _trim(self, bucket: list[dict[str, Any]]) -> None:
        bucket.sort(
            key=lambda item: (
                float(item.get("quality", 0.0) or 0.0),
                int(item.get("created_step") or -1),
            ),
            reverse=True,
        )
        del bucket[self._max :]

    def ingest(
        self,
        category: str,
        demo: str,
        *,
        tag: str | None = None,
        reward: float = 1.0,
        task_id: str | None = None,
        step: int | None = None,
        quality: float | None = None,
    ) -> None:
        if not demo:
            return
        entry = self._entry(
            category,
            demo,
            tag=tag,
            reward=reward,
            task_id=task_id,
            step=step,
            quality=quality,
        )
        bucket = self._success[self._key(category, tag)]
        bucket.append(entry)
        self._trim(bucket)
        self._ingest_count += 1

    def ingest_critique(
        self,
        category: str,
        critique: str,
        *,
        tag: str | None = None,
        reward: float = 0.0,
        task_id: str | None = None,
        step: int | None = None,
        quality: float | None = None,
    ) -> None:
        if not critique:
            return
        entry = self._entry(
            category,
            critique,
            tag=tag,
            reward=reward,
            task_id=task_id,
            step=step,
            quality=quality if quality is not None else 1.0 - float(reward),
        )
        bucket = self._critique[self._key(category, tag)]
        bucket.append(entry)
        self._trim(bucket)
        self._critique_count += 1

    def _candidates(
        self,
        source: Dict[str, list[dict[str, Any]]],
        category: str,
        tag: str | None,
    ) -> list[dict[str, Any]]:
        keys = [self._key(category, tag)] if tag else []
        keys.append(self._key(category, "unknown"))
        exact = []
        for key in keys:
            exact.extend(source.get(key, []))
        if exact:
            return exact
        fallback: list[dict[str, Any]] = []
        prefix = f"{category}::"
        for key, entries in source.items():
            if key.startswith(prefix):
                fallback.extend(entries)
        return fallback

    def sample(self, category: str, tag: str | None = None) -> Optional[str]:
        candidates = self._candidates(self._success, category, tag)
        if not candidates:
            return None
        ranked = sorted(candidates, key=lambda item: float(item.get("quality", 0.0)), reverse=True)
        pool = ranked[: max(1, min(len(ranked), self._max))]
        return str(random.choice(pool).get("text") or "")

    def sample_critique(self, category: str, tag: str | None = None) -> Optional[str]:
        candidates = self._candidates(self._critique, category, tag)
        if not candidates:
            return None
        ranked = sorted(candidates, key=lambda item: float(item.get("quality", 0.0)), reverse=True)
        pool = ranked[: max(1, min(len(ranked), self._max))]
        return str(random.choice(pool).get("text") or "")

    @property
    def stats(self) -> Dict[str, int]:
        counts: Dict[str, int] = defaultdict(int)
        for entries in self._success.values():
            for entry in entries:
                counts[str(entry.get("category") or "unknown")] += 1
        return dict(counts)

    @property
    def critique_stats(self) -> Dict[str, int]:
        counts: Dict[str, int] = defaultdict(int)
        for entries in self._critique.values():
            for entry in entries:
                counts[str(entry.get("category") or "unknown")] += 1
        return dict(counts)

    @property
    def total_ingested(self) -> int:
        return self._ingest_count

    @property
    def total_critiques(self) -> int:
        return self._critique_count

    @property
    def total_stored(self) -> int:
        return (
            sum(len(entries) for entries in self._success.values())
            + sum(len(entries) for entries in self._critique.values())
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "success": {key: list(entries) for key, entries in self._success.items()},
            "critique": {key: list(entries) for key, entries in self._critique.items()},
            "max_per_category": self._max,
            "ingest_count": self._ingest_count,
            "critique_count": self._critique_count,
        }

    def load(self, path: str | None = None) -> None:
        target = path or self.path
        if not target or not os.path.exists(target):
            return
        try:
            with open(target, "r", encoding="utf-8") as handle:
                raw = json.load(handle)
        except (OSError, json.JSONDecodeError):
            return
        self._success = defaultdict(list, {str(k): list(v) for k, v in dict(raw.get("success") or {}).items()})
        self._critique = defaultdict(list, {str(k): list(v) for k, v in dict(raw.get("critique") or {}).items()})
        stored_success = sum(len(entries) for entries in self._success.values())
        stored_critique = sum(len(entries) for entries in self._critique.values())
        self._ingest_count = int(raw.get("ingest_count", stored_success) or stored_success)
        self._critique_count = int(raw.get("critique_count", stored_critique) or stored_critique)

    def save(self, path: str | None = None) -> None:
        target = path or self.path
        if not target:
            return
        os.makedirs(os.path.dirname(target) or ".", exist_ok=True)
        with open(target, "w", encoding="utf-8") as handle:
            json.dump(self.to_dict(), handle, ensure_ascii=False, indent=2)


def extract_tool_call_demo(messages: List[dict]) -> str:
    """Extract a compact behavioural summary from a successful trajectory.

    For trajectories that contain tool calls (base / miss_param / long_context),
    the output looks like::

        func_a(x=1) → [result snippet] → func_b(y=2) → ...

    For trajectories that succeed *without* tool calls (miss_func abstention),
    the output is a short natural-language note so the experience bank also
    covers the "correct-abstain" behaviour.

    Truncated to at most 8 calls to keep the demo short.
    """
    parts: List[str] = []
    max_calls = 8
    has_any_tool_call = False
    assistant_text_parts: List[str] = []

    for msg in messages:
        if len(parts) >= max_calls * 2:
            break
        role = msg.get("role", "")
        if role == "assistant":
            if msg.get("tool_calls"):
                has_any_tool_call = True
                for tc in msg["tool_calls"]:
                    func = tc.get("function", {})
                    name = func.get("name", "?")
                    args = func.get("arguments", "")
                    if isinstance(args, str) and len(args) > 80:
                        args = args[:77] + "..."
                    parts.append(f"{name}({args})")
                    if len(parts) >= max_calls:
                        break
            elif msg.get("content"):
                text = str(msg["content"]).strip()
                if text and len(assistant_text_parts) < 2:
                    assistant_text_parts.append(text[:120])
        elif role == "tool":
            content = str(msg.get("content", ""))[:60]
            if content:
                parts.append(f"→ {content}")

    if parts:
        return " → ".join(parts[:max_calls * 2])

    if not has_any_tool_call and assistant_text_parts:
        return (
            "[Abstained from tool calls] The model correctly replied in "
            "text without invoking any tools. Example reply: "
            + assistant_text_parts[0]
        )

    return ""


def epatch_enabled(config: Any) -> bool:
    tocf_on = bool(_cfg_get(config, "tocf.enable", False))
    ep_on = bool(_cfg_get(config, "tocf.self_evolution.enable", False))
    return tocf_on and ep_on


def apply_experience_injection(
    task: Task,
    bank: ExperienceBank,
    config: Any,
    mode: str | None = None,
    capability_state: TOCFCapabilityState | None = None,
) -> bool:
    """Inject self-evolved experience (success demo + optional failure critique).

    Contrastive injection strategy:
      - With probability ``inject_prob``, append a success demo.
      - With probability ``critique_prob``, additionally append a failure
        critique for the same category (if one exists).

    Returns True if the query was modified.
    """
    if not epatch_enabled(config):
        return False

    se_cfg = _cfg_get(config, "tocf.self_evolution", {}) or {}
    apply_to_val = bool(_cfg_get(se_cfg, "apply_to_validation", False))
    if mode in ("validate", "val") and not apply_to_val:
        return False

    inject_prob = float(_cfg_get(se_cfg, "inject_prob", 0.5) or 0.5)
    if random.random() > inject_prob:
        return False

    from agentevolver.module.tocf.category import infer_task_category
    category = (task.metadata or {}).get("category")
    if not category:
        category = infer_task_category(task.task_id, task.env_type, task.metadata)
    if not category:
        return False

    tag = None
    if capability_state is not None and task.task_id:
        task_state = capability_state.tasks.get(str(task.task_id)) or {}
        tag = task_state.get("last_tag")

    demo = bank.sample(category, tag=tag)
    if not demo:
        return False

    if task.query is None:
        return False

    template = str(_cfg_get(se_cfg, "template", _DEFAULT_TEMPLATE) or _DEFAULT_TEMPLATE)
    experience_text = template.format(demo=demo)

    critique_prob = float(_cfg_get(se_cfg, "critique_prob", 0.5) or 0.5)
    critique_text = ""
    if random.random() < critique_prob:
        critique = bank.sample_critique(category, tag=tag)
        if critique:
            crit_tmpl = str(
                _cfg_get(se_cfg, "critique_template", _DEFAULT_CRITIQUE_TEMPLATE)
                or _DEFAULT_CRITIQUE_TEMPLATE
            )
            critique_text = crit_tmpl.format(critique=critique)

    metadata = task.metadata if task.metadata is not None else {}
    metadata.setdefault("tocf", {})
    metadata["tocf"]["self_evolution_injected"] = True
    metadata["tocf"]["self_critique_injected"] = bool(critique_text)
    metadata["tocf"]["self_evolution_tag"] = tag
    task.metadata = metadata

    task.query = f"{task.query}{experience_text}{critique_text}"
    return True


def ingest_from_trajectories(
    bank: ExperienceBank,
    trajectories: list,
    config: Any,
    global_step: int | None = None,
) -> Dict[str, float]:
    """Ingest both success demos and failure critiques from trajectories.

    Returns metrics dict.
    """
    if not epatch_enabled(config):
        return {}

    from agentevolver.module.tocf.category import infer_task_category

    ingested_success = 0
    ingested_critique = 0

    for traj in trajectories:
        messages = getattr(traj, "steps", None) or []
        if not messages:
            continue

        meta = getattr(traj, "metadata", {}) or {}
        category = meta.get("category")
        if not category:
            task_id = meta.get("task_id") or getattr(traj, "data_id", "")
            category = infer_task_category(task_id, "bfcl", meta)
        if not category:
            continue

        success = getattr(traj, "success", False)
        task_id = meta.get("task_id") or getattr(traj, "data_id", "")
        reward = float(getattr(getattr(traj, "reward", None), "outcome", 0.0) or 0.0)
        reward_meta = (
            getattr(getattr(traj, "reward", None), "metadata", None) or {}
        )
        progress_info = reward_meta.get("bfcl_dense_progress_info", {}) or {}
        failure_tags = list(progress_info.get("failure_tags") or [])
        tag = dominant_failure_tag(failure_tags)

        if success:
            demo = extract_tool_call_demo(messages)
            if demo:
                bank.ingest(
                    category,
                    demo,
                    tag=tag,
                    reward=reward,
                    task_id=task_id,
                    step=global_step,
                )
                ingested_success += 1
        else:
            critique = _extract_critique_from_messages(messages)
            if critique:
                bank.ingest_critique(
                    category,
                    critique,
                    tag=tag,
                    reward=reward,
                    task_id=task_id,
                    step=global_step,
                )
                ingested_critique += 1

    bank.save()

    return {
        "tocf/epatch/ingested_success": float(ingested_success),
        "tocf/epatch/ingested_critique": float(ingested_critique),
        "tocf/epatch/bank_success": float(bank.total_ingested),
        "tocf/epatch/bank_critique": float(bank.total_critiques),
        "tocf/epatch/bank_total": float(bank.total_stored),
        **{f"tocf/epatch/success/{cat}": float(n)
           for cat, n in bank.stats.items()},
        **{f"tocf/epatch/critique/{cat}": float(n)
           for cat, n in bank.critique_stats.items()},
    }
