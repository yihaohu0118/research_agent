"""Capability-gap targeted re-exploration (SEAL §3.4.2) for GCCE.

After :class:`HindsightAttributionRouter` classifies a failure as a
capability gap, the trajectory carries a strong contrastive signal:

    * the environment did expose enough information to succeed;
    * the policy already knows (in hindsight) what it *should* have done;
    * yet at rollout time it picked the wrong action.

Rather than throwing this signal away, we convert it into a small
amount of directed exploration data. For each seed we build query
variants that *prime* the policy with a light-weight hint derived from
its own corrected action, then run fresh rollouts in the *unchanged*
environment and keep only successful ones for the next GRPO step.

This is purely additive: it injects extra Task objects into the batch
without modifying any reward signal, any environment, or the GRPO
objective. Ablations can enable/disable it cleanly.
"""
from __future__ import annotations

import json
import random
from collections import defaultdict, deque
from typing import Any, Deque, Dict, List, Optional

try:
    from loguru import logger  # type: ignore
except ImportError:  # pragma: no cover
    import logging

    logger = logging.getLogger(__name__)

from agentevolver.schema.task import Task


class CapGapExplorer:
    """Turn hindsight-flagged capability gaps into targeted exploration tasks.

    All configuration is optional; the module is a no-op when
    ``enable=False``.

    Config keys (all under ``seal.cap_gap_explore``):
      enable: bool (default False)
      buffer_max_size: int (default 256) — capped deque of seeds
      max_seeds_per_step: int (default 8)
      variants_per_seed: int (default 2)
      success_threshold: float (default 1.0) — outcome reward to keep a rollout
      sample_frequency: int (default 2) — generate exploration tasks every N steps
      prefer_corrected: bool (default True) — prioritise seeds with a parsed correction
    """

    def __init__(self, config: Dict[str, Any]):
        cfg = config or {}
        self.enabled = bool(cfg.get("enable", False))
        self.buffer_max_size = int(cfg.get("buffer_max_size", 256))
        self.max_seeds_per_step = int(cfg.get("max_seeds_per_step", 8))
        self.variants_per_seed = int(cfg.get("variants_per_seed", 2))
        self.success_threshold = float(cfg.get("success_threshold", 1.0))
        self.sample_frequency = int(cfg.get("sample_frequency", 2))
        self.prefer_corrected = bool(cfg.get("prefer_corrected", True))

        # Rolling buffer; new entries push out the oldest.
        self._buffer: Deque[Dict[str, Any]] = deque(maxlen=self.buffer_max_size)

        # Lightweight metrics.
        self._total_pushes: int = 0
        self._total_explore_tasks_built: int = 0
        self._total_survivors: int = 0

    # --------------------------------------------------- buffer management

    def push(
        self,
        *,
        task_id: str,
        first_query: str,
        messages: List[Dict[str, Any]],
        corrected_action: Optional[str] = None,
        category: str = "",
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Insert a capability-gap seed into the rolling buffer."""
        if not self.enabled:
            return
        if not task_id or not first_query:
            return
        self._buffer.append(
            {
                "task_id": task_id,
                "first_query": first_query,
                "messages": messages,
                "corrected_action": corrected_action,
                "category": category,
                "metadata": metadata or {},
            }
        )
        self._total_pushes += 1

    def buffer_size(self) -> int:
        return len(self._buffer)

    def clear_buffer(self) -> None:
        self._buffer.clear()

    # --------------------------------------------------- scheduling

    def should_explore(self, global_steps: int) -> bool:
        if not self.enabled or self.sample_frequency <= 0 or not self._buffer:
            return False
        return global_steps % self.sample_frequency == 0

    # --------------------------------------------------- seed sampling

    def sample_seeds(self) -> List[Dict[str, Any]]:
        """Sample up to ``max_seeds_per_step`` diverse seeds.

        Prioritises seeds carrying a parsed ``corrected_action`` (stronger
        contrastive signal). Within each group, prefers distinct
        ``task_id`` to avoid variant collapse.
        """
        if not self._buffer:
            return []
        buf = list(self._buffer)
        n = min(self.max_seeds_per_step, len(buf))

        with_correction = [s for s in buf if s.get("corrected_action")]
        without_correction = [s for s in buf if not s.get("corrected_action")]

        primary = with_correction if self.prefer_corrected else buf
        secondary = without_correction if self.prefer_corrected else []

        seeds: List[Dict[str, Any]] = []
        if primary:
            by_task: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
            for s in primary:
                by_task[s.get("task_id", "")].append(s)
            task_ids = list(by_task.keys())
            random.shuffle(task_ids)
            for tid in task_ids:
                if len(seeds) >= n:
                    break
                seeds.append(random.choice(by_task[tid]))

        if len(seeds) < n and secondary:
            remaining = n - len(seeds)
            seeds.extend(
                random.sample(secondary, min(remaining, len(secondary)))
            )

        return seeds[:n]

    # --------------------------------------------------- task construction

    def build_exploration_tasks(
        self,
        seeds: List[Dict[str, Any]],
        env_type: str,
    ) -> List[Task]:
        """Construct fresh ``Task`` objects from seeds."""
        tasks: List[Task] = []
        if not seeds:
            return tasks

        for seed in seeds:
            tid = seed.get("task_id", "")
            base_query = seed.get("first_query") or _extract_first_user(
                seed.get("messages", [])
            )
            if not base_query or not tid:
                continue
            corrected = seed.get("corrected_action")
            variants = _build_variants(base_query, corrected)[
                : self.variants_per_seed
            ]

            for idx, variant_query in enumerate(variants):
                meta = {
                    "cap_gap_explore": True,
                    "original_task_id": tid,
                    "has_correction_hint": corrected is not None,
                    "variant_index": idx,
                    "source_category": seed.get("category", ""),
                }
                meta.update(seed.get("metadata") or {})
                # Keep the generated task_id distinct so the exploration
                # rollout never collides with a real evaluation sample.
                explore_task_id = f"{tid}::cap_gap_explore::{idx}"
                tasks.append(
                    Task(
                        task_id=explore_task_id,
                        env_type=env_type,
                        open_query=False,
                        query=variant_query,
                        metadata=meta,
                    )
                )
        self._total_explore_tasks_built += len(tasks)
        logger.info(
            "[CapGapExplore] built {} exploration tasks from {} seeds "
            "({} with correction hints)",
            len(tasks),
            len(seeds),
            sum(1 for s in seeds if s.get("corrected_action")),
        )
        return tasks

    # --------------------------------------------------- quality gate

    def filter_successful(self, trajectories: List[Any]) -> List[Any]:
        """Keep only exploration rollouts that achieved ``success_threshold``."""
        survivors: List[Any] = []
        for traj in trajectories:
            r = getattr(traj, "reward", None)
            if r is None:
                continue
            outcome = float(getattr(r, "outcome", 0.0) or 0.0)
            if outcome >= self.success_threshold:
                survivors.append(traj)
        self._total_survivors += len(survivors)
        logger.info(
            "[CapGapExplore] {}/{} exploration rollouts survived threshold={:.2f}",
            len(survivors),
            len(trajectories),
            self.success_threshold,
        )
        return survivors

    # --------------------------------------------------- metrics

    def metrics(self) -> Dict[str, float]:
        return {
            "seal/cap_gap_explore/buffer_size": float(len(self._buffer)),
            "seal/cap_gap_explore/total_pushes": float(self._total_pushes),
            "seal/cap_gap_explore/tasks_built": float(
                self._total_explore_tasks_built
            ),
            "seal/cap_gap_explore/survivors": float(self._total_survivors),
        }


# ---------------------------------------------------------------- helpers

def _extract_first_user(messages: List[Dict[str, Any]]) -> str:
    for m in messages or []:
        if m.get("role") == "user":
            return str(m.get("content") or m.get("value") or "")
    return ""


def _build_variants(
    base_query: str, corrected_action: Optional[str]
) -> List[str]:
    """Turn the base query into variants primed by the corrected action.

    When the attributor parsed a corrected tool_call, we splice a light
    textual hint ("consider using <func> with parameters <args>") that
    nudges the policy toward the action it already identified in
    hindsight. Without a correction we fall back to generic step-by-step
    suffixes.
    """
    variants: List[str] = []
    hint = ""
    if corrected_action:
        try:
            data = json.loads(corrected_action)
            func = data.get("name", "") if isinstance(data, dict) else ""
            args = data.get("arguments", {}) if isinstance(data, dict) else {}
            if isinstance(args, dict):
                key_args = list(args.keys())[:3]
            else:
                key_args = []
            if func:
                hint = f" (Hint: consider using {func}"
                if key_args:
                    hint += f" with parameters: {', '.join(key_args)}"
                hint += ")"
        except (json.JSONDecodeError, TypeError, AttributeError):
            hint = ""

    if hint:
        variants.append(base_query + hint)
        variants.append(
            base_query
            + " Please think carefully about which function to call "
            "and verify the required parameters before each step."
        )
    else:
        variants.append(
            base_query
            + " Before calling any function, list what information you need "
            "and check available tools."
        )
        variants.append(
            base_query
            + " If your first approach fails, try an alternative strategy."
        )
    return variants


# ---------------------------------------------------------------- config

def cap_gap_explore_enabled(config: Any) -> bool:
    """Safe accessor for ``seal.cap_gap_explore.enable``."""
    try:
        return bool(
            config.get("seal", {})
            .get("cap_gap_explore", {})
            .get("enable", False)
        )
    except AttributeError:
        seal = getattr(config, "seal", None)
        if seal is None:
            return False
        cfg = getattr(seal, "cap_gap_explore", None)
        if cfg is None:
            return False
        return bool(getattr(cfg, "enable", False))
