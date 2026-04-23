"""S-Patch: Strategy Bandit for Model Self-Exploration.

Unlike E-Patch (which passively replays past successes/failures) S-Patch
actively learns *which prompt-level strategy* best corrects each failure
mode, and injects that strategy on future rollouts of similar tasks.

Mechanism
---------
For each ``(category, failure_tag)`` pair the patch maintains a small
Thompson-sampling bandit over a seed library of strategy hints (e.g.
"prefer a text answer when no matching tool exists"). On rollout:

    1. Look up the *last* failure tag this ``task_id`` produced.
    2. Query the ``(category, tag)`` bandit → sample a strategy via
       Thompson sampling from Beta(wins + 1, pulls - wins + 1).
    3. Append the strategy text to ``task.query``.

After each rollout batch, the bandit is updated with ``success`` →
``wins += 1`` and always ``pulls += 1``. The model therefore *learns*
which strategy works against each failure mode purely from its own
rollout outcomes — genuine self-exploration with zero extra rollout
compute.

Design notes
~~~~~~~~~~~~
* **GRPO group integrity**: every rollout in a GRPO group shares the
  same prompt (and thus the same injected strategy), so the group-wise
  advantage stays unbiased. The strategy competes with other strategies
  *across* groups, not within one group.
* **Composability**: S-Patch and E-Patch both modify ``task.query``.
  They compose naturally — the injection order is E-Patch (demo +
  critique) first, S-Patch (strategy) second, so the strategy cue is
  closest to the user's actual request.
* **Cold start**: a uniform Beta(1, 1) prior + Thompson sampling means
  the first few pulls are exploratory regardless of bandit history.

Config path (all under ``tocf.strategy``):
    enable: bool (default false)
    inject_prob: float (default 0.7) — fraction of tasks that receive
        a strategy injection.
    apply_to_validation: bool (default false)
    prior_alpha / prior_beta: Beta prior parameters (default 1.0, 1.0)
    strategies: optional dict overriding the default seed library,
        mapping strategy_id → {text, tags} where tags is the list of
        failure-tag keys this strategy is considered a candidate for
        (``"*"`` means "always a candidate").
    template: optional wrapping template, default is a minimal one-line
        prefix.
"""
from __future__ import annotations

import math
import random
from collections import defaultdict
from typing import Any, Dict, List, Optional, Tuple

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


# ── Seed strategy library ─────────────────────────────────────────────────────
# Each entry: (strategy_id, text, applicable_tags).
# ``applicable_tags`` defines which failure-tag keys this strategy is a
# candidate for.  ``"*"`` means "always a candidate" (useful when no prior
# failure history exists for a task).
#
# The texts are intentionally short (<= ~30 words) to keep token overhead
# negligible and to force the model to interpret rather than copy.

_SEED_STRATEGIES: List[Tuple[str, str, List[str]]] = [
    # ── Abstention strategies (miss_func / irrelevance) ──────────────────
    (
        "abstain_when_unmatched",
        "If no tool in the provided list matches the user's request, "
        "answer in plain text and do NOT attempt a tool call.",
        ["spurious_tool_call", "*"],
    ),
    (
        "verify_tool_name",
        "Before calling a tool, confirm its exact name appears in the "
        "tools section. If uncertain, prefer a text reply.",
        ["spurious_tool_call"],
    ),
    (
        "text_first_on_irrelevance",
        "If the request can be answered conversationally without tools, "
        "do so directly rather than forcing a tool call.",
        ["spurious_tool_call", "correct_abstention"],
    ),
    # ── Acting strategies (empty / scorable turns) ───────────────────────
    (
        "never_empty_turn",
        "Always respond: either issue a valid tool call or produce a "
        "direct text answer. Do not return nothing.",
        ["empty_turn_model_response", "*"],
    ),
    (
        "decompose_action",
        "Break multi-step requests into one tool call at a time; "
        "inspect each result before choosing the next action.",
        ["empty_turn_model_response", "state_mismatch"],
    ),
    # ── State / parameter strategies (base / miss_param / state_mismatch) ─
    (
        "read_before_write",
        "When the task involves modifying state, first issue a read-only "
        "call to confirm the current state, then act.",
        ["state_mismatch", "instance_mismatch"],
    ),
    (
        "verify_arguments",
        "Before every tool call, cross-check each argument against values "
        "explicitly stated by the user. Do not fabricate parameters.",
        ["state_mismatch", "response_mismatch", "*"],
    ),
    (
        "minimal_calls",
        "Prefer the shortest tool-call sequence that satisfies the "
        "request; avoid redundant or exploratory calls.",
        ["state_mismatch", "*"],
    ),
    # ── Response strategies (response_mismatch) ──────────────────────────
    (
        "address_all_asks",
        "Re-read the user's last message before concluding and make sure "
        "every explicit ask is addressed in your final reply.",
        ["response_mismatch"],
    ),
    (
        "summarise_tool_output",
        "When replying after tool calls, concisely summarise what was "
        "done and what the result was, in natural language.",
        ["response_mismatch", "*"],
    ),
    # ── Generic baseline (always a candidate) ────────────────────────────
    (
        "think_before_act",
        "Briefly reason about which tool (if any) is needed before "
        "acting. Prefer caution over extra calls.",
        ["*"],
    ),
]


_DEFAULT_TEMPLATE = (
    "\n\n[Strategy] {text}\n"
)


def _resolve_strategy_library(config: Any) -> Dict[str, Dict[str, Any]]:
    """Return the strategy library, overriding defaults with config if present.

    Config format (optional)::

        tocf:
          strategy:
            strategies:
              my_strat_id:
                text: "Always verify tool names."
                tags: ["spurious_tool_call"]
    """
    library: Dict[str, Dict[str, Any]] = {}
    for sid, text, tags in _SEED_STRATEGIES:
        library[sid] = {"text": text, "tags": list(tags)}

    override = _cfg_get(config, "tocf.strategy.strategies", None)
    if isinstance(override, dict):
        for sid, spec in override.items():
            if not isinstance(spec, dict):
                continue
            txt = str(spec.get("text", "") or "").strip()
            if not txt:
                continue
            raw_tags = spec.get("tags") or ["*"]
            tags_list = [str(t) for t in raw_tags] if isinstance(raw_tags, (list, tuple)) else ["*"]
            library[str(sid)] = {"text": txt, "tags": tags_list}

    return library


# ─────────────────────────────────────────────────────────────────────────────
# StrategyBandit
# ─────────────────────────────────────────────────────────────────────────────


class StrategyBandit:
    """Thompson-sampling bandit over strategy hints, keyed by (category, tag).

    State is process-local (no persistence). ``None`` category or tag are
    coerced to the strings ``"_none"`` / ``"unknown"`` so the bandit
    always has a valid lookup key.
    """

    _UNKNOWN_TAG = "unknown"
    _NONE_CAT = "_none"

    def __init__(
        self,
        library: Dict[str, Dict[str, Any]],
        prior_alpha: float = 1.0,
        prior_beta: float = 1.0,
    ):
        self._library = library
        self._prior_alpha = float(prior_alpha)
        self._prior_beta = float(prior_beta)
        # (category, tag) -> strategy_id -> (wins, pulls)
        self._stats: Dict[Tuple[str, str], Dict[str, List[float]]] = defaultdict(dict)
        # task_id -> last failure tag observed
        self._last_tag_by_task: Dict[str, str] = {}
        # diagnostics
        self._total_pulls = 0
        self._total_wins = 0

    # ── Tag / key handling ────────────────────────────────────────────────

    def _key(self, category: Optional[str], tag: Optional[str]) -> Tuple[str, str]:
        cat = category or self._NONE_CAT
        t = tag or self._UNKNOWN_TAG
        return (cat, t)

    def get_last_tag(self, task_id: Optional[str]) -> str:
        if not task_id:
            return self._UNKNOWN_TAG
        return self._last_tag_by_task.get(str(task_id), self._UNKNOWN_TAG)

    def record_failure_tag(self, task_id: Optional[str], tag: Optional[str]) -> None:
        if not task_id or not tag:
            return
        self._last_tag_by_task[str(task_id)] = str(tag)

    # ── Candidate filtering ───────────────────────────────────────────────

    def _candidates(self, tag: str) -> List[str]:
        """Return strategy ids that list ``tag`` (or ``"*"``) in their tags."""
        out: List[str] = []
        for sid, spec in self._library.items():
            tags = spec.get("tags") or ["*"]
            if tag in tags or "*" in tags:
                out.append(sid)
        if not out:
            out = list(self._library.keys())
        return out

    # ── Selection ─────────────────────────────────────────────────────────

    def select(
        self,
        category: Optional[str],
        tag: Optional[str],
        rng: Optional[random.Random] = None,
    ) -> Optional[str]:
        """Thompson-sample a strategy id for the given (category, tag) key.

        Returns ``None`` if the library is empty.
        """
        rng = rng or random
        key = self._key(category, tag)
        candidates = self._candidates(key[1])
        if not candidates:
            return None

        bucket = self._stats[key]
        best_sid: Optional[str] = None
        best_score: float = -1.0
        for sid in candidates:
            wins, pulls = bucket.get(sid, [0.0, 0.0])
            alpha = self._prior_alpha + wins
            beta = self._prior_beta + max(0.0, pulls - wins)
            score = rng.betavariate(alpha, beta)
            if score > best_score:
                best_score = score
                best_sid = sid
        return best_sid

    # ── Updates ───────────────────────────────────────────────────────────

    def update(
        self,
        category: Optional[str],
        tag: Optional[str],
        strategy_id: str,
        success: bool,
    ) -> None:
        if not strategy_id or strategy_id not in self._library:
            return
        key = self._key(category, tag)
        bucket = self._stats[key]
        wins, pulls = bucket.get(strategy_id, [0.0, 0.0])
        pulls += 1.0
        if success:
            wins += 1.0
            self._total_wins += 1
        self._total_pulls += 1
        bucket[strategy_id] = [wins, pulls]

    # ── Diagnostics ───────────────────────────────────────────────────────

    def strategy_text(self, strategy_id: str) -> str:
        spec = self._library.get(strategy_id) or {}
        return str(spec.get("text", "") or "")

    def metrics(self, prefix: str = "tocf/spatch") -> Dict[str, float]:
        m: Dict[str, float] = {
            f"{prefix}/total_pulls": float(self._total_pulls),
            f"{prefix}/total_wins": float(self._total_wins),
            f"{prefix}/overall_winrate": (
                float(self._total_wins) / float(self._total_pulls)
                if self._total_pulls
                else 0.0
            ),
            f"{prefix}/num_keys": float(len(self._stats)),
        }
        # Per-strategy aggregated winrate across all keys.
        sid_wins: Dict[str, float] = defaultdict(float)
        sid_pulls: Dict[str, float] = defaultdict(float)
        for bucket in self._stats.values():
            for sid, (w, p) in bucket.items():
                sid_wins[sid] += w
                sid_pulls[sid] += p
        for sid, pulls in sid_pulls.items():
            if pulls <= 0:
                continue
            m[f"{prefix}/winrate/{sid}"] = float(sid_wins[sid]) / float(pulls)
            m[f"{prefix}/pulls/{sid}"] = float(pulls)
        return m


# ─────────────────────────────────────────────────────────────────────────────
# Helpers around trajectories / tasks
# ─────────────────────────────────────────────────────────────────────────────


def spatch_enabled(config: Any) -> bool:
    tocf_on = bool(_cfg_get(config, "tocf.enable", False))
    sp_on = bool(_cfg_get(config, "tocf.strategy.enable", False))
    return tocf_on and sp_on


def _dominant_failure_tag(tags: List[str]) -> str:
    """Return the most informative tag from a per-turn failure-tag list.

    Priority:
      1. If any non-pass failure tag exists, return the most frequent one.
      2. Else if all tags are ``pass``, return ``pass``.
      3. Else return ``unknown``.
    """
    if not tags:
        return StrategyBandit._UNKNOWN_TAG
    from collections import Counter
    non_pass = [t for t in tags if t and t != "pass"]
    if non_pass:
        return Counter(non_pass).most_common(1)[0][0]
    if any(t == "pass" for t in tags):
        return "pass"
    return StrategyBandit._UNKNOWN_TAG


def apply_strategy_injection(
    task: Task,
    bandit: StrategyBandit,
    config: Any,
    mode: str | None = None,
) -> Optional[str]:
    """Inject a bandit-selected strategy hint into ``task.query``.

    Returns the chosen ``strategy_id`` (or ``None`` if nothing was injected).
    The id is also stored on ``task.metadata["tocf"]["spatch_strategy"]``
    so we can later credit-assign the bandit update to the right arm.
    """
    if not spatch_enabled(config):
        return None

    sp_cfg = _cfg_get(config, "tocf.strategy", {}) or {}
    apply_to_val = bool(_cfg_get(sp_cfg, "apply_to_validation", False))
    if mode in ("validate", "val") and not apply_to_val:
        return None

    inject_prob = float(_cfg_get(sp_cfg, "inject_prob", 0.7) or 0.0)
    if inject_prob <= 0.0 or random.random() > inject_prob:
        return None

    if task.query is None:
        return None

    from agentevolver.module.tocf.category import infer_task_category
    category = (task.metadata or {}).get("category")
    if not category:
        category = infer_task_category(task.task_id, task.env_type, task.metadata)

    tag = bandit.get_last_tag(task.task_id)
    strategy_id = bandit.select(category, tag)
    if not strategy_id:
        return None

    text = bandit.strategy_text(strategy_id)
    if not text:
        return None

    template = str(_cfg_get(sp_cfg, "template", _DEFAULT_TEMPLATE) or _DEFAULT_TEMPLATE)
    injected = template.format(text=text)

    metadata = task.metadata if task.metadata is not None else {}
    metadata.setdefault("tocf", {})
    metadata["tocf"]["spatch_strategy"] = strategy_id
    metadata["tocf"]["spatch_prior_tag"] = tag
    metadata["tocf"]["spatch_category"] = category
    task.metadata = metadata

    task.query = f"{task.query}{injected}"
    return strategy_id


def update_bandit_from_trajectories(
    bandit: StrategyBandit,
    trajectories: list,
    config: Any,
) -> Dict[str, float]:
    """Post-rollout: update bandit statistics and task→tag cache.

    For each trajectory we look up the ``spatch_strategy`` that was
    injected (if any), credit the bandit arm with the trajectory outcome,
    and cache the *new* dominant failure tag so the next rollout on the
    same task can pick a strategy targeted at the current failure mode.

    Returns a metrics dict suitable for ``metrics.update(...)``.
    """
    if not spatch_enabled(config):
        return {}

    from agentevolver.module.tocf.category import infer_task_category

    updated = 0
    injected_count = 0
    success_updates = 0

    for traj in trajectories:
        meta = getattr(traj, "metadata", {}) or {}
        task_id = meta.get("task_id") or getattr(traj, "data_id", "") or ""
        category = meta.get("category")
        if not category:
            category = infer_task_category(task_id, "bfcl", meta)

        # Extract failure tags from trajectory reward metadata — same
        # structure A-Patch reads.
        reward_meta = (
            getattr(getattr(traj, "reward", None), "metadata", None) or {}
        )
        progress_info = reward_meta.get("bfcl_dense_progress_info", {}) or {}
        failure_tags = list(progress_info.get("failure_tags") or [])
        new_tag = _dominant_failure_tag(failure_tags)

        # The prior_tag is what the bandit selected against; read from
        # task metadata (injected by apply_strategy_injection).
        tocf_meta = meta.get("tocf") or {}
        strategy_id = tocf_meta.get("spatch_strategy")
        prior_tag = tocf_meta.get("spatch_prior_tag") or bandit.get_last_tag(task_id)

        success = bool(getattr(traj, "success", False))

        if strategy_id:
            bandit.update(category, prior_tag, strategy_id, success)
            updated += 1
            injected_count += 1
            if success:
                success_updates += 1

        # Always refresh the last-tag cache (even if no injection happened
        # this step) so future selections are informed.
        bandit.record_failure_tag(task_id, new_tag)

    metrics = bandit.metrics()
    metrics["tocf/spatch/updates"] = float(updated)
    metrics["tocf/spatch/injected_batch"] = float(injected_count)
    metrics["tocf/spatch/successful_batch_updates"] = float(success_updates)
    return metrics
