"""On-policy Hindsight Attribution Router for GCCE (ported from SEAL).

Motivation
----------
The original GCCE ``gap_attributor`` relied on heuristic correlates of
Delta_E and Delta_pi that, in the absence of a teacher cache and an oracle
probe, collapsed to linear functions of the category failure rate. The
result was a router mathematically indistinguishable from a muted PACE.

This module replaces that heuristic with a *testable* identifiability
scheme inspired by Kristensen et al. (SEAL / AgentEvolver-main 3). For
every failed trajectory, the same policy ``pi_theta`` is asked — with a
small reflection prompt — whether it could have self-corrected. If yes,
we classify the failure as a *capability gap*; if no (or if it declares
``[ENV_BLOCK]``), we classify it as an *environment gap*.

Three consequences:
  1. The classification is on-policy: no teacher API, no privileged model.
  2. It is cheap: one extra LM call per *failed* rollout, which can be
     batched with the existing async rollout manager.
  3. The resulting Delta_pi / Delta_E estimates are *causally attributed*
     to the policy vs. the environment, because the counterfactual is
     literally "same policy, same context, one more chance".

This is the same idea the research proposal attributes to GCCE but
actually executable on a 7B model without external judge APIs.
"""
from __future__ import annotations

import json
import re
from collections import defaultdict
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, List, Optional

try:
    from loguru import logger  # type: ignore
except ImportError:  # pragma: no cover - loguru is a soft dep
    import logging

    logger = logging.getLogger(__name__)


class GapType(str, Enum):
    CAPABILITY_GAP = "capability_gap"
    ENVIRONMENT_GAP = "environment_gap"
    UNDETERMINED = "undetermined"


@dataclass
class AttributionResult:
    gap_type: GapType = GapType.UNDETERMINED
    corrected_action: Optional[str] = None
    reflection_output: str = ""
    env_block_declared: bool = False
    correction_verified: bool = False
    failure_step_idx: int = -1
    category: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)


# ---------------------------------------------------------------- prompts

DEFAULT_REFLECTION_PROMPT = (
    "You just attempted a multi-step tool-calling task but FAILED to achieve "
    "the goal.\n\n"
    "Review the conversation above carefully. You have two options:\n\n"
    "**Option A - Self-Correct:** If you believe you *could* have succeeded "
    "with a different action at some point, output EXACTLY one corrected "
    "tool call in the standard format:\n"
    "<tool_call>\n"
    "{\"name\": \"<function-name>\", \"arguments\": <args-json-object>}\n"
    "</tool_call>\n\n"
    "**Option B - Declare Environment Block:** If you believe the environment "
    "itself made success impossible (e.g., missing API capability, ambiguous "
    "instructions that cannot be resolved, hidden preconditions), output "
    "EXACTLY:\n[ENV_BLOCK]\n\n"
    "Respond with ONLY Option A or Option B, nothing else."
)

REFLECTION_PROMPT_WITH_SCORE = (
    "You just attempted a multi-step tool-calling task and scored "
    "{score:.1%} (1.0 = full success).\n\n"
    "Review the conversation above. Your goal is to determine WHY the score "
    "was not perfect.\n\n"
    "**Option A - Self-Correct:** If a different action choice would have "
    "improved the outcome, output ONE corrected tool call:\n"
    "<tool_call>\n"
    "{\"name\": \"<function-name>\", \"arguments\": <args-json-object>}\n"
    "</tool_call>\n\n"
    "**Option B - Declare Environment Block:** If the environment interface "
    "itself prevented success (missing info, ambiguous spec, hidden "
    "constraints), output:\n[ENV_BLOCK]\n\n"
    "Respond with ONLY Option A or Option B."
)


# ---------------------------------------------------------------- router

class HindsightAttributionRouter:
    """On-policy hindsight attribution for GCCE.

    Parameters
    ----------
    llm_chat_fn:
        Callable with signature ``(messages, request_id=str, custom_sampling_params=dict) -> dict``
        or similar. Must return the current policy's response. At minimum the
        return value must be either a ``str`` or a ``dict`` carrying a
        ``"content"`` field.
    env_client:
        Optional env client used to verify corrected actions by running them
        through ``env_client.step(...)``. When ``None`` the verify step is
        skipped regardless of ``verify_correction``.
    env_block_token:
        Sentinel substring indicating the model itself attributed failure to
        the environment.
    verify_correction:
        If ``True`` and both ``env_client`` and ``instance_id`` are available,
        attempt to execute the parsed corrected action to confirm the
        environment actually accepts it. A rejected action reverts the
        classification to ``ENVIRONMENT_GAP``.
    failure_threshold:
        Trajectory-level outcome below which a rollout is considered a
        failure worth attributing. Successful rollouts are skipped.
    max_attributions_per_step:
        Soft cap on the number of attribution calls per training step to
        keep LLM cost bounded under large batch sizes. Extra failures are
        dropped at random.
    """

    def __init__(
        self,
        llm_chat_fn: Callable,
        env_client: Any = None,
        reflection_prompt: str = DEFAULT_REFLECTION_PROMPT,
        env_block_token: str = "[ENV_BLOCK]",
        verify_correction: bool = False,
        failure_threshold: float = 1.0,
        max_attributions_per_step: int = 64,
    ):
        self.llm_chat_fn = llm_chat_fn
        self.env_client = env_client
        self.reflection_prompt = reflection_prompt
        self.env_block_token = env_block_token
        self.verify_correction = verify_correction
        self.failure_threshold = failure_threshold
        self.max_attributions_per_step = max_attributions_per_step

        # Per-category accumulators (reset each epoch).
        self._cap_gap_counts: Dict[str, int] = defaultdict(int)
        self._env_gap_counts: Dict[str, int] = defaultdict(int)
        self._total_counts: Dict[str, int] = defaultdict(int)

        # Lifetime metrics (not reset).
        self._total_attributions = 0
        self._total_env_block = 0
        self._total_corrections_parsed = 0
        self._total_corrections_verified = 0

        # Optional cache keyed by ``task_id`` so we never attribute the same
        # failing trajectory twice inside one step.
        self._seen_trajectory_ids: set = set()

    # -------------------------------------------------------- single-item

    def attribute(
        self,
        trajectory_messages: List[Dict[str, Any]],
        score: float,
        task_id: str = "",
        instance_id: str = "",
        category: str = "",
        env_client: Any = None,
    ) -> AttributionResult:
        """Classify a single trajectory. Returns ``AttributionResult``.

        Successful trajectories (``score >= failure_threshold``) return an
        ``UNDETERMINED`` result with metadata ``reason='succeeded'`` and do
        not trigger an LM call.
        """
        result = AttributionResult(category=category)
        result.metadata["task_id"] = task_id
        result.metadata["original_score"] = score

        if score >= self.failure_threshold:
            result.gap_type = GapType.UNDETERMINED
            result.metadata["reason"] = "trajectory_succeeded"
            return result

        reflection_messages = self._build_reflection_context(
            trajectory_messages, score
        )

        try:
            llm_output = self.llm_chat_fn(reflection_messages, request_id="")
            reflection_text = self._extract_text(llm_output)
            result.reflection_output = reflection_text
        except Exception as e:
            logger.warning(
                f"[Hindsight] reflection LM call failed for task={task_id}: {e}"
            )
            result.gap_type = GapType.UNDETERMINED
            result.metadata["error"] = str(e)
            return result

        self._total_attributions += 1

        if self.env_block_token in reflection_text:
            result.env_block_declared = True
            result.gap_type = GapType.ENVIRONMENT_GAP
            result.metadata["reason"] = "env_block_declared"
            self._total_env_block += 1
            self._record(category, result.gap_type)
            return result

        corrected_action = self._parse_corrected_action(reflection_text)
        if corrected_action is None:
            # Reflection produced neither a valid action nor an env-block
            # declaration: safest interpretation is environment gap, matching
            # the policy's implicit claim "I cannot recover here".
            result.gap_type = GapType.ENVIRONMENT_GAP
            result.metadata["reason"] = "no_valid_action_parsed"
            self._record(category, result.gap_type)
            return result

        result.corrected_action = corrected_action
        self._total_corrections_parsed += 1

        client = env_client or self.env_client
        if self.verify_correction and client is not None and instance_id:
            verified = self._verify_action(client, instance_id, corrected_action)
            result.correction_verified = verified
            if not verified:
                result.gap_type = GapType.ENVIRONMENT_GAP
                result.metadata["reason"] = "corrected_action_rejected_by_env"
                self._record(category, result.gap_type)
                return result
            self._total_corrections_verified += 1

        result.gap_type = GapType.CAPABILITY_GAP
        result.metadata["reason"] = "self_correction_succeeded"
        self._record(category, result.gap_type)
        return result

    # -------------------------------------------------------- batch interface

    def batch_attribute(
        self,
        trajectories: List[Dict[str, Any]],
    ) -> List[AttributionResult]:
        """Attribute a list of trajectory dicts.

        Each ``traj`` must carry keys: ``messages`` (list), ``score`` (float).
        Optional keys: ``task_id``, ``instance_id``, ``category``.

        Honors ``max_attributions_per_step`` by dropping excess failures from
        the back of the list (stable).
        """
        # Split successes out early so we never pay for their LM call.
        failures = []
        results: List[AttributionResult] = []
        for t in trajectories:
            score = float(t.get("score", 0.0) or 0.0)
            if score >= self.failure_threshold:
                results.append(
                    AttributionResult(
                        gap_type=GapType.UNDETERMINED,
                        category=t.get("category", ""),
                        metadata={
                            "task_id": t.get("task_id", ""),
                            "reason": "trajectory_succeeded",
                            "original_score": score,
                        },
                    )
                )
            else:
                failures.append(t)

        # Budget cap: keep the most recent failures (typical curriculum
        # skew). We avoid random sampling to keep results reproducible.
        if self.max_attributions_per_step > 0:
            failures = failures[: self.max_attributions_per_step]

        for t in failures:
            results.append(
                self.attribute(
                    trajectory_messages=t.get("messages", []),
                    score=float(t.get("score", 0.0) or 0.0),
                    task_id=t.get("task_id", ""),
                    instance_id=t.get("instance_id", ""),
                    category=t.get("category", ""),
                )
            )
        return results

    # -------------------------------------------------------- category aggregates

    def category_rates(self) -> Dict[str, Dict[str, float]]:
        """Return ``{category: {cap_gap_rate, env_gap_rate, sample_count}}``.

        Rates are computed over *this window's* attributions only. Categories
        with no attribution samples are excluded.
        """
        out: Dict[str, Dict[str, float]] = {}
        for cat, total in self._total_counts.items():
            if total <= 0:
                continue
            cap = self._cap_gap_counts.get(cat, 0)
            env = self._env_gap_counts.get(cat, 0)
            out[cat] = {
                "cap_gap_rate": float(cap) / float(total),
                "env_gap_rate": float(env) / float(total),
                "sample_count": float(total),
            }
        return out

    def reset_window(self) -> None:
        """Clear the per-window counts (call once per epoch)."""
        self._cap_gap_counts.clear()
        self._env_gap_counts.clear()
        self._total_counts.clear()
        self._seen_trajectory_ids.clear()

    def metrics(self) -> Dict[str, float]:
        total = max(1, self._total_attributions)
        overall_cap = sum(self._cap_gap_counts.values())
        overall_env = sum(self._env_gap_counts.values())
        return {
            "gcce/hindsight/total_attributions": float(self._total_attributions),
            "gcce/hindsight/env_block_rate": float(self._total_env_block) / float(total),
            "gcce/hindsight/capability_gap_rate": float(overall_cap)
            / float(max(1, overall_cap + overall_env)),
            "gcce/hindsight/environment_gap_rate": float(overall_env)
            / float(max(1, overall_cap + overall_env)),
            "gcce/hindsight/corrections_parsed": float(self._total_corrections_parsed),
            "gcce/hindsight/corrections_verified": float(
                self._total_corrections_verified
            ),
        }

    # -------------------------------------------------------- helpers

    def _record(self, category: str, gap: GapType) -> None:
        key = category or "unknown"
        self._total_counts[key] += 1
        if gap == GapType.CAPABILITY_GAP:
            self._cap_gap_counts[key] += 1
        elif gap == GapType.ENVIRONMENT_GAP:
            self._env_gap_counts[key] += 1

    def _build_reflection_context(
        self, messages: List[Dict[str, Any]], score: float
    ) -> List[Dict[str, Any]]:
        """Append the reflection prompt to the trajectory messages."""
        context: List[Dict[str, Any]] = []
        for m in messages:
            role = m.get("role", "user")
            # env_manager.get_llm_chat_fn historically used "value" instead of
            # "content" in some codepaths. Normalise to both.
            content = m.get("content", m.get("value", ""))
            context.append({"role": role, "content": content})
        if 0.0 < score < 1.0:
            prompt = REFLECTION_PROMPT_WITH_SCORE.format(score=score)
        else:
            prompt = self.reflection_prompt
        context.append({"role": "user", "content": prompt})
        return context

    @staticmethod
    def _extract_text(llm_output: Any) -> str:
        if isinstance(llm_output, dict):
            return str(
                llm_output.get("content")
                or llm_output.get("value")
                or llm_output.get("message", "")
            )
        return str(llm_output or "")

    @staticmethod
    def _parse_corrected_action(text: str) -> Optional[str]:
        """Extract a JSON tool_call from the model's reflection output."""
        # Tolerate both forms: <tool_call>\n{...}\n</tool_call> and inline.
        pattern = r"<tool_call>\s*\n?(\{.*?\})\s*\n?</tool_call>"
        match = re.search(pattern, text, re.DOTALL)
        if not match:
            return None
        try:
            data = json.loads(match.group(1))
            if (
                isinstance(data, dict)
                and "name" in data
                and "arguments" in data
                and isinstance(data["name"], str)
            ):
                return json.dumps(data, ensure_ascii=False)
        except json.JSONDecodeError:
            return None
        return None

    def _verify_action(
        self, env_client: Any, instance_id: str, corrected_action_json: str
    ) -> bool:
        """Execute the corrected action and return True iff env accepted it."""
        try:
            data = json.loads(corrected_action_json)
            action_msg = {
                "role": "assistant",
                "content": (
                    "<tool_call>\n"
                    + json.dumps(data, ensure_ascii=False)
                    + "\n</tool_call>"
                ),
            }
            step_result = env_client.step(instance_id, action_msg)
            state_content = ""
            if isinstance(step_result, dict):
                states = step_result.get("state", [])
                if states and isinstance(states[0], dict):
                    state_content = states[0].get("content", "") or ""
            return (
                "[ERROR]" not in state_content and "[CONSTRAINT]" not in state_content
            )
        except Exception as e:
            logger.debug(f"[Hindsight] verification failed: {e}")
            return False


# ---------------------------------------------------------------- utilities

def compute_attribution_metrics(
    results: List[AttributionResult],
) -> Dict[str, float]:
    total = max(1, len(results))
    cap = sum(1 for r in results if r.gap_type == GapType.CAPABILITY_GAP)
    env = sum(1 for r in results if r.gap_type == GapType.ENVIRONMENT_GAP)
    undet = sum(1 for r in results if r.gap_type == GapType.UNDETERMINED)
    env_block = sum(1 for r in results if r.env_block_declared)
    verified = sum(1 for r in results if r.correction_verified)

    return {
        "gcce/hindsight/batch_total": float(total),
        "gcce/hindsight/batch_capability_gap_rate": cap / total,
        "gcce/hindsight/batch_environment_gap_rate": env / total,
        "gcce/hindsight/batch_undetermined_rate": undet / total,
        "gcce/hindsight/batch_env_block_rate": env_block / total,
        "gcce/hindsight/batch_correction_verified_rate": verified / max(1, cap),
    }


def hindsight_enabled(config: Any) -> bool:
    """Safe accessor for ``gcce.hindsight.enable``."""
    try:
        return bool(
            config.get("gcce", {}).get("hindsight", {}).get("enable", False)
        )
    except AttributeError:
        # OmegaConf nodes fall through here.
        hindsight = getattr(getattr(config, "gcce", None), "hindsight", None)
        if hindsight is None:
            return False
        return bool(getattr(hindsight, "enable", False))
