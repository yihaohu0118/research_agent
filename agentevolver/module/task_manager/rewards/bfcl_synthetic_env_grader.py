from __future__ import annotations

import json
import threading
from typing import Any

from agentevolver.client.env_client import EnvClient
from agentevolver.module.agent_flow.reward_calculator import GraderResult, RewardCalculator
from agentevolver.module.tocf.bfcl_synthetic import (
    bfcl_synthetic_env_params,
    compare_tool_turns,
    extract_observed_tool_turns,
    normalize_turns_for_tool_schema,
    normalize_tool_turns,
    replay_tool_turns_in_bfcl_env,
    serialize_tool_turns,
    tool_schemas_from_prompt,
    tool_turns_to_strings,
    trajectory_has_tool_error,
)
from agentevolver.schema.task import Task
from agentevolver.schema.trajectory import Trajectory

from . import grader_manager

try:
    from omegaconf import OmegaConf
except Exception:  # pragma: no cover - OmegaConf is present in training.
    OmegaConf = None


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


def _first_system_prompt(messages: list[dict[str, Any]]) -> str:
    for message in messages:
        if message.get("role") == "system":
            return str(message.get("content") or "")
    return ""


@grader_manager.reg("bfcl-synthetic-env")
class BfclSyntheticEnvGrader(RewardCalculator):
    """Rule-based grader for co-evolved BFCL synthetic tasks.

    The task still runs inside the BFCL environment, so tool calls are executed
    against the real sandbox state. The reward is computed by structurally
    comparing the rollout's tool calls with the normalized executable
    ground-truth tool-call sequence produced during co-evolution finalization.
    """

    _gt_replay_cache: dict[tuple[str, str, str, str], tuple[bool, str]] = {}
    _gt_replay_lock = threading.Lock()

    def __init__(self, task: Task):
        super().__init__(task)
        self._config = None

    def set_config(self, config):
        self._config = config

    def _bfcl_env_params(self) -> dict[str, Any]:
        params: dict[str, Any] = {"is_open_query": bool(self.task.open_query)}
        bfcl_params = _cfg_get(self._config, "env_service.bfcl", None)
        if bfcl_params is not None:
            if OmegaConf is not None:
                bfcl_params = OmegaConf.to_container(bfcl_params, resolve=True)
            if isinstance(bfcl_params, dict):
                params.update(bfcl_params)
        params.setdefault("is_open_query", True)
        params.setdefault("strict_tool_parser", True)
        return params

    def _replay_enabled(self) -> bool:
        metadata_cfg = ((self.task.metadata or {}).get("tocf", {}) or {}).get("coevo", {})
        if "replay_gt_before_reward" in metadata_cfg:
            return bool(metadata_cfg["replay_gt_before_reward"])
        return bool(_cfg_get(self._config, "tocf.coevo.replay_gt_before_reward", True))

    def _trajectory_tool_schemas(self, trajectory: Trajectory) -> dict[str, dict[str, Any]]:
        prompt = _first_system_prompt(getattr(trajectory, "steps", []) or [])
        return tool_schemas_from_prompt(prompt)

    def _replay_cache_key(
        self,
        env: EnvClient,
        expected_gt: str,
        params: dict[str, Any],
    ) -> tuple[str, str, str, str]:
        params_key = json.dumps(params, ensure_ascii=False, sort_keys=True, default=str)
        return (str(env.base_url), str(self.task.task_id), expected_gt, params_key)

    def _replay_expected_gt(
        self,
        env: EnvClient,
        expected_turns,
    ) -> tuple[bool, str]:
        expected_gt = serialize_tool_turns(expected_turns)
        params, overlay_reason = bfcl_synthetic_env_params(
            self.task,
            self._bfcl_env_params(),
            turns=expected_turns,
        )
        if "synthetic_case_overlay" not in params:
            return False, overlay_reason
        cache_key = self._replay_cache_key(env, expected_gt, params)
        with self._gt_replay_lock:
            cached = self._gt_replay_cache.get(cache_key)
        if cached is not None:
            return cached

        result = replay_tool_turns_in_bfcl_env(
            env,
            str(self.task.task_id),
            expected_turns,
            params=params,
        )

        with self._gt_replay_lock:
            self._gt_replay_cache[cache_key] = result
        return result

    def calculate_reward(
        self, trajectory: Trajectory, env: EnvClient, instance_id: str
    ) -> GraderResult:
        expected_turns = normalize_tool_turns(self.task.ground_truth)
        overlay_params, overlay_reason = bfcl_synthetic_env_params(
            self.task,
            self._bfcl_env_params(),
            turns=expected_turns,
        )
        overlay = overlay_params.get("synthetic_case_overlay")
        semantic_alignment = (
            overlay.get("semantic_alignment", {})
            if isinstance(overlay, dict)
            else {"ok": False, "reason": overlay_reason, "score": 0.0}
        )
        tool_schemas = self._trajectory_tool_schemas(trajectory)
        expected_for_compare = (
            normalize_turns_for_tool_schema(expected_turns, tool_schemas)
            if tool_schemas
            else expected_turns
        )
        observed_turns = extract_observed_tool_turns(getattr(trajectory, "steps", []) or [])
        comparison = (
            compare_tool_turns(expected_for_compare, observed_turns)
            if expected_for_compare is not None
            else {
                "score": 0.0,
                "matched": 0,
                "expected": 0,
                "observed": len(observed_turns),
                "success": False,
                "failure_tag": "synthetic_gt_payload_unmappable",
            }
        )
        has_tool_error = trajectory_has_tool_error(getattr(trajectory, "steps", []) or [])

        score = float(comparison["score"])
        failure_tag = str(comparison["failure_tag"])
        replay_ok = True
        replay_reason = "disabled"
        if not expected_turns:
            score = 0.0
            failure_tag = "synthetic_gt_unparseable"
        elif (
            bool(_cfg_get(self._config, "tocf.coevo.require_semantic_alignment", True))
            and not bool(semantic_alignment.get("ok", False))
        ):
            score = 0.0
            failure_tag = str(semantic_alignment.get("reason") or "synthetic_semantic_unanchored")
        elif expected_for_compare is None:
            score = 0.0
            failure_tag = "synthetic_gt_payload_unmappable"
        elif self._replay_enabled():
            replay_ok, replay_reason = self._replay_expected_gt(env, expected_turns)
            if not replay_ok:
                score = 0.0
                failure_tag = "synthetic_gt_not_executable"
        if has_tool_error and replay_ok:
            score = 0.0
            failure_tag = "tool_error"

        success = score >= 1.0 and replay_ok and not has_tool_error
        if success:
            failure_tag = "pass"

        expected_metadata = expected_for_compare if expected_for_compare is not None else expected_turns
        metadata = {
            "bfcl_synthetic_expected": tool_turns_to_strings(expected_metadata),
            "bfcl_synthetic_expected_raw": tool_turns_to_strings(expected_turns),
            "bfcl_synthetic_observed": tool_turns_to_strings(observed_turns),
            "bfcl_synthetic_gt": serialize_tool_turns(expected_turns),
            "bfcl_synthetic_has_tool_error": has_tool_error,
            "bfcl_synthetic_gt_replay_ok": replay_ok,
            "bfcl_synthetic_gt_replay_reason": replay_reason,
            "bfcl_synthetic_overlay_ready": bool(overlay),
            "bfcl_synthetic_overlay_reason": overlay_reason,
            "bfcl_synthetic_overlay_hash": (
                overlay.get("objective_hash") if isinstance(overlay, dict) else None
            ),
            "bfcl_synthetic_semantic_alignment": semantic_alignment,
            "bfcl_dense_raw_accuracy": 1.0 if success else 0.0,
            "bfcl_dense_progress": score,
            "bfcl_dense_progress_info": {
                "failure_tags": [failure_tag],
                "passed_turns": int(success),
                "scorable_turns": 1,
                "passed_irrelevance_turns": 0,
                "irrelevance_turns": 0,
                "terminated_early": False,
                "error": None,
            },
        }

        reason = (
            "BFCL synthetic env verifier: "
            f"score={score:.4f}, matched={comparison['matched']}/"
            f"{comparison['expected']}, observed={comparison['observed']}, "
            f"tool_error={has_tool_error}, gt_replay={replay_ok} "
            f"({replay_reason}), tag={failure_tag}"
        )
        return {"score": score, "reason": reason, "metadata": metadata}
