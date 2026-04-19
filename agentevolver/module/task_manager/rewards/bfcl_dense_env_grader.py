from __future__ import annotations

import re
from typing import Any

from agentevolver.client.env_client import EnvClient
from agentevolver.module.agent_flow.reward_calculator import GraderResult, RewardCalculator
from agentevolver.schema.task import Task
from agentevolver.schema.trajectory import Trajectory

from . import grader_manager


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


@grader_manager.reg("bfcl-dense-env")
class BfclDenseEnvGrader(RewardCalculator):
    """BFCL environment grader with capped objective partial credit."""

    def __init__(self, task: Task):
        super().__init__(task)
        self._config = None

    def set_config(self, config):
        self._config = config

    def _training_progress(self) -> float:
        epoch_value = (self.task.metadata or {}).get("epoch")
        if epoch_value is None:
            return 0.0

        epoch_index = None
        if isinstance(epoch_value, int):
            epoch_index = epoch_value
        elif isinstance(epoch_value, str):
            match = re.match(r"train\.(\d+)\.", epoch_value)
            if match:
                epoch_index = int(match.group(1))

        if epoch_index is None:
            return 0.0

        total_epochs = float(_cfg_get(self._config, "trainer.total_epochs", 1) or 1)
        return min(1.0, max(0.0, epoch_index / max(1.0, total_epochs)))

    def _partial_credit_cap(self) -> float:
        cfg_base = "tocf.feedback.dense_reward"
        cap = float(_cfg_get(self._config, f"{cfg_base}.partial_credit_cap", 0.5))
        if bool(_cfg_get(self._config, f"{cfg_base}.cap_decay", False)):
            min_cap = float(_cfg_get(self._config, f"{cfg_base}.min_cap", 0.1))
            cap = max(min_cap, cap * (1.0 - self._training_progress()))

        metadata_cfg = ((self.task.metadata or {}).get("tocf", {}) or {}).get("dense_reward", {})
        if "partial_credit_cap" in metadata_cfg:
            cap = float(metadata_cfg["partial_credit_cap"])
        return max(0.0, min(1.0, cap))

    def _partial_credit_weight(self) -> float:
        cfg_base = "tocf.feedback.dense_reward"
        weight = float(_cfg_get(self._config, f"{cfg_base}.partial_credit_weight", 1.0))
        metadata_cfg = ((self.task.metadata or {}).get("tocf", {}) or {}).get("dense_reward", {})
        if "partial_credit_weight" in metadata_cfg:
            weight = float(metadata_cfg["partial_credit_weight"])
        return max(0.0, weight)

    # ---------------------------------------------------------------- F-Patch
    # Multi-level partial credit.
    #
    # The original grader clamped every non-passing trajectory to 0.0 because
    # BFCL's multi_turn accuracy is 0/1. This gave a formally "dense" grader
    # that was in practice binary. The F-Patch restores a real dense signal
    # by combining four cheap per-rollout shape statistics emitted by
    # env_handler._diagnose_trajectory:
    #
    #   s1 = turn_coverage_rate         (made any call in each user turn)
    #   s2 = tool_call_accept_rate      (calls not rejected by parser/avail)
    #   s3 = tool_exec_success_rate     (executions that did not error)
    #   s4 = BFCL pass  (0/1)
    #
    # These four numbers are all in [0, 1] and are ordered by "closeness to
    # correctness". A weighted sum, capped at `partial_credit_cap` for any
    # non-passing trajectory, preserves the semantics you want: a passing
    # rollout still gets 1.0, a completely silent rollout gets ~0, and
    # intermediate shapes get intermediate reward. The cap stays configurable
    # so it can be dropped back to 0 for a fair F-Patch-off ablation.
    _F_WEIGHTS = (
        ("turn_coverage_rate", 0.25),
        ("tool_call_accept_rate", 0.25),
        ("tool_exec_success_rate", 0.50),
    )

    def _partial_signal(self, result: dict) -> tuple[float, dict]:
        components = {}
        accum = 0.0
        for key, weight in self._F_WEIGHTS:
            raw_key = f"trajectory_{key}" if not key.startswith("trajectory_") else key
            value = float(result.get(raw_key, 0.0) or 0.0)
            value = max(0.0, min(1.0, value))
            components[key] = value
            accum += weight * value
        return accum, components

    def calculate_reward(self, trajectory: Trajectory, env: EnvClient, instance_id: str) -> GraderResult:
        result = env.evaluate(instance_id, params={"sparse": False})
        if isinstance(result, dict):
            raw_accuracy = float(result.get("accuracy", 0.0) or 0.0)
            completed = bool(result.get("completed", False))
        else:
            raw_accuracy = float(result or 0.0)
            completed = raw_accuracy >= 1.0
            result = {"accuracy": raw_accuracy, "completed": completed}

        cap = self._partial_credit_cap()
        weight = self._partial_credit_weight()

        if raw_accuracy >= 1.0:
            score = 1.0
            partial_value = 1.0
            components = {k: 1.0 for k, _ in self._F_WEIGHTS}
        else:
            partial_value, components = self._partial_signal(
                result if isinstance(result, dict) else {}
            )
            score = min(cap, weight * partial_value)

        metadata = {
            "bfcl_dense_raw": result,
            "bfcl_dense_raw_accuracy": raw_accuracy,
            "bfcl_dense_cap": cap,
            "bfcl_dense_completed": completed,
            "bfcl_dense_partial_value": float(partial_value),
            "bfcl_dense_components": components,
        }
        reason = (
            f"BFCL dense F-Patch: score={score:.4f}, "
            f"raw_accuracy={raw_accuracy:.4f}, partial={partial_value:.4f}, "
            f"cap={cap:.4f}, completed={completed}"
        )
        return {"score": score, "reason": reason, "metadata": metadata}
