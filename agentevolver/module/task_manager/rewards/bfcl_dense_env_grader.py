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

    def calculate_reward(self, trajectory: Trajectory, env: EnvClient, instance_id: str) -> GraderResult:
        result = env.evaluate(instance_id, params={"sparse": False})
        if isinstance(result, dict):
            raw_accuracy = float(result.get("accuracy", 0.0) or 0.0)
            completed = bool(result.get("completed", False))
        else:
            raw_accuracy = float(result or 0.0)
            completed = raw_accuracy >= 1.0
            result = {"accuracy": raw_accuracy, "completed": completed}

        if raw_accuracy >= 1.0:
            score = 1.0
        else:
            score = min(self._partial_credit_cap(), raw_accuracy * self._partial_credit_weight())

        metadata = {
            "bfcl_dense_raw": result,
            "bfcl_dense_raw_accuracy": raw_accuracy,
            "bfcl_dense_cap": self._partial_credit_cap(),
            "bfcl_dense_completed": completed,
        }
        reason = (
            f"BFCL dense env reward: score={score:.4f}, "
            f"raw_accuracy={raw_accuracy:.4f}, cap={metadata['bfcl_dense_cap']:.4f}, completed={completed}"
        )
        return {"score": score, "reason": reason, "metadata": metadata}
