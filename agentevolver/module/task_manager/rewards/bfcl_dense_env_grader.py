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
    """BFCL environment grader with outcome-aligned per-turn progress.

    Historical note
    ---------------
    The first version of this grader clamped every non-passing trajectory
    to ``0.0`` because BFCL's multi_turn accuracy is binary, producing a
    "dense" grader that was in fact sparse. The second version shaped the
    reward with *process* statistics (turn_coverage_rate, tool_call_accept_rate,
    tool_exec_success_rate). Those signals are not monotone-aligned with
    correctness: a model can score high on them by emitting any well-formed
    tool call, which actively hurts ``miss_func`` tasks where the correct
    behaviour is to ask rather than call.

    This version uses the *same* checker semantics as the final binary
    accuracy, evaluated per user turn. ``bfcl_progress`` is computed by
    ``env_handler`` via ``multi_turn_progress.safe_compute_progress`` and
    equals ``passed_gt_turns / scorable_gt_turns``. It is outcome-aligned,
    monotone in correctness, and not gameable by spamming tool calls.
    """

    def __init__(self, task: Task):
        super().__init__(task)
        self._config = None

    def set_config(self, config):
        self._config = config

    # ---------------------------------------------------------------- utils
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
        cap = float(_cfg_get(self._config, f"{cfg_base}.partial_credit_cap", 0.7))
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

    # ------------------------------------------------------------- scoring
    def calculate_reward(
        self, trajectory: Trajectory, env: EnvClient, instance_id: str
    ) -> GraderResult:
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

        # Outcome-aligned partial credit: fraction of GT user turns passed.
        # Populated by env_handler.evaluate() for multi_turn categories;
        # missing (single_turn / relevance) defaults to 0.0 and the branch
        # below falls back to the binary accuracy.
        progress = float(result.get("bfcl_progress", 0.0) or 0.0)
        progress = max(0.0, min(1.0, progress))
        progress_info = result.get("bfcl_progress_info", {}) or {}

        if raw_accuracy >= 1.0:
            # Full pass always scores 1.0, regardless of cap -- the cap only
            # bounds partial credit, not the reward of a fully correct
            # trajectory.
            score = 1.0
        else:
            score = min(cap, weight * progress)

        metadata = {
            "bfcl_dense_raw": {
                k: v for k, v in result.items() if k != "bfcl_progress_info"
            },
            "bfcl_dense_raw_accuracy": raw_accuracy,
            "bfcl_dense_cap": cap,
            "bfcl_dense_weight": weight,
            "bfcl_dense_completed": completed,
            "bfcl_dense_progress": progress,
            "bfcl_dense_progress_info": progress_info,
        }
        reason = (
            f"BFCL dense (per-turn progress): score={score:.4f}, "
            f"raw_accuracy={raw_accuracy:.4f}, progress={progress:.4f}, "
            f"cap={cap:.4f}, completed={completed}, "
            f"passed/scorable="
            f"{progress_info.get('passed_turns', '-')}/"
            f"{progress_info.get('scorable_turns', '-')}"
        )
        return {"score": score, "reason": reason, "metadata": metadata}
