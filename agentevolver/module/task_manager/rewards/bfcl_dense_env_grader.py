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

    Two modes are supported via ``tocf.feedback.dense_reward.mode``:

    * ``capped`` (default, legacy): ``score = min(cap, weight * progress)``
      when the trajectory does not fully pass, ``1.0`` otherwise.
      The cap defaults to ``0.7``.

    * ``t3rl``: direct mirror of the reward shaping used in
      ``/Users/huyihao/Desktop/T3RL-main/t3rl/rollout/bfcl.py:res_to_sample``.
      ``score = progress_with_irrelevance`` (no cap, no weight). The
      ``progress_with_irrelevance`` metric — computed in
      ``env_service/environments/bfcl/multi_turn_progress.py`` — augments
      the legacy turn-pass ratio with an irrelevance-turn component that
      rewards correctly abstaining (score 1.0 when the model does not call
      any tool on a GT-empty turn) and penalises spurious tool calls
      (score 0.0). This is a strictly better dense signal for
      ``multi_turn_miss_func`` because the correct behaviour — *not*
      calling a tool — now receives an on-policy gradient rather than
      being invisible.

    Historical note
    ---------------
    The first version of this grader clamped every non-passing trajectory
    to ``0.0`` because BFCL's multi_turn accuracy is binary, producing a
    "dense" grader that was in fact sparse. The second version shaped the
    reward with *process* statistics (turn_coverage_rate, tool_call_accept_rate,
    tool_exec_success_rate). Those signals are not monotone-aligned with
    correctness: a model can score high on them by emitting any well-formed
    tool call, which actively hurts ``miss_func`` tasks where the correct
    behaviour is to ask rather than call. The current capped/t3rl modes
    use the *same* checker semantics as the final binary accuracy, just
    evaluated per user turn.
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

    def _mode(self) -> str:
        """Reward-shaping mode. ``capped`` keeps the legacy min(cap, w*p)
        formula; ``t3rl`` mirrors T3RL-main's uncapped progress-with-
        irrelevance signal. Task metadata can override the config so
        individual tasks can opt into a different mode if needed.
        """
        cfg_base = "tocf.feedback.dense_reward"
        mode = str(_cfg_get(self._config, f"{cfg_base}.mode", "capped") or "capped").lower()
        metadata_cfg = ((self.task.metadata or {}).get("tocf", {}) or {}).get("dense_reward", {})
        if "mode" in metadata_cfg:
            mode = str(metadata_cfg["mode"] or "capped").lower()
        if mode not in ("capped", "t3rl"):
            mode = "capped"
        return mode

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

        # Outcome-aligned partial credit: fraction of GT user turns passed.
        # Populated by env_handler.evaluate() for multi_turn categories;
        # missing (single_turn / relevance) defaults to 0.0 and the branch
        # below falls back to the binary accuracy.
        progress = float(result.get("bfcl_progress", 0.0) or 0.0)
        progress = max(0.0, min(1.0, progress))
        # T3RL-style metric that also scores irrelevance turns. Falls back
        # to ``progress`` when env_handler did not emit it (e.g. older
        # cache entries or single-turn categories).
        progress_t3rl = float(
            result.get("bfcl_progress_with_irrelevance", progress) or 0.0
        )
        progress_t3rl = max(0.0, min(1.0, progress_t3rl))
        progress_info = result.get("bfcl_progress_info", {}) or {}

        mode = self._mode()

        if mode == "t3rl":
            # T3RL-main training reward: uncapped progress with irrelevance.
            # Full-pass safety net: BFCL's binary accuracy is authoritative,
            # so we still clamp a fully-correct trajectory to 1.0 in the
            # rare case where the progress scorer disagreed.
            score = 1.0 if raw_accuracy >= 1.0 else progress_t3rl
            cap = 1.0
            weight = 1.0
        else:
            # Legacy capped mode.
            cap = self._partial_credit_cap()
            weight = self._partial_credit_weight()
            if raw_accuracy >= 1.0:
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
            "bfcl_dense_progress_t3rl": progress_t3rl,
            "bfcl_dense_mode": mode,
            "bfcl_dense_progress_info": progress_info,
        }
        if mode == "t3rl":
            reason = (
                f"BFCL dense[t3rl]: score={score:.4f}, "
                f"raw_accuracy={raw_accuracy:.4f}, "
                f"progress_with_irrelevance={progress_t3rl:.4f}, "
                f"legacy_progress={progress:.4f}, completed={completed}, "
                f"passed/scorable="
                f"{progress_info.get('passed_turns', '-')}/"
                f"{progress_info.get('scorable_turns', '-')}, "
                f"irrelevance(passed/total)="
                f"{progress_info.get('passed_irrelevance_turns', '-')}/"
                f"{progress_info.get('irrelevance_turns', '-')}"
            )
        else:
            reason = (
                f"BFCL dense[capped]: score={score:.4f}, "
                f"raw_accuracy={raw_accuracy:.4f}, progress={progress:.4f}, "
                f"cap={cap:.4f}, completed={completed}, "
                f"passed/scorable="
                f"{progress_info.get('passed_turns', '-')}/"
                f"{progress_info.get('scorable_turns', '-')}"
            )
        return {"score": score, "reason": reason, "metadata": metadata}
