from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

from loguru import logger

from .stats import TOCFStats


def _cfg_get(config: Any, key: str, default=None):
    if config is None:
        return default
    if isinstance(config, dict):
        return config.get(key, default)
    return getattr(config, key, default)


@dataclass
class TOCFPatchDecision:
    category_weights: dict[str, float]
    accepted: bool
    reason: str
    complexity: float


class TOCFController:
    """T-Patch controller driven by TOCFStats.

    Configurable knobs (all backward compatible - old defaults reproduce the
    original failure-rate controller behavior):

      - ``objective`` ∈ {"failure_rate", "learning_progress"}
          * ``failure_rate`` (default): legacy behavior. weight ∝ 1 + α · (1 - p).
            Under sparse reward this tends to over-allocate budget to
            ``p ≈ 0`` categories where GRPO groups are all-zero and produce
            no gradient (zero-variance groups), which can actively hurt
            training. Kept as default to preserve the existing ablation rows.
          * ``learning_progress``: weight ∝ 1 + α · 4 · p · (1 - p). Peaks at
            p = 0.5 (maximum GRPO group variance) and decays at the saturated
            and dead ends. The factor 4 rescales so the lever matches the
            failure-rate formula at its peak.
      - ``prior_strength`` (β₀): Beta(β₀·μ, β₀·(1-μ)) shrinkage strength on
        the per-category success rate. Default ``0.0`` disables shrinkage so
        the legacy MLE estimator is unchanged.
      - ``prior_mean`` (μ): prior mean for the Beta shrinkage. Default ``0.5``.
      - ``ema_beta`` ∈ (0, 1]: inter-epoch smoothing. Default ``1.0`` reverts
        to the legacy "overwrite with this-epoch estimate" behavior. Smaller
        values damp epoch-to-epoch oscillation, e.g. 0.5 means 50/50 blend
        between the last accepted weight and the new target.
    """

    def __init__(self, config: Any = None):
        task_cfg = _cfg_get(config, "task_distribution", {})
        self.enabled = bool(_cfg_get(config, "enable", False)) and bool(_cfg_get(task_cfg, "enable", False))
        self.alpha = float(_cfg_get(task_cfg, "alpha", 1.0))
        self.min_weight = float(_cfg_get(task_cfg, "min_weight", 0.2))
        self.max_weight = float(_cfg_get(task_cfg, "max_weight", 3.0))
        self.min_samples = int(_cfg_get(task_cfg, "min_samples", 8))
        self.complexity_lambda = float(_cfg_get(task_cfg, "complexity_lambda", 0.0))
        self.accept_threshold = float(_cfg_get(task_cfg, "accept_threshold", -1.0))
        self.base_weights = dict(_cfg_get(task_cfg, "category_weights", {}) or {})

        objective = str(_cfg_get(task_cfg, "objective", "failure_rate")).lower()
        if objective not in ("failure_rate", "learning_progress"):
            logger.warning(
                f"[TOCF] Unknown objective '{objective}', falling back to 'failure_rate'."
            )
            objective = "failure_rate"
        self.objective = objective

        self.prior_strength = max(0.0, float(_cfg_get(task_cfg, "prior_strength", 0.0)))
        self.prior_mean = float(_cfg_get(task_cfg, "prior_mean", 0.5))
        self.prior_mean = min(1.0, max(0.0, self.prior_mean))

        ema_beta = float(_cfg_get(task_cfg, "ema_beta", 1.0))
        self.ema_beta = min(1.0, max(0.0, ema_beta))

        self.current_weights = dict(self.base_weights)
        self.last_decision: TOCFPatchDecision | None = None
        self.proposed_count = 0
        self.accepted_count = 0

    def _clamp(self, value: float) -> float:
        return min(self.max_weight, max(self.min_weight, value))

    def _shrunk_success_rate(self, success: float, count: float) -> float:
        """Apply Beta-shrinkage to the per-category success rate."""
        if self.prior_strength <= 0.0 or count <= 0:
            return float(success) / float(count) if count else 0.0
        alpha = self.prior_strength * self.prior_mean
        beta = self.prior_strength * (1.0 - self.prior_mean)
        return (float(success) + alpha) / (float(count) + alpha + beta)

    def _score(self, p: float) -> float:
        """Objective-dependent re-weighting score in [0, 1]."""
        p = min(1.0, max(0.0, float(p)))
        if self.objective == "learning_progress":
            # 4 * p * (1 - p) peaks at 1.0 when p=0.5 so that, at the peak,
            # the weight multiplier (1 + alpha * score) matches what the
            # failure_rate formula produces at p=0.
            return 4.0 * p * (1.0 - p)
        return 1.0 - p  # failure_rate

    def propose(self, stats: TOCFStats) -> TOCFPatchDecision | None:
        if not self.enabled:
            return None

        snapshot = stats.snapshot(window=True)
        categories = snapshot["categories"]
        proposed = dict(self.current_weights)

        for category, item in categories.items():
            count = int(item["count"])
            if count < self.min_samples:
                continue
            success = float(item["success_rate"]) * count  # recover raw success count
            p_hat = self._shrunk_success_rate(success, count)
            score = self._score(p_hat)
            base = float(self.base_weights.get(category, 1.0))
            target = self._clamp(base * (1.0 + self.alpha * score))
            if self.ema_beta < 1.0:
                prev = float(self.current_weights.get(category, base))
                blended = (1.0 - self.ema_beta) * prev + self.ema_beta * target
                proposed[category] = self._clamp(blended)
            else:
                proposed[category] = target

        complexity = sum(
            abs(float(proposed.get(k, 1.0)) - float(self.current_weights.get(k, 1.0)))
            for k in proposed
        )
        score_val = -self.complexity_lambda * complexity
        # Tolerance-aware diff check so floating point noise alone never
        # triggers an "accept" with identical weights.
        changed = any(
            not math.isclose(
                float(proposed.get(k, 1.0)),
                float(self.current_weights.get(k, 1.0)),
                rel_tol=1e-6,
                abs_tol=1e-6,
            )
            for k in proposed
        )
        accepted = score_val >= self.accept_threshold and changed
        reason = (
            f"objective={self.objective}, "
            f"ema_beta={self.ema_beta:.3f}, prior_strength={self.prior_strength:.2f}, "
            f"score={score_val:.4f}, complexity={complexity:.4f}, "
            f"categories={list(categories.keys())}"
        )
        return TOCFPatchDecision(
            category_weights=proposed,
            accepted=accepted,
            reason=reason,
            complexity=complexity,
        )

    def accept(self, decision: TOCFPatchDecision | None) -> TOCFPatchDecision | None:
        if decision is None:
            return None
        self.proposed_count += 1
        self.last_decision = decision
        if decision.accepted:
            self.accepted_count += 1
            self.current_weights = dict(decision.category_weights)
            logger.info(f"Accepted TOCF T-Patch: {decision.reason}, weights={self.current_weights}")
        else:
            logger.info(f"Rejected TOCF T-Patch: {decision.reason}")
        return decision

    def metrics(self, prefix: str = "tocf/controller") -> dict[str, float]:
        out = {
            f"{prefix}/proposed_count": float(self.proposed_count),
            f"{prefix}/accepted_count": float(self.accepted_count),
            f"{prefix}/acceptance_rate": (
                float(self.accepted_count) / float(self.proposed_count)
                if self.proposed_count
                else 0.0
            ),
        }
        # Surface the current weights so the dashboard can track them directly
        # without having to diff the accepted-weights log stream.
        for c, w in self.current_weights.items():
            out[f"{prefix}/weight/{c}"] = float(w)
        return out
