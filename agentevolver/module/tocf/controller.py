from __future__ import annotations

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
    """Small deterministic T-Patch controller driven by TOCFStats."""

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
        self.current_weights = dict(self.base_weights)
        self.last_decision: TOCFPatchDecision | None = None
        self.proposed_count = 0
        self.accepted_count = 0

    def _clamp(self, value: float) -> float:
        return min(self.max_weight, max(self.min_weight, value))

    def propose(self, stats: TOCFStats) -> TOCFPatchDecision | None:
        if not self.enabled:
            return None

        snapshot = stats.snapshot(window=True)
        categories = snapshot["categories"]
        proposed = dict(self.current_weights)

        for category, item in categories.items():
            if int(item["count"]) < self.min_samples:
                continue
            failure_rate = 1.0 - float(item["success_rate"])
            base = float(self.base_weights.get(category, 1.0))
            proposed[category] = self._clamp(base * (1.0 + self.alpha * failure_rate))

        complexity = sum(abs(float(proposed.get(k, 1.0)) - float(self.current_weights.get(k, 1.0))) for k in proposed)
        score = -self.complexity_lambda * complexity
        accepted = score >= self.accept_threshold and proposed != self.current_weights
        reason = f"score={score:.4f}, complexity={complexity:.4f}, categories={list(categories.keys())}"
        return TOCFPatchDecision(category_weights=proposed, accepted=accepted, reason=reason, complexity=complexity)

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
        return {
            f"{prefix}/proposed_count": float(self.proposed_count),
            f"{prefix}/accepted_count": float(self.accepted_count),
            f"{prefix}/acceptance_rate": (
                float(self.accepted_count) / float(self.proposed_count)
                if self.proposed_count
                else 0.0
            ),
        }
