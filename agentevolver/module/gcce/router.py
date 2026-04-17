"""Gap-conditioned routing for GCCE.

Turns per-category (Delta_E, Delta_pi) pairs into two scalar weights

    r_E(c)  = Delta_E(c) / (Delta_E(c) + Delta_pi(c) + eta)
    r_pi(c) = 1 - r_E(c)

These weights are fed to two downstream modules:

    * the **TOCF controller** uses ``r_E(c)`` as the category-level pressure
      for its T-Patch proposal, so environment edits only happen where the
      failure is identifiable as an environment gap;
    * the **policy-side advantage reweighter** (see ``advantage_weighting``)
      uses ``r_pi(c)`` to up-weight positive advantages in categories where
      the bottleneck is identifiable as a policy gap.

The router also emits an *absolute* patch-budget signal ``b(c)`` proportional
to Delta_E(c) * p(c), which the TOCF controller can consume to decide how
aggressively to modify a given category.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping, Optional

from loguru import logger

from .gap_attributor import CategoryGap


@dataclass
class RouteDecision:
    r_env: dict[str, float] = field(default_factory=dict)
    r_policy: dict[str, float] = field(default_factory=dict)
    patch_budget: dict[str, float] = field(default_factory=dict)
    env_category_weights: dict[str, float] = field(default_factory=dict)
    total_regret_bound: float = 0.0
    summary: str = ""


class GCCERouter:
    def __init__(self, config: Any = None):
        gcce_cfg = _cfg_get(config, "gcce", {}) or {}
        router_cfg = _cfg_get(gcce_cfg, "router", {}) or {}
        self.enabled = bool(_cfg_get(gcce_cfg, "enable", False))
        self.eta = float(_cfg_get(router_cfg, "eta", 1e-3))
        self.alpha_env = float(_cfg_get(router_cfg, "alpha_env", 2.0))
        self.alpha_policy = float(_cfg_get(router_cfg, "alpha_policy", 1.0))
        self.min_weight = float(_cfg_get(router_cfg, "min_weight", 0.2))
        self.max_weight = float(_cfg_get(router_cfg, "max_weight", 3.0))
        self.base_weights: dict[str, float] = dict(
            _cfg_get(router_cfg, "base_weights", {}) or {}
        )
        self.total_budget = float(_cfg_get(router_cfg, "total_budget", 1.0))
        self.latest: RouteDecision | None = None

    def route(self, gaps: Mapping[str, CategoryGap], category_probs: Mapping[str, float] | None = None) -> RouteDecision:
        if not self.enabled or not gaps:
            self.latest = RouteDecision(summary="disabled_or_empty")
            return self.latest

        r_env: dict[str, float] = {}
        r_pol: dict[str, float] = {}
        env_weights: dict[str, float] = {}
        patch_budget: dict[str, float] = {}

        total_gap = sum(max(0.0, g.delta_e) * _cat_prob(category_probs, c) for c, g in gaps.items())

        for category, gap in gaps.items():
            denom = max(0.0, gap.delta_e) + max(0.0, gap.delta_pi) + self.eta
            r_e = max(0.0, gap.delta_e) / denom if denom > 0 else 0.0
            r_p = 1.0 - r_e
            r_env[category] = float(r_e)
            r_pol[category] = float(r_p)

            base_w = float(self.base_weights.get(category, 1.0))
            env_weights[category] = _clip(
                base_w * (1.0 + self.alpha_env * r_e * gap.delta_e),
                self.min_weight,
                self.max_weight,
            )

            pc = _cat_prob(category_probs, category)
            if total_gap > 0:
                patch_budget[category] = float(
                    self.total_budget * (max(0.0, gap.delta_e) * pc) / total_gap
                )
            else:
                patch_budget[category] = 0.0

        summary = ", ".join(
            f"{c}(r_E={r_env[c]:.2f}, r_π={r_pol[c]:.2f}, w_env={env_weights[c]:.2f}, b={patch_budget[c]:.2f})"
            for c in gaps
        )
        total_bound = sum(max(0.0, g.delta_e) + max(0.0, g.delta_pi) for g in gaps.values())
        logger.info(f"[GCCE] Route decision: {summary} | total_regret_bound={total_bound:.3f}")

        self.latest = RouteDecision(
            r_env=r_env,
            r_policy=r_pol,
            patch_budget=patch_budget,
            env_category_weights=env_weights,
            total_regret_bound=float(total_bound),
            summary=summary,
        )
        return self.latest

    # ------------------------------------------------------------------ accessors
    def r_policy(self, category: str) -> float:
        if self.latest is None:
            return 0.0
        return float(self.latest.r_policy.get(category, 0.0))

    def r_env(self, category: str) -> float:
        if self.latest is None:
            return 0.0
        return float(self.latest.r_env.get(category, 0.0))

    def advantage_weight(self, category: str) -> float:
        """Multiplicative weight applied to positive advantages by the policy side."""
        if self.latest is None:
            return 1.0
        r_p = self.latest.r_policy.get(category, 0.0)
        return _clip(
            1.0 + self.alpha_policy * float(r_p),
            self.min_weight,
            self.max_weight,
        )

    def env_category_weights(self) -> dict[str, float]:
        if self.latest is None:
            return {}
        return dict(self.latest.env_category_weights)

    # ------------------------------------------------------------------ logging
    def metrics(self, prefix: str = "gcce/router") -> dict[str, float]:
        result: dict[str, float] = {f"{prefix}/enabled": 1.0 if self.enabled else 0.0}
        if self.latest is None:
            return result
        result[f"{prefix}/total_regret_bound"] = float(self.latest.total_regret_bound)
        for category, value in self.latest.r_env.items():
            safe = str(category).replace("/", "_")
            result[f"{prefix}/{safe}/r_env"] = float(value)
            result[f"{prefix}/{safe}/r_policy"] = float(self.latest.r_policy.get(category, 0.0))
            result[f"{prefix}/{safe}/patch_budget"] = float(self.latest.patch_budget.get(category, 0.0))
            result[f"{prefix}/{safe}/env_weight"] = float(self.latest.env_category_weights.get(category, 1.0))
        return result


def _clip(value: float, lo: float, hi: float) -> float:
    return float(min(hi, max(lo, value)))


def _cat_prob(category_probs: Mapping[str, float] | None, category: str) -> float:
    if not category_probs:
        return 1.0
    return float(category_probs.get(category, 1.0))


def _cfg_get(config: Any, key: str, default: Any = None) -> Any:
    if config is None:
        return default
    if isinstance(config, Mapping):
        return config.get(key, default)
    return getattr(config, key, default)
