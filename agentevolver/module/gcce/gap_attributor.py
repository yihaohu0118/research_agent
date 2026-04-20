"""Counterfactual Gap Attribution (CGA).

For each TOCF category c we estimate two reference quantities per epoch:

    Delta_E(c) = J(E*, pi_t | c) - J(E_t, pi_t | c)
    Delta_pi(c) = J(E_t, pi_teacher | c) - J(E_t, pi_t | c)

These are upper bounds under Assumptions (A1)-(A3) in the GCCE paper.
Estimators are deliberately conservative:

    hat_Delta_E(c) = clamp(oracle_success(c) - current_success(c), 0, 1)
    hat_Delta_pi(c) = clamp(teacher_success(c) - current_success(c), 0, 1)

We apply Bayesian shrinkage towards a global mean so that small-sample
categories do not dominate the routing decision:

    tilde_Delta(c) = (n_c * hat_Delta(c) + n0 * mean_Delta) / (n_c + n0)

where n0 is ``prior_strength`` from config.

The attributor is deliberately pure (no side effects). It consumes a TOCF
stats snapshot plus the teacher cache / oracle probe and emits a per-category
:class:`CategoryGap`. Downstream, :class:`~agentevolver.module.gcce.router.GCCERouter`
turns the gap pair into routing weights ``r_E, r_pi``.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Iterable, Mapping, Optional

from loguru import logger

from .oracle_probe import OracleProbe
from .teacher_cache import TeacherCache


@dataclass
class CategoryGap:
    category: str
    count: int = 0
    current_success: float = 0.0
    oracle_success: Optional[float] = None
    teacher_success: Optional[float] = None
    delta_e_raw: float = 0.0
    delta_pi_raw: float = 0.0
    delta_e: float = 0.0
    delta_pi: float = 0.0

    def total(self) -> float:
        return max(0.0, self.delta_e) + max(0.0, self.delta_pi)


class GapAttributor:
    """Combine TOCFStats + teacher cache + oracle probe into CategoryGaps."""

    def __init__(
        self,
        config: Any = None,
        teacher_cache: TeacherCache | None = None,
        oracle_probe: OracleProbe | None = None,
        hindsight_router: Any = None,
    ):
        gcce_cfg = _cfg_get(config, "gcce", {}) or {}
        attr_cfg = _cfg_get(gcce_cfg, "attribution", {}) or {}
        self.enabled = bool(_cfg_get(gcce_cfg, "enable", False))
        self.min_samples = int(_cfg_get(attr_cfg, "min_samples", 4))
        self.prior_strength = float(_cfg_get(attr_cfg, "prior_strength", 4.0))
        self.use_total_stats = bool(_cfg_get(attr_cfg, "use_total_stats", False))
        self.clip_min = float(_cfg_get(attr_cfg, "clip_min", 0.0))
        self.clip_max = float(_cfg_get(attr_cfg, "clip_max", 1.0))
        # "heuristic" (legacy) uses reward_mean + teacher_cache as proxies.
        # "hindsight" uses per-category cap_gap_rate / env_gap_rate from the
        # on-policy attribution router — the identifiable GCCE path.
        self.attribution_source = str(
            _cfg_get(attr_cfg, "source", "heuristic")
        ).lower()
        self.teacher_cache = teacher_cache
        self.oracle_probe = oracle_probe
        self.hindsight_router = hindsight_router
        self.latest: dict[str, CategoryGap] = {}

    def attribute(self, tocf_stats: Any, tasks: Iterable[Any] | None = None, env_type: str | None = None) -> dict[str, CategoryGap]:
        """Returns a dict ``{category: CategoryGap}`` for the current epoch.

        Missing teacher / oracle signals are handled gracefully:
        Delta_pi defaults to category failure rate (PACE-equivalent);
        Delta_E defaults to the oracle probe prior (e.g. 1.0) times the
        failure rate, so the two always have comparable scale.
        """
        if not self.enabled or tocf_stats is None:
            self.latest = {}
            return {}

        snapshot = tocf_stats.snapshot(window=not self.use_total_stats)
        categories = snapshot.get("categories", {}) or {}

        teacher_rates = {}
        if self.teacher_cache is not None:
            teacher_rates = self.teacher_cache.category_success_rate(
                tasks=list(tasks) if tasks else None, env_type=env_type
            )

        # On-policy hindsight signal (only when source="hindsight" and a
        # router is wired in). Keys: category -> {cap_gap_rate, env_gap_rate,
        # sample_count}.
        hindsight_rates: dict[str, dict[str, float]] = {}
        if self.attribution_source == "hindsight" and self.hindsight_router is not None:
            try:
                hindsight_rates = self.hindsight_router.category_rates() or {}
            except Exception as exc:  # pragma: no cover - defensive
                logger.warning(f"[GCCE] hindsight router failed: {exc}")
                hindsight_rates = {}

        raw: dict[str, CategoryGap] = {}
        for category, item in categories.items():
            count = int(item.get("count", 0))
            if count < self.min_samples:
                continue
            current_success = float(item.get("success_rate", 0.0))
            # F-Patch dense-reward average. When F-Patch is off this equals the
            # success rate; when F-Patch is on it sits strictly above success
            # rate whenever some trajectories collect partial credit.
            reward_mean = float(item.get("reward_mean", current_success))
            failure_rate = max(0.0, 1.0 - current_success)

            oracle_success: Optional[float] = None
            teacher_success = (
                teacher_rates.get(category, (None, 0))[0] if teacher_rates else None
            )

            if self.attribution_source == "hindsight" and category in hindsight_rates:
                # ── Identifiable mode ─────────────────────────────────────
                # The router's on-policy reflection gives a direct estimate of
                # P(failure classified as capability gap | fail). Multiply by
                # the category failure rate to get a gap-mass estimate on the
                # same [0,1] scale as the heuristic branch.
                rates = hindsight_rates[category]
                cap_rate = max(0.0, min(1.0, float(rates.get("cap_gap_rate", 0.0))))
                env_rate = max(0.0, min(1.0, float(rates.get("env_gap_rate", 0.0))))
                delta_pi_raw = failure_rate * cap_rate
                delta_e_raw = failure_rate * env_rate
                delta_e_source = "hindsight"
                delta_pi_source = "hindsight"
            else:
                # ── Heuristic (legacy) fallback ───────────────────────────
                # Delta_E prefers oracle_probe, then F-Patch reward mean;
                # Delta_pi prefers teacher_cache, then failure residual.
                delta_e_source = "none"
                if self.oracle_probe is not None:
                    oracle_success = self.oracle_probe.oracle_success_rate(category)
                    if oracle_success is not None:
                        delta_e_source = "oracle_probe"
                if oracle_success is not None:
                    delta_e_raw = max(0.0, oracle_success - current_success)
                elif reward_mean > current_success + 1e-6:
                    delta_e_raw = min(failure_rate, reward_mean - current_success)
                    delta_e_source = "f_patch_reward_mean"
                else:
                    delta_e_raw = 0.0
                    delta_e_source = "zero_fallback"

                delta_pi_source = "none"
                if teacher_success is not None:
                    delta_pi_raw = max(0.0, float(teacher_success) - current_success)
                    delta_pi_source = "teacher_cache"
                else:
                    delta_pi_raw = max(0.0, failure_rate - delta_e_raw)
                    delta_pi_source = (
                        "failure_residual" if delta_e_raw > 0 else "failure_rate"
                    )

            raw[category] = CategoryGap(
                category=category,
                count=count,
                current_success=current_success,
                oracle_success=oracle_success,
                teacher_success=teacher_success,
                delta_e_raw=float(delta_e_raw),
                delta_pi_raw=float(delta_pi_raw),
            )
            # Stash provenance for logging; not part of the dataclass schema.
            setattr(raw[category], "_delta_e_source", delta_e_source)
            setattr(raw[category], "_delta_pi_source", delta_pi_source)

        if not raw:
            self.latest = {}
            return {}

        mean_e = sum(g.delta_e_raw for g in raw.values()) / len(raw)
        mean_pi = sum(g.delta_pi_raw for g in raw.values()) / len(raw)
        n0 = max(1e-6, self.prior_strength)

        for gap in raw.values():
            nc = max(1, gap.count)
            shrunk_e = (nc * gap.delta_e_raw + n0 * mean_e) / (nc + n0)
            shrunk_pi = (nc * gap.delta_pi_raw + n0 * mean_pi) / (nc + n0)
            gap.delta_e = min(self.clip_max, max(self.clip_min, float(shrunk_e)))
            gap.delta_pi = min(self.clip_max, max(self.clip_min, float(shrunk_pi)))

        self.latest = raw
        logger.info(
            "[GCCE] Gap attribution: "
            + ", ".join(
                f"{c}(count={g.count}, sr={g.current_success:.3f}, "
                f"dE={g.delta_e:.3f}, dπ={g.delta_pi:.3f})"
                for c, g in raw.items()
            )
        )
        return raw

    def metrics(self, prefix: str = "gcce/gap") -> dict[str, float]:
        result: dict[str, float] = {}
        if not self.latest:
            return result
        total_e = sum(g.delta_e for g in self.latest.values())
        total_pi = sum(g.delta_pi for g in self.latest.values())
        result[f"{prefix}/mean_delta_e"] = float(total_e / len(self.latest))
        result[f"{prefix}/mean_delta_pi"] = float(total_pi / len(self.latest))
        result[f"{prefix}/total_regret_bound"] = float(total_e + total_pi)
        for category, gap in self.latest.items():
            safe = str(category).replace("/", "_")
            base = f"{prefix}/{safe}"
            result[f"{base}/delta_e"] = float(gap.delta_e)
            result[f"{base}/delta_pi"] = float(gap.delta_pi)
            result[f"{base}/current_success"] = float(gap.current_success)
            result[f"{base}/count"] = float(gap.count)
            if gap.teacher_success is not None:
                result[f"{base}/teacher_success"] = float(gap.teacher_success)
            if gap.oracle_success is not None:
                result[f"{base}/oracle_success"] = float(gap.oracle_success)
            de_src = getattr(gap, "_delta_e_source", None)
            dpi_src = getattr(gap, "_delta_pi_source", None)
            if de_src:
                # 0=none, 1=oracle_probe, 2=f_patch_reward_mean,
                # 3=zero_fallback, 4=hindsight
                result[f"{base}/delta_e_source_id"] = float(
                    {
                        "none": 0,
                        "oracle_probe": 1,
                        "f_patch_reward_mean": 2,
                        "zero_fallback": 3,
                        "hindsight": 4,
                    }.get(de_src, 0)
                )
            if dpi_src:
                # 0=none, 1=teacher_cache, 2=failure_residual,
                # 3=failure_rate, 4=hindsight
                result[f"{base}/delta_pi_source_id"] = float(
                    {
                        "none": 0,
                        "teacher_cache": 1,
                        "failure_residual": 2,
                        "failure_rate": 3,
                        "hindsight": 4,
                    }.get(dpi_src, 0)
                )
        return result


def _cfg_get(config: Any, key: str, default: Any = None) -> Any:
    if config is None:
        return default
    if isinstance(config, Mapping):
        return config.get(key, default)
    return getattr(config, key, default)
