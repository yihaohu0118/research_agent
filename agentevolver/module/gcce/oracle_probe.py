"""Oracle environment probe for GCCE Delta_E estimation.

An *oracle environment* E* is the current environment with the TOCF O/C/F
patches pushed to their maximal setting: schema is maximally exposed, action
constraints validate every tool call, and the feedback channel returns dense
partial credit plus recovery hints. It represents an upper bound on how much
pure environment shaping can help the *current* policy.

Running a full oracle rollout for every training step is prohibitive. We
therefore (i) maintain an *override config* that, when applied, makes the
environment service behave as E*, and (ii) accept a thin callback from the
trainer that performs a small probe batch (e.g. 16 held-out tasks) every K
epochs. The probe result is a per-category success rate that we expose as
``Delta_E(c) = success_oracle(c) - success_current(c)`` to the router.

When no callback is wired up, :class:`OracleProbe` degrades to a static
``oracle_success_prior`` config value (default = 1.0), which means "treat the
oracle upper bound as perfect" -- conservative from a router standpoint, and
matches the assumption that TOCF can always close most of the environment
gap if pushed to maximum exposure.
"""
from __future__ import annotations

import copy
import json
import os
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Mapping, Optional

from loguru import logger


ProbeFn = Callable[[dict[str, Any]], dict[str, tuple[float, int]]]
# Signature: probe_fn(overrides) -> {category: (success_rate, n_samples)}


@dataclass
class OracleProbeResult:
    timestamp: float = 0.0
    global_step: int = 0
    category_success: dict[str, tuple[float, int]] = field(default_factory=dict)

    def to_json(self) -> dict[str, Any]:
        return {
            "timestamp": self.timestamp,
            "global_step": self.global_step,
            "category_success": {
                k: {"success_rate": v[0], "count": v[1]}
                for k, v in self.category_success.items()
            },
        }


class OracleProbe:
    """Schedules and caches the oracle-env probe used for Delta_E estimation."""

    def __init__(self, config: Any = None, probe_fn: ProbeFn | None = None):
        gcce_cfg = _cfg_get(config, "gcce", {}) or {}
        oracle_cfg = _cfg_get(gcce_cfg, "oracle_env", {}) or {}
        self.enabled = bool(_cfg_get(oracle_cfg, "enable", False))
        self.probe_every_epochs = int(_cfg_get(oracle_cfg, "probe_every_epochs", 2))
        self.probe_sample_size = int(_cfg_get(oracle_cfg, "probe_sample_size", 16))
        self.oracle_success_prior = float(_cfg_get(oracle_cfg, "oracle_success_prior", 1.0))
        self.dump_dir: Optional[str] = _cfg_get(oracle_cfg, "dump_dir", None)
        self.overrides: dict[str, Any] = dict(
            _cfg_get(oracle_cfg, "overrides", {}) or {}
        )
        self.probe_fn = probe_fn
        self.latest: OracleProbeResult | None = None
        self.last_probe_epoch: int = -1
        self.probe_count: int = 0

    def should_probe(self, epoch: int) -> bool:
        if not self.enabled:
            return False
        if self.probe_fn is None:
            return False
        if self.last_probe_epoch < 0:
            return True
        return (epoch - self.last_probe_epoch) >= self.probe_every_epochs

    def set_probe_fn(self, probe_fn: ProbeFn) -> None:
        self.probe_fn = probe_fn

    def build_oracle_overrides(self) -> dict[str, Any]:
        """Return a shallow copy of the override dict that turns E_t into E*.

        The override dict is applied by the probe callback to the environment
        service configuration. For BFCL the conventional overrides are
        ``tocf.query_suffix.enable = true``, ``feedback.dense_reward.enable =
        true``, and a higher ``partial_credit_cap``. Callers can extend this
        mapping from config.
        """
        default_overrides = {
            "tocf.query_suffix.enable": True,
            "tocf.query_suffix.apply_to_validation": True,
            "tocf.feedback.dense_reward.enable": True,
            "tocf.feedback.dense_reward.partial_credit_cap": 0.8,
            "tocf.feedback.dense_reward.partial_credit_weight": 1.0,
        }
        merged = dict(default_overrides)
        merged.update(self.overrides)
        return merged

    def run_probe(self, epoch: int, global_step: int) -> OracleProbeResult | None:
        if not self.enabled or self.probe_fn is None:
            return None
        overrides = self.build_oracle_overrides()
        logger.info(
            f"[GCCE] Oracle probe @ epoch={epoch} step={global_step} "
            f"overrides={list(overrides.keys())}"
        )
        category_success = self.probe_fn(overrides)
        result = OracleProbeResult(
            timestamp=time.time(),
            global_step=global_step,
            category_success=dict(category_success or {}),
        )
        self.latest = result
        self.last_probe_epoch = epoch
        self.probe_count += 1
        self._dump(epoch, result)
        return result

    def oracle_success_rate(self, category: str) -> Optional[float]:
        if self.latest is None or not self.latest.category_success:
            return None
        item = self.latest.category_success.get(category)
        if not item:
            return None
        return float(item[0])

    def prior(self) -> float:
        """Fallback oracle success rate when no probe has been run yet."""
        return self.oracle_success_prior

    def metrics(self, prefix: str = "gcce/oracle") -> dict[str, float]:
        result = {
            f"{prefix}/enabled": 1.0 if self.enabled else 0.0,
            f"{prefix}/probe_count": float(self.probe_count),
        }
        if self.latest is not None:
            for category, (success_rate, count) in self.latest.category_success.items():
                safe = str(category).replace("/", "_")
                result[f"{prefix}/{safe}/success_rate"] = float(success_rate)
                result[f"{prefix}/{safe}/count"] = float(count)
        return result

    def _dump(self, epoch: int, result: OracleProbeResult) -> None:
        if not self.dump_dir:
            return
        os.makedirs(self.dump_dir, exist_ok=True)
        path = os.path.join(self.dump_dir, f"oracle_probe_epoch_{epoch}.json")
        with open(path, "w", encoding="utf-8") as f:
            json.dump(result.to_json(), f, ensure_ascii=False, indent=2)


def _cfg_get(config: Any, key: str, default: Any = None) -> Any:
    if config is None:
        return default
    if isinstance(config, Mapping):
        return config.get(key, default)
    return getattr(config, key, default)
