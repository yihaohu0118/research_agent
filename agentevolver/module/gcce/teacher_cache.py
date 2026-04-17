"""Offline teacher-policy success cache for GCCE Delta_pi estimation.

We avoid online teacher rollouts during training for cost and determinism
reasons. A precompute script (``scripts/precompute_teacher_scores.py``)
evaluates a strong reference model (Qwen-Max / GPT / Claude / etc.) on the
training split once, under the *baseline* environment E_0, and dumps a
per-task success indicator. GCCE consumes that file here.

Schema of the JSON file::

    {
        "meta": {
            "teacher_model": "qwen-max-2025-09-xx",
            "env_type": "bfcl",
            "env_mode": "baseline",
            "num_tasks": 400
        },
        "scores": {
            "multi_turn_base_0": {"success": 1.0, "reward": 1.0},
            "multi_turn_miss_param_3": {"success": 0.0, "reward": 0.25},
            ...
        }
    }

When the file is missing or a task is not listed, the cache returns
``None`` for that task; Delta_pi estimation will gracefully fall back to
category-level failure rate (equivalent to PACE behaviour), so the training
run never crashes because of a missing teacher score.
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from typing import Any, Mapping, Optional

from loguru import logger

from agentevolver.module.tocf.category import infer_task_category


@dataclass
class TeacherScore:
    success: float = 0.0
    reward: float = 0.0


@dataclass
class TeacherCache:
    path: Optional[str] = None
    enabled: bool = False
    default_success: float = 0.0
    scores: dict[str, TeacherScore] = field(default_factory=dict)
    meta: dict[str, Any] = field(default_factory=dict)
    _category_cache: dict[str, tuple[float, int]] = field(default_factory=dict)

    @classmethod
    def from_config(cls, config: Any) -> "TeacherCache":
        if config is None:
            return cls()
        gcce_cfg = _cfg_get(config, "gcce", {}) or {}
        teacher_cfg = _cfg_get(gcce_cfg, "teacher", {}) or {}
        enabled = bool(_cfg_get(teacher_cfg, "enable", False))
        path = _cfg_get(teacher_cfg, "cache_path", None)
        default_success = float(_cfg_get(teacher_cfg, "default_success", 0.0))
        cache = cls(path=path, enabled=enabled, default_success=default_success)
        if enabled and path:
            cache.load()
        return cache

    def load(self) -> None:
        if not self.path or not os.path.exists(self.path):
            logger.warning(
                f"[GCCE] Teacher cache path missing or not found: {self.path}. "
                "Delta_pi estimation will fall back to failure-rate proxy."
            )
            self.scores = {}
            return
        with open(self.path, "r", encoding="utf-8") as f:
            blob = json.load(f)
        raw_scores = blob.get("scores", {}) or {}
        self.scores = {
            str(k): TeacherScore(
                success=float(v.get("success", 0.0)),
                reward=float(v.get("reward", v.get("success", 0.0))),
            )
            for k, v in raw_scores.items()
        }
        self.meta = dict(blob.get("meta", {}))
        logger.info(
            f"[GCCE] Loaded teacher cache: {len(self.scores)} tasks from {self.path}"
        )
        self._category_cache.clear()

    def get(self, task_id: str) -> Optional[TeacherScore]:
        if not self.enabled:
            return None
        return self.scores.get(str(task_id))

    def category_success_rate(
        self,
        tasks: list[Any] | None = None,
        env_type: str | None = None,
    ) -> dict[str, tuple[float, int]]:
        """Aggregate teacher success by TOCF category.

        Returns mapping ``{category: (mean_success, n_tasks)}``. If ``tasks`` is
        provided we iterate over the training split (so each category's mean is
        a proper expectation over the training population). Otherwise we
        aggregate over every scored task id, which is less faithful but works
        as a fallback.
        """
        if not self.enabled or not self.scores:
            return {}
        if self._category_cache and tasks is None:
            return dict(self._category_cache)

        buckets: dict[str, list[float]] = {}
        if tasks:
            for task in tasks:
                task_id = getattr(task, "task_id", None)
                if not task_id:
                    continue
                score = self.get(task_id)
                if score is None:
                    continue
                category = infer_task_category(
                    task_id,
                    env_type=env_type or getattr(task, "env_type", None),
                    metadata=getattr(task, "metadata", None),
                )
                buckets.setdefault(category, []).append(score.success)
        else:
            for task_id, score in self.scores.items():
                category = infer_task_category(task_id, env_type=env_type, metadata=None)
                buckets.setdefault(category, []).append(score.success)

        result = {
            category: (sum(values) / len(values), len(values))
            for category, values in buckets.items()
            if values
        }
        self._category_cache = dict(result)
        return result

    def metrics(self, prefix: str = "gcce/teacher") -> dict[str, float]:
        if not self.enabled or not self.scores:
            return {f"{prefix}/enabled": 0.0, f"{prefix}/num_tasks": 0.0}
        mean_success = sum(s.success for s in self.scores.values()) / len(self.scores)
        return {
            f"{prefix}/enabled": 1.0,
            f"{prefix}/num_tasks": float(len(self.scores)),
            f"{prefix}/mean_success": float(mean_success),
        }


def _cfg_get(config: Any, key: str, default: Any = None) -> Any:
    if config is None:
        return default
    if isinstance(config, Mapping):
        return config.get(key, default)
    return getattr(config, key, default)
