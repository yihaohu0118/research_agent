from __future__ import annotations

import copy
import random
from collections import Counter
from typing import Optional, Sequence

from loguru import logger

from agentevolver.module.task_manager.data_mixture import MixtureStrategy
from agentevolver.module.tocf.category import infer_task_category
from agentevolver.schema.task import TaskObjective


class AdaptiveMixtureStrategy(MixtureStrategy):
    """Mixture strategy with category-level weighted resampling for T-Patch."""

    def __init__(
        self,
        use_original: bool = True,
        synthetic_ratio: float = 0.0,
        category_weights: Optional[dict[str, float]] = None,
        shuffle: bool = True,
        seed: Optional[int] = None,
        target_size: Optional[int] = None,
        replacement: bool = True,
        default_weight: float = 1.0,
    ):
        if synthetic_ratio < 0:
            raise ValueError("synthetic_ratio must be non-negative")
        if default_weight < 0:
            raise ValueError("default_weight must be non-negative")

        self._use_original = use_original
        self._synthetic_ratio = synthetic_ratio
        self._category_weights = dict(category_weights or {})
        self._shuffle = shuffle
        self._seed = seed
        self._target_size = target_size
        self._replacement = replacement
        self._default_weight = default_weight
        self._rebuild_count = 0

    @property
    def need_synthetic(self) -> bool:
        return self._synthetic_ratio > 0

    @property
    def category_weights(self) -> dict[str, float]:
        return dict(self._category_weights)

    def set_category_weights(self, weights: dict[str, float]):
        self._category_weights = {str(k): max(0.0, float(v)) for k, v in dict(weights).items()}
        logger.info(f"TOCF category weights updated: {self._category_weights}")

    def _rng(self):
        if self._seed is None:
            return random
        return random.Random(self._seed + self._rebuild_count)

    def _category(self, item: TaskObjective) -> str:
        return infer_task_category(item.task.task_id, item.task.env_type, item.task.metadata)

    def _weight(self, item: TaskObjective) -> float:
        return max(0.0, float(self._category_weights.get(self._category(item), self._default_weight)))

    def _weighted_sample(self, items: Sequence[TaskObjective], count: int, rng) -> list[TaskObjective]:
        if count <= 0 or not items:
            return []

        weights = [self._weight(item) for item in items]
        if sum(weights) <= 0:
            weights = [1.0 for _ in items]

        if self._replacement or count > len(items):
            selected = rng.choices(list(items), weights=weights, k=count)
            return [copy.deepcopy(item) for item in selected]

        pool = list(items)
        pool_weights = list(weights)
        selected = []
        for _ in range(min(count, len(pool))):
            picked = rng.choices(range(len(pool)), weights=pool_weights, k=1)[0]
            selected.append(copy.deepcopy(pool.pop(picked)))
            pool_weights.pop(picked)
            if not pool:
                break
            if sum(pool_weights) <= 0:
                pool_weights = [1.0 for _ in pool]
        return selected

    def mix_data(
        self,
        synthetic_objectives: list[TaskObjective],
        original_tasks: Sequence[TaskObjective],
    ) -> list[TaskObjective]:
        rng = self._rng()
        self._rebuild_count += 1

        mixed: list[TaskObjective] = []
        original_target = self._target_size if self._target_size is not None else len(original_tasks)
        if self._use_original:
            mixed.extend(self._weighted_sample(original_tasks, original_target, rng))

        synthetic_target = int(len(original_tasks) * self._synthetic_ratio)
        if synthetic_target > 0:
            mixed.extend(self._weighted_sample(synthetic_objectives, synthetic_target, rng))

        if self._shuffle:
            rng.shuffle(mixed)

        logger.info(
            "TOCF mixture rebuilt: "
            f"#original_seed={len(original_tasks)}, #synthetic_seed={len(synthetic_objectives)}, "
            f"#mixed={len(mixed)}, distribution={dict(Counter(self._category(x) for x in mixed))}"
        )
        return mixed

    def __repr__(self):
        return (
            "AdaptiveMixtureStrategy("
            f"use_original={self._use_original}, synthetic_ratio={self._synthetic_ratio}, "
            f"category_weights={self._category_weights}, shuffle={self._shuffle}, "
            f"seed={self._seed}, target_size={self._target_size}, replacement={self._replacement})"
        )
