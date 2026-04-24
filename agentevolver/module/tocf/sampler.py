from __future__ import annotations

import copy
import math
import random
from collections import Counter, defaultdict
from typing import Optional, Sequence

from loguru import logger

from agentevolver.module.task_manager.data_mixture import MixtureStrategy
from agentevolver.module.tocf.category import infer_task_category
from agentevolver.schema.task import TaskObjective


class AdaptiveMixtureStrategy(MixtureStrategy):
    """Mixture strategy with category-level weighted resampling for T-Patch.

    New knobs (all default to the legacy behavior so existing ablation rows
    that instantiate this class with only the old kwargs are unaffected):

      - ``stratified``: when True, allocates per-category quotas using a
        largest-remainder split of ``count`` and samples **without**
        replacement inside each category whenever the quota fits in the pool.
        This avoids the "same hard task picked 4× in one epoch" failure mode
        that the legacy ``rng.choices`` path exhibits under skewed weights,
        which otherwise corrupts GRPO's per-prompt group baseline.
      - ``stable_seed``: when True, the RNG is seeded once from ``seed`` at
        construction time. The legacy path mixes ``seed + rebuild_count``
        into the seed on every ``mix_data`` call, which makes mixture draws
        drift across runs as a function of how often the T-Patch controller
        accepts a patch - ruining cross-ablation reproducibility.
    """

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
        stratified: bool = False,
        stable_seed: bool = False,
    ):
        if synthetic_ratio < 0:
            raise ValueError("synthetic_ratio must be non-negative")
        if default_weight < 0:
            raise ValueError("default_weight must be non-negative")

        self._use_original = use_original
        self._synthetic_ratio = synthetic_ratio
        self._category_weights = dict(category_weights or {})
        self._task_weights: dict[str, float] = {}
        self._shuffle = shuffle
        self._seed = seed
        self._target_size = target_size
        self._replacement = replacement
        self._default_weight = default_weight
        self._stratified = stratified
        self._stable_seed = stable_seed
        self._rebuild_count = 0
        self._stable_rng: Optional[random.Random] = (
            random.Random(seed) if stable_seed and seed is not None else None
        )

    @property
    def need_synthetic(self) -> bool:
        return self._synthetic_ratio > 0

    @property
    def category_weights(self) -> dict[str, float]:
        return dict(self._category_weights)

    def set_category_weights(self, weights: dict[str, float]):
        self._category_weights = {str(k): max(0.0, float(v)) for k, v in dict(weights).items()}
        logger.info(f"TOCF category weights updated: {self._category_weights}")

    def set_task_weights(self, weights: dict[str, float]):
        self._task_weights = {str(k): max(0.0, float(v)) for k, v in dict(weights).items()}
        logger.info(f"TOCF task weights updated: #tasks={len(self._task_weights)}")

    def _rng(self):
        if self._stable_rng is not None:
            return self._stable_rng
        if self._seed is None:
            return random
        return random.Random(self._seed + self._rebuild_count)

    def _category(self, item: TaskObjective) -> str:
        return infer_task_category(item.task.task_id, item.task.env_type, item.task.metadata)

    def _task_id(self, item: TaskObjective) -> str:
        return str(item.task.task_id or "")

    def _weight_for_category(self, category: str) -> float:
        return max(0.0, float(self._category_weights.get(category, self._default_weight)))

    def _weight_for_task(self, task_id: str) -> float:
        return max(0.0, float(self._task_weights.get(task_id, 1.0)))

    def _weight(self, item: TaskObjective) -> float:
        return self._weight_for_category(self._category(item)) * self._weight_for_task(self._task_id(item))

    def _weighted_without_replacement(
        self,
        pool: Sequence[TaskObjective],
        count: int,
        rng,
    ) -> list[TaskObjective]:
        remaining = list(pool)
        selected: list[TaskObjective] = []
        for _ in range(min(count, len(remaining))):
            weights = [self._weight(item) for item in remaining]
            if sum(weights) <= 0.0:
                weights = [1.0 for _ in remaining]
            picked = rng.choices(range(len(remaining)), weights=weights, k=1)[0]
            selected.append(remaining.pop(picked))
            if not remaining:
                break
        return selected

    def _stratified_sample(
        self, items: Sequence[TaskObjective], count: int, rng
    ) -> list[TaskObjective]:
        """Largest-remainder quota allocation + within-category sampling.

        Within-category sampling is **without replacement** unless the quota
        exceeds the pool, in which case the full pool is taken and the
        remainder is filled with replacement from the same pool.
        """
        if count <= 0 or not items:
            return []

        groups: dict[str, list[TaskObjective]] = defaultdict(list)
        for it in items:
            groups[self._category(it)].append(it)

        cat_weights = {c: self._weight_for_category(c) for c in groups}
        total_weight = sum(cat_weights.values())
        if total_weight <= 0.0:
            cat_weights = {c: 1.0 for c in groups}
            total_weight = float(len(cat_weights))

        raw_quotas = {c: count * cat_weights[c] / total_weight for c in groups}
        quotas = {c: int(math.floor(q)) for c, q in raw_quotas.items()}
        remainder = count - sum(quotas.values())
        if remainder > 0:
            order = sorted(
                raw_quotas.keys(),
                key=lambda c: raw_quotas[c] - quotas[c],
                reverse=True,
            )
            for c in order[:remainder]:
                quotas[c] += 1

        selected: list[TaskObjective] = []
        for category, quota in quotas.items():
            if quota <= 0:
                continue
            pool = groups[category]
            if not pool:
                continue
            if quota <= len(pool):
                picks = self._weighted_without_replacement(pool, quota, rng)
            else:
                # Quota exceeds unique pool size: take the whole pool first
                # (each distinct task appears at least once) then fill the
                # rest with replacement, which is the minimum amount of
                # duplication consistent with the requested count.
                weights = [self._weight(item) for item in pool]
                if sum(weights) <= 0.0:
                    weights = [1.0 for _ in pool]
                picks = list(pool) + rng.choices(pool, weights=weights, k=quota - len(pool))
            selected.extend(copy.deepcopy(p) for p in picks)
        return selected

    def _weighted_sample(self, items: Sequence[TaskObjective], count: int, rng) -> list[TaskObjective]:
        if count <= 0 or not items:
            return []

        if self._stratified:
            return self._stratified_sample(items, count, rng)

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
            f"#mixed={len(mixed)}, stratified={self._stratified}, "
            f"distribution={dict(Counter(self._category(x) for x in mixed))}"
        )
        return mixed

    def __repr__(self):
        return (
            "AdaptiveMixtureStrategy("
            f"use_original={self._use_original}, synthetic_ratio={self._synthetic_ratio}, "
            f"category_weights={self._category_weights}, shuffle={self._shuffle}, "
            f"task_weights={len(self._task_weights)}, "
            f"seed={self._seed}, target_size={self._target_size}, replacement={self._replacement}, "
            f"stratified={self._stratified}, stable_seed={self._stable_seed})"
        )
