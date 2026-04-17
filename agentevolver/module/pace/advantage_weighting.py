from __future__ import annotations

import json
from collections import defaultdict
from typing import Any

import torch

from agentevolver.module.tocf.category import infer_task_category


def _cfg_get(config: Any, key: str, default=None):
    if config is None:
        return default
    if isinstance(config, dict):
        return config.get(key, default)
    getter = getattr(config, "get", None)
    if callable(getter):
        try:
            return getter(key, default)
        except Exception:
            pass
    try:
        return getattr(config, key)
    except Exception:
        return default


def _as_dict(value: Any) -> dict:
    if isinstance(value, dict):
        return value
    if isinstance(value, str):
        try:
            parsed = json.loads(value)
            return parsed if isinstance(parsed, dict) else {}
        except (TypeError, json.JSONDecodeError):
            return {}
    return {}


def _array_get(values: Any, idx: int, default=None):
    if values is None:
        return default
    try:
        return values[idx]
    except Exception:
        return default


def _nested_cfg(config: Any, *keys: str, default=None):
    node = config
    for key in keys:
        node = _cfg_get(node, key, None)
        if node is None:
            return default
    return node


def _advantage_cfg(config: Any):
    return _nested_cfg(config, "pace", "advantage_weighting", default={}) or {}


def pace_advantage_weighting_enabled(config: Any) -> bool:
    pace_cfg = _cfg_get(config, "pace", {}) or {}
    adv_cfg = _cfg_get(pace_cfg, "advantage_weighting", {}) or {}
    return bool(_cfg_get(pace_cfg, "enable", False)) and bool(_cfg_get(adv_cfg, "enable", False))


def _reward_item(batch: Any, idx: int) -> dict:
    return _as_dict(_array_get(batch.non_tensor_batch.get("reward_scores"), idx, {}))


def _reward_value(batch: Any, idx: int) -> float:
    item = _reward_item(batch, idx)
    try:
        return float(item.get("outcome", 0.0))
    except (TypeError, ValueError):
        return 0.0


def _metadata_from_batch(batch: Any, idx: int) -> dict:
    reward_metadata = _as_dict(_reward_item(batch, idx).get("metadata", {}))
    if "test_category" in reward_metadata and "category" not in reward_metadata:
        reward_metadata["category"] = reward_metadata["test_category"]

    extras = _as_dict(_array_get(batch.non_tensor_batch.get("extras"), idx, {}))
    extra_metadata = _as_dict(extras.get("metadata", {}))
    metadata = {**extra_metadata, **reward_metadata}
    for key in ("category", "dataset_type", "data_source"):
        if key in extras and key not in metadata:
            metadata[key] = extras[key]
    return metadata


def _task_id_from_batch(batch: Any, idx: int) -> str:
    task_id = _array_get(batch.non_tensor_batch.get("task_ids"), idx, None)
    if task_id:
        return str(task_id)

    extras = _as_dict(_array_get(batch.non_tensor_batch.get("extras"), idx, {}))
    for key in ("task_id", "index", "original_id"):
        if extras.get(key):
            return str(extras[key])
    return ""


def _sample_advantage_score(batch: Any) -> torch.Tensor:
    advantages = batch.batch["advantages"]
    if "response_mask" in batch.batch.keys():
        mask = batch.batch["response_mask"].to(device=advantages.device, dtype=advantages.dtype)
    else:
        mask = torch.ones_like(advantages)
    denom = mask.sum(dim=-1).clamp_min(1.0)
    return (advantages * mask).sum(dim=-1) / denom


def _should_apply(batch: Any, idx: int, sample_adv_scores: torch.Tensor, cfg: Any) -> bool:
    apply_to = str(_cfg_get(cfg, "apply_to", "positive_advantage"))
    if apply_to == "all":
        return True
    if apply_to == "positive_reward":
        threshold = float(_cfg_get(cfg, "reward_threshold", 0.0))
        return _reward_value(batch, idx) > threshold
    if apply_to == "successful_reward":
        threshold = float(_cfg_get(cfg, "success_reward_threshold", 1.0))
        return _reward_value(batch, idx) >= threshold

    threshold = float(_cfg_get(cfg, "advantage_threshold", 0.0))
    return bool(sample_adv_scores[idx].detach().item() > threshold)


def _category_failure_rates(stats: Any, *, scope: str, min_samples: int) -> dict[str, tuple[float, int]]:
    if stats is None:
        return {}
    snapshot = stats.snapshot(window=(scope != "total"))
    categories = snapshot.get("categories", {})
    result = {}
    for category, item in categories.items():
        count = int(item.get("count", 0))
        success_rate = float(item.get("success_rate", 0.0))
        if count < min_samples:
            continue
        result[str(category)] = (max(0.0, 1.0 - success_rate), count)
    return result


def apply_curriculum_advantage_weighting(batch: Any, stats: Any, config: Any, env_type: str | None = None):
    """Boost positive learning signal from categories that are currently hard.

    This implements PACE's curriculum-aware advantage weighting. TOCF changes which
    tasks are sampled; this module changes how strongly successful hard-category
    rollouts contribute to the policy gradient.
    """
    if not pace_advantage_weighting_enabled(config):
        return batch, {}

    cfg = _advantage_cfg(config)
    min_samples = int(_cfg_get(cfg, "min_samples", 8))
    scope = str(_cfg_get(cfg, "stats_scope", "window"))
    failure_rates = _category_failure_rates(stats, scope=scope, min_samples=min_samples)
    if not failure_rates:
        return batch, {"pace/advantage_weighting/enabled": 1.0, "pace/advantage_weighting/weighted_count": 0.0}

    alpha = float(_cfg_get(cfg, "alpha", 0.5))
    min_weight = float(_cfg_get(cfg, "min_weight", 1.0))
    max_weight = float(_cfg_get(cfg, "max_weight", 2.0))
    prefix = str(_cfg_get(cfg, "metric_prefix", "pace/advantage_weighting"))

    advantages = batch.batch["advantages"]
    weights = torch.ones(advantages.shape[0], device=advantages.device, dtype=advantages.dtype)
    sample_adv_scores = _sample_advantage_score(batch)

    by_category = defaultdict(lambda: {"count": 0, "weighted": 0, "weight_sum": 0.0})
    for idx in range(advantages.shape[0]):
        task_id = _task_id_from_batch(batch, idx)
        metadata = _metadata_from_batch(batch, idx)
        category = infer_task_category(task_id, env_type, metadata)
        failure_rate, _count = failure_rates.get(category, (0.0, 0))
        weight = min(max_weight, max(min_weight, 1.0 + alpha * failure_rate))

        by_category[category]["count"] += 1
        by_category[category]["weight_sum"] += weight
        if weight > 1.0 and _should_apply(batch, idx, sample_adv_scores, cfg):
            weights[idx] = weight
            by_category[category]["weighted"] += 1

    batch.batch["advantages"] = advantages * weights.view(-1, 1)

    weighted_mask = weights > 1.0
    metrics = {
        f"{prefix}/enabled": 1.0,
        f"{prefix}/mean_weight": float(weights.mean().detach().item()),
        f"{prefix}/max_weight": float(weights.max().detach().item()),
        f"{prefix}/weighted_count": float(weighted_mask.sum().detach().item()),
        f"{prefix}/weighted_ratio": float(weighted_mask.float().mean().detach().item()),
    }
    for category, item in by_category.items():
        safe_category = str(category).replace("/", "_")
        count = max(1, int(item["count"]))
        metrics[f"{prefix}/{safe_category}/batch_count"] = float(item["count"])
        metrics[f"{prefix}/{safe_category}/weighted_count"] = float(item["weighted"])
        metrics[f"{prefix}/{safe_category}/avg_candidate_weight"] = float(item["weight_sum"] / count)
        if category in failure_rates:
            metrics[f"{prefix}/{safe_category}/failure_rate"] = float(failure_rates[category][0])

    return batch, metrics
