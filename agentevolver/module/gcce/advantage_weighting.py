"""GCCE advantage weighting -- the policy-side projection of the router.

This is the drop-in replacement for
``agentevolver.module.pace.advantage_weighting.apply_curriculum_advantage_weighting``.
Instead of reweighting positive advantages by category failure rate, we use
the router's policy-gap weight ``r_pi(c)`` which is the *causal* share of
each category's regret that the policy (not the environment) is responsible
for.

When GCCE is disabled or the router has no decision yet, this function is a
no-op and training silently falls back to PACE / plain GRPO.
"""
from __future__ import annotations

import json
from collections import defaultdict
from typing import Any, Mapping

import torch

from agentevolver.module.tocf.category import infer_task_category


def _cfg_get(config: Any, key: str, default=None):
    if config is None:
        return default
    if isinstance(config, Mapping):
        return config.get(key, default)
    getter = getattr(config, "get", None)
    if callable(getter):
        try:
            return getter(key, default)
        except Exception:
            pass
    return getattr(config, key, default)


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


def _is_demo_row(batch: Any, idx: int) -> bool:
    """Detect whether this row was injected by D-Patch.

    Three redundant channels are checked because the flag flows through
    different containers depending on whether it was set at trajectory
    construction time (``metadata.is_demo``) or stamped onto the reward
    scores dict by ``env_manager.trajectories_to_samples``:

      1. ``non_tensor_batch["extras"][idx]["is_demo"]``
      2. ``non_tensor_batch["reward_scores"][idx]["metadata"]["is_demo"]``
      3. ``non_tensor_batch["rollout_ids"][idx]`` starting with ``demo::``
    """
    try:
        extras = _as_dict(_array_get(batch.non_tensor_batch.get("extras"), idx, {}))
        if bool(extras.get("is_demo", False)):
            return True
        reward_meta = _as_dict(_reward_item(batch, idx).get("metadata", {}))
        if bool(reward_meta.get("is_demo", False)):
            return True
        rid = _array_get(batch.non_tensor_batch.get("rollout_ids"), idx, "")
        if isinstance(rid, str) and rid.startswith("demo::"):
            return True
    except Exception:
        return False
    return False


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


def gcce_enabled(config: Any) -> bool:
    gcce_cfg = _cfg_get(config, "gcce", {}) or {}
    return bool(_cfg_get(gcce_cfg, "enable", False))


def gcce_advantage_weighting_enabled(config: Any) -> bool:
    if not gcce_enabled(config):
        return False
    gcce_cfg = _cfg_get(config, "gcce", {}) or {}
    adv_cfg = _cfg_get(gcce_cfg, "advantage_weighting", {}) or {}
    return bool(_cfg_get(adv_cfg, "enable", False))


def apply_gcce_advantage_weighting(batch: Any, router: Any, config: Any, env_type: str | None = None):
    """Multiply positive advantages by ``1 + alpha_policy * r_pi(c)``.

    This is the GCCE counterpart to PACE's ``apply_curriculum_advantage_weighting``.
    It is safe to call unconditionally: when GCCE is disabled, when the router
    has no decision yet, or when the router returns no per-category weight for
    a sample, the advantage is left untouched.
    """
    if not gcce_advantage_weighting_enabled(config) or router is None:
        return batch, {}
    if getattr(router, "latest", None) is None:
        return batch, {"gcce/advantage_weighting/enabled": 1.0, "gcce/advantage_weighting/weighted_count": 0.0}

    gcce_cfg = _cfg_get(config, "gcce", {}) or {}
    adv_cfg = _cfg_get(gcce_cfg, "advantage_weighting", {}) or {}
    prefix = str(_cfg_get(adv_cfg, "metric_prefix", "gcce/advantage_weighting"))

    advantages = batch.batch["advantages"]
    weights = torch.ones(advantages.shape[0], device=advantages.device, dtype=advantages.dtype)
    sample_adv_scores = _sample_advantage_score(batch)

    # Demo amplification is orthogonal to PACE-style reweighting: it only
    # acts on rows the D-Patch injector flagged as demonstrations, and it
    # pushes positive advantages on those rows harder so the policy
    # actually moves toward the gold trajectory. When D-Patch is disabled
    # there will simply be no demo rows and this code path is a no-op.
    demo_amplify_enabled = bool(_cfg_get(adv_cfg, "demo_amplify_enable", True))

    by_category: dict[str, dict[str, float]] = defaultdict(
        lambda: {
            "count": 0,
            "weighted": 0,
            "weight_sum": 0.0,
            "demo_count": 0,
            "demo_weighted": 0,
            "demo_weight_sum": 0.0,
        }
    )
    for idx in range(advantages.shape[0]):
        task_id = _task_id_from_batch(batch, idx)
        metadata = _metadata_from_batch(batch, idx)
        category = infer_task_category(task_id, env_type, metadata)

        is_demo = _is_demo_row(batch, idx)
        by_category[category]["count"] += 1
        if is_demo:
            by_category[category]["demo_count"] += 1

        weight = 1.0
        if is_demo and demo_amplify_enabled and hasattr(router, "demo_advantage_scale"):
            # Use the paired CGA signal: the same router that drives
            # demo_rate(c) on the env side drives demo_advantage_scale(c)
            # on the policy side. This is the "co-" in co-evolution.
            weight = float(router.demo_advantage_scale(category))
            by_category[category]["demo_weight_sum"] += weight
            if weight > 1.0 and _should_apply(batch, idx, sample_adv_scores, adv_cfg):
                weights[idx] = weight
                by_category[category]["demo_weighted"] += 1
            by_category[category]["weight_sum"] += weight
        else:
            weight = float(router.advantage_weight(category))
            by_category[category]["weight_sum"] += weight
            if weight > 1.0 and _should_apply(batch, idx, sample_adv_scores, adv_cfg):
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
        metrics[f"{prefix}/{safe_category}/r_policy"] = float(router.r_policy(category))
        metrics[f"{prefix}/{safe_category}/demo_count"] = float(item["demo_count"])
        metrics[f"{prefix}/{safe_category}/demo_weighted_count"] = float(item["demo_weighted"])
        if item["demo_count"] > 0:
            metrics[f"{prefix}/{safe_category}/demo_avg_weight"] = float(
                item["demo_weight_sum"] / max(1, int(item["demo_count"]))
            )

    return batch, metrics
