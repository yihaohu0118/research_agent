from __future__ import annotations

import copy
import hashlib
from collections import defaultdict
from dataclasses import dataclass
from typing import Any, Sequence

from agentevolver.module.tocf.category import infer_task_category, patch_task_metadata
from agentevolver.module.tocf.bfcl_synthetic import (
    build_bfcl_synthetic_case_overlay,
    normalize_tool_turns,
    serialize_tool_turns,
    tool_turns_to_strings,
    validate_bfcl_synthetic_alignment,
)
from agentevolver.module.tocf.state import (
    EXCLUDED_FAILURE_TAGS,
    PASS_TAG,
    TOCFCapabilityState,
    UNKNOWN_TAG,
    unpack_category_tag,
)
from agentevolver.schema.task import Task, TaskObjective


def _cfg_get(config: Any, path: str, default=None):
    cur = config
    for part in path.split("."):
        if cur is None:
            return default
        if isinstance(cur, dict):
            cur = cur.get(part, default)
        else:
            cur = getattr(cur, part, default)
    return cur


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


_TARGET_TAG_GUIDANCE: dict[str, str] = {
    "spurious_tool_call": (
        "Prefer tasks where the model must distinguish between truly relevant tool use and turns "
        "that should be answered directly without forcing a tool call."
    ),
    "correct_abstention": (
        "Prefer tasks where the correct solution requires restraint on some turns, even though tools are available."
    ),
    "empty_turn_model_response": (
        "Prefer tasks where each user turn requires a concrete action or direct answer, so passive or empty responses fail clearly."
    ),
    "state_mismatch": (
        "Prefer tasks that require careful state tracking across multiple turns, where stale values or incorrect state transitions would break the task."
    ),
    "instance_mismatch": (
        "Prefer tasks involving multiple similar entities or instances, so the model must identify the correct target before acting."
    ),
    "response_mismatch": (
        "Prefer tasks where the final natural-language response must fully reflect tool results and cover every explicit user ask."
    ),
}


@dataclass
class CoEvoSeedSpec:
    task_id: str
    category: str
    target_tag: str
    pressure: float
    count: int
    source: str


def coevo_enabled(config: Any) -> bool:
    return bool(_cfg_get(config, "tocf.enable", False)) and bool(
        _cfg_get(config, "tocf.coevo.enable", False)
    )


def _valid_failure_tag(tag: str | None) -> bool:
    return bool(tag) and tag not in (PASS_TAG, UNKNOWN_TAG) and tag not in EXCLUDED_FAILURE_TAGS


def _annotate_seed_task(task: Task, spec: CoEvoSeedSpec, *, epoch: int | str | None) -> Task:
    seed = task.copy(deep=True)
    metadata = patch_task_metadata(seed.metadata, task_id=seed.task_id, env_type=seed.env_type)
    tocf_meta = metadata.setdefault("tocf", {})
    coevo_meta = dict(tocf_meta.get("coevo") or {})
    coevo_meta.update(
        {
            "enabled": True,
            "source": spec.source,
            "parent_task_id": spec.task_id,
            "target_tag": spec.target_tag,
            "pressure": float(spec.pressure),
            "observations": int(spec.count),
            "evolution_epoch": epoch,
        }
    )
    tocf_meta["coevo"] = coevo_meta
    metadata["tocf"] = tocf_meta
    seed.metadata = metadata
    seed.query = None
    return seed


def select_coevo_seed_tasks(
    seed_tasks: Sequence[Task],
    capability_state: TOCFCapabilityState | None,
    config: Any,
    *,
    epoch: int | str | None = None,
) -> list[Task]:
    if capability_state is None or not coevo_enabled(config):
        return []

    cfg = _cfg_get(config, "tocf.coevo", {}) or {}
    max_seed_tasks = max(1, int(_cfg_get(cfg, "max_seed_tasks", 8) or 8))
    min_task_observations = max(1, int(_cfg_get(cfg, "min_task_observations", 2) or 2))
    min_samples = max(1, int(_cfg_get(cfg, "min_samples", 4) or 4))
    min_task_weight = float(_cfg_get(cfg, "min_task_weight", 1.05) or 1.05)
    selection_alpha = float(_cfg_get(cfg, "selection_alpha", 0.75) or 0.75)
    selection_max_weight = float(_cfg_get(cfg, "selection_max_weight", 2.5) or 2.5)
    category_fallback = bool(_cfg_get(cfg, "category_fallback", True))

    task_lookup = {str(task.task_id): task for task in seed_tasks if task.task_id}
    selected_ids: set[str] = set()
    selected: list[Task] = []

    task_weights = capability_state.task_weight_targets(
        min_weight=1.0,
        max_weight=selection_max_weight,
        alpha=selection_alpha,
        min_samples=min_samples,
    )

    ranked_task_specs: list[CoEvoSeedSpec] = []
    for task_id, task_weight in sorted(task_weights.items(), key=lambda item: item[1], reverse=True):
        task_state = capability_state.tasks.get(str(task_id)) or {}
        tag = str(task_state.get("last_tag") or UNKNOWN_TAG)
        if task_weight < min_task_weight or not _valid_failure_tag(tag):
            continue
        count = int(task_state.get("count", 0) or 0)
        if count < min_task_observations or task_id not in task_lookup:
            continue
        category = str(task_state.get("category") or infer_task_category(task_id, task_lookup[task_id].env_type, task_lookup[task_id].metadata))
        ranked_task_specs.append(
            CoEvoSeedSpec(
                task_id=str(task_id),
                category=category,
                target_tag=tag,
                pressure=float(task_weight),
                count=count,
                source="task",
            )
        )

    for spec in ranked_task_specs:
        if len(selected) >= max_seed_tasks:
            break
        if spec.task_id in selected_ids:
            continue
        selected.append(_annotate_seed_task(task_lookup[spec.task_id], spec, epoch=epoch))
        selected_ids.add(spec.task_id)

    if len(selected) >= max_seed_tasks or not category_fallback:
        return selected

    category_pool: dict[str, list[Task]] = defaultdict(list)
    for task in seed_tasks:
        if str(task.task_id) in selected_ids:
            continue
        category = infer_task_category(task.task_id, task.env_type, task.metadata)
        category_pool[category].append(task)

    ranked_categories: list[CoEvoSeedSpec] = []
    source = capability_state.window_category_tags or capability_state.total_category_tags
    for key, stats in source.items():
        category, tag = unpack_category_tag(str(key))
        if not _valid_failure_tag(tag) or int(stats.count) < min_samples:
            continue
        pressure = (1.0 - float(stats.reward_mean)) * min(
            1.0, float(stats.count) / float(max(1, min_samples * 4))
        )
        if pressure <= 0.0 or not category_pool.get(category):
            continue
        ranked_categories.append(
            CoEvoSeedSpec(
                task_id=str(category_pool[category][0].task_id),
                category=category,
                target_tag=tag,
                pressure=pressure,
                count=int(stats.count),
                source="category",
            )
        )

    ranked_categories.sort(key=lambda item: item.pressure, reverse=True)
    for spec in ranked_categories:
        if len(selected) >= max_seed_tasks:
            break
        pool = category_pool.get(spec.category) or []
        next_task = next((task for task in pool if str(task.task_id) not in selected_ids), None)
        if next_task is None:
            continue
        selected.append(_annotate_seed_task(next_task, spec, epoch=epoch))
        selected_ids.add(str(next_task.task_id))

    return selected


def coevo_guidance_for_task(task: Task | None) -> str:
    if task is None:
        return ""
    metadata = task.metadata or {}
    coevo = (metadata.get("tocf") or {}).get("coevo", {}) or {}
    if not coevo.get("enabled"):
        return ""

    category = infer_task_category(task.task_id, task.env_type, metadata)
    target_tag = str(coevo.get("target_tag") or UNKNOWN_TAG)
    guidance = _TARGET_TAG_GUIDANCE.get(target_tag)
    if not guidance:
        return ""

    source = str(coevo.get("source") or "task")
    pressure = _safe_float(coevo.get("pressure"), 0.0)
    guidance_text = (
        "# Co-Evolution Target\n"
        "You are generating targeted task variants for training.\n"
        f"- Keep the task in the same capability family/category: `{category}`.\n"
        f"- Target the observed failure mode: `{target_tag}`.\n"
        f"- Failure pressure from recent training: `{pressure:.3f}` (source: `{source}`).\n"
        f"- {guidance}\n"
        "- Make the task realistic, concrete, and objectively verifiable.\n"
        "- Prefer a harder variant over a simple paraphrase.\n"
        "- Preserve the same environment/tool family instead of inventing unrelated capabilities.\n"
    )
    if task.env_type == "bfcl":
        guidance_text += (
            "\n# BFCL Synthetic Case Contract\n"
            "- Generate a task that can be solved using the SAME available BFCL tools/state as the seed case.\n"
            "- Do not invent new files, APIs, tools, classes, accounts, or database rows that were not visible in the interaction history.\n"
            "- The `action_sequence` must be executable BFCL tool calls, grouped with `# step0`, `# step1`, ... when there are multiple user turns.\n"
            "- If the task needs multiple user turns, include `question_schedule` in the JSON: an array of user utterances, one per step. "
            "The first item must exactly match `query`, and the number of items must equal the number of `# step` blocks.\n"
            "- Make the query semantically anchor the GT: mention the operation intent or concrete entities/filenames/values used by the tool calls.\n"
        )
    return guidance_text


def finalize_coevo_objectives(
    objectives: Sequence[TaskObjective],
    *,
    synthetic_grader: str,
    bfcl_synthetic_grader: str | None = None,
    require_executable_gt: bool = True,
    require_semantic_alignment: bool = True,
    min_semantic_score: float = 0.34,
    epoch: int | str | None,
) -> list[TaskObjective]:
    finalized: list[TaskObjective] = []
    for objective in objectives:
        if objective.task.query is None or objective.ground_truth is None:
            continue
        cleaned = copy.deepcopy(objective)
        cleaned.task.open_query = True
        evaluator = synthetic_grader
        normalized_gt = None
        if cleaned.task.env_type == "bfcl" and bfcl_synthetic_grader:
            normalized_gt = normalize_tool_turns(cleaned.ground_truth)
            if require_executable_gt and not normalized_gt:
                continue
            if normalized_gt:
                overlay, overlay_reason = build_bfcl_synthetic_case_overlay(
                    task_id=str(cleaned.task.task_id),
                    query=cleaned.task.query,
                    ground_truth=cleaned.ground_truth,
                    metadata=cleaned.task.metadata,
                    turns=normalized_gt,
                )
                if require_executable_gt and overlay is None:
                    continue
                semantic_alignment = validate_bfcl_synthetic_alignment(
                    cleaned.task.query,
                    normalized_gt,
                    cleaned.task.metadata,
                    min_score=min_semantic_score,
                )
                if require_semantic_alignment and not bool(semantic_alignment["ok"]):
                    continue
                evaluator = bfcl_synthetic_grader
                if overlay:
                    cleaned.task.open_query = bool(len(overlay.get("question") or []) <= 1)
                cleaned.ground_truth = serialize_tool_turns(normalized_gt)
            else:
                overlay = None
                overlay_reason = "synthetic_gt_unparseable"
                semantic_alignment = {
                    "ok": False,
                    "score": 0.0,
                    "reason": "synthetic_gt_unparseable",
                }
        else:
            overlay = None
            overlay_reason = "not_bfcl_synthetic"
            semantic_alignment = {"ok": True, "score": 1.0, "reason": "not_bfcl_synthetic"}
        cleaned.task.evaluator = evaluator
        metadata = patch_task_metadata(
            cleaned.task.metadata,
            task_id=cleaned.task.task_id,
            env_type=cleaned.task.env_type,
        )
        tocf_meta = metadata.setdefault("tocf", {})
        coevo_meta = dict(tocf_meta.get("coevo") or {})
        coevo_meta["enabled"] = True
        coevo_meta["source"] = "coevo"
        coevo_meta["parent_task_id"] = str(coevo_meta.get("parent_task_id") or cleaned.task.task_id)
        coevo_meta["target_tag"] = str(coevo_meta.get("target_tag") or UNKNOWN_TAG)
        coevo_meta["evolution_epoch"] = epoch
        coevo_meta["verifier"] = evaluator
        coevo_meta["gt_parseable"] = bool(normalized_gt)
        coevo_meta["gt_executable"] = False
        coevo_meta["overlay_ready"] = bool(overlay)
        coevo_meta["overlay_reason"] = overlay_reason
        coevo_meta["semantic_alignment"] = semantic_alignment
        if normalized_gt and evaluator == bfcl_synthetic_grader:
            coevo_meta["gt_execution_check"] = "reward_time_env_replay"
        coevo_meta["objective_hash"] = (
            str(overlay.get("objective_hash"))
            if overlay and overlay.get("objective_hash")
            else hashlib.sha1(str(cleaned.task.query or "").encode("utf-8")).hexdigest()[:12]
        )
        tocf_meta["coevo"] = coevo_meta
        if normalized_gt:
            tocf_meta["bfcl_synthetic_gt"] = tool_turns_to_strings(normalized_gt)
        metadata["tocf"] = tocf_meta
        cleaned.task.metadata = metadata
        finalized.append(cleaned)
    return finalized
