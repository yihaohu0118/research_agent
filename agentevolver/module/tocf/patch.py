from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from agentevolver.schema.task import Task

from agentevolver.module.tocf.category import infer_task_category


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


def _mapping_get(obj: Any, key: str) -> Any:
    """Safely ``get(key)`` on a dict-like that may be an OmegaConf DictConfig.

    ``isinstance(obj, Mapping)`` is not reliable for every OmegaConf version,
    so we duck-type on ``.get`` + ``.keys`` and swallow any access error.
    """
    if obj is None:
        return None
    if isinstance(obj, Mapping):
        return obj.get(key)
    if hasattr(obj, "get") and hasattr(obj, "keys"):
        try:
            return obj.get(key)
        except Exception:
            return None
    return None


def _mapping_nonempty(obj: Any) -> bool:
    """True if ``obj`` looks like a non-empty dict or DictConfig."""
    if obj is None:
        return False
    if isinstance(obj, Mapping):
        return len(obj) > 0
    if hasattr(obj, "keys"):
        try:
            return len(list(obj.keys())) > 0  # type: ignore[arg-type]
        except Exception:
            return False
    return False


def resolve_query_suffix(task: Task, config: Any = None, mode: str | None = None) -> str:
    """Resolve a C-Patch query suffix for ``task``.

    Priority (highest first):

    1. ``task.metadata["tocf"]["query_suffix"]`` — per-task override. Lets
       exploration / synthesis paths attach a bespoke suffix to a single
       task without touching global config. Bypasses the ``enable`` and
       ``apply_to_validation`` gates because the caller explicitly asked
       for this text on this task.

    2. ``tocf.query_suffix.by_category[task.category]`` — per-category
       lookup. Preferred over the legacy single ``suffix`` knob because
       different BFCL failure modes need structurally different cues
       (e.g. miss_func benefits from "no-matching-tool → reply in text"
       whereas long_context needs "most-recent-turn-only" framing). Cues
       are intentionally written as behaviour rules that do NOT mention
       the category label itself, to avoid turning the suffix into a
       supervised classification signal that would cause train/val drift.

    3. ``tocf.query_suffix.suffix`` — legacy global fallback. Kept so
       old configs keep working unchanged. When ``by_category`` is
       configured, the behaviour on a category miss is governed by
       ``tocf.query_suffix.category_fallback`` (``"suffix"`` to fall
       back to the global string, ``"empty"`` to emit no cue).

    The ``apply_to_validation`` flag gates everything below the per-task
    override on ``mode == "validate"`` / ``"val"``. Cues are typically
    training-only so that eval scores reflect the policy, not the cue.
    """
    metadata_tocf = (task.metadata or {}).get("tocf", {}) or {}
    per_task_override = metadata_tocf.get("query_suffix")
    if per_task_override:
        return str(per_task_override).strip()

    cfg_enabled = bool(_cfg_get(config, "tocf.query_suffix.enable", False))
    if not cfg_enabled:
        return ""

    apply_to_validation = bool(
        _cfg_get(config, "tocf.query_suffix.apply_to_validation", False)
    )
    if mode in ("validate", "val") and not apply_to_validation:
        return ""

    by_category = _cfg_get(config, "tocf.query_suffix.by_category", None)
    global_suffix = str(_cfg_get(config, "tocf.query_suffix.suffix", "") or "")
    fallback_policy = str(
        _cfg_get(config, "tocf.query_suffix.category_fallback", "suffix") or "suffix"
    ).lower()

    if _mapping_nonempty(by_category):
        category = (task.metadata or {}).get("category")
        if not category:
            category = infer_task_category(
                task.task_id, task.env_type, task.metadata
            )
        cue = _mapping_get(by_category, category)
        if cue is not None:
            # Empty string is a valid "hold out this category" signal and
            # must not fall through to the global suffix.
            return str(cue or "").strip()
        if fallback_policy == "empty":
            return ""
        return global_suffix.strip()

    return global_suffix.strip()


def apply_query_suffix(task: Task, config: Any = None, mode: str | None = None) -> bool:
    """Append a non-semantic exploration suffix to a task query if configured."""
    suffix = resolve_query_suffix(task, config=config, mode=mode)
    if not suffix or task.query is None:
        return False

    metadata = task.metadata if task.metadata is not None else {}
    metadata.setdefault("tocf", {})
    metadata["tocf"].setdefault("original_query", task.query)
    metadata["tocf"]["applied_query_suffix"] = suffix
    task.metadata = metadata

    if suffix not in task.query:
        task.query = f"{task.query}\n\n{suffix}"
        return True
    return False
