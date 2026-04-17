from __future__ import annotations

from typing import Any

from agentevolver.schema.task import Task


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


def resolve_query_suffix(task: Task, config: Any = None, mode: str | None = None) -> str:
    """Resolve finite query-suffix templates from task metadata and config."""
    metadata_tocf = (task.metadata or {}).get("tocf", {})
    suffix = metadata_tocf.get("query_suffix")

    cfg_enabled = bool(_cfg_get(config, "tocf.query_suffix.enable", False))
    if cfg_enabled:
        apply_to_validation = bool(_cfg_get(config, "tocf.query_suffix.apply_to_validation", False))
        if mode in ("validate", "val") and not apply_to_validation:
            return suffix or ""
        suffix = suffix or _cfg_get(config, "tocf.query_suffix.suffix", "")

    return str(suffix or "").strip()


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
