from __future__ import annotations

import copy
import json
import os
import re
from collections import defaultdict
from difflib import SequenceMatcher
from typing import Any, Iterable, Sequence

from agentevolver.schema.task import TaskObjective


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


class EvolvedTaskBank:
    """Persistent bank of accepted co-evolved task objectives."""

    def __init__(
        self,
        *,
        path: str | None = None,
        max_total: int = 256,
        max_per_parent: int = 16,
    ):
        self.path = path
        self.max_total = max(1, int(max_total))
        self.max_per_parent = max(1, int(max_per_parent))
        self._objectives: list[TaskObjective] = []
        self._accepted_total = 0
        self._rejected_total = 0
        self._retired_total = 0
        if path:
            self.load(path)

    def _query_key(self, objective: TaskObjective) -> str:
        return self.normalize_query(objective.task.query)

    @staticmethod
    def normalize_query(query: Any) -> str:
        text = str(query or "").strip().lower()
        return re.sub(r"\s+", " ", text)

    @staticmethod
    def token_set(query: Any) -> set[str]:
        normalized = EvolvedTaskBank.normalize_query(query)
        return {token for token in re.split(r"[^a-z0-9_]+", normalized) if token}

    @staticmethod
    def _jaccard(a: set[str], b: set[str]) -> float:
        if not a or not b:
            return 0.0
        inter = len(a & b)
        union = len(a | b)
        return float(inter) / float(union) if union else 0.0

    def _score(self, objective: TaskObjective) -> float:
        confidence = _safe_float(objective.confidence, 0.0)
        coevo = ((objective.task.metadata or {}).get("tocf") or {}).get("coevo", {}) or {}
        pressure = _safe_float(coevo.get("pressure"), 0.0)
        epoch = _safe_float(coevo.get("evolution_epoch"), 0.0)
        return confidence + 0.15 * pressure + 0.01 * epoch

    def _parent_task_id(self, objective: TaskObjective) -> str:
        coevo = ((objective.task.metadata or {}).get("tocf") or {}).get("coevo", {}) or {}
        parent = coevo.get("parent_task_id") or objective.task.task_id
        return str(parent or "")

    def _trim(self) -> None:
        self._objectives.sort(key=self._score, reverse=True)
        del self._objectives[self.max_total :]

    def retire_stale(
        self,
        *,
        current_epoch: int | str | None,
        max_staleness: int | None = None,
    ) -> int:
        if max_staleness is None or not isinstance(current_epoch, int):
            return 0
        kept: list[TaskObjective] = []
        retired = 0
        for objective in self._objectives:
            coevo = ((objective.task.metadata or {}).get("tocf") or {}).get("coevo", {}) or {}
            created_epoch = coevo.get("evolution_epoch")
            if isinstance(created_epoch, int) and current_epoch - created_epoch > max_staleness:
                retired += 1
                continue
            kept.append(objective)
        self._objectives = kept
        self._retired_total += retired
        return retired

    def objectives(self) -> list[TaskObjective]:
        return [copy.deepcopy(objective) for objective in self._objectives]

    def existing_query_keys(self) -> set[str]:
        return {self._query_key(objective) for objective in self._objectives if self._query_key(objective)}

    def add_candidates(
        self,
        objectives: Sequence[TaskObjective],
        *,
        min_confidence: float = 0.0,
        max_new: int | None = None,
        existing_queries: Iterable[str] | None = None,
        min_query_chars: int = 24,
        min_query_tokens: int = 5,
        max_query_similarity: float = 0.9,
        max_jaccard_similarity: float = 0.8,
    ) -> list[TaskObjective]:
        accepted: list[TaskObjective] = []
        if not objectives:
            return accepted

        seen = self.existing_query_keys()
        seen_signatures: list[tuple[str, set[str]]] = [
            (normalized, self.token_set(normalized))
            for normalized in seen
            if normalized
        ]
        for query in existing_queries or ():
            normalized = self.normalize_query(query)
            if normalized:
                seen.add(normalized)
                seen_signatures.append((normalized, self.token_set(normalized)))

        parent_counts: dict[str, int] = defaultdict(int)
        for objective in self._objectives:
            parent_counts[self._parent_task_id(objective)] += 1

        ranked = sorted(objectives, key=self._score, reverse=True)
        for objective in ranked:
            if max_new is not None and len(accepted) >= int(max_new):
                break
            if objective.task.query is None or objective.ground_truth is None:
                self._rejected_total += 1
                continue
            if _safe_float(objective.confidence, 0.0) < float(min_confidence):
                self._rejected_total += 1
                continue

            key = self._query_key(objective)
            tokens = self.token_set(key)
            if not key or key in seen:
                self._rejected_total += 1
                continue
            if len(key) < int(min_query_chars) or len(tokens) < int(min_query_tokens):
                self._rejected_total += 1
                continue
            too_similar = False
            for prior_text, prior_tokens in seen_signatures:
                if SequenceMatcher(None, key, prior_text).ratio() >= float(max_query_similarity):
                    too_similar = True
                    break
                if self._jaccard(tokens, prior_tokens) >= float(max_jaccard_similarity):
                    too_similar = True
                    break
            if too_similar:
                self._rejected_total += 1
                continue

            parent_task_id = self._parent_task_id(objective)
            if parent_counts[parent_task_id] >= self.max_per_parent:
                self._rejected_total += 1
                continue

            copied = copy.deepcopy(objective)
            accepted.append(copied)
            self._objectives.append(copied)
            parent_counts[parent_task_id] += 1
            seen.add(key)
            seen_signatures.append((key, tokens))
            self._accepted_total += 1

        self._trim()
        return [copy.deepcopy(objective) for objective in accepted]

    def metrics(self, prefix: str = "tocf/coevo") -> dict[str, float]:
        metrics: dict[str, float] = {
            f"{prefix}/bank_size": float(len(self._objectives)),
            f"{prefix}/accepted_total": float(self._accepted_total),
            f"{prefix}/rejected_total": float(self._rejected_total),
            f"{prefix}/retired_total": float(self._retired_total),
        }
        per_tag: dict[str, int] = defaultdict(int)
        per_category: dict[str, int] = defaultdict(int)
        for objective in self._objectives:
            metadata = objective.task.metadata or {}
            category = str(metadata.get("category") or "unknown")
            coevo = (metadata.get("tocf") or {}).get("coevo", {}) or {}
            tag = str(coevo.get("target_tag") or "unknown")
            per_category[category] += 1
            per_tag[tag] += 1
        for category, count in per_category.items():
            metrics[f"{prefix}/bank_category/{category}"] = float(count)
        for tag, count in per_tag.items():
            metrics[f"{prefix}/bank_tag/{tag}"] = float(count)
        return metrics

    def to_dict(self) -> dict[str, Any]:
        return {
            "objectives": [objective.dict() for objective in self._objectives],
            "accepted_total": self._accepted_total,
            "rejected_total": self._rejected_total,
            "retired_total": self._retired_total,
            "max_total": self.max_total,
            "max_per_parent": self.max_per_parent,
        }

    def load(self, path: str | None = None) -> None:
        target = path or self.path
        if not target or not os.path.exists(target):
            return
        try:
            with open(target, "r", encoding="utf-8") as handle:
                raw = json.load(handle)
        except (OSError, json.JSONDecodeError):
            return

        self.max_total = max(1, int(raw.get("max_total", self.max_total) or self.max_total))
        self.max_per_parent = max(1, int(raw.get("max_per_parent", self.max_per_parent) or self.max_per_parent))
        self._accepted_total = int(raw.get("accepted_total", 0) or 0)
        self._rejected_total = int(raw.get("rejected_total", 0) or 0)
        self._retired_total = int(raw.get("retired_total", 0) or 0)
        self._objectives = [
            TaskObjective.parse_obj(item) for item in list(raw.get("objectives") or [])
        ]
        self._trim()

    def save(self, path: str | None = None) -> None:
        target = path or self.path
        if not target:
            return
        os.makedirs(os.path.dirname(target) or ".", exist_ok=True)
        with open(target, "w", encoding="utf-8") as handle:
            json.dump(self.to_dict(), handle, ensure_ascii=False, indent=2)
