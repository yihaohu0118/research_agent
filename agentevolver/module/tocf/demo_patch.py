"""D-Patch: Demonstration Patch for the CoEvo-D (Demonstration-gated
Co-Evolution) algorithm.

This is the environment-side counterpart of GCCE's policy-side advantage
reweighting: whenever the CGA router signals that a given category is
dominated by an *environment* gap (i.e. the current policy is starved of
informative trajectories for that category), the environment emits an
expert demonstration rendered from the task's ground-truth action
sequence.

Architecturally D-Patch is a sibling of TOCF's T/F/C patches:

  T-Patch  : reweight which tasks get sampled              (category prior)
  F-Patch  : reshape the reward signal per rollout         (dense reward)
  C-Patch  : augment the user query with a process prompt  (query suffix)
  D-Patch  : inject gold-action rollouts per category      (this module)

Key design points
-----------------
* The demo pool is built **offline** (see
  ``scripts/build_bfcl_demo_pool.py``). Runtime tokenisation happens once
  per demo the first time it is drawn, and is cached thereafter.
* A demo, once materialised, is a minimal cmt-like object that quacks the
  way ``agentevolver.module.env_manager.EnvManager.trajectories_to_samples``
  expects: it exposes ``metadata``, ``reward``, ``query``, ``data_id`` /
  ``task_id`` / ``rollout_id`` / ``is_terminated`` attributes and a
  ``group_tokenize()`` method that returns a valid
  ``agentevolver.schema.trajectory.Sample``.
* The injection rate per category is driven by the GCCE router's
  ``demo_rate(category)`` method (router extension for CoEvo-D). When the
  router is disabled or absent the rate falls back to a uniform
  ``uniform_rate`` from config so that pure-ablation runs (no CGA) are
  well-defined.
* Demos are flagged via ``metadata["is_demo"] = True``. Downstream
  advantage weighting looks at this flag to apply the
  ``demo_advantage_scale`` amplification, which is the policy-side signal
  that shares the same CGA source as the injection rate - that is the
  "co-evolution" coupling.

This module is deliberately tokenizer-and-chat-template agnostic: the
tokenizer is supplied at injection time by the trainer. It only relies on
the HF ``apply_chat_template`` contract (with ``return_assistant_tokens_mask``
support where available; a best-effort fallback is provided).
"""
from __future__ import annotations

import copy
import json
import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from loguru import logger

from agentevolver.schema.trajectory import Reward, Sample


__all__ = [
    "DemoPool",
    "DemoTrajectory",
    "DemoInjector",
    "DemoConfig",
    "demo_patch_enabled",
]


# ---------------------------------------------------------------------------
# Config helpers
# ---------------------------------------------------------------------------
def _cfg_get(config: Any, key: str, default: Any = None) -> Any:
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


def demo_patch_enabled(config: Any) -> bool:
    tocf_cfg = _cfg_get(config, "tocf", {}) or {}
    demo_cfg = _cfg_get(tocf_cfg, "demo_patch", {}) or {}
    return bool(_cfg_get(demo_cfg, "enable", False))


@dataclass
class DemoConfig:
    """Runtime knobs for D-Patch. Mirrors ``tocf.demo_patch`` YAML block."""

    enable: bool = False
    pool_path: str = ""
    # Fraction of each task's rollout group that gets replaced with demos.
    # If router is enabled, this is the cap; actual rate = router.demo_rate(c).
    # When router is disabled, this rate is used uniformly.
    uniform_rate: float = 0.125  # 1 / n=8 by default
    max_rate: float = 0.5
    min_rate: float = 0.0
    # Multiplicative scale on advantages of demo rows (policy-side
    # amplification). Effective scale = 1 + alpha_demo * r_pi(c).
    alpha_demo: float = 1.5
    # If > 0 and the batch's category demos count is below this fraction,
    # force-inject to reach it (guarantees a non-zero signal in cold start).
    min_per_category_fraction: float = 0.0
    # Decay demo rate linearly over training. 0.0 disables decay.
    rate_decay_per_epoch: float = 0.0
    # Seed for reproducible injection (None = non-deterministic).
    seed: int | None = None

    @classmethod
    def from_config(cls, config: Any) -> "DemoConfig":
        tocf_cfg = _cfg_get(config, "tocf", {}) or {}
        demo_cfg = _cfg_get(tocf_cfg, "demo_patch", {}) or {}
        return cls(
            enable=bool(_cfg_get(demo_cfg, "enable", False)),
            pool_path=str(_cfg_get(demo_cfg, "pool_path", "") or ""),
            uniform_rate=float(_cfg_get(demo_cfg, "uniform_rate", 0.125)),
            max_rate=float(_cfg_get(demo_cfg, "max_rate", 0.5)),
            min_rate=float(_cfg_get(demo_cfg, "min_rate", 0.0)),
            alpha_demo=float(_cfg_get(demo_cfg, "alpha_demo", 1.5)),
            min_per_category_fraction=float(
                _cfg_get(demo_cfg, "min_per_category_fraction", 0.0)
            ),
            rate_decay_per_epoch=float(
                _cfg_get(demo_cfg, "rate_decay_per_epoch", 0.0)
            ),
            seed=_cfg_get(demo_cfg, "seed", None),
        )


# ---------------------------------------------------------------------------
# Demo pool: offline-built per-task JSON spec
# ---------------------------------------------------------------------------
class DemoPool:
    """Lazy loader + category-indexed accessor for the offline demo pool.

    The underlying JSON layout is documented in
    ``scripts/build_bfcl_demo_pool.py``.
    """

    def __init__(self, path: str):
        self._path = str(path)
        self._by_task: dict[str, dict[str, Any]] = {}
        self._by_category: dict[str, list[str]] = {}
        self._loaded = False

    @property
    def is_loaded(self) -> bool:
        return self._loaded

    @property
    def path(self) -> str:
        return self._path

    def __len__(self) -> int:
        return len(self._by_task)

    def load(self) -> "DemoPool":
        if self._loaded:
            return self
        if not self._path:
            logger.warning("[D-Patch] demo pool path is empty; pool will be inert")
            self._loaded = True
            return self

        p = Path(self._path)
        if not p.exists():
            logger.warning(f"[D-Patch] demo pool file not found: {p}; pool inert")
            self._loaded = True
            return self

        with p.open("r", encoding="utf-8") as f:
            raw = json.load(f)
        if not isinstance(raw, dict):
            raise ValueError(
                f"Unexpected demo pool format at {p}: expected dict, got {type(raw).__name__}"
            )
        for task_id, spec in raw.items():
            if not isinstance(spec, dict):
                continue
            cat = str(spec.get("category") or "unknown")
            self._by_task[str(task_id)] = spec
            self._by_category.setdefault(cat, []).append(str(task_id))

        total = len(self._by_task)
        cat_sizes = {k: len(v) for k, v in self._by_category.items()}
        logger.info(
            f"[D-Patch] loaded {total} demos from {p} "
            f"across {len(self._by_category)} categories: {cat_sizes}"
        )
        self._loaded = True
        return self

    def get(self, task_id: str) -> dict[str, Any] | None:
        return self._by_task.get(str(task_id))

    def sample_by_category(
        self, category: str, rng: random.Random
    ) -> dict[str, Any] | None:
        ids = self._by_category.get(category)
        if not ids:
            return None
        return self._by_task[rng.choice(ids)]

    def categories(self) -> list[str]:
        return list(self._by_category.keys())

    def size_of(self, category: str) -> int:
        return len(self._by_category.get(category, []))


# ---------------------------------------------------------------------------
# DemoTrajectory: quacks like a cmt for env_manager.to_dataproto()
# ---------------------------------------------------------------------------
@dataclass
class _Tokenised:
    prompt_ids: list[int]
    response_ids: list[int]
    prompt_attention_mask: list[int]
    response_attention_mask: list[int]
    prompt_position_ids: list[int]
    response_position_ids: list[int]
    prompt_loss_mask: list[int]
    response_loss_mask: list[int]


class DemoTrajectory:
    """Minimal cmt-compatible wrapper around an offline demo spec.

    Exposes the attributes and methods that
    ``env_manager.trajectories_to_samples`` consults on a cmt object:
      - ``.metadata`` (dict) - for get_extra() and downstream advantage
      - ``.reward`` (Reward)
      - ``.data_id`` / ``.task_id`` / ``.rollout_id``
      - ``.is_terminated`` / ``.steps`` / ``.query``
      - ``.group_tokenize()`` returning ``List[Sample]``

    A DemoTrajectory is materialised once and then cached on the
    ``DemoInjector`` by ``task_id``. Tokenisation is a non-trivial cost
    (runs the chat template + tokenizer once per demo) but is amortised.
    """

    def __init__(
        self,
        spec: dict[str, Any],
        tokenizer: Any,
        *,
        max_prompt_length: int,
        max_response_length: int,
        rollout_id: str,
        data_id: str | None = None,
    ):
        self._spec = spec
        self._tokenizer = tokenizer
        self.max_prompt_length = int(max_prompt_length)
        self.max_response_length = int(max_response_length)

        self.task_id = str(spec.get("task_id") or data_id or "demo")
        self.data_id = str(data_id or self.task_id)
        self.rollout_id = str(rollout_id)
        self.query = self._infer_query(spec.get("messages") or [])
        self.is_terminated = True
        self.steps = []  # env_manager.get_extra doesn't require steps

        reward_value = float(spec.get("reward", 1.0))
        self.reward = Reward(
            outcome=reward_value,
            success_rate=1.0 if reward_value > 0 else 0.0,
            metadata={
                "is_demo": True,
                "category": str(spec.get("category") or "unknown"),
                "source": "d_patch_demo_pool",
            },
        )
        self.metadata: dict[str, Any] = {
            "is_demo": True,
            "category": str(spec.get("category") or "unknown"),
            "task_id": self.task_id,
            "add_exp": False,
            "task_train_exp_mode": None,
            "experience_list": [],
        }

        self._cached: _Tokenised | None = None

    # ---------------- convenience ----------------
    @staticmethod
    def _infer_query(messages: Sequence[Mapping[str, Any]]) -> str:
        for msg in messages:
            if msg.get("role") == "user":
                return str(msg.get("content", ""))
        return ""

    # ---------------- tokenisation ----------------
    def _messages(self) -> list[dict[str, Any]]:
        return copy.deepcopy(self._spec.get("messages") or [])

    def _prompt_messages(self) -> list[dict[str, Any]]:
        """Prompt = through the first user turn (inclusive)."""
        out: list[dict[str, Any]] = []
        saw_user = False
        for msg in self._messages():
            out.append(msg)
            if msg.get("role") == "user":
                saw_user = True
                break
        if not saw_user:
            return []
        return out

    def _apply_chat_template(
        self, messages: list[dict[str, Any]], *, add_generation_prompt: bool
    ) -> list[int]:
        tok = self._tokenizer
        try:
            return tok.apply_chat_template(
                messages,
                tokenize=True,
                add_generation_prompt=add_generation_prompt,
            )
        except Exception as exc:
            logger.warning(
                f"[D-Patch] apply_chat_template failed ({exc}); "
                "falling back to plain concatenation."
            )
            text = "\n".join(f"{m.get('role','')}: {m.get('content','')}" for m in messages)
            return tok.encode(text, add_special_tokens=False)

    def _compute_assistant_mask(
        self, response_ids: list[int], messages: list[dict[str, Any]]
    ) -> list[int]:
        """Best-effort: try HF return_assistant_tokens_mask; if unsupported,
        mark everything in the response as supervised (conservative choice
        that still trains on gold assistant tokens; tool-result tokens
        being trained adds a tiny noise but empirically does not hurt at
        small scale and avoids a fragile hand-rolled boundary search).
        """
        tok = self._tokenizer
        try:
            encoded = tok.apply_chat_template(
                messages,
                tokenize=True,
                add_generation_prompt=False,
                return_dict=True,
                return_assistant_tokens_mask=True,
            )
            mask = encoded.get("assistant_masks")
            if mask is not None:
                full_mask = list(mask)
                tail = full_mask[-len(response_ids):]
                if len(tail) == len(response_ids):
                    return [int(bool(x)) for x in tail]
        except Exception:
            pass
        return [1] * len(response_ids)

    def _tokenise(self) -> _Tokenised:
        if self._cached is not None:
            return self._cached

        prompt_msgs = self._prompt_messages()
        full_msgs = self._messages()

        prompt_ids = self._apply_chat_template(
            prompt_msgs, add_generation_prompt=True
        )
        full_ids = self._apply_chat_template(
            full_msgs, add_generation_prompt=False
        )

        if len(full_ids) <= len(prompt_ids) or full_ids[: len(prompt_ids)] != prompt_ids:
            prefix_len = 0
            lim = min(len(prompt_ids), len(full_ids))
            for k in range(lim):
                if prompt_ids[k] != full_ids[k]:
                    break
                prefix_len = k + 1
            prompt_ids = prompt_ids[:prefix_len]

        response_ids = full_ids[len(prompt_ids):]
        if not response_ids:
            response_ids = [self._tokenizer.eos_token_id or prompt_ids[-1]]

        if len(prompt_ids) > self.max_prompt_length:
            prompt_ids = prompt_ids[-self.max_prompt_length:]
        if len(response_ids) > self.max_response_length:
            response_ids = response_ids[: self.max_response_length]

        response_loss_mask = self._compute_assistant_mask(response_ids, full_msgs)
        if len(response_loss_mask) != len(response_ids):
            response_loss_mask = [1] * len(response_ids)

        tokenised = _Tokenised(
            prompt_ids=list(prompt_ids),
            response_ids=list(response_ids),
            prompt_attention_mask=[1] * len(prompt_ids),
            response_attention_mask=[1] * len(response_ids),
            prompt_position_ids=list(range(len(prompt_ids))),
            response_position_ids=list(
                range(len(prompt_ids), len(prompt_ids) + len(response_ids))
            ),
            prompt_loss_mask=[0] * len(prompt_ids),
            response_loss_mask=response_loss_mask,
        )
        self._cached = tokenised
        return tokenised

    # ---------------- cmt-compatible surface ----------------
    def group_tokenize(self) -> list[Sample]:
        t = self._tokenise()
        input_ids = t.prompt_ids + t.response_ids
        attention_mask = t.prompt_attention_mask + t.response_attention_mask
        position_ids = t.prompt_position_ids + t.response_position_ids
        loss_mask = t.prompt_loss_mask + t.response_loss_mask

        sample = Sample(
            data_id=self.data_id,
            rollout_id=self.rollout_id,
            task_id=self.task_id,
            minor_index_id=0,
            messages=self._messages(),
            input_ids=input_ids,
            prompt_ids=t.prompt_ids,
            response_ids=t.response_ids,
            attention_mask=attention_mask,
            prompt_attention_mask=t.prompt_attention_mask,
            response_attention_mask=t.response_attention_mask,
            loss_mask=loss_mask,
            prompt_loss_mask=t.prompt_loss_mask,
            response_loss_mask=t.response_loss_mask,
            position_ids=position_ids,
            prompt_position_ids=t.prompt_position_ids,
            response_position_ids=t.response_position_ids,
            reward_scores=self.reward.model_dump(),
            max_prompt_len=self.max_prompt_length,
            max_response_len=self.max_response_length,
            max_model_len=self.max_prompt_length + self.max_response_length,
        )
        sample.truncate_output_ids()
        return [sample]


# ---------------------------------------------------------------------------
# Injector: turns a batch's (tasks, trajectories) into
# (tasks, trajectories + demos) according to CGA-gated per-category rate.
# ---------------------------------------------------------------------------
@dataclass
class InjectionReport:
    total_tasks: int = 0
    total_rollouts: int = 0
    injected: int = 0
    by_category: dict[str, int] = field(default_factory=dict)
    rate_by_category: dict[str, float] = field(default_factory=dict)


class DemoInjector:
    """Injects D-Patch demos into the post-rollout trajectory list.

    The contract is:
      - Called *after* ``env_manager.rollout(...)`` returns ``trajectories``
        but *before* ``env_manager.to_dataproto(trajectories)``.
      - Receives the same ``tasks`` list and the rollout-multiplicity
        ``n`` used at rollout time.
      - Returns (new_tasks, new_trajectories) with demo DemoTrajectory
        objects appended for tasks in categories where the router signals
        injection.

    The injector does **not** replace failed rollouts in-place because
    advantage computation groups rollouts by task_id; we add *extra*
    rollout slots for that task, which keeps the group-based GRPO
    normalisation well-defined (the group size grows by the number of
    demos injected, and the demo contributes a guaranteed positive
    advantage).
    """

    def __init__(
        self,
        config: DemoConfig,
        pool: DemoPool,
        tokenizer: Any,
        max_prompt_length: int,
        max_response_length: int,
    ):
        self._cfg = config
        self._pool = pool
        self._tokenizer = tokenizer
        self._max_prompt = int(max_prompt_length)
        self._max_response = int(max_response_length)
        self._rng = (
            random.Random(config.seed) if config.seed is not None else random.Random()
        )

    @property
    def enabled(self) -> bool:
        return bool(self._cfg.enable and self._pool.is_loaded and len(self._pool) > 0)

    # ------------------------------------------------------------------
    def _rate_for_category(
        self,
        category: str,
        router: Any | None,
        epoch: int | None,
    ) -> float:
        # Router path: use demo_rate(c) if available. Fallback to uniform.
        if router is not None and hasattr(router, "demo_rate"):
            try:
                r = float(router.demo_rate(category))
            except Exception:
                r = self._cfg.uniform_rate
        else:
            r = self._cfg.uniform_rate

        if epoch is not None and self._cfg.rate_decay_per_epoch > 0.0:
            r -= self._cfg.rate_decay_per_epoch * float(epoch)

        return max(self._cfg.min_rate, min(self._cfg.max_rate, r))

    # ------------------------------------------------------------------
    def _build_demo_for_task(
        self, task_id: str, category: str, rollout_id: str
    ) -> DemoTrajectory | None:
        spec = self._pool.get(task_id) or self._pool.sample_by_category(
            category, self._rng
        )
        if spec is None:
            return None
        return DemoTrajectory(
            spec=spec,
            tokenizer=self._tokenizer,
            max_prompt_length=self._max_prompt,
            max_response_length=self._max_response,
            rollout_id=rollout_id,
            data_id=task_id,
        )

    # ------------------------------------------------------------------
    def inject(
        self,
        tasks: list[Any],
        trajectories: list[Any],
        *,
        router: Any | None = None,
        rollout_n: int = 1,
        epoch: int | None = None,
        category_getter: Any = None,
    ) -> tuple[list[Any], list[Any], InjectionReport]:
        """Return (new_tasks, new_trajectories, report).

        Parameters
        ----------
        tasks              The input Task list handed to env_manager.rollout.
        trajectories       The cmt-like rollout objects returned by it.
        router             Any object exposing ``demo_rate(category) -> float``.
        rollout_n          The GRPO group size used at rollout time.
        category_getter    Optional callable(task, traj) -> category. If
                           None we look at ``task.metadata.get("category")``
                           or fall back to "unknown".
        """
        report = InjectionReport()
        if not self.enabled:
            return tasks, trajectories, report

        # Group trajectories by their source task so we know which group to
        # extend with a demo.
        by_task: dict[str, list[int]] = {}
        for idx, traj in enumerate(trajectories):
            tid = getattr(traj, "data_id", None) or getattr(traj, "task_id", None) or ""
            by_task.setdefault(str(tid), []).append(idx)

        report.total_tasks = len(tasks)
        report.total_rollouts = len(trajectories)

        new_trajectories = list(trajectories)

        seen_categories: set[str] = set()
        for task in tasks:
            tid = str(getattr(task, "task_id", "") or "")
            category = "unknown"
            if category_getter is not None:
                try:
                    category = str(category_getter(task) or "unknown")
                except Exception:
                    category = "unknown"
            else:
                md = getattr(task, "metadata", None) or {}
                if isinstance(md, Mapping):
                    category = str(md.get("category") or md.get("dataset_type") or "unknown")

            rate = self._rate_for_category(category, router, epoch)
            if category not in seen_categories:
                report.rate_by_category[category] = rate
                seen_categories.add(category)

            # Expected group demos = rate * rollout_n, rounded stochastically
            # so that small rates still occasionally trigger a demo.
            mu = rate * float(rollout_n)
            n_demos = int(mu) + (1 if self._rng.random() < (mu - int(mu)) else 0)
            if n_demos <= 0:
                continue

            for k in range(n_demos):
                rid = f"demo::{tid}::{k}"
                demo = self._build_demo_for_task(tid, category, rid)
                if demo is None:
                    continue
                new_trajectories.append(demo)
                report.injected += 1
                report.by_category[category] = report.by_category.get(category, 0) + 1

        if report.injected > 0:
            logger.info(
                f"[D-Patch] injected {report.injected} demos "
                f"(base rollouts={report.total_rollouts}, rate_by_cat={report.rate_by_category}, "
                f"injected_by_cat={report.by_category})"
            )
        return tasks, new_trajectories, report

    # ------------------------------------------------------------------
    def metrics(self, report: InjectionReport, prefix: str = "d_patch") -> dict[str, float]:
        out: dict[str, float] = {
            f"{prefix}/enabled": 1.0 if self.enabled else 0.0,
            f"{prefix}/total_rollouts": float(report.total_rollouts),
            f"{prefix}/injected": float(report.injected),
            f"{prefix}/inject_ratio": (
                float(report.injected) / float(report.total_rollouts)
                if report.total_rollouts > 0
                else 0.0
            ),
        }
        for cat, cnt in report.by_category.items():
            safe = str(cat).replace("/", "_")
            out[f"{prefix}/{safe}/injected"] = float(cnt)
        for cat, rate in report.rate_by_category.items():
            safe = str(cat).replace("/", "_")
            out[f"{prefix}/{safe}/rate"] = float(rate)
        return out

    # ------------------------------------------------------------------
    def alpha_demo(self) -> float:
        return float(self._cfg.alpha_demo)
