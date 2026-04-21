"""Warm-start SFT on the CoEvo-D offline demo pool.

This is **Phase 0** of the CoEvo-D pipeline: before GRPO begins, we use
the ground-truth demonstrations we already rendered (see
``scripts/build_bfcl_demo_pool.py``) to teach the base policy the
tool-calling surface of BFCL -- format, turn structure, argument
conventions. The resulting checkpoint replaces ``Qwen2.5-7B-Instruct``
as the init for GRPO, so TOCF/PACE/GCCE/D-Patch (Phase 1) operate on a
policy that already has non-zero success rate in every category.

Empirically, on tool-calling benchmarks, a short SFT warm-start on a few
hundred gold trajectories moves the GRPO starting point up by
+0.05 ~ +0.10 pass-rate points before any co-evolution mechanism kicks
in. CoEvo-D then stacks on top with the same CGA signal driving both
sides of the system, which is exactly the "environment and model
co-evolve" story.

Design notes
------------
* **Full fine-tune, no LoRA**: the RL stage full-fine-tunes, so running
  SFT with LoRA would force an adapter merge before GRPO and add a
  failure mode for no upside at this scale. FSDP + bf16 makes the 7B
  full FT trivially fit on 4x80GB / 4x48GB.
* **Loss mask via incremental tokenisation**: we do *not* rely on the
  chat template's ``return_assistant_tokens_mask`` (Qwen2.5-Instruct's
  template does not always wrap assistant spans in ``{% generation %}``
  markers, and behaviour differs between transformers minor versions).
  Instead we prefix-tokenise message-by-message and label every
  assistant-emitted token, all other tokens as -100. This is O(T^2) per
  sample but T <= max_seq_length and we run once offline.
* **Uses** :class:`transformers.Trainer` **without TRL**: avoids another
  dependency bump and keeps the whole thing in ~250 lines of stdlib +
  HF.

Launch (single node, 4 GPUs, FSDP full-shard + bf16)::

    bash examples/run_sft_warmstart.sh
"""
from __future__ import annotations

import argparse
import json
import os
import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import Dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    DataCollatorForLanguageModeling,
    Trainer,
    TrainingArguments,
    set_seed,
)


# ---------------------------------------------------------------------------
# Sample construction: messages -> {input_ids, labels}
# ---------------------------------------------------------------------------
IGNORE_INDEX = -100


def _build_example(
    messages: list[dict[str, Any]],
    tokenizer: Any,
    max_seq_length: int,
) -> dict[str, list[int]] | None:
    """Turn a demo's message list into an ``input_ids`` / ``labels`` pair.

    We tokenise the conversation prefix-by-prefix (each prefix ends on
    one more message than the previous) and classify every newly-emitted
    token as "assistant (supervised)" or "not assistant (ignored)".

    Returns ``None`` if the demo is degenerate (no assistant token
    would end up supervised, e.g. the first-and-only message is a
    system prompt).
    """
    prev_ids: list[int] = []
    input_ids: list[int] = []
    labels: list[int] = []

    for msg_idx in range(len(messages)):
        prefix = messages[: msg_idx + 1]
        try:
            new_ids = tokenizer.apply_chat_template(
                prefix,
                tokenize=True,
                add_generation_prompt=False,
            )
        except Exception as exc:
            raise RuntimeError(
                f"apply_chat_template failed on prefix of length "
                f"{msg_idx + 1}: {exc}"
            ) from exc

        if len(new_ids) < len(prev_ids) or new_ids[: len(prev_ids)] != prev_ids:
            # Templates occasionally re-emit a trailing token (e.g.
            # final <|im_end|>) between prefixes. Fall back to the
            # longest common prefix in that case.
            common = 0
            for a, b in zip(new_ids, prev_ids):
                if a != b:
                    break
                common += 1
            delta = new_ids[common:]
        else:
            delta = new_ids[len(prev_ids):]

        is_assistant = messages[msg_idx].get("role") == "assistant"
        label_chunk = delta if is_assistant else [IGNORE_INDEX] * len(delta)
        input_ids.extend(delta)
        labels.extend(label_chunk)
        prev_ids = new_ids

    if len(input_ids) == 0:
        return None
    if all(label == IGNORE_INDEX for label in labels):
        return None

    if len(input_ids) > max_seq_length:
        # Keep the tail: the assistant tokens most often live at the end
        # of multi-turn conversations, and right-truncation would drop
        # them. Left-truncate the prompt/tool-result prefix instead.
        input_ids = input_ids[-max_seq_length:]
        labels = labels[-max_seq_length:]
        if all(label == IGNORE_INDEX for label in labels):
            return None

    return {"input_ids": input_ids, "labels": labels}


class DemoSFTDataset(Dataset):
    """HF-Dataset-like wrapper over the pre-tokenised demos.

    We materialise all demos up front (they are small: ~400 conversations).
    """

    def __init__(
        self,
        demos: list[dict[str, Any]],
        tokenizer: Any,
        max_seq_length: int,
    ):
        self._records: list[dict[str, list[int]]] = []
        self._tasks: list[str] = []
        self._categories: list[str] = []
        for demo in demos:
            rec = _build_example(
                demo.get("messages") or [],
                tokenizer,
                max_seq_length,
            )
            if rec is None:
                continue
            self._records.append(rec)
            self._tasks.append(str(demo.get("task_id", "")))
            self._categories.append(str(demo.get("category", "unknown")))

    def __len__(self) -> int:
        return len(self._records)

    def __getitem__(self, idx: int) -> dict[str, list[int]]:
        return self._records[idx]

    @property
    def num_assistant_tokens(self) -> int:
        return sum(
            sum(1 for lab in rec["labels"] if lab != IGNORE_INDEX)
            for rec in self._records
        )

    @property
    def num_total_tokens(self) -> int:
        return sum(len(rec["input_ids"]) for rec in self._records)

    def category_histogram(self) -> dict[str, int]:
        out: dict[str, int] = {}
        for cat in self._categories:
            out[cat] = out.get(cat, 0) + 1
        return out


# ---------------------------------------------------------------------------
# Collator: pad + build attention mask, preserve label=-100
# ---------------------------------------------------------------------------
@dataclass
class SFTCollator:
    """Right-pad variable-length records to the longest item in the batch.

    We intentionally do *not* use
    :class:`transformers.DataCollatorForLanguageModeling` because that
    either (a) masks 15% MLM-style, or (b) in the ``mlm=False`` causal
    mode it overwrites labels from input_ids, nuking our IGNORE_INDEX
    mask. A tiny custom collator is safer.
    """

    pad_token_id: int
    label_pad_id: int = IGNORE_INDEX

    def __call__(self, features: list[dict[str, list[int]]]) -> dict[str, torch.Tensor]:
        max_len = max(len(f["input_ids"]) for f in features)
        input_ids = torch.full(
            (len(features), max_len), self.pad_token_id, dtype=torch.long
        )
        attention_mask = torch.zeros((len(features), max_len), dtype=torch.long)
        labels = torch.full(
            (len(features), max_len), self.label_pad_id, dtype=torch.long
        )
        for i, f in enumerate(features):
            n = len(f["input_ids"])
            input_ids[i, :n] = torch.tensor(f["input_ids"], dtype=torch.long)
            attention_mask[i, :n] = 1
            labels[i, :n] = torch.tensor(f["labels"], dtype=torch.long)
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
        }


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
def _load_pool(path: str) -> list[dict[str, Any]]:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"demo pool not found: {p}")
    with p.open("r", encoding="utf-8") as f:
        raw = json.load(f)
    if not isinstance(raw, dict):
        raise ValueError(
            f"expected dict keyed by task_id, got {type(raw).__name__}"
        )
    return list(raw.values())


def _split(
    demos: list[dict[str, Any]], eval_ratio: float, seed: int
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    if eval_ratio <= 0.0:
        return demos, []
    rng = random.Random(seed)
    indices = list(range(len(demos)))
    rng.shuffle(indices)
    n_eval = max(1, int(round(len(indices) * eval_ratio)))
    eval_idx = set(indices[:n_eval])
    train, ev = [], []
    for i, demo in enumerate(demos):
        (ev if i in eval_idx else train).append(demo)
    return train, ev


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--demo-pool", required=True)
    parser.add_argument("--model-name-or-path", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--num-train-epochs", type=float, default=2.0)
    parser.add_argument("--learning-rate", type=float, default=5e-6)
    parser.add_argument("--lr-scheduler-type", default="cosine")
    parser.add_argument("--warmup-ratio", type=float, default=0.03)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--per-device-train-batch-size", type=int, default=1)
    parser.add_argument("--per-device-eval-batch-size", type=int, default=1)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=8)
    parser.add_argument("--max-seq-length", type=int, default=12288)
    parser.add_argument("--eval-ratio", type=float, default=0.1)
    parser.add_argument("--logging-steps", type=int, default=5)
    parser.add_argument("--save-strategy", default="epoch")
    # Renamed from ``evaluation_strategy`` to ``eval_strategy`` in
    # transformers >= 4.46; we keep the CLI flag under the new name.
    parser.add_argument("--eval-strategy", default="epoch")
    parser.add_argument("--save-total-limit", type=int, default=1)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--bf16",
        action="store_true",
        default=True,
        help="Enable bf16 training (default: on).",
    )
    parser.add_argument(
        "--gradient-checkpointing",
        action="store_true",
        default=True,
    )
    parser.add_argument(
        "--fsdp",
        default="full_shard auto_wrap",
        help="FSDP config string for TrainingArguments.fsdp.",
    )
    parser.add_argument(
        "--fsdp-transformer-layer-cls-to-wrap",
        default="Qwen2DecoderLayer",
        help="FSDP auto-wrap policy (architecture-specific).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    set_seed(args.seed)

    # ---- tokenizer + model ----
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_name_or_path, use_fast=True, trust_remote_code=True
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    # ---- data ----
    demos = _load_pool(args.demo_pool)
    train_demos, eval_demos = _split(demos, args.eval_ratio, args.seed)
    train_ds = DemoSFTDataset(train_demos, tokenizer, args.max_seq_length)
    eval_ds: DemoSFTDataset | None = (
        DemoSFTDataset(eval_demos, tokenizer, args.max_seq_length)
        if eval_demos
        else None
    )

    is_main_process = (
        int(os.environ.get("RANK", "0")) == 0
        and int(os.environ.get("LOCAL_RANK", "0")) == 0
    )
    if is_main_process:
        print(
            f"[sft] demo pool: {args.demo_pool}  "
            f"total_demos={len(demos)}  "
            f"train={len(train_ds)}  eval={len(eval_ds) if eval_ds else 0}",
            flush=True,
        )
        print(
            f"[sft] train tokens total={train_ds.num_total_tokens}  "
            f"assistant (supervised)={train_ds.num_assistant_tokens}  "
            f"assistant_ratio={train_ds.num_assistant_tokens / max(1, train_ds.num_total_tokens):.3f}",
            flush=True,
        )
        print(f"[sft] category hist (train): {train_ds.category_histogram()}", flush=True)

    model = AutoModelForCausalLM.from_pretrained(
        args.model_name_or_path,
        torch_dtype=torch.bfloat16 if args.bf16 else torch.float32,
        trust_remote_code=True,
        use_cache=False,
    )

    # ---- training ----
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=args.num_train_epochs,
        per_device_train_batch_size=args.per_device_train_batch_size,
        per_device_eval_batch_size=args.per_device_eval_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        lr_scheduler_type=args.lr_scheduler_type,
        warmup_ratio=args.warmup_ratio,
        weight_decay=args.weight_decay,
        logging_steps=args.logging_steps,
        save_strategy=args.save_strategy,
        eval_strategy=args.eval_strategy if eval_ds else "no",
        save_total_limit=args.save_total_limit,
        bf16=args.bf16,
        gradient_checkpointing=args.gradient_checkpointing,
        # gradient_checkpointing_kwargs avoids the silent re-entrant warning
        gradient_checkpointing_kwargs={"use_reentrant": False},
        fsdp=args.fsdp,
        fsdp_config={
            # Wrap policy: one FSDP unit per transformer decoder layer.
            # For Qwen2.5 this is "Qwen2DecoderLayer"; override via the
            # CLI flag for other architectures.
            "transformer_layer_cls_to_wrap": [args.fsdp_transformer_layer_cls_to_wrap],
            # NB: do NOT set activation_checkpointing here. We already
            # set gradient_checkpointing=True at the TrainingArguments
            # level; FSDP's own AC on top of that causes a re-entrant
            # autograd warning in transformers 4.53.x.
        },
        report_to=["none"],
        ddp_find_unused_parameters=False,
        seed=args.seed,
        dataloader_num_workers=2,
        remove_unused_columns=False,
        label_names=["labels"],
        optim="adamw_torch",
    )

    collator = SFTCollator(pad_token_id=tokenizer.pad_token_id)

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        data_collator=collator,
        tokenizer=tokenizer,
    )

    trainer.train()
    trainer.save_model(args.output_dir)
    # Save tokenizer alongside so GRPO can load it with just model.path=...
    if is_main_process:
        tokenizer.save_pretrained(args.output_dir)
        manifest = {
            "base_model": args.model_name_or_path,
            "demo_pool": args.demo_pool,
            "num_train": len(train_ds),
            "num_eval": len(eval_ds) if eval_ds else 0,
            "num_train_epochs": args.num_train_epochs,
            "learning_rate": args.learning_rate,
            "max_seq_length": args.max_seq_length,
            "assistant_tokens": train_ds.num_assistant_tokens,
            "total_tokens": train_ds.num_total_tokens,
            "category_histogram": train_ds.category_histogram(),
        }
        (Path(args.output_dir) / "sft_manifest.json").write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2)
        )
        print(f"[sft] wrote SFT checkpoint -> {args.output_dir}", flush=True)


if __name__ == "__main__":
    main()
