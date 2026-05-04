#!/usr/bin/env python3
"""Convert a verl FSDP DTensor actor checkpoint to HuggingFace format.

Run with the same world size used by the checkpoint, for example:

  torchrun --standalone --nproc_per_node=4 scripts/convert_verl_fsdp_dtensor_to_hf.py \
    --actor-dir checkpoints/.../global_step_90/actor \
    --target-dir checkpoints/.../global_step_90_hf
"""

from __future__ import annotations

import argparse
import os
import shutil
from collections import OrderedDict

import torch
import torch.distributed as dist
from accelerate import init_empty_weights
from torch.distributed.tensor import DTensor
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--actor-dir", required=True, help="Directory containing model_world_size_*_rank_*.pt")
    parser.add_argument("--target-dir", required=True, help="Output HuggingFace model directory")
    parser.add_argument("--torch-dtype", default="bfloat16", choices=["float16", "bfloat16", "float32"])
    parser.add_argument("--max-shard-size", default="2GB")
    parser.add_argument(
        "--backend",
        default="gloo",
        choices=["gloo", "nccl"],
        help="Distributed backend for DTensor collectives. Defaults to gloo because checkpoints are loaded on CPU.",
    )
    parser.add_argument(
        "--metadata-source",
        default=None,
        help=(
            "Optional HF model directory/repo to use for config, tokenizer, and generation metadata. "
            "Useful for Llama/ToolACE checkpoints where verl actor metadata may rewrite multi-EOS "
            "or tokenizer fields."
        ),
    )
    return parser.parse_args()


def dtype_from_name(name: str) -> torch.dtype:
    return {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
    }[name]


def copy_metadata_files(source_dir: str, target_dir: str) -> None:
    """Preserve HF metadata files exactly when a source model directory is provided."""
    metadata_files = [
        "config.json",
        "generation_config.json",
        "tokenizer.json",
        "tokenizer_config.json",
        "special_tokens_map.json",
        "added_tokens.json",
        "chat_template.jinja",
        "tokenizer.model",
        "merges.txt",
        "vocab.json",
    ]
    for filename in metadata_files:
        source_path = os.path.join(source_dir, filename)
        if os.path.exists(source_path):
            shutil.copy2(source_path, os.path.join(target_dir, filename))


def main() -> None:
    args = parse_args()
    dtype = dtype_from_name(args.torch_dtype)

    dist.init_process_group(backend=args.backend)
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    local_rank = int(os.environ.get("LOCAL_RANK", rank))
    if args.backend == "nccl":
        cuda_device_count = torch.cuda.device_count()
        if cuda_device_count < world_size:
            raise RuntimeError(
                f"NCCL backend needs at least {world_size} visible CUDA devices, "
                f"but only {cuda_device_count} are visible."
            )
        torch.cuda.set_device(local_rank)
    elif rank == 0:
        print("Using CPU/gloo DTensor collectives for checkpoint merge.")

    shard_path = os.path.join(args.actor_dir, f"model_world_size_{world_size}_rank_{rank}.pt")
    if not os.path.exists(shard_path):
        raise FileNotFoundError(f"Missing shard for rank {rank}: {shard_path}")

    checkpoint = torch.load(shard_path, map_location="cpu", weights_only=False)
    full_state = OrderedDict()

    for name, value in checkpoint.items():
        if isinstance(value, DTensor):
            full_value = value.full_tensor()
        else:
            full_value = value

        if rank == 0:
            if torch.is_tensor(full_value):
                full_value = full_value.detach().cpu()
                if full_value.is_floating_point():
                    full_value = full_value.to(dtype)
            full_state[name] = full_value

        del full_value

    if rank == 0:
        os.makedirs(args.target_dir, exist_ok=True)
        metadata_source = args.metadata_source or args.actor_dir
        config = AutoConfig.from_pretrained(metadata_source, trust_remote_code=True)
        with init_empty_weights():
            model = AutoModelForCausalLM.from_config(config, torch_dtype=dtype, trust_remote_code=True)
        model.save_pretrained(
            args.target_dir,
            state_dict=full_state,
            safe_serialization=True,
            max_shard_size=args.max_shard_size,
        )
        tokenizer = AutoTokenizer.from_pretrained(metadata_source, trust_remote_code=True)
        tokenizer.save_pretrained(args.target_dir)

        generation_config = os.path.join(metadata_source, "generation_config.json")
        if os.path.exists(generation_config):
            from transformers import GenerationConfig

            GenerationConfig.from_pretrained(metadata_source).save_pretrained(args.target_dir)

        if args.metadata_source:
            copy_metadata_files(args.metadata_source, args.target_dir)

        print(f"Saved HuggingFace checkpoint to {args.target_dir}")

    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
