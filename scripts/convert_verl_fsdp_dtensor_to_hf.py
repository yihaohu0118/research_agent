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
    return parser.parse_args()


def dtype_from_name(name: str) -> torch.dtype:
    return {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
    }[name]


def main() -> None:
    args = parse_args()
    dtype = dtype_from_name(args.torch_dtype)

    backend = "nccl" if torch.cuda.is_available() else "gloo"
    dist.init_process_group(backend=backend)
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    local_rank = int(os.environ.get("LOCAL_RANK", rank))
    if torch.cuda.is_available():
        torch.cuda.set_device(local_rank)

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
        config = AutoConfig.from_pretrained(args.actor_dir, trust_remote_code=True)
        with init_empty_weights():
            model = AutoModelForCausalLM.from_config(config, torch_dtype=dtype, trust_remote_code=True)
        model.save_pretrained(
            args.target_dir,
            state_dict=full_state,
            safe_serialization=True,
            max_shard_size=args.max_shard_size,
        )
        tokenizer = AutoTokenizer.from_pretrained(args.actor_dir, trust_remote_code=True)
        tokenizer.save_pretrained(args.target_dir)

        generation_config = os.path.join(args.actor_dir, "generation_config.json")
        if os.path.exists(generation_config):
            from transformers import GenerationConfig

            GenerationConfig.from_pretrained(args.actor_dir).save_pretrained(args.target_dir)

        print(f"Saved HuggingFace checkpoint to {args.target_dir}")

    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
