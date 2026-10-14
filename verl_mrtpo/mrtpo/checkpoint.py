"""Checkpoint export and data-order restoration utilities."""

import json
import os
import shutil
from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from torchdata.stateful_dataloader import StatefulDataLoader


def restore_dataloader_state(dataloader: "StatefulDataLoader", state: dict, global_step: int) -> None:
    """Restore a fresh loader, advancing an untracked RandomSampler across completed epochs.

    The sampler must have the same initial seed and dataset as the saved run.
    Stateful and custom samplers retain their own restoration behavior.
    """
    sampler = dataloader.sampler
    if type(sampler) is torch.utils.data.RandomSampler and sampler.generator is not None:
        if "_snapshot" in state:
            snapshot = state["_snapshot"]
            position = snapshot["_snapshot_step"] + state["_steps_since_snapshot"]
            sampler_state = snapshot["_main_snapshot"]["_index_sampler_state"]
        else:
            position = state["_num_yielded"]
            sampler_state = state["_index_sampler_state"]
        if sampler_state is None:
            completed_steps = global_step - position
            if completed_steps < 0 or completed_steps % len(dataloader):
                raise ValueError(
                    f"Checkpoint step {global_step} and loader position {position} "
                    f"are inconsistent with {len(dataloader)} batches per epoch"
                )
            completed_epochs = completed_steps // len(dataloader)
            for _ in range(completed_epochs):
                # Exhaustion also consumes RandomSampler's trailing randperm call.
                for _ in sampler:
                    pass
            print(f"Restored RandomSampler: completed_epochs={completed_epochs} "
                  f"epoch_position={position} next_global_step={global_step + 1}")
    dataloader.load_state_dict(state)


def export_hf_bf16(source, target):
    """Move `source` (verl's actor/huggingface dir) to `target`, casting fp32 weights to bf16.

    verl writes the HF export with the fp32 master weights; evaluation loads bf16 anyway, so
    casting halves disk use without changing what vLLM sees. The target appears atomically.
    """
    from safetensors.torch import load_file, save_file

    if not os.path.isdir(source):
        raise FileNotFoundError(f"no HF export at {source}")
    staging = target + ".partial"
    shutil.rmtree(staging, ignore_errors=True)
    os.makedirs(os.path.dirname(target), exist_ok=True)
    shutil.move(source, staging)
    total = 0
    for name in sorted(os.listdir(staging)):
        if not name.endswith(".safetensors"):
            continue
        path = os.path.join(staging, name)
        tensors = load_file(path)
        tensors = {key: (value.to(torch.bfloat16) if value.dtype == torch.float32 else value).contiguous()
                   for key, value in tensors.items()}
        total += sum(value.numel() * value.element_size() for value in tensors.values())
        save_file(tensors, path, metadata={"format": "pt"})
    index = os.path.join(staging, "model.safetensors.index.json")
    if os.path.exists(index):
        with open(index) as stream:
            data = json.load(stream)
        data.setdefault("metadata", {})["total_size"] = total
        with open(index, "w") as stream:
            json.dump(data, stream, indent=2)
    config = os.path.join(staging, "config.json")
    if os.path.exists(config):
        with open(config) as stream:
            data = json.load(stream)
        data["torch_dtype"] = "bfloat16"
        with open(config, "w") as stream:
            json.dump(data, stream, indent=2)
    shutil.rmtree(target, ignore_errors=True)
    os.rename(staging, target)
    return target
