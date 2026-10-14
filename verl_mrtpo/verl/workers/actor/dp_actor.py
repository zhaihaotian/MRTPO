# Copyright 2024 Bytedance Ltd. and/or its affiliates
# Copyright 2023-2024 SGLang Team
# Copyright 2025 ModelBest Inc. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Single Process Actor
"""

import logging
import os

import torch
from torch import nn
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP

import verl.utils.torch_functional as verl_F
from verl import DataProto
from verl.trainer.ppo.core_algos import agg_loss, compute_policy_loss, get_policy_loss_fn, kl_penalty
from verl.utils.device import get_device_name, is_cuda_available, is_npu_available
from verl.utils.fsdp_utils import FSDPModule, fsdp2_clip_grad_norm_
from verl.utils.profiler import GPUMemoryLogger
from verl.utils.py_functional import append_to_dict
from verl.utils.seqlen_balancing import prepare_dynamic_batch, restore_dynamic_batch
from verl.utils.torch_functional import logprobs_from_logits
from verl.utils.ulysses import gather_outputs_and_unpad, ulysses_pad, ulysses_pad_and_slice_inputs
from verl.workers.actor import BasePPOActor

if is_cuda_available:
    from flash_attn.bert_padding import index_first_axis, pad_input, rearrange, unpad_input
elif is_npu_available:
    from transformers.integrations.npu_flash_attention import index_first_axis, pad_input, rearrange, unpad_input


__all__ = ["DataParallelPPOActor"]

logger = logging.getLogger(__file__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))


class DataParallelPPOActor(BasePPOActor):
    def __init__(self, config, actor_module: nn.Module, actor_optimizer: torch.optim.Optimizer = None):
        """When optimizer is None, it is Reference Policy"""
        super().__init__(config)
        self.actor_module = actor_module
        self.actor_optimizer = actor_optimizer

        self.use_remove_padding = self.config.get("use_remove_padding", False)
        if torch.distributed.get_rank() == 0:
            print(f"Actor use_remove_padding={self.use_remove_padding}")
        self.use_fused_kernels = self.config.get("use_fused_kernels", False)
        if torch.distributed.get_rank() == 0:
            print(f"Actor use_fused_kernels={self.use_fused_kernels}")

        self.ulysses_sequence_parallel_size = self.config.ulysses_sequence_parallel_size
        self.use_ulysses_sp = self.ulysses_sequence_parallel_size > 1

        if self.config.entropy_from_logits_with_chunking:
            entropy_from_logits = verl_F.entropy_from_logits_with_chunking
        else:
            entropy_from_logits = verl_F.entropy_from_logits

        self.compute_entropy_from_logits = (
            torch.compile(entropy_from_logits, dynamic=True)
            if self.config.get("use_torch_compile", True)  #  use torch compile by default
            else entropy_from_logits
        )
        self.device_name = get_device_name()

    def _forward_micro_batch(
        self, micro_batch, temperature, calculate_entropy=False, topk=None
    ):
        """
        Returns:
            entropy: # (bs, response_len)
            log_probs: # (bs, response_len)
            distribution (OPSD only): gathered student log-probs or teacher top-K tensors
        """
        response_length = micro_batch["responses"].size(-1)
        opsd_student = "opsd_topk_ids" in micro_batch
        need_distribution = topk is not None or opsd_student
        if need_distribution:
            from mrtpo.opsd import gather_student_log_probs, teacher_topk_log_probs

            if self.use_fused_kernels:
                raise ValueError("OPSD requires unfused full-vocabulary logits")
            distribution = {}
        multi_modal_inputs = {}
        if "multi_modal_inputs" in micro_batch.keys():
            if "image_bound" in micro_batch["multi_modal_inputs"][0]:  # minicpm-o logic
                for key in micro_batch["multi_modal_inputs"][0].keys():
                    multi_modal_inputs[key] = [inputs[key] for inputs in micro_batch["multi_modal_inputs"]]
            else:
                for key in micro_batch["multi_modal_inputs"][0].keys():
                    multi_modal_inputs[key] = torch.cat(
                        [inputs[key] for inputs in micro_batch["multi_modal_inputs"]], dim=0
                    )

        with torch.autocast(device_type=self.device_name, dtype=torch.bfloat16):
            input_ids = micro_batch["input_ids"]
            batch_size, seqlen = input_ids.shape
            attention_mask = micro_batch["attention_mask"]
            position_ids = micro_batch["position_ids"]
            entropy = None
            if position_ids.dim() == 3:  # qwen2vl mrope
                position_ids = position_ids.transpose(0, 1)  # (bsz, 3, seqlen) -> (3, bsz, seqlen)

            if self.use_remove_padding:
                input_ids_rmpad, indices, cu_seqlens, *_ = unpad_input(
                    input_ids.unsqueeze(-1), attention_mask
                )  # input_ids_rmpad (total_nnz, ...)
                input_ids_rmpad = input_ids_rmpad.transpose(0, 1)  # (1, total_nnz)

                # unpad the position_ids to align the rotary
                if position_ids.dim() == 3:
                    position_ids_rmpad = (
                        index_first_axis(rearrange(position_ids, "c b s ... -> (b s) c ..."), indices)
                        .transpose(0, 1)
                        .unsqueeze(1)
                    )  # (3, bsz, seqlen) -> (3, 1, bsz * seqlen)
                else:
                    position_ids_rmpad = index_first_axis(
                        rearrange(position_ids.unsqueeze(-1), "b s ... -> (b s) ..."), indices
                    ).transpose(0, 1)

                if "image_bound" in multi_modal_inputs:
                    from verl.utils.dataset.vision_utils import process_multi_modal_inputs_for_minicpmo

                    multi_modal_inputs = process_multi_modal_inputs_for_minicpmo(
                        input_ids, attention_mask, position_ids, cu_seqlens, multi_modal_inputs
                    )

                # for compute the log_prob
                input_ids_rmpad_rolled = torch.roll(input_ids_rmpad, shifts=-1, dims=1)  # (1, total_nnz)

                if opsd_student:
                    # Align teacher IDs with the states that predict each response token.
                    teacher_ids = micro_batch["opsd_topk_ids"]
                    padded_ids = teacher_ids.new_zeros((batch_size, seqlen, teacher_ids.size(-1)))
                    padded_ids[:, -response_length - 1 : -1] = teacher_ids
                    packed_ids = index_first_axis(padded_ids.flatten(0, 1), indices)
                    if self.use_ulysses_sp:
                        packed_ids, _, _ = ulysses_pad_and_slice_inputs(
                            packed_ids.transpose(0, 1), position_ids_rmpad=None,
                            sp_size=self.ulysses_sequence_parallel_size,
                        )
                        packed_ids = packed_ids.transpose(0, 1)

                # pad and slice the inputs if sp > 1
                if self.use_ulysses_sp:
                    is_vlm_model = "multi_modal_inputs" in micro_batch.keys()
                    if is_vlm_model:
                        # vlm model's inputs will be sliced after embedding
                        input_ids_rmpad, position_ids_rmpad, pad_size = ulysses_pad(
                            input_ids_rmpad,
                            position_ids_rmpad=position_ids_rmpad,
                            sp_size=self.ulysses_sequence_parallel_size,
                        )
                    else:
                        input_ids_rmpad, position_ids_rmpad, pad_size = ulysses_pad_and_slice_inputs(
                            input_ids_rmpad,
                            position_ids_rmpad=position_ids_rmpad,
                            sp_size=self.ulysses_sequence_parallel_size,
                        )
                    input_ids_rmpad_rolled, _, _ = ulysses_pad_and_slice_inputs(
                        input_ids_rmpad_rolled,
                        position_ids_rmpad=None,
                        sp_size=self.ulysses_sequence_parallel_size,
                    )

                input_ids_rmpad_rolled = input_ids_rmpad_rolled.squeeze(0)  # ((total_nnz / sp) + pad)

                # only pass input_ids and position_ids to enable flash_attn_varlen
                extra_args = {}
                if self.use_fused_kernels:
                    extra_args["temperature"] = temperature
                    extra_args["return_dict"] = True

                output = self.actor_module(
                    input_ids=input_ids_rmpad,
                    attention_mask=None,
                    position_ids=position_ids_rmpad,
                    **multi_modal_inputs,
                    use_cache=False,
                    **extra_args,
                )  # prevent model thinks we are generating

                if self.use_fused_kernels:
                    log_probs = output.log_probs.squeeze(0)  # (total_nnz,)
                    entropy_rmpad = output.entropy.squeeze(0)  # (total_nnz,)

                else:
                    logits_rmpad = output.logits.squeeze(0)  # (total_nnz, vocab_size)
                    logits_rmpad.div_(temperature)

                    if topk is not None:
                        distribution = teacher_topk_log_probs(logits_rmpad, topk)
                    elif opsd_student:
                        distribution["student_topk_logp"] = gather_student_log_probs(logits_rmpad, packed_ids)

                    # if use_sp: ((total_nnz / sp) + pad) ; if not use_sp: (batch, seqlen)
                    inplace_backward = True
                    if calculate_entropy:
                        inplace_backward = False
                    if opsd_student:
                        inplace_backward = False
                    log_probs = logprobs_from_logits(
                        logits=logits_rmpad,
                        labels=input_ids_rmpad_rolled,
                        inplace_backward=inplace_backward,
                    )

                    # compute entropy
                    if calculate_entropy:
                        if not self.config.entropy_checkpointing:
                            entropy_rmpad = self.compute_entropy_from_logits(logits_rmpad)  # ((total_nnz / sp) + pad)
                        else:
                            entropy_rmpad = torch.utils.checkpoint.checkpoint(
                                self.compute_entropy_from_logits, logits_rmpad
                            )

                # gather log_prob if sp > 1
                if self.use_ulysses_sp:
                    # gather and unpad for the ulysses sp
                    log_probs = gather_outputs_and_unpad(
                        log_probs,
                        gather_dim=0,
                        unpad_dim=0,
                        padding_size=pad_size,
                    )
                    if calculate_entropy:
                        entropy_rmpad = gather_outputs_and_unpad(
                            entropy_rmpad,
                            gather_dim=0,
                            unpad_dim=0,
                            padding_size=pad_size,
                        )
                # pad back to (bsz, seqlen)
                if calculate_entropy:
                    full_entropy = pad_input(
                        hidden_states=entropy_rmpad.unsqueeze(-1),
                        indices=indices,
                        batch=batch_size,
                        seqlen=seqlen,
                    )
                full_log_probs = pad_input(
                    hidden_states=log_probs.unsqueeze(-1),
                    indices=indices,
                    batch=batch_size,
                    seqlen=seqlen,
                )

                # only return response part:
                if calculate_entropy:
                    entropy = full_entropy.squeeze(-1)[:, -response_length - 1 : -1]  # (bsz, response_length)
                log_probs = full_log_probs.squeeze(-1)[:, -response_length - 1 : -1]  # (bsz, response_length)

                if need_distribution:
                    for key, values in distribution.items():
                        if self.use_ulysses_sp:
                            values = gather_outputs_and_unpad(
                                values, gather_dim=0, unpad_dim=0, padding_size=pad_size,
                            )
                        distribution[key] = pad_input(
                            hidden_states=values, indices=indices, batch=batch_size, seqlen=seqlen,
                        )[:, -response_length - 1 : -1]

            else:  # not using rmpad and no ulysses sp
                extra_args = {}
                if self.use_fused_kernels:
                    extra_args["temperature"] = temperature
                    extra_args["return_dict"] = True

                output = self.actor_module(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    position_ids=position_ids,
                    **multi_modal_inputs,
                    use_cache=False,
                    **extra_args,
                )  # prevent model thinks we are generating

                if self.use_fused_kernels:
                    log_probs = output.log_probs[:, -response_length - 1 : -1]
                    entropy = output.entropy[:, -response_length - 1 : -1]  # (bsz, response_length)

                else:
                    logits = output.logits

                    logits.div_(temperature)
                    logits = logits[:, -response_length - 1 : -1, :]  # (bsz, response_length, vocab_size)
                    if topk is not None:
                        distribution = teacher_topk_log_probs(logits, topk)
                    elif opsd_student:
                        distribution["student_topk_logp"] = gather_student_log_probs(
                            logits, micro_batch["opsd_topk_ids"],
                        )
                    if opsd_student:
                        log_probs = logprobs_from_logits(logits, micro_batch["responses"], inplace_backward=False)
                    else:
                        log_probs = logprobs_from_logits(logits, micro_batch["responses"])
                    if calculate_entropy:
                        if not self.config.entropy_checkpointing:
                            entropy = verl_F.entropy_from_logits(logits)  # (bsz, response_length)
                        else:
                            entropy = torch.utils.checkpoint.checkpoint(verl_F.entropy_from_logits, logits)

            if need_distribution:
                return entropy, log_probs, distribution
            return entropy, log_probs

    def _optimizer_step(self):
        assert self.config.grad_clip is not None

        if isinstance(self.actor_module, FSDP):
            grad_norm = self.actor_module.clip_grad_norm_(max_norm=self.config.grad_clip)
        elif isinstance(self.actor_module, FSDPModule):
            grad_norm = fsdp2_clip_grad_norm_(self.actor_module.parameters(), max_norm=self.config.grad_clip)
        else:
            grad_norm = torch.nn.utils.clip_grad_norm_(self.actor_module.parameters(), max_norm=self.config.grad_clip)

        # if grad_norm is not finite, skip the update
        if not torch.isfinite(grad_norm):
            print(f"WARN: rank {torch.distributed.get_rank()} grad_norm is not finite: {grad_norm}")
            self.actor_optimizer.zero_grad()
        else:
            self.actor_optimizer.step()
        return grad_norm

    def _forward_response_distribution(self, micro_batch, prefix=""):
        """Native-dtype logits at response predecessors, using the log-prob attention path."""
        if self.use_ulysses_sp or self.use_fused_kernels:
            raise ValueError("Criterion JS requires sequence parallel size 1 and unfused logits")
        response_length = micro_batch["responses"].size(-1)
        input_ids = micro_batch[f"{prefix}input_ids"]
        attention_mask = micro_batch[f"{prefix}attention_mask"]
        position_ids = micro_batch[f"{prefix}position_ids"]
        response_mask = micro_batch["response_mask"].bool()
        batch_size, seqlen = input_ids.shape
        with torch.autocast(device_type=self.device_name, dtype=torch.bfloat16):
            if self.use_remove_padding:
                ids, indices, *_ = unpad_input(input_ids.unsqueeze(-1), attention_mask)
                ids = ids.transpose(0, 1)
                positions = index_first_axis(
                    rearrange(position_ids.unsqueeze(-1), "b s ... -> (b s) ..."), indices
                ).transpose(0, 1)
                output = self.actor_module(
                    input_ids=ids, attention_mask=None, position_ids=positions, use_cache=False
                )
                flat_to_unpadded = torch.full(
                    (batch_size * seqlen,), -1, dtype=torch.long, device=input_ids.device
                )
                flat_to_unpadded[indices] = torch.arange(indices.numel(), device=input_ids.device)
                row_offsets = torch.arange(batch_size, device=input_ids.device)[:, None] * seqlen
                predecessors = torch.arange(seqlen - response_length - 1, seqlen - 1, device=input_ids.device)
                selected = flat_to_unpadded[(row_offsets + predecessors)[response_mask]]
                if torch.any(selected < 0):
                    raise ValueError("A response token is missing its predecessor state")
                logits = output.logits.squeeze(0).index_select(0, selected)
            else:
                output = self.actor_module(
                    input_ids=input_ids, attention_mask=attention_mask, position_ids=position_ids, use_cache=False
                )
                logits = output.logits[:, -response_length - 1 : -1][response_mask]
        return logits

    @torch.no_grad()
    def compute_criterion_js(self, data: DataProto):
        """MRTPO replay: per-token JS between the full-context and replay-context distributions of fixed responses."""
        from mrtpo.js import chunked_distribution_js

        self.actor_module.eval()
        if data.meta_info["use_dynamic_bsz"]:
            # P0 is longer than P-minus-j; size each paired forward by P0.
            sizing = DataProto.from_dict({"attention_mask": data.batch["full_attention_mask"]})
            _, indices = prepare_dynamic_batch(sizing, max_token_len=data.meta_info["max_token_len"])
            micro_batches = [data[index] for index in indices]
        else:
            micro_batches = data.split(data.meta_info["micro_batch_size"])
        outputs = []
        for micro_batch in micro_batches:
            full_logits = self._forward_response_distribution(micro_batch.batch, "full_")
            minus_logits = self._forward_response_distribution(micro_batch.batch)
            response_mask = micro_batch.batch["response_mask"].bool()
            js = chunked_distribution_js(
                full_logits, minus_logits,
                chunk_size=data.meta_info["js_chunk_size"], temperature=data.meta_info["temperature"],
            )
            result = torch.zeros_like(response_mask, dtype=torch.float32)
            result[response_mask] = js
            outputs.append(result)
            del full_logits, minus_logits
        output = torch.cat(outputs, dim=0)
        if data.meta_info["use_dynamic_bsz"]:
            output = restore_dynamic_batch(output, indices)
        return output

    @GPUMemoryLogger(role="dp actor", logger=logger)
    def compute_log_prob(self, data: DataProto, calculate_entropy=False) -> torch.Tensor:
        """Compute the log probability of the responses given input_ids, attention_mask and position_ids

        Args:
            data (DataProto): a DataProto containing keys

                ``input_ids``: tensor of shape [batch_size, sequence_length]. torch.int64. Note that input_ids is the
                concatenation of prompt and response. Note that ``sequence_length = prompt_length + response_length``.

                ``attention_mask``: tensor of shape [batch_size, sequence_length]. torch.int64.

                ``position_ids``: tensor of shape [batch_size, sequence_length]. torch.int64.

                ``responses``:  tensor of shape [batch_size, response_length]. torch.int64.

        Returns:
            torch.Tensor: the log_prob tensor
        """
        # set to eval
        self.actor_module.eval()

        micro_batch_size = data.meta_info["micro_batch_size"]
        temperature = data.meta_info["temperature"]  # temperature must be in the data.meta_info to avoid silent error
        use_dynamic_bsz = data.meta_info["use_dynamic_bsz"]
        has_multi_modal_inputs = "multi_modal_inputs" in data.non_tensor_batch.keys()
        select_keys = ["responses", "input_ids", "attention_mask", "position_ids"]
        non_tensor_select_keys = ["multi_modal_inputs"] if has_multi_modal_inputs else []

        data = data.select(batch_keys=select_keys, non_tensor_batch_keys=non_tensor_select_keys)

        if use_dynamic_bsz:
            max_token_len = data.meta_info["max_token_len"] * self.ulysses_sequence_parallel_size
            micro_batches, batch_idx_list = prepare_dynamic_batch(data, max_token_len=max_token_len)
        else:
            micro_batches = data.split(micro_batch_size)

        log_probs_lst = []
        entropy_lst = []
        for micro_batch in micro_batches:
            model_inputs = {**micro_batch.batch, **micro_batch.non_tensor_batch}
            with torch.no_grad():
                entropy, log_probs = self._forward_micro_batch(
                    model_inputs, temperature=temperature, calculate_entropy=calculate_entropy
                )
            log_probs_lst.append(log_probs)
            if calculate_entropy:
                entropy_lst.append(entropy)

        log_probs = torch.concat(log_probs_lst, dim=0)
        entropys = None
        if calculate_entropy:
            entropys = torch.concat(entropy_lst, dim=0)

        if use_dynamic_bsz:
            log_probs = restore_dynamic_batch(log_probs, batch_idx_list)
            if calculate_entropy:
                entropys = restore_dynamic_batch(entropys, batch_idx_list)

        return log_probs, entropys

    @torch.no_grad()
    def compute_topk_log_probs(self, data: DataProto, k):
        """Evaluate the frozen teacher with the reference log-prob batching policy."""
        self.actor_module.eval()
        data = data.select(batch_keys=["responses", "response_mask", "input_ids", "attention_mask", "position_ids"])
        if data.meta_info["use_dynamic_bsz"]:
            max_token_len = data.meta_info["max_token_len"] * self.ulysses_sequence_parallel_size
            micro_batches, indices = prepare_dynamic_batch(data, max_token_len=max_token_len)
        else:
            micro_batches = data.split(data.meta_info["micro_batch_size"])
        outputs = []
        for micro_batch in micro_batches:
            _, _, distribution = self._forward_micro_batch(
                micro_batch.batch, temperature=data.meta_info["temperature"], topk=k,
            )
            distribution["teacher_topk_mass"] = distribution["opsd_topk_logp"].exp().sum(-1)
            outputs.append(distribution)
        result = {key: torch.cat([output[key] for output in outputs], dim=0) for key in outputs[0]}
        if data.meta_info["use_dynamic_bsz"]:
            result = {key: restore_dynamic_batch(value, indices) for key, value in result.items()}
        return result

    @GPUMemoryLogger(role="dp actor", logger=logger)
    def update_policy(self, data: DataProto):
        # make sure we are in training mode
        self.actor_module.train()

        temperature = data.meta_info["temperature"]  # temperature must be in the data.meta_info to avoid silent error

        select_keys = [
            "responses",
            "response_mask",
            "input_ids",
            "attention_mask",
            "position_ids",
            "old_log_probs",
            "advantages",
        ]
        if self.config.use_kl_loss:
            select_keys.append("ref_log_prob")
        opsd_loss = self.config.get("opsd_loss", False)
        if opsd_loss:
            select_keys.extend(["opsd_topk_ids", "opsd_topk_logp"])

        has_multi_modal_inputs = "multi_modal_inputs" in data.non_tensor_batch.keys()
        non_tensor_select_keys = ["multi_modal_inputs"] if has_multi_modal_inputs else []

        data = data.select(batch_keys=select_keys, non_tensor_batch_keys=non_tensor_select_keys)

        # Split to make minibatch iterator for updating the actor
        # See PPO paper for details. https://arxiv.org/abs/1707.06347
        mini_batches = data.split(self.config.ppo_mini_batch_size)

        metrics = {}
        if opsd_loss:
            opsd_totals = torch.zeros(3, device=data.batch["responses"].device, dtype=torch.float64)
        global_token_loss = self.config.get("global_token_loss", False)
        if global_token_loss and self.config.loss_agg_mode != "token-mean":
            raise ValueError("global_token_loss requires loss_agg_mode=token-mean")
        for _ in range(self.config.ppo_epochs):
            for batch_idx, mini_batch in enumerate(mini_batches):
                if self.config.use_dynamic_bsz:
                    max_token_len = self.config.ppo_max_token_len_per_gpu * self.ulysses_sequence_parallel_size
                    micro_batches, _ = prepare_dynamic_batch(mini_batch, max_token_len=max_token_len)
                else:
                    self.gradient_accumulation = (
                        self.config.ppo_mini_batch_size // self.config.ppo_micro_batch_size_per_gpu
                    )
                    micro_batches = mini_batch.split(self.config.ppo_micro_batch_size_per_gpu)

                if global_token_loss:
                    global_tokens = mini_batch.batch["response_mask"].sum().clone()
                    world_size = 1
                    if torch.distributed.is_initialized():
                        world_size = torch.distributed.get_world_size()
                        torch.distributed.all_reduce(global_tokens)
                    if global_tokens.item() == 0:
                        raise ValueError("An optimizer mini-batch must contain response tokens")
                    append_to_dict(metrics, {
                        "actor/global_response_tokens_per_update": global_tokens.item(),
                        "actor/micro_batches_per_update": len(micro_batches),
                    })

                self.actor_optimizer.zero_grad()

                for micro_batch in micro_batches:
                    micro_batch_metrics = {}
                    model_inputs = {**micro_batch.batch, **micro_batch.non_tensor_batch}
                    response_mask = model_inputs["response_mask"]
                    old_log_prob = model_inputs["old_log_probs"]
                    advantages = model_inputs["advantages"]

                    clip_ratio = self.config.clip_ratio
                    clip_ratio_low = (
                        self.config.clip_ratio_low if self.config.clip_ratio_low is not None else clip_ratio
                    )
                    clip_ratio_high = (
                        self.config.clip_ratio_high if self.config.clip_ratio_high is not None else clip_ratio
                    )
                    clip_ratio_c = self.config.get("clip_ratio_c", 3.0)
                    entropy_coeff = self.config.entropy_coeff
                    loss_agg_mode = self.config.loss_agg_mode

                    # all return: (bsz, response_length)
                    calculate_entropy = False
                    if entropy_coeff != 0:
                        calculate_entropy = True
                    if opsd_loss:
                        from mrtpo.opsd import opsd_token_loss

                        entropy, log_prob, distribution = self._forward_micro_batch(
                            model_inputs, temperature=temperature, calculate_entropy=calculate_entropy
                        )
                        token_loss, clip_fraction = opsd_token_loss(
                            distribution["student_topk_logp"], model_inputs["opsd_topk_logp"],
                            self.config.opsd_token_clip,
                        )
                        pg_loss = agg_loss(token_loss, response_mask, loss_agg_mode)
                        with torch.no_grad():
                            opsd_totals[0] += pg_loss.detach().double() * response_mask.sum()
                            opsd_totals[1] += (clip_fraction.double() * response_mask).sum()
                            opsd_totals[2] += response_mask.sum()
                    else:
                        entropy, log_prob = self._forward_micro_batch(
                            model_inputs, temperature=temperature, calculate_entropy=calculate_entropy
                        )

                    loss_mode = self.config.policy_loss.get("loss_mode", "vanilla")

                    if opsd_loss:
                        pass
                    elif self.config.policy_loss.loss_mode == "vanilla":
                        pg_loss, pg_clipfrac, ppo_kl, pg_clipfrac_lower = compute_policy_loss(
                            old_log_prob=old_log_prob,
                            log_prob=log_prob,
                            advantages=advantages,
                            response_mask=response_mask,
                            cliprange=clip_ratio,
                            cliprange_low=clip_ratio_low,
                            cliprange_high=clip_ratio_high,
                            clip_ratio_c=clip_ratio_c,
                            loss_agg_mode=loss_agg_mode,
                        )

                    else:
                        policy_loss_fn = get_policy_loss_fn(loss_mode)
                        pg_loss, pg_clipfrac, ppo_kl, pg_clipfrac_lower = policy_loss_fn(
                            old_log_prob=old_log_prob,
                            log_prob=log_prob,
                            advantages=advantages,
                            response_mask=response_mask,
                            loss_agg_mode=loss_agg_mode,
                            config=self.config,
                        )

                    if entropy_coeff != 0:
                        entropy_loss = agg_loss(loss_mat=entropy, loss_mask=response_mask, loss_agg_mode=loss_agg_mode)

                        # compute policy loss
                        policy_loss = pg_loss - entropy_loss * entropy_coeff
                    else:
                        policy_loss = pg_loss

                    if self.config.use_kl_loss:
                        ref_log_prob = model_inputs["ref_log_prob"]
                        # compute kl loss
                        kld = kl_penalty(
                            logprob=log_prob, ref_logprob=ref_log_prob, kl_penalty=self.config.kl_loss_type
                        )
                        kl_loss = agg_loss(loss_mat=kld, loss_mask=response_mask, loss_agg_mode=loss_agg_mode)

                        policy_loss = policy_loss + kl_loss * self.config.kl_loss_coef
                        micro_batch_metrics["actor/kl_loss"] = kl_loss.detach().item()
                        micro_batch_metrics["actor/kl_coef"] = self.config.kl_loss_coef

                    if global_token_loss:
                        # FSDP averages rank gradients; undo that average and use
                        # the all-rank optimizer mini-batch's token denominator.
                        loss = policy_loss * (world_size * response_mask.sum() / global_tokens)
                    elif self.config.use_dynamic_bsz:
                        # relative to the dynamic bsz
                        loss = policy_loss * (response_mask.shape[0] / self.config.ppo_mini_batch_size)
                    else:
                        loss = policy_loss / self.gradient_accumulation
                    loss.backward()

                    if not opsd_loss:
                        micro_batch_metrics.update(
                            {
                                "actor/pg_loss": pg_loss.detach().item(),
                                "actor/pg_clipfrac": pg_clipfrac.detach().item(),
                                "actor/ppo_kl": ppo_kl.detach().item(),
                                "actor/pg_clipfrac_lower": pg_clipfrac_lower.detach().item(),
                            }
                        )
                    append_to_dict(metrics, micro_batch_metrics)

                grad_norm = self._optimizer_step()
                mini_batch_metrics = {"actor/grad_norm": grad_norm.detach().item()}
                append_to_dict(metrics, mini_batch_metrics)
        if opsd_loss:
            if torch.distributed.is_initialized():
                torch.distributed.all_reduce(opsd_totals)
            append_to_dict(metrics, {
                "actor/opsd_loss": (opsd_totals[0] / opsd_totals[2]).item(),
                "actor/opsd_clip_frac": (opsd_totals[1] / opsd_totals[2]).item(),
            })
        self.actor_optimizer.zero_grad()
        return metrics
