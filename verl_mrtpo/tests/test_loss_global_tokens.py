"""Token-mean loss over the whole optimizer mini-batch (global_token_loss): the gradient does not depend on how rows
are split into micro-batches or ranks."""
from types import MethodType

import pytest
import torch
from omegaconf import OmegaConf

from verl import DataProto
from verl.workers.actor.dp_actor import DataParallelPPOActor


def batch_fixture():
    mask = (torch.arange(5)[None, :] < torch.tensor([1, 2, 5, 3, 4, 1])[:, None]).long()
    ids = torch.arange(30).reshape(6, 5) % 11
    features = (ids.double() - 5) / 10
    advantages = torch.linspace(-1.3, 1.7, 30, dtype=torch.double).reshape(6, 5)
    old = features * .2 + torch.linspace(-.4, .4, 30).reshape(6, 5)
    return DataProto.from_dict({
        'responses': ids, 'response_mask': mask,
        'input_ids': torch.cat((torch.ones(6, 1, dtype=torch.long), ids), -1),
        'attention_mask': torch.cat((torch.ones(6, 1, dtype=torch.long), mask), -1),
        'position_ids': torch.arange(6).expand(6, 6),
        'old_log_probs': old, 'advantages': advantages,
    }, meta_info={'temperature': 1.})


def run_production_gradient(data, monkeypatch, world_size, global_tokens, partitions, dynamic):
    import verl.workers.actor.dp_actor as module
    actor = DataParallelPPOActor.__new__(DataParallelPPOActor)
    actor.config = OmegaConf.create({
        'ppo_mini_batch_size': len(data), 'ppo_micro_batch_size_per_gpu': len(partitions[0]),
        'ppo_epochs': 1, 'use_kl_loss': False, 'global_token_loss': True,
        'use_dynamic_bsz': dynamic, 'ppo_max_token_len_per_gpu': 32,
        'clip_ratio': .2, 'clip_ratio_low': .2, 'clip_ratio_high': .27,
        'entropy_coeff': 0, 'loss_agg_mode': 'token-mean', 'policy_loss': {'loss_mode': 'vanilla'},
    })
    actor.ulysses_sequence_parallel_size = 1
    actor.actor_module = torch.nn.Linear(1, 1, bias=False).double()
    with torch.no_grad():
        actor.actor_module.weight.fill_(.2)
    actor.actor_optimizer = torch.optim.SGD(actor.actor_module.parameters(), lr=.1)
    def forward(self, inputs, temperature, calculate_entropy=False):
        features = (inputs['responses'].double() - 5) / 10
        return None, self.actor_module(features.unsqueeze(-1)).squeeze(-1)
    gradients = []
    def optimizer_step(self):
        grad = self.actor_module.weight.grad.clone()
        gradients.append(grad)
        return grad.norm()
    actor._forward_micro_batch = MethodType(forward, actor)
    actor._optimizer_step = MethodType(optimizer_step, actor)
    with monkeypatch.context() as patch:
        # The production decorator queries CUDA allocator counters; the loss
        # and optimizer loop below still run unchanged on the CPU toy model.
        patch.setattr('verl.utils.profiler.performance._get_current_mem_info', lambda: (0, 0, 0, 0))
        patch.setattr(torch.distributed, 'is_initialized', lambda: world_size > 1)
        patch.setattr(torch.distributed, 'get_world_size', lambda: world_size)
        patch.setattr(torch.distributed, 'all_reduce', lambda tensor: tensor.fill_(global_tokens))
        if dynamic:
            patch.setattr(module, 'prepare_dynamic_batch', lambda batch, max_token_len: ([batch[rows] for rows in partitions], partitions))
        metrics = actor.update_policy(data)
    assert len(gradients) == 1
    assert metrics['actor/global_response_tokens_per_update'] == [global_tokens]
    return gradients[0]


@pytest.mark.parametrize('dynamic', [False, True])
@pytest.mark.parametrize('world_size,micro_size', [(1, 1), (1, 2), (1, 3), (1, 6), (2, 1), (2, 3)])
def test_update_gradient_independent_of_microbatch_and_rank_partition(monkeypatch, dynamic, world_size, micro_size):
    data = batch_fixture()
    weight = torch.tensor(.2, dtype=torch.double, requires_grad=True)
    logits = (data.batch['responses'].double() - 5) / 10 * weight
    ratio = (logits - data.batch['old_log_probs']).exp()
    advantage = data.batch['advantages']
    loss = torch.maximum(-advantage * ratio, -advantage * ratio.clamp(.8, 1.27))
    tokens = data.batch['response_mask'].sum().item()
    ((loss * data.batch['response_mask']).sum() / tokens).backward()
    reference = weight.grad
    # The rank split deliberately assigns unequal token counts to the ranks.
    rank_rows = [list(range(6))] if world_size == 1 else [[0, 1, 5], [2, 3, 4]]
    gradients = []
    for rows in rank_rows:
        local = data[rows]
        partitions = [list(range(i, min(i + micro_size, len(rows)))) for i in range(0, len(rows), micro_size)]
        if dynamic and len(partitions) > 1:
            partitions.reverse()
        gradients.append(run_production_gradient(local, monkeypatch, world_size, tokens, partitions, dynamic))
    averaged = torch.stack(gradients).mean()
    torch.testing.assert_close(averaged, reference, rtol=1e-6, atol=1e-8)


def test_uneven_dynamic_partition(monkeypatch):
    data = batch_fixture()
    tokens = data.batch['response_mask'].sum().item()
    full = run_production_gradient(data, monkeypatch, 1, tokens, [list(range(6))], True)
    split = run_production_gradient(data, monkeypatch, 1, tokens, [[5], [2, 4, 0], [1, 3]], True)
    torch.testing.assert_close(full, split, rtol=1e-6, atol=1e-8)
