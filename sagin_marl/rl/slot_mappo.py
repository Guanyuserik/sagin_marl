"""Feed-forward MAPPO update for the K=1 hybrid SAGIN action interface.

The native collector supplies conditional local observations for motion, satellite
selection and bandwidth. Their joint density belongs to one UAV. Other UAVs are
separate PPO samples. A single pre-motion centralized value supplies slot GAE.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F

from .structured_mappo import _index_dataclass


def slot_gae(rewards, values, next_values, done, *, gamma, gae_lambda):
    """[T, E] GAE; done includes the finite episode horizon, not rollout cuts."""
    if not (rewards.shape == values.shape == next_values.shape == done.shape) or rewards.ndim != 2:
        raise ValueError("slot GAE expects equal [time, env] tensors")
    alive = (~done.bool()).to(values.dtype)
    delta = rewards + float(gamma) * alive * next_values - values
    advantage = torch.zeros_like(values)
    tail = torch.zeros_like(values[0])
    for t in range(values.shape[0] - 1, -1, -1):
        tail = delta[t] + float(gamma) * float(gae_lambda) * alive[t] * tail
        advantage[t] = tail
    return advantage + values, advantage


def agent_ppo_loss(new_logprob, old_logprob, advantages, clip_ratio):
    if new_logprob.ndim != 2 or old_logprob.shape != new_logprob.shape:
        raise ValueError("MAPPO log probabilities must have shape [slot, UAV]")
    if advantages.shape != new_logprob.shape[:1]:
        raise ValueError("one common team advantage is required per slot")
    log_ratio = new_logprob - old_logprob.detach()
    # Fail on corrupted likelihoods instead of silently bounding PPO ratios.
    if not bool(torch.isfinite(log_ratio).all()):
        raise RuntimeError("non-finite MAPPO log ratio")
    ratio = log_ratio.exp()
    advantage = advantages.detach()[:, None]
    loss = -torch.minimum(ratio * advantage, ratio.clamp(1 - clip_ratio, 1 + clip_ratio) * advantage).mean()
    with torch.no_grad():
        metrics = {
            "approx_kl": ((ratio - 1) - log_ratio).mean(),
            "clip_frac": (torch.abs(ratio - 1) > clip_ratio).float().mean(),
            "ratio_mean": ratio.mean(),
        }
    return loss, metrics


def hybrid_logprobs(actor, batches, indices, *, entropy=True):
    """Raw conditional densities, never averaged over BW simplex dimensions."""
    agents = batches[0].num_agents
    local_indices = (indices[:, None] * agents + torch.arange(agents, device=indices.device)).reshape(-1)
    parts, entropies, collector_parts = [], [], []
    for stage, policy in enumerate((actor.accel_policy, actor.sat_subset_policy, actor.bw_policy)):
        batch = batches[stage]
        local = _index_dataclass(batch.local_batch, local_indices)
        action = batch.actions.index_select(0, indices)
        if stage == 0:
            if batch.latent_actions is None:
                raise RuntimeError("MAPPO motion requires recorded pre-squash Gaussian samples")
            out = policy.evaluate_actions(local, action.reshape(-1, action.shape[-1]),
                latent_action=batch.latent_actions.index_select(0, indices).reshape(-1, 2), compute_entropy=entropy)
        elif stage == 1:
            out = policy.evaluate_actions(local, action.reshape(-1), compute_entropy=entropy)
        else:
            out = policy.evaluate_actions(local, action.reshape(-1, action.shape[-1]), compute_entropy=entropy)
        parts.append((out.logprob_raw if stage == 2 else out.logprob).reshape(-1, agents))
        entropies.append((out.entropy_raw if stage == 2 else out.entropy).reshape(-1, agents))
        collector_parts.append(out.logprob.reshape(-1, agents))
    return torch.stack(parts, dim=-1), torch.stack(entropies, dim=-1), collector_parts


@dataclass
class SlotMAPPOBatch:
    stages: dict[int, Any]
    old_logprobs: torch.Tensor  # [slot, UAV, hybrid component]
    old_values: torch.Tensor
    returns: torch.Tensor
    advantages: torch.Tensor
    parity_errors: dict[str, float]


@torch.no_grad()
def prepare_slot_batch(actor, critic, views, *, num_envs, gamma, gae_lambda, chunk_size=256):
    stages = views.training_view.stage_batches
    motion = stages[0]
    n = motion.num_samples
    if n == 0 or n % num_envs:
        raise ValueError("incomplete time-major native rollout")
    expected_envs = np.tile(np.arange(num_envs), n // num_envs)
    for stage in range(3):
        b = stages[stage]
        if b.num_samples != n or not np.array_equal(b.env_indices, expected_envs):
            raise ValueError("slot MAPPO requires aligned K=1 time-major stage samples")
        if not np.array_equal(b.transition_indices, motion.transition_indices + stage):
            raise ValueError("hybrid stage transitions do not belong to matching slots")
    device = motion.values.device
    old_parts = []
    parity = {name: 0.0 for name in ("accel", "sat", "bw", "bw_raw")}
    for start in range(0, n, chunk_size):
        idx = torch.arange(start, min(start + chunk_size, n), device=device)
        parts, _, collector = hybrid_logprobs(actor, stages, idx, entropy=False)
        old_parts.append(parts)
        for stage, name in enumerate(("accel", "sat", "bw")):
            stored = stages[stage].old_logprobs.index_select(0, idx)
            error = (collector[stage].sum(-1) - stored).abs()
            parity[name] = max(parity[name], float(error.max()))
            # Check elementwise history where available; sums alone can cancel.
            stored_agent = getattr(stages[stage], "old_logprobs_per_agent", None)
            if stored_agent is not None:
                parity[name] = max(parity[name], float((collector[stage] - stored_agent.index_select(0, idx)).abs().max()))
        stored_raw = getattr(stages[2], "bw_logprob_raw_per_agent", None)
        if stored_raw is not None:
            parity["bw_raw"] = max(parity["bw_raw"], float((parts[..., 2] - stored_raw.index_select(0, idx)).abs().max()))
    if any(not np.isfinite(v) or v > 1e-3 for v in parity.values()):
        raise RuntimeError(f"native/replayed old policy likelihood mismatch: {parity}")
    old_values = motion.values.detach()
    old_parts_t = torch.cat(old_parts)
    if not bool(torch.isfinite(old_parts_t).all()):
        raise RuntimeError("non-finite old hybrid log probability")
    # Native rollout cuts bootstrap from the actual successor world. Terminal
    # episodes, including the task's configured finite horizon, mask this value.
    next_world = views.return_view.stage_batches[2].next_world_batch
    if next_world is None:
        raise RuntimeError("slot MAPPO requires successor pre-motion world states")
    next_values = []
    for start in range(0, n, chunk_size):
        idx = torch.arange(start, min(start + chunk_size, n), device=device)
        next_values.append(critic.value_accel(_index_dataclass(next_world, idx)))
    shape = (n // num_envs, num_envs)
    returns, advantage = slot_gae(stages[2].rewards.reshape(shape), old_values.reshape(shape),
        torch.cat(next_values).reshape(shape), (stages[2].terminated | stages[2].truncated).reshape(shape),
        gamma=gamma, gae_lambda=gae_lambda)
    advantage = advantage.reshape(-1)
    advantage = (advantage - advantage.mean()) / advantage.std(unbiased=False).clamp_min(1e-8)
    return SlotMAPPOBatch(stages, old_parts_t.detach(), old_values, returns.reshape(-1).detach(),
        advantage.detach(), parity)


def update_slot_mappo(actor, critic, batch, actor_optimizer, critic_optimizer, *,
        epochs=5, minibatches=4, microbatch_size=256, clip_ratio=0.2,
        value_coef=0.5, max_grad_norm=0.5, entropy_coefs=(0.001, 0.001, 0.0)):
    n = batch.returns.numel()
    device = batch.returns.device
    # PopArt preserves unnormalized predictions when the return scale changes.
    critic.update_popart_stats(0, batch.returns)
    old_norm = critic.popart_normalize(0, batch.old_values).detach()
    target = critic.popart_normalize(0, batch.returns).detach()
    actor_params = [p for p in actor.parameters() if p.requires_grad]
    critic_params = [p for p in critic.parameters() if p.requires_grad]
    totals = {key: 0.0 for key in ("policy_loss", "value_loss", "approx_kl", "clip_frac", "ratio_mean")}
    updates = 0
    grad_norms = {name: 0.0 for name in ("actor", "critic")}
    for _ in range(epochs):
        for outer in torch.randperm(n, device=device).tensor_split(min(minibatches, n)):
            actor_optimizer.zero_grad(set_to_none=True)
            critic_optimizer.zero_grad(set_to_none=True)
            for idx in outer.split(microbatch_size):
                weight = idx.numel() / outer.numel()
                parts, entropy, _ = hybrid_logprobs(actor, batch.stages, idx)
                policy_loss, info = agent_ppo_loss(parts.sum(-1), batch.old_logprobs.index_select(0, idx).sum(-1),
                    batch.advantages.index_select(0, idx), clip_ratio)
                entropy_weight = entropy.new_tensor(entropy_coefs)
                actor_loss = policy_loss - (entropy * entropy_weight).sum(-1).mean()
                world = _index_dataclass(batch.stages[0].world_batch, idx)
                value = critic.popart_normalize(0, critic.value_accel(world))
                old = old_norm.index_select(0, idx)
                value_clipped = old + (value - old).clamp(-clip_ratio, clip_ratio)
                truth = target.index_select(0, idx)
                loss_unclipped = F.huber_loss(value, truth, reduction="none", delta=10.0)
                loss_clipped = F.huber_loss(value_clipped, truth, reduction="none", delta=10.0)
                value_loss = torch.maximum(loss_unclipped, loss_clipped).mean()
                (actor_loss * weight).backward()
                (value_coef * value_loss * weight).backward()
                for key, value_ in {"policy_loss": policy_loss, "value_loss": value_loss, **info}.items():
                    totals[key] += weight * float(value_.detach())
            grad_norms["actor"] += float(torch.nn.utils.clip_grad_norm_(actor_params, max_grad_norm, error_if_nonfinite=True))
            grad_norms["critic"] += float(torch.nn.utils.clip_grad_norm_(critic_params, max_grad_norm, error_if_nonfinite=True))
            actor_optimizer.step()
            critic_optimizer.step()
            updates += 1
    with torch.no_grad():
        fitted = torch.cat([critic.value_accel(_index_dataclass(batch.stages[0].world_batch,
            torch.arange(start, min(start + microbatch_size, n), device=device)))
            for start in range(0, n, microbatch_size)])
        variance = batch.returns.var(unbiased=False).clamp_min(1e-8)
        ev_before = 1 - (batch.returns - batch.old_values).var(unbiased=False) / variance
        ev_after = 1 - (batch.returns - fitted).var(unbiased=False) / variance
    return {**{k: v / updates for k, v in totals.items()}, "optimizer_steps": updates,
        **{f"grad_norm_{k}": v / updates for k, v in grad_norms.items()},
        "value_ev_before": float(ev_before), "value_ev_after": float(ev_after),
        "return_mean": float(batch.returns.mean()), "return_std": float(batch.returns.std(unbiased=False)),
        **{f"old_logprob_parity_{k}": v for k, v in batch.parity_errors.items()}}
