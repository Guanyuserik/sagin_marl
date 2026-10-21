from __future__ import annotations

import copy
import csv
from dataclasses import fields, is_dataclass
import math
import os
from typing import Any, Dict, List, Sequence, Tuple

import numpy as np
import torch

from sagin_marl.env import native_cuda
from sagin_marl.env.structured_driver import StructuredBatchStepResult, StructuredControlDriver
from sagin_marl.env.structured_gpu_rollout_runtime import StructuredGpuNativeRolloutProgram
from sagin_marl.rl.baselines import (
    cluster_center_queue_aware_policy,
    demand_priority_policy,
    feasible_random_bw_policy,
    feasible_random_sat_policy,
    feasible_uniform_bw_policy,
    feasible_uniform_sat_policy,
    link_priority_policy,
    lyapunov_queue_aware_policy_step,
    observable_cluster_queue_aware_policy,
    queue_aware_bw_policy,
    queue_aware_policy,
    random_feasible_policy,
    static_uniform_policy,
    topology_dpp_policy,
    topology_dpp_policy_step,
    uniform_bw_policy,
    zero_accel_policy,
)
from sagin_marl.rl.structured_critic import ZeroStructuredCritic
from sagin_marl.rl.structured_mappo import (
    StructuredMAPPO,
    _collate_dataclass,
    _normalize_exec_source,
    _sat_action_select_k_from_cfg,
    _sat_mask_to_ids,
    validate_satellite_control_consistency,
)
from sagin_marl.rl.structured_parallel_eval import (
    _refresh_stage_obs_cache,
    batched_policy_accel_actions,
    batched_policy_accel_actions_from_world_batch,
    batched_policy_bw_outputs,
    batched_policy_bw_outputs_from_batch,
    batched_policy_sat_subset_indices,
    batched_policy_sat_subset_indices_from_world_batch,
    current_obs_many,
    last_reward_parts_many,
    looks_like_driver_group,
    reset_at,
    reset_many,
)
from sagin_marl.rl.structured_train import (
    as_structured_drivers,
    close_structured_env_group,
    make_structured_env_group,
)

_STAGE_FEASIBLE_BASELINES = {
    "static",
    "static_uniform",
    "uniform",
    "random",
    "random_feasible",
    "link_priority",
    "demand_priority",
}
from sagin_marl.rl.structured_buffer import (
    StructuredNativeRolloutTrainingBatchView,
    _accel_local_from_flat_history_ring,
    _build_rollout_views_from_native_training_ring,
    _bw_local_from_flat_history_ring,
    _sat_local_from_flat_history_ring,
    _where_world_state,
    _world_state_from_flat_history_ring,
)


def _as_driver_list(env_group) -> list[StructuredControlDriver]:
    drivers = as_structured_drivers(env_group)
    if looks_like_driver_group(drivers):
        return [drivers[int(index)] for index in range(len(drivers))]
    return list(drivers)


def _bind_native_eval_tensor_device(drivers: Any, cfg) -> None:
    setter = getattr(drivers, "set_tensor_device", None)
    if not callable(setter):
        return
    backend = str(getattr(cfg, "structured_env_tensor_backend", "cuda") or "cuda").strip().lower()
    if backend in {"cuda", "auto"} and torch.cuda.is_available():
        setter(torch.device("cuda"))
    elif backend in {"cpu", "torch_cpu"}:
        setter(torch.device("cpu"))


def _checkpoint_eval_best_summary_from_state(state: Dict[str, float]) -> Dict[str, float] | None:
    pre_backlog = float(state.get("best_pre_backlog_steps_eval", float("inf")))
    if not np.isfinite(pre_backlog):
        return None
    return {
        "reward_sum": float(state.get("best_model_reward_sum", -float("inf"))),
        "processed_ratio_eval": float(state.get("best_processed_ratio_eval", -float("inf"))),
        "drop_ratio_eval": float(state.get("best_drop_ratio_eval", float("inf"))),
        "pre_backlog_steps_eval": pre_backlog,
        "collision_episode_fraction": float(state.get("best_collision_episode_fraction", float("inf"))),
        "sat_overlap_eval": float(state.get("best_sat_overlap_eval", float("inf"))),
    }


def _checkpoint_eval_model_better(
    summary: Dict[str, float],
    best_summary: Dict[str, float] | None,
    cfg,
) -> bool:
    if best_summary is None:
        return True
    model_collision_threshold = max(
        float(getattr(cfg, "checkpoint_eval_model_collision_threshold", 0.0) or 0.0),
        0.0,
    )
    reward_tie_rel_tol = max(
        float(getattr(cfg, "checkpoint_eval_reward_tie_rel_tol", 0.05) or 0.0),
        0.0,
    )
    use_sat_overlap = bool(getattr(cfg, "checkpoint_eval_use_sat_overlap", False))
    current_collision = float(summary["collision_episode_fraction"])
    best_collision = float(best_summary["collision_episode_fraction"])
    current_collision_pass = current_collision <= model_collision_threshold
    best_collision_pass = best_collision <= model_collision_threshold
    if current_collision_pass != best_collision_pass:
        return current_collision_pass
    if not current_collision_pass:
        if current_collision < best_collision - 1e-12:
            return True
        if current_collision > best_collision + 1e-12:
            return False
    current_pre = float(summary["pre_backlog_steps_eval"])
    best_pre = float(best_summary["pre_backlog_steps_eval"])
    tie_band = reward_tie_rel_tol * max(abs(best_pre), 1.0)
    if current_pre < best_pre - tie_band:
        return True
    if current_pre > best_pre + tie_band:
        return False
    current_processed = float(summary["processed_ratio_eval"])
    best_processed = float(best_summary["processed_ratio_eval"])
    current_reward = float(summary["reward_sum"])
    best_reward = float(best_summary["reward_sum"])
    if use_sat_overlap:
        if current_processed > best_processed + 1e-12:
            return True
        if current_processed < best_processed - 1e-12:
            return False
        current_overlap = float(summary.get("sat_overlap_eval", 0.0))
        best_overlap = float(best_summary.get("sat_overlap_eval", float("inf")))
        if current_overlap < best_overlap - 1e-12:
            return True
        if current_overlap > best_overlap + 1e-12:
            return False
        return current_reward > best_reward + 1e-12
    if current_reward > best_reward + 1e-12:
        return True
    if current_reward < best_reward - 1e-12:
        return False
    return current_processed > best_processed + 1e-12


def structured_checkpoint_eval_fieldnames() -> List[str]:
    return [
        "update",
        "checkpoint_suffix",
        "episodes",
        "reward_sum",
        "bw_weighted_workload_delta_sum",
        "bw_weighted_workload_level_sum",
        "processed_ratio_eval",
        "drop_ratio_eval",
        "pre_backlog_steps_eval",
        "D_sys_report",
        "x_acc_mean",
        "x_rel_mean",
        "g_pre_mean",
        "d_pre_mean",
        "sat_overlap_eval",
        "collision_episode_fraction",
        "fixed_reward_sum",
        "fixed_bw_weighted_workload_delta_sum",
        "fixed_bw_weighted_workload_level_sum",
        "fixed_processed_ratio_eval",
        "fixed_drop_ratio_eval",
        "fixed_pre_backlog_steps_eval",
        "fixed_D_sys_report",
        "fixed_x_acc_mean",
        "fixed_x_rel_mean",
        "fixed_g_pre_mean",
        "fixed_d_pre_mean",
        "fixed_sat_overlap_eval",
        "fixed_collision_episode_fraction",
        "processed_improved",
        "drop_improved",
        "pre_backlog_improved",
        "model_improved",
        "quality_worsened",
        "quality_worse_streak",
        "reward_improved",
        "reward_plateau_streak",
        "collision_gate_passed",
        "quality_early_stop_triggered",
        "reward_early_stop_triggered",
        "early_stop_triggered",
    ]


def append_structured_checkpoint_eval_row(path: str, row: Dict[str, object]) -> None:
    fieldnames = structured_checkpoint_eval_fieldnames()
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    write_header = not os.path.exists(path)
    with open(path, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if write_header:
            writer.writeheader()
        writer.writerow({name: row.get(name, "") for name in fieldnames})


def update_structured_checkpoint_eval_state(
    state: Dict[str, float],
    summary: Dict[str, float],
    cfg,
) -> Dict[str, object]:
    reward_sum = float(summary["reward_sum"])
    processed = float(summary["processed_ratio_eval"])
    drop_ratio = float(summary["drop_ratio_eval"])
    pre_backlog = float(summary["pre_backlog_steps_eval"])
    collision_frac = float(summary["collision_episode_fraction"])
    rel_tol = max(float(getattr(cfg, "checkpoint_eval_front_queue_rel_improve_tol", 0.05) or 0.0), 0.0)
    worsen_delta = max(float(getattr(cfg, "checkpoint_eval_sat_drop_worsen_delta", 0.0) or 0.0), 0.0)
    patience = max(int(getattr(cfg, "checkpoint_eval_worsen_patience", 2) or 0), 1)
    sat_drop_early_stop_enabled = bool(getattr(cfg, "checkpoint_eval_sat_drop_early_stop_enabled", True))
    reward_early_stop_enabled = bool(getattr(cfg, "checkpoint_eval_reward_early_stop_enabled", False))
    reward_patience = max(int(getattr(cfg, "checkpoint_eval_reward_patience", 5) or 0), 1)
    reward_min_delta_rel = max(float(getattr(cfg, "checkpoint_eval_reward_min_delta_rel", 0.0) or 0.0), 0.0)
    reward_collision_threshold = max(
        float(getattr(cfg, "checkpoint_eval_reward_collision_threshold", 1.0) or 0.0),
        0.0,
    )
    best_processed_prev = float(state.get("best_processed_ratio_eval", -float("inf")))
    best_drop_prev = float(state.get("best_drop_ratio_eval", float("inf")))
    best_pre_backlog_prev = float(state.get("best_pre_backlog_steps_eval", float("inf")))
    prev_processed = state.get("prev_processed_ratio_eval", None)
    prev_drop = state.get("prev_drop_ratio_eval", None)
    prev_pre_backlog = state.get("prev_pre_backlog_steps_eval", None)
    best_reward_prev = float(state.get("best_reward_sum", -float("inf")))
    processed_improved = processed > best_processed_prev + worsen_delta
    drop_improved = drop_ratio < best_drop_prev - worsen_delta
    pre_backlog_improved = (not np.isfinite(best_pre_backlog_prev)) or pre_backlog < best_pre_backlog_prev * (1.0 - rel_tol)
    best_summary_prev = _checkpoint_eval_best_summary_from_state(state)
    model_improved = _checkpoint_eval_model_better(summary, best_summary_prev, cfg)
    quality_worsened = (
        prev_processed is not None
        and prev_drop is not None
        and prev_pre_backlog is not None
        and processed < float(prev_processed) - worsen_delta
        and drop_ratio > float(prev_drop) + worsen_delta
        and pre_backlog > float(prev_pre_backlog) * (1.0 + rel_tol)
    )
    reward_margin = reward_min_delta_rel * max(abs(best_reward_prev), 1.0) if np.isfinite(best_reward_prev) else 0.0
    reward_improved = (not np.isfinite(best_reward_prev)) or reward_sum > (best_reward_prev + reward_margin)
    if model_improved:
        state["best_processed_ratio_eval"] = processed
        state["best_drop_ratio_eval"] = drop_ratio
        state["best_pre_backlog_steps_eval"] = pre_backlog
        state["best_model_reward_sum"] = reward_sum
        state["best_collision_episode_fraction"] = collision_frac
        state["best_sat_overlap_eval"] = float(summary.get("sat_overlap_eval", 0.0))
    if (not np.isfinite(best_reward_prev)) or reward_sum > best_reward_prev:
        state["best_reward_sum"] = reward_sum
    prev_streak = int(state.get("quality_worse_streak", 0))
    quality_worse_streak = prev_streak + 1 if quality_worsened and not model_improved else 0
    state["quality_worse_streak"] = float(quality_worse_streak)
    state["prev_processed_ratio_eval"] = processed
    state["prev_drop_ratio_eval"] = drop_ratio
    state["prev_pre_backlog_steps_eval"] = pre_backlog
    reward_prev_streak = int(state.get("reward_plateau_streak", 0))
    reward_plateau_streak = 0 if reward_improved else reward_prev_streak + 1
    state["reward_plateau_streak"] = float(reward_plateau_streak)
    collision_gate_passed = collision_frac <= reward_collision_threshold
    quality_should_stop = sat_drop_early_stop_enabled and quality_worse_streak >= patience
    reward_should_stop = reward_early_stop_enabled and reward_plateau_streak >= reward_patience and collision_gate_passed
    should_stop = quality_should_stop or reward_should_stop
    return {
        "processed_improved": float(processed_improved),
        "drop_improved": float(drop_improved),
        "pre_backlog_improved": float(pre_backlog_improved),
        "model_improved": float(model_improved),
        "quality_worsened": float(quality_worsened),
        "quality_worse_streak": float(quality_worse_streak),
        "reward_improved": float(reward_improved),
        "reward_plateau_streak": float(reward_plateau_streak),
        "collision_gate_passed": float(collision_gate_passed),
        "quality_early_stop_triggered": float(quality_should_stop),
        "reward_early_stop_triggered": float(reward_should_stop),
        "early_stop_triggered": float(should_stop),
    }


def _baseline_actions(baseline: str, obs_list, cfg, env):
    baseline = str(baseline).strip().lower()
    num_agents = len(env.agents)
    rng = getattr(env, "rng", None)
    if baseline == "zero":
        return zero_accel_policy(num_agents), None, None
    if baseline in {"static_uniform", "static", "uniform"}:
        return static_uniform_policy(obs_list, cfg, rng=rng)
    if baseline in {"random", "random_feasible"}:
        return random_feasible_policy(obs_list, cfg, rng=rng)
    if baseline == "link_priority":
        return link_priority_policy(obs_list, cfg)
    if baseline == "demand_priority":
        return demand_priority_policy(obs_list, cfg)
    if baseline == "uniform_bw":
        return zero_accel_policy(num_agents), feasible_uniform_bw_policy(obs_list, cfg), None
    if baseline == "random_bw":
        return zero_accel_policy(num_agents), feasible_random_bw_policy(obs_list, cfg, rng=rng), None
    if baseline == "uniform_sat":
        return zero_accel_policy(num_agents), None, feasible_uniform_sat_policy(obs_list, cfg, rng=rng)
    if baseline == "random_sat":
        return zero_accel_policy(num_agents), None, feasible_random_sat_policy(obs_list, cfg, rng=rng)
    if baseline == "queue_aware_bw":
        return zero_accel_policy(num_agents), queue_aware_bw_policy(obs_list, cfg), None
    if baseline == "queue_aware":
        return queue_aware_policy(obs_list, cfg)
    if baseline == "cluster_center_queue_aware":
        centers = getattr(env, "gu_cluster_centers", None)
        counts = getattr(env, "gu_cluster_counts", None)
        return cluster_center_queue_aware_policy(obs_list, cfg, centers, counts)
    if baseline == "observable_cluster_queue_aware":
        return observable_cluster_queue_aware_policy(obs_list, cfg)
    if baseline == "topology_dpp":
        return topology_dpp_policy(obs_list, cfg)
    if baseline in {"dpp_resource_hybrid", "topology_dpp_resource"}:
        centers = getattr(env, "gu_cluster_centers", None)
        counts = getattr(env, "gu_cluster_counts", None)
        accel_actions, _queue_bw, _queue_sat = cluster_center_queue_aware_policy(obs_list, cfg, centers, counts)
        _dpp_accel, bw_logits, sat_logits = topology_dpp_policy(obs_list, cfg)
        return accel_actions, bw_logits, sat_logits
    raise ValueError(f"Unsupported structured baseline policy: {baseline}")


def _run_structured_actor_step(actor, driver: StructuredControlDriver, device: torch.device, deterministic: bool):
    z_accel = driver.begin_step()
    accel_states = driver.build_local_accel_states(z_accel)
    accel_batch = _collate_dataclass(accel_states, device)
    with torch.inference_mode():
        accel_out = actor.act_accel(accel_batch, deterministic=deterministic)
    z_sat = driver.run_accel_stage(accel_out.action.detach().cpu().numpy())
    sat_states = driver.build_local_sat_states(z_sat)
    sat_batch = _collate_dataclass(sat_states, device)
    with torch.inference_mode():
        sat_out = actor.act_sat(sat_batch, deterministic=deterministic)
    sat_action = driver.decode_sat_subset_actions(sat_states, sat_out.subset_index.detach().cpu().tolist())
    z_bw = driver.run_sat_stage(sat_action)
    bw_states = driver.build_bw_valid_context(z_bw)
    bw_batch = _collate_dataclass(bw_states, device)
    with torch.inference_mode():
        bw_out = actor.act_bw(bw_batch, deterministic=deterministic)
    return driver.execute_stage_bw_and_step(bw_out.action.detach().cpu().numpy())


def _new_episode_accumulator(cfg=None) -> Dict[str, float]:
    num_gu = float(max(int(getattr(cfg, "num_gu", 1) or 1), 1)) if cfg is not None else 1.0
    num_uav = float(max(int(getattr(cfg, "num_uav", 1) or 1), 1)) if cfg is not None else 1.0
    num_sat = float(max(int(getattr(cfg, "num_sat", 1) or 1), 1)) if cfg is not None else 1.0
    target_steps = float(max(int(getattr(cfg, "T_steps", 0) or 0), 0)) if cfg is not None else 0.0
    return {
        "reward_sum": 0.0,
        "bw_weighted_workload_delta_sum": 0.0,
        "bw_weighted_workload_level_sum": 0.0,
        "processed_ratio_sum": 0.0,
        "drop_ratio_sum": 0.0,
        "pre_backlog_sum": 0.0,
        "d_sys_sum": 0.0,
        "x_acc_sum": 0.0,
        "x_rel_sum": 0.0,
        "g_pre_sum": 0.0,
        "d_pre_sum": 0.0,
        "sat_overlap_sum": 0.0,
        "collision_any": 0.0,
        "gu_queue_sum_total": 0.0,
        "uav_queue_sum_total": 0.0,
        "sat_queue_sum_total": 0.0,
        "queue_total_sum_total": 0.0,
        "terminal_queue_sum": 0.0,
        "arrival_sum_total": 0.0,
        "outflow_sum_total": 0.0,
        "backhaul_sum_total": 0.0,
        "sat_processed_sum_total": 0.0,
        "drop_sum_total": 0.0,
        "drop_sum_active_total": 0.0,
        "gu_drop_sum_total": 0.0,
        "uav_drop_sum_total": 0.0,
        "sat_drop_sum_total": 0.0,
        "num_gu": num_gu,
        "num_uav": num_uav,
        "num_sat": num_sat,
        "target_steps": target_steps,
        "steps": 0.0,
    }


def _metric_float(
    reward_parts: Dict[str, Any],
    runtime_trace: Dict[str, Any] | None,
    key: str,
    default: float = 0.0,
) -> float:
    value = reward_parts.get(key, None)
    if value is None and runtime_trace is not None:
        value = runtime_trace.get(key, None)
    if value is None:
        return float(default)
    try:
        return float(value)
    except (TypeError, ValueError):
        return float(default)


def _accumulate_step_metrics(
    acc: Dict[str, float],
    *,
    reward_value: float,
    bw_delta_value: float,
    bw_level_value: float,
    reward_parts: Dict[str, Any],
    runtime_trace: Dict[str, Any] | None = None,
) -> None:
    acc["reward_sum"] += float(reward_value)
    acc["bw_weighted_workload_delta_sum"] += float(bw_delta_value)
    acc["bw_weighted_workload_level_sum"] += float(bw_level_value)
    acc["steps"] += 1.0
    acc["processed_ratio_sum"] += _metric_float(reward_parts, runtime_trace, "processed_ratio_eval")
    acc["drop_ratio_sum"] += _metric_float(reward_parts, runtime_trace, "drop_ratio_eval")
    acc["pre_backlog_sum"] += _metric_float(reward_parts, runtime_trace, "pre_backlog_steps_eval")
    acc["d_sys_sum"] += _metric_float(reward_parts, runtime_trace, "D_sys_report")
    acc["x_acc_sum"] += _metric_float(reward_parts, runtime_trace, "x_acc")
    acc["x_rel_sum"] += _metric_float(reward_parts, runtime_trace, "x_rel")
    acc["g_pre_sum"] += _metric_float(reward_parts, runtime_trace, "g_pre")
    acc["d_pre_sum"] += _metric_float(reward_parts, runtime_trace, "d_pre")
    acc["sat_overlap_sum"] += _metric_float(reward_parts, runtime_trace, "sat_overlap_eval")
    acc["collision_any"] = max(
        acc["collision_any"],
        _metric_float(reward_parts, runtime_trace, "collision_event"),
    )

    gu_queue_sum = _metric_float(reward_parts, runtime_trace, "gu_queue_sum")
    uav_queue_sum = _metric_float(reward_parts, runtime_trace, "uav_queue_sum")
    sat_queue_sum = _metric_float(reward_parts, runtime_trace, "sat_queue_sum")
    queue_total_sum = _metric_float(
        reward_parts,
        runtime_trace,
        "queue_total_sum",
        gu_queue_sum + uav_queue_sum + sat_queue_sum,
    )
    acc["gu_queue_sum_total"] += gu_queue_sum
    acc["uav_queue_sum_total"] += uav_queue_sum
    acc["sat_queue_sum_total"] += sat_queue_sum
    acc["queue_total_sum_total"] += queue_total_sum
    acc["terminal_queue_sum"] = queue_total_sum
    acc["arrival_sum_total"] += _metric_float(reward_parts, runtime_trace, "arrival_sum")
    acc["outflow_sum_total"] += _metric_float(reward_parts, runtime_trace, "outflow_sum")
    acc["backhaul_sum_total"] += _metric_float(reward_parts, runtime_trace, "backhaul_sum")
    acc["sat_processed_sum_total"] += _metric_float(reward_parts, runtime_trace, "sat_processed_sum")
    acc["drop_sum_total"] += _metric_float(reward_parts, runtime_trace, "drop_sum")
    acc["drop_sum_active_total"] += _metric_float(reward_parts, runtime_trace, "drop_sum_active")
    acc["gu_drop_sum_total"] += _metric_float(reward_parts, runtime_trace, "gu_drop_sum")
    acc["uav_drop_sum_total"] += _metric_float(reward_parts, runtime_trace, "uav_drop_sum")
    acc["sat_drop_sum_total"] += _metric_float(reward_parts, runtime_trace, "sat_drop_sum")


def _episode_row_from_accumulator(episode_index: int, acc: Dict[str, float]) -> Dict[str, float]:
    denom = float(max(int(acc["steps"]), 1))
    arrival_total = float(acc.get("arrival_sum_total", 0.0))
    backhaul_total = float(acc.get("backhaul_sum_total", 0.0))
    target_steps = float(acc.get("target_steps", 0.0))
    return {
        "episode": float(episode_index),
        "step_count": float(acc["steps"]),
        "episode_length": float(acc["steps"]),
        "terminated_early": float(1.0 if target_steps > 0.0 and float(acc["steps"]) < target_steps else 0.0),
        "reward_sum": float(acc["reward_sum"]),
        "bw_weighted_workload_delta_sum": float(acc["bw_weighted_workload_delta_sum"]),
        "bw_weighted_workload_level_sum": float(acc["bw_weighted_workload_level_sum"]),
        "processed_ratio_total": float(acc["processed_ratio_sum"]),
        "drop_ratio_total": float(acc["drop_ratio_sum"]),
        "pre_backlog_total": float(acc["pre_backlog_sum"]),
        "D_sys_total": float(acc["d_sys_sum"]),
        "x_acc_total": float(acc["x_acc_sum"]),
        "x_rel_total": float(acc["x_rel_sum"]),
        "g_pre_total": float(acc["g_pre_sum"]),
        "d_pre_total": float(acc["d_pre_sum"]),
        "sat_overlap_total": float(acc["sat_overlap_sum"]),
        "processed_ratio_eval": float(acc["processed_ratio_sum"] / denom),
        "drop_ratio_eval": float(acc["drop_ratio_sum"] / denom),
        "pre_backlog_steps_eval": float(acc["pre_backlog_sum"] / denom),
        "D_sys_report": float(acc["d_sys_sum"] / denom),
        "x_acc_mean": float(acc["x_acc_sum"] / denom),
        "x_rel_mean": float(acc["x_rel_sum"] / denom),
        "g_pre_mean": float(acc["g_pre_sum"] / denom),
        "d_pre_mean": float(acc["d_pre_sum"] / denom),
        "sat_overlap_eval": float(acc["sat_overlap_sum"] / denom),
        "collision_episode_fraction": float(acc["collision_any"]),
        "gu_queue_mean": float(acc["gu_queue_sum_total"] / denom / max(acc.get("num_gu", 1.0), 1.0)),
        "uav_queue_mean": float(acc["uav_queue_sum_total"] / denom / max(acc.get("num_uav", 1.0), 1.0)),
        "sat_queue_mean": float(acc["sat_queue_sum_total"] / denom / max(acc.get("num_sat", 1.0), 1.0)),
        "queue_total_mean": float(acc["queue_total_sum_total"] / denom),
        "terminal_queue_sum": float(acc["terminal_queue_sum"]),
        "arrival_sum": arrival_total,
        "arrival_step_mean": float(arrival_total / denom),
        "outflow_sum": float(acc["outflow_sum_total"]),
        "outflow_step_mean": float(acc["outflow_sum_total"] / denom),
        "backhaul_sum": backhaul_total,
        "backhaul_step_mean": float(backhaul_total / denom),
        "sat_processed_sum": float(acc["sat_processed_sum_total"]),
        "sat_processed_step_mean": float(acc["sat_processed_sum_total"] / denom),
        "drop_sum": float(acc["drop_sum_total"]),
        "drop_sum_active": float(acc["drop_sum_active_total"]),
        "gu_drop_sum": float(acc["gu_drop_sum_total"]),
        "uav_drop_sum": float(acc["uav_drop_sum_total"]),
        "sat_drop_sum": float(acc["sat_drop_sum_total"]),
        "outflow_arrival_ratio": float(acc["outflow_sum_total"] / max(arrival_total, 1.0e-9)),
        "sat_incoming_arrival_ratio": float(backhaul_total / max(arrival_total, 1.0e-9)),
        "sat_processed_arrival_ratio": float(acc["sat_processed_sum_total"] / max(arrival_total, 1.0e-9)),
        "sat_processed_incoming_ratio": float(acc["sat_processed_sum_total"] / max(backhaul_total, 1.0e-9)),
        "drop_ratio": float(acc["drop_sum_total"] / max(arrival_total, 1.0e-9)),
        "active_drop_ratio": float(acc["drop_sum_active_total"] / max(arrival_total, 1.0e-9)),
        "gu_drop_ratio": float(acc["gu_drop_sum_total"] / max(arrival_total, 1.0e-9)),
        "uav_drop_ratio": float(acc["uav_drop_sum_total"] / max(arrival_total, 1.0e-9)),
        "sat_drop_ratio": float(acc["sat_drop_sum_total"] / max(arrival_total, 1.0e-9)),
    }


def _summary_from_rows(rows: List[Dict[str, float]], episodes: int) -> Dict[str, float]:
    denom = float(max(int(episodes), 1))
    summary: Dict[str, float] = {"episodes": denom}
    keys: set[str] = set()
    for row in rows:
        keys.update(str(key) for key in row.keys())
    for key in sorted(keys):
        if key == "episode":
            continue
        total = 0.0
        count = 0
        for row in rows:
            value = row.get(key, None)
            if value is None:
                continue
            try:
                total += float(value)
            except (TypeError, ValueError):
                continue
            count += 1
        if count > 0:
            summary[key] = total / float(max(count, 1))
    return summary


def _step_reward_parts(step_result, drivers, *, indices=None):
    if isinstance(step_result, StructuredBatchStepResult):
        tensors = step_result.reward_part_tensors or {}
        parts: Dict[str, Any] = {}
        for key, value in tensors.items():
            row = value.reshape(int(step_result.num_envs), -1)[0, 0].detach().cpu()
            parts[str(key)] = bool(row.item()) if value.dtype == torch.bool else float(row.item())
        if step_result.reward_mode_active is not None:
            parts["reward_mode_active"] = str(step_result.reward_mode_active)
        if parts:
            return parts
    reward_parts = dict(getattr(step_result, "reward_parts", {}) or {})
    if reward_parts:
        return reward_parts
    if indices is None:
        reward_rows = last_reward_parts_many(drivers)
        return dict(reward_rows[0] or {}) if reward_rows else {}
    reward_rows = last_reward_parts_many(drivers, indices=indices)
    return dict(reward_rows[0] or {}) if reward_rows else {}


def _step_team_reward(step_result) -> float:
    if isinstance(step_result, StructuredBatchStepResult):
        return float(step_result.team_rewards.reshape(-1)[0].detach().cpu().item())
    rewards = getattr(step_result, "rewards", {}) or {}
    if rewards:
        return float(next(iter(rewards.values())))
    return float(getattr(step_result, "team_reward", 0.0) or 0.0)


def _step_done(step_result) -> bool:
    if isinstance(step_result, StructuredBatchStepResult):
        terminated = bool(step_result.terminated.reshape(-1)[0].detach().cpu().item())
        truncated = bool(step_result.truncated.reshape(-1)[0].detach().cpu().item())
        return bool(terminated or truncated)
    terminations = getattr(step_result, "terminations", {}) or {}
    truncations = getattr(step_result, "truncations", {}) or {}
    if terminations or truncations:
        terminated = bool(next(iter(terminations.values()))) if terminations else False
        truncated = bool(next(iter(truncations.values()))) if truncations else False
        return bool(terminated or truncated)
    return bool(getattr(step_result, "terminated", False) or getattr(step_result, "truncated", False))


def _batch_step_scalar(step_result, attr_name: str, default: float = 0.0) -> float:
    if not isinstance(step_result, StructuredBatchStepResult):
        return float(getattr(step_result, attr_name, default) or default)
    tensor_map = {
        "bw_access_reward": step_result.bw_access_rewards,
        "bw_weighted_workload_delta_reward": step_result.bw_weighted_workload_delta_rewards,
        "bw_weighted_workload_level_reward": step_result.bw_weighted_workload_level_rewards,
        "bw_gu_queue_level_reward": step_result.bw_gu_queue_level_rewards,
        "bw_system_queue_level_reward": step_result.bw_system_queue_level_rewards,
        "bw_gu_service_queue_reward": step_result.bw_gu_service_queue_rewards,
    }
    value = tensor_map.get(str(attr_name))
    if value is None:
        return float(default)
    return float(value.reshape(-1)[0].detach().cpu().item())


def _batch_step_result_arrays(step_results: StructuredBatchStepResult) -> Dict[str, Any]:
    def _float_array(tensor: torch.Tensor | None) -> np.ndarray:
        if tensor is None:
            return np.zeros((int(step_results.num_envs),), dtype=np.float32)
        return tensor.detach().to(dtype=torch.float32).reshape(int(step_results.num_envs)).cpu().numpy()

    def _bool_array(tensor: torch.Tensor) -> np.ndarray:
        return tensor.detach().to(dtype=torch.bool).reshape(int(step_results.num_envs)).cpu().numpy().astype(bool, copy=False)

    reward_part_arrays: Dict[str, np.ndarray] = {}
    for key, tensor in dict(step_results.reward_part_tensors or {}).items():
        reward_part_arrays[str(key)] = _float_array(tensor)
    return {
        "team_reward": _float_array(step_results.team_rewards),
        "terminated": _bool_array(step_results.terminated),
        "truncated": _bool_array(step_results.truncated),
        "bw_access_reward": _float_array(step_results.bw_access_rewards),
        "bw_weighted_workload_delta_reward": _float_array(step_results.bw_weighted_workload_delta_rewards),
        "bw_weighted_workload_level_reward": _float_array(step_results.bw_weighted_workload_level_rewards),
        "bw_gu_queue_level_reward": _float_array(step_results.bw_gu_queue_level_rewards),
        "bw_system_queue_level_reward": _float_array(step_results.bw_system_queue_level_rewards),
        "bw_gu_service_queue_reward": _float_array(step_results.bw_gu_service_queue_rewards),
        "reward_parts": reward_part_arrays,
        "reward_mode_active": step_results.reward_mode_active,
    }


def _batch_reward_parts_at(batch_arrays: Dict[str, Any], slot: int) -> Dict[str, float | str]:
    reward_parts = {
        key: float(values[int(slot)])
        for key, values in dict(batch_arrays.get("reward_parts", {}) or {}).items()
    }
    reward_mode = batch_arrays.get("reward_mode_active")
    if reward_mode is not None:
        reward_parts["reward_mode_active"] = str(reward_mode)
    return reward_parts


def _run_structured_baseline_step(
    baseline: str,
    driver: StructuredControlDriver,
    cfg,
    obs_list,
    baseline_state: Dict[str, np.ndarray] | None = None,
):
    step_result, baseline_state, _ = _run_structured_baseline_step_with_actions(
        baseline,
        driver,
        cfg,
        obs_list,
        baseline_state=baseline_state,
    )
    return step_result, baseline_state


def _candidate_index_matrix_from_groups(cfg, candidate_groups: List[List[int]] | None) -> np.ndarray:
    out = np.full((int(cfg.num_uav), int(cfg.users_obs_max)), -1, dtype=np.int64)
    if not candidate_groups:
        return out
    for uav_index in range(min(int(cfg.num_uav), len(candidate_groups))):
        candidate_arr = np.asarray(list(candidate_groups[uav_index])[: int(cfg.users_obs_max)], dtype=np.int64)
        if candidate_arr.size > 0:
            out[uav_index, : candidate_arr.size] = candidate_arr
    return out


def _bw_semantic_action_from_candidates(cfg, candidate_groups: List[List[int]] | None, bw_action: np.ndarray) -> np.ndarray:
    semantic = np.zeros((int(cfg.num_uav), int(cfg.num_gu)), dtype=np.float32)
    bw_arr = np.asarray(bw_action, dtype=np.float32)
    if bw_arr.shape == semantic.shape:
        return bw_arr.astype(np.float32, copy=True)
    if not candidate_groups:
        return semantic
    for uav_index in range(min(int(cfg.num_uav), len(candidate_groups), int(bw_arr.shape[0]))):
        candidates = list(candidate_groups[uav_index])[: int(cfg.users_obs_max)]
        max_slots = min(len(candidates), int(bw_arr.shape[1]))
        for slot_index in range(max_slots):
            gu_index = int(candidates[slot_index])
            if 0 <= gu_index < int(cfg.num_gu):
                semantic[uav_index, gu_index] = float(bw_arr[uav_index, slot_index])
    return semantic


def _to_numpy_tape_value(value, *, dtype):
    if torch.is_tensor(value):
        return value.detach().cpu().numpy().astype(dtype, copy=True)
    return np.asarray(value, dtype=dtype).copy()


def _batch_core_from_driver_group(group):
    core = getattr(group, "batch_core", None)
    if core is not None:
        return core
    core = getattr(group, "_batch_core", None)
    if core is not None:
        return core
    batch_env = getattr(group, "batch_env", None)
    if batch_env is None:
        batch_env = getattr(group, "_batch_env", None)
    core = None if batch_env is None else getattr(batch_env, "_core", None)
    if core is not None:
        return core
    return getattr(group, "_core", None)


def _native_replay_runtime_and_tensor_state(group):
    core = _batch_core_from_driver_group(group)
    if core is None:
        return None, None
    runtime = getattr(core, "_native_sub_batch_runtime", None)
    tensor_state = getattr(core, "_native_sub_batch_tensor_state", None)
    if runtime is None or tensor_state is None:
        return None, None
    main = getattr(runtime, "main", None)
    if main is None or int(getattr(main, "num_envs", 0) or 0) <= 0:
        return None, None
    return runtime, tensor_state


def _require_native_replay_runtime_and_tensor_state(group, *, context: str):
    runtime, tensor_state = _native_replay_runtime_and_tensor_state(group)
    if runtime is None or tensor_state is None:
        raise RuntimeError(
            f"{context} requires native_sub_batch_rollout_program() state; "
            "refusing to read official native_rollout_runtime for non-record replay."
        )
    return runtime, tensor_state


def _copy_native_bw_link_transition_tape(group, *, runtime=None) -> Dict[str, np.ndarray] | None:
    if runtime is None:
        raise RuntimeError("native BW link-transition tape capture requires an explicit native replay runtime.")
    link_transition = None
    if runtime is not None:
        link_transition = getattr(getattr(runtime, "main", None), "bw_link_transition", None)
    if not link_transition:
        return None
    dtypes = {
        "uav_energy": np.float32,
        "last_energy_cost": np.float32,
        "rate_matrix": np.float32,
        "sat_loads": np.float32,
        "last_sat_score": np.float32,
    }
    tape: Dict[str, np.ndarray] = {}
    for key, dtype in dtypes.items():
        if isinstance(link_transition, dict):
            if key not in link_transition:
                return None
            value = link_transition[key]
        else:
            if not hasattr(link_transition, key):
                return None
            value = getattr(link_transition, key)
        arr = value.detach().cpu().numpy() if torch.is_tensor(value) else np.asarray(value)
        if arr.ndim > 0 and arr.shape[0] == 1:
            arr = arr[0]
        tape[key] = np.asarray(arr, dtype=dtype).copy()
    return tape


def _native_random_tape_runtime_state(group, *, index: int = 0, runtime=None) -> Dict[str, Any] | None:
    if runtime is None:
        raise RuntimeError("native random-tape runtime-state capture requires an explicit native replay runtime.")
    core = _batch_core_from_driver_group(group)
    export_slot = None if core is None else getattr(core, "_export_runtime_state_from_core_slot", None)
    if callable(export_slot):
        replay_runtime = getattr(core, "_native_sub_batch_runtime", None)
        replay_tensor_state = getattr(core, "_native_sub_batch_tensor_state", None)
        replay_cfg = getattr(core, "_native_sub_batch_cfg", None)
        replay_rng = getattr(core, "_native_sub_batch_torch_rng", None)
        replay_bound_kernels = getattr(core, "_native_sub_batch_bound_kernels", None)
        workspace_context = getattr(core, "_native_main_kernel_workspace_context", None)
        if (
            runtime is not None
            and runtime is replay_runtime
            and replay_tensor_state is not None
            and replay_cfg is not None
            and replay_rng is not None
            and isinstance(replay_bound_kernels, dict)
            and callable(workspace_context)
        ):
            with workspace_context(
                runtime=runtime,
                tensor_state=replay_tensor_state,
                cfg=replay_cfg,
                rng=replay_rng,
                bound_kernels=replay_bound_kernels,
            ):
                return export_slot(int(index))
        return export_slot(int(index))
    export_batch = getattr(group, "export_runtime_state_batch", None)
    if callable(export_batch):
        return export_batch(indices=[int(index)])[0]
    return None


def _native_random_tape_traffic_state_after_step(group, cfg, *, runtime=None) -> Dict[str, Any] | None:
    if str(getattr(cfg, "traffic_model", "homogeneous") or "homogeneous").strip().lower() != "sticky_subset_hotspot":
        return None
    if runtime is None:
        raise RuntimeError("native random-tape traffic-state capture requires an explicit native replay runtime.")
    random_state = None if runtime is None else getattr(runtime, "random", None)
    active_tape = None if random_state is None else getattr(random_state, "hotspot_active_rollout_tape", None)
    mask_tape = None if random_state is None else getattr(random_state, "hotspot_mask_rollout_tape", None)
    num_gu = int(getattr(cfg, "num_gu", 0) or 0)
    if torch.is_tensor(active_tape) and torch.is_tensor(mask_tape) and active_tape.ndim >= 2 and mask_tape.ndim >= 3:
        current_step = max(int(getattr(random_state, "step", 0) or 0) - 1, 0)
        current_step = min(current_step, int(mask_tape.shape[0]) - 1)
        next_step = min(current_step + 1, int(active_tape.shape[0]) - 1)
        active_idx = int(active_tape[next_step, 0].detach().cpu().item())
        mask = mask_tape[current_step, 0].detach().cpu().numpy().astype(np.float32, copy=True)
        if mask.shape == (num_gu,):
            return {
                "_hotspot_active_idx": active_idx,
                "last_hotspot_index": active_idx,
                "last_hotspot_mask": mask,
            }
    core = _batch_core_from_driver_group(group)
    payloads = getattr(core, "_slot_state_payloads", None)
    if isinstance(payloads, (list, tuple)) and payloads:
        payload = payloads[0]
        active_idx = int(payload.get("_hotspot_active_idx", payload.get("last_hotspot_index", -1)))
        return {
            "_hotspot_active_idx": active_idx,
            "last_hotspot_index": int(payload.get("last_hotspot_index", active_idx)),
            "last_hotspot_mask": np.asarray(
                payload.get("last_hotspot_mask", np.zeros((num_gu,), dtype=np.float32)),
                dtype=np.float32,
            ).copy(),
        }
    return {
        "_hotspot_active_idx": -1,
        "last_hotspot_index": -1,
        "last_hotspot_mask": np.zeros((num_gu,), dtype=np.float32),
    }


def _canonical_step_result_payload_from_batch(batch_result: StructuredBatchStepResult) -> Dict[str, Any]:
    reward = float(batch_result.team_rewards.reshape(-1)[0].detach().cpu().item())
    terminated = bool(batch_result.terminated.reshape(-1)[0].detach().cpu().item())
    truncated = bool(batch_result.truncated.reshape(-1)[0].detach().cpu().item())
    aux_fields = {
        "bw_access_reward": batch_result.bw_access_rewards,
        "bw_weighted_workload_delta_reward": batch_result.bw_weighted_workload_delta_rewards,
        "bw_weighted_workload_level_reward": batch_result.bw_weighted_workload_level_rewards,
        "bw_gu_queue_level_reward": batch_result.bw_gu_queue_level_rewards,
        "bw_system_queue_level_reward": batch_result.bw_system_queue_level_rewards,
        "bw_gu_service_queue_reward": batch_result.bw_gu_service_queue_rewards,
    }
    aux_payload: Dict[str, float | None] = {}
    for key, tensor in aux_fields.items():
        aux_payload[key] = None if tensor is None else float(tensor.reshape(-1)[0].detach().cpu().item())
    reward_parts: Dict[str, Any] = {}
    for key, value in (batch_result.reward_part_tensors or {}).items():
        row = value.reshape(int(batch_result.num_envs), -1)[0, 0].detach().cpu()
        reward_parts[str(key)] = bool(row.item()) if value.dtype == torch.bool else float(row.item())
    if batch_result.reward_mode_active is not None:
        reward_parts["reward_mode_active"] = str(batch_result.reward_mode_active)
    return {
        "reward": reward,
        "terminated": terminated,
        "truncated": truncated,
        **aux_payload,
        "reward_parts": reward_parts,
    }


def _compare_canonical_step_result_payload(
    errors: list[str],
    *,
    context: str,
    actual: StructuredBatchStepResult,
    expected: Dict[str, Any],
    atol: float,
    rtol: float,
) -> None:
    actual_payload = _canonical_step_result_payload_from_batch(actual)
    for key in (
        "reward",
        "terminated",
        "truncated",
        "bw_access_reward",
        "bw_weighted_workload_delta_reward",
        "bw_weighted_workload_level_reward",
        "bw_gu_queue_level_reward",
        "bw_system_queue_level_reward",
        "bw_gu_service_queue_reward",
    ):
        actual_value = actual_payload.get(key)
        expected_value = expected.get(key)
        if actual_value is None or expected_value is None:
            if actual_value is not expected_value:
                errors.append(f"{context}.{key}: actual={actual_value!r} expected={expected_value!r}")
            continue
        if isinstance(actual_value, bool) or isinstance(expected_value, bool):
            if bool(actual_value) != bool(expected_value):
                errors.append(f"{context}.{key}: actual={actual_value!r} expected={expected_value!r}")
            continue
        if not math.isclose(float(actual_value), float(expected_value), rel_tol=float(rtol), abs_tol=float(atol)):
            errors.append(
                f"{context}.{key}: actual={float(actual_value):.8f} expected={float(expected_value):.8f}"
            )
    actual_parts = dict(actual_payload.get("reward_parts", {}) or {})
    expected_parts = dict(expected.get("reward_parts", {}) or {})
    if set(actual_parts) != set(expected_parts):
        errors.append(
            f"{context}.reward_parts keys: actual={sorted(actual_parts)} expected={sorted(expected_parts)}"
        )
    for key in sorted(set(actual_parts) & set(expected_parts)):
        actual_value = actual_parts[key]
        expected_value = expected_parts[key]
        if isinstance(actual_value, str) or isinstance(expected_value, str):
            if str(actual_value) != str(expected_value):
                errors.append(f"{context}.reward_parts.{key}: actual={actual_value!r} expected={expected_value!r}")
            continue
        if not math.isclose(float(actual_value), float(expected_value), rel_tol=float(rtol), abs_tol=float(atol)):
            errors.append(
                f"{context}.reward_parts.{key}: actual={float(actual_value):.8f} "
                f"expected={float(expected_value):.8f}"
            )


def _apply_canonical_step_result_payload(step_result, payload: Dict[str, Any], env=None) -> None:
    reward = float(payload["reward"])
    terminated = bool(payload["terminated"])
    truncated = bool(payload["truncated"])
    reward_parts = dict(payload.get("reward_parts", {}) or {})
    step_result.team_reward = reward
    step_result.terminated = terminated
    step_result.truncated = truncated
    step_result.reward_parts = reward_parts
    step_result.bw_access_reward = float(reward_parts.get("x_acc", 0.0) or 0.0)
    step_result.bw_weighted_workload_delta_reward = float(
        reward_parts.get("bw_weighted_workload_delta_reward", 0.0) or 0.0
    )
    step_result.bw_weighted_workload_level_reward = float(
        reward_parts.get("bw_weighted_workload_level_reward", 0.0) or 0.0
    )
    step_result.bw_gu_queue_level_reward = float(reward_parts.get("bw_gu_queue_level_reward", 0.0) or 0.0)
    step_result.bw_system_queue_level_reward = float(reward_parts.get("bw_system_queue_level_reward", 0.0) or 0.0)
    step_result.bw_gu_service_queue_reward = float(reward_parts.get("bw_gu_service_queue_reward", 0.0) or 0.0)
    step_result.rewards = {agent: reward for agent in (step_result.rewards or {})}
    step_result.terminations = {agent: terminated for agent in (step_result.terminations or {})}
    step_result.truncations = {agent: truncated for agent in (step_result.truncations or {})}
    if env is not None:
        env.last_reward_parts = dict(reward_parts)


class _NativeActionTraceReplayBridge:
    def __init__(
        self,
        *,
        cfg,
        accel_action: np.ndarray,
        sat_action: np.ndarray,
        bw_semantic_action: np.ndarray,
        arrival_tape,
        arrival_rate_tape,
        access_gain_tape,
        doppler_residual_after_tape,
        traffic_state_after_tape,
        bw_link_transition_tape,
        materialize_outputs: bool = True,
        exec_sources: Sequence[str] | None = None,
    ) -> None:
        self.cfg = cfg
        self.accel_action = np.asarray(accel_action, dtype=np.float32)
        self.sat_action = np.asarray(sat_action, dtype=np.int64)
        self.bw_semantic_action = np.asarray(bw_semantic_action, dtype=np.float32)
        self.exec_sources = _acceptance_source_modes(exec_sources)
        self.arrival_tape = arrival_tape
        self.arrival_rate_tape = arrival_rate_tape
        self.access_gain_tape = access_gain_tape
        self.doppler_residual_after_tape = doppler_residual_after_tape
        self.traffic_state_after_tape = traffic_state_after_tape
        self.bw_link_transition_tape = bw_link_transition_tape
        self.materialize_outputs = bool(materialize_outputs)
        self.candidate_indices_batch: torch.Tensor | None = None

    def _write_random_tape_override(self, runtime, *, num_envs: int) -> None:
        if self.arrival_tape is None and self.arrival_rate_tape is None:
            return
        random = getattr(runtime, "random", None)
        if random is None:
            raise RuntimeError("native main-kernel replay requires runtime random tape buffers.")
        slot = int(getattr(runtime.history, "cursor", 0))

        def _copy_step(value, target, *, name: str) -> None:
            if value is None:
                return
            if not torch.is_tensor(target):
                raise RuntimeError(f"native action replay {name} override requires an existing rollout tape tensor.")
            if slot < 0 or slot >= int(target.shape[0]):
                raise RuntimeError(
                    f"native action replay {name} override slot {slot} is outside tape capacity {int(target.shape[0])}."
                )
            value_t = torch.as_tensor(value, dtype=torch.float32, device=target.device)
            if value_t.ndim == len(target.shape[2:]) and int(num_envs) == 1:
                value_t = value_t.unsqueeze(0)
            expected_shape = (int(num_envs),) + tuple(target.shape[2:])
            if tuple(value_t.shape) != expected_shape:
                raise RuntimeError(
                    f"native action replay {name} override has shape {tuple(value_t.shape)}, "
                    f"expected {expected_shape}."
                )
            target[slot].copy_(value_t.to(dtype=target.dtype))

        _copy_step(self.arrival_tape, getattr(random, "arrival_rollout_tape", None), name="arrival_tape")
        _copy_step(
            self.arrival_rate_tape,
            getattr(random, "arrival_rate_rollout_tape", None),
            name="arrival_rate_tape",
        )

    def _write_bw_link_transition_override(self, runtime) -> None:
        main = getattr(runtime, "main", None)
        active_t = None if main is None else getattr(main, "bw_link_transition_override_active", None)
        override = None if main is None else getattr(main, "bw_link_transition_override", None)
        if not torch.is_tensor(active_t):
            return
        if self.bw_link_transition_tape is None:
            active_t.zero_()
            return
        if override is None:
            raise RuntimeError("native main-kernel replay requires persistent BW link-transition override buffers.")
        for key in ("uav_energy", "last_energy_cost", "rate_matrix", "sat_loads", "last_sat_score"):
            if not hasattr(override, key):
                raise RuntimeError(f"native main-kernel replay missing BW link-transition override buffer {key!r}.")
            if key not in self.bw_link_transition_tape:
                raise RuntimeError(f"native main-kernel replay missing BW link-transition tape field {key!r}.")
            target = getattr(override, key)
            if not torch.is_tensor(target):
                raise RuntimeError(f"BW link-transition override buffer {key!r} must be a tensor.")
            value_t = torch.as_tensor(self.bw_link_transition_tape[key], dtype=torch.float32, device=target.device)
            if tuple(value_t.shape) == tuple(target.shape[1:]) and int(target.shape[0]) == 1:
                value_t = value_t.unsqueeze(0)
            if tuple(value_t.shape) != tuple(target.shape):
                raise RuntimeError(
                    f"BW link-transition tape field {key!r} has shape {tuple(value_t.shape)}, "
                    f"expected {tuple(target.shape)}."
                )
            target.copy_(value_t)
        active_t.fill_(1.0)

    @staticmethod
    def _copy_live_tensor(
        runtime,
        *,
        target_tensor: torch.Tensor,
        attr_name: str,
        value: torch.Tensor,
        dtype: torch.dtype | None = None,
        num_envs: int,
    ) -> None:
        if not torch.is_tensor(target_tensor):
            raise RuntimeError(f"native main-kernel replay missing live buffer {attr_name!r}.")
        tensor = runtime._require_runtime_tensor(value, field_name=f"live.{attr_name}", dtype=dtype)
        expected_shape = (int(num_envs),) + tuple(target_tensor.shape[1:])
        if tuple(tensor.shape) == tuple(target_tensor.shape[1:]) and int(num_envs) == 1:
            tensor = tensor.unsqueeze(0)
        if tuple(tensor.shape) != expected_shape:
            raise RuntimeError(
                f"live tensor {attr_name!r} has shape {tuple(tensor.shape)}, expected {expected_shape}."
            )
        target_tensor.copy_(tensor.to(device=target_tensor.device, dtype=target_tensor.dtype))

    @staticmethod
    def _sat_action_to_subset_index(runtime, sat_action: np.ndarray, *, num_envs: int) -> torch.Tensor:
        action_t = torch.as_tensor(sat_action, dtype=torch.long, device=runtime.device)
        if action_t.ndim == 1:
            action_t = action_t.unsqueeze(0)
        if action_t.ndim == 2:
            if int(action_t.shape[0]) == int(num_envs):
                return action_t
            action_t = action_t.unsqueeze(0)
        if action_t.ndim != 3:
            raise RuntimeError(f"SAT replay action must be subset indices or selection ids, got shape {tuple(action_t.shape)}.")
        stage_fields = runtime.main.sat_stage_fields
        visible_ids_t = None if stage_fields is None else getattr(stage_fields, "visible_ids", None)
        visible_mask_t = None if stage_fields is None else getattr(stage_fields, "visible_mask", None)
        subset_members_base_t = runtime.main.sat_subset_members_base
        if not torch.is_tensor(visible_ids_t) or not torch.is_tensor(visible_mask_t) or not torch.is_tensor(subset_members_base_t):
            raise RuntimeError("SAT selection-id replay requires visible SAT ids and subset constants.")
        if int(action_t.shape[0]) != int(num_envs):
            raise RuntimeError(f"SAT selection-id replay has batch {int(action_t.shape[0])}, expected {int(num_envs)}.")
        select_k = int(subset_members_base_t.shape[1])
        if int(action_t.shape[-1]) != select_k:
            raise RuntimeError(f"SAT selection-id replay has K={int(action_t.shape[-1])}, expected {select_k}.")
        subset_count = int(subset_members_base_t.shape[0])
        visible_width = int(visible_ids_t.shape[-1])
        local_subset_t = subset_members_base_t.to(device=runtime.device, dtype=torch.long)
        local_safe_t = local_subset_t.clamp(min=0, max=max(visible_width - 1, 0))
        gather_idx_t = local_safe_t.view(1, 1, subset_count, select_k).expand(
            int(num_envs),
            int(action_t.shape[1]),
            subset_count,
            select_k,
        )
        visible_ids_expanded_t = visible_ids_t.to(device=runtime.device, dtype=torch.long).unsqueeze(2).expand(
            int(num_envs),
            int(action_t.shape[1]),
            subset_count,
            visible_width,
        )
        visible_mask_expanded_t = visible_mask_t.to(device=runtime.device, dtype=torch.bool).unsqueeze(2).expand(
            int(num_envs),
            int(action_t.shape[1]),
            subset_count,
            visible_width,
        )
        selected_ids_t = torch.gather(visible_ids_expanded_t, 3, gather_idx_t)
        selected_valid_t = torch.gather(visible_mask_expanded_t, 3, gather_idx_t) & local_subset_t.ge(0).view(
            1,
            1,
            subset_count,
            select_k,
        )
        decoded_t = torch.where(selected_valid_t, selected_ids_t, selected_ids_t * 0 - 1)
        desired_t = action_t.unsqueeze(2)
        desired_valid_t = desired_t.ge(0)
        decoded_valid_t = decoded_t.ge(0)
        desired_in_decoded_t = decoded_t.unsqueeze(-1).eq(desired_t.unsqueeze(-2)).any(dim=-2)
        decoded_in_desired_t = decoded_t.unsqueeze(-1).eq(desired_t.unsqueeze(-2)).any(dim=-1)
        matches_t = (
            (desired_in_decoded_t | ~desired_valid_t).all(dim=-1)
            & (decoded_in_desired_t | ~decoded_valid_t).all(dim=-1)
        )
        if not bool(matches_t.any(dim=-1).all().detach().cpu().item()):
            bad = (~matches_t.any(dim=-1)).nonzero(as_tuple=False)[0].detach().cpu().tolist()
            env_i, uav_i = int(bad[0]), int(bad[1])
            desired_dbg = action_t[env_i, uav_i].detach().cpu().tolist()
            decoded_dbg = decoded_t[env_i, uav_i, : min(subset_count, 8)].detach().cpu().tolist()
            visible_dbg = visible_ids_t[env_i, : min(visible_width, 16)].detach().cpu().tolist()
            raise RuntimeError(
                "SAT selection-id replay could not map every selection to a subset index: "
                f"env={env_i} uav={uav_i} desired={desired_dbg} "
                f"visible={visible_dbg} decoded_first={decoded_dbg}"
            )
        return matches_t.to(dtype=torch.long).argmax(dim=-1)

    @staticmethod
    def _runtime_native_abi(runtime) -> native_cuda.NativeCudaRuntimeABI:
        abi = getattr(getattr(runtime, "main", None), "native_cuda_abi", None)
        if not isinstance(abi, native_cuda.NativeCudaRuntimeABI):
            raise RuntimeError("native main-kernel replay requires a prebuilt native CUDA runtime ABI.")
        return abi

    @staticmethod
    def _source_mode_codes(runtime) -> tuple[int, int, int]:
        main = runtime.main
        return (
            int(main.accel_actor_source_mode_code),
            int(main.sat_actor_source_mode_code),
            int(main.bw_actor_source_mode_code),
        )

    def write_accel_action(self, _accel_obs, *, runtime, num_envs: int, deterministic: bool) -> None:
        source = self.exec_sources[0]
        if source == "zero":
            return
        accel_mode, sat_mode, bw_mode = self._source_mode_codes(runtime)
        if source == "queue_aware":
            native_cuda.queue_aware_accel_live(
                self._runtime_native_abi(runtime),
                active_idx=int(runtime.main.accel_active_idx),
                accel_source_mode=accel_mode,
                sat_source_mode=sat_mode,
                bw_source_mode=bw_mode,
            )
            return
        if source == "cluster_center_queue_aware":
            native_cuda.cluster_center_accel_live(
                self._runtime_native_abi(runtime),
                active_idx=int(runtime.main.accel_active_idx),
                accel_source_mode=accel_mode,
                sat_source_mode=sat_mode,
                bw_source_mode=bw_mode,
            )
            return
        if source in {
            "uniform",
            "random",
            "link_priority",
            "demand_priority",
            "lyapunov",
            "topology_dpp_accel",
            "observable_cluster_queue_aware",
        }:
            native_cuda.baseline_accel_live(
                self._runtime_native_abi(runtime),
                active_idx=int(runtime.main.accel_active_idx),
                accel_source_mode=accel_mode,
                sat_source_mode=sat_mode,
                bw_source_mode=bw_mode,
            )
            return
        self._copy_live_tensor(
            runtime,
            target_tensor=runtime.main.live_accel_action,
            attr_name="accel_action",
            value=torch.as_tensor(self.accel_action, dtype=torch.float32, device=runtime.device).unsqueeze(0),
            dtype=torch.float32,
            num_envs=int(num_envs),
        )
        if not torch.is_tensor(runtime.main.live_accel_old_logprob):
            raise RuntimeError("native main-kernel replay missing live accel logprob buffer.")
        runtime.main.live_accel_old_logprob.zero_()

    def write_sat_action(
        self,
        _sat_obs,
        *,
        runtime,
        num_envs: int,
        sat_max_select: int,
        deterministic: bool,
    ) -> None:
        source = self.exec_sources[1]
        if source == "zero":
            return
        if source in {"queue_aware", "cluster_center_queue_aware", "observable_cluster_queue_aware"}:
            accel_mode, sat_mode, bw_mode = self._source_mode_codes(runtime)
            native_cuda.queue_aware_sat_live(
                self._runtime_native_abi(runtime),
                accel_source_mode=accel_mode,
                sat_source_mode=sat_mode,
                bw_source_mode=bw_mode,
            )
            return
        if source in {"uniform", "random", "link_priority", "demand_priority", "lyapunov", "topology_dpp_sat"}:
            accel_mode, sat_mode, bw_mode = self._source_mode_codes(runtime)
            native_cuda.baseline_sat_live(
                self._runtime_native_abi(runtime),
                accel_source_mode=accel_mode,
                sat_source_mode=sat_mode,
                bw_source_mode=bw_mode,
            )
            return
        subset_index_t = self._sat_action_to_subset_index(runtime, self.sat_action, num_envs=int(num_envs))
        self._copy_live_tensor(
            runtime,
            target_tensor=runtime.main.live_sat_subset_index,
            attr_name="sat_subset_index",
            value=subset_index_t,
            dtype=torch.long,
            num_envs=int(num_envs),
        )
        if not torch.is_tensor(runtime.main.live_sat_old_logprobs_per_agent):
            raise RuntimeError("native main-kernel replay missing live SAT per-agent logprob buffer.")
        runtime.main.live_sat_old_logprobs_per_agent.zero_()
        if torch.is_tensor(getattr(runtime.main, "live_sat_entropy_per_agent", None)):
            runtime.main.live_sat_entropy_per_agent.zero_()

    def write_bw_action(self, _bw_obs, *, runtime, num_envs: int, deterministic: bool) -> None:
        self._write_random_tape_override(runtime, num_envs=int(num_envs))
        if self.access_gain_tape is not None:
            stage_fields = getattr(runtime.main, "bw_stage_fields", None)
            access_gain_target = None if stage_fields is None else getattr(stage_fields, "access_gain_matrix", None)
            if not torch.is_tensor(access_gain_target):
                raise RuntimeError("native action replay access-gain tape requires BW stage access_gain_matrix.")
            access_gain_t = torch.as_tensor(self.access_gain_tape, dtype=torch.float32, device=access_gain_target.device)
            if access_gain_t.ndim == 2 and int(num_envs) == 1:
                access_gain_t = access_gain_t.unsqueeze(0)
            expected_shape = (int(num_envs),) + tuple(access_gain_target.shape[1:])
            if tuple(access_gain_t.shape) != expected_shape:
                raise RuntimeError(
                    f"native action replay access-gain tape has shape {tuple(access_gain_t.shape)}, "
                    f"expected {expected_shape}."
                )
            access_gain_target.copy_(access_gain_t.to(dtype=access_gain_target.dtype))
            direct_fields = getattr(runtime.main, "bw_direct_input_fields", None)
            direct_gain_target = None if direct_fields is None else getattr(direct_fields, "access_gain_matrix", None)
            if torch.is_tensor(direct_gain_target) and tuple(direct_gain_target.shape) == tuple(access_gain_target.shape):
                direct_gain_target.copy_(access_gain_target)
        source = self.exec_sources[2]
        if source == "zero":
            self._write_bw_link_transition_override(runtime)
            return
        if source in {"queue_aware", "cluster_center_queue_aware", "observable_cluster_queue_aware"}:
            accel_mode, sat_mode, bw_mode = self._source_mode_codes(runtime)
            native_cuda.queue_aware_bw_live(
                self._runtime_native_abi(runtime),
                accel_source_mode=accel_mode,
                sat_source_mode=sat_mode,
                bw_source_mode=bw_mode,
            )
            self._write_bw_link_transition_override(runtime)
            return
        if source in {
            "uniform",
            "random",
            "link_priority",
            "demand_priority",
            "lyapunov",
            "dpp_resource_bw",
            "topology_dpp_bw",
        }:
            accel_mode, sat_mode, bw_mode = self._source_mode_codes(runtime)
            native_cuda.baseline_bw_live(
                self._runtime_native_abi(runtime),
                accel_source_mode=accel_mode,
                sat_source_mode=sat_mode,
                bw_source_mode=bw_mode,
            )
            self._write_bw_link_transition_override(runtime)
            return
        bw_action_t = torch.as_tensor(self.bw_semantic_action, dtype=torch.float32, device=runtime.device)
        expected_tail = (int(self.cfg.num_uav), int(self.cfg.num_gu))
        if bw_action_t.ndim == 2 and tuple(bw_action_t.shape) != expected_tail:
            raise RuntimeError(
                f"native BW replay requires full-G semantic action shape {expected_tail}, got {tuple(bw_action_t.shape)}."
            )
        if bw_action_t.ndim == 3 and tuple(bw_action_t.shape[1:]) != expected_tail:
            raise RuntimeError(
                f"native BW replay requires full-G semantic action tail {expected_tail}, got {tuple(bw_action_t.shape)}."
            )
        self._copy_live_tensor(
            runtime,
            target_tensor=runtime.main.live_bw_action,
            attr_name="bw_action",
            value=bw_action_t,
            dtype=torch.float32,
            num_envs=int(num_envs),
        )
        self._copy_live_tensor(
            runtime,
            target_tensor=runtime.main.live_bw_ref_action,
            attr_name="bw_ref_action",
            value=bw_action_t,
            dtype=torch.float32,
            num_envs=int(num_envs),
        )
        for name in (
            "live_bw_old_logprob",
            "live_bw_old_logprobs_per_agent",
        ):
            target = getattr(runtime.main, name, None)
            if not torch.is_tensor(target):
                raise RuntimeError(f"native main-kernel replay missing {name} buffer.")
            target.zero_()
        self._write_bw_link_transition_override(runtime)


class _NativeMainKernelActionReplayProgram:
    """Single native replay boundary for formal/eval action traces."""

    def __init__(
        self,
        drivers,
        cfg,
        *,
        capacity: int | None = None,
        exec_sources: Sequence[str] | None = None,
    ) -> None:
        self.drivers = drivers
        self.cfg = cfg
        self.exec_sources = _acceptance_source_modes(exec_sources)
        replay_program_factory = getattr(drivers, "native_sub_batch_rollout_program", None)
        if not callable(replay_program_factory):
            raise RuntimeError(
                "native main-kernel action replay requires native_sub_batch_rollout_program(); "
                "refusing to use legacy hot replay or official training runtime for non-record replay."
            )
        base_capacity = max(
            int(capacity) if capacity is not None else int(getattr(cfg, "T_steps", 3) or 3),
            1,
        )
        existing_runtime, _existing_tensor_state = _native_replay_runtime_and_tensor_state(drivers)
        existing_cursor = (
            int(getattr(existing_runtime.rollout_history, "cursor", 0) or 0)
            if existing_runtime is not None
            else 0
        )
        replay_capacity = base_capacity if capacity is not None else max(base_capacity, existing_cursor + 1)
        replay_envs = len(drivers)
        self.program = replay_program_factory(
            capacity=replay_capacity,
            selected_indices=tuple(range(replay_envs)),
        )
        if self.program is None:
            raise RuntimeError("native_sub_batch_rollout_program returned no replay program.")
        if not isinstance(self.program, StructuredGpuNativeRolloutProgram):
            raise RuntimeError("native_sub_batch_rollout_program must return StructuredGpuNativeRolloutProgram.")
        self.runtime = getattr(self.program, "runtime", None)
        if self.runtime is None:
            raise RuntimeError("native_sub_batch_rollout_program returned a replay program without runtime.")
        _bind_native_replay_source_modes(self.runtime, self.exec_sources)
        self._tape_step_program = getattr(self.program, "_step_program", None)
        if self.program is not None and self._tape_step_program is None:
            raise RuntimeError("native main-kernel replay requires a private tape step executor.")

    @property
    def available(self) -> bool:
        return self.runtime is not None and self.program is not None

    def begin_horizon(self, *, num_steps: int) -> None:
        if not self.available:
            raise RuntimeError("native main-kernel replay program is not available.")
        begin_horizon = getattr(self._tape_step_program.executor, "_runtime_begin_horizon", None)
        if not callable(begin_horizon):
            raise RuntimeError("native main-kernel replay requires a final begin_horizon executor entry.")
        _bind_native_replay_source_modes(self.runtime, self.exec_sources)
        begin_horizon(num_steps=max(int(num_steps), 1))

    def _require_horizon_active(self) -> None:
        if not self.available:
            raise RuntimeError("native main-kernel replay program is not available.")
        horizon_steps = int(getattr(self.runtime.result, "horizon_num_steps", 0) or 0)
        if horizon_steps <= 0:
            raise RuntimeError("native main-kernel segmented replay must be entered through begin_horizon first.")

    def publish_accel_obs(self):
        self._require_horizon_active()
        accel_obs = self._tape_step_program._begin()
        if accel_obs is None:
            raise RuntimeError("native replay tape did not publish accel obs.")
        return accel_obs

    def publish_sat_obs_from_accel_action(
        self,
        *,
        actor_bridge: Any,
        deterministic: bool = True,
    ):
        self._require_horizon_active()
        program = self.program
        if hasattr(actor_bridge, "begin_step"):
            actor_bridge.begin_step(deterministic=deterministic)
        accel_obs = self._tape_step_program._begin()
        if accel_obs is None:
            raise RuntimeError("native replay tape did not publish accel obs.")
        actor_bridge.write_accel_action(
            accel_obs,
            runtime=program.runtime,
            num_envs=program.num_envs,
            deterministic=deterministic,
        )
        sat_obs, sat_max_select = self._tape_step_program._after_accel_action(
            max_visible=program.fixed_visible_sat_width,
        )
        if sat_obs is None:
            raise RuntimeError("native replay tape did not publish SAT obs.")
        if hasattr(actor_bridge, "after_sat_obs"):
            actor_bridge.after_sat_obs(sat_obs=sat_obs, sat_max_select=int(sat_max_select), runtime=program.runtime)
        return sat_obs

    def replay_sat_bw_tail(
        self,
        *,
        actor_bridge: Any,
        deterministic: bool = True,
    ) -> StructuredBatchStepResult:
        self._require_horizon_active()
        program = self.program
        actor_bridge.write_sat_action(
            None,
            runtime=program.runtime,
            num_envs=program.num_envs,
            sat_max_select=0,
            deterministic=deterministic,
        )
        bw_obs = self._tape_step_program._after_sat_action(
            max_visible=program.fixed_visible_sat_width,
        )
        if bw_obs is None:
            raise RuntimeError("native replay tape did not publish BW obs.")
        actor_bridge.write_bw_action(
            bw_obs,
            runtime=program.runtime,
            num_envs=program.num_envs,
            deterministic=deterministic,
        )
        finish_kwargs = {
            "rollout_tail": False,
        }
        step_result = self._tape_step_program._after_bw_action(**finish_kwargs)
        if not isinstance(step_result, StructuredBatchStepResult):
            raise RuntimeError("native GPU replay tape requires StructuredBatchStepResult.")
        return step_result

    def replay_action_trace(
        self,
        *,
        accel_action: np.ndarray,
        sat_action: np.ndarray,
        bw_semantic_action: np.ndarray,
        arrival_tape,
        arrival_rate_tape,
        access_gain_tape,
        doppler_residual_after_tape,
        traffic_state_after_tape,
        bw_link_transition_tape,
    ):
        if not self.available:
            raise RuntimeError("native main-kernel replay program is not available.")
        bridge = _NativeActionTraceReplayBridge(
            cfg=self.cfg,
            accel_action=accel_action,
            sat_action=sat_action,
            bw_semantic_action=bw_semantic_action,
            arrival_tape=arrival_tape,
            arrival_rate_tape=arrival_rate_tape,
            access_gain_tape=access_gain_tape,
            doppler_residual_after_tape=doppler_residual_after_tape,
            traffic_state_after_tape=traffic_state_after_tape,
            bw_link_transition_tape=bw_link_transition_tape,
            exec_sources=self.exec_sources,
        )
        return self.program.replay_step(
            actor_bridge=bridge,
            deterministic=True,
            rollout_tail=False,
        )

    def replay_action_horizon(
        self,
        action_traces: Sequence[Dict[str, Any]],
        *,
        deterministic: bool = True,
        step_callback: Any | None = None,
    ) -> list[StructuredBatchStepResult]:
        if not self.available:
            raise RuntimeError("native main-kernel replay program is not available.")
        traces = list(action_traces)
        self._write_horizon_random_tape_overrides(traces)

        def _bridge_factory(*, step_index: int, runtime: Any) -> _NativeActionTraceReplayBridge:
            del runtime
            trace = traces[int(step_index)]
            return _NativeActionTraceReplayBridge(
                cfg=self.cfg,
                accel_action=np.asarray(trace["accel_action"], dtype=np.float32),
                sat_action=np.asarray(trace["sat_action"], dtype=np.int64),
                bw_semantic_action=np.asarray(trace["bw_semantic_action"], dtype=np.float32),
                arrival_tape=trace.get("arrival_tape"),
                arrival_rate_tape=trace.get("arrival_rate_tape"),
                access_gain_tape=trace.get("access_gain_tape"),
                doppler_residual_after_tape=trace.get("doppler_residual_after_tape"),
                traffic_state_after_tape=trace.get("traffic_state_after_tape"),
                bw_link_transition_tape=trace.get("bw_link_transition_tape"),
                exec_sources=self.exec_sources,
            )

        return self.program.replay_horizon(
            actor_bridge_factory=_bridge_factory,
            horizon=len(traces),
            deterministic=bool(deterministic),
            step_callback=step_callback,
        )

    def _write_horizon_random_tape_overrides(self, traces: Sequence[Dict[str, Any]]) -> None:
        if not traces:
            return
        random = getattr(self.runtime, "random", None)
        if random is None:
            raise RuntimeError("native main-kernel replay requires persistent random tape buffers.")
        num_envs = int(self.program.num_envs)
        steps = int(len(traces))

        def _copy_stacked(
            *,
            trace_key: str,
            target_attr: str,
            dtype: torch.dtype,
            required: bool,
        ) -> None:
            values = [trace.get(trace_key) for trace in traces]
            if not any(value is not None for value in values):
                if required:
                    raise RuntimeError(f"native main-kernel replay trace is missing required {trace_key}.")
                return
            if any(value is None for value in values):
                raise RuntimeError(f"native main-kernel replay trace has a partial {trace_key}.")
            target = getattr(random, target_attr, None)
            if not torch.is_tensor(target):
                if trace_key == "fading_gain_tape":
                    expected = torch.ones(
                        (num_envs, int(self.cfg.num_gu), int(self.cfg.num_uav)),
                        dtype=dtype,
                        device=self.runtime.device,
                    )
                elif trace_key == "doppler_noise_tape":
                    expected = torch.zeros(
                        (num_envs, int(self.cfg.num_uav), int(self.cfg.num_sat)),
                        dtype=dtype,
                        device=self.runtime.device,
                    )
                else:
                    expected = None
                if expected is not None:
                    for step_index, value in enumerate(values):
                        value_t = torch.as_tensor(value, dtype=dtype, device=expected.device)
                        if value_t.ndim == expected.ndim - 1 and num_envs == 1:
                            value_t = value_t.unsqueeze(0)
                        if tuple(value_t.shape) != tuple(expected.shape):
                            raise RuntimeError(
                                f"native main-kernel replay {trace_key} at step {step_index} has shape "
                                f"{tuple(value_t.shape)}, expected fallback shape {tuple(expected.shape)}."
                            )
                        if not torch.allclose(value_t, expected, rtol=0.0, atol=0.0):
                            raise RuntimeError(
                                f"native main-kernel replay {trace_key} has no runtime tape buffer, "
                                "so only the fixed disabled-mode fallback is valid."
                            )
                    return
                raise RuntimeError(f"native main-kernel replay requires random.{target_attr}.")
            if steps > int(target.shape[0]):
                raise RuntimeError(
                    f"native main-kernel replay {target_attr} capacity {int(target.shape[0])} "
                    f"is smaller than horizon {steps}."
                )
            per_step_shape = (num_envs,) + tuple(target.shape[2:])
            stacked: list[torch.Tensor] = []
            for step_index, value in enumerate(values):
                value_t = torch.as_tensor(value, dtype=dtype, device=target.device)
                if value_t.ndim == len(per_step_shape) - 1 and num_envs == 1:
                    value_t = value_t.unsqueeze(0)
                if tuple(value_t.shape) != per_step_shape:
                    raise RuntimeError(
                        f"native main-kernel replay {trace_key} at step {step_index} has shape "
                        f"{tuple(value_t.shape)}, expected {per_step_shape}."
                    )
                stacked.append(value_t.to(dtype=target.dtype))
            target[:steps].copy_(torch.stack(stacked, dim=0))

        _copy_stacked(
            trace_key="arrival_tape",
            target_attr="arrival_rollout_tape",
            dtype=torch.float32,
            required=True,
        )
        _copy_stacked(
            trace_key="arrival_rate_tape",
            target_attr="arrival_rate_rollout_tape",
            dtype=torch.float32,
            required=True,
        )
        _copy_stacked(
            trace_key="fading_gain_tape",
            target_attr="fading_gain_rollout_tape",
            dtype=torch.float32,
            required=True,
        )
        _copy_stacked(
            trace_key="doppler_noise_tape",
            target_attr="doppler_noise_rollout_tape",
            dtype=torch.float32,
            required=True,
        )


def _apply_reference_accel_cache_from_native_stage_batch(driver: StructuredControlDriver, stage_batch, cfg) -> None:
    env = driver.env
    value = getattr(stage_batch, "value", None)
    if not callable(value) and hasattr(stage_batch, "_fields"):
        def value(name: str):
            return getattr(stage_batch, name, None)
    if not callable(value):
        raise RuntimeError("native reference cache replay requires a fixed tensor stage cache.")

    def _required(name: str, *, dtype):
        tensor = value(name)
        if tensor is None:
            raise RuntimeError(f"native fixed stage cache is missing {name!r}.")
        arr = _to_numpy_tape_value(tensor, dtype=dtype)
        if arr.shape and int(arr.shape[0]) == 1:
            arr = arr[0]
        return arr

    assoc = _required("assoc", dtype=np.int32)
    candidate_indices = _required("candidate_indices", dtype=np.int64)
    candidate_mask = _required("candidate_mask", dtype=np.float32)
    candidates = []
    for uav_index in range(int(cfg.num_uav)):
        row: list[int] = []
        for slot_index in range(int(cfg.users_obs_max)):
            if float(candidate_mask[uav_index, slot_index]) > 0.5:
                gu_index = int(candidate_indices[uav_index, slot_index])
                if 0 <= gu_index < int(cfg.num_gu):
                    row.append(gu_index)
        candidates.append(row)
    bw_valid_mask = _required("bw_valid_mask", dtype=np.float32)
    eta_slots = _required("eta_slots", dtype=np.float32)
    access_gain = _required("access_gain_matrix", dtype=np.float32)
    env._cached_bw_valid_mask = bw_valid_mask.copy()
    env._store_cached_access_stage_context(
        assoc,
        candidates,
        eta=eta_slots,
        bw_valid_mask=bw_valid_mask,
        access_snapshot=access_gain,
        snapshot_step_t=int(env.t),
        uav_pos=_required("uav_pos", dtype=np.float32),
        gu_pos=_required("gu_pos", dtype=np.float32),
    )
    sat_pos = _required("sat_pos", dtype=np.float32)
    sat_vel = _required("sat_vel", dtype=np.float32)
    visible_ids = value("visible_ids")
    visible_mask = value("visible_mask")
    visible = None
    if visible_ids is not None and visible_mask is not None:
        ids_arr = _to_numpy_tape_value(visible_ids, dtype=np.int64)
        mask_arr = _to_numpy_tape_value(visible_mask, dtype=np.float32)
        if ids_arr.shape and int(ids_arr.shape[0]) == 1:
            ids_arr = ids_arr[0]
        if mask_arr.shape and int(mask_arr.shape[0]) == 1:
            mask_arr = mask_arr[0]
        visible = []
        for uav_index in range(int(cfg.num_uav)):
            row: list[int] = []
            for slot_index in range(int(ids_arr.shape[1])):
                sat_id = int(ids_arr[uav_index, slot_index])
                if (
                    float(mask_arr[uav_index, slot_index]) > 0.5
                    and 0 <= sat_id < int(cfg.num_sat)
                ):
                    row.append(sat_id)
            visible.append(row)
    if visible is None:
        visible = env._visible_sats_sorted(sat_pos)
    for cache_name in (
        "_cached_elevation_t",
        "_cached_elevation_matrix",
        "_cached_backhaul_loss_t",
        "_cached_backhaul_loss_matrix",
        "_cached_uav_ecef",
        "_cached_uav_vel_ecef",
        "_cached_uav_neighbor_t",
        "_cached_uav_neighbor_order",
    ):
        if hasattr(env, cache_name):
            setattr(env, cache_name, None)
    env._cache_sat_obs(sat_pos, sat_vel, [list(row) for row in visible])
    env._cached_obs_runtime_context = None
    env._cached_global_state = None


def _run_structured_baseline_step_with_actions(
    baseline: str,
    driver: StructuredControlDriver,
    cfg,
    obs_list,
    baseline_state: Dict[str, np.ndarray] | None = None,
    random_tape_group=None,
    exec_sources: Sequence[str] | None = None,
):
    source_modes = _acceptance_source_modes(exec_sources)
    baseline_key = str(baseline).strip().lower()
    if baseline_key == "lyapunov":
        accel_actions, bw_logits, sat_logits, baseline_state = lyapunov_queue_aware_policy_step(
            obs_list,
            cfg,
            state=baseline_state,
            compute_accel=True,
            compute_bw=False,
            compute_sat=False,
            update_pressure=True,
            update_service=False,
        )
    elif baseline_key == "topology_dpp":
        accel_actions, bw_logits, sat_logits, baseline_state = topology_dpp_policy_step(
            obs_list,
            cfg,
            state=baseline_state,
            compute_accel=True,
            compute_bw=False,
            compute_sat=False,
            update_pressure=True,
            update_service=False,
        )
    elif baseline_key in {"dpp_resource_hybrid", "topology_dpp_resource"}:
        centers = getattr(driver.env, "gu_cluster_centers", None)
        counts = getattr(driver.env, "gu_cluster_counts", None)
        accel_actions, _queue_bw, _queue_sat = cluster_center_queue_aware_policy(obs_list, cfg, centers, counts)
        _dpp_accel, bw_logits, sat_logits, baseline_state = topology_dpp_policy_step(
            obs_list,
            cfg,
            state=baseline_state,
            compute_accel=False,
            compute_bw=False,
            compute_sat=False,
            update_pressure=True,
            update_service=False,
        )
    else:
        accel_actions, bw_logits, sat_logits = _baseline_actions(baseline, obs_list, cfg, driver.env)
    if accel_actions is None:
        accel_actions = zero_accel_policy(len(driver.env.agents))
    policy_accel_actions = np.asarray(accel_actions, dtype=np.float32)
    policy_bw_logits = None if bw_logits is None else np.asarray(bw_logits, dtype=np.float32)
    policy_sat_logits = None if sat_logits is None else np.asarray(sat_logits, dtype=np.float32)

    queue_bundle = None
    cluster_bundle = None

    def _queue_bundle():
        nonlocal queue_bundle
        if queue_bundle is None:
            queue_bundle = queue_aware_policy(obs_list, cfg)
        return queue_bundle

    def _cluster_bundle():
        nonlocal cluster_bundle
        if cluster_bundle is None:
            centers = getattr(driver.env, "gu_cluster_centers", None)
            counts = getattr(driver.env, "gu_cluster_counts", None)
            cluster_bundle = cluster_center_queue_aware_policy(obs_list, cfg, centers, counts)
        return cluster_bundle

    def _stage_value(source: str, stage: int):
        source_s = str(source).strip().lower()
        if source_s in {"policy", "teacher"}:
            return (policy_accel_actions, policy_bw_logits, policy_sat_logits)[stage]
        if source_s == "queue_aware":
            return _queue_bundle()[stage]
        if source_s == "cluster_center_queue_aware":
            return _cluster_bundle()[stage]
        if source_s == "zero":
            return None
        raise ValueError(f"Unsupported native acceptance source {source_s!r}.")

    accel_source_value = _stage_value(source_modes[0], 0)
    accel_actions = (
        np.zeros((int(cfg.num_uav), 2), dtype=np.float32)
        if accel_source_value is None
        else np.asarray(accel_source_value, dtype=np.float32)
    )
    access_gain_tape = None
    if random_tape_group is not None:
        tape_program = _NativeMainKernelActionReplayProgram(random_tape_group, cfg, exec_sources=source_modes)
        if not tape_program.available:
            raise RuntimeError("native random tape group must expose a native rollout program.")
        zero_sat = np.full(
            (int(cfg.num_uav), _sat_action_select_k_from_cfg(cfg)),
            -1,
            dtype=np.int64,
        )
        zero_bw = np.zeros((int(cfg.num_uav), int(cfg.num_gu)), dtype=np.float32)
        tape_bridge = _NativeActionTraceReplayBridge(
            cfg=cfg,
            accel_action=accel_actions,
            sat_action=zero_sat,
            bw_semantic_action=zero_bw,
            arrival_tape=None,
            arrival_rate_tape=None,
            access_gain_tape=None,
            doppler_residual_after_tape=None,
            traffic_state_after_tape=None,
            bw_link_transition_tape=None,
            exec_sources=source_modes,
        )
        tape_program.publish_sat_obs_from_accel_action(
            actor_bridge=tape_bridge,
            deterministic=True,
        )
        tape_runtime = tape_program.runtime
        tape_stage_batch = None if tape_runtime is None else tape_runtime.main.sat_stage_fields
        if tape_stage_batch is None and tape_runtime is not None:
            tape_stage_batch = tape_runtime.main.sat_stage
        if tape_stage_batch is not None and hasattr(tape_stage_batch, "__bool__") and not hasattr(tape_stage_batch, "_fields") and not bool(tape_stage_batch):
            tape_stage_batch = None
        if tape_stage_batch is None:
            raise RuntimeError("native random tape main-kernel did not publish SAT stage fields.")
        if hasattr(tape_stage_batch, "_fields"):
            access_gain_tensor = getattr(tape_stage_batch, "access_gain_matrix", None)
            if torch.is_tensor(access_gain_tensor):
                access_gain_tensor = access_gain_tensor.to(dtype=torch.float32)
        else:
            access_gain_tensor = tape_stage_batch.value("access_gain_matrix", dtype=torch.float32)
        if access_gain_tensor is not None:
            access_gain_tape = access_gain_tensor[0].detach().cpu().numpy().astype(np.float32, copy=True)

    driver.begin_step()
    driver.run_accel_stage(accel_actions, access_gain_override=access_gain_tape)
    if baseline_key == "lyapunov":
        _refresh_stage_obs_cache(driver)
        stage_obs_list = current_obs_many([driver], indices=[0])[0]
        _stage_accel, policy_bw_logits, policy_sat_logits, baseline_state = lyapunov_queue_aware_policy_step(
            stage_obs_list,
            cfg,
            state=baseline_state,
            compute_accel=False,
            compute_bw=True,
            compute_sat=True,
            update_pressure=False,
            update_service=True,
        )
    elif baseline_key == "topology_dpp":
        _refresh_stage_obs_cache(driver)
        stage_obs_list = current_obs_many([driver], indices=[0])[0]
        _stage_accel, policy_bw_logits, policy_sat_logits, baseline_state = topology_dpp_policy_step(
            stage_obs_list,
            cfg,
            state=baseline_state,
            compute_accel=False,
            compute_bw=True,
            compute_sat=True,
            update_pressure=False,
            update_service=True,
        )
    elif baseline_key in {"dpp_resource_hybrid", "topology_dpp_resource"}:
        _refresh_stage_obs_cache(driver)
        stage_obs_list = current_obs_many([driver], indices=[0])[0]
        _stage_accel, policy_bw_logits, policy_sat_logits, baseline_state = topology_dpp_policy_step(
            stage_obs_list,
            cfg,
            state=baseline_state,
            compute_accel=False,
            compute_bw=True,
            compute_sat=True,
            update_pressure=False,
            update_service=True,
        )
    elif baseline_key in _STAGE_FEASIBLE_BASELINES:
        _refresh_stage_obs_cache(driver)
        stage_obs_list = current_obs_many([driver], indices=[0])[0]
        _stage_accel, policy_bw_logits, policy_sat_logits = _baseline_actions(
            baseline,
            stage_obs_list,
            cfg,
            driver.env,
        )
    if access_gain_tape is None:
        access_gain_tape = (
            None
            if getattr(driver, "_stage_access_gain_matrix", None) is None
            else np.asarray(driver._stage_access_gain_matrix, dtype=np.float32).copy()
        )
    select_k = _sat_action_select_k_from_cfg(cfg)
    sat_logits_source = _stage_value(source_modes[1], 2)
    if sat_logits_source is None:
        sat_actions = np.full((cfg.num_uav, select_k), -1, dtype=np.int64)
    else:
        sat_actions = _sat_mask_to_ids(driver, np.asarray(sat_logits_source, dtype=np.float32))
    if source_modes[1] == "zero":
        sat_actions = _zero_sat_subset_zero_ids(driver, cfg)
    driver.run_sat_stage(sat_actions)
    legacy_candidates = [list(candidates) for candidates in (driver._stage_candidates or [[] for _ in range(cfg.num_uav)])]
    bw_logits_source = _stage_value(source_modes[2], 1)
    if bw_logits_source is None:
        bw_actions = np.zeros((cfg.num_uav, cfg.num_gu), dtype=np.float32)
    else:
        bw_actions = np.asarray(bw_logits_source, dtype=np.float32)
    if source_modes[2] == "zero":
        bw_actions = np.zeros((int(cfg.num_uav), int(cfg.num_gu)), dtype=np.float32)
    bw_semantic_actions = _bw_semantic_action_from_candidates(cfg, legacy_candidates, bw_actions)
    action_trace = {
        "accel_action": np.asarray(accel_actions, dtype=np.float32).copy(),
        "sat_action": np.asarray(sat_actions, dtype=np.int64).copy(),
        "bw_action": np.asarray(bw_actions, dtype=np.float32).copy(),
        "bw_semantic_action": bw_semantic_actions.copy(),
        "legacy_candidate_indices": _candidate_index_matrix_from_groups(cfg, legacy_candidates),
    }
    if access_gain_tape is not None:
        action_trace["access_gain_tape"] = access_gain_tape
    arrival_tape = None
    arrival_rate_tape = None
    doppler_residual_after_tape = None
    traffic_state_after_tape = None
    bw_link_transition_tape = None
    canonical_step_result_payload = None
    canonical_runtime_state_payload = None
    tape_runtime = None
    if random_tape_group is not None:
        tape_program = _NativeMainKernelActionReplayProgram(random_tape_group, cfg, exec_sources=source_modes)
        if not tape_program.available:
            raise RuntimeError("native random tape group must expose a native rollout program.")
        tape_bridge = _NativeActionTraceReplayBridge(
            cfg=cfg,
            accel_action=accel_actions,
            sat_action=sat_actions,
            bw_semantic_action=action_trace["bw_semantic_action"],
            arrival_tape=None,
            arrival_rate_tape=None,
            access_gain_tape=access_gain_tape,
            doppler_residual_after_tape=None,
            traffic_state_after_tape=None,
            bw_link_transition_tape=None,
            materialize_outputs=False,
            exec_sources=source_modes,
        )
        canonical_batch_result = tape_program.replay_sat_bw_tail(
            actor_bridge=tape_bridge,
            deterministic=True,
        )
        canonical_step_result_payload = _canonical_step_result_payload_from_batch(canonical_batch_result)
        tape_runtime = tape_program.runtime
        canonical_runtime_state_payload = _native_random_tape_runtime_state(random_tape_group, index=0, runtime=tape_runtime)
        bw_link_transition_tape = _copy_native_bw_link_transition_tape(random_tape_group, runtime=tape_runtime)
        tape_core = _batch_core_from_driver_group(random_tape_group)
        if tape_core is not None:
            tape_trace_runtime, tape_tensor_state = _require_native_replay_runtime_and_tensor_state(
                random_tape_group,
                context="native random-tape state capture",
            )
            if tape_runtime is not tape_trace_runtime:
                raise RuntimeError("native random-tape capture runtime does not match the active replay runtime.")
            replay_slot = max(int(getattr(tape_runtime.history, "cursor", 1)) - 1, 0)
            random_buffers = getattr(tape_runtime, "random", None)
            arrival_rollout_t = None if random_buffers is None else getattr(random_buffers, "arrival_rollout_tape", None)
            arrival_rate_rollout_t = None if random_buffers is None else getattr(random_buffers, "arrival_rate_rollout_tape", None)
            fading_rollout_t = None if random_buffers is None else getattr(random_buffers, "fading_gain_rollout_tape", None)
            doppler_noise_rollout_t = (
                None if random_buffers is None else getattr(random_buffers, "doppler_noise_rollout_tape", None)
            )
            if (
                torch.is_tensor(arrival_rollout_t)
                and 0 <= replay_slot < int(arrival_rollout_t.shape[0])
                and int(arrival_rollout_t.shape[1]) > 0
            ):
                arrival_tensor = arrival_rollout_t[replay_slot, 0]
            else:
                arrival_tensor = tape_tensor_state.last_gu_arrival[0]
            arrival_tape = arrival_tensor.detach().cpu().numpy().astype(np.float32, copy=True)
            if (
                torch.is_tensor(arrival_rate_rollout_t)
                and 0 <= replay_slot < int(arrival_rate_rollout_t.shape[0])
                and int(arrival_rate_rollout_t.shape[1]) > 0
            ):
                arrival_rate_tensor = arrival_rate_rollout_t[replay_slot, 0]
            else:
                arrival_rate_tensor = tape_tensor_state.last_gu_arrival_rate_vec[0]
            arrival_rate_tape = arrival_rate_tensor.detach().cpu().numpy().astype(np.float32, copy=True)
            if (
                torch.is_tensor(fading_rollout_t)
                and 0 <= replay_slot < int(fading_rollout_t.shape[0])
                and int(fading_rollout_t.shape[1]) > 0
            ):
                action_trace["fading_gain_tape"] = (
                    fading_rollout_t[replay_slot, 0].detach().cpu().numpy().astype(np.float32, copy=True)
                )
            else:
                action_trace["fading_gain_tape"] = np.ones(
                    (int(cfg.num_gu), int(cfg.num_uav)),
                    dtype=np.float32,
                )
            if (
                torch.is_tensor(doppler_noise_rollout_t)
                and 0 <= replay_slot < int(doppler_noise_rollout_t.shape[0])
                and int(doppler_noise_rollout_t.shape[1]) > 0
            ):
                action_trace["doppler_noise_tape"] = (
                    doppler_noise_rollout_t[replay_slot, 0].detach().cpu().numpy().astype(np.float32, copy=True)
                )
            else:
                action_trace["doppler_noise_tape"] = np.zeros(
                    (int(cfg.num_uav), int(cfg.num_sat)),
                    dtype=np.float32,
                )
            doppler_tensor = tape_tensor_state.doppler_residual[0]
            doppler_residual_after_tape = doppler_tensor.detach().cpu().numpy().astype(np.float32, copy=True)
        traffic_state_after_tape = _native_random_tape_traffic_state_after_step(random_tape_group, cfg, runtime=tape_runtime)
    step_result = driver.execute_stage_bw_and_step(
        bw_semantic_actions,
        arrival_override=arrival_tape,
        doppler_residual_after_override=doppler_residual_after_tape,
        traffic_state_after_override=traffic_state_after_tape,
        bw_link_transition_override=bw_link_transition_tape,
    )
    if canonical_step_result_payload is not None:
        _apply_canonical_step_result_payload(step_result, canonical_step_result_payload, env=driver.env)
    if canonical_runtime_state_payload is not None:
        driver.env.load_runtime_state(
            copy.deepcopy(canonical_runtime_state_payload),
            refresh_observation_cache=False,
            refresh_global_state_cache=True,
        )
    if random_tape_group is not None and not bool(getattr(step_result, "terminated", False) or getattr(step_result, "truncated", False)):
        next_stage_tape = None
        if tape_runtime is not None:
            stage_buffers = tape_runtime.main.accel_stage_field_buffers
            active_idx = int(tape_runtime.main.accel_active_idx)
            if stage_buffers is not None and active_idx in {0, 1}:
                next_stage_tape = stage_buffers[active_idx]
        if next_stage_tape is not None and hasattr(next_stage_tape, "__bool__") and not hasattr(next_stage_tape, "_fields") and not bool(next_stage_tape):
            next_stage_tape = None
        if next_stage_tape is None:
            raise RuntimeError("native random tape main-kernel did not publish next accel stage fields.")
        if next_stage_tape is not None:
            _apply_reference_accel_cache_from_native_stage_batch(driver, next_stage_tape, cfg)
            step_result.obs = {
                agent: driver.env._get_obs(agent_index)
                for agent_index, agent in enumerate(driver.env.agents)
            }
    if arrival_tape is None:
        arrival_tape = np.asarray(
            getattr(driver.env, "last_gu_arrival", np.zeros((int(cfg.num_gu),), dtype=np.float32)),
            dtype=np.float32,
        ).copy()
    action_trace["arrival_tape"] = np.asarray(arrival_tape, dtype=np.float32).copy()
    if arrival_rate_tape is None:
        arrival_rate_tape = np.asarray(
            getattr(driver.env, "last_gu_arrival_rate_vec", np.zeros((int(cfg.num_gu),), dtype=np.float32)),
            dtype=np.float32,
        ).copy()
    action_trace["arrival_rate_tape"] = np.asarray(arrival_rate_tape, dtype=np.float32).copy()
    if doppler_residual_after_tape is not None:
        action_trace["doppler_residual_after_tape"] = np.asarray(doppler_residual_after_tape, dtype=np.float32).copy()
    if traffic_state_after_tape is not None:
        action_trace["traffic_state_after_tape"] = copy.deepcopy(traffic_state_after_tape)
    if bw_link_transition_tape is not None:
        action_trace["bw_link_transition_tape"] = {
            key: np.asarray(value, dtype=np.float32).copy()
            for key, value in bw_link_transition_tape.items()
        }
    if canonical_step_result_payload is not None:
        action_trace["expected_step_result_payload"] = copy.deepcopy(canonical_step_result_payload)
    return step_result, baseline_state, action_trace


def _runtime_state_trace_row_from_state(state: Dict[str, Any]) -> Dict[str, float]:
    gu_queue = np.asarray(state.get("gu_queue", []), dtype=np.float32)
    uav_queue = np.asarray(state.get("uav_queue", []), dtype=np.float32)
    sat_queue = np.asarray(state.get("sat_queue", []), dtype=np.float32)
    gu_queue_sum = float(np.sum(gu_queue, dtype=np.float64)) if gu_queue.size else 0.0
    uav_queue_sum = float(np.sum(uav_queue, dtype=np.float64)) if uav_queue.size else 0.0
    sat_queue_sum = float(np.sum(sat_queue, dtype=np.float64)) if sat_queue.size else 0.0
    return {
        "t": float(state.get("t", 0.0) or 0.0),
        "global_step": float(state.get("global_step", 0.0) or 0.0),
        "gu_queue_sum": gu_queue_sum,
        "uav_queue_sum": uav_queue_sum,
        "sat_queue_sum": sat_queue_sum,
        "queue_total_sum": gu_queue_sum + uav_queue_sum + sat_queue_sum,
    }


def _runtime_state_trace_rows(
    drivers,
    *,
    indices: List[int],
    runtime=None,
    tensor_state=None,
) -> List[Dict[str, float]]:
    if runtime is None:
        runtime = getattr(drivers, "native_rollout_runtime", None)
    core = _batch_core_from_driver_group(drivers)
    if tensor_state is None:
        tensor_state = None if core is None else getattr(core, "_runtime_tensor_state", None)
    runtime_active = bool(
        runtime is not None
        and getattr(runtime, "main", None) is not None
        and int(getattr(runtime.main, "num_envs", 0) or 0) > 0
        and tensor_state is not None
        and torch.is_tensor(getattr(tensor_state, "gu_queue", None))
    )
    if runtime_active:
        rows: List[Dict[str, float]] = []
        for index in indices:
            idx = int(index)
            gu_sum_t = tensor_state.gu_queue[idx].to(dtype=torch.float64).sum()
            uav_sum_t = tensor_state.uav_queue[idx].to(dtype=torch.float64).sum()
            sat_sum_t = tensor_state.sat_queue[idx].to(dtype=torch.float64).sum()
            gu_queue_sum = float(gu_sum_t.detach().cpu().item())
            uav_queue_sum = float(uav_sum_t.detach().cpu().item())
            sat_queue_sum = float(sat_sum_t.detach().cpu().item())
            rows.append(
                {
                    "t": float(tensor_state.t[idx].detach().cpu().item()),
                    "global_step": float(tensor_state.global_step[idx].detach().cpu().item()),
                    "gu_queue_sum": gu_queue_sum,
                    "uav_queue_sum": uav_queue_sum,
                    "sat_queue_sum": sat_queue_sum,
                    "queue_total_sum": gu_queue_sum + uav_queue_sum + sat_queue_sum,
                }
            )
        return rows
    export_batch = getattr(drivers, "export_runtime_state_batch", None)
    if callable(export_batch):
        return [
            _runtime_state_trace_row_from_state(state)
            for state in export_batch(indices=[int(index) for index in indices])
        ]
    driver_list = _as_driver_list(drivers) if looks_like_driver_group(drivers) else list(drivers)
    rows: List[Dict[str, float]] = []
    for index in indices:
        driver = driver_list[int(index)]
        env = getattr(driver, "env", driver)
        rows.append(
            {
                "t": float(getattr(env, "t", 0.0) or 0.0),
                "global_step": float(getattr(env, "global_step", 0.0) or 0.0),
                "gu_queue_sum": float(np.sum(np.asarray(getattr(env, "gu_queue", []), dtype=np.float32), dtype=np.float64)),
                "uav_queue_sum": float(np.sum(np.asarray(getattr(env, "uav_queue", []), dtype=np.float32), dtype=np.float64)),
                "sat_queue_sum": float(np.sum(np.asarray(getattr(env, "sat_queue", []), dtype=np.float32), dtype=np.float64)),
                "queue_total_sum": float(
                    np.sum(np.asarray(getattr(env, "gu_queue", []), dtype=np.float32), dtype=np.float64)
                    + np.sum(np.asarray(getattr(env, "uav_queue", []), dtype=np.float32), dtype=np.float64)
                    + np.sum(np.asarray(getattr(env, "sat_queue", []), dtype=np.float32), dtype=np.float64)
                ),
            }
        )
    return rows


def _step_trace_row(
    *,
    episode_index: int,
    step_index: int,
    step_result,
    reward_parts: Dict[str, float],
    runtime_trace: Dict[str, float],
) -> Dict[str, float]:
    if isinstance(step_result, StructuredBatchStepResult):
        terminated = bool(step_result.terminated.reshape(-1)[0].detach().cpu().item())
        truncated = bool(step_result.truncated.reshape(-1)[0].detach().cpu().item())
    else:
        terminations = getattr(step_result, "terminations", {}) or {}
        truncations = getattr(step_result, "truncations", {}) or {}
        terminated = bool(next(iter(terminations.values()))) if terminations else bool(getattr(step_result, "terminated", False))
        truncated = bool(next(iter(truncations.values()))) if truncations else bool(getattr(step_result, "truncated", False))
    return {
        "episode": float(episode_index),
        "step": float(step_index),
        "reward": float(_step_team_reward(step_result)),
        "processed_ratio_eval": float(reward_parts.get("processed_ratio_eval", 0.0) or 0.0),
        "drop_ratio_eval": float(reward_parts.get("drop_ratio_eval", 0.0) or 0.0),
        "pre_backlog_steps_eval": float(reward_parts.get("pre_backlog_steps_eval", 0.0) or 0.0),
        "D_sys_report": float(reward_parts.get("D_sys_report", 0.0) or 0.0),
        "x_acc": float(reward_parts.get("x_acc", 0.0) or 0.0),
        "x_rel": float(reward_parts.get("x_rel", 0.0) or 0.0),
        "g_pre": float(reward_parts.get("g_pre", 0.0) or 0.0),
        "d_pre": float(reward_parts.get("d_pre", 0.0) or 0.0),
        "bw_weighted_workload_level_reward": float(
            reward_parts.get(
                "bw_weighted_workload_level_reward",
                _batch_step_scalar(step_result, "bw_weighted_workload_level_reward"),
            )
            or 0.0
        ),
        "sat_overlap_eval": float(reward_parts.get("sat_overlap_eval", 0.0) or 0.0),
        "collision_event": float(reward_parts.get("collision_event", 0.0) or 0.0),
        "terminated": float(terminated),
        "truncated": float(truncated),
        "done": float(terminated or truncated),
        "t": float(step_index + 1),
        "global_step": float(runtime_trace.get("global_step", 0.0) or 0.0),
        "gu_queue_sum": float(reward_parts.get("gu_queue_sum", runtime_trace.get("gu_queue_sum", 0.0)) or 0.0),
        "uav_queue_sum": float(reward_parts.get("uav_queue_sum", runtime_trace.get("uav_queue_sum", 0.0)) or 0.0),
        "sat_queue_sum": float(reward_parts.get("sat_queue_sum", runtime_trace.get("sat_queue_sum", 0.0)) or 0.0),
        "queue_total_sum": float(reward_parts.get("queue_total_sum", runtime_trace.get("queue_total_sum", 0.0)) or 0.0),
    }


def _history_flat_numpy(
    tensor: torch.Tensor | None,
    *,
    steps: int,
    num_envs: int,
    dtype=np.float32,
    default: float = 0.0,
) -> np.ndarray:
    shape = (max(int(steps), 0), max(int(num_envs), 1))
    if tensor is None or not torch.is_tensor(tensor):
        return np.full(shape, default, dtype=dtype)
    count = int(steps) * int(num_envs)
    if count <= 0:
        return np.zeros(shape, dtype=dtype)
    return (
        tensor.detach()
        .reshape(-1)[:count]
        .reshape(int(steps), int(num_envs))
        .cpu()
        .numpy()
        .astype(dtype, copy=True)
    )


def _history_reward_part_arrays(
    history: Any,
    *,
    steps: int,
    num_envs: int,
) -> Dict[str, np.ndarray]:
    reward_parts = getattr(getattr(history, "bw_stage", None), "reward_part_tensors", None)
    if not isinstance(reward_parts, dict):
        return {}
    arrays: Dict[str, np.ndarray] = {}
    for key, tensor in reward_parts.items():
        if torch.is_tensor(tensor):
            arrays[str(key)] = _history_flat_numpy(tensor, steps=steps, num_envs=num_envs, dtype=np.float32)
    return arrays


def _native_replay_rows_and_traces_from_history(
    runtime: Any,
    *,
    episode_index: int,
    num_steps: int,
    num_envs: int = 1,
    cfg=None,
) -> tuple[Dict[str, float], List[Dict[str, float]]]:
    history = getattr(runtime, "history", None)
    if history is None or getattr(history, "bw_stage", None) is None:
        raise RuntimeError("native replay trace extraction requires runtime history.")
    steps = max(int(num_steps), 0)
    n = max(int(num_envs), 1)
    rewards = _history_flat_numpy(history.bw_stage.rewards, steps=steps, num_envs=n, dtype=np.float32)
    terminated = _history_flat_numpy(history.terminated, steps=steps, num_envs=n, dtype=np.bool_)
    truncated = _history_flat_numpy(history.truncated, steps=steps, num_envs=n, dtype=np.bool_)
    parts = _history_reward_part_arrays(history, steps=steps, num_envs=n)

    def part(name: str, default: float = 0.0) -> np.ndarray:
        value = parts.get(str(name))
        if value is None:
            return np.full((steps, n), float(default), dtype=np.float32)
        return value

    gu_queue_sum = part("gu_queue_sum")
    uav_queue_sum = part("uav_queue_sum")
    sat_queue_sum = part("sat_queue_sum")
    queue_total_sum = parts.get("queue_total_sum")
    if queue_total_sum is None:
        queue_total_sum = gu_queue_sum + uav_queue_sum + sat_queue_sum

    acc = _new_episode_accumulator(cfg)
    traces: List[Dict[str, float]] = []
    for step_index in range(steps):
        slot = 0
        term = bool(terminated[step_index, slot])
        trunc = bool(truncated[step_index, slot])
        done = bool(term or trunc)
        reward_parts = {
            "processed_ratio_eval": float(part("processed_ratio_eval")[step_index, slot]),
            "drop_ratio_eval": float(part("drop_ratio_eval")[step_index, slot]),
            "pre_backlog_steps_eval": float(part("pre_backlog_steps_eval")[step_index, slot]),
            "D_sys_report": float(part("D_sys_report")[step_index, slot]),
            "x_acc": float(part("x_acc")[step_index, slot]),
            "x_rel": float(part("x_rel")[step_index, slot]),
            "g_pre": float(part("g_pre")[step_index, slot]),
            "d_pre": float(part("d_pre")[step_index, slot]),
            "sat_overlap_eval": float(part("sat_overlap_eval")[step_index, slot]),
            "collision_event": float(part("collision_event")[step_index, slot]),
        }
        for name in (
            "gu_queue_sum",
            "uav_queue_sum",
            "sat_queue_sum",
            "queue_total_sum",
            "arrival_sum",
            "outflow_sum",
            "backhaul_sum",
            "sat_processed_sum",
            "drop_sum",
            "drop_sum_active",
            "gu_drop_sum",
            "uav_drop_sum",
            "sat_drop_sum",
        ):
            reward_parts[name] = float(part(name)[step_index, slot])
        reward_value = float(rewards[step_index, slot])
        bw_delta_value = float(part("bw_weighted_workload_delta_reward")[step_index, slot])
        bw_level_value = float(part("bw_weighted_workload_level_reward")[step_index, slot])
        _accumulate_step_metrics(
            acc,
            reward_value=reward_value,
            bw_delta_value=bw_delta_value,
            bw_level_value=bw_level_value,
            reward_parts=reward_parts,
        )
        traces.append(
            {
                "episode": float(episode_index),
                "step": float(step_index),
                "reward": reward_value,
                **reward_parts,
                "terminated": float(term),
                "truncated": float(trunc),
                "done": float(done),
                "t": float(step_index + 1),
                "global_step": float(step_index + 1),
                "gu_queue_sum": float(gu_queue_sum[step_index, slot]),
                "uav_queue_sum": float(uav_queue_sum[step_index, slot]),
                "sat_queue_sum": float(sat_queue_sum[step_index, slot]),
                "queue_total_sum": float(queue_total_sum[step_index, slot]),
                "arrival_sum": reward_parts["arrival_sum"],
                "outflow_sum": reward_parts["outflow_sum"],
                "backhaul_sum": reward_parts["backhaul_sum"],
                "sat_processed_sum": reward_parts["sat_processed_sum"],
                "drop_sum": reward_parts["drop_sum"],
            }
        )
    return _episode_row_from_accumulator(int(episode_index), acc), traces


def _compare_expected_step_payloads_from_history(
    errors: list[str],
    *,
    context: str,
    runtime: Any,
    action_traces: Sequence[Dict[str, Any]],
    atol: float,
    rtol: float,
) -> None:
    expected_payloads = [trace.get("expected_step_result_payload") for trace in action_traces]
    if not any(isinstance(payload, dict) for payload in expected_payloads):
        return
    if not all(isinstance(payload, dict) for payload in expected_payloads):
        errors.append(f"{context}.expected_step_result_payload: partial payloads")
        return
    history = getattr(runtime, "history", None)
    steps = len(expected_payloads)
    rewards = _history_flat_numpy(history.bw_stage.rewards, steps=steps, num_envs=1, dtype=np.float32).reshape(-1)
    terminated = _history_flat_numpy(history.terminated, steps=steps, num_envs=1, dtype=np.bool_).reshape(-1)
    truncated = _history_flat_numpy(history.truncated, steps=steps, num_envs=1, dtype=np.bool_).reshape(-1)
    parts = {
        key: value.reshape(-1)
        for key, value in _history_reward_part_arrays(history, steps=steps, num_envs=1).items()
    }
    aux_actual = {
        "bw_access_reward": _history_flat_numpy(history.bw_stage.bw_access_rewards, steps=steps, num_envs=1).reshape(-1),
        "bw_weighted_workload_delta_reward": _history_flat_numpy(
            history.bw_stage.bw_weighted_workload_delta_rewards,
            steps=steps,
            num_envs=1,
        ).reshape(-1),
        "bw_weighted_workload_level_reward": _history_flat_numpy(
            history.bw_stage.bw_weighted_workload_level_rewards,
            steps=steps,
            num_envs=1,
        ).reshape(-1),
        "bw_gu_queue_level_reward": _history_flat_numpy(
            history.bw_stage.bw_gu_queue_level_rewards,
            steps=steps,
            num_envs=1,
        ).reshape(-1),
        "bw_system_queue_level_reward": _history_flat_numpy(
            history.bw_stage.bw_system_queue_level_rewards,
            steps=steps,
            num_envs=1,
        ).reshape(-1),
        "bw_gu_service_queue_reward": _history_flat_numpy(
            history.bw_stage.bw_gu_service_queue_rewards,
            steps=steps,
            num_envs=1,
        ).reshape(-1),
    }
    reward_mode_active = str(getattr(getattr(runtime, "main", None), "bw_reward_mode_active", "") or "")
    for step_index, expected in enumerate(expected_payloads):
        if not math.isclose(
            float(rewards[step_index]),
            float(expected.get("reward", 0.0)),
            rel_tol=float(rtol),
            abs_tol=float(atol),
        ):
            errors.append(
                f"{context} step={step_index}.reward: "
                f"actual={float(rewards[step_index]):.8f} expected={float(expected.get('reward', 0.0)):.8f}"
            )
        if bool(terminated[step_index]) != bool(expected.get("terminated", False)):
            errors.append(f"{context} step={step_index}.terminated: actual={bool(terminated[step_index])} expected={bool(expected.get('terminated', False))}")
        if bool(truncated[step_index]) != bool(expected.get("truncated", False)):
            errors.append(f"{context} step={step_index}.truncated: actual={bool(truncated[step_index])} expected={bool(expected.get('truncated', False))}")
        for key, actual_values in aux_actual.items():
            expected_value = expected.get(key)
            if expected_value is None:
                continue
            if not math.isclose(
                float(actual_values[step_index]),
                float(expected_value),
                rel_tol=float(rtol),
                abs_tol=float(atol),
            ):
                errors.append(
                    f"{context} step={step_index}.{key}: "
                    f"actual={float(actual_values[step_index]):.8f} expected={float(expected_value):.8f}"
                )
        expected_parts = dict(expected.get("reward_parts", {}) or {})
        for key, expected_value in expected_parts.items():
            if str(key) == "reward_mode_active":
                if str(expected_value) != reward_mode_active:
                    errors.append(
                        f"{context} step={step_index}.reward_parts.{key}: "
                        f"actual={reward_mode_active!r} expected={expected_value!r}"
                    )
                continue
            actual_values = parts.get(str(key))
            if actual_values is None:
                errors.append(f"{context} step={step_index}.reward_parts.{key}: missing actual")
                continue
            if not math.isclose(
                float(actual_values[step_index]),
                float(expected_value),
                rel_tol=float(rtol),
                abs_tol=float(atol),
            ):
                errors.append(
                    f"{context} step={step_index}.reward_parts.{key}: "
                    f"actual={float(actual_values[step_index]):.8f} expected={float(expected_value):.8f}"
                )


_FIXED_POLICY_EXEC_SOURCE_MAP: dict[str, tuple[str, str, str]] = {
    "distributed_queue_a": ("distributed_queue_a", "queue_aware", "queue_aware"),
    "distributed_queue_b": ("distributed_queue_b", "queue_aware", "queue_aware"),
    "distributed_queue_c": ("distributed_queue_c", "lyapunov", "lyapunov"),
    "zero": ("zero", "zero", "zero"),
    "static_uniform": ("zero", "uniform", "uniform"),
    "static": ("zero", "uniform", "uniform"),
    "uniform": ("zero", "uniform", "uniform"),
    "random": ("random", "random", "random"),
    "random_feasible": ("random", "random", "random"),
    "link_priority": ("zero", "link_priority", "link_priority"),
    "demand_priority": ("zero", "demand_priority", "demand_priority"),
    # `lyapunov` is kept as a compatibility alias. The clearer current name is
    # `maxweight_lyapunov`, because the implementation is a stage-wise
    # MaxWeight/Lyapunov queue-pressure controller rather than the archived
    # topology-enumerating DPP prototype.
    "lyapunov": ("lyapunov", "lyapunov", "lyapunov"),
    "maxweight_lyapunov": ("lyapunov", "lyapunov", "lyapunov"),
    "lyapunov_maxweight": ("lyapunov", "lyapunov", "lyapunov"),
    "queue_aware_bw": ("zero", "zero", "queue_aware"),
    "queue_aware": ("queue_aware", "queue_aware", "queue_aware"),
    "cluster_center_queue_aware": (
        "cluster_center_queue_aware",
        "queue_aware",
        "queue_aware",
    ),
    "observable_cluster_queue_aware": (
        "observable_cluster_queue_aware",
        "queue_aware",
        "queue_aware",
    ),
    # Lightweight DPP/MaxWeight ablations built from existing native source
    # modes. These isolate which decision layer carries the non-learning
    # controller's gains before adding a heavier topology-enumerating DPP.
    "dpp_no_mobility": ("zero", "lyapunov", "lyapunov"),
    "dpp_equal_bw": ("lyapunov", "lyapunov", "uniform"),
    "dpp_greedy_sat": ("lyapunov", "queue_aware", "lyapunov"),
    "dpp_resource_bw": ("zero", "zero", "topology_dpp_bw"),
    "topology_dpp_bw": ("zero", "zero", "topology_dpp_bw"),
    "dpp_resource_hybrid_native": (
        "cluster_center_queue_aware",
        "queue_aware",
        "topology_dpp_bw",
    ),
    "topology_dpp_resource_native": (
        "cluster_center_queue_aware",
        "queue_aware",
        "topology_dpp_bw",
    ),
    "topology_dpp_native_bw_sat_cached": (
        "cluster_center_queue_aware",
        "topology_dpp_sat",
        "topology_dpp_bw",
    ),
    "full_topology_dpp_joint": (
        "topology_dpp_accel",
        "topology_dpp_sat",
        "topology_dpp_bw",
    ),
    "topology_dpp_joint_native": (
        "topology_dpp_accel",
        "topology_dpp_sat",
        "topology_dpp_bw",
    ),
}


def _fixed_policy_exec_sources(baseline_policy: str) -> tuple[str, str, str] | None:
    baseline = str(baseline_policy).strip().lower()
    return _FIXED_POLICY_EXEC_SOURCE_MAP.get(baseline)


def _acceptance_source_modes(exec_sources: Sequence[str] | None) -> tuple[str, str, str]:
    if exec_sources is None:
        return ("policy", "policy", "policy")
    if len(exec_sources) != 3:
        raise ValueError("native acceptance exec_sources must contain accel/sat/bw entries.")
    allowed = {
        "policy",
        "zero",
        "queue_aware",
        "cluster_center_queue_aware",
        "observable_cluster_queue_aware",
        "teacher",
        "uniform",
        "random",
        "link_priority",
        "demand_priority",
        "lyapunov",
        "dpp_resource_bw",
        "topology_dpp_accel",
        "topology_dpp_bw",
        "topology_dpp_sat",
    }
    modes: list[str] = []
    for source in exec_sources:
        source_s = str(source or "policy").strip().lower()
        if source_s not in allowed:
            raise ValueError(
                f"native acceptance exec source {source_s!r} is not supported; expected one of {sorted(allowed)}."
            )
        modes.append(source_s)
    return (modes[0], modes[1], modes[2])


def _native_acceptance_source_mode_code(source: str) -> int:
    source_s = str(source or "policy").strip().lower()
    table = {
        "policy": native_cuda.SOURCE_POLICY,
        "teacher": native_cuda.SOURCE_POLICY,
        "zero": native_cuda.SOURCE_ZERO,
        "uniform": native_cuda.SOURCE_UNIFORM,
        "random": native_cuda.SOURCE_RANDOM,
        "link_priority": native_cuda.SOURCE_LINK_PRIORITY,
        "demand_priority": native_cuda.SOURCE_DEMAND_PRIORITY,
        "queue_aware": native_cuda.SOURCE_QUEUE_AWARE,
        "cluster_center_queue_aware": native_cuda.SOURCE_CLUSTER_CENTER_QUEUE_AWARE,
        "observable_cluster_queue_aware": native_cuda.SOURCE_OBSERVABLE_CLUSTER_QUEUE_AWARE,
        "lyapunov": native_cuda.SOURCE_LYAPUNOV,
        "dpp_resource_bw": native_cuda.SOURCE_DPP_RESOURCE_BW,
        "topology_dpp_accel": native_cuda.SOURCE_TOPOLOGY_DPP_ACCEL,
        "topology_dpp_bw": native_cuda.SOURCE_TOPOLOGY_DPP_BW,
        "topology_dpp_sat": native_cuda.SOURCE_TOPOLOGY_DPP_SAT,
    }
    if source_s not in table:
        raise RuntimeError(f"native replay source {source_s!r} is not supported.")
    return int(table[source_s])


def _bind_native_replay_source_modes(runtime: Any, exec_sources: Sequence[str] | None) -> None:
    if runtime is None or getattr(runtime, "main", None) is None:
        raise RuntimeError("native replay source modes require a live runtime.")
    accel_source, sat_source, bw_source = _acceptance_source_modes(exec_sources)
    runtime.main.accel_actor_source_mode_code = _native_acceptance_source_mode_code(accel_source)
    runtime.main.sat_actor_source_mode_code = _native_acceptance_source_mode_code(sat_source)
    runtime.main.bw_actor_source_mode_code = _native_acceptance_source_mode_code(bw_source)


def _zero_sat_subset_zero_ids(driver: StructuredControlDriver, cfg) -> np.ndarray:
    sat_states = driver.build_local_sat_states()
    subset_indices: list[int] = []
    for state in sat_states:
        valid = torch.nonzero(state.subset_mask[0], as_tuple=False).flatten()
        subset_indices.append(int(valid[0].item()) if int(valid.numel()) > 0 else 0)
    _selections, sat_actions = driver._sat_selections_from_subset_indices(subset_indices)
    return np.asarray(sat_actions, dtype=np.int64)


def _native_history_prefix_tensor(
    value: torch.Tensor | None,
    *,
    capacity: int,
    num_steps: int,
    field_name: str,
) -> torch.Tensor | None:
    if value is None:
        return None
    if not torch.is_tensor(value):
        raise RuntimeError(f"native history field {field_name} must be a tensor.")
    per_step = int(value.shape[0]) // max(int(capacity), 1)
    return value[: int(num_steps) * per_step]


def _compare_native_tensor_field(
    errors: list[str],
    name: str,
    actual: torch.Tensor | None,
    expected: torch.Tensor | None,
    *,
    atol: float,
    rtol: float,
) -> None:
    if actual is None or expected is None:
        if actual is not expected:
            errors.append(f"{name}: None mismatch actual={actual is None} expected={expected is None}")
        return
    if not torch.is_tensor(actual) or not torch.is_tensor(expected):
        errors.append(f"{name}: non-tensor field")
        return
    actual_t = actual.detach()
    expected_t = expected.detach()
    if expected_t.device != actual_t.device:
        expected_t = expected_t.to(device=actual_t.device)
    if tuple(actual_t.shape) != tuple(expected_t.shape):
        errors.append(f"{name}: shape actual={tuple(actual_t.shape)} expected={tuple(expected_t.shape)}")
        return
    if actual_t.dtype == torch.bool or expected_t.dtype == torch.bool:
        if not torch.equal(actual_t.to(dtype=torch.bool), expected_t.to(dtype=torch.bool)):
            errors.append(f"{name}: bool mismatch")
        return
    if not torch.is_floating_point(actual_t) and not torch.is_floating_point(expected_t):
        if not torch.equal(actual_t, expected_t.to(dtype=actual_t.dtype)):
            errors.append(f"{name}: integer mismatch")
        return
    actual_f = actual_t.to(dtype=torch.float32)
    expected_f = expected_t.to(dtype=torch.float32)
    actual_nan = torch.isnan(actual_f)
    expected_nan = torch.isnan(expected_f)
    if not torch.equal(actual_nan, expected_nan):
        errors.append(f"{name}: nan mask mismatch")
        return
    finite_mask = ~(actual_nan | expected_nan)
    if not torch.any(finite_mask):
        return
    actual_finite = actual_f[finite_mask]
    expected_finite = expected_f[finite_mask]
    if not torch.allclose(actual_finite, expected_finite, atol=float(atol), rtol=float(rtol)):
        max_diff = (actual_finite - expected_finite).abs().max().detach().cpu().item()
        errors.append(f"{name}: max_abs_diff={float(max_diff):.8g}")


def _compare_native_dataclass_tensors(
    errors: list[str],
    name: str,
    actual: Any,
    expected: Any,
    *,
    atol: float,
    rtol: float,
) -> None:
    if actual is None or expected is None:
        if actual is not expected:
            errors.append(f"{name}: None mismatch actual={actual is None} expected={expected is None}")
        return
    if not (is_dataclass(actual) and is_dataclass(expected)):
        errors.append(f"{name}: expected dataclass tensors")
        return
    for field_info in fields(actual):
        field_name = str(field_info.name)
        actual_value = getattr(actual, field_name)
        expected_value = getattr(expected, field_name, None)
        child_name = f"{name}.{field_name}"
        if torch.is_tensor(actual_value) or torch.is_tensor(expected_value) or actual_value is None or expected_value is None:
            _compare_native_tensor_field(errors, child_name, actual_value, expected_value, atol=atol, rtol=rtol)
        elif is_dataclass(actual_value):
            _compare_native_dataclass_tensors(errors, child_name, actual_value, expected_value, atol=atol, rtol=rtol)


def _native_rollout_field_contract_errors(
    runtime: Any,
    *,
    num_steps: int,
    num_envs: int,
    atol: float,
    rtol: float,
) -> list[str]:
    errors: list[str] = []
    if runtime is None or getattr(runtime, "history", None) is None:
        return ["native field contract: missing runtime/history"]
    history = runtime.history
    capacity = int(getattr(history, "capacity", 0) or 0)
    steps = max(int(num_steps), 0)
    n = max(int(num_envs), 0)
    if steps <= 0 or n <= 0:
        return errors
    if int(getattr(history, "cursor", 0) or 0) < steps:
        errors.append(f"native field contract: cursor={int(getattr(history, 'cursor', 0) or 0)} steps={steps}")
        return errors
    views = _build_rollout_views_from_native_training_ring(
        StructuredNativeRolloutTrainingBatchView(history=history, num_steps=steps, num_envs=n),
        device=torch.device(getattr(runtime, "device", "cpu")),
    )
    stage_batches = views.training_view.stage_batches
    stage_specs = (
        (0, history.accel_stage, capacity + 1, ("actions", "old_logprobs", "values")),
        (1, history.sat_stage, capacity, ("actions", "old_logprobs", "values")),
        (2, history.bw_stage, capacity, ("actions", "old_logprobs", "values", "rewards")),
    )
    expected_local_batches = {
        0: _accel_local_from_flat_history_ring(
            history.accel_stage,
            num_steps=steps,
            capacity=capacity,
            device=torch.device(getattr(runtime, "device", "cpu")),
        ),
        1: _sat_local_from_flat_history_ring(
            history.sat_stage,
            num_steps=steps,
            capacity=capacity,
            device=torch.device(getattr(runtime, "device", "cpu")),
        ),
        2: _bw_local_from_flat_history_ring(
            history.bw_stage,
            num_steps=steps,
            capacity=capacity,
            device=torch.device(getattr(runtime, "device", "cpu")),
        ),
    }
    for stage_id, hist_stage, world_capacity, tensor_fields in stage_specs:
        batch = stage_batches[int(stage_id)]
        for field_info in fields(batch.world_batch):
            field_name = str(field_info.name)
            if not hasattr(hist_stage.world_batch, field_name):
                continue
            _compare_native_tensor_field(
                errors,
                f"stage{stage_id}.world_batch.{field_name}",
                getattr(batch.world_batch, field_name),
                _native_history_prefix_tensor(
                    getattr(hist_stage.world_batch, field_name),
                    capacity=world_capacity,
                    num_steps=steps,
                    field_name=f"stage{stage_id}.world_batch.{field_name}",
                ),
                atol=atol,
                rtol=rtol,
            )
        _compare_native_dataclass_tensors(
            errors,
            f"stage{stage_id}.local_batch",
            batch.local_batch,
            expected_local_batches[int(stage_id)],
            atol=atol,
            rtol=rtol,
        )
        for tensor_name in tensor_fields:
            _compare_native_tensor_field(
                errors,
                f"stage{stage_id}.{tensor_name}",
                getattr(batch, tensor_name if tensor_name != "old_logprobs" else "old_logprobs"),
                _native_history_prefix_tensor(
                    getattr(hist_stage, tensor_name),
                    capacity=capacity,
                    num_steps=steps,
                    field_name=f"stage{stage_id}.{tensor_name}",
                ),
                atol=atol,
            rtol=rtol,
        )
    bw_batch = stage_batches[2]
    accel_batch = stage_batches[0]
    accel_optional_pairs = (
        ("danger_imitation_targets", "danger_imitation_targets"),
        ("danger_imitation_masks", "danger_imitation_masks"),
    )
    for batch_name, hist_name in accel_optional_pairs:
        _compare_native_tensor_field(
            errors,
            f"stage0.{batch_name}",
            getattr(accel_batch, batch_name),
            _native_history_prefix_tensor(
                getattr(history.accel_stage, hist_name),
                capacity=capacity,
                num_steps=steps,
                field_name=f"history.accel_stage.{hist_name}",
            ),
            atol=atol,
            rtol=rtol,
        )
    optional_pairs = (
        ("bw_ref_actions", "bw_ref_actions"),
        ("old_logprobs_per_agent", "bw_old_logprobs_per_agent"),
        ("bw_flow_proxy_scores", "bw_flow_proxy_scores"),
        ("bw_flow_proxy_masks", "bw_flow_proxy_masks"),
        ("bw_flow_proxy_deltas", "bw_flow_proxy_deltas"),
    )
    for batch_name, hist_name in optional_pairs:
        _compare_native_tensor_field(
            errors,
            f"stage2.{batch_name}",
            getattr(bw_batch, batch_name),
            _native_history_prefix_tensor(
                getattr(history.bw_stage, hist_name),
                capacity=capacity,
                num_steps=steps,
                field_name=f"history.bw_stage.{hist_name}",
            ),
            atol=atol,
            rtol=rtol,
        )
    terminal_mask = _native_history_prefix_tensor(
        history.terminal_next_world_mask,
        capacity=capacity,
        num_steps=steps,
        field_name="terminal_next_world_mask",
    )
    if terminal_mask is None:
        errors.append("terminal_next_world_mask: missing")
    else:
        terminated_prefix = _native_history_prefix_tensor(
            history.terminated,
            capacity=capacity,
            num_steps=steps,
            field_name="terminated",
        )
        truncated_prefix = _native_history_prefix_tensor(
            history.truncated,
            capacity=capacity,
            num_steps=steps,
            field_name="truncated",
        )
        if terminated_prefix is not None and truncated_prefix is not None:
            _compare_native_tensor_field(
                errors,
                "terminal_next_world_mask.done_semantics",
                terminal_mask,
                terminated_prefix.to(dtype=torch.bool) | truncated_prefix.to(dtype=torch.bool),
                atol=atol,
                rtol=rtol,
            )
        terminal_next_world = _world_state_from_flat_history_ring(
            history.terminal_next_world,
            num_steps=steps,
            capacity=capacity,
            stage_id=0,
            device=torch.device(getattr(runtime, "device", "cpu")),
        )
        next_actor_world = _world_state_from_flat_history_ring(
            history.accel_stage.world_batch,
            num_steps=steps,
            capacity=capacity + 1,
            stage_id=0,
            device=torch.device(getattr(runtime, "device", "cpu")),
            start_slot=1,
        )
        expected_next_world = _where_world_state(
            terminal_mask.to(device=next_actor_world.uav_nodes.device, dtype=torch.bool),
            terminal_next_world,
            next_actor_world,
        )
        _compare_native_dataclass_tensors(
            errors,
            "stage2.next_world_batch",
            views.return_view.stage_batches[2].next_world_batch,
            expected_next_world,
            atol=atol,
            rtol=rtol,
        )

    hist_reward_parts_all = getattr(history.bw_stage, "reward_part_tensors", None)
    if isinstance(hist_reward_parts_all, dict):
      stage_reward_arrays = dict(getattr(views.return_view.stage_batches[2], "reward_part_arrays", {}) or {})
      return_reward_arrays = dict(getattr(views.return_view, "reward_part_arrays", {}) or {})
      if set(stage_reward_arrays) != set(hist_reward_parts_all):
          errors.append(
              "return_view.stage2.reward_part_arrays keys: "
              f"actual={sorted(stage_reward_arrays)} expected={sorted(hist_reward_parts_all)}"
          )
      if set(return_reward_arrays) != set(hist_reward_parts_all):
          errors.append(
              "return_view.reward_part_arrays keys: "
              f"actual={sorted(return_reward_arrays)} expected={sorted(hist_reward_parts_all)}"
          )
      for key, tensor in hist_reward_parts_all.items():
          expected_tensor = _native_history_prefix_tensor(
              tensor,
              capacity=capacity,
              num_steps=steps,
              field_name=f"history.bw_stage.reward_part_tensors.{key}",
          )
          if expected_tensor is None:
              continue
          expected_flat = expected_tensor.detach().to(dtype=torch.float32).cpu().reshape(-1)
          actual_stage = stage_reward_arrays.get(str(key))
          _compare_native_tensor_field(
              errors,
              f"return_view.stage2.reward_part_arrays.{key}",
              None if actual_stage is None else torch.as_tensor(actual_stage, dtype=torch.float32),
              expected_flat,
              atol=atol,
              rtol=rtol,
          )
          actual_return = return_reward_arrays.get(str(key))
          expected_return = torch.zeros((expected_flat.numel() * 3,), dtype=torch.float32)
          expected_return[2::3] = expected_flat
          _compare_native_tensor_field(
              errors,
              f"return_view.reward_part_arrays.{key}",
              None if actual_return is None else torch.as_tensor(actual_return, dtype=torch.float32),
              expected_return,
              atol=atol,
              rtol=rtol,
          )
    step_views = list(getattr(runtime.result, "step_result_views", [])[:steps])
    if len(step_views) != steps:
        errors.append(f"step_result_views: count={len(step_views)} steps={steps}")
    else:
        def _cat_view_field(field_name: str) -> torch.Tensor | None:
            values = [getattr(view, field_name, None) for view in step_views]
            if all(value is None for value in values):
                return None
            if any(value is None or not torch.is_tensor(value) for value in values):
                errors.append(f"step_result_views.{field_name}: partial or non-tensor field")
                return None
            return torch.cat([value for value in values if torch.is_tensor(value)], dim=0)

        flat_count = steps * n
        _compare_native_tensor_field(errors, "step_views.team_rewards", _cat_view_field("team_rewards"), history.bw_stage.rewards[:flat_count], atol=atol, rtol=rtol)
        _compare_native_tensor_field(errors, "step_views.terminated", _cat_view_field("terminated"), history.terminated[:flat_count], atol=atol, rtol=rtol)
        _compare_native_tensor_field(errors, "step_views.truncated", _cat_view_field("truncated"), history.truncated[:flat_count], atol=atol, rtol=rtol)
        _compare_native_tensor_field(errors, "step_views.bw_access_rewards", _cat_view_field("bw_access_rewards"), None if history.bw_stage.bw_access_rewards is None else history.bw_stage.bw_access_rewards[:flat_count], atol=atol, rtol=rtol)
        _compare_native_tensor_field(errors, "step_views.bw_weighted_workload_delta_rewards", _cat_view_field("bw_weighted_workload_delta_rewards"), None if history.bw_stage.bw_weighted_workload_delta_rewards is None else history.bw_stage.bw_weighted_workload_delta_rewards[:flat_count], atol=atol, rtol=rtol)
        _compare_native_tensor_field(errors, "step_views.bw_weighted_workload_level_rewards", _cat_view_field("bw_weighted_workload_level_rewards"), None if history.bw_stage.bw_weighted_workload_level_rewards is None else history.bw_stage.bw_weighted_workload_level_rewards[:flat_count], atol=atol, rtol=rtol)
        _compare_native_tensor_field(errors, "step_views.bw_gu_queue_level_rewards", _cat_view_field("bw_gu_queue_level_rewards"), None if history.bw_stage.bw_gu_queue_level_rewards is None else history.bw_stage.bw_gu_queue_level_rewards[:flat_count], atol=atol, rtol=rtol)
        _compare_native_tensor_field(errors, "step_views.bw_system_queue_level_rewards", _cat_view_field("bw_system_queue_level_rewards"), None if history.bw_stage.bw_system_queue_level_rewards is None else history.bw_stage.bw_system_queue_level_rewards[:flat_count], atol=atol, rtol=rtol)
        _compare_native_tensor_field(errors, "step_views.bw_gu_service_queue_rewards", _cat_view_field("bw_gu_service_queue_rewards"), None if history.bw_stage.bw_gu_service_queue_rewards is None else history.bw_stage.bw_gu_service_queue_rewards[:flat_count], atol=atol, rtol=rtol)
        _compare_native_tensor_field(errors, "step_views.danger_imitation_target", _cat_view_field("danger_imitation_target"), None if history.accel_stage.danger_imitation_targets is None else history.accel_stage.danger_imitation_targets[:flat_count], atol=atol, rtol=rtol)
        _compare_native_tensor_field(errors, "step_views.danger_imitation_mask", _cat_view_field("danger_imitation_mask"), None if history.accel_stage.danger_imitation_masks is None else history.accel_stage.danger_imitation_masks[:flat_count], atol=atol, rtol=rtol)
        _compare_native_tensor_field(errors, "step_views.bw_flow_proxy_scores", _cat_view_field("bw_flow_proxy_scores"), None if history.bw_stage.bw_flow_proxy_scores is None else history.bw_stage.bw_flow_proxy_scores[:flat_count], atol=atol, rtol=rtol)
        _compare_native_tensor_field(errors, "step_views.bw_flow_proxy_mask", _cat_view_field("bw_flow_proxy_mask"), None if history.bw_stage.bw_flow_proxy_masks is None else history.bw_stage.bw_flow_proxy_masks[:flat_count], atol=atol, rtol=rtol)
        _compare_native_tensor_field(errors, "step_views.bw_flow_proxy_deltas", _cat_view_field("bw_flow_proxy_deltas"), None if history.bw_stage.bw_flow_proxy_deltas is None else history.bw_stage.bw_flow_proxy_deltas[:flat_count], atol=atol, rtol=rtol)
        hist_reward_parts = getattr(history.bw_stage, "reward_part_tensors", None)
        if isinstance(hist_reward_parts, dict):
            view_part_keys = None
            for view in step_views:
                step_parts = getattr(view, "reward_part_tensors", None)
                if not isinstance(step_parts, dict):
                    errors.append("step_views.reward_part_tensors: missing dict")
                    view_part_keys = set()
                    break
                keys = set(step_parts)
                view_part_keys = keys if view_part_keys is None else view_part_keys & keys
            if view_part_keys is not None and view_part_keys != set(hist_reward_parts):
                errors.append(
                    "step_views.reward_part_tensors: key mismatch "
                    f"actual={sorted(view_part_keys)} expected={sorted(hist_reward_parts)}"
                )
            for key, tensor in hist_reward_parts.items():
                values = [
                    view.reward_part_tensors.get(key)
                    for view in step_views
                    if isinstance(getattr(view, "reward_part_tensors", None), dict)
                ]
                actual = torch.cat(values, dim=0) if values and all(torch.is_tensor(value) for value in values) else None
                _compare_native_tensor_field(
                    errors,
                    f"step_views.reward_part_tensors.{key}",
                    actual,
                    tensor[:flat_count] if torch.is_tensor(tensor) else None,
                    atol=atol,
                    rtol=rtol,
                )
    return errors


def _evaluate_structured_actor_exec_sources_internal(
    cfg,
    actor,
    *,
    device: torch.device,
    episodes: int,
    episode_seed_base: int | None = None,
    deterministic: bool = True,
    num_envs: int = 1,
    vec_backend: str = "sync",
    exec_accel_source: str | None = None,
    exec_sat_source: str | None = None,
    exec_bw_source: str | None = None,
    collect_step_traces: bool = False,
) -> Tuple[Dict[str, float], List[Dict[str, float]], List[List[Dict[str, float]]] | None]:
    active_slots = max(min(int(num_envs), int(episodes)), 1)
    resolved_accel_source = _normalize_exec_source(
        getattr(cfg, "exec_accel_source", "policy") if exec_accel_source is None else exec_accel_source
    )
    resolved_sat_source = _normalize_exec_source(
        getattr(cfg, "exec_sat_source", "policy") if exec_sat_source is None else exec_sat_source
    )
    resolved_bw_source = _normalize_exec_source(
        getattr(cfg, "exec_bw_source", "policy") if exec_bw_source is None else exec_bw_source
    )
    validate_satellite_control_consistency(
        cfg,
        train_sat=resolved_sat_source == "policy",
        exec_sat_source=resolved_sat_source,
        context="evaluate_structured_actor_exec_sources",
    )
    env_group = make_structured_env_group(cfg, num_envs=active_slots, backend=vec_backend, mode="eval")
    drivers = env_group if looks_like_driver_group(env_group) else _as_driver_list(env_group)
    actor.eval()
    learner = StructuredMAPPO(
        actor=actor,
        critic=ZeroStructuredCritic(device),
        gamma=float(getattr(cfg, "gamma", 0.99) or 0.99),
        gae_lambda=float(getattr(cfg, "gae_lambda", 0.95) or 0.95),
        clip_ratio=float(getattr(cfg, "clip_ratio", 0.2) or 0.2),
        value_coef=0.0,
        entropy_coef=0.0,
        max_grad_norm=float(getattr(cfg, "max_grad_norm", 0.5) or 0.5),
        ppo_epochs=1,
        num_mini_batch=1,
        actor_optimizer=None,
        critic_optimizer=None,
        device=device,
        target_mode="step_level",
        cfg=cfg,
        train_accel=resolved_accel_source == "policy",
        train_sat=resolved_sat_source == "policy",
        train_bw=resolved_bw_source == "policy",
        exec_accel_source=resolved_accel_source,
        exec_sat_source=resolved_sat_source,
        exec_bw_source=resolved_bw_source,
    )
    rows: List[Dict[str, float]] = []
    episode_traces: Dict[int, List[Dict[str, float]]] = {}
    horizon = max(int(getattr(cfg, "T_steps", 1) or 1), 1)
    episode_cursor = 0
    try:
        while episode_cursor < int(episodes):
            batch_episodes = min(active_slots, int(episodes) - int(episode_cursor))
            seeds = [
                None
                if episode_seed_base is None
                else int(episode_seed_base) + int(episode_cursor) + slot
                for slot in range(active_slots)
            ]
            reset_many(drivers, seeds)
            slot_step = [0 for _ in range(active_slots)]
            slot_done = [False for _ in range(active_slots)]
            slot_acc = [_new_episode_accumulator(cfg) for _ in range(active_slots)]
            learner.begin_native_rollout(
                drivers,
                rollout_env_steps=horizon,
                num_envs=active_slots,
            )
            horizon_results = learner.collect_env_horizon_native_tensor_policy(
                drivers,
                None,
                horizon=horizon,
                deterministic=deterministic,
            )
            if not horizon_results:
                raise RuntimeError("native actor evaluation produced no horizon results.")
            for step_results in horizon_results:
                batch_step_arrays = (
                    _batch_step_result_arrays(step_results)
                    if isinstance(step_results, StructuredBatchStepResult)
                    else None
                )
                step_count = (
                    int(step_results.num_envs)
                    if isinstance(step_results, StructuredBatchStepResult)
                    else len(step_results)
                )
                for slot in range(min(step_count, active_slots)):
                    if slot >= batch_episodes or slot_done[slot]:
                        continue
                    step_result = step_results if batch_step_arrays is not None else step_results[slot]
                    reward_parts = (
                        _batch_reward_parts_at(batch_step_arrays, slot)
                        if batch_step_arrays is not None
                        else dict(getattr(step_result, "reward_parts", {}) or {})
                    )
                    acc = slot_acc[slot]
                    if batch_step_arrays is None:
                        reward_value = _step_team_reward(step_result)
                        bw_delta_value = _batch_step_scalar(step_result, "bw_weighted_workload_delta_reward")
                        bw_level_value = _batch_step_scalar(step_result, "bw_weighted_workload_level_reward")
                    else:
                        reward_value = float(batch_step_arrays["team_reward"][slot])
                        bw_delta_value = float(batch_step_arrays["bw_weighted_workload_delta_reward"][slot])
                        bw_level_value = float(batch_step_arrays["bw_weighted_workload_level_reward"][slot])
                    _accumulate_step_metrics(
                        acc,
                        reward_value=reward_value,
                        bw_delta_value=bw_delta_value,
                        bw_level_value=bw_level_value,
                        reward_parts=reward_parts,
                    )
                    if collect_step_traces:
                        episode_index = int(episode_cursor) + int(slot)
                        episode_traces.setdefault(episode_index, []).append(
                            _step_trace_row(
                                episode_index=episode_index,
                                step_index=int(slot_step[slot]),
                                step_result=step_result,
                                reward_parts=reward_parts,
                                runtime_trace={},
                            )
                        )
                    slot_step[slot] += 1

                    done = (
                        bool(batch_step_arrays["terminated"][slot] or batch_step_arrays["truncated"][slot])
                        if batch_step_arrays is not None
                        else _step_done(step_result)
                    )
                    if done:
                        rows.append(_episode_row_from_accumulator(int(episode_cursor) + int(slot), slot_acc[slot]))
                        slot_done[slot] = True
            for slot in range(batch_episodes):
                if not slot_done[slot]:
                    rows.append(_episode_row_from_accumulator(int(episode_cursor) + int(slot), slot_acc[slot]))
            episode_cursor += int(batch_episodes)
        rows.sort(key=lambda row: int(row["episode"]))
        traces = None
        if collect_step_traces:
            traces = [episode_traces.get(ep, []) for ep in range(int(episodes))]
        return _summary_from_rows(rows, episodes=int(episodes)), rows, traces
    finally:
        close_structured_env_group(env_group)


def _evaluate_structured_actor_group(
    cfg,
    actor,
    *,
    device: torch.device,
    episodes: int,
    episode_seed_base: int | None = None,
    deterministic: bool = True,
    active_slots: int,
    vec_backend: str,
) -> Tuple[Dict[str, float], List[Dict[str, float]]]:
    return evaluate_structured_actor_exec_sources(
        cfg,
        actor,
        device=device,
        episodes=episodes,
        episode_seed_base=episode_seed_base,
        deterministic=deterministic,
        num_envs=active_slots,
        vec_backend=vec_backend,
        exec_accel_source="policy",
        exec_sat_source="policy",
        exec_bw_source="policy",
    )


def evaluate_structured_actor_parallel(
    cfg,
    actor,
    *,
    device: torch.device,
    episodes: int,
    episode_seed_base: int | None = None,
    deterministic: bool = True,
    num_envs: int = 8,
    vec_backend: str = "sync",
) -> Tuple[Dict[str, float], List[Dict[str, float]]]:
    active_slots = max(min(int(num_envs), int(episodes)), 1)
    return _evaluate_structured_actor_group(
        cfg,
        actor,
        device=device,
        episodes=episodes,
        episode_seed_base=episode_seed_base,
        deterministic=deterministic,
        active_slots=active_slots,
        vec_backend=vec_backend,
    )


def evaluate_structured_actor(
    cfg,
    actor,
    *,
    device: torch.device,
    episodes: int,
    episode_seed_base: int | None = None,
    deterministic: bool = True,
    num_envs: int = 1,
    vec_backend: str = "sync",
) -> Tuple[Dict[str, float], List[Dict[str, float]]]:
    active_slots = max(min(int(num_envs), int(episodes)), 1)
    return _evaluate_structured_actor_group(
        cfg,
        actor,
        device=device,
        episodes=episodes,
        episode_seed_base=episode_seed_base,
        deterministic=deterministic,
        active_slots=active_slots,
        vec_backend=vec_backend,
    )


def evaluate_structured_actor_exec_sources(
    cfg,
    actor,
    *,
    device: torch.device,
    episodes: int,
    episode_seed_base: int | None = None,
    deterministic: bool = True,
    num_envs: int = 1,
    vec_backend: str = "sync",
    exec_accel_source: str | None = None,
    exec_sat_source: str | None = None,
    exec_bw_source: str | None = None,
) -> Tuple[Dict[str, float], List[Dict[str, float]]]:
    summary, rows, _ = _evaluate_structured_actor_exec_sources_internal(
        cfg,
        actor,
        device=device,
        episodes=episodes,
        episode_seed_base=episode_seed_base,
        deterministic=deterministic,
        num_envs=num_envs,
        vec_backend=vec_backend,
        exec_accel_source=exec_accel_source,
        exec_sat_source=exec_sat_source,
        exec_bw_source=exec_bw_source,
        collect_step_traces=False,
    )
    return summary, rows


def evaluate_structured_actor_exec_sources_with_traces(
    cfg,
    actor,
    *,
    device: torch.device,
    episodes: int,
    episode_seed_base: int | None = None,
    deterministic: bool = True,
    num_envs: int = 1,
    vec_backend: str = "sync",
    exec_accel_source: str | None = None,
    exec_sat_source: str | None = None,
    exec_bw_source: str | None = None,
) -> Tuple[Dict[str, float], List[Dict[str, float]], List[List[Dict[str, float]]]]:
    summary, rows, traces = _evaluate_structured_actor_exec_sources_internal(
        cfg,
        actor,
        device=device,
        episodes=episodes,
        episode_seed_base=episode_seed_base,
        deterministic=deterministic,
        num_envs=num_envs,
        vec_backend=vec_backend,
        exec_accel_source=exec_accel_source,
        exec_sat_source=exec_sat_source,
        exec_bw_source=exec_bw_source,
        collect_step_traces=True,
    )
    return summary, rows, traces or []


def _clone_cfg_with_structured_backend(cfg, *, structured_env_backend: str, structured_env_tensor_backend: str):
    cloned = copy.deepcopy(cfg)
    cloned.structured_env_backend = str(structured_env_backend)
    cloned.structured_env_tensor_backend = str(structured_env_tensor_backend)
    return cloned


def _max_abs_update(bucket: Dict[str, float], key: str, a: float, b: float) -> None:
    bucket[key] = max(float(bucket.get(key, 0.0)), abs(float(a) - float(b)))


def _evaluate_structured_baseline_policy_with_traces(
    cfg,
    *,
    baseline_policy: str,
    episodes: int,
    episode_seed_base: int | None = None,
    random_tape_cfg=None,
    exec_sources: Sequence[str] | None = None,
) -> Tuple[
    Dict[str, float],
    List[Dict[str, float]],
    List[List[Dict[str, float]]],
    List[List[Dict[str, np.ndarray]]],
    List[Dict[str, Any] | None],
]:
    source_modes = _acceptance_source_modes(exec_sources)
    rows: List[Dict[str, float]] = []
    traces: List[List[Dict[str, float]]] = []
    action_rollouts: List[List[Dict[str, np.ndarray]]] = []
    reset_rollouts: List[Dict[str, Any] | None] = []
    env_group = make_structured_env_group(cfg, num_envs=1, backend="sync", mode="eval")
    random_tape_group = (
        None
        if random_tape_cfg is None
        else make_structured_env_group(random_tape_cfg, num_envs=1, backend="sync", mode="eval")
    )
    try:
        drivers = env_group if looks_like_driver_group(env_group) else as_structured_drivers(env_group)
        _bind_native_eval_tensor_device(drivers, cfg)
        if random_tape_group is not None:
            _bind_native_eval_tensor_device(random_tape_group, random_tape_cfg)
        driver = drivers[0]
        for ep in range(int(episodes)):
            seed = None if episode_seed_base is None else int(episode_seed_base) + ep
            reset_state_tape = None
            if random_tape_group is not None:
                reset_many(random_tape_group, [None if seed is None else int(seed)])
                export_reset = getattr(random_tape_group, "export_runtime_state_batch", None)
                if not callable(export_reset):
                    raise RuntimeError("native random tape group requires runtime-state export.")
                reset_state_tape = export_reset(indices=[0])[0]
                load_states = getattr(drivers, "load_runtime_state_batch", None)
                if callable(load_states):
                    load_states(
                        [reset_state_tape],
                        indices=[0],
                        refresh_observation_cache=False,
                        refresh_global_state_cache=True,
                    )
                else:
                    load_driver_state = getattr(driver, "load_runtime_state", None)
                    if not callable(load_driver_state):
                        load_driver_state = getattr(driver.env, "load_runtime_state")
                    load_driver_state(
                        reset_state_tape,
                        refresh_observation_cache=False,
                        refresh_global_state_cache=True,
                    )
                initial_stage_tape = None
                begin_tape_rollout = getattr(random_tape_group, "begin_native_main_kernel_rollout", None)
                if callable(begin_tape_rollout):
                    begin_tape_rollout(capacity=max(int(getattr(cfg, "T_steps", 1) or 1), 1), num_envs=1)
                tape_program = _NativeMainKernelActionReplayProgram(
                    random_tape_group,
                    cfg,
                    exec_sources=source_modes,
                )
                tape_runtime = getattr(tape_program, "runtime", None)
                if tape_runtime is not None and tape_program.available:
                    tape_program.begin_horizon(num_steps=max(int(getattr(cfg, "T_steps", 1) or 1), 1))
                    tape_program.publish_accel_obs()
                    initial_stage_tape = tape_runtime.main.accel_stage_fields
                    if initial_stage_tape is None:
                        initial_stage_tape = tape_runtime.main.accel_stage
                    if initial_stage_tape is not None and hasattr(initial_stage_tape, "__bool__") and not hasattr(initial_stage_tape, "_fields") and not bool(initial_stage_tape):
                        initial_stage_tape = None
                if initial_stage_tape is None:
                    raise RuntimeError("native random tape group must publish an initial main-kernel accel stage fields.")
                _apply_reference_accel_cache_from_native_stage_batch(driver, initial_stage_tape, cfg)
            else:
                reset_many(drivers, [None if seed is None else int(seed)])
            clear_step = getattr(driver, "_clear_step", None)
            if callable(clear_step):
                clear_step()
            reset_rollouts.append(copy.deepcopy(reset_state_tape))
            baseline_state = None
            obs_list = current_obs_many([driver], indices=[0])[0]
            done = False
            step_index = 0
            acc = _new_episode_accumulator(cfg)
            episode_trace: List[Dict[str, float]] = []
            episode_actions: List[Dict[str, np.ndarray]] = []
            while not done:
                step_result, baseline_state, action_trace = _run_structured_baseline_step_with_actions(
                    baseline_policy,
                    driver,
                    cfg,
                    obs_list,
                    baseline_state=baseline_state,
                    random_tape_group=random_tape_group,
                    exec_sources=source_modes,
                )
                episode_actions.append(action_trace)
                obs_list = list(step_result.obs.values())
                reward_parts = dict(getattr(step_result, "reward_parts", {}) or {})
                if not reward_parts:
                    reward_parts = dict(last_reward_parts_many([driver])[0] or {})
                runtime_trace = _runtime_state_trace_rows([driver], indices=[0])[0]
                episode_trace.append(
                    _step_trace_row(
                        episode_index=ep,
                        step_index=step_index,
                        step_result=step_result,
                        reward_parts=reward_parts,
                        runtime_trace=runtime_trace,
                    )
                )
                _accumulate_step_metrics(
                    acc,
                    reward_value=_step_team_reward(step_result),
                    bw_delta_value=_batch_step_scalar(step_result, "bw_weighted_workload_delta_reward"),
                    bw_level_value=_batch_step_scalar(step_result, "bw_weighted_workload_level_reward"),
                    reward_parts=reward_parts,
                    runtime_trace=runtime_trace,
                )
                done = _step_done(step_result)
                step_index += 1
            rows.append(_episode_row_from_accumulator(ep, acc))
            traces.append(episode_trace)
            action_rollouts.append(episode_actions)
    finally:
        close_structured_env_group(env_group)
        if random_tape_group is not None:
            close_structured_env_group(random_tape_group)
    return _summary_from_rows(rows, episodes=int(episodes)), rows, traces, action_rollouts, reset_rollouts


def _evaluate_structured_action_replay_with_traces(
    cfg,
    *,
    action_rollouts: List[List[Dict[str, np.ndarray]]],
    reset_rollouts: List[Dict[str, Any] | None] | None = None,
    episode_seed_base: int | None = None,
    vec_backend: str = "sync",
    field_contract_atol: float | None = None,
    field_contract_rtol: float | None = None,
    exec_sources: Sequence[str] | None = None,
) -> Tuple[Dict[str, float], List[Dict[str, float]], List[List[Dict[str, float]]]] | Tuple[
    Dict[str, float],
    List[Dict[str, float]],
    List[List[Dict[str, float]]],
    List[str],
]:
    rows: List[Dict[str, float]] = []
    traces: List[List[Dict[str, float]]] = []
    field_contract_errors: List[str] = []
    source_modes = _acceptance_source_modes(exec_sources)
    env_group = make_structured_env_group(cfg, num_envs=1, backend=vec_backend, mode="eval")
    drivers = env_group if looks_like_driver_group(env_group) else _as_driver_list(env_group)
    _bind_native_eval_tensor_device(drivers, cfg)
    try:
        for ep, episode_actions in enumerate(action_rollouts):
            reset_state_tape = (
                None
                if reset_rollouts is None or ep >= len(reset_rollouts)
                else reset_rollouts[ep]
            )
            seed = None if episode_seed_base is None else int(episode_seed_base) + ep
            if reset_state_tape is not None:
                # Match the random-tape generator position used to create the
                # reset tape, then overwrite the runtime state with the exact
                # recorded reset payload.
                reset_many(drivers, [None if seed is None else int(seed)])
                clear_native_kernel = getattr(drivers, "clear_native_main_kernel", None)
                if callable(clear_native_kernel):
                    clear_native_kernel()
                load_states = getattr(drivers, "load_runtime_state_batch", None)
                if callable(load_states):
                    load_states(
                        [copy.deepcopy(reset_state_tape)],
                        indices=[0],
                        refresh_observation_cache=not looks_like_driver_group(drivers),
                        refresh_global_state_cache=True,
                    )
                else:
                    driver_list = _as_driver_list(drivers) if looks_like_driver_group(drivers) else list(drivers)
                    driver_list[0].env.load_runtime_state(
                        copy.deepcopy(reset_state_tape),
                        refresh_observation_cache=True,
                        refresh_global_state_cache=True,
                    )
            else:
                reset_many(drivers, [None if seed is None else int(seed)])
            begin_replay_rollout = getattr(drivers, "begin_native_main_kernel_rollout", None)
            if callable(begin_replay_rollout):
                begin_replay_rollout(capacity=max(len(episode_actions), 1), num_envs=1)
            if not looks_like_driver_group(drivers):
                current_obs_many(drivers, indices=[0])
            acc = _new_episode_accumulator(cfg)
            episode_trace: List[Dict[str, float]] = []
            replay_runtime = None

            replay_program = _NativeMainKernelActionReplayProgram(
                drivers,
                cfg,
                capacity=max(len(episode_actions), 1),
                exec_sources=source_modes,
            )
            if not replay_program.available:
                raise RuntimeError("structured action replay requires native main-kernel BW result APIs.")

            replay_program.replay_action_horizon(
                episode_actions,
                deterministic=True,
                step_callback=None,
            )
            replay_runtime = replay_program.runtime
            native_row, episode_trace = _native_replay_rows_and_traces_from_history(
                replay_runtime,
                episode_index=ep,
                num_steps=len(episode_actions),
                num_envs=1,
                cfg=cfg,
            )
            rows.append(native_row)
            traces.append(episode_trace)
            if field_contract_atol is not None and replay_runtime is not None:
                _compare_expected_step_payloads_from_history(
                    field_contract_errors,
                    context=f"episode={ep} step_result_payload",
                    runtime=replay_runtime,
                    action_traces=episode_actions,
                    atol=float(field_contract_atol),
                    rtol=float(0.0 if field_contract_rtol is None else field_contract_rtol),
                )
                episode_errors = _native_rollout_field_contract_errors(
                    replay_runtime,
                    num_steps=len(episode_trace),
                    num_envs=1,
                    atol=float(field_contract_atol),
                    rtol=float(0.0 if field_contract_rtol is None else field_contract_rtol),
                )
                field_contract_errors.extend(f"episode={ep} {error}" for error in episode_errors)
    finally:
        close_structured_env_group(env_group)
    result = (_summary_from_rows(rows, episodes=int(len(action_rollouts))), rows, traces)
    if field_contract_atol is not None:
        return (*result, field_contract_errors)
    return result


def _structured_long_rollout_acceptance_thresholds(cfg) -> Dict[str, Dict[str, Dict[str, float]]]:
    def _threshold(attr: str, default: float) -> float:
        value = getattr(cfg, attr, default)
        return float(default if value is None else value)

    return {
        "episode": {
            "reward_sum": {"abs": _threshold("long_rollout_episode_reward_abs_tol", 5.0e-2), "rel": _threshold("long_rollout_episode_reward_rel_tol", 1.0e-4)},
            "step_count": {"abs": 0.0, "rel": 0.0},
            "processed_ratio_total": {"abs": _threshold("long_rollout_episode_processed_abs_tol", 1.0e-2), "rel": _threshold("long_rollout_episode_processed_rel_tol", 5.0e-5)},
            "drop_ratio_total": {"abs": _threshold("long_rollout_episode_drop_abs_tol", 5.0e-4), "rel": _threshold("long_rollout_episode_drop_rel_tol", 1.0e-5)},
            "pre_backlog_total": {"abs": _threshold("long_rollout_episode_backlog_abs_tol", 5.0), "rel": _threshold("long_rollout_episode_backlog_rel_tol", 5.0e-3)},
            "D_sys_total": {"abs": _threshold("long_rollout_episode_dsys_abs_tol", 10.0), "rel": _threshold("long_rollout_episode_dsys_rel_tol", 5.0e-3)},
            "processed_ratio_eval": {"abs": _threshold("long_rollout_episode_processed_eval_abs_tol", 5.0e-5), "rel": _threshold("long_rollout_episode_processed_eval_rel_tol", 5.0e-5)},
            "drop_ratio_eval": {"abs": _threshold("long_rollout_episode_drop_eval_abs_tol", 1.0e-4), "rel": _threshold("long_rollout_episode_drop_eval_rel_tol", 1.0e-5)},
            "pre_backlog_steps_eval": {"abs": _threshold("long_rollout_episode_backlog_eval_abs_tol", 2.0e-2), "rel": _threshold("long_rollout_episode_backlog_eval_rel_tol", 5.0e-3)},
            "D_sys_report": {"abs": _threshold("long_rollout_episode_dsys_report_abs_tol", 5.0e-2), "rel": _threshold("long_rollout_episode_dsys_report_rel_tol", 5.0e-3)},
            "x_acc_mean": {"abs": _threshold("long_rollout_episode_x_acc_abs_tol", 5.0e-5), "rel": _threshold("long_rollout_episode_x_acc_rel_tol", 5.0e-5)},
            "x_rel_mean": {"abs": _threshold("long_rollout_episode_x_rel_abs_tol", 5.0e-5), "rel": _threshold("long_rollout_episode_x_rel_rel_tol", 5.0e-5)},
            "g_pre_mean": {"abs": _threshold("long_rollout_episode_g_pre_abs_tol", 5.0e-5), "rel": _threshold("long_rollout_episode_g_pre_rel_tol", 5.0e-5)},
            "d_pre_mean": {"abs": _threshold("long_rollout_episode_d_pre_abs_tol", 1.0e-5), "rel": _threshold("long_rollout_episode_d_pre_rel_tol", 1.0e-4)},
            "sat_overlap_eval": {"abs": _threshold("long_rollout_episode_sat_overlap_abs_tol", 1.0e-6), "rel": 0.0},
            "collision_episode_fraction": {"abs": _threshold("long_rollout_episode_collision_abs_tol", 1.0e-6), "rel": 0.0},
        },
        "trace": {
            "reward": {"abs": _threshold("long_rollout_trace_reward_abs_tol", 3.5e-1), "rel": 0.0},
            "processed_ratio_eval": {"abs": _threshold("long_rollout_trace_processed_abs_tol", 3.0e-1), "rel": 0.0},
            "drop_ratio_eval": {"abs": _threshold("long_rollout_trace_drop_abs_tol", 1.0e-3), "rel": 0.0},
            "pre_backlog_steps_eval": {"abs": _threshold("long_rollout_trace_backlog_abs_tol", 5.0e-1), "rel": 0.0},
            "D_sys_report": {"abs": _threshold("long_rollout_trace_dsys_abs_tol", 1.2e1), "rel": 0.0},
            "x_acc": {"abs": _threshold("long_rollout_trace_x_acc_abs_tol", 3.0e-1), "rel": 0.0},
            "x_rel": {"abs": _threshold("long_rollout_trace_x_rel_abs_tol", 3.0e-1), "rel": 0.0},
            "g_pre": {"abs": _threshold("long_rollout_trace_g_pre_abs_tol", 3.0e-1), "rel": 0.0},
            "d_pre": {"abs": _threshold("long_rollout_trace_d_pre_abs_tol", 1.0e-3), "rel": 0.0},
            "queue_total_sum": {"abs": _threshold("long_rollout_trace_queue_total_abs_tol", 5.0e6), "rel": _threshold("long_rollout_trace_queue_total_rel_tol", 8.0e-2)},
            "gu_queue_sum": {"abs": _threshold("long_rollout_trace_gu_queue_abs_tol", 5.0e6), "rel": _threshold("long_rollout_trace_gu_queue_rel_tol", 8.0e-2)},
            "uav_queue_sum": {"abs": _threshold("long_rollout_trace_uav_queue_abs_tol", 2.0e6), "rel": _threshold("long_rollout_trace_uav_queue_rel_tol", 1.5e-1)},
            "sat_queue_sum": {"abs": _threshold("long_rollout_trace_sat_queue_abs_tol", 1.0e6), "rel": _threshold("long_rollout_trace_sat_queue_rel_tol", 8.0e-2)},
            "t": {"abs": 0.0, "rel": 0.0},
            "done": {"abs": 0.0, "rel": 0.0},
            "terminated": {"abs": 0.0, "rel": 0.0},
            "truncated": {"abs": 0.0, "rel": 0.0},
        },
        "summary": {
            "reward_sum": {"abs": _threshold("long_rollout_summary_reward_abs_tol", 5.0e-2), "rel": _threshold("long_rollout_summary_reward_rel_tol", 1.0e-4)},
            "processed_ratio_eval": {"abs": _threshold("long_rollout_summary_processed_abs_tol", 1.0e-3), "rel": _threshold("long_rollout_summary_processed_rel_tol", 5.0e-5)},
            "drop_ratio_eval": {"abs": _threshold("long_rollout_summary_drop_abs_tol", 1.0e-4), "rel": _threshold("long_rollout_summary_drop_rel_tol", 1.0e-5)},
            "pre_backlog_steps_eval": {"abs": _threshold("long_rollout_summary_backlog_abs_tol", 2.0e-2), "rel": _threshold("long_rollout_summary_backlog_rel_tol", 5.0e-3)},
            "D_sys_report": {"abs": _threshold("long_rollout_summary_dsys_abs_tol", 5.0e-2), "rel": _threshold("long_rollout_summary_dsys_rel_tol", 5.0e-3)},
            "x_acc_mean": {"abs": _threshold("long_rollout_summary_x_acc_abs_tol", 5.0e-5), "rel": _threshold("long_rollout_summary_x_acc_rel_tol", 5.0e-5)},
            "x_rel_mean": {"abs": _threshold("long_rollout_summary_x_rel_abs_tol", 5.0e-5), "rel": _threshold("long_rollout_summary_x_rel_rel_tol", 5.0e-5)},
            "g_pre_mean": {"abs": _threshold("long_rollout_summary_g_pre_abs_tol", 5.0e-5), "rel": _threshold("long_rollout_summary_g_pre_rel_tol", 5.0e-5)},
            "d_pre_mean": {"abs": _threshold("long_rollout_summary_d_pre_abs_tol", 1.0e-5), "rel": _threshold("long_rollout_summary_d_pre_rel_tol", 1.0e-4)},
            "sat_overlap_eval": {"abs": _threshold("long_rollout_summary_sat_overlap_abs_tol", 1.0e-6), "rel": 0.0},
            "collision_episode_fraction": {"abs": _threshold("long_rollout_summary_collision_abs_tol", 1.0e-6), "rel": 0.0},
        },
    }


def _structured_long_rollout_within_threshold(
    native_value: float,
    legacy_value: float,
    *,
    abs_tol: float,
    rel_tol: float,
) -> bool:
    allowed = max(float(abs_tol), float(rel_tol) * max(abs(float(legacy_value)), 1.0))
    return abs(float(native_value) - float(legacy_value)) <= allowed


def _long_rollout_trace_coverage(traces: Sequence[Sequence[Dict[str, float]]]) -> Dict[str, bool]:
    coverage = {
        "no_done_row": False,
        "terminated_row": False,
        "truncated_row": False,
        "rollout_tail_row": False,
        "episode_done_and_rollout_tail_row": False,
    }
    for trace in traces:
        last_idx = len(trace) - 1
        for idx, step in enumerate(trace):
            terminated = bool(float(step.get("terminated", 0.0)))
            truncated = bool(float(step.get("truncated", 0.0)))
            done = bool(float(step.get("done", 1.0 if terminated or truncated else 0.0)))
            tail = int(idx) == int(last_idx) and last_idx >= 0
            coverage["no_done_row"] = coverage["no_done_row"] or not done
            coverage["terminated_row"] = coverage["terminated_row"] or terminated
            coverage["truncated_row"] = coverage["truncated_row"] or truncated
            coverage["rollout_tail_row"] = coverage["rollout_tail_row"] or tail
            coverage["episode_done_and_rollout_tail_row"] = (
                coverage["episode_done_and_rollout_tail_row"] or (done and tail)
            )
    return coverage


def validate_structured_fixed_seed_long_rollout(
    cfg,
    *,
    baseline_policy: str,
    episodes: int,
    episode_seed_base: int,
    num_envs: int = 1,
    vec_backend: str = "sync",
    native_tensor_backend: str | None = None,
    atol: float = 6.0e-5,
    rtol: float = 1.0e-6,
    exec_sources: Sequence[str] | None = None,
    precomputed_legacy_eval_result: Any | None = None,
) -> Dict[str, Any]:
    fixed_sources = _fixed_policy_exec_sources(baseline_policy)
    if fixed_sources is None:
        raise ValueError(
            "validate_structured_fixed_seed_long_rollout only supports fixed policies that map to exec sources."
        )
    source_modes = _acceptance_source_modes(fixed_sources if exec_sources is None else exec_sources)
    native_tensor_backend_value = (
        str(native_tensor_backend)
        if native_tensor_backend is not None
        else ("cuda" if torch.cuda.is_available() else "cpu")
    )
    native_cfg = _clone_cfg_with_structured_backend(
        cfg,
        structured_env_backend="native",
        structured_env_tensor_backend=native_tensor_backend_value,
    )
    legacy_cfg = _clone_cfg_with_structured_backend(
        cfg,
        structured_env_backend="legacy",
        structured_env_tensor_backend=native_tensor_backend_value,
    )
    legacy_cfg.structured_native_main_kernel_require_compiled_segments = False
    legacy_cfg.structured_native_main_kernel_required_compile_names = ""
    random_tape_cfg = copy.deepcopy(native_cfg)
    random_tape_cfg.structured_native_main_kernel_require_compiled_segments = False
    random_tape_cfg.structured_native_main_kernel_required_compile_names = ""
    legacy_eval_result = precomputed_legacy_eval_result
    if legacy_eval_result is None:
        legacy_eval_result = _evaluate_structured_baseline_policy_with_traces(
            legacy_cfg,
            baseline_policy=str(baseline_policy),
            episodes=episodes,
            episode_seed_base=episode_seed_base,
            random_tape_cfg=random_tape_cfg,
            exec_sources=source_modes,
        )
    if len(legacy_eval_result) == 5:
        legacy_summary, legacy_rows, legacy_traces, legacy_action_rollouts, legacy_reset_rollouts = legacy_eval_result
    elif len(legacy_eval_result) == 4:
        legacy_summary, legacy_rows, legacy_traces, legacy_action_rollouts = legacy_eval_result
        legacy_reset_rollouts = None
    else:
        raise RuntimeError("structured baseline trace evaluation returned an unexpected result shape.")
    native_eval_result = _evaluate_structured_action_replay_with_traces(
        native_cfg,
        action_rollouts=legacy_action_rollouts,
        reset_rollouts=legacy_reset_rollouts,
        episode_seed_base=episode_seed_base,
        vec_backend=vec_backend,
        field_contract_atol=float(atol),
        field_contract_rtol=float(rtol),
        exec_sources=source_modes,
    )
    if len(native_eval_result) == 4:
        native_summary, native_rows, native_traces, native_field_contract_errors = native_eval_result
    else:
        native_summary, native_rows, native_traces = native_eval_result
        native_field_contract_errors = []

    episode_fields = (
        "reward_sum",
        "step_count",
        "processed_ratio_total",
        "drop_ratio_total",
        "pre_backlog_total",
        "D_sys_total",
        "processed_ratio_eval",
        "drop_ratio_eval",
        "pre_backlog_steps_eval",
        "D_sys_report",
        "x_acc_mean",
        "x_rel_mean",
        "g_pre_mean",
        "d_pre_mean",
        "sat_overlap_eval",
        "collision_episode_fraction",
    )
    step_fields = (
        "reward",
        "processed_ratio_eval",
        "drop_ratio_eval",
        "pre_backlog_steps_eval",
        "D_sys_report",
        "x_acc",
        "x_rel",
        "g_pre",
        "d_pre",
        "sat_overlap_eval",
        "collision_event",
        "t",
        "gu_queue_sum",
        "uav_queue_sum",
        "sat_queue_sum",
        "queue_total_sum",
        "terminated",
        "truncated",
        "done",
    )
    summary_fields = (
        "reward_sum",
        "processed_ratio_eval",
        "drop_ratio_eval",
        "pre_backlog_steps_eval",
        "D_sys_report",
        "x_acc_mean",
        "x_rel_mean",
        "g_pre_mean",
        "d_pre_mean",
        "sat_overlap_eval",
        "collision_episode_fraction",
    )

    episode_diff_max: Dict[str, float] = {}
    trace_diff_max: Dict[str, float] = {}
    summary_diff_max: Dict[str, float] = {}
    exact_matches = True
    comparison_errors: List[str] = []
    acceptance_thresholds = _structured_long_rollout_acceptance_thresholds(cfg)
    system_acceptance_passed = True
    system_acceptance_errors: List[str] = []
    if native_field_contract_errors:
        exact_matches = False
        system_acceptance_passed = False
        system_acceptance_errors.extend(native_field_contract_errors)

    if len(native_rows) != len(legacy_rows):
        comparison_errors.append(f"episode_count native={len(native_rows)} legacy={len(legacy_rows)}")
        exact_matches = False
    if len(native_traces) != len(legacy_traces):
        comparison_errors.append(f"trace_episode_count native={len(native_traces)} legacy={len(legacy_traces)}")
        exact_matches = False

    for native_row, legacy_row in zip(native_rows, legacy_rows):
        if int(native_row["episode"]) != int(legacy_row["episode"]):
            exact_matches = False
            comparison_errors.append(
                f"episode_index native={int(native_row['episode'])} legacy={int(legacy_row['episode'])}"
            )
        for field in episode_fields:
            _max_abs_update(episode_diff_max, field, float(native_row[field]), float(legacy_row[field]))
            if not math.isclose(
                float(native_row[field]),
                float(legacy_row[field]),
                rel_tol=float(rtol),
                abs_tol=float(atol),
            ):
                exact_matches = False
                comparison_errors.append(
                    f"episode_field {field} ep={int(native_row['episode'])} "
                    f"native={float(native_row[field]):.8f} legacy={float(legacy_row[field]):.8f}"
                )
            threshold = acceptance_thresholds["episode"].get(field)
            if threshold is not None and not _structured_long_rollout_within_threshold(
                float(native_row[field]),
                float(legacy_row[field]),
                abs_tol=float(threshold["abs"]),
                rel_tol=float(threshold["rel"]),
            ):
                system_acceptance_passed = False
                system_acceptance_errors.append(
                    f"episode_field {field} ep={int(native_row['episode'])} "
                    f"diff={abs(float(native_row[field]) - float(legacy_row[field])):.8f} "
                    f"abs_tol={float(threshold['abs']):.8f} rel_tol={float(threshold['rel']):.8f}"
                )

    for native_trace, legacy_trace in zip(native_traces, legacy_traces):
        if len(native_trace) != len(legacy_trace):
            exact_matches = False
            comparison_errors.append(
                f"trace_length native={len(native_trace)} legacy={len(legacy_trace)}"
            )
        for native_step, legacy_step in zip(native_trace, legacy_trace):
            if int(native_step["step"]) != int(legacy_step["step"]):
                exact_matches = False
                comparison_errors.append(
                    f"trace_step episode={int(native_step['episode'])} "
                    f"native={int(native_step['step'])} legacy={int(legacy_step['step'])}"
                )
            for field in step_fields:
                _max_abs_update(trace_diff_max, field, float(native_step[field]), float(legacy_step[field]))
                if not math.isclose(
                    float(native_step[field]),
                    float(legacy_step[field]),
                    rel_tol=float(rtol),
                    abs_tol=float(atol),
                ):
                    exact_matches = False
                    comparison_errors.append(
                        f"trace_field {field} episode={int(native_step['episode'])} step={int(native_step['step'])} "
                        f"native={float(native_step[field]):.8f} legacy={float(legacy_step[field]):.8f}"
                    )
                threshold = acceptance_thresholds["trace"].get(field)
                if threshold is not None and not _structured_long_rollout_within_threshold(
                    float(native_step[field]),
                    float(legacy_step[field]),
                    abs_tol=float(threshold["abs"]),
                    rel_tol=float(threshold["rel"]),
                ):
                    system_acceptance_passed = False
                    system_acceptance_errors.append(
                        f"trace_field {field} episode={int(native_step['episode'])} step={int(native_step['step'])} "
                        f"diff={abs(float(native_step[field]) - float(legacy_step[field])):.8f} "
                        f"abs_tol={float(threshold['abs']):.8f} rel_tol={float(threshold['rel']):.8f}"
                    )

    for field in summary_fields:
        _max_abs_update(summary_diff_max, field, float(native_summary[field]), float(legacy_summary[field]))
        if not math.isclose(
            float(native_summary[field]),
            float(legacy_summary[field]),
            rel_tol=float(rtol),
            abs_tol=float(atol),
        ):
            exact_matches = False
            comparison_errors.append(
                f"summary_field {field} native={float(native_summary[field]):.8f} "
                f"legacy={float(legacy_summary[field]):.8f}"
            )
        threshold = acceptance_thresholds["summary"].get(field)
        if threshold is not None and not _structured_long_rollout_within_threshold(
            float(native_summary[field]),
            float(legacy_summary[field]),
            abs_tol=float(threshold["abs"]),
            rel_tol=float(threshold["rel"]),
        ):
            system_acceptance_passed = False
            system_acceptance_errors.append(
                f"summary_field {field} "
                f"diff={abs(float(native_summary[field]) - float(legacy_summary[field])):.8f} "
                f"abs_tol={float(threshold['abs']):.8f} rel_tol={float(threshold['rel']):.8f}"
            )

    return {
        "baseline_policy": str(baseline_policy),
        "exec_sources": list(source_modes),
        "episodes": int(episodes),
        "episode_seed_base": int(episode_seed_base),
        "native_tensor_backend": native_tensor_backend_value,
        "comparison_mode": "native_random_tape_action_replay",
        "native_summary": native_summary,
        "legacy_summary": legacy_summary,
        "native_rows": native_rows,
        "legacy_rows": legacy_rows,
        "native_traces": native_traces,
        "legacy_traces": legacy_traces,
        "episode_diff_max": episode_diff_max,
        "trace_diff_max": trace_diff_max,
        "summary_diff_max": summary_diff_max,
        "passed": bool(system_acceptance_passed),
        "exact_passed": bool(exact_matches),
        "comparison_errors": comparison_errors,
        "system_acceptance_errors": system_acceptance_errors,
        "native_field_contract_errors": native_field_contract_errors,
        "coverage": _long_rollout_trace_coverage(native_traces),
        "acceptance_thresholds": acceptance_thresholds,
        "tolerances": {"atol": float(atol), "rtol": float(rtol)},
    }


def _cfg_for_flow_proxy_acceptance_case(cfg, *, enabled: bool, base_action_mode: str):
    case_cfg = copy.deepcopy(cfg)
    case_cfg.bw_flow_proxy_aux_enabled = bool(enabled)
    for attr in (
        "bw_counterfactual_credit_enabled",
        "bw_marginal_teacher_sample_enabled",
        "structured_bw_per_slot_surrogate_enabled",
    ):
        if hasattr(case_cfg, attr):
            setattr(case_cfg, attr, False)
    if bool(enabled):
        reward_mode = str(getattr(case_cfg, "reward_mode", "weighted_workload_level") or "weighted_workload_level").strip().lower()
        if reward_mode not in {"controllable_flow", "weighted_workload_level", "weighted_workload_delta", "relative_weighted_workload_delta"}:
            case_cfg.reward_mode = "weighted_workload_level"
    case_cfg.bw_flow_proxy_base_action_mode = str(base_action_mode).strip().lower()
    return case_cfg


def _cfg_for_done_acceptance_case(cfg, *, kind: str):
    case_cfg = copy.deepcopy(cfg)
    kind_s = str(kind).strip().lower()
    if kind_s == "terminated":
        case_cfg.d_safe = max(float(getattr(case_cfg, "d_safe", 0.0) or 0.0), 2000.0)
        case_cfg.uav_init_min_spacing = 0.0
        case_cfg.T_steps = max(int(getattr(case_cfg, "T_steps", 2) or 2), 2)
    elif kind_s == "truncated":
        case_cfg.d_safe = 0.0
        case_cfg.uav_init_min_spacing = 0.0
        case_cfg.T_steps = 1
    else:
        raise ValueError(f"unknown native acceptance done case {kind!r}")
    return case_cfg


def _merge_coverage(dst: Dict[str, bool], src: Dict[str, bool]) -> Dict[str, bool]:
    for key, value in src.items():
        dst[key] = bool(dst.get(key, False) or value)
    return dst


def validate_structured_fixed_seed_long_rollout_acceptance_matrix(
    cfg,
    *,
    baseline_policy: str,
    episodes: int,
    episode_seed_base: int,
    num_envs: int = 1,
    vec_backend: str = "sync",
    native_tensor_backend: str | None = None,
    atol: float = 1.0e-5,
    rtol: float = 1.0e-5,
) -> Dict[str, Any]:
    matrix_steps = max(int(getattr(cfg, "structured_acceptance_matrix_steps", 8) or 8), 1)

    def _matrix_case_cfg(case_cfg):
        out = copy.deepcopy(case_cfg)
        out.T_steps = min(max(int(getattr(out, "T_steps", matrix_steps) or matrix_steps), 1), matrix_steps)
        return out

    source_cases_list: list[tuple[str, tuple[str, str, str]]] = [
        ("policy_policy_policy", ("policy", "policy", "policy")),
        ("queue_aware_queue_aware_queue_aware", ("queue_aware", "queue_aware", "queue_aware")),
        (
            "cluster_center_queue_aware_cluster_center_queue_aware_cluster_center_queue_aware",
            ("cluster_center_queue_aware", "cluster_center_queue_aware", "cluster_center_queue_aware"),
        ),
        ("zero_policy_policy", ("zero", "policy", "policy")),
        ("policy_zero_policy", ("policy", "zero", "policy")),
        ("policy_policy_zero", ("policy", "policy", "zero")),
        ("queue_aware_policy_queue_aware", ("queue_aware", "policy", "queue_aware")),
        (
            "cluster_center_queue_aware_policy_queue_aware",
            ("cluster_center_queue_aware", "policy", "queue_aware"),
        ),
        ("zero_zero_zero", ("zero", "zero", "zero")),
    ]
    if getattr(cfg, "exec_teacher_actor_path", None):
        source_cases_list.append(("teacher_teacher_teacher", ("teacher", "teacher", "teacher")))
        source_cases_list.append(("teacher_policy_queue_aware", ("teacher", "policy", "queue_aware")))
    source_cases = tuple(source_cases_list)
    flow_cases = (
        ("flow_disabled", False, "executed"),
        ("flow_enabled_executed", True, "executed"),
        ("flow_enabled_deterministic", True, "deterministic"),
        ("flow_enabled_external_live_override", True, "external_live_override"),
    )
    done_cases = (
        ("done_terminated", _cfg_for_done_acceptance_case(cfg, kind="terminated")),
        ("done_truncated", _cfg_for_done_acceptance_case(cfg, kind="truncated")),
    )
    matrix_cases: list[dict[str, Any]] = []
    coverage = {
        "no_done_row": False,
        "terminated_row": False,
        "truncated_row": False,
        "rollout_tail_row": False,
        "episode_done_and_rollout_tail_row": False,
    }
    passed = True
    exact_passed = True
    legacy_eval_cache: dict[tuple[Any, ...], Any] = {}

    def _legacy_cache_key(case_cfg, source_modes: tuple[str, str, str]) -> tuple[Any, ...]:
        attrs = (
            "num_uav",
            "num_gu",
            "num_sat",
            "users_obs_max",
            "visible_sats_max",
            "sat_num_select",
            "T_steps",
            "d_safe",
            "uav_init_min_spacing",
            "reward_mode",
            "traffic_model",
            "structured_env_tensor_backend",
        )
        return (
            tuple(source_modes),
            int(episodes),
            int(episode_seed_base),
            str(native_tensor_backend),
            tuple((name, getattr(case_cfg, name, None)) for name in attrs),
        )

    def _legacy_eval_for(case_cfg, source_modes: tuple[str, str, str]):
        key = _legacy_cache_key(case_cfg, source_modes)
        cached = legacy_eval_cache.get(key)
        if cached is not None:
            return cached
        native_tensor_backend_value = (
            str(native_tensor_backend)
            if native_tensor_backend is not None
            else ("cuda" if torch.cuda.is_available() else "cpu")
        )
        native_cfg = _clone_cfg_with_structured_backend(
            case_cfg,
            structured_env_backend="native",
            structured_env_tensor_backend=native_tensor_backend_value,
        )
        legacy_cfg = _clone_cfg_with_structured_backend(
            case_cfg,
            structured_env_backend="legacy",
            structured_env_tensor_backend=native_tensor_backend_value,
        )
        legacy_cfg.structured_native_main_kernel_require_compiled_segments = False
        legacy_cfg.structured_native_main_kernel_required_compile_names = ""
        random_tape_cfg = copy.deepcopy(native_cfg)
        random_tape_cfg.structured_native_main_kernel_require_compiled_segments = False
        random_tape_cfg.structured_native_main_kernel_required_compile_names = ""
        cached = _evaluate_structured_baseline_policy_with_traces(
            legacy_cfg,
            baseline_policy=str(baseline_policy),
            episodes=episodes,
            episode_seed_base=episode_seed_base,
            random_tape_cfg=random_tape_cfg,
            exec_sources=source_modes,
        )
        legacy_eval_cache[key] = cached
        return cached

    def _append_case(
        *,
        name: str,
        source_name: str,
        source_modes: tuple[str, str, str],
        flow_name: str,
        flow_enabled: bool,
        flow_base: str,
        case_cfg,
    ) -> None:
        nonlocal passed, exact_passed
        report = validate_structured_fixed_seed_long_rollout(
            case_cfg,
            baseline_policy=baseline_policy,
            episodes=episodes,
            episode_seed_base=episode_seed_base,
            num_envs=num_envs,
            vec_backend=vec_backend,
            native_tensor_backend=native_tensor_backend,
            atol=atol,
            rtol=rtol,
            exec_sources=source_modes,
            precomputed_legacy_eval_result=_legacy_eval_for(case_cfg, source_modes),
        )
        case_passed = bool(report["passed"])
        case_exact = bool(report.get("exact_passed", report["passed"]))
        passed = passed and case_passed
        exact_passed = exact_passed and case_exact
        _merge_coverage(coverage, dict(report.get("coverage", {})))
        matrix_cases.append(
            {
                "name": name,
                "source_case": source_name,
                "exec_sources": list(source_modes),
                "flow_case": flow_name,
                "flow_proxy_enabled": bool(flow_enabled),
                "flow_proxy_base_action_mode": str(flow_base),
                "passed": case_passed,
                "exact_passed": case_exact,
                "episode_diff_max": report.get("episode_diff_max", {}),
                "trace_diff_max": report.get("trace_diff_max", {}),
                "comparison_errors": report.get("comparison_errors", []),
                "system_acceptance_errors": report.get("system_acceptance_errors", []),
                "coverage": report.get("coverage", {}),
            }
        )

    for source_name, source_modes in source_cases:
        flow_name, flow_enabled, flow_base = flow_cases[0]
        _append_case(
            name=f"{source_name}+{flow_name}",
            source_name=source_name,
            source_modes=source_modes,
            flow_name=flow_name,
            flow_enabled=bool(flow_enabled),
            flow_base=str(flow_base),
            case_cfg=_matrix_case_cfg(
                _cfg_for_flow_proxy_acceptance_case(
                    cfg,
                    enabled=bool(flow_enabled),
                    base_action_mode=str(flow_base),
                )
            ),
        )

    flow_source_modes = _fixed_policy_exec_sources(baseline_policy) or ("policy", "policy", "policy")
    for flow_name, flow_enabled, flow_base in flow_cases[1:]:
        _append_case(
            name=f"flow_source_reference+{flow_name}",
            source_name="flow_source_reference",
            source_modes=flow_source_modes,
            flow_name=flow_name,
            flow_enabled=bool(flow_enabled),
            flow_base=str(flow_base),
            case_cfg=_matrix_case_cfg(
                _cfg_for_flow_proxy_acceptance_case(
                    cfg,
                    enabled=bool(flow_enabled),
                    base_action_mode=str(flow_base),
                )
            ),
        )

    for done_name, done_cfg in done_cases:
        done_case_cfg = _matrix_case_cfg(done_cfg) if done_name == "done_terminated" else done_cfg
        report = validate_structured_fixed_seed_long_rollout(
            done_case_cfg,
            baseline_policy=baseline_policy,
            episodes=max(int(episodes), 1),
            episode_seed_base=episode_seed_base,
            num_envs=num_envs,
            vec_backend=vec_backend,
            native_tensor_backend=native_tensor_backend,
            atol=atol,
            rtol=rtol,
            exec_sources=("policy", "policy", "policy"),
            precomputed_legacy_eval_result=_legacy_eval_for(done_case_cfg, ("policy", "policy", "policy")),
        )
        case_passed = bool(report["passed"])
        case_exact = bool(report.get("exact_passed", report["passed"]))
        passed = passed and case_passed
        exact_passed = exact_passed and case_exact
        _merge_coverage(coverage, dict(report.get("coverage", {})))
        matrix_cases.append(
            {
                "name": done_name,
                "done_case": done_name,
                "exec_sources": ["policy", "policy", "policy"],
                "flow_case": "configured",
                "passed": case_passed,
                "exact_passed": case_exact,
                "episode_diff_max": report.get("episode_diff_max", {}),
                "trace_diff_max": report.get("trace_diff_max", {}),
                "comparison_errors": report.get("comparison_errors", []),
                "system_acceptance_errors": report.get("system_acceptance_errors", []),
                "coverage": report.get("coverage", {}),
            }
        )

    required_coverage = {
        "no_done_row",
        "terminated_row",
        "truncated_row",
        "rollout_tail_row",
        "episode_done_and_rollout_tail_row",
    }
    missing_coverage = sorted(key for key in required_coverage if not bool(coverage.get(key, False)))
    if missing_coverage:
        passed = False
        exact_passed = False
    return {
        "baseline_policy": str(baseline_policy),
        "episodes": int(episodes),
        "episode_seed_base": int(episode_seed_base),
        "native_tensor_backend": (
            str(native_tensor_backend)
            if native_tensor_backend is not None
            else ("cuda" if torch.cuda.is_available() else "cpu")
        ),
        "source_cases": [name for name, _ in source_cases],
        "flow_cases": [name for name, _enabled, _base in flow_cases],
        "done_cases": [name for name, _cfg in done_cases],
        "matrix_steps": int(matrix_steps),
        "coverage": coverage,
        "missing_coverage": missing_coverage,
        "cases": matrix_cases,
        "passed": bool(passed),
        "exact_passed": bool(exact_passed),
        "tolerances": {"atol": float(atol), "rtol": float(rtol)},
    }


def evaluate_structured_fixed_policy(
    cfg,
    *,
    baseline_policy: str,
    episodes: int,
    episode_seed_base: int | None = None,
    num_envs: int = 1,
) -> Dict[str, float]:
    exec_sources = _fixed_policy_exec_sources(baseline_policy)
    if exec_sources is not None:
        dummy_actor = torch.nn.Linear(1, 1)
        summary, _ = evaluate_structured_actor_exec_sources(
            cfg,
            dummy_actor,
            device=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
            episodes=episodes,
            episode_seed_base=episode_seed_base,
            deterministic=True,
            num_envs=max(int(num_envs), 1),
            vec_backend="sync",
            exec_accel_source=exec_sources[0],
            exec_sat_source=exec_sources[1],
            exec_bw_source=exec_sources[2],
        )
        return summary

    env_group = make_structured_env_group(cfg, num_envs=1, backend="sync", mode="eval")
    drivers = as_structured_drivers(env_group)
    rows: List[Dict[str, float]] = []
    baseline_state = None
    try:
        for ep in range(int(episodes)):
            seed = None if episode_seed_base is None else int(episode_seed_base) + ep
            reset_many(drivers, [seed])
            driver = drivers[0]
            baseline_state = None
            obs_list = current_obs_many(drivers, indices=[0])[0]
            done = False
            acc = _new_episode_accumulator(cfg)
            while not done:
                step_result, baseline_state = _run_structured_baseline_step(
                    baseline_policy,
                    driver,
                    cfg,
                    obs_list,
                    baseline_state=baseline_state,
                )
                obs_list = list(step_result.obs.values())
                reward_parts = _step_reward_parts(step_result, drivers, indices=[0])
                runtime_trace = _runtime_state_trace_rows(drivers, indices=[0])[0]
                _accumulate_step_metrics(
                    acc,
                    reward_value=_step_team_reward(step_result),
                    bw_delta_value=_batch_step_scalar(step_result, "bw_weighted_workload_delta_reward"),
                    bw_level_value=_batch_step_scalar(step_result, "bw_weighted_workload_level_reward"),
                    reward_parts=reward_parts,
                    runtime_trace=runtime_trace,
                )
                done = _step_done(step_result)
            rows.append(_episode_row_from_accumulator(ep, acc))
        return _summary_from_rows(rows, episodes=int(episodes))
    finally:
        close_structured_env_group(env_group)
