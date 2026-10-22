"""Train a flat, centralized-value MAPPO adapter independently of STARS updates."""
from __future__ import annotations

import argparse
import csv
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import torch
import yaml

from sagin_marl.env.config import load_config
from sagin_marl.rl.slot_mappo import prepare_slot_batch, update_slot_mappo
from sagin_marl.rl.structured_buffer import StructuredRolloutBuffer
from sagin_marl.rl.structured_train import make_structured_driver_group, close_structured_env_group
from scripts.train_joint_mcgae import (
    _evaluate_joint_checkpoint, _force_joint_config, _make_joint_learner,
    _rng_state_payload, _restore_rng_state_payload, _set_seed,
)


def write_json(path, data):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(data, indent=2, default=str) + "\n")
    temporary.replace(path)


def append_csv(path, row):
    exists = path.exists()
    with path.open("a", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(row))
        if not exists:
            writer.writeheader()
        writer.writerow(row)


def actor_digest(actor):
    digest = hashlib.sha256()
    for key, value in sorted(actor.state_dict().items()):
        digest.update(key.encode())
        digest.update(value.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--run_dir", required=True)
    parser.add_argument("--seed", type=int, default=45211)
    parser.add_argument("--num_envs", type=int, default=64)
    parser.add_argument("--rollout_env_steps", type=int, default=250)
    parser.add_argument("--updates", type=int, default=25)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--minibatches", type=int, default=4)
    parser.add_argument("--microbatch", type=int, default=256)
    parser.add_argument("--actor_lr", type=float, default=1e-4)
    parser.add_argument("--critic_lr", type=float, default=3e-4)
    parser.add_argument("--hidden", type=int, default=512)
    parser.add_argument("--eval_interval", type=int, default=5)
    parser.add_argument("--eval_episodes", type=int, default=32)
    parser.add_argument("--eval_seed", type=int, default=20510000)
    parser.add_argument("--resume")
    args = parser.parse_args()
    for key in ("num_envs", "rollout_env_steps", "updates", "epochs", "minibatches", "microbatch", "hidden", "eval_interval", "eval_episodes"):
        if getattr(args, key) <= 0:
            parser.error(f"--{key} must be positive")
    if not torch.cuda.is_available():
        raise RuntimeError("native SAGIN MAPPO training requires CUDA")
    device = torch.device("cuda")
    _set_seed(args.seed)
    cfg = load_config(args.config)
    if int(cfg.access_bw_decision_interval) != 1 or int(cfg.sat_decision_interval) != 1:
        raise ValueError("slot MAPPO currently requires K_BW=K_sat=1")
    _force_joint_config(cfg, reward_mode=cfg.reward_mode)
    cfg.seed = args.seed
    cfg.structured_actor_backbone = "flat_mlp"
    cfg.critic_value_mode = "flat_mlp"
    cfg.actor_hidden = cfg.critic_hidden = cfg.critic_value_head_hidden = args.hidden
    cfg.structured_actor_input_norm_enabled = True
    cfg.structured_critic_input_norm_enabled = True
    cfg.critic_popart_enabled = True
    cfg.critic_popart_beta = 0.9
    cfg.danger_imitation_enabled = False
    cfg.danger_imitation_coef = 0.0
    cfg.actor_lr, cfg.critic_lr = args.actor_lr, args.critic_lr
    cfg.stage_mcgae_critic_ev_gate_enabled = False
    for name in ("critic_compile_enabled", "stage_actor_compile_enabled", "accel_actor_compile_enabled", "sat_actor_compile_enabled", "bw_actor_compile_enabled"):
        setattr(cfg, name, False)
    output = Path(args.run_dir).resolve()
    output.mkdir(parents=True, exist_ok=True)
    if any(output.iterdir()) and not args.resume:
        raise FileExistsError("use an empty run directory; historical results are never overwritten")
    config_text = yaml.safe_dump(asdict(cfg), sort_keys=False)
    config_hash = hashlib.sha256(config_text.encode()).hexdigest()
    learner = _make_joint_learner(cfg, device=device)
    actor, critic = learner.actor, learner.critic
    # StructuredMAPPO is used only as the existing native environment/actor
    # adapter. Neither its update() nor STARS return computation is called.
    actor_optimizer = torch.optim.Adam([p for p in actor.parameters() if p.requires_grad], lr=args.actor_lr, eps=1e-5)
    critic_optimizer = torch.optim.Adam(critic.parameters(), lr=args.critic_lr, eps=1e-5)
    start = 0
    best_reward = -float("inf")
    initial_digest = actor_digest(actor)
    if args.resume:
        saved = torch.load(args.resume, map_location=device, weights_only=False)
        if saved.get("algorithm") != "slot_mappo_v1" or saved["config_sha256"] != config_hash:
            raise ValueError("checkpoint algorithm/config does not match this run")
        for key in ("seed", "num_envs", "rollout_env_steps", "epochs", "minibatches", "microbatch", "eval_seed", "eval_episodes", "eval_interval"):
            if saved["args"][key] != getattr(args, key):
                raise ValueError(f"resume protocol changed: {key}")
        actor.load_state_dict(saved["actor"])
        critic.load_state_dict(saved["critic"])
        actor_optimizer.load_state_dict(saved["actor_optimizer"])
        critic_optimizer.load_state_dict(saved["critic_optimizer"])
        start = saved["update"]
        best_reward = saved["best_reward"]
        initial_digest = saved["initial_actor_sha256"]
        _restore_rng_state_payload(saved["rng_state"], device)
        # An interrupted update must not duplicate an already recorded update.
        for name in ("metrics.csv", "checkpoint_eval.csv"):
            path = output / name
            if path.exists():
                with path.open() as handle:
                    rows = list(csv.DictReader(handle))
                rows = [row for row in rows if int(float(row["update"])) <= start]
                path.unlink()
                for row in rows:
                    append_csv(path, row)
    else:
        (output / "config_source.yaml").write_text(config_text)
        write_json(output / "manifest.json", {
            "algorithm": "slot_mappo_v1", "source_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
            "config_sha256": config_hash, "initial_actor_sha256": initial_digest, "args": vars(args),
            "torch": torch.__version__, "cuda": torch.version.cuda, "device": torch.cuda.get_device_name(0),
            "ratio_unit": "per_UAV_joint_hybrid_action", "value_time": "pre_motion_once_per_slot",
            "bw_density": "raw_Dirichlet", "episode_horizon": "finite_terminal", "popart": True,
            "critic_ev_gate": False, "danger_imitation": False, "created_at": time.time(),
        })

    def save(name, update):
        temporary = output / (name + ".tmp")
        torch.save({"algorithm": "slot_mappo_v1", "actor": actor.state_dict(), "critic": critic.state_dict(),
            "actor_optimizer": actor_optimizer.state_dict(), "critic_optimizer": critic_optimizer.state_dict(),
            "cfg": vars(cfg), "args": vars(args), "update": update, "best_reward": best_reward,
            "config_sha256": config_hash, "initial_actor_sha256": initial_digest,
            "rng_state": _rng_state_payload(device)}, temporary)
        temporary.replace(output / name)

    def evaluate(update):
        nonlocal best_reward
        state = _rng_state_payload(device)
        try:
            summary = _evaluate_joint_checkpoint(cfg, learner, device=device, episodes=args.eval_episodes,
                episode_seed_base=args.eval_seed, num_envs=min(args.num_envs, args.eval_episodes), policy_mode="deterministic")
        finally:
            _restore_rng_state_payload(state, device)
        row = {"update": update, **summary}
        append_csv(output / "checkpoint_eval.csv", row)
        if summary["collision_episode_fraction"] == 0 and summary["reward_sum"] > best_reward:
            best_reward = summary["reward_sum"]
            save("best_model.pt", update)
        print(json.dumps({"evaluation": row}), flush=True)

    group = make_structured_driver_group(cfg, num_envs=args.num_envs, backend="sync", mode="train")
    try:
        if not args.resume:
            save("initial.pt", 0)
            evaluate(0)
        for update in range(start + 1, args.updates + 1):
            begun = time.perf_counter()
            write_json(output / "status.json", {"status": "running", "update": update, "phase": "collect", "time": time.time()})
            group.reset_many([args.seed + (update - 1) * 100000 + env for env in range(args.num_envs)])
            learner.begin_native_rollout(group, rollout_env_steps=args.rollout_env_steps, num_envs=args.num_envs)
            buffer = StructuredRolloutBuffer()
            with torch.no_grad():
                learner.collect_env_horizon_native_tensor_policy(group, buffer=buffer, horizon=args.rollout_env_steps, deterministic=False)
                views = buffer.build_rollout_views(device)
                batch = prepare_slot_batch(actor, critic, views, num_envs=args.num_envs,
                    gamma=cfg.gamma, gae_lambda=cfg.gae_lambda, chunk_size=args.microbatch)
            write_json(output / "status.json", {"status": "running", "update": update, "phase": "PPO", "time": time.time()})
            before = {name: next(module.parameters()).detach().clone() for name, module in
                (("accel", actor.accel_policy), ("sat", actor.sat_subset_policy), ("bw", actor.bw_policy))}
            info = update_slot_mappo(actor, critic, batch, actor_optimizer, critic_optimizer,
                epochs=args.epochs, minibatches=args.minibatches, microbatch_size=args.microbatch,
                clip_ratio=cfg.clip_ratio, value_coef=cfg.value_coef, max_grad_norm=cfg.max_grad_norm)
            for name, module in (("accel", actor.accel_policy), ("sat", actor.sat_subset_policy), ("bw", actor.bw_policy)):
                info[f"parameter_delta_{name}"] = float((next(module.parameters()).detach() - before[name]).abs().max())
            row = {"update": update, "iteration_sec": time.perf_counter() - begun,
                "reward_step_mean": float(batch.stages[2].rewards.mean()), **info}
            append_csv(output / "metrics.csv", row)
            print(json.dumps(row), flush=True)
            del batch, views, buffer
            if update % args.eval_interval == 0 or update == args.updates:
                evaluate(update)
                save(f"checkpoint_update{update:04d}.pt", update)
            save("final.pt", update)
        write_json(output / "status.json", {"status": "complete", "completed_updates": args.updates, "time": time.time()})
    except Exception as exc:
        write_json(output / "status.json", {"status": "failed", "error": repr(exc), "time": time.time()})
        raise
    finally:
        close_structured_env_group(group)


if __name__ == "__main__":
    main()
