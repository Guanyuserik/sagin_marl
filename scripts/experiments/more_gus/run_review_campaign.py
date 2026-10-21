"""Frozen two-GPU IoTJ review campaign: policy evaluation and learning controls."""
from __future__ import annotations

import argparse
import csv
from dataclasses import asdict
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import tarfile
import time
from types import SimpleNamespace

REPO = Path(__file__).resolve().parents[3]
SEEDS = (45211, 45210, 61723)
TEST_BASES = (20421000, 20422000)
CONTROL_LABEL = "FlatActorCritic structural learning control (not vanilla MAPPO)"


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n")
    tmp.replace(path)


def write_csv(path, rows):
    if not rows:
        raise ValueError("No episode records")
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    names = sorted({key for row in rows for key in row})
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=names)
        writer.writeheader()
        writer.writerows(rows)


def conditions(base):
    result = [{"name": "nominal", "changes": {}}]
    for scale in (1.5, 2.0):
        result.append({"name": f"load_{scale:g}",
                       "changes": {"task_arrival_rate": base["task_arrival_rate"] * scale}})
    for axis, field in (("access", "b_acc"), ("backhaul", "b_backhaul_per_sat"),
                        ("cpu", "sat_cpu_freq")):
        for scale in (0.5, 0.75, 1.5, 2.0):
            changes = {field: base[field] * scale}
            if axis == "backhaul" and "b_sat_total" in base:
                changes["b_sat_total"] = changes[field]
            result.append({"name": f"{axis}_{scale:g}", "changes": changes})
    for rho in (1.0, 8.0):
        result.append({"name": f"hotspot_rho_{rho:g}", "changes": {"hotspot_rho": rho}})
    for duration in (5.0, 50.0):
        result.append({"name": f"hotspot_duration_{duration:g}",
                       "changes": {"hotspot_on_mean_steps": duration}})
    for scale in (1.0, 1.5):
        result.append({"name": f"long1000_load_{scale:g}", "changes": {
            "T_steps": 1000, "task_arrival_rate": base["task_arrival_rate"] * scale}})
    return result


def learning_config(base, seed):
    result = dict(base, seed=seed, structured_actor_backbone="flat_mlp", critic_value_mode="flat_mlp")
    changed = {key for key in base if result.get(key) != base[key]}
    if changed - {"seed", "structured_actor_backbone", "critic_value_mode"}:
        raise ValueError(f"Unexpected learning-control changes: {changed}")
    return result


def normalize_episode(row, horizon):
    def number(key):
        value = float(row[key])
        if not math.isfinite(value):
            raise ValueError(f"Nonfinite metric: {key}")
        return value
    arrivals = number("arrival_sum")
    steps = number("episode_length")
    if arrivals <= 0 or steps <= 0:
        raise ValueError("Missing arrivals or elapsed slots")
    processed = number("sat_processed_sum") / arrivals
    dropped = number("drop_sum") / arrivals
    residual = number("terminal_queue_sum") / arrivals
    balance_error = processed + dropped + residual - 1.0
    if abs(balance_error) > 1e-4:
        raise ValueError(f"Workload conservation failed: {balance_error}")
    return dict(
        arrival_bits=arrivals, processed_bits=number("sat_processed_sum"),
        dropped_bits=number("drop_sum"), terminal_inventory_bits=number("terminal_queue_sum"),
        processed_workload_fraction=processed, dropped_workload_fraction=dropped,
        terminal_inventory_fraction=residual, conservation_error=balance_error,
        mean_queue_mbit=number("queue_total_mean") / 1e6,
        objective_cost_per_observed_slot=-number("bw_weighted_workload_level_sum") / steps,
        reward_sum=number("reward_sum"), collision=number("collision_episode_fraction"),
        observed_slots=steps, configured_horizon=horizon,
        early_termination=float(steps < horizon),
        complete_horizon=float(steps == horizon),
    )


def load_module(source):
    spec = importlib.util.spec_from_file_location("frozen_review_bootstrap", source / "scripts/experiments/more_gus/run_bootstrap_training.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def find_reference(root, seed):
    matches = [p for p in root.glob(f"bootstrap_phase4_3uav100gu_k1_seed{seed}_*")
               if "continuation" not in p.name and "_u500_" not in p.name and (p / "manifest.json").exists()]
    if len(matches) != 1:
        raise ValueError(f"Expected one original training run for {seed}: {matches}")
    return matches[0]


def prepare(args):
    import yaml
    sys.path.insert(0, str(REPO))
    from sagin_marl.env.config import load_config
    from scripts.experiments.more_gus.run_fixed_baselines import check_protocol
    root = Path(args.run_dir).resolve()
    if root.exists():
        raise ValueError("Use a new campaign directory")
    dirty = subprocess.check_output(["git", "status", "--porcelain"], cwd=REPO, text=True)
    if dirty.strip():
        raise RuntimeError("Freeze the experiment branch before preparing")
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO, text=True).strip()
    reference_root = Path(args.reference_root).resolve()
    refs = [find_reference(reference_root, seed) for seed in SEEDS]
    loaded = []
    provenance = {}
    for ref in refs:
        meta = json.loads((ref / "manifest.json").read_text())
        if sha(ref / "config.yaml") != meta["config_sha256"]["config.yaml"]:
            raise ValueError(f"Reference config hash mismatch: {ref}")
        if json.loads((ref / "status.json").read_text())["status"] != "complete" or (ref / "failure.json").exists():
            raise ValueError(f"Incomplete reference: {ref}")
        cfg = json.loads(json.dumps(asdict(load_config(str(ref / "config.yaml")))))
        check_protocol(cfg, 64, 32, list(TEST_BASES))
        loaded.append(cfg)
        for file in (ref / "manifest.json", ref / "config.yaml", ref / "train/best_checkpoint.pt", ref / "train/final.pt"):
            provenance[str(file)] = sha(file)
    base = loaded[0]
    for cfg in loaded[1:]:
        different = {key for key in base.keys() | cfg.keys() if base.get(key) != cfg.get(key)}
        if different - {"seed"}:
            raise ValueError(f"Reference training protocols differ: {different}")
    source_fields = ["scripts/train_joint_mcgae.py", "sagin_marl/rl/structured_mappo.py",
                     "sagin_marl/rl/structured_critic.py", "sagin_marl/rl/structured_actor.py",
                     "sagin_marl/rl/structured_factory.py", "sagin_marl/env/native_cuda/kernels.cu"]
    for name in source_fields:
        if sha(REPO / name) != sha(refs[0] / "source" / name):
            raise ValueError(f"Reference learning/environment source changed: {name}")
    root.mkdir(parents=True)
    archive = root / "source.tar"
    with archive.open("wb") as handle:
        subprocess.run(["git", "archive", commit, "sagin_marl", "scripts", "configs"], cwd=REPO, stdout=handle, check=True)
    source = root / "source"
    source.mkdir()
    with tarfile.open(archive) as handle:
        handle.extractall(source, filter="data")
    cfg_hashes = {}
    def config_file(relative, payload):
        path = root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(yaml.safe_dump(payload, sort_keys=False))
        cfg_hashes[relative] = sha(path)
        return str(path)
    evaluation_jobs = []
    scenario_rows = conditions(base)
    models = [{"name": f"stars_seed{s}", "kind": "learned", "training_seed": s,
               "checkpoint": str(r / "train/best_checkpoint.pt")} for s, r in zip(SEEDS, refs)]
    models += [{"name": "dqs", "kind": "fixed", "baseline": "distributed_queue_c"},
               {"name": "lyapunov", "kind": "fixed", "baseline": "maxweight_lyapunov"}]
    for point in scenario_rows:
        cfg = dict(base, **point["changes"])
        config = config_file(f"evaluation/configs/{point['name']}.yaml", cfg)
        for model in models:
            for episode_base in TEST_BASES:
                evaluation_jobs.append(dict(point=point["name"], config=config, model=model,
                    episode_seed_base=episode_base, episodes=64, horizon=cfg["T_steps"]))
    nominal = str(root / "evaluation/configs/nominal.yaml")
    for seed, ref in zip(SEEDS, refs):
        model = dict(name=f"stars_final_seed{seed}", kind="learned", training_seed=seed,
                     checkpoint=str(ref / "train/final.pt"))
        for episode_base in TEST_BASES:
            evaluation_jobs.append(dict(point="nominal", config=nominal, model=model,
                episode_seed_base=episode_base, episodes=64, horizon=250))
    training_jobs = []
    for seed in SEEDS:
        child = root / "training" / f"flat_actor_critic_seed{seed}"
        cfg = learning_config(base, seed)
        config_file(str(child.relative_to(root) / "config.yaml"), cfg)
        small = dict(cfg, checkpoint_eval_interval_updates=1, checkpoint_eval_start_update=1,
                     checkpoint_eval_episodes=8, checkpoint_eval_episode_seed_base=1910000,
                     checkpoint_eval_early_stop_enabled=False,
                     checkpoint_eval_reward_early_stop_enabled=False)
        scale = dict(cfg, checkpoint_eval_interval_updates=3, checkpoint_eval_start_update=3,
                     checkpoint_eval_episode_seed_base=20420000,
                     checkpoint_eval_early_stop_enabled=False,
                     checkpoint_eval_reward_early_stop_enabled=False)
        config_file(str(child.relative_to(root) / "small_config.yaml"), small)
        config_file(str(child.relative_to(root) / "scale_config.yaml"), scale)
        training_jobs.append(dict(seed=seed, path=str(child), model="flat_actor_critic",
            differences={key: [base.get(key), cfg[key]] for key in cfg if base.get(key) != cfg[key]}))
    write_json(root / "manifest.json", dict(source_commit=commit,
        source_archive_sha256=sha(archive), reference_root=str(reference_root), provenance=provenance,
        learning_source_sha256={name: sha(REPO / name) for name in source_fields},
        config_sha256=cfg_hashes, scenarios=scenario_rows, evaluation_jobs=evaluation_jobs,
        training_jobs=training_jobs, evaluation_gpu=0, training_gpu=1,
        test_seed_bases=TEST_BASES, episodes_per_base=64, num_envs_eval=32,
        training_updates_cap=700, training_envs=64, training_rollout_slots=250,
        learning_control_label=CONTROL_LABEL,
        checkpoint_selection="Unchanged reference validation and early stopping; selected and final both retained",
        evaluation_role="Held-out review evaluation; no model selection or tuning on these episodes",
        failure_policy="Keep all episodes; report collisions, observed horizon and fixed-horizon completion separately",
        metrics_version="episode-cumulative-workload-and-terminal-balance-v1", created_at=time.time()))
    for line in ("evaluation", "training"):
        write_json(root / line / "status.json", dict(status="prepared", completed=0, time=time.time()))
    print(json.dumps(dict(run_dir=str(root), evaluation_jobs=len(evaluation_jobs),
                         training_jobs=len(training_jobs), scenarios=len(scenario_rows))))


def configure_environment(gpu):
    for key, value in {"CUDA_VISIBLE_DEVICES": str(gpu), "OMP_NUM_THREADS": "1",
        "OPENBLAS_NUM_THREADS": "1", "MKL_NUM_THREADS": "1", "PYTHONUNBUFFERED": "1",
        "PYTHONFAULTHANDLER": "1", "PYTHONPATH": ""}.items():
        os.environ[key] = value


def verify(root):
    manifest = json.loads((root / "manifest.json").read_text())
    if sha(root / "source.tar") != manifest["source_archive_sha256"]:
        raise ValueError("Frozen source archive changed")
    for name, digest in manifest["config_sha256"].items():
        if sha(root / name) != digest:
            raise ValueError(f"Frozen config changed: {name}")
    for name, digest in manifest["provenance"].items():
        if sha(name) != digest:
            raise ValueError(f"Reference artifact changed: {name}")
    return manifest


def read_csv(path):
    with Path(path).open() as handle:
        return list(csv.DictReader(handle))


def evaluate_job(root, jobs, job, *, pilot=False):
    model = job["model"]
    label = f"{job['point']}__{model['name']}__seed{job['episode_seed_base']}"
    dest = root / "evaluation" / ("preflight" if pilot else "results") / label
    if dest.exists():
        raise ValueError(f"Evaluation output already exists: {dest}")
    dest.mkdir(parents=True)
    source = root / "source"
    if model["kind"] == "learned":
        command = [sys.executable, str(source / "scripts/evaluate_structured_mixed_heads_native.py"),
            "--config", job["config"], "--base_checkpoint", model["checkpoint"], "--device", "cuda",
            "--episodes", str(job["episodes"]), "--num_envs", "32", "--episode_seed_base", str(job["episode_seed_base"]),
            "--policy_mode", "deterministic", "--out_dir", str(dest), "--label", "policy"]
        rows_file, summary_file = dest / "policy_episodes.csv", dest / "policy_summary.json"
    else:
        command = [sys.executable, str(source / "scripts/evaluation/evaluate_structured_fixed_policy.py"),
            "--config", job["config"], "--baseline", model["baseline"], "--episodes", str(job["episodes"]),
            "--num_envs", "32", "--episode_seed_base", str(job["episode_seed_base"]),
            "--structured_env_tensor_backend", "cuda", "--out", str(dest / "episodes.csv"),
            "--summary_out", str(dest / "summary.json")]
        if model["baseline"] == "distributed_queue_c":
            command += ["--dq_movement_weight", "0", "--dq_switch_weight", "0"]
        rows_file, summary_file = dest / "episodes.csv", dest / "summary.json"
    jobs.run_command(dest, source, "evaluation", command)
    summary = json.loads(summary_file.read_text())
    if any(summary.get("load_info", {}).get(k) for k in ("missing_keys", "unexpected_keys", "adapted_keys", "skipped_keys")):
        raise ValueError("Checkpoint did not load strictly")
    rows = read_csv(rows_file)
    if len(rows) != job["episodes"]:
        raise ValueError("Incomplete episode output")
    normalized = []
    for index, row in enumerate(rows):
        record = dict(point=job["point"], model=model["name"], training_seed=model.get("training_seed", -1),
                      episode_seed=job["episode_seed_base"] + index, **normalize_episode(row, job["horizon"]))
        normalized.append(record)
    write_csv(dest / "normalized_episodes.csv", normalized)
    write_json(dest / "completion.json", dict(status="complete", episodes=len(rows), job=job,
        raw_rows_sha256=sha(rows_file), normalized_rows_sha256=sha(dest / "normalized_episodes.csv"), time=time.time()))
    return normalized


def run_evaluation(root, manifest):
    configure_environment(manifest["evaluation_gpu"])
    source = root / "source"
    sys.path.insert(0, str(source))
    jobs = load_module(source)
    if json.loads((root / "evaluation/status.json").read_text())["status"] != "prepared":
        raise ValueError("Evaluation queue was already started")
    pilot = dict(manifest["evaluation_jobs"][0], episodes=32, episode_seed_base=20420000)
    write_json(root / "evaluation/status.json", dict(status="running", phase="preflight", pid=os.getpid(), gpu=0, completed=0, time=time.time()))
    evaluate_job(root, jobs, pilot, pilot=True)
    all_rows = []
    for index, job in enumerate(manifest["evaluation_jobs"]):
        write_json(root / "evaluation/status.json", dict(status="running", phase="evaluate", pid=os.getpid(), gpu=0,
            completed=index, total=len(manifest["evaluation_jobs"]), current_job=job, time=time.time()))
        all_rows.extend(evaluate_job(root, jobs, job))
        write_csv(root / "evaluation/all_episodes.csv", all_rows)
    write_json(root / "evaluation/status.json", dict(status="complete", completed=len(manifest["evaluation_jobs"]),
        total=len(manifest["evaluation_jobs"]), episodes=len(all_rows), time=time.time()))


def run_training(root, manifest):
    configure_environment(manifest["training_gpu"])
    source = root / "source"
    sys.path.insert(0, str(source))
    jobs = load_module(source)
    if json.loads((root / "training/status.json").read_text())["status"] != "prepared":
        raise ValueError("Training queue was already started")
    completed = []
    for job in manifest["training_jobs"]:
        child = Path(job["path"])
        args = SimpleNamespace(seed=job["seed"], gpu=1, max_updates=manifest["training_updates_cap"])
        write_json(root / "training/status.json", dict(status="running", phase="preflight", pid=os.getpid(),
            gpu=1, seed=job["seed"], completed_seeds=completed, time=time.time()))
        jobs.preflight(child, source, args)
        write_json(root / "training/status.json", dict(status="running", phase="train", pid=os.getpid(),
            gpu=1, seed=job["seed"], completed_seeds=completed, time=time.time()))
        jobs.train(child, source, args)
        write_json(root / "training/status.json", dict(status="running", phase="heldout_evaluation", pid=os.getpid(),
            gpu=1, seed=job["seed"], completed_seeds=completed, time=time.time()))
        fresh_rows = []
        for kind, filename in (("selected", "best_checkpoint.pt"), ("final", "final.pt")):
            for episode_base in manifest["test_seed_bases"]:
                label = f"review_{kind}_seed{episode_base}"
                jobs.evaluate(child, source, label, "config.yaml", child / "train" / filename,
                    seed=episode_base, episodes=64, envs=32)
                rows = read_csv(child / "evaluations" / label / f"{label}_episodes.csv")
                for index, row in enumerate(rows):
                    fresh_rows.append(dict(model="flat_actor_critic", training_seed=job["seed"],
                        checkpoint_kind=kind, episode_seed=episode_base + index,
                        **normalize_episode(row, 250)))
        write_csv(child / "review_heldout_episodes.csv", fresh_rows)
        write_json(child / "status.json", dict(status="complete", seed=job["seed"], time=time.time()))
        completed.append(job["seed"])
    write_json(root / "training/status.json", dict(status="complete", completed_seeds=completed, time=time.time()))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", choices=("prepare", "evaluation", "training"), required=True)
    parser.add_argument("--run_dir", required=True)
    parser.add_argument("--reference_root")
    args = parser.parse_args()
    root = Path(args.run_dir).resolve()
    if args.phase == "prepare":
        if not args.reference_root:
            parser.error("Preparation requires --reference_root")
        prepare(args)
        return
    manifest = verify(root)
    line = "evaluation" if args.phase == "evaluation" else "training"
    try:
        (run_evaluation if line == "evaluation" else run_training)(root, manifest)
    except BaseException as exc:
        payload = dict(status="failed", error=repr(exc), pid=os.getpid(), time=time.time())
        write_json(root / line / "failure.json", payload)
        write_json(root / line / "status.json", payload)
        raise


if __name__ == "__main__":
    main()
