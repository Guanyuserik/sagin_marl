"""Run the 100-GU Phase4 protocol with isolated sources and CUDA acceptance gates."""
from __future__ import annotations

import argparse
import csv
from dataclasses import asdict
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import tarfile
import time

import yaml

REPO = Path(__file__).resolve().parents[3]
SCENARIO = 'configs/experiments/more_gus/structured_joint_mcgae_3uav100gu_22clusters_t250.yaml'
PROTOCOL = 'configs/experiments/phase4_learning_ablation/structured_joint_mcgae_3uav20gu_t250_relational_critic.yaml'


def write_json(path: Path, value) -> None:
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(value, indent=2, default=str) + '\n')
    tmp.replace(path)


def read_rows(path: Path) -> list[dict]:
    with path.open() as handle:
        return list(csv.DictReader(handle))


def run_command(root: Path, source: Path, label: str, command: list[str]) -> float:
    log = root / f'{label}.log'
    started = time.time()
    record = dict(label=label, command=command, started_at=started, log=str(log))
    write_json(root / 'status.json', dict(phase=label, status='running', pid=os.getpid(), **record))
    print(f'START {label}', flush=True)
    with log.open('a') as handle:
        result = subprocess.run(command, cwd=source, stdout=handle, stderr=subprocess.STDOUT)
    elapsed = time.time() - started
    record.update(returncode=result.returncode, elapsed_seconds=elapsed, finished_at=time.time())
    with (root / 'commands.jsonl').open('a') as handle:
        handle.write(json.dumps(record) + '\n')
    print(f'END {label} rc={result.returncode} seconds={elapsed:.1f}', flush=True)
    if result.returncode:
        raise RuntimeError(f'{label} failed ({result.returncode}); inspect {log}')
    return elapsed


def prepare(root: Path, args) -> tuple[Path, dict]:
    manifest_path = root / 'manifest.json'
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text())
        for key in ('seed', 'max_updates', 'gpu'):
            if manifest[key] != getattr(args, key):
                raise ValueError(f'Run identity differs: {key}')
        source = root / 'source'
        for name, digest in manifest['config_sha256'].items():
            if hashlib.sha256((root / name).read_bytes()).hexdigest() != digest:
                raise RuntimeError(f'Frozen config changed: {name}')
        return source, manifest
    if any(root.iterdir()):
        raise RuntimeError('Use an empty run directory for a new campaign.')
    dirty = subprocess.check_output(['git', 'status', '--porcelain'], cwd=REPO, text=True)
    if dirty.strip():
        raise RuntimeError('Commit source changes before freezing a training campaign.')
    commit = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=REPO, text=True).strip()
    archive = root / 'source.tar'
    with archive.open('wb') as handle:
        subprocess.run(['git', 'archive', commit, 'sagin_marl', 'scripts', 'configs'],
                       cwd=REPO, stdout=handle, check=True)
    source = root / 'source'
    source.mkdir()
    with tarfile.open(archive) as handle:
        handle.extractall(source, filter='data')
    sys.path.insert(0, str(source))
    from sagin_marl.env.config import load_config
    from scripts.train_joint_mcgae import _force_joint_config
    import torch

    cfg = load_config(str(source / SCENARIO))
    phase4 = yaml.safe_load((source / PROTOCOL).read_text())
    for key, value in phase4.items():
        if key.startswith('checkpoint_eval_'):
            setattr(cfg, key, value)
    # Reference baseline design is tracked separately; it does not select models.
    cfg.checkpoint_eval_fixed_policy = ''
    cfg.seed = args.seed
    _force_joint_config(cfg, reward_mode=cfg.reward_mode)
    for key in ('critic_compile_enabled', 'stage_actor_compile_enabled',
                'accel_actor_compile_enabled', 'sat_actor_compile_enabled', 'bw_actor_compile_enabled'):
        setattr(cfg, key, False)
    if args.actor_microbatch is not None:
        cfg.actor_update_microbatch_size = args.actor_microbatch
    if args.critic_microbatch is not None:
        cfg.stage_mcgae_critic_update_microbatch_size = args.critic_microbatch
    effective = asdict(cfg)
    small = dict(effective, checkpoint_eval_interval_updates=1, checkpoint_eval_start_update=1,
                 checkpoint_eval_episodes=8, checkpoint_eval_episode_seed_base=1910000,
                 checkpoint_eval_early_stop_enabled=False)
    scale = dict(effective, checkpoint_eval_interval_updates=3, checkpoint_eval_start_update=3,
                 checkpoint_eval_episode_seed_base=1920000, checkpoint_eval_early_stop_enabled=False)
    for name, payload in [('config.yaml', effective), ('small_config.yaml', small), ('scale_config.yaml', scale)]:
        (root / name).write_text(yaml.safe_dump(payload, sort_keys=False))
    manifest = dict(created_at=time.time(), source_commit=commit, source_archive_sha256=hashlib.sha256(archive.read_bytes()).hexdigest(),
                    seed=args.seed, max_updates=args.max_updates, gpu=args.gpu,
                    scenario=SCENARIO, protocol=PROTOCOL, return_target='bootstrap_gae',
                    num_envs=64, rollout_env_steps=250, python=sys.version, torch=torch.__version__,
                    cuda=torch.version.cuda, device=torch.cuda.get_device_name(0),
                    validation_seed_base=effective['checkpoint_eval_episode_seed_base'],
                    postrun_seed_bases=[1980000, 1981000],
                    postrun_role='Held out from this training and model selection; first-run screening, not final baseline evidence',
                    source_isolated=True,
                    protocol_overrides={'checkpoint_eval_fixed_policy': '', 'torch_compile': False},
                    config_sha256={name:hashlib.sha256((root / name).read_bytes()).hexdigest()
                                   for name in ('config.yaml', 'small_config.yaml', 'scale_config.yaml')})
    write_json(manifest_path, manifest)
    return source, manifest


def train_command(root, config, output, args, *, envs, updates, save_every, resume=None):
    command = [sys.executable, 'scripts/train_joint_mcgae.py', '--config', str(root / config),
               '--run_dir', str(output), '--device', 'cuda', '--num_envs', str(envs),
               '--rollout_env_steps', '250', '--return_target', 'bootstrap_gae',
               '--return_target_schedule', 'fixed', '--max_updates', str(updates),
               '--seed', str(args.seed), '--save_every', str(save_every), '--disable_torch_compile']
    if resume is not None:
        command.extend(['--resume', str(resume)])
    return command


def evaluate(root, source, label, config, checkpoint, *, seed, episodes, envs):
    output = root / 'evaluations' / label
    command = [sys.executable, 'scripts/evaluate_structured_mixed_heads_native.py',
               '--config', str(root / config), '--base_checkpoint', str(checkpoint),
               '--device', 'cuda', '--episodes', str(episodes), '--num_envs', str(envs),
               '--episode_seed_base', str(seed), '--policy_mode', 'deterministic',
               '--out_dir', str(output), '--label', label]
    run_command(root, source, label, command)
    result = json.loads((output / f'{label}_summary.json').read_text())
    if any(result.get('load_info', {}).get(key) for key in ('missing_keys', 'unexpected_keys', 'adapted_keys', 'skipped_keys')):
        raise RuntimeError('Evaluation checkpoint did not load strictly.')
    return result['summary']


def check_training(output: Path, expected: int) -> dict:
    import torch
    rows = read_rows(output / 'metrics.csv')
    if [int(float(row['update'])) for row in rows] != list(range(1, expected + 1)):
        raise RuntimeError('Missing/duplicate update rows.')
    state = torch.load(output / 'final.pt', map_location='cpu', weights_only=False)
    if state['update'] != expected or not state['completed']:
        raise RuntimeError('Incomplete checkpoint.')
    stages = {}
    for stage_id, name in enumerate(('accel', 'sat', 'bw')):
        for row in rows:
            for key in (f'{name}_raw_adv_mean', f'{name}_norm_adv_std', f'{name}_critic_final_ev'):
                if not math.isfinite(float(row[key])):
                    raise RuntimeError(f'Non-finite training field: {key}')
        opt = state['actor_optimizers'][stage_id]['state']
        steps = [float(item['step']) for item in opt.values() if 'step' in item]
        stages[name] = dict(optimizer_steps=max(steps, default=0),
                           critic_ev=float(rows[-1][f'{name}_critic_final_ev']),
                           raw_adv_std=float(rows[-1][f'{name}_raw_adv_std']),
                           skipped_updates=sum(float(row.get(f'{name}_critic_critic_skip_actor_update', 0)) for row in rows))
    report = dict(updates=expected, stages=stages, iteration_seconds=[float(row['iteration_sec']) for row in rows],
                  memory={k:float(v) for k,v in rows[-1].items() if 'cuda_' in k})
    del state
    return report


def preflight(root, source, args):
    if (root / 'preflight.json').exists():
        raise RuntimeError('Preflight already recorded; do not overwrite evidence.')
    small = root / 'preflight' / 'small_resume'
    scale = root / 'preflight' / 'scale64'
    stop_file = small / 'training_stop.json'
    completed = int(json.loads(stop_file.read_text())['completed_updates']) if stop_file.exists() else 0
    if completed < 2:
        run_command(root, source, 'small_2u', train_command(root, 'small_config.yaml', small, args, envs=8, updates=2, save_every=1))
    if completed < 3:
        run_command(root, source, 'small_resume_u3', train_command(root, 'small_config.yaml', small, args,
                    envs=8, updates=3, save_every=1, resume=small / 'checkpoint_update0002.pt'))
    small_report = check_training(small, 3)
    small_eval_seed = yaml.safe_load((root / 'small_config.yaml').read_text())['checkpoint_eval_episode_seed_base']
    if small_eval_seed is None:
        raise ValueError('Evaluation parity needs an explicit checkpoint evaluation seed.')
    external = evaluate(root, source, 'small_eval_parity', 'small_config.yaml', small / 'final.pt',
                        seed=int(small_eval_seed), episodes=8, envs=8)
    internal = read_rows(small / 'checkpoint_eval.csv')[-1]
    parity = {}
    for key in ('reward_sum', 'processed_ratio_eval', 'drop_ratio_eval', 'pre_backlog_steps_eval', 'collision_episode_fraction'):
        parity[key] = abs(float(internal[key]) - float(external[key]))
        tolerance = 0.0 if key == 'collision_episode_fraction' else 1e-5
        if not math.isclose(float(internal[key]), float(external[key]), rel_tol=tolerance, abs_tol=tolerance):
            raise RuntimeError(f'Internal/external evaluation mismatch: {key} {parity[key]}')
    scale_seconds = run_command(root, source, 'scale64_3u', train_command(root, 'scale_config.yaml', scale, args,
                                         envs=64, updates=3, save_every=1))
    scale_report = check_training(scale, 3)
    for name, info in scale_report['stages'].items():
        if info['optimizer_steps'] <= 0:
            raise RuntimeError(f'{name} never updated in scale smoke; inspect critic gate.')
    write_json(root / 'preflight.json', dict(status='passed', small=small_report, scale=scale_report,
                                            evaluation_seed=int(small_eval_seed), evaluation_abs_error=parity, evaluation_tolerance={'rtol':1e-5,'atol':1e-5,'collision_exact':True}, scale_wall_seconds=scale_seconds,
                                            finished_at=time.time()))


def train(root, source, args):
    if json.loads((root / 'preflight.json').read_text())['status'] != 'passed':
        raise RuntimeError('CUDA acceptance gates have not passed.')
    output = root / 'train'
    if output.exists():
        raise RuntimeError('Formal run already exists; resume must be reviewed explicitly.')
    run_command(root, source, 'train', train_command(root, 'config.yaml', output, args,
                                                    envs=64, updates=args.max_updates, save_every=25))
    stop = json.loads((output / 'training_stop.json').read_text())
    training_report = check_training(output, int(stop['completed_updates']))
    summaries = {}
    for label, filename in [('selected', 'best_checkpoint.pt'), ('final', 'final.pt')]:
        checkpoint = output / filename
        for seed in (1980000, 1981000):
            key = f'{label}_seed{seed}'
            summaries[key] = evaluate(root, source, key, 'config.yaml', checkpoint, seed=seed, episodes=32, envs=32)
    write_json(root / 'result.json', dict(stop=stop, training=training_report, evaluations=summaries))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run_dir', required=True)
    parser.add_argument('--phase', choices=('prepare', 'preflight', 'train'), required=True)
    parser.add_argument('--seed', type=int, default=45211)
    parser.add_argument('--max_updates', type=int, default=700)
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--actor_microbatch', type=int)
    parser.add_argument('--critic_microbatch', type=int)
    args = parser.parse_args()
    if args.max_updates < 1 or args.gpu < 0 or any(x is not None and x < 1 for x in (args.actor_microbatch, args.critic_microbatch)):
        parser.error('GPU must be nonnegative; update and microbatch sizes must be positive.')
    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu)
    os.environ['PYTHONUNBUFFERED'] = '1'
    os.environ['PYTHONFAULTHANDLER'] = '1'
    os.environ['OMP_NUM_THREADS'] = '1'
    os.environ['OPENBLAS_NUM_THREADS'] = '1'
    os.environ['MKL_NUM_THREADS'] = '1'
    os.environ['PYTHONPATH'] = ''
    root = Path(args.run_dir).resolve()
    root.mkdir(parents=True, exist_ok=True)
    try:
        source, manifest = prepare(root, args)
        sys.path.insert(0, str(source))
        with (root / 'controller_invocations.jsonl').open('a') as handle:
            handle.write(json.dumps(dict(phase=args.phase, time=time.time(), command=sys.argv,
                                        sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())) + '\n')
        if args.phase == 'preflight':
            preflight(root, source, args)
        elif args.phase == 'train':
            train(root, source, args)
        write_json(root / 'status.json', dict(status='complete', phase=args.phase, time=time.time()))
        print(f'COMPLETE {args.phase} {root}', flush=True)
    except BaseException as exc:
        failure = dict(status='failed', phase=args.phase, error=repr(exc), time=time.time())
        write_json(root / 'failure.json', failure)
        write_json(root / 'status.json', failure)
        raise


if __name__ == '__main__':
    main()
