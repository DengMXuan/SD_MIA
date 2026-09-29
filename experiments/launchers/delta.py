"""Parallel exact-p/q collection with the original joint diagnostic analysis."""
import argparse
from dataclasses import asdict
import fcntl
import json
import os
from pathlib import Path
import sys

from huggingface_hub.constants import HF_HUB_CACHE

from experiments.paths import ROOT
from experiments.shared.core import gpu_pool
from experiments.launchers.devices import run_jobs


def jobs_for(args):
    jobs = []
    for source in args.datasets:
        for seed in args.seeds:
            output = args.output / 'conditions' / source / f'seed{seed}'
            command = [sys.executable, '-B', '-u', str(ROOT / 'standalone/pythia_delta/verify.py'),
                '--stage', 'collect', '--datasets', source, '--seeds', str(seed),
                '--device', 'cuda:0', '--dtype', args.dtype, '--per-class', str(args.per_class),
                '--sample-seed', str(args.sample_seed), '--threads', str(args.threads),
                '--bootstrap', str(args.bootstrap), '--data-root', str(args.data_root),
                '--audit-root', str(args.audit_root), '--model-root', str(args.model_root),
                '--output', str(output)]
            jobs.append(gpu_pool.Job(f'{source}/seed{seed}', command, seed))
    return jobs


def merge_caches(args, plan, items):
    """Share validated immutable worker caches, then correct across ALL conditions."""
    from standalone.pythia_delta import verify
    for item in items:
        source = args.output / 'conditions' / item['domain'] / f'seed{item["seed"]}'
        for role in ('target', 'draft'):
            verify.load_cached(source, role, item)
            destination = verify.cache_path(args.output, role, item)
            destination.parent.mkdir(parents=True, exist_ok=True)
            if not destination.exists():
                os.link(verify.cache_path(source, role, item), destination)
            verify.load_cached(args.output, role, item)
    verify.analyze(args, plan, items)


def main(argv=None):
    from standalone.pythia_delta import verify
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument('command', nargs='?', choices=('dry-run', 'run', 'summarize'), default='dry-run')
    parser.add_argument('--datasets', nargs='+', choices=verify.SOURCES, default=list(verify.SOURCES[:3]))
    parser.add_argument('--seeds', nargs='+', type=int, default=[1919])
    parser.add_argument('--per-class', type=int, default=64)
    parser.add_argument('--sample-seed', type=int, default=20260927)
    parser.add_argument('--bootstrap', type=int, default=2000)
    parser.add_argument('--threads', type=int, default=8)
    parser.add_argument('--dtype', choices=('float32', 'bfloat16'), default='float32')
    parser.add_argument('--data-root', type=Path, default=verify.DEFAULT_DATA)
    parser.add_argument('--audit-root', type=Path, default=verify.DEFAULT_AUDIT)
    parser.add_argument('--model-root', type=Path, default=Path(HF_HUB_CACHE))
    parser.add_argument('--output', type=Path, default=ROOT / 'artifacts/audits/pythia_delta_gpu_v1')
    gpu_pool.add_arguments(parser)
    args = parser.parse_args(argv)
    if (args.bootstrap < 100 or args.threads < 1 or args.per_class < 0
            or len(set(args.datasets)) != len(args.datasets) or len(set(args.seeds)) != len(args.seeds)):
        parser.error('invalid computation sizes or duplicate conditions')
    args.output = args.output.resolve()
    args.device = 'cuda:0'
    from standalone.pretraining_baselines.contract import separate_output
    separate_output(args.output, [args.data_root, args.audit_root, args.model_root,
                                   ROOT / 'experiments', ROOT / 'standalone', ROOT / 'tests', ROOT / '.git'])
    scheduling = gpu_pool.configuration(args)
    plan, items = verify.selection(args)
    jobs = jobs_for(args)
    if args.command == 'dry-run':
        print(json.dumps(dict(plan=plan, jobs=[asdict(job) for job in jobs], scheduling=scheduling), indent=2))
        return
    args.output.mkdir(parents=True, exist_ok=True)
    try:
        with (args.output / '.lock').open('a') as lock:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            saved = args.output / 'PLAN.json'
            if saved.exists():
                verify.require(verify.read_json(saved) == plan, 'plan changed; use a new output directory')
            else:
                verify.require(args.command == 'run', 'run collection before summarizing')
                verify.require(set(p.name for p in args.output.iterdir()) <= {'.lock'}, 'output directory already in use')
                verify.write_json(saved, plan)
            if args.command == 'run':
                log_root = args.log_root or args.output / 'executions'
                if args.log_root:
                    separate_output(log_root, [args.output, args.data_root, args.audit_root,
                        args.model_root, ROOT / 'experiments', ROOT / 'standalone', ROOT / 'tests', ROOT / '.git'])
                rows = run_jobs(jobs, scheduling=scheduling, cwd=ROOT, log_root=log_root)
                if any(row['state'] != 'complete' for row in rows):
                    raise SystemExit(2)
            merge_caches(args, plan, items)
    except KeyboardInterrupt:
        raise SystemExit(130)


if __name__ == '__main__':
    main()
