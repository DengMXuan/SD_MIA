#!/usr/bin/env python3
"""GPU queue and matched main/baseline summaries for Pythia and Qwen pretraining."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from experiments.shared.core import gpu_pool
from standalone.pretraining_baselines.contract import (
    METHODS, SEEDS, SOURCES, frozen_contract, assert_main_partitions, separate_output,
)

ORIGINAL = Path('/home/mxd/lib/SD_MIA')
DATA = Path('/home/mxd/lib/SD_MIA-pretraining-data')


def plan(args):
    from experiments.pretraining.data import TARGET, DRAFT
    from experiments.pretraining.datasets import QWEN_MODELS
    from experiments.shared.audit.artifacts import digest
    tasks = []
    conditions = args.sources if args.experiment == 'mimir' else args.variants
    for name in conditions:
        for seed in args.seeds:
            if args.experiment == 'mimir':
                manifest_path = args.data_root / name / f'seed{seed}/manifest.json'
                count = 2000 if name == 'full_pile' else 400
                kind, models = 'mimir_pretraining_v1', dict(target=TARGET, draft=DRAFT)
                split = 'none' if name == 'full_pile' else 'ngram_13_0.8'
                benchmark = f'mimir/{name}/{split}'
            else:
                from standalone.qwen_temporal_clean.prepare import validate_output
                validate_output(args.data_root / f'seed{seed}')
                manifest_path = args.data_root / f'seed{seed}' / name / 'manifest.json'
                count, kind, models = 2000, 'temporal_pretraining_v1', QWEN_MODELS
                benchmark = f'wiki_temporal/qwen_temporal_clean_v2/{name}'
            manifest, partitions, *_ = frozen_contract(manifest_path, seed)
            if (manifest['kind'] != kind or manifest['models'] != models
                    or manifest['counts'] != dict(member=count, nonmember=count, auxiliary=600)
                    or manifest['benchmark'] != benchmark or manifest['max_tokens'] != 512):
                raise ValueError(f'model/source/count contract mismatch: {manifest_path}')
            if args.experiment == 'temporal' and manifest.get('cleaning_variant') != name:
                raise ValueError('temporal cleaning variant mismatch')
            output = args.output_root / name / f'seed{seed}'
            main_output = args.main_root / name / f'seed{seed}'
            assert_main_partitions(main_output, partitions)
            tasks.append(dict(experiment=args.experiment, source=name, seed=seed,
                manifest=str(manifest_path), output=str(output), main_output=str(main_output),
                manifest_sha256=partitions['manifest_sha256'], partitions_sha256=digest(partitions),
                counts=manifest['counts'], methods=list(METHODS)))
    return tasks


def worker(task):
    from experiments.shared.audit.artifacts import digest
    from standalone.pretraining_baselines.evaluate import evaluate
    _, partitions, *_ = frozen_contract(task['manifest'], task['seed'])
    if (partitions['manifest_sha256'] != task['manifest_sha256']
            or digest(partitions) != task['partitions_sha256'] or task['methods'] != list(METHODS)):
        raise ValueError('worker data/partition/methods changed after scheduling')
    reports = evaluate(task['manifest'], task['output'], seed=task['seed'],
                       main_dir=task['main_output'], device='cuda:0')
    print(json.dumps(dict(source=task['source'], seed=task['seed'],
                          metrics={m: r['metrics'] for m, r in reports.items()})), flush=True)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument('experiment', choices=('mimir', 'temporal'))
    parser.add_argument('command', choices=('dry-run', 'run', 'summarize'))
    parser.add_argument('--seeds', type=int, nargs='+', choices=SEEDS, default=list(SEEDS))
    parser.add_argument('--sources', nargs='+', choices=SOURCES, help='MIMIR domains; default all eight')
    parser.add_argument('--variants', nargs='+', choices=('clean_only', 'length_matched'),
                        help='Qwen temporal only; default length_matched')
    parser.add_argument('--data-root', type=Path,
        help='MIMIR prepared root (contains DOMAIN/seedSEED), or Qwen clean root (contains seedSEED/VARIANT)')
    parser.add_argument('--output-root', type=Path, help='new baseline task root; never the existing main root')
    parser.add_argument('--main-root', type=Path, help='read-only main task root for partition checks/comparison')
    gpu_pool.add_arguments(parser)
    args = parser.parse_args(argv)
    if args.experiment == 'mimir' and args.variants is not None:
        parser.error('--variants only applies to temporal')
    if args.experiment == 'temporal' and args.sources is not None:
        parser.error('--sources only applies to mimir')
    args.sources = args.sources or list(SOURCES)
    args.variants = args.variants or ['length_matched']
    for values in (args.sources, args.variants, args.seeds):
        if len(set(values)) != len(values):
            parser.error('duplicate conditions')
    name = 'pythia_mimir_baselines7_v1' if args.experiment == 'mimir' else 'qwen_temporal_clean_baselines7_v1'
    args.data_root = (args.data_root or (DATA / 'mimir/prepared' if args.experiment == 'mimir'
        else ROOT / 'artifacts/data/qwen_temporal_clean_v2')).resolve()
    args.output_root = (args.output_root or ROOT / 'artifacts/audits' / name / 'tasks').resolve()
    args.main_root = (args.main_root or (ORIGINAL / 'artifacts/audits/pythia_mimir_v1/tasks'
        if args.experiment == 'mimir' else ROOT / 'artifacts/audits/qwen_temporal_clean_v2/tasks')).resolve()
    protected = [args.data_root, args.main_root, ROOT / 'experiments', ROOT / 'standalone',
                 ROOT / 'tests', ROOT / '.git', ORIGINAL / 'experiments', ORIGINAL / 'standalone',
                 Path('/home/mxd/.cache/huggingface/hub')]
    try:
        separate_output(args.output_root, protected)
        if args.log_root:
            separate_output(args.log_root, [*protected, args.output_root])
        gpu_pool.configuration(args)
    except ValueError as error:
        parser.error(str(error))
    return args


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    if argv and argv[0] == '_worker':
        if len(argv) != 2:
            raise ValueError('_worker requires exactly one saved JSON task')
        worker(json.loads(argv[1]))
        return
    args = parse_args(argv)
    tasks = plan(args)
    scheduling = gpu_pool.configuration(args)
    if args.command == 'dry-run':
        print(json.dumps(dict(experiment=args.experiment, conditions=len(tasks), methods=METHODS,
            method_rows=len(tasks) * len(METHODS), scheduling=scheduling,
            auxiliary=dict(reference=400, threshold_calibration=200),
            reports=str(args.output_root / '_summary'), tasks=tasks), indent=2))
        return
    rows = []
    if args.command == 'run':
        jobs = [gpu_pool.Job(f'{t["source"]}/seed{t["seed"]}',
            [sys.executable, '-B', '-u', str(Path(__file__).resolve()), '_worker', json.dumps(t)],
            t['seed']) for t in tasks]
        rows = gpu_pool.run_jobs(jobs, scheduling=scheduling, cwd=ROOT, use_cuda=True,
            log_root=args.log_root or args.output_root / '_executions')
    from standalone.pretraining_baselines.reporting import summarize
    result = summarize(tasks, args.output_root / '_summary')
    print(json.dumps({k: result[k] for k in
        ('baseline_complete', 'comparison_complete', 'completed_baseline_rows', 'expected_baseline_rows', 'reports')}, indent=2))
    if (not result['baseline_complete'] or any(r['state'] != 'complete' for r in rows)
            or any(r['state'] == 'invalid' for r in result['rows'])):
        raise SystemExit(2)


if __name__ == '__main__':
    os.environ.setdefault('HF_HUB_OFFLINE', '1')
    os.environ.setdefault('TRANSFORMERS_OFFLINE', '1')
    try:
        main()
    except KeyboardInterrupt:
        raise SystemExit(130)
