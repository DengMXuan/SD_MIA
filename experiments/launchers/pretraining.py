"""One benchmark per invocation, with independent main and baseline queues."""
from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
import os
from pathlib import Path
import sys

from experiments.paths import ROOT
from experiments.shared.core import gpu_pool

SEEDS = (1919, 1949, 1978)
# Existing frozen data/results retain their paths and request identities.
AUDIT_ROOT = ROOT / 'artifacts/audits/paper_positive_controls_v1'


@dataclass(frozen=True)
class Benchmark:
    name: str
    legacy_group: str
    sources: tuple[str, ...]
    data_root: Path
    manifest_pattern: str
    benchmark_pattern: str
    counts: dict[str, tuple[int, int]]
    note: str
    kind: str = 'mimir_pretraining_v1'
    source_prefix: str = ''
    historical_main: Path | None = None


def plan(spec, sources, seeds, *, data_root=None, output_root=None):
    from experiments.pretraining.data import TARGET, DRAFT
    from experiments.shared.audit.artifacts import digest
    from standalone.pretraining_baselines.contract import (
        METHODS, assert_main_partitions, frozen_contract, separate_output,
    )
    data_root = Path(data_root or spec.data_root).resolve()
    output_root = Path(output_root or AUDIT_ROOT / 'tasks' / spec.legacy_group).resolve()
    protected = [data_root, ROOT / 'experiments', ROOT / 'standalone', ROOT / 'tests', ROOT / '.git',
                 ROOT / 'artifacts/audits/pythia_mimir_v1', ROOT.parent / 'SD_MIA-pretraining-data']
    if spec.historical_main:
        protected.append(spec.historical_main)
    separate_output(output_root, protected)
    tasks = []
    for source in sources:
        for seed in seeds:
            label = spec.source_prefix + source
            manifest_path = data_root / spec.manifest_pattern.format(source=source, seed=seed)
            manifest, partitions, *_ = frozen_contract(manifest_path, seed)
            member, nonmember = spec.counts[source]
            if (manifest['benchmark'] != spec.benchmark_pattern.format(source=source)
                    or manifest['models'] != dict(target=TARGET, draft=DRAFT)
                    or manifest['counts'] != dict(member=member, nonmember=nonmember, auxiliary=600)
                    or manifest['max_tokens'] != 512 or manifest['kind'] != spec.kind):
                raise ValueError(f'unexpected frozen experiment: {manifest_path}')
            folder = output_root / label / f'seed{seed}'
            main = (spec.historical_main / label / f'seed{seed}'
                    if spec.historical_main else folder / 'main')
            assert_main_partitions(main, partitions)
            tasks.append(dict(experiment=spec.legacy_group, source=label, seed=seed,
                manifest=str(manifest_path), main_output=str(main), output=str(folder / 'baselines7'),
                manifest_sha256=partitions['manifest_sha256'], partitions_sha256=digest(partitions),
                methods=list(METHODS), counts=manifest['counts'],
                historical_main=spec.historical_main is not None))
    return tasks


def worker(task, role):
    from experiments.launchers.devices import check_idle
    from experiments.shared.audit.artifacts import digest
    from standalone.pretraining_baselines.contract import MAIN, METHODS, frozen_contract
    check_idle([0])
    _, partitions, *_ = frozen_contract(task['manifest'], task['seed'])
    if (partitions['manifest_sha256'] != task['manifest_sha256']
            or digest(partitions) != task['partitions_sha256'] or task['methods'] != list(METHODS)):
        raise ValueError('frozen data/partition/methods changed after scheduling')
    if role == 'main':
        if task['historical_main']:
            raise ValueError('historical main results are read-only')
        from experiments.pretraining.evaluation import evaluate_main
        evaluate_main(task['manifest'], task['main_output'], seed=task['seed'], device='cuda:0')
    elif role == 'baseline':
        if task['historical_main'] and not (Path(task['main_output']) / MAIN / 'REPORT.json').is_file():
            raise ValueError('historical main result missing; refusing to regenerate it')
        from standalone.pretraining_baselines.evaluate import evaluate
        evaluate(task['manifest'], task['output'], seed=task['seed'],
                 main_dir=task['main_output'], device='cuda:0')
    else:
        raise ValueError('worker role must be main or baseline')


def summarize(spec, tasks, role, destination):
    if role == 'main':
        from experiments.pretraining.matrix import summarize as main_summary
        result = main_summary([{**task, 'output': task['main_output'], 'detector_epochs': 30}
                               for task in tasks])
        result['selected_complete'] = result['complete'] == result['conditions']
        destination.mkdir(parents=True, exist_ok=True)
        (destination / 'MAIN.json').write_text(json.dumps(result, indent=2) + '\n')
    else:
        from standalone.pretraining_baselines.reporting import summarize as comparison
        result = comparison(tasks, destination)
        result['selected_complete'] = result['baseline_complete']
        # The generic reporting module is a frozen numerical-source dependency.
        # Add the benchmark-specific interpretation here, outside that module.
        path = destination / 'COMPARISON.md'
        body = path.read_text().split('\n', 1)[1].replace(
            'Qwen temporal classes are historical/recent proxies, not verified membership.', spec.note)
        path.write_text(f'# {spec.name}\n' + body)
    result['reports'] = str(destination)
    return result


def run(spec, argv=None):
    os.environ.setdefault('HF_HUB_OFFLINE', '1')
    os.environ.setdefault('TRANSFORMERS_OFFLINE', '1')
    parser = argparse.ArgumentParser(description=spec.note, allow_abbrev=False)
    parser.add_argument('role', choices=('baseline',) if spec.historical_main else ('main', 'baseline'))
    parser.add_argument('command', nargs='?', default='dry-run', choices=('dry-run', 'run', 'summarize'))
    parser.add_argument('--sources', nargs='+', choices=spec.sources, default=list(spec.sources))
    parser.add_argument('--seeds', type=int, nargs='+', choices=SEEDS, default=list(SEEDS))
    parser.add_argument('--data-root', type=Path, default=spec.data_root)
    parser.add_argument('--output-root', type=Path, default=AUDIT_ROOT / 'tasks' / spec.legacy_group)
    gpu_pool.add_arguments(parser)
    args = parser.parse_args(argv)
    if any(len(set(values)) != len(values) for values in (args.sources, args.seeds)):
        parser.error('duplicate conditions')
    if args.command == 'run' and args.gpu is None and args.gpus is None:
        parser.error('run requires --gpu or --gpus with explicitly selected idle devices')
    try:
        scheduling = gpu_pool.configuration(args)
        tasks = plan(spec, args.sources, args.seeds, data_root=args.data_root, output_root=args.output_root)
        if args.command == 'dry-run':
            print(json.dumps(dict(experiment=spec.name, role=args.role, tasks=tasks,
                                  scheduling=scheduling, gpu_started=False), indent=2))
            return
        destination = args.output_root.resolve() / 'reports' / args.role
        if args.command == 'run':
            from experiments.launchers.devices import run_jobs
            jobs = [gpu_pool.Job(f'{spec.name}/{args.role}/{t["source"]}/seed{t["seed"]}',
                [sys.executable, '-B', '-u', '-m', 'experiments.launchers.pretraining',
                 args.role, json.dumps(t)], t['seed']) for t in tasks]
            log_root = args.log_root or args.output_root.resolve().with_name(
                args.output_root.name + '_executions') / args.role
            from standalone.pretraining_baselines.contract import separate_output
            separate_output(log_root, [args.data_root, args.output_root, ROOT / 'experiments',
                                      ROOT / 'standalone', ROOT / 'tests', ROOT / '.git'])
            rows = run_jobs(jobs, scheduling=scheduling, cwd=ROOT, log_root=log_root)
            result = summarize(spec, tasks, args.role, destination)
            print(json.dumps(dict(executions=rows, **result), indent=2))
            if any(row['state'] != 'complete' for row in rows) or not result['selected_complete']:
                raise SystemExit(2)
        else:
            result = summarize(spec, tasks, args.role, destination)
            print(json.dumps(result, indent=2))
            if not result['selected_complete']:
                raise SystemExit(2)
    except (OSError, ValueError, KeyError, RuntimeError) as error:
        parser.error(str(error))
    except KeyboardInterrupt:
        raise SystemExit(130)


if __name__ == '__main__':
    if len(sys.argv) != 3:
        raise SystemExit('internal worker requires ROLE TASK_JSON')
    worker(json.loads(sys.argv[2]), sys.argv[1])
