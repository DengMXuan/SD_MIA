"""Plan, run, resume and summarize target generalization and KD acceptance."""
import argparse
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
import csv
import fcntl
import json
import os
from pathlib import Path
from queue import Queue, Empty
import subprocess
import sys
import time

import numpy as np

from experiments.paths import ROOT
from experiments.shared.audit.devices import check_gpus
from experiments.shared.core.audit_runtime import _write_json
from experiments.shared.evaluation.quality import (
    OUTPUT_ROOT, KINDS, condition_task, evaluate_quality, inspect_task, read_report, validate_output,
)
from experiments.shared.models.registry import MODEL_PAIRS


def make_tasks(args):
    tasks = []
    for name in args.pairs:
        for benchmark in args.benchmarks:
            for seed in args.seeds:
                for kind in args.evaluations:
                    settings = dict(bootstrap_repeats=args.bootstrap_repeats)
                    if kind == 'generalization':
                        settings.update(samples=args.samples, batch_size=args.batch_size)
                    else:
                        settings.update(per_class=args.per_class)
                    tasks.append(condition_task(MODEL_PAIRS[name], benchmark, seed, kind,
                                                output_root=args.output_root, **settings))
    return tasks


def metric_rows(report):
    summary = report['summary']
    if report['task']['evaluation'] == 'acceptance':
        for role, metrics in summary.items():
            for metric, values in metrics.items():
                yield role, metric, values['mean'], values['ci95_low'], values['ci95_high']
    else:
        for metric, values in summary['member_minus_nonmember'].items():
            yield 'member', metric, values['member_mean'], None, None
            yield 'nonmember', metric, values['nonmember_mean'], None, None
            yield 'member_minus_nonmember', metric, values['delta'], values['ci95_low'], values['ci95_high']
        for name, values in summary['base_minus_tuned'].items():
            role, metric = name.split('/')
            yield f'{role}/base_minus_tuned', metric, values['delta'], values['ci95_low'], values['ci95_high']


def summarize(tasks, output_root):
    for task in tasks:
        validate_output(task['run_dir'], Path(output_root) / 'reports')
    states, rows = {}, []
    for task in tasks:
        states[task['id']] = state = inspect_task(task)
        if state['status'] != 'complete':
            continue
        report = read_report(task)
        for role, metric, value, low, high in metric_rows(report):
            rows.append(dict(**task['condition'], evaluation=task['evaluation'], role=role,
                             metric=metric, value=value, ci95_low=low, ci95_high=high,
                             report=str(Path(task['output']) / 'REPORT.json')))
    groups = defaultdict(list)
    for row in rows:
        key = tuple(row[name] for name in ('model_pair', 'benchmark', 'epoch', 'evaluation', 'role', 'metric'))
        groups[key].append(row)
    aggregates = []
    for key, entries in sorted(groups.items()):
        values = [r['value'] for r in entries]
        aggregates.append(dict(zip(('model_pair', 'benchmark', 'epoch', 'evaluation', 'role', 'metric'), key),
                               completed_seeds=len(values), seeds=sorted(r['condition_seed'] for r in entries),
                               mean=float(np.mean(values)), std=float(np.std(values, ddof=1)) if len(values) > 1 else None))
    result = dict(complete=all(s['status'] == 'complete' for s in states.values()),
                  expected_tasks=len(tasks), completed_tasks=sum(s['status'] == 'complete' for s in states.values()),
                  states=states, rows=rows, seed_summary=aggregates,
                  aggregation='equal weight across completed condition seeds; std is sample std, not a confidence interval')
    destination = Path(output_root) / 'reports'
    _write_json(destination / 'SUMMARY.json', result)
    for filename, entries in (('conditions.csv', rows), ('seeds.csv', aggregates)):
        with (destination / filename).open('w', newline='') as stream:
            if entries:
                writer = csv.DictWriter(stream, fieldnames=list(entries[0]))
                writer.writeheader()
                writer.writerows(entries)
    return result


def run(tasks, output_root, gpus):
    for task in tasks:
        validate_output(task['run_dir'], output_root)
    states = {task['id']: inspect_task(task) for task in tasks}
    blocked = {key: state for key, state in states.items() if state['status'] == 'blocked'}
    if blocked:
        raise ValueError(f'preflight failed: {json.dumps(blocked)}')
    pending = [task for task in tasks if states[task['id']]['status'] != 'complete']
    if not pending:
        return True
    check_gpus(gpus)
    output_root = Path(output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    with (output_root / '.matrix.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        execution = output_root / 'executions' / str(time.time_ns())
        execution.mkdir(parents=True)
        queue = Queue()
        for index, task in enumerate(pending):
            queue.put((index, task))

        def on_gpu(gpu):
            success = True
            while True:
                try:
                    index, task = queue.get_nowait()
                except Empty:
                    return success
                task_file = execution / f'{index}.json'
                logfile = execution / f'{index}.log'
                _write_json(task_file, task)
                env = {**os.environ, 'CUDA_VISIBLE_DEVICES': str(gpu),
                       'PYTHONHASHSEED': str(task['condition']['condition_seed']),
                       'OMP_NUM_THREADS': '2', 'TOKENIZERS_PARALLELISM': 'false',
                       'HF_HUB_OFFLINE': '1', 'TRANSFORMERS_OFFLINE': '1'}
                print(json.dumps(dict(event='start', task=task['id'], gpu=gpu, log=str(logfile))), flush=True)
                try:
                    check_gpus([gpu])
                    with logfile.open('w') as log:
                        process = subprocess.run([sys.executable, '-m', 'experiments.model_quality.cli',
                                                  'worker', '--task-file', str(task_file)], cwd=ROOT, env=env,
                                                 stdout=log, stderr=subprocess.STDOUT)
                    state = inspect_task(task)
                    ok = process.returncode == 0 and state['status'] == 'complete'
                    _write_json(execution / f'{index}.result.json', dict(returncode=process.returncode, state=state))
                except Exception as error:
                    ok = False
                    _write_json(execution / f'{index}.result.json', dict(error=str(error)))
                success = success and ok
                print(json.dumps(dict(event='finished', task=task['id'], gpu=gpu, success=ok)), flush=True)
        with ThreadPoolExecutor(max_workers=len(gpus)) as pool:
            results = list(pool.map(on_gpu, gpus))
        return all(results)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=('dry-run', 'status', 'run', 'summarize', 'worker'))
    parser.add_argument('--pairs', '--model-pairs', nargs='+', choices=tuple(MODEL_PAIRS), default=list(MODEL_PAIRS))
    parser.add_argument('--benchmarks', nargs='+', choices=('wikitection', 'newstection', 'arxivtection'),
                        default=['wikitection', 'newstection', 'arxivtection'])
    parser.add_argument('--seeds', nargs='+', type=int, choices=(1919, 1949, 1978), default=[1919, 1949, 1978])
    parser.add_argument('--evaluations', nargs='+', choices=KINDS, default=list(KINDS))
    parser.add_argument('--output-root', type=Path, default=OUTPUT_ROOT)
    parser.add_argument('--samples', type=int, default=500, help='generalization records per member/nonmember class')
    parser.add_argument('--per-class', type=int, default=256, help='acceptance records per member/nonmember/KD auxiliary class')
    parser.add_argument('--batch-size', type=int, default=4, help='generation batch size; acceptance uses one document per forward')
    parser.add_argument('--bootstrap-repeats', type=int, default=1000)
    parser.add_argument('--gpus', nargs='+', type=int, default=[0])
    parser.add_argument('--task-file', type=Path)
    args = parser.parse_args()
    if args.command == 'worker':
        if args.task_file is None:
            parser.error('worker requires --task-file')
        evaluate_quality(json.loads(args.task_file.read_text()))
        return
    for values in (args.pairs, args.benchmarks, args.seeds, args.evaluations, args.gpus):
        if len(values) != len(set(values)):
            parser.error('duplicate matrix entries are not allowed')
    if any(g < 0 for g in args.gpus):
        parser.error('GPU indices must be nonnegative')
    args.output_root = args.output_root.resolve()
    tasks = make_tasks(args)
    if args.command in ('dry-run', 'status'):
        states = {task['id']: inspect_task(task) for task in tasks}
        print(json.dumps(dict(conditions=len(tasks) // len(args.evaluations), tasks=len(tasks),
                              output_root=str(args.output_root), states=states), indent=2))
        if any(s['status'] == 'blocked' for s in states.values()):
            raise SystemExit(2)
        return
    success = run(tasks, args.output_root, args.gpus) if args.command == 'run' else True
    result = summarize(tasks, args.output_root)
    print(json.dumps({k: result[k] for k in ('complete', 'expected_tasks', 'completed_tasks')}))
    if not success or not result['complete']:
        raise SystemExit(2)


if __name__ == '__main__':
    main()
