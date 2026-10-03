"""Run/resume the three-seed WikiTection epsilon-8 DP quality study."""
import argparse
import json
from pathlib import Path

from experiments.model_quality.cli import run, summarize
from experiments.shared.evaluation.quality import KINDS
from .quality import OUTPUT_ROOT, SEEDS, evaluate_quality, inspect_task, make_task, read_report


def make_tasks(args):
    tasks = []
    for seed in args.seeds:
        for kind in args.evaluations:
            settings = dict(bootstrap_repeats=args.bootstrap_repeats)
            if kind == 'generalization':
                settings.update(samples=args.samples, batch_size=args.batch_size)
            else:
                settings.update(per_class=args.per_class)
            tasks.append(make_task(seed, kind, output_root=args.output_root, **settings))
    return tasks


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=('dry-run', 'status', 'run', 'summarize', 'worker'))
    parser.add_argument('--seeds', nargs='+', type=int, choices=SEEDS, default=list(SEEDS))
    parser.add_argument('--evaluations', nargs='+', choices=KINDS, default=list(KINDS))
    parser.add_argument('--output-root', type=Path, default=OUTPUT_ROOT)
    parser.add_argument('--samples', type=int, default=500)
    parser.add_argument('--per-class', type=int, default=256)
    parser.add_argument('--batch-size', type=int, default=4)
    parser.add_argument('--bootstrap-repeats', type=int, default=1000)
    parser.add_argument('--gpus', nargs='+', type=int, default=[0])
    parser.add_argument('--task-file', type=Path)
    args = parser.parse_args()
    if args.command == 'worker':
        if args.task_file is None:
            parser.error('worker requires --task-file')
        evaluate_quality(json.loads(args.task_file.read_text()))
        return
    for values in (args.seeds, args.evaluations, args.gpus):
        if len(values) != len(set(values)):
            parser.error('duplicate matrix entries are not allowed')
    if any(g < 0 for g in args.gpus):
        parser.error('GPU indices must be nonnegative')
    args.output_root = args.output_root.resolve()
    tasks = make_tasks(args)
    if args.command in ('dry-run', 'status'):
        states = {task['id']: inspect_task(task) for task in tasks}
        print(json.dumps(dict(conditions=len(args.seeds), tasks=len(tasks),
                              output_root=str(args.output_root), states=states), indent=2))
        if any(state['status'] == 'blocked' for state in states.values()):
            raise SystemExit(2)
        return
    success = (run(tasks, args.output_root, args.gpus, inspector=inspect_task,
                   worker_module='experiments.dp_defense.quality_cli') if args.command == 'run' else True)
    result = summarize(tasks, args.output_root, inspector=inspect_task, report_reader=read_report)
    print(json.dumps({key: result[key] for key in ('complete', 'expected_tasks', 'completed_tasks')}))
    if not success or not result['complete']:
        raise SystemExit(2)


if __name__ == '__main__':
    main()
