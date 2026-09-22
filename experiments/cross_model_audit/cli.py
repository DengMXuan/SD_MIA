"""Opt-in cross-model validation with separate tasks, cache and summaries."""
import argparse
import json
from pathlib import Path

from experiments.paths import QWEN_AUDIT
from experiments.cross_model_audit import engine
from experiments.cross_model_audit.model_registry import MODEL_PAIRS
from experiments.cross_model_audit.storage import OUTPUT_ROOT


def selected_tasks(pairs, model_root, output_root, benchmarks, epochs, seeds, settings):
    if model_root is not None and len(pairs) != 1:
        raise ValueError('--model-root requires exactly one model pair')
    output_root = output_root.resolve()
    # Old and new task contracts must never share output files, even via aliases.
    if output_root == QWEN_AUDIT or output_root.is_relative_to(QWEN_AUDIT) or QWEN_AUDIT.is_relative_to(output_root):
        raise ValueError('cross-model output must be separate from the existing Qwen audit')
    tasks = []
    for name in pairs:
        root = model_root if model_root is not None else MODEL_PAIRS[name].run_root
        tasks.extend(engine.make_tasks(root.resolve(), output_root, benchmarks, epochs, seeds,
                                       settings, model_pair=name))
    if any(Path(task["output"]).is_relative_to(QWEN_AUDIT) for task in tasks):
        raise ValueError("resolved task output must be separate from the existing Qwen audit")
    return tasks


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=('dry-run', 'status', 'run', 'summarize'))
    parser.add_argument('--model-pairs', '--pairs', nargs='+', choices=tuple(MODEL_PAIRS), default=['gemma4'])
    parser.add_argument('--model-root', type=Path, help='training run root override, single pair only')
    parser.add_argument('--output-root', type=Path, default=OUTPUT_ROOT)
    parser.add_argument('--benchmarks', nargs='+', choices=('wikitection', 'newstection', 'arxivtection'),
                        default=['wikitection', 'newstection', 'arxivtection'])
    parser.add_argument('--epochs', nargs='+', type=int, choices=(1, 3), default=[1, 3])
    parser.add_argument('--seeds', nargs='+', type=int, choices=(1919, 1949, 1978), default=[1919, 1949, 1978])
    parser.add_argument('--gpus', nargs='+', type=int, default=[0])
    parser.add_argument('--audit-seed', type=int, default=20260914)
    parser.add_argument('--detector-epochs', type=int, default=30)
    args = parser.parse_args()
    if args.detector_epochs < 1:
        parser.error('detector-epochs must be positive')
    for values in (args.model_pairs, args.benchmarks, args.epochs, args.seeds):
        if len(values) != len(set(values)):
            parser.error('duplicate matrix entries are not allowed')
    settings = dict(starts=['suffix64'], rounds_per_start=32, audit_seed=args.audit_seed,
                    detector_epochs=args.detector_epochs, baseline=engine.BASELINE_DEFAULTS)
    try:
        tasks = selected_tasks(args.model_pairs, args.model_root, args.output_root, args.benchmarks,
                               args.epochs, args.seeds, settings)
    except ValueError as error:
        parser.error(str(error))
    summary_root = args.output_root.resolve() / 'fixed_only_summary'
    if args.command in ('dry-run', 'status'):
        conditions = sum(task['kind'] == 'baseline' for task in tasks)
        print(json.dumps(dict(model_pairs=args.model_pairs, worker_tasks=len(tasks),
                              audit_configurations=conditions * 2, expected_method_rows=conditions * 24,
                              summary_directory=str(summary_root),
                              states={task['id']: engine.inspect_task(task) for task in tasks}), indent=2))
        return
    completed = engine.run_tasks(tasks, args.output_root.resolve(), args.gpus) if args.command == 'run' else True
    result = engine.summarize(tasks, summary_root)
    selected = {task['id'] for task in tasks}
    attempts = [json.loads(path.read_text()) for path in (args.output_root / 'executions').glob('*/STATUS.json')]
    result['attempted_worker_wall_seconds_sum'] = sum(row.get('worker_wall_seconds', 0.) for row in attempts
                                                     if row.get('task') in selected)
    engine._write_json(summary_root / 'SUMMARY.json', result)
    print(json.dumps({key: result[key] for key in ('complete', 'expected_rows', 'completed_rows', 'errors')}))
    if not completed or not result['complete']:
        raise SystemExit(2)


if __name__ == '__main__':
    main()
