"""Opt-in cross-model validation with separate tasks, cache and summaries."""
from experiments.shared.audit.config import audit_settings
import argparse
import json
from pathlib import Path

from experiments.paths import QWEN_AUDIT, audit_reports, audit_executions
from experiments.cross_model_audit import engine
from experiments.shared.models.registry import MODEL_PAIRS
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
    parser.add_argument('--audit-seed', type=int, default=None, help="must match the condition seed; defaults to it")
    parser.add_argument('--detector-epochs', type=int, default=30)
    args = parser.parse_args()
    if args.detector_epochs < 1:
        parser.error('detector-epochs must be positive')
    for values in (args.model_pairs, args.benchmarks, args.epochs, args.seeds):
        if len(values) != len(set(values)):
            parser.error('duplicate matrix entries are not allowed')
    settings = audit_settings(audit_seed=args.audit_seed, detector_epochs=args.detector_epochs)
    try:
        tasks = selected_tasks(args.model_pairs, args.model_root, args.output_root, args.benchmarks,
                               args.epochs, args.seeds, settings)
    except ValueError as error:
        parser.error(str(error))
    summary_root = audit_reports(args.output_root)
    if args.command in ('dry-run', 'status'):
        main_tasks = [task for task in tasks if task['kind'] == 'main']
        baseline_count = len(engine.METHODS)
        print(json.dumps(dict(model_pairs=args.model_pairs, worker_tasks=len(tasks),
                              audit_configurations=len(main_tasks),
                              expected_method_rows=sum(len(task['methods']) + baseline_count for task in main_tasks),
                              summary_directory=str(summary_root),
                              states={task['id']: engine.inspect_task(task) for task in tasks}), indent=2))
        return
    completed = engine.run_tasks(tasks, args.output_root.resolve(), args.gpus) if args.command == 'run' else True
    result = engine.summarize(tasks, summary_root, execution_root=audit_executions(args.output_root))
    print(json.dumps({key: result[key] for key in ('complete', 'expected_rows', 'completed_rows', 'errors')}))
    if not completed or not result['complete']:
        raise SystemExit(2)


if __name__ == '__main__':
    main()
