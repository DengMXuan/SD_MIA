"""Seven Qwen3 epoch-1 baselines on three datasets and three matching seeds.

Each condition reuses its frozen 2,000 members, 2,000 nonmembers and 600 audit
auxiliaries (400 reference / 200 calibration). Only the target is evaluated.
The shared worker checks the training/data seed against the requested condition
before inference and resumes only completed methods with matching provenance.
"""
import argparse
import json
from pathlib import Path

from experiments.cross_model_audit import engine
from experiments.paths import AUDITS, audit_executions, audit_reports
from experiments.shared.audit import reporting, scheduler
from experiments.shared.audit.config import audit_settings
from experiments.shared.audit.devices import check_gpus
from experiments.shared.audit.evaluation import _validate_output
from experiments.shared.audit.provenance import read_result
from experiments.shared.models.registry import MODEL_PAIRS

BENCHMARKS = ('wikitection', 'newstection', 'arxivtection')
SEEDS = (1919, 1949, 1978)
METHODS = ('loss', 'min_k_prob', 'min_k_pp', 'sead', 'petal', 'recall', 'icp_mia')
OUTPUT_ROOT = AUDITS / 'qwen_baselines7_epoch1_condition_seed_v1/tasks'


def validate_scope(task):
    """Reject expanded scope or a mismatched seed before dispatching a worker."""
    condition = task['condition']
    if (task.get('model_pair') != 'qwen3' or condition.get('model_pair') != 'qwen3'
            or task.get('kind') != 'baseline' or task.get('methods') != list(METHODS)
            or 'draft_role' in task or 'protocol' in task
            or condition.get('benchmark') not in BENCHMARKS or condition.get('epoch') != 1
            or condition.get('condition_seed') not in SEEDS):
        raise ValueError('this experiment requires the seven Qwen3 epoch-1 baselines only')
    if (task['settings'].get('audit_seed') != condition['condition_seed']
            or task['settings'].get('seed_policy') != 'condition_v1'):
        raise ValueError('baseline, auxiliary partition and AUC seeds must match the condition seed')
    _validate_output(Path(task['run_dir']).resolve(), Path(task['output']).resolve())


def make_tasks(output_root=OUTPUT_ROOT, *, model_root=None):
    model_root = Path(model_root or MODEL_PAIRS['qwen3'].run_root).resolve()
    output_root = Path(output_root).resolve()
    for destination in (output_root, audit_reports(output_root), audit_executions(output_root)):
        _validate_output(model_root, destination)
    candidates = engine.make_tasks(model_root, output_root, BENCHMARKS, [1], SEEDS,
                                   audit_settings(), model_pair='qwen3')
    tasks = [task for task in candidates if task['kind'] == 'baseline']
    for task in tasks:
        task['methods'] = list(METHODS)
        validate_scope(task)
    if len(tasks) != 9 or len({task['output'] for task in tasks}) != 9:
        raise ValueError('expected nine distinct baseline conditions')
    return tasks


def inspect_task(task):
    validate_scope(task)
    return engine.inspect_task(task)


def summarize(tasks, output_root):
    for task in tasks:
        validate_scope(task)
        _validate_output(Path(task['run_dir']).resolve(), audit_reports(output_root))
    spec = MODEL_PAIRS['qwen3']
    # One target-only row per method; do not duplicate baselines for draft roles.
    return reporting.summarize(
        tasks, audit_reports(output_root), methods=METHODS, baseline_methods=METHODS,
        describe=lambda task: (dict(model_pair=spec.name, adapter=spec.adapter,
                                    target_model=spec.target), ('target_only',)),
        title='Qwen3 epoch-1 seven baselines (condition seeds)', read_result=read_result,
        execution_root=audit_executions(output_root),
    )


def execute_worker(task):
    validate_scope(task)
    engine.execute_worker(task)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=('dry-run', 'status', 'run', 'summarize', 'worker'))
    parser.add_argument('--model-root', type=Path, default=MODEL_PAIRS['qwen3'].run_root,
                        help='training run root containing DATASET/epoch1/seedSEED/results.json')
    parser.add_argument('--output-root', type=Path, default=OUTPUT_ROOT,
                        help='task root for this separate baseline batch')
    parser.add_argument('--gpus', nargs='+', type=int, default=[0],
                        help='physical GPU indices; one condition at a time per GPU')
    parser.add_argument('--task-file', type=Path, help='internal saved worker request')
    args = parser.parse_args()
    if args.command == 'worker':
        if args.task_file is None:
            parser.error('worker requires --task-file')
        execute_worker(json.loads(args.task_file.read_text()))
        return
    if len(args.gpus) != len(set(args.gpus)) or any(gpu < 0 for gpu in args.gpus):
        parser.error('choose distinct nonnegative GPU indices')
    args.output_root = args.output_root.resolve()
    try:
        tasks = make_tasks(args.output_root, model_root=args.model_root)
    except ValueError as error:
        parser.error(str(error))
    if args.command in ('dry-run', 'status'):
        payload = dict(
            model_pair='qwen3', target_epoch=1, benchmarks=BENCHMARKS, methods=METHODS,
            worker_tasks=len(tasks), expected_method_rows=len(tasks) * len(METHODS),
            main_tasks=0, data_counts=dict(member=2000, nonmember=2000, audit_auxiliary=600),
            auxiliary_partitions=dict(reference=400, calibration=200),
            seed_mapping=[dict(training_data_seed=seed, baseline_seed=seed,
                               auxiliary_split_seed=seed, auc_bootstrap_seed=seed) for seed in SEEDS],
            output_root=str(args.output_root), summary_directory=str(audit_reports(args.output_root)),
        )
        if args.command == 'dry-run':
            # Planning only: no model/data reads, output directories or GPU calls.
            payload['tasks'] = tasks
        else:
            payload['states'] = {task['id']: inspect_task(task) for task in tasks}
        print(json.dumps(payload, indent=2))
        if any(state['state'] in ('pending', 'invalid', 'stale')
               for state in payload.get('states', {}).values()):
            raise SystemExit(2)
        return
    success = True
    if args.command == 'run':
        success = scheduler.run_tasks(tasks, args.output_root, args.gpus, inspect_task=inspect_task,
                                      check_gpus=check_gpus, worker_module=__spec__.name)
    result = summarize(tasks, args.output_root)
    print(json.dumps({key: result[key] for key in ('complete', 'expected_rows', 'completed_rows', 'errors')}))
    if not success or not result['complete']:
        raise SystemExit(2)


if __name__ == '__main__':
    main()
