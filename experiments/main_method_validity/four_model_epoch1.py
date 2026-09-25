"""Epoch-1 auxiliary-draft main method for the four non-Qwen3 model pairs."""
import argparse
import json
from pathlib import Path

from experiments.paths import AUDITS, audit_executions, audit_reports
from experiments.cross_model_audit import engine
from experiments.shared.audit import reporting, scheduler
from experiments.shared.audit.config import audit_settings
from experiments.shared.audit.devices import check_gpus
from experiments.shared.audit.evaluation import _validate_output
from experiments.shared.audit.provenance import read_result
from experiments.shared.core.audit_runtime import _write_json
from experiments.shared.core.deployment_archive import sha256_file
from experiments.shared.models.registry import MODEL_PAIRS

PAIR_ROLES = {
    'gemma4': 'draft_auxiliary_distilled',
    'qwen3_8b_eagle3': 'auxiliary_head',
    'llama31_8b_eagle3': 'auxiliary_head',
    'qwen35_9b_mtp': 'auxiliary_head',
}
BENCHMARKS = ('wikitection', 'newstection', 'arxivtection')
SEEDS = (1919, 1949, 1978)
METHOD = 'main_fixed_sparse_positive'
OUTPUT_ROOT = AUDITS / 'four_model_epoch1_main_condition_seed_v1/tasks'
_OTHER_BATCHES = (
    AUDITS / 'qwen_kd_epoch1_condition_seed_v1',
    AUDITS / 'qwen_condition_seed_v1',
    AUDITS / 'cross_model_condition_seed_v1',
)


def _source_sha256():
    # This standalone selector sits outside the shared/cross-model provenance
    # roots, so its digest is bound to each task request separately.
    return sha256_file(Path(__file__))


def _output_guard(output_root):
    output_root = Path(output_root).resolve()
    if output_root == AUDITS or AUDITS.is_relative_to(output_root):
        raise ValueError('choose an isolated audit task root, not its parent')
    if output_root.is_relative_to(AUDITS) and (output_root.name != 'tasks'
                                               or output_root.parent.parent != AUDITS):
        raise ValueError('an audit output root must be <batch>/tasks')
    for batch in _OTHER_BATCHES:
        batch = batch.resolve()
        if (output_root == batch or output_root.is_relative_to(batch)
                or batch.is_relative_to(output_root)):
            raise ValueError('four-model output must be separate from existing audit batches')
    for name in PAIR_ROLES:
        run_root = MODEL_PAIRS[name].run_root.resolve()
        for destination in (output_root, audit_reports(output_root), audit_executions(output_root)):
            _validate_output(run_root, destination)
    return output_root


def _batch_manifest(output_root):
    return dict(schema='four_model_epoch1_main_condition_seed_v1',
                output_root=str(output_root), pairs=PAIR_ROLES,
                benchmarks=list(BENCHMARKS), seeds=list(SEEDS), epoch=1,
                method=METHOD, source_sha256=_source_sha256())


def _check_batch(output_root, *, create=False, require_marker=False):
    output_root = _output_guard(output_root)
    marker = output_root / 'BATCH.json'
    expected = _batch_manifest(output_root)
    if marker.exists():
        if json.loads(marker.read_text()) != expected:
            raise ValueError('four-model batch identity or selector source changed; use a new output root')
    elif require_marker:
        raise ValueError('four-model worker requires a coordinator-created batch marker')
    elif output_root.is_relative_to(AUDITS) and any(
            (output_root.parent / name).exists() for name in ('reports', 'executions', 'intermediate')):
        raise ValueError('refusing to adopt an existing unmarked audit batch')
    elif output_root.exists() and any(output_root.iterdir()):
        raise ValueError('refusing to adopt an unmarked existing audit directory')
    elif create:
        output_root.mkdir(parents=True, exist_ok=True)
        _write_json(marker, expected)


def make_tasks(output_root=OUTPUT_ROOT, *, detector_epochs=30):
    if type(detector_epochs) is not int or detector_epochs < 1:
        raise ValueError('detector epochs must be positive')
    output_root = _output_guard(output_root)
    tasks = []
    for name, role in PAIR_ROLES.items():
        spec = MODEL_PAIRS[name]
        if role not in spec.roles or spec.roles[0] != role:
            raise ValueError(f'auxiliary draft role changed for {name}')
        candidates = engine.make_tasks(spec.run_root.resolve(), output_root, BENCHMARKS, [1], SEEDS,
                                       audit_settings(detector_epochs=detector_epochs), model_pair=name)
        for task in candidates:
            if task['kind'] == 'main' and task['draft_role'] == role:
                task['settings']['selector_source_sha256'] = _source_sha256()
                tasks.append(task)
    if len(tasks) != 36 or len({task['id'] for task in tasks}) != 36:
        raise ValueError('expected 36 distinct epoch-1 auxiliary-draft main tasks')
    return tasks


def validate_scope(task):
    """Reject changed worker requests before any model loading or output write."""
    try:
        task_id = Path(task['id'])
        output = Path(task['output']).resolve()
        if not task_id.parts or tuple(output.parts[-len(task_id.parts):]) != task_id.parts:
            raise ValueError('task output does not match its identity')
        output_root = output.parents[len(task_id.parts) - 1]
        epochs = task['settings']['detector_epochs']
        expected = {row['id']: row for row in make_tasks(output_root, detector_epochs=epochs)}
        if task != expected.get(task['id']):
            raise ValueError('task is outside the fixed four-model main-method scope')
    except (KeyError, IndexError, TypeError) as error:
        raise ValueError('invalid four-model task identity') from error


def inspect_task(task):
    validate_scope(task)
    return engine.inspect_task(task)


def summarize(tasks, output_root):
    for task in tasks:
        validate_scope(task)
    return reporting.summarize(
        tasks, audit_reports(output_root), methods=(METHOD,), baseline_methods=(),
        describe=lambda task: (
            dict(model_pair=task['model_pair'], adapter=MODEL_PAIRS[task['model_pair']].adapter,
                 target_model=MODEL_PAIRS[task['model_pair']].target,
                 draft_model=MODEL_PAIRS[task['model_pair']].draft),
            (PAIR_ROLES[task['model_pair']],)),
        title='Four-model epoch-1 auxiliary-draft main method (condition seeds)',
        read_result=read_result, execution_root=audit_executions(output_root),
    )


def execute_worker(task):
    validate_scope(task)
    task_id = Path(task['id'])
    output_root = Path(task['output']).resolve().parents[len(task_id.parts) - 1]
    _check_batch(output_root, require_marker=True)
    engine.execute_worker(task)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=('dry-run', 'status', 'run', 'summarize', 'worker'))
    parser.add_argument('--output-root', type=Path, default=OUTPUT_ROOT)
    parser.add_argument('--gpus', nargs='+', type=int, default=[0])
    parser.add_argument('--detector-epochs', type=int, default=30)
    parser.add_argument('--task-file', type=Path)
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
        tasks = make_tasks(args.output_root, detector_epochs=args.detector_epochs)
        _check_batch(args.output_root)
    except ValueError as error:
        parser.error(str(error))
    if args.command in ('dry-run', 'status'):
        states = {task['id']: inspect_task(task) for task in tasks}
        print(json.dumps(dict(worker_tasks=len(tasks), expected_method_rows=len(tasks),
                              baseline_tasks=0, member_draft_tasks=0, model_pairs=list(PAIR_ROLES),
                              target_epoch=1, roles=PAIR_ROLES,
                              seed_mapping=[dict(training_data_seed=seed, main_seed=seed,
                                                 auxiliary_split_seed=seed, auc_bootstrap_seed=seed)
                                            for seed in SEEDS],
                              output_root=str(args.output_root),
                              summary_directory=str(audit_reports(args.output_root)),
                              states=states), indent=2))
        if any(state['state'] in ('pending', 'invalid', 'stale') for state in states.values()):
            raise SystemExit(2)
        return
    _check_batch(args.output_root, create=True)
    success = True
    if args.command == 'run':
        success = scheduler.run_tasks(tasks, args.output_root, args.gpus, inspect_task=inspect_task,
                                      check_gpus=check_gpus,
                                      worker_module='experiments.main_method_validity.four_model_epoch1')
    result = summarize(tasks, args.output_root)
    print(json.dumps({key: result[key] for key in ('complete', 'expected_rows', 'completed_rows', 'errors')}))
    if not success or not result['complete']:
        raise SystemExit(2)


if __name__ == '__main__':
    main()
