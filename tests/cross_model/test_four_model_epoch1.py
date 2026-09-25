"""Fixed four-model main-only matrix and isolation from an active Qwen batch."""
import copy
import json
from pathlib import Path

import pytest

from experiments.main_method_validity import four_model_epoch1 as experiment
from experiments.paths import AUDITS
from experiments.shared.audit.provenance import runtime_files
from tests.sft.test_qwen_audit_matrix import write_example_result


def tasks_at(tmp_path):
    return experiment.make_tasks(tmp_path / 'four_model_tasks')


def test_exact_36_main_tasks_use_auxiliary_drafts_and_condition_seeds(tmp_path):
    tasks = tasks_at(tmp_path)
    assert len(tasks) == len({task['id'] for task in tasks}) == 36
    assert {(task['model_pair'], task['condition']['benchmark'], task['condition']['condition_seed'])
            for task in tasks} == {(pair, benchmark, seed)
                                  for pair in experiment.PAIR_ROLES
                                  for benchmark in experiment.BENCHMARKS
                                  for seed in experiment.SEEDS}
    for task in tasks:
        assert task['condition']['epoch'] == 1
        assert task['kind'] == 'main'
        assert task['protocol'] == 'fixed'
        assert task['methods'] == [experiment.METHOD]
        assert task['draft_role'] == experiment.PAIR_ROLES[task['model_pair']]
        assert task['settings']['audit_seed'] == task['condition']['condition_seed']
        assert task['settings']['seed_policy'] == 'condition_v1'
        assert task['settings']['selector_source_sha256'] == experiment._source_sha256()
    assert not (tmp_path / 'four_model_tasks').exists()
    assert Path(experiment.__file__) not in runtime_files()


@pytest.mark.parametrize('change', ['baseline', 'member_draft', 'epoch', 'model', 'seed',
                                     'method', 'source', 'output'])
def test_changed_worker_task_is_rejected_before_engine_execution(tmp_path, monkeypatch, change):
    task = copy.deepcopy(tasks_at(tmp_path)[0])
    if change == 'baseline':
        task.update(kind='baseline', methods=['loss'])
    elif change == 'member_draft':
        task['draft_role'] = 'draft_member_sft'
    elif change == 'epoch':
        task['condition']['epoch'] = 3
    elif change == 'model':
        task['model_pair'] = 'qwen3'
    elif change == 'seed':
        task['settings']['audit_seed'] = 20260914
    elif change == 'method':
        task['methods'].append('loss')
    elif change == 'source':
        task['settings']['selector_source_sha256'] = 'old'
    else:
        task['output'] = str(tmp_path / 'other' / task['id'])
    monkeypatch.setattr(experiment.engine, 'execute_worker', lambda _: pytest.fail('engine ran'))
    with pytest.raises(ValueError):
        experiment.execute_worker(task)
    assert not (tmp_path / 'other').exists()


def test_output_protection_and_batch_marker(tmp_path, monkeypatch):
    for batch in ('qwen_kd_epoch1_condition_seed_v1', 'qwen_condition_seed_v1',
                  'cross_model_condition_seed_v1'):
        with pytest.raises(ValueError, match='separate'):
            experiment.make_tasks(AUDITS / batch / 'tasks')
    out = tmp_path / 'new' / 'tasks'
    experiment._check_batch(out, create=True)
    assert (out / 'BATCH.json').is_file()
    experiment._check_batch(out)
    monkeypatch.setattr(experiment, '_source_sha256', lambda: 'changed')
    with pytest.raises(ValueError, match='source changed'):
        experiment._check_batch(out)


def test_existing_unmarked_audit_batch_is_not_adopted(tmp_path, monkeypatch):
    audit_root = tmp_path / 'audits'
    monkeypatch.setattr(experiment, 'AUDITS', audit_root)
    old = audit_root / 'another_batch'
    (old / 'reports').mkdir(parents=True)
    with pytest.raises(ValueError, match='unmarked audit batch'):
        experiment._check_batch(old / 'tasks', create=True)
    assert not (old / 'tasks').exists()


def test_dry_run_is_read_only_and_has_no_baselines(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(experiment.engine, 'ready', lambda _: (True, 'ready'))
    monkeypatch.setattr(experiment.scheduler, 'run_tasks', lambda *a, **k: pytest.fail('scheduled work'))
    out = tmp_path / 'new_tasks'
    monkeypatch.setattr('sys.argv', ['four_model_epoch1', 'dry-run', '--output-root', str(out)])
    experiment.main()
    result = json.loads(capsys.readouterr().out)
    assert result['worker_tasks'] == result['expected_method_rows'] == 36
    assert result['baseline_tasks'] == result['member_draft_tasks'] == 0
    assert len(result['states']) == 36
    assert all(state['state'] == 'ready' for state in result['states'].values())
    assert all(set(mapping.values()) == {seed}
               for mapping, seed in zip(result['seed_mapping'], experiment.SEEDS))
    assert not out.exists()


def test_summary_keeps_four_models_and_three_seeds_separate(tmp_path):
    tasks = tasks_at(tmp_path)
    out = tmp_path / 'four_model_tasks'
    for task in tasks:
        value = {1919: .7, 1949: .8, 1978: .9}[task['condition']['condition_seed']]
        write_example_result(task, experiment.METHOD, value)
    result = experiment.summarize(tasks, out)
    assert result['complete'] and result['expected_rows'] == result['completed_rows'] == 36
    assert len(result['seed_summary']) == 12
    assert all(row['expected_seeds'] == row['completed_seeds'] == 3
               and row['auc_mean'] == pytest.approx(.8) for row in result['seed_summary'])
    assert all(not row['reused_target_only'] for row in result['rows'])


def test_run_dispatches_only_36_main_tasks(tmp_path, monkeypatch, capsys):
    dispatched = []
    def run(tasks, output_root, gpus, **kwargs):
        dispatched.extend(tasks)
        assert gpus == [0, 2]
        assert kwargs['worker_module'] == experiment.__name__
        return True
    monkeypatch.setattr(experiment.scheduler, 'run_tasks', run)
    monkeypatch.setattr(experiment, 'summarize', lambda *a: dict(
        complete=True, expected_rows=36, completed_rows=36, errors=[]))
    out = tmp_path / 'new_tasks'
    monkeypatch.setattr('sys.argv', ['four_model_epoch1', 'run', '--output-root', str(out),
                                   '--gpus', '0', '2'])
    experiment.main()
    assert len(dispatched) == 36
    assert {task['kind'] for task in dispatched} == {'main'}
    assert (out / 'BATCH.json').exists()
    assert json.loads(capsys.readouterr().out)['complete']
