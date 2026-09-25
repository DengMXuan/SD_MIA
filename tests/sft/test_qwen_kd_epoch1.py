"""Restricted Qwen rerun scope, condition seeds, safe output and main-only summaries."""
import copy
import json
from pathlib import Path

import pytest

from experiments.sd_membership_sft.audit import qwen_kd_epoch1 as experiment
from experiments.shared.audit.artifacts import runtime_files
from tests.sft.test_qwen_audit_matrix import write_example_result


def tasks_at(tmp_path):
    return experiment.make_tasks(tmp_path / 'output', model_root=tmp_path / 'models')


def test_exact_nine_conditions_with_paired_seeds_and_no_other_methods(tmp_path):
    tasks = tasks_at(tmp_path)
    assert len(tasks) == len({t['id'] for t in tasks}) == 9
    assert {(t['condition']['benchmark'], t['condition']['condition_seed']) for t in tasks} == {
        (benchmark, seed) for benchmark in experiment.BENCHMARKS for seed in experiment.SEEDS}
    for task in tasks:
        assert task['condition']['epoch'] == 1
        assert task['model_pair'] == 'qwen3'
        assert task['kind'] == 'main'
        assert task['draft_role'] == 'draft_auxiliary_distilled'
        assert task['methods'] == ['main_fixed_sparse_positive']
        assert task['settings']['audit_seed'] == task['condition']['condition_seed']
        assert task['settings']['seed_policy'] == 'condition_v1'
    assert not (tmp_path / 'output').exists()
    assert Path(experiment.__file__) in runtime_files()


@pytest.mark.parametrize('change', ['epoch', 'draft', 'baseline', 'model', 'seed'])
def test_saved_worker_cannot_expand_scope(tmp_path, monkeypatch, change):
    task = copy.deepcopy(tasks_at(tmp_path)[0])
    if change == 'epoch':
        task['condition']['epoch'] = 3
    elif change == 'draft':
        task['draft_role'] = 'draft_member_sft'
    elif change == 'baseline':
        task.update(kind='baseline', methods=['loss'])
    elif change == 'model':
        task['model_pair'] = 'gemma4'
    else:
        task['settings']['audit_seed'] = 20260914
    monkeypatch.setattr(experiment.engine, 'execute_worker', lambda _: pytest.fail('must reject before launching'))
    with pytest.raises(ValueError):
        experiment.execute_worker(task)
    assert not Path(task['output']).exists()


def test_dry_run_is_read_only_and_displays_all_four_matching_seeds(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(experiment.engine, 'ready', lambda _: (True, 'ready'))
    monkeypatch.setattr(experiment.scheduler, 'run_tasks', lambda *a, **k: pytest.fail('dry-run scheduled work'))
    monkeypatch.setattr('sys.argv', ['qwen_kd_epoch1', 'dry-run', '--output-root', str(tmp_path / 'out')])
    experiment.main()
    result = json.loads(capsys.readouterr().out)
    assert result['worker_tasks'] == result['expected_method_rows'] == 9
    assert result['baseline_tasks'] == 0
    assert len(result['states']) == 9
    assert all(s['state'] == 'ready' for s in result['states'].values())
    for row, seed in zip(result['seed_mapping'], experiment.SEEDS):
        assert set(row.values()) == {seed}
    assert not (tmp_path / 'out').exists()


def test_main_only_summary_has_nine_rows_and_three_independent_seeds_per_dataset(tmp_path):
    tasks = tasks_at(tmp_path)
    empty = experiment.summarize(tasks, tmp_path / 'output')
    assert not empty['complete'] and empty['expected_rows'] == 9 and empty['completed_rows'] == 0
    for task in tasks:
        value = {1919: .7, 1949: .8, 1978: .9}[task['condition']['condition_seed']]
        write_example_result(task, experiment.METHOD, value)
    result = experiment.summarize(tasks, tmp_path / 'output')
    assert result['complete'] and result['expected_rows'] == result['completed_rows'] == 9
    assert len(result['seed_summary']) == 3
    assert result['unique_successful_execution_groups'] == 9
    assert result['unique_successful_measured_method_seconds'] == 90
    for row in result['seed_summary']:
        assert row['expected_seeds'] == row['completed_seeds'] == 3
        assert row['auc_mean'] == pytest.approx(.8)
        assert row['auc_std'] == pytest.approx(.1)
    assert all(not row['reused_target_only'] for row in result['rows'])
    scores = Path(tasks[0]['output']) / experiment.METHOD / 'scores.npz'
    with scores.open('ab') as stream:
        stream.write(b'modified')
    invalid = experiment.summarize(tasks, tmp_path / 'output')
    assert not invalid['complete'] and invalid['completed_rows'] == 8 and len(invalid['errors']) == 1


def test_historical_output_alias_is_rejected_without_writing(tmp_path):
    from experiments.paths import QWEN_AUDIT
    alias = tmp_path / 'old'
    alias.symlink_to(QWEN_AUDIT, target_is_directory=True)
    with pytest.raises(ValueError, match='separate'):
        experiment.make_tasks(alias)


def test_run_dispatches_only_nine_tasks_through_the_shared_scheduler(tmp_path, monkeypatch, capsys):
    dispatched = []
    def run(tasks, output_root, gpus, **kwargs):
        dispatched.extend(tasks)
        assert gpus == [1, 3]
        assert kwargs['worker_module'] == experiment.__name__
        return True
    monkeypatch.setattr(experiment.scheduler, 'run_tasks', run)
    monkeypatch.setattr(experiment, 'summarize', lambda *a: dict(
        complete=True, expected_rows=9, completed_rows=9, errors=[]))
    monkeypatch.setattr('sys.argv', ['qwen_kd_epoch1', 'run', '--output-root', str(tmp_path / 'out'),
                                   '--gpus', '1', '3'])
    experiment.main()
    assert len(dispatched) == 9
    assert {task['kind'] for task in dispatched} == {'main'}
    assert json.loads(capsys.readouterr().out)['complete']
