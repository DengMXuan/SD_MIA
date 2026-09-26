"""Check launch boundaries without starting experiments or accessing a GPU."""
import json
import os
from pathlib import Path
import subprocess
import sys

import pytest

from experiments.paths import ROOT
from experiments.pretraining import matrix


@pytest.fixture
def fake_python(tmp_path):
    script = tmp_path / 'python'
    script.write_text(f'#!{sys.executable}\nimport json, os, sys\n'
                      'print(json.dumps(dict(argv=sys.argv[1:], cwd=os.getcwd())))\n')
    script.chmod(0o755)
    return {**os.environ, 'SD_AUDIT_PYTHON': str(script)}


@pytest.mark.parametrize('script,module,prefix', (
    ('pretraining/scripts/run_pythia_mimir.sh', 'experiments.pretraining.matrix', ['mimir']),
    ('pretraining/scripts/run_qwen_temporal.sh', 'experiments.pretraining.matrix', ['temporal']),
    ('dp_defense/scripts/train_qwen_epoch1_kd.sh', 'experiments.dp_defense.sweep', ['dry-run']),
    ('dp_defense/scripts/run_qwen_epoch1_kd_main.sh', 'experiments.dp_defense.sweep', ['dry-run']),
))
def test_wrappers_default_preview_from_any_directory(fake_python, tmp_path, script, module, prefix):
    result = subprocess.run(['bash', str(ROOT / 'experiments' / script)], cwd=tmp_path,
                            env=fake_python, capture_output=True, text=True, check=True)
    call = json.loads(result.stdout)
    assert call['cwd'] == str(ROOT)
    assert call['argv'][:4] == ['-B', '-u', '-m', module]
    assert call['argv'][4:4 + len(prefix)] == prefix


@pytest.mark.parametrize('gpu_args', [['--gpu', '2'], ['--gpus', '2', '3', '--workers', '2', '--log-root', '/tmp/logs']])
@pytest.mark.parametrize('script,stage', (
    ('train_qwen_epoch1_kd.sh', 'train'), ('run_qwen_epoch1_kd_main.sh', 'audit'),
))
def test_dp_stage_isolation_and_forwarding(fake_python, tmp_path, script, stage, gpu_args):
    filters = ['--seeds', '1949', '--benchmarks', 'arxivtection', '--epsilons', '4', *gpu_args]
    result = subprocess.run(['bash', str(ROOT / 'experiments/dp_defense/scripts' / script), 'run', *filters],
                            cwd=tmp_path, env=fake_python, capture_output=True, text=True, check=True)
    args = json.loads(result.stdout)['argv']
    assert args[4:] == [stage, '--model-pairs', 'qwen3', '--epochs', '1', '--draft-variants', 'kd', *filters]


@pytest.mark.parametrize('option', ['--epochs', '--epoc', '--model-pairs', '--draft-v', '--include-b'])
def test_dp_fixed_condition_cannot_be_overridden(fake_python, tmp_path, option):
    for script in ('train_qwen_epoch1_kd.sh', 'run_qwen_epoch1_kd_main.sh'):
        result = subprocess.run(['bash', str(ROOT / 'experiments/dp_defense/scripts' / script), 'run', option, '3'],
                                cwd=tmp_path, env=fake_python, capture_output=True, text=True)
        assert result.returncode == 2 and not result.stdout


@pytest.mark.parametrize('experiment,count', [('mimir', 24), ('temporal', 3)])
def test_default_matrix_dry_run_has_no_worker_or_output(monkeypatch, tmp_path, capsys, experiment, count):
    def fake_plan(args):
        sources = args.sources if experiment == 'mimir' else ['wikitection']
        return [dict(source=source, seed=seed) for source in sources for seed in args.seeds]
    monkeypatch.setattr(matrix, 'plan', fake_plan)
    monkeypatch.setattr(matrix.gpu_pool, 'run_jobs', lambda *a, **k: pytest.fail('dry-run launched a worker'))
    matrix.main([experiment, '--output-root', str(tmp_path / 'audit')])
    output = json.loads(capsys.readouterr().out)
    assert output['conditions'] == count
    assert {t['seed'] for t in output['tasks']} == {1919, 1949, 1978}
    assert not (tmp_path / 'audit').exists()


def test_prepare_never_calls_evaluator_and_run_passes_condition(monkeypatch, tmp_path):
    from experiments.pretraining import evaluation, temporal_reuse
    prepared, evaluated = [], []
    monkeypatch.setattr(temporal_reuse, 'prepare_from_shared_split', lambda *a, **k: prepared.append((a, k)))
    def evaluate(*args, **kwargs):
        evaluated.append((args, kwargs))
        return dict(metrics={})
    monkeypatch.setattr(evaluation, 'evaluate_main', evaluate)
    task = dict(source='wikitection', seed=1978, history='history', shared='shared', pool='pool',
                manifest=str(tmp_path / 'data/manifest.json'), output=str(tmp_path / 'audit'),
                gpu=2, detector_epochs=30)
    matrix.worker(task, prepare_only=True)
    assert len(prepared) == 1 and not evaluated
    matrix.worker(task)
    assert prepared[-1][1] == dict(seed=1978)
    assert evaluated == [((task['manifest'], task['output']),
                          dict(seed=1978, device='cuda:2', detector_epochs=30))]


def test_matrix_propagates_failures_and_preserves_each_condition(monkeypatch, tmp_path):
    tasks = [dict(source='arxiv', seed=seed) for seed in matrix.SEEDS]
    monkeypatch.setattr(matrix, 'plan', lambda args: tasks)
    launched = []
    def run(jobs, **kwargs):
        launched.extend(json.loads(job.command[-1]) for job in jobs)
        assert [job.seed for job in jobs] == list(matrix.SEEDS)
        assert kwargs['scheduling']['active_gpus'] == [2, 3]
        return [dict(state='failed', id=jobs[0].id, exit_code=1)]
    monkeypatch.setattr(matrix.gpu_pool, 'run_jobs', run)
    with pytest.raises(SystemExit) as error:
        matrix.main(['mimir', 'run', '--output-root', str(tmp_path), '--gpus', '2', '3', '--workers', '2'])
    assert error.value.code == 2 and launched == tasks


@pytest.mark.parametrize('stage', ['train', 'audit'])
def test_dp_pool_preserves_full_matrix_and_runs_only_selected_stage(monkeypatch, tmp_path, stage):
    from experiments.dp_defense import sweep
    calls, schedules = [], []
    def run(jobs, **kwargs):
        calls.append(jobs)
        schedules.append(kwargs['scheduling'])
        return [dict(id=job.id, state='complete') for job in jobs]
    monkeypatch.setattr(sweep.gpu_pool, 'run_jobs', run)
    sweep.main([stage, '--epochs', '1', '--draft-variants', 'kd',
                '--gpus', '1', '3', '--workers', '2', '--log-root', str(tmp_path)])
    sweep.main([stage, '--epochs', '1', '--draft-variants', 'kd', '--gpu', '2', '--log-root', str(tmp_path)])
    assert calls[0] == calls[1]
    assert [s['active_gpus'] for s in schedules] == [[1, 3], [2]]
    jobs = calls[0]
    assert len(jobs) == len({job.id for job in jobs}) == 27
    assert {job.seed for job in jobs} == {1919, 1949, 1978}
    for job in jobs:
        assert f'experiments.dp_defense.{stage}' in job.command
        assert job.command[-2:] == ['--draft-variants', 'kd']
        flag, device = ('--gpu', '0') if stage == 'train' else ('--device', 'cuda:0')
        assert job.command[job.command.index(flag) + 1] == device
        assert f'/seed{job.seed}' in job.id and '/epoch1/' in job.id


def test_pretraining_worker_requests_stay_identical_when_gpu_pool_changes(monkeypatch, tmp_path):
    # Scheduling changes must not alter seed, output path, or the worker device
    # embedded in the provenance-bound experiment request.
    from experiments.pretraining.data import TARGET, DRAFT, TOKEN_CONTRACT, sha256
    folder = tmp_path / 'mimir/prepared/arxiv/seed1919'
    folder.mkdir(parents=True)
    (folder / 'records.jsonl').write_text('fixture')
    (folder / 'manifest.json').write_text(json.dumps(dict(kind='mimir_pretraining_v1',
        selection_seed=1919, models=dict(target=TARGET, draft=DRAFT), max_tokens=512,
        counts=dict(member=400, nonmember=400, auxiliary=600), token_contract=TOKEN_CONTRACT,
        benchmark='mimir/arxiv/ngram_13_0.8', records_file='records.jsonl',
        records_sha256=sha256(folder / 'records.jsonl'))))
    calls = []
    def run(jobs, **kwargs):
        calls.append(jobs)
        return [dict(state='complete')]
    monkeypatch.setattr(matrix.gpu_pool, 'run_jobs', run)
    common = ['mimir', 'run', '--data-root', str(tmp_path), '--output-root', str(tmp_path / 'audit'),
              '--sources', 'arxiv', '--seeds', '1919']
    matrix.main([*common, '--gpu', '2'])
    matrix.main([*common, '--gpus', '0', '1', '3', '--workers', '3'])
    assert calls[0] == calls[1]
    assert json.loads(calls[0][0].command[-1])['gpu'] == 0


def test_manifest_mismatch_is_rejected(tmp_path):
    from experiments.pretraining.data import TOKEN_CONTRACT, sha256
    records = tmp_path / 'records.jsonl'
    records.write_text('fixture')
    manifest = tmp_path / 'manifest.json'
    saved = dict(kind='mimir_pretraining_v1', selection_seed=1919, models={}, counts={},
                 token_contract=TOKEN_CONTRACT, max_tokens=512,
                 records_file=records.name, records_sha256=sha256(records))
    manifest.write_text(json.dumps(saved))
    args = (manifest, 1919, 'mimir_pretraining_v1', {}, {})
    assert matrix._checked_manifest(*args) == saved
    with pytest.raises(ValueError, match='contract mismatch'):
        matrix._checked_manifest(manifest, 1949, 'mimir_pretraining_v1', {}, {})
    records.write_text('changed')
    with pytest.raises(ValueError, match='checksum mismatch'):
        matrix._checked_manifest(*args)


@pytest.mark.parametrize('args', [
    ['temporal', '--sources', 'arxiv'], ['mimir', '--reference-root', '/tmp'],
    ['mimir', '--seeds', '1919', '1919'], ['temporal', '--gpu', '-1'],
    ['mimir', '--gpus', '0', '0'], ['temporal', '--gpus', '0', '1', '--workers', '3'],
    ['mimir', '--workers', '0'], ['temporal', '--gpu', '0', '--gpus', '1'],
])
def test_invalid_selections_fail_before_planning(monkeypatch, args):
    monkeypatch.setattr(matrix, 'plan', lambda args: pytest.fail('planned invalid request'))
    with pytest.raises(SystemExit) as error:
        matrix.main(args)
    assert error.value.code == 2
