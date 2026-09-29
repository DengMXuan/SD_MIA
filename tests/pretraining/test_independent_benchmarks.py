"""Independent plans, stages, provenance and failures without any GPU/model."""
from dataclasses import replace
import json
from pathlib import Path

import pytest

from experiments.launchers import pretraining as runner
from experiments.pretraining.benchmarks import mimir13, mimir7, wikimia
from experiments.pretraining.data import TARGET, DRAFT, TOKEN_CONTRACT, sha256
from standalone.pretraining_baselines.contract import MAIN


def frozen_spec(tmp_path, spec):
    spec = replace(spec, data_root=tmp_path / 'data',
                   historical_main=tmp_path / 'old_main' if spec.historical_main else None)
    for source in spec.sources:
        for seed in runner.SEEDS:
            manifest = spec.data_root / spec.manifest_pattern.format(source=source, seed=seed)
            manifest.parent.mkdir(parents=True, exist_ok=True)
            m, n = spec.counts[source]
            counts = dict(auxiliary=600, member=m, nonmember=n)
            rows = [dict(record_id=f'{group}-{i}', group=group, label=int(group == 'member'))
                    for group, count in counts.items() for i in range(count)]
            records = manifest.with_name('records.jsonl')
            records.write_text(''.join(json.dumps(row) + '\n' for row in rows))
            manifest.write_text(json.dumps(dict(kind=spec.kind, selection_seed=seed,
                token_contract=TOKEN_CONTRACT, counts=counts, records_file=records.name,
                records_sha256=sha256(records), benchmark=spec.benchmark_pattern.format(source=source),
                models=dict(target=TARGET, draft=DRAFT), max_tokens=512)))
    return spec


@pytest.mark.parametrize('spec,count', [(mimir13.SPEC, 21), (mimir7.SPEC, 6), (wikimia.SPEC, 6)])
def test_each_benchmark_plans_with_only_its_own_data(tmp_path, spec, count, capsys, monkeypatch):
    spec = frozen_spec(tmp_path, spec)
    output = tmp_path / 'results'
    monkeypatch.setattr(runner.gpu_pool, 'run_jobs', lambda *a, **k: pytest.fail('preview started a worker'))
    runner.run(spec, ['baseline', 'dry-run', '--output-root', str(output), '--gpus', '2', '3'])
    result = json.loads(capsys.readouterr().out)
    assert len(result['tasks']) == count and result['gpu_started'] is False
    assert result['scheduling']['active_gpus'] == [2, 3]
    assert {t['experiment'] for t in result['tasks']} == {spec.legacy_group}
    assert not output.exists()


def test_corrupted_frozen_data_and_duplicate_selection_are_rejected(tmp_path):
    spec = frozen_spec(tmp_path, mimir7.SPEC)
    with pytest.raises(SystemExit) as error:
        runner.run(spec, ['main', '--sources', 'github', 'github'])
    assert error.value.code == 2
    records = spec.data_root / 'github/seed1919/records.jsonl'
    records.write_text(records.read_text() + '{}\n')
    with pytest.raises(ValueError, match='checksum'):
        runner.plan(spec, ['github'], [1919], output_root=tmp_path / 'results')


def test_worker_roles_are_independent_and_historical_main_is_protected(tmp_path, monkeypatch):
    from experiments.launchers import devices
    from experiments.pretraining import evaluation
    from standalone.pretraining_baselines import evaluate
    monkeypatch.setattr(devices, 'check_idle', lambda _: None)
    calls = []
    monkeypatch.setattr(evaluation, 'evaluate_main', lambda *a, **k: calls.append('main'))
    monkeypatch.setattr(evaluate, 'evaluate', lambda *a, **k: calls.append('baseline'))
    spec = frozen_spec(tmp_path, mimir7.SPEC)
    task, = runner.plan(spec, ['github'], [1919], output_root=tmp_path / 'results')
    runner.worker(task, 'main')
    assert calls == ['main']
    runner.worker(task, 'baseline')
    assert calls == ['main', 'baseline']
    task['historical_main'] = True
    with pytest.raises(ValueError, match='read-only'):
        runner.worker(task, 'main')
    with pytest.raises(ValueError, match='historical main result missing'):
        runner.worker(task, 'baseline')
    task['manifest_sha256'] = 'tampered'
    with pytest.raises(ValueError, match='changed'):
        runner.worker(task, 'baseline')


def test_run_requires_devices_and_propagates_worker_failure(tmp_path, monkeypatch):
    from experiments.launchers import devices
    spec = frozen_spec(tmp_path, mimir7.SPEC)
    with pytest.raises(SystemExit) as error:
        runner.run(spec, ['main', 'run'])
    assert error.value.code == 2
    queued = []
    def dispatch(jobs, **kwargs):
        queued.extend(jobs)
        assert kwargs['scheduling']['active_gpus'] == [2, 3]
        return [dict(state='failed')]
    monkeypatch.setattr(devices, 'run_jobs', dispatch)
    monkeypatch.setattr(runner, 'summarize', lambda *a: dict(selected_complete=False))
    with pytest.raises(SystemExit) as error:
        runner.run(spec, ['main', 'run', '--gpus', '2', '3', '--output-root', str(tmp_path / 'results')])
    assert error.value.code == 2
    assert len(queued) == 6
    assert all(job.command[-2] == 'main' for job in queued)
    assert {json.loads(job.command[-1])['experiment'] for job in queued} == {'mimir02'}


def test_summary_fails_on_incomplete_selected_role(tmp_path):
    spec = frozen_spec(tmp_path, mimir7.SPEC)
    for role in ('main', 'baseline'):
        with pytest.raises(SystemExit) as error:
            runner.run(spec, [role, 'summarize', '--output-root', str(tmp_path / 'results')])
        assert error.value.code == 2
    assert (tmp_path / 'results/reports/main/MAIN.json').is_file()
    comparison = json.loads((tmp_path / 'results/reports/baseline/COMPARISON.json').read_text())
    assert comparison['expected_baseline_rows'] == 6 * 7
    assert not comparison['comparison_complete']


def test_mimir7_prepare_uses_explicit_auxiliary_root(tmp_path, monkeypatch, capsys):
    from experiments.launchers import prepare_mimir7
    from experiments.pretraining.benchmarks import prepare

    auxiliary_root = tmp_path / 'pinned_mimir13'
    auxiliary_file = auxiliary_root / 'official/cache_100_200_1000_512/test/github_ngram_13_0.8.jsonl'
    auxiliary_file.parent.mkdir(parents=True)
    auxiliary_file.write_text('"auxiliary record"\n')
    manifest = auxiliary_root / 'prepared/github/seed1919/manifest.json'
    manifest.parent.mkdir(parents=True)
    manifest.write_text(json.dumps({
        'source_provenance': {'revision': prepare.MIMIR_REVISION},
        'source_files': {'nonmember': {
            'path': str(auxiliary_file), 'sha256': sha256(auxiliary_file),
        }},
    }))
    official_root = tmp_path / 'mimir7_official'
    for role in ('train', 'test'):
        source = official_root / f'cache_100_200_1000_512/{role}/github_ngram_7_0.2.jsonl'
        source.parent.mkdir(parents=True)
        source.write_text('"test record"\n')

    frozen = []
    def freeze(member, nonmember, auxiliary, output, **kwargs):
        frozen.append((member, nonmember, auxiliary, output))
        return output / 'manifest.json'
    monkeypatch.setattr(prepare, '_freeze_mimir_with_external_aux', freeze)
    data_root = tmp_path / 'mimir7_prepared'
    prepare_mimir7.main([
        '--source', 'github', '--seeds', '1919', '--data-root', str(data_root),
        '--official-root', str(official_root), '--auxiliary-root', str(auxiliary_root),
    ])
    result = json.loads(capsys.readouterr().out)
    assert result['manifest'] == str(data_root / 'mimir_7_0.2/github/seed1919/manifest.json')
    assert frozen[0][2] == auxiliary_file
    assert frozen[0][0].is_relative_to(official_root)
