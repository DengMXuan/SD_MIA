import copy
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from experiments.resource_curves import qwen_matrix as matrix
from experiments.resource_curves import qwen_data as data
from experiments.resource_curves.config import AuxiliaryBudget
from experiments.resource_curves.partitions import base_partitions
from experiments.resource_curves.storage import code_fingerprint
from experiments.shared.data.data import SFTRecord, _hash_ids
from tests.resource_curves.test_resource_curves import ToyAdapter, shared_split


def args(tmp_path, experiment):
    return SimpleNamespace(experiment=experiment,
        benchmarks=['wikitection', 'arxivtection'] if experiment == 'domain' else list(data.BENCHMARKS),
        seeds=list(data.SEEDS), model_root=tmp_path / 'models', output_root=tmp_path / 'results',
        extension_root=tmp_path / 'extensions', detector_epochs=1, bootstrap=0)


def shared(benchmark, seed=1919):
    value = shared_split(seed)
    value['benchmark'] = benchmark
    for entries in value['splits'].values():
        for row in entries:
            row['record_id'] = benchmark + ':' + row['record_id']
            row['text_sha256'] = row['record_id']
    return value


@pytest.fixture
def fake_preflight(monkeypatch):
    monkeypatch.setattr(matrix, 'read_condition', lambda run, benchmark, seed: dict(shared=shared(benchmark, seed)))
    monkeypatch.setattr(matrix, 'read_extension', lambda *a: None)


@pytest.mark.parametrize('experiment,count,curve_rows', [('auxiliary', 45, 54), ('domain', 6, 6), ('query', 45, 45)])
def test_matrix_counts_scope_and_seed_mapping(tmp_path, fake_preflight, experiment, count, curve_rows):
    tasks = matrix.plan(args(tmp_path, experiment))
    assert len(tasks) == count
    assert len({t['output'] for t in tasks}) == count
    assert sum(len(t['curves']) for t in tasks) == curve_rows
    for task in tasks:
        matrix.validate_task(task)
        assert f"epoch1/seed{task['seed']}" in task['run_dir']
        assert matrix.condition(task)['draft_role'] == 'draft_auxiliary_distilled'
        if experiment == 'domain':
            assert task['donor_run'].endswith(f"newstection/epoch1/seed{task['seed']}")
        if experiment == 'query':
            assert task['budget'] == AuxiliaryBudget().to_dict()
    if experiment == 'auxiliary':
        assert sum(t['curves'] == ['fitting', 'calibration'] for t in tasks) == 9


def test_dry_run_is_read_only_and_queue_passes_seeds(tmp_path, fake_preflight, monkeypatch, capsys):
    root = tmp_path / 'new-output'
    common = ['--output-root', str(root), '--model-root', str(tmp_path / 'models'),
              '--gpus', '1', '3', '--workers', '2']
    matrix.main(['query', 'dry-run', *common])
    output = json.loads(capsys.readouterr().out)
    assert not root.exists()
    assert output['scheduling']['active_gpus'] == [1, 3]
    assert all(len(set(row.values())) == 1 for row in output['seed_mapping'])
    calls = []
    def run(jobs, **kw):
        calls.append((jobs, kw))
        return [dict(state='complete') for j in jobs]
    monkeypatch.setattr(matrix.gpu_pool, 'run_jobs', run)
    monkeypatch.setattr(matrix, 'summarize', lambda tasks, **kw: dict(complete=len(tasks)))
    matrix.main(['query', 'run', *common])
    jobs, kwargs = calls[0]
    assert len(jobs) == 45
    assert kwargs['use_cuda']
    assert kwargs['scheduling']['workers'] == 2
    for job in jobs:
        task = json.loads(job.command[-1])
        assert job.seed == task['seed']
        assert job.id == task['id']
        assert job.command[3:5] == ['-m', matrix.__spec__.name]


@pytest.mark.parametrize('tail', [
    ['--seeds', '1919', '1919'], ['--seeds', '42'], ['--gpus', '0', '0'],
    ['--gpus', '0', '--workers', '2'], ['--gpu', '0', '--gpus', '1'],
    ['--benchmarks', 'newstection'], ['--detector-epochs', '0'],
])
def test_invalid_cli_rejected(tmp_path, fake_preflight, tail):
    with pytest.raises(SystemExit) as error:
        matrix.main(['domain', 'dry-run', '--output-root', str(tmp_path / 'results'), *tail])
    assert error.value.code == 2


def test_domain_roles_match_news_seed_and_target_test():
    for seed in data.SEEDS:
        target, donor = shared('wikitection', seed), shared('newstection', seed)
        study = data.domain_study(target, donor)
        parts = study['points'][0]['partitions']
        assert study['seed'] == seed
        for role in ('train', 'validation', 'calibration'):
            assert parts[role] == base_partitions(donor)[role]
        assert parts['test'] == base_partitions(target)['test']
        assert not set(sum((parts[r] for r in ('train', 'validation', 'calibration')), [])).intersection(
            r['record_id'] for r in target['splits']['audit_auxiliary'])
    wrong = copy.deepcopy(donor)
    wrong['seed'] = 1919
    with pytest.raises(ValueError, match='seed'):
        data.domain_study(target, wrong)
    wrong = copy.deepcopy(donor)
    wrong['splits']['audit_auxiliary'][0]['record_id'] = target['splits']['auxiliary'][0]['record_id']
    with pytest.raises(ValueError, match='overlap'):
        data.domain_study(target, wrong)


def record(record_id, tokens):
    return SFTRecord(record_id=record_id, source='synthetic', response_ids=tuple(tokens),
                     response_hash=_hash_ids(tokens), prompt_ids=(1, 2),
                     prompt_hash=_hash_ids([1, 2]), prompt_text='prompt')


@pytest.mark.parametrize('duplicate', ['raw', 'tokens', 'near'])
def test_cross_domain_leaks_include_target_draft_training(duplicate):
    target, donor = shared('wikitection'), shared('newstection')
    anchor = record('target', range(50))
    tokens = list(range(100, 150))
    if duplicate == 'tokens':
        tokens = list(range(50))
    if duplicate == 'near':
        tokens = list(range(40)) + list(range(200, 210))
    if duplicate == 'raw':
        donor['splits']['audit_auxiliary'][0]['text_sha256'] = target['splits']['auxiliary'][0]['text_sha256']
    ts = SimpleNamespace(members=[], nonmembers=[], draft_auxiliary=[anchor], audit_auxiliary=[])
    ds = SimpleNamespace(audit_auxiliary=[record(donor['splits']['audit_auxiliary'][0]['record_id'], tokens)])
    with pytest.raises(ValueError, match='duplicates'):
        data.check_domain_overlap(target, donor, ts, ds)


def test_domain_prepared_records_exclude_target_auxiliaries():
    target, donor = shared('arxivtection'), shared('newstection')
    def split(value, offset):
        groups = {}
        for i, (role, rows) in enumerate(value['splits'].items()):
            groups[role] = [record(r['record_id'], range(offset + j * 100, offset + j * 100 + 40))
                            for j, r in enumerate(rows)]
            offset += 100000
        return SimpleNamespace(members=groups['member'], nonmembers=groups['nonmember'],
                               draft_auxiliary=groups['auxiliary'], audit_auxiliary=groups['audit_auxiliary'])
    ts, ds = split(target, 0), split(donor, 1000000)
    prepared, study = data.prepare_domain(target, donor, ts, ds, object())
    assert len(prepared.records) == 604
    assert prepared.record_ids[:600].tolist() == [r.record_id for r in ds.audit_auxiliary]
    assert prepared.record_ids[600:].tolist() == [r.record_id for r in ts.members + ts.nonmembers]
    assert study['cross_domain_audit']['gate'] == 'PASS'


def test_full_curve_built_before_selecting_fitting_point(tmp_path, fake_preflight, monkeypatch):
    # Building a one-point 800 curve would move the 1200 validation reservoir.
    import experiments.shared.models.loading as loading
    from experiments.resource_curves.storage import digest
    values = args(tmp_path, 'auxiliary')
    values.benchmarks, values.seeds = ['wikitection'], [1919]
    tasks = matrix.plan(values)
    frozen = shared('wikitection')
    ext = dict(seed=1919, shared_split_digest=digest(frozen), records=[dict(record_id=f'e{i}') for i in range(1000)])
    extension_path = Path(tasks[0]['extension_path'])
    extension_path.parent.mkdir(parents=True)
    extension_path.write_text(json.dumps(ext))
    monkeypatch.setattr(matrix, 'read_condition', lambda *a: dict(shared=frozen, files=[],
        passport=dict(config=dict(draft_model='toy', draft_revision='v1'))))
    monkeypatch.setattr(matrix, 'read_extension', lambda *a: ext)
    monkeypatch.setattr(loading, 'local_tokenizer', lambda *a: None)
    monkeypatch.setattr(matrix, 'materialize', lambda *a: None)
    monkeypatch.setattr(matrix, 'prepare_in_domain', lambda *a: None)
    points = [matrix.prepare_task(t)[1] for t in tasks if t['budget']['fitting'] > 400]
    for role in ('train', 'validation'):
        assert points[1]['partitions'][role][:len(points[0]['partitions'][role])] == points[0]['partitions'][role]
    assert points[0]['partitions']['calibration'] == points[1]['partitions']['calibration']


def test_cpu_worker_collection_fitting_resume_and_summary(tmp_path, fake_preflight, monkeypatch):
    import experiments.shared.models.loading as loading
    import experiments.shared.audit.provenance as provenance
    previous_threads = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        values = args(tmp_path, 'query')
        values.benchmarks, values.seeds = ['wikitection'], [1919]
        tasks = matrix.plan(values)[:2]
        frozen = shared('wikitection')
        ids = [r['record_id'] for role in ('audit_auxiliary', 'member', 'nonmember') for r in frozen['splits'][role]]
        rows = [record(r, [0, 1, 2, 0]) for r in ids]
        tokenizer = SimpleNamespace(encode=lambda text, **kw: [1, 2], bos_token_id=None, eos_token_id=2)
        prepared = data.prepared_records(rows, ['audit_auxiliary'] * 600 + ['member'] * 2 + ['nonmember'] * 2,
                                         tokenizer, 1919)
        point = dict(seed=1919, budget=AuxiliaryBudget().to_dict(), partitions=base_partitions(frozen))
        monkeypatch.setattr(matrix, 'prepare_task', lambda task: (prepared, point, dict(files=[])))
        monkeypatch.setattr(provenance, 'sources_for', lambda *a, **kw: dict(files=[], checkpoints=[]))
        adapters = []
        def load(*a):
            adapter = ToyAdapter()
            adapters.append(adapter)
            return adapter
        monkeypatch.setattr(loading, 'load_adapter', load)
        for task in tasks:
            matrix.execute_task(task, device='cpu')
        assert len(adapters) == 2  # Each B was actually collected independently.
        reports = [matrix.read_report(t, code_fingerprint()) for t in tasks]
        assert reports[1]['cost']['acceptance_judgments'] == 2 * reports[0]['cost']['acceptance_judgments']
        assert reports[0]['detector']['key'] != reports[1]['detector']['key']
        assert all(r['cost']['timing_origin'] == 'measured_same_budget_archive' for r in reports)
        for task in tasks:
            matrix.execute_task(task, device='cpu')
        assert len(adapters) == 2  # Full, checked recovery without model loading.
        summary = matrix.summarize(tasks, write_to=tmp_path / 'summary')
        assert summary['complete'] == 2
        assert len(summary['metric_rows']) == 2
        assert (tmp_path / 'summary/per_seed.csv').is_file()
        saved = Path(tasks[0]['output']) / 'scores.npz'
        saved.write_bytes(b'corrupted')
        assert matrix.summarize(tasks)['rows'][0]['state'] == 'invalid'
    finally:
        torch.set_num_threads(previous_threads)
