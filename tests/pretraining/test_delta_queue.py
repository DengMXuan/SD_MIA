"""GPU diagnostics retain one joint statistical analysis across conditions."""
from types import SimpleNamespace

import numpy as np

from experiments.launchers import delta
from standalone.pythia_delta import verify


def test_collection_jobs_do_not_run_per_condition_analysis(tmp_path):
    args = SimpleNamespace(datasets=['arxiv', 'github'], seeds=[1919, 1949], output=tmp_path,
        dtype='float32', per_class=64, sample_seed=1, threads=1, bootstrap=100,
        data_root=tmp_path / 'data', audit_root=tmp_path / 'audit', model_root=tmp_path / 'models')
    jobs = delta.jobs_for(args)
    assert len(jobs) == 4 and len({job.id for job in jobs}) == 4
    assert {job.seed for job in jobs} == {1919, 1949}
    for job in jobs:
        assert job.command[job.command.index('--stage') + 1] == 'collect'
        assert job.command[job.command.index('--device') + 1] == 'cuda:0'
    assert len({job.command[-1] for job in jobs}) == 4


def test_joint_analysis_receives_validated_caches_for_all_conditions(tmp_path, monkeypatch):
    items = [dict(domain=source, seed=1919, key=source, token_ids=[1, 2, 3]) for source in ('arxiv', 'github')]
    for item in items:
        output = tmp_path / 'conditions' / item['domain'] / 'seed1919'
        for role in ('target', 'draft'):
            path = verify.cache_path(output, role, item)
            path.parent.mkdir(parents=True)
            np.savez_compressed(path, key=np.asarray(item['key']), token_ids=np.asarray(item['token_ids']),
                                logps=np.array([-2., -1.]))
    plan = dict(conditions=['arxiv/seed1919', 'github/seed1919'])
    calls = []
    monkeypatch.setattr(verify, 'analyze', lambda args, selected, records: calls.append((selected, records)))
    delta.merge_caches(SimpleNamespace(output=tmp_path), plan, items)
    assert calls == [(plan, items)]
    for item in items:
        np.testing.assert_array_equal(verify.load_cached(tmp_path, 'target', item), [-2., -1.])
    # A completed collection can be summarized again without duplicate output.
    delta.merge_caches(SimpleNamespace(output=tmp_path), plan, items)
    assert len(calls) == 2
