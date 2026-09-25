"""The DP experiment uses the same registered model/draft routes as ordinary audits."""
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from experiments.dp_defense import audit, sweep
from experiments.shared.audit.config import audit_settings, condition_settings
from experiments.shared.models.registry import MODEL_PAIRS


@pytest.mark.parametrize('is_head', [False, True])
def test_new_audit_family_requires_explicit_dp_training_support(tmp_path, monkeypatch, is_head):
    from experiments.dp_defense import api

    reference = tmp_path / 'reference'
    reference.mkdir()
    (reference / 'results.json').write_text(json.dumps({'config': {}}))
    monkeypatch.setattr(api, 'identify_pair', lambda _: SimpleNamespace(adapter='new_family', is_head=is_head))
    output = tmp_path / 'private'
    with pytest.raises(ValueError, match='DP training is not implemented'):
        api.plan_private_training(reference, output, epsilon=4.)
    assert not output.exists()


def passport(pair):
    spec = MODEL_PAIRS[pair]
    return dict(config=dict(benchmark='wikitection', target_epochs=1, seed=1919,
                            target_model=spec.target, draft_model=spec.draft,
                            target_revision=spec.target_revision, draft_revision=spec.draft_revision),
                protocol_track=dict(pair=pair),
                privacy=dict(request_key='private-request',
                             stages={'target': {'privacy': {'epsilon': 4.}}},
                             pairs={'draft_auxiliary_distilled': dict(epsilon=4., delta=5e-6),
                                    'draft_member_sft': dict(epsilon=8., delta=1e-5)}))


@pytest.mark.parametrize('pair', list(MODEL_PAIRS))
def test_dp_execution_selects_each_registered_draft_and_attaches_its_budget(tmp_path, monkeypatch, pair):
    import torch
    from experiments.shared.audit import baselines
    from tests.sft.test_qwen_audit_matrix import write_example_result

    artifact = passport(pair)
    spec = MODEL_PAIRS[pair]
    run = tmp_path / 'model'
    run.mkdir()
    (run / '.dp.lock').touch()
    monkeypatch.setattr(audit, 'verify_run', lambda path: artifact)
    monkeypatch.setattr(torch.cuda, 'is_available', lambda: True)
    prepared, sources = [], []
    def prepare(path, adapter, role):
        prepared.append((adapter, role))
        return None, None
    def source(path, roles, *, adapter):
        sources.append((adapter, tuple(roles)))
        return {}
    def execute(task, *args):
        for method in task['methods']:
            write_example_result(task, method, .7)
    monkeypatch.setattr(audit, 'prepare_records', prepare)
    monkeypatch.setattr(audit, 'audit_sources', source)
    monkeypatch.setattr(audit, 'run_main', execute)
    monkeypatch.setattr(baselines, 'run_baselines', execute)
    result = audit.run_audit(run, tmp_path / 'audit', baselines=True)
    assert result['complete']
    assert prepared == [(spec.adapter, spec.roles[0]), (spec.adapter, spec.roles[1]), (spec.adapter, None)]
    assert sources == [(spec.adapter, ('target', role)) for role in spec.roles] + [(spec.adapter, ('target',))]
    main = [row for row in result['rows'] if row['method'] == 'main_fixed_sparse_positive']
    assert [row['privacy']['epsilon'] for row in main] == [4., 8.]
    assert all(row['privacy']['scope'] == 'target_only' for row in result['rows'] if row['draft_role'] == 'target_only')
    tasks = audit.make_tasks(run, tmp_path / 'audit', artifact)
    assert all(task['settings'] == condition_settings(audit_settings(), 1919) for task in tasks)


def test_dp_sweep_covers_models_without_output_collisions(tmp_path):
    args = SimpleNamespace(model_pairs=list(MODEL_PAIRS), reference_root=None,
                           benchmarks=['wikitection'], epochs=[1], seeds=[1919], epsilons=[1., 4.],
                           model_root=tmp_path / 'models', audit_root=tmp_path / 'audits',
                           gpu=0, include_baselines=True)
    tasks = sweep.commands(args)
    assert len(tasks) == 10
    assert len({t['train'][t['train'].index('--output-dir') + 1] for t in tasks}) == 10
    for task in tasks:
        reference = task['train'][task['train'].index('--reference-run') + 1]
        assert Path(reference).is_relative_to(MODEL_PAIRS[task['model_pair']].run_root)
    args.reference_root = tmp_path / 'ambiguous'
    with pytest.raises(ValueError, match='exactly one'):
        sweep.commands(args)
