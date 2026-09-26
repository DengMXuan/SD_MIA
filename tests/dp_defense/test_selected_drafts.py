"""KD-only DP conditions keep stage budgets, seed identity and resume aligned."""
import json
from types import SimpleNamespace

import pytest
import torch
pytest.importorskip('opacus')

from experiments.dp_defense import api, audit, sweep
from experiments.dp_defense.accounting import make_plan
from experiments.dp_defense.artifacts import verify_run
from experiments.dp_defense.conditions import seed_policy, variants
from experiments.shared.core.deployment_archive import sha256_file
from experiments.shared.data.data import SFTRecord, records_metadata
from experiments.shared.models.registry import MODEL_PAIRS
from experiments.shared.training.config import Config
from tests.dp_defense.test_dp_defense import TinyTokenizer, tiny_model, single_thread


@pytest.mark.parametrize('seed', [1919, 1949, 1978])
def test_plan_kd_only_validates_actual_split_seed_without_creating_outputs(tmp_path, seed):
    spec = MODEL_PAIRS['qwen3']
    reference = tmp_path / 'reference'
    reference.mkdir()
    manifest = reference / 'split.json'
    manifest.write_text(json.dumps(dict(benchmark='wikitection', seed=seed)))
    manifest.with_suffix('.audit.json').write_text('{}')
    cfg = Config(trainer='full', target_model=spec.target, draft_model=spec.draft,
                 target_revision=spec.target_revision, draft_revision=spec.draft_revision,
                 benchmark='wikitection', target_epochs=1, seed=seed, data_seed=seed)
    artifact = dict(config=cfg.as_dict(), material_passport=dict(status='COMPLETED'),
                    data=dict(shared_split_manifest=str(manifest), shared_split_sha256=sha256_file(manifest),
                              pool_path=str(reference / 'pool.jsonl')))
    passport = reference / 'results.json'
    passport.write_text(json.dumps(artifact))
    output = tmp_path / 'dp'
    request = api.plan_private_training(reference, output, epsilon=4, draft_variants=['kd'])
    assert not output.exists()
    assert request['draft_variants'] == ['kd']
    assert set(request['plans']) == {'target'} and request['plans']['target']['steps'] == 125
    assert request['config']['run_auxiliary_draft'] and not request['config']['run_member_draft']
    assert request['config']['pool_path'] == str(reference / 'pool.jsonl')
    assert request['execution'] == dict(accumulator_device='cpu', accumulator_dtype='float32')
    gpu_request = api.plan_private_training(reference, output, epsilon=4, draft_variants=['kd'],
                                            accumulator_device='cuda')
    assert not output.exists()
    assert gpu_request['execution']['accumulator_device'] == 'cuda'
    assert gpu_request['plans'] == request['plans']
    assert gpu_request['seed_policy'] == request['seed_policy']
    from experiments.dp_defense.artifacts import stage_key
    assert stage_key(gpu_request, 'target') != stage_key(request, 'target')
    assert all(v == seed for k, v in request['seed_policy'].items() if k != 'private_randomness')
    assert 'unpublished' in request['seed_policy']['private_randomness']
    manifest.write_text(json.dumps(dict(benchmark='wikitection', seed=seed+1)))
    artifact['data']['shared_split_sha256'] = sha256_file(manifest)
    passport.write_text(json.dumps(artifact))
    with pytest.raises(ValueError, match='condition/data/shared-split seeds'):
        api.plan_private_training(reference, output, epsilon=4, draft_variants=['kd'])


def test_epoch1_kd_sweep_routes_every_seed_and_counts_only_selected_artifacts(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr('sys.argv', ['sweep', 'dry-run', '--epochs', '1', '--draft-variants', 'kd'])
    sweep.main()
    plan = json.loads(capsys.readouterr().out)
    assert (plan['conditions'], plan['artifacts'], plan['draft_audits']) == (27, 54, 27)
    for task in plan['tasks']:
        assert '/epoch1/' in task['condition'] and 'epoch3' not in task['condition']
        assert task['stages'] == ['target', 'draft_auxiliary_distilled']
        assert task['train'][task['train'].index('--accumulator-device') + 1] == 'cpu'
        for command in ('train', 'audit'):
            assert task[command][-2:] == ['--draft-variants', 'kd']
        seed = int(task['condition'].rsplit('seed', 1)[1])
        assert seed in (1919, 1949, 1978)
        for flag, command in (('--reference-run', 'train'), ('--output-dir', 'train'), ('--run-dir', 'audit')):
            assert task[command][task[command].index(flag)+1].endswith(task['condition'])
    for selection in ([], ['kd', 'kd'], ['unknown']):
        with pytest.raises(ValueError, match='distinct'):
            variants(selection)


@pytest.mark.parametrize('pair,seed', [('qwen3', 1919), ('qwen3_8b_eagle3', 1949), ('qwen35_9b_mtp', 1978)])
def test_kd_only_training_real_cpu_dp_target_and_stage_seed_resume(tmp_path, monkeypatch, pair, seed):
    from experiments.dp_defense import train, head_train, training as private_training
    from experiments.shared.drafts import plain, common
    from experiments.shared.data import splits
    from experiments.shared.training import training
    from experiments.shared.audit import evaluation
    from experiments.dp_defense.artifacts import evaluation_verification

    spec = MODEL_PAIRS[pair]
    runner = head_train if spec.is_head else train
    output = tmp_path / 'run'
    cfg = Config(trainer='full', optimizer='adamw', target_model=spec.target, draft_model=spec.draft,
        target_revision=spec.target_revision, draft_revision=spec.draft_revision,
        output_dir=output, seed=seed, data_seed=seed, n_per_class=2, n_aux=2, n_audit_aux=2,
        target_batch_size=1, target_grad_accum=2, run_member_draft=False)
    manifest = tmp_path / 'split.json'
    manifest.write_text(json.dumps(dict(seed=seed, benchmark=cfg.benchmark)))
    metadata = dict(shared_split_manifest=str(manifest), shared_split_sha256=sha256_file(manifest),
                    pool_sha256='pool', counts={'member': 2}, split_seed=seed, tokenizer_source='source')
    records = [[SFTRecord(f'{role}-{i}', 'test', (4, 5), f'{role}-{i}', prompt_ids=(1,))
                for i in range(2)] for role in ('member', 'nonmember', 'aux', 'audit')]
    plan = make_plan(epsilon=4, population=2, expected_batch_size=2)
    request = dict(config=cfg.as_dict(), draft_variants=['kd'], sources=[], source_head=None,
                   plans={'target': plan.as_dict()}, scope='models_only', seed_policy=seed_policy(cfg.as_dict(), manifest))
    if spec.is_head:
        request['head_pair'] = pair
        details = dict(pair=pair, kind=spec.adapter, manifest=manifest, pool=tmp_path/'pool', data=metadata)
        prepared = (cfg, details, {'target': plan}, request)
    else:
        reference = dict(material_passport=dict(status='COMPLETED'), records=dict(zip(
            ('members', 'nonmembers', 'auxiliary', 'audit_auxiliary'), [records_metadata(r) for r in records])))
        prepared = (cfg, reference, manifest, {'target': plan}, request)
    monkeypatch.setattr(runner, 'prepare_request', lambda *a: prepared)
    monkeypatch.setattr(plain, '_load_condition_split', lambda *a: (*records, metadata))
    monkeypatch.setattr(splits, 'build_controlled_split_from_shared_manifest', lambda *a:
        SimpleNamespace(members=records[0], draft_auxiliary=records[2], metadata=metadata))
    tokenizer = TinyTokenizer()
    tokenizer.get_vocab = lambda: {str(i): i for i in range(32)}
    monkeypatch.setattr(training, 'load_tokenizer', lambda *a, **kw: tokenizer)
    monkeypatch.setattr(common, 'tokenizer_for', lambda *a: tokenizer)
    monkeypatch.setattr(torch.cuda, 'is_available', lambda: True)
    monkeypatch.setattr(torch.cuda, 'set_device', lambda *a: None)
    (tmp_path / 'eagle3.py').write_text('# tiny fixture')
    monkeypatch.setattr(head_train, 'cached_snapshot', lambda *a: tmp_path)
    loaded, private_calls, kd_rng, kd_ids, kd_seeds = [], [], [], [], []

    def load(model_id, *a, **kw):
        loaded.append(model_id)
        return tiny_model()

    def load_head(*a):
        model = tiny_model()
        model.config.num_speculative_steps = 1
        return model

    real_train = private_training.dp_sft_train
    def cpu_train(model, members, tok, device, plan, **kw):
        private_calls.append([r.record_id for r in members])
        with monkeypatch.context() as cpu:
            cpu.setattr(torch.cuda, 'is_available', lambda: False)
            cpu.setattr(torch.accelerator, 'is_available', lambda: False)
            return real_train(model, members, tok, 'cpu', plan, lr=.01, optimizer_name='adamw')

    def distill(model, teacher, auxiliary, *a, **kw):
        kd_rng.append(torch.rand(4))
        kd_ids.append([r.record_id for r in auxiliary])
        kd_seeds.append(kw['seed'] if spec.is_head else a[-1])
        if len(kd_rng) == 1:
            raise RuntimeError('interrupt KD')

    monkeypatch.setattr(training, 'load_causal_lm', load)
    monkeypatch.setattr(head_train, 'load_causal_lm', load)
    monkeypatch.setattr(head_train, 'load_initial_head', load_head)
    monkeypatch.setattr(private_training, 'dp_sft_train', cpu_train)
    monkeypatch.setattr(head_train, 'dp_sft_train', cpu_train)
    monkeypatch.setattr(training, 'distill_on_auxiliary', distill)
    monkeypatch.setattr(head_train, 'fit_auxiliary', distill)
    with pytest.raises(RuntimeError, match='interrupt KD'):
        runner.run(tmp_path / 'ref', output, 4, 1., 0, ['kd'])
    runner.run(tmp_path / 'ref', output, 4, 1., 0, ['kd'])
    artifact = verify_run(output)
    assert len(private_calls) == 1  # Reuses the completed target; no member training.
    assert set(artifact['privacy']['stages']) == {'target', 'draft_auxiliary_distilled'}
    assert not (output / 'checkpoints/draft_member_sft').exists() and not (output / 'heads/member_head').exists()
    assert kd_seeds == [seed, seed] and kd_ids == [[r.record_id for r in records[2]]] * 2
    torch.testing.assert_close(kd_rng[0], kd_rng[1], rtol=0, atol=0)
    budget = artifact['privacy']['pairs']
    assert set(budget) == {'draft_auxiliary_distilled'}
    assert budget['draft_auxiliary_distilled']['epsilon_cap'] == 4 and budget['draft_auxiliary_distilled']['delta'] == 5e-6
    tasks = audit.make_tasks(output, tmp_path / 'audit', artifact)
    assert len(tasks) == 1 and tasks[0]['draft_role'] == spec.roles[0]
    assert tasks[0]['settings']['audit_seed'] == seed
    with pytest.raises(ValueError, match='not trained'):
        audit.make_tasks(output, tmp_path / 'audit', artifact, draft_variants=['member'])
    with pytest.raises(ValueError, match='must equal'):
        audit.make_tasks(output, tmp_path / 'audit', artifact, seed=seed+1)
    checked = []
    monkeypatch.setattr('experiments.shared.models.readiness.ready', lambda t: (checked.append(t['draft_role']) or True, 'ready'))
    info = evaluation.inspect_run(output, verification=evaluation_verification())
    assert checked == info['draft_roles'] == [spec.roles[0]]
    count = len(loaded)
    runner.run(tmp_path / 'ref', output, 4, 1., 0, ['kd'])
    assert len(loaded) == count
    artifact['config']['seed'] += 1
    (output / 'results.json').write_text(json.dumps(artifact))
    with pytest.raises(ValueError, match='config differs'):
        verify_run(output)
