"""DP head mechanisms and interrupted stage recovery (CPU only)."""
from types import SimpleNamespace
import json
import pytest
import torch
pytest.importorskip("opacus")
from experiments.dp_defense.accounting import make_plan, pair_budgets
from experiments.dp_defense.training import dp_sft_train
from experiments.dp_defense.artifacts import owned_run, read_stage, save_stage, stage_key, verify_run
from experiments.shared.data.data import SFTRecord
from experiments.shared.audit.artifacts import digest
from tests.dp_defense.test_dp_defense import tiny_model, TinyTokenizer, single_thread

def test_head_plan_has_384_updates_independent_of_target_epochs():
    a = make_plan(epsilon=4, steps=384, epochs=1)
    b = make_plan(epsilon=4, steps=384, epochs=3)
    assert a == b and a.steps == 384
    assert a.sample_rate == .008
    with pytest.raises(ValueError, match='explicit optimizer'):
        make_plan(epsilon=4, steps=0)


def test_head_private_callback_updates_only_trainable_parameters():
    class Head(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = torch.nn.Parameter(torch.tensor(0.))
            self.verifier = torch.nn.Parameter(torch.tensor(7.), requires_grad=False)
    model = Head()
    plan = make_plan(epsilon=4, population=1, expected_batch_size=1)
    record = SFTRecord('doc', 'test', (4, 5), 'hash', prompt_ids=(1,))
    seen = []
    def loss(head, batch):
        seen.append(batch['input_ids'].shape)
        return (head.weight - head.verifier).square()
    dp_sft_train(model, [record], TinyTokenizer(), 'cpu', plan, lr=.1, optimizer_name='adamw',
                 allow_frozen_parameters=True, document_loss=loss)
    assert seen == [torch.Size([1, 4])]
    assert model.weight.item() != 0. and model.verifier.item() == 7.
    assert model.verifier.grad is None
    with pytest.raises(ValueError, match='full-parameter'):
        dp_sft_train(model, [record], TinyTokenizer(), 'cpu', plan, document_loss=loss)


def test_head_stages_use_legacy_head_layout_and_bind_member_teacher(tmp_path):
    assert read_stage(tmp_path / 'not-started', 'target', 'key') is None
    plan = make_plan(epsilon=4, population=2, expected_batch_size=2)
    dp = {**plan.as_dict(), 'completed_steps': plan.steps}
    request = dict(head_pair='qwen3_8b_eagle3', sources=[],
                   plans={'target': plan.as_dict(), 'draft_member_sft': plan.as_dict()})
    with owned_run(tmp_path / 'run', request) as output:
        stages = {}
        for role in ('target', 'draft_auxiliary_distilled', 'draft_member_sft'):
            teacher = stages['target']['checkpoint_sha256'] if role != 'target' else None
            key = stage_key(request, role, teacher)
            stages[role] = save_stage(output, role, key, tiny_model(), TinyTokenizer(),
                dp if role != 'draft_auxiliary_distilled' else {'teacher_sha256': teacher},
                marker={'stage': role})
        assert (output / 'heads/member_head/_COMPLETE.json').exists()
        assert not (output / 'checkpoints/draft_member_sft').exists()
        write = {'privacy': dict(request_key=digest(request), stages=stages, pairs=pair_budgets(dp, dp))}
        (output / 'results.json').write_text(json.dumps(write))
        assert verify_run(output) == write
        with pytest.raises(ValueError, match='teacher mismatch'):
            read_stage(output, 'draft_member_sft', stage_key(request, 'draft_member_sft', 'another-target'))


def test_mtp_head_objectives_align_teacher_and_native_response_mask():
    from experiments.dp_defense.head_train import head_loss
    import torch.nn.functional as F
    class Target(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = torch.nn.Parameter(torch.ones(1), requires_grad=False)
            self.eval()
        def forward(self, input_ids, **kwargs):
            logits = torch.arange(20.).reshape(1, 5, 4).sin()
            return SimpleNamespace(logits=logits, hidden_states=(logits,))
    class Head(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.logits = torch.nn.Parameter(torch.arange(12.).reshape(1, 3, 4).cos())
            self.masks = []
        def forward(self, input_ids, hidden_states, loss_mask, **kwargs):
            assert not hidden_states.requires_grad
            self.masks.append(loss_mask.clone())
            valid = loss_mask[:, 2:]
            ce = F.cross_entropy(self.logits[valid], input_ids[:, 2:][valid])
            return [self.logits], ce, {}
    target, head = Target(), Head()
    ids = torch.tensor([[0, 1, 2, 3, 1]])
    labels = torch.tensor([[-100, -100, -100, 3, 1]])
    batch = dict(input_ids=ids, labels=labels, attention_mask=torch.ones_like(ids))
    native = head_loss('mtp', target, head, batch, torch.device('cpu'), member=True)
    assert torch.allclose(native, F.cross_entropy(head.logits[:, 1:].reshape(-1, 4), ids[:, 3:].flatten()))
    kd = head_loss('mtp', target, head, batch, torch.device('cpu'), member=False)
    teacher = target(ids).logits[:, 2:4].reshape(-1, 4)
    expected = F.kl_div(F.log_softmax(head.logits[:, 1:].reshape(-1, 4) / 2., -1),
                        F.softmax(teacher / 2., -1), reduction='batchmean') * 4
    assert torch.allclose(kd, expected)
    kd.backward()
    assert head.logits.grad is not None and target.weight.grad is None
    assert all(torch.equal(m, labels.ne(-100)) for m in head.masks)
    target.train()
    with pytest.raises(ValueError, match='frozen eval'):
        head_loss('mtp', target, head, batch, torch.device('cpu'), member=True)


@pytest.mark.parametrize('member', [False, True])
def test_eagle_member_and_aux_use_same_temperature_kl(member, monkeypatch):
    from experiments.dp_defense.head_train import head_loss
    from experiments.shared.drafts import eagle3
    calls = []
    target = torch.nn.Linear(1, 1).requires_grad_(False).eval()
    head = torch.nn.Linear(1, 1)
    def loss(model, teacher, batch, device, temperature):
        calls.append((model, teacher, temperature))
        return model.weight.sum(), None
    monkeypatch.setattr(eagle3, '_eagle_kd_loss', loss)
    head_loss('eagle3', target, head, {}, 'cpu', member=member).backward()
    assert calls == [(head, target, 2.)]
    assert target.weight.grad is None and head.weight.grad is not None


@pytest.mark.parametrize('pair', ['qwen3_8b_eagle3', 'qwen35_9b_mtp'])
def test_dp_head_lifecycle_routes_data_freezes_target_and_resumes(tmp_path, monkeypatch, pair):
    from experiments.dp_defense import head_train as runner
    from experiments.shared.data import splits
    from experiments.shared.drafts import common
    from experiments.shared.training.config import Config
    spec = common.PAIR_MODELS[pair]
    output = tmp_path / 'run'
    cfg = Config(trainer='full', optimizer='adamw', output_dir=output, n_per_class=2,
                 target_model=spec['target'], target_revision=spec['target_revision'])
    plan = make_plan(epsilon=4, population=2, expected_batch_size=2)
    plans = {'target': plan, 'draft_member_sft': plan}
    request = dict(head_pair=pair, sources=[], source_head=None, scope='models_only',
                   plans={k: v.as_dict() for k, v in plans.items()})
    records = [[SFTRecord(f'{role}-{i}', 'test', (4, 5), f'{role}-{i}', prompt_ids=(1,))
                for i in range(2)] for role in ('member', 'aux')]
    metadata = dict(shared_split_sha256='split', pool_sha256='pool', counts={'member': 2},
                    split_seed=1919, tokenizer_source='source')
    details = dict(pair=pair, kind=spec['kind'], manifest=tmp_path / 'split', pool=tmp_path / 'pool', data=metadata)
    monkeypatch.setattr(runner, 'prepare_request', lambda *a: (cfg, details, plans, request))
    monkeypatch.setattr(splits, 'build_controlled_split_from_shared_manifest', lambda *a:
                        SimpleNamespace(members=records[0], draft_auxiliary=records[1], metadata=metadata))
    monkeypatch.setattr(common, 'tokenizer_for', lambda *a: TinyTokenizer())
    monkeypatch.setattr(torch.cuda, 'is_available', lambda: True)
    monkeypatch.setattr(torch.cuda, 'set_device', lambda d: None)
    monkeypatch.setattr(runner, 'set_seed', lambda seed: None)
    (tmp_path / 'eagle3.py').write_text('# pinned remote implementation')
    monkeypatch.setattr(runner, 'cached_snapshot', lambda *a: tmp_path)
    loaded, heads, trained, distilled = [], [], [], []
    def load(model_id, *a, **kw):
        loaded.append(model_id)
        return tiny_model()
    def load_head(pair, source, target, device):
        heads.append((pair, source, target))
        model = tiny_model()
        model.config.num_speculative_steps = 1
        return model
    def train(model, records, tokenizer, device, plan, **kwargs):
        trained.append(([r.record_id for r in records], kwargs))
        if len(trained) == 2:
            raise RuntimeError('interrupted member head')
        return {**plan.as_dict(), 'completed_steps': plan.steps}
    def distill(model, target, records, *args, **kwargs):
        assert not target.training and not any(p.requires_grad for p in target.parameters())
        distilled.append([r.record_id for r in records])
    monkeypatch.setattr(runner, 'load_causal_lm', load)
    monkeypatch.setattr(runner, 'load_initial_head', load_head)
    monkeypatch.setattr(runner, 'dp_sft_train', train)
    monkeypatch.setattr(runner, 'fit_auxiliary', distill)
    with pytest.raises(RuntimeError, match='interrupted member'):
        runner.run(tmp_path / 'ref', output, 4., 1., 0)
    assert (output / 'heads/auxiliary_head/DP_STAGE.json').exists()
    assert not (output / 'results.json').exists()
    runner.run(tmp_path / 'ref', output, 4., 1., 0)
    artifact = verify_run(output)
    assert loaded == [spec['target']] + [str(output / 'checkpoints/target')] * 3
    assert len(heads) == 3 and all(h[2] == output / 'checkpoints/target' for h in heads)
    assert distilled == [[r.record_id for r in records[1]]]
    assert all(ids == [r.record_id for r in records[0]] for ids, _ in trained)
    assert trained[-1][1]['allow_frozen_parameters'] is True
    assert callable(trained[-1][1]['document_loss'])
    assert artifact['privacy']['pairs']['draft_member_sft']['epsilon_cap'] == 8.
    runner.run(tmp_path / 'ref', output, 4., 1., 0)
    assert len(heads) == 3
