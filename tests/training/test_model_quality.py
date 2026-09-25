"""Quality protocol math, frozen record selection, resumption and matrix scope."""
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from experiments.shared.evaluation import acceptance, quality, data
from experiments.shared.models.registry import MODEL_PAIRS
from experiments.shared.training import generalization as gen
from tests.training.test_generalization import StubTokenizer, _record


def test_epoch1_matrix_has_45_conditions_90_tasks_and_only_auxiliary_kd(tmp_path):
    from experiments.model_quality.cli import make_tasks
    args = SimpleNamespace(pairs=list(MODEL_PAIRS), benchmarks=['wikitection', 'newstection', 'arxivtection'],
                           seeds=[1919, 1949, 1978], evaluations=list(quality.KINDS),
                           bootstrap_repeats=1000, samples=500, batch_size=4, per_class=256,
                           output_root=tmp_path / 'outputs')
    tasks = make_tasks(args)
    assert len(tasks) == len({t['output'] for t in tasks}) == 90
    assert {t['condition']['epoch'] for t in tasks} == {1}
    for task in tasks:
        spec = MODEL_PAIRS[task['model_pair']]
        assert task['draft_role'] == (spec.roles[0] if task['evaluation'] == 'acceptance' else None)
        assert f"seed{task['condition']['condition_seed']}" in task['run_dir']
    assert not args.output_root.exists()


def test_short_samples_keep_exact_ids_lengths_and_seed_without_consuming_split():
    records = [_record(length, i) for i, length in enumerate((128, 300, 384, 512))]
    samples = gen.build_eval_samples(records, StubTokenizer(), 4, 256, 128, 1919)
    lengths = {s['record_id']: (s['context_tokens'], s['reference_tokens']) for s in samples}
    assert list(lengths) == [records[i].record_id for i in np.random.default_rng(1919).permutation(4)]
    assert lengths == {'sft:test:0': (85, 43), 'sft:test:1': (200, 100),
                       'sft:test:2': (256, 128), 'sft:test:3': (256, 128)}
    assert len(records) == 4
    with pytest.raises(ValueError, match='enough distinct'):
        gen.build_eval_samples(records, StubTokenizer(), 5, 256, 128, 1919)


def test_equal_class_sizes_do_not_make_member_nonmember_bootstrap_paired():
    x = np.linspace(0, 1, 40)
    scores = {role: {metric: x.copy() for metric in gen.GenerationQualityScorer.METRICS}
              for role in ('member', 'nonmember')}
    result = gen.summarize_model_scores(scores, scores, gen.GenerationQualityScorer.METRICS, 300, 1919, .03)
    for metric in gen.GenerationQualityScorer.METRICS:
        gap = result['member_minus_nonmember'][metric]
        assert gap['delta'] == 0 and gap['ci95_low'] < 0 < gap['ci95_high']
        for role in ('member', 'nonmember'):
            paired = result['base_minus_tuned'][f'{role}/{metric}']
            assert paired['ci95_low'] == paired['ci95_high'] == 0


def test_generation_buckets_by_reference_budget_and_restores_order():
    tokenizer = SimpleNamespace(padding_side='right', pad_token_id=0,
                                decode=lambda ids, **kwargs: str(len(ids)))
    budgets = []
    class Model:
        def generate(self, input_ids, attention_mask, max_new_tokens, **kwargs):
            budgets.append(max_new_tokens)
            assert torch.equal(input_ids.ne(0), attention_mask.bool())
            return torch.cat((input_ids, torch.ones((len(input_ids), max_new_tokens), dtype=torch.long)), 1)
    samples = [dict(prompt_ids=[1, 2], reference_tokens=n) for n in (43, 128, 100, 43)]
    results = gen.generate_continuations(Model(), samples, tokenizer, torch.device('cpu'), 128, 4)
    assert results == ['43', '128', '100', '43']
    assert sorted(budgets) == [43, 100, 128]
    assert tokenizer.padding_side == 'right'


@pytest.mark.parametrize('kind', ['plain', 'eagle3', 'mtp'])
def test_distribution_overlap_includes_zero_q_truth_and_masks_prompt(kind):
    class Adapter:
        def rows(self, tokens):
            p = torch.tensor([.2, .5, .3]).log().repeat(len(tokens), 1)
            q = torch.tensor([.5, 0., .5]).log().repeat(len(tokens), 1)
            if kind == 'mtp':
                q[0] = -torch.inf
            return p, q
    record = SimpleNamespace(prompt_ids=(2, 2), response_ids=(1, 0), append_eos=True)
    result = acceptance.record_acceptance(Adapter(), record, SimpleNamespace(eos_token_id=2))
    assert result['response_positions'] == 3
    assert result['exact_acceptance'] == pytest.approx(.5)
    assert result['top1_agreement'] == 0
    assert result['truth_vocab_coverage'] == pytest.approx(2 / 3)


def test_acceptance_summary_weights_documents_not_lengths():
    rows = [dict(role=role, exact_acceptance=value, top1_agreement=value,
                 truth_vocab_coverage=1., response_positions=length)
            for role in ('member', 'nonmember', 'auxiliary') for value, length in ((0., 1), (1., 100))]
    result = acceptance.summarize_acceptance(rows, 50, 1919)
    assert result['overall']['exact_acceptance']['mean'] == .5
    assert result['overall']['exact_acceptance']['count'] == 6


def test_acceptance_sampling_matches_generalization_seed_and_preserves_records():
    records = [_record(128, i) for i in range(20)]
    split = SimpleNamespace(members=records, nonmembers=records.copy(), draft_auxiliary=records.copy())
    selected = data.sample_classes(split, 5, 1949)
    generalization = gen.build_eval_samples(records, StubTokenizer(), 10, 256, 128, 1949)
    assert [r.record_id for r in selected['member']] == [s['record_id'] for s in generalization[:5]]
    assert len(split.members) == 20


def test_generation_resumes_without_loading_models_and_checks_cache(tmp_path, monkeypatch):
    cfg = SimpleNamespace(target_model='target', target_revision='rev', benchmark='wikitection')
    spec = MODEL_PAIRS['qwen3_8b_eagle3']
    task = quality.condition_task(spec, 'wikitection', 1919, 'generalization',
                                  output_root=tmp_path, samples=2, batch_size=2, bootstrap_repeats=20)
    split = SimpleNamespace(members=[_record(128, i) for i in range(2)],
                            nonmembers=[_record(300, i + 2) for i in range(2)])
    tokenizer = StubTokenizer()
    tokenizer.padding_side, tokenizer.pad_token_id = 'right', 0
    loads = []
    class Model(torch.nn.Module):
        def __init__(self, label):
            super().__init__()
            self.config = SimpleNamespace(use_cache=False)
            loads.append(label)
        def generate(self, input_ids, attention_mask, max_new_tokens, **kwargs):
            return torch.cat((input_ids, torch.ones((len(input_ids), max_new_tokens), dtype=torch.long)), 1)
    monkeypatch.setattr(quality, 'load_causal_lm', lambda *a, **kw: Model('base'))
    monkeypatch.setattr(gen, 'load_finetuned_model', lambda *a, **kw: Model('tuned'))
    cache, output = tmp_path / 'cache', tmp_path / 'output'
    cache.mkdir(); output.mkdir()
    first = quality._generalization(task, cfg, tokenizer, split, cache, output, torch.device('cpu'))
    assert loads == ['base', 'tuned']  # No draft or head model is loaded.
    again = quality._generalization(task, cfg, tokenizer, split, cache, output, torch.device('cpu'))
    assert loads == ['base', 'tuned'] and first[0] == again[0]
    shard = next(cache.glob('*.json'))
    saved = json.loads(shard.read_text()); saved['rows'][0]['bleu4'] = .99
    shard.write_text(json.dumps(saved))
    with pytest.raises(ValueError, match='cached evaluation rows changed'):
        quality._generalization(task, cfg, tokenizer, split, cache, output, torch.device('cpu'))


def test_inspection_is_read_only_and_rejects_changed_report_scores(tmp_path, monkeypatch):
    task = quality.condition_task(MODEL_PAIRS['qwen3'], 'wikitection', 1919, 'acceptance', output_root=tmp_path)
    monkeypatch.setattr(quality, 'validate_task', lambda _: None)
    assert quality.inspect_task(task) == {'status': 'ready'}
    assert not Path(task['output']).exists()
    output = Path(task['output']); output.mkdir(parents=True)
    scores = output / 'scores.npz'; scores.write_bytes(b'original')
    report = dict(schema='model_quality_report_v1', task=task, sources={'files': [], 'checkpoints': []},
                  outputs={'scores.npz': quality.sha256_file(scores)})
    (output / 'REPORT.json').write_text(json.dumps(report))
    assert quality.inspect_task(task) == {'status': 'complete'}
    scores.write_bytes(b'changed')
    assert quality.inspect_task(task)['status'] == 'blocked'


def test_worker_binds_seed_sources_and_resume_without_repeating_evaluation(tmp_path, monkeypatch):
    task = quality.condition_task(MODEL_PAIRS['qwen3'], 'wikitection', 1949, 'acceptance', output_root=tmp_path)
    monkeypatch.setattr(quality, 'validate_task', lambda _: None)
    monkeypatch.setattr(quality, 'sources_for', lambda _: dict(files=[], checkpoints=[]))
    monkeypatch.setattr(torch.cuda, 'is_available', lambda: True)
    monkeypatch.setattr(torch.cuda, 'set_device', lambda _: None)
    monkeypatch.setattr(torch.cuda, 'get_device_name', lambda _: 'test-only mocked GPU')
    seeds, calls = [], []
    monkeypatch.setattr(quality, 'set_seed', seeds.append)
    monkeypatch.setattr(quality, 'frozen_split', lambda _: (None, None, SimpleNamespace(metadata={'split_seed': 1949})))
    def evaluate(task, cfg, tokenizer, split, cache, output, device):
        calls.append(task['id'])
        assert json.loads((cache / 'REQUEST.json').read_text())['task'] == task
        (output / 'SAMPLES.json').write_text('{}')
        return {'test': True}, {'record_ids': np.array(['record'])}, {'seed': 1949}, 'test report'
    monkeypatch.setattr(quality, '_acceptance', evaluate)
    report = quality.evaluate_quality(task)
    assert report['protocol']['seed'] == report['split']['split_seed'] == 1949
    assert quality.evaluate_quality(task) == report
    assert calls == [task['id']] and seeds == [1949, 1949]
    (Path(task['output']) / 'REPORT.md').write_text('modified')
    with pytest.raises(ValueError, match='checksum mismatch'):
        quality.evaluate_quality(task)


def test_summary_cannot_write_to_training_assets(tmp_path):
    from experiments.model_quality.cli import summarize
    from experiments.paths import TRAINING
    task = quality.condition_task(MODEL_PAIRS['qwen3'], 'wikitection', 1919, 'acceptance', output_root=tmp_path)
    with pytest.raises(ValueError, match='separate'):
        summarize([task], TRAINING)


def test_acceptance_resume_revalidates_auxiliary_probe_and_records_diagnostics(tmp_path, monkeypatch):
    from experiments.shared.models import validation
    task = quality.condition_task(MODEL_PAIRS['qwen3'], 'wikitection', 1949, 'acceptance',
                                  output_root=tmp_path, per_class=1, bootstrap_repeats=2)
    records = [SimpleNamespace(record_id=f'id{i}', response_hash=f'h{i}',
                               prompt_ids=(0, 1), response_ids=(2, 3)) for i in range(6600)]
    split = SimpleNamespace(audit_auxiliary=records[:600], members=records[600:2600],
                            nonmembers=records[2600:4600], draft_auxiliary=records[4600:])
    validations, evaluated = [], []
    monkeypatch.setattr(quality, 'load_adapter', lambda *args: object())
    def gate(adapter, prompt, response, *, seed):
        validations.append(seed)
        return dict(status='passed', schema=validation.VALIDATION_SCHEMA)
    monkeypatch.setattr(validation, 'validate_adapter', gate)
    def accept(adapter, record, tokenizer):
        evaluated.append(record.record_id)
        return dict(exact_acceptance=.8, top1_agreement=.7, truth_vocab_coverage=1., response_positions=2)
    monkeypatch.setattr(quality, 'record_acceptance', accept)
    cache, output = tmp_path / 'cache', tmp_path / 'output'
    cache.mkdir(); output.mkdir()
    for _ in range(2):
        _, _, protocol, _ = quality._acceptance(task, None, None, split, cache, output, torch.device('cpu'))
        gate_report = json.loads((output / 'VALIDATION.json').read_text())
        assert protocol['adapter_validation'] == gate_report
        index = int(np.sort(np.random.default_rng(1949).permutation(600)[:320])[0])
        assert gate_report['record_id'] == records[index].record_id
        assert gate_report['record_role'] == 'audit_auxiliary_train'
    assert validations == [1949, 1949]
    assert len(evaluated) == 3  # Cached metrics reused, but validation never skipped.
