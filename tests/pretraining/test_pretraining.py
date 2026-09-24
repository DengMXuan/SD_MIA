import argparse
import json
from pathlib import Path

import numpy as np
import pytest
import torch
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import Whitespace
from transformers import GPTNeoXConfig, GPTNeoXForCausalLM, PreTrainedTokenizerFast

from experiments.pretraining.prepare import freeze
from experiments.pretraining.data import load_evaluation, load_model
from experiments.pretraining.extract import extract
from experiments.pretraining.cache import ROLE
from experiments.shared.data.data import make_sft_example
from experiments.sd_membership_sft.archive.m1_fit import load_m1_data, make_partitions, run_experiment


@pytest.fixture(scope='module')
def pretrained_fixture(tmp_path_factory):
    root = tmp_path_factory.mktemp('pythia')
    torch.set_num_threads(1)
    vocab = {'[PAD]': 0, '[EOS]': 1, '[UNK]': 2, 'document': 3, 'alpha': 4, 'beta': 5}
    vocab.update({f'item{i}': i + 6 for i in range(90)})
    backend = Tokenizer(WordLevel(vocab, unk_token='[UNK]'))
    backend.pre_tokenizer = Whitespace()
    tokenizer = PreTrainedTokenizerFast(tokenizer_object=backend, pad_token='[PAD]', eos_token='[EOS]', unk_token='[UNK]')
    models = {}
    for i, (role, padded) in enumerate((('target', 112), ('draft', 104))):
        path = root / role
        torch.manual_seed(i + 12)
        config = GPTNeoXConfig(vocab_size=padded, hidden_size=16, intermediate_size=32,
            num_hidden_layers=4, num_attention_heads=4, max_position_embeddings=128,
            rotary_pct=.5, hidden_dropout=0., attention_dropout=0.,
            bos_token_id=1, eos_token_id=1, pad_token_id=0)
        GPTNeoXForCausalLM(config).save_pretrained(path)
        tokenizer.save_pretrained(path)
        models[role] = dict(repo_id=str(path), revision='local-test')
    members, nonmembers = root / 'member.jsonl', root / 'nonmember.jsonl'
    members.write_text(''.join(json.dumps(f'document item{i} alpha beta') + '\n' for i in range(40)))
    nonmembers.write_text(''.join(json.dumps(f'document item{i} alpha beta') + '\n' for i in range(40, 90)))
    manifest = freeze(members, nonmembers, root / 'data', source='fixture', split='test',
                      n_per_class=40, n_aux=5, models=models)
    return root, manifest, models


def test_raw_text_contract_and_frozen_labels(pretrained_fixture):
    root, manifest, _ = pretrained_fixture
    data = load_evaluation(manifest, verify_draft=True)
    assert (len(data.members), len(data.nonmembers), len(data.auxiliary)) == (40, 40, 5)
    for record in data.members + data.nonmembers + data.auxiliary:
        example = make_sft_example(record, data.tokenizer)
        assert len(example['input_ids']) == 4
        assert example['labels'][0] == -100
        assert example['labels'][1:] == list(record.response_ids)
        assert data.tokenizer.eos_token_id not in example['input_ids']
        assert record.prompt_ids == (3,)
    assert set(r.record_id for r in data.members).isdisjoint(r.record_id for r in data.nonmembers)
    with pytest.raises(FileExistsError):
        freeze(root / 'member.jsonl', root / 'nonmember.jsonl', manifest.parent, source='fixture', split='test')


def test_valid_vocab_removes_different_padded_heads(pretrained_fixture):
    _, _, models = pretrained_fixture
    for spec in models.values():
        model = load_model(spec, torch.device('cpu'))
        with torch.inference_mode():
            output = model(input_ids=torch.tensor([[3, 6, 4]]))
        assert output.logits.shape[-1] == 96
        class AttractToPadding:
            def __call__(self, input_ids, scores):
                scores[:, 96:] = 1e6
                return scores
        generated = model.generate(input_ids=torch.tensor([[3, 6]]), max_new_tokens=3,
                                   do_sample=False, logits_processor=[AttractToPadding()])
        assert int(generated.max()) < 96


def test_baseline_all_methods_on_pretrained_target(pretrained_fixture, monkeypatch):
    from experiments.baseline import cli as run
    root, manifest, _ = pretrained_fixture
    output = root / 'baselines'
    monkeypatch.setattr('sys.argv', ['baseline', '--pretraining-manifest', str(manifest),
        '--output-dir', str(output), '--methods', 'all',
        '--recall-shots', '1', '--icp-top-k', '1', '--sead-samples', '50', '--samia-samples', '1',
        '--generation-batch-size', '8'])
    monkeypatch.setattr(torch.cuda, 'is_available', lambda: False)
    run.main()
    report = json.loads((output / 'baseline_metrics.json').read_text())
    assert report['protocol']['training_regime'] == 'pretraining'
    assert report['protocol']['token_contract']['append_eos'] is False
    assert set(report['metrics']) == set(run.METHODS)
    expected_queries = dict(loss=1, min_k_prob=1, min_k_pp=1, recall=2, icp_mia=2,
                            petal=85/80, sead=1, ws=2, rs=1, bt=2, samia=1)
    for method, count in expected_queries.items():
        cost = report['costs'][method]
        prefix = 'physical_incremental_' if method in ('rs', 'bt') else ''
        assert cost[prefix + 'target_sequences_per_record'] == pytest.approx(count)
        assert cost[prefix + 'amortized_ms_per_record'] > 0
        if prefix:
            assert 'amortized_ms_per_record' not in cost
            assert cost['reference_reused'] is True
    for method in ('loss', 'min_k_prob', 'min_k_pp', 'sead'):
        assert report['costs'][method]['tokens_per_record'] == 4
    assert report['costs']['petal']['tokens_per_record'] == 4.25
    assert (output / 'BASELINE_COSTS.md').exists()
    single = root / 'single_loss'
    monkeypatch.setattr('sys.argv', ['baseline', '--pretraining-manifest', str(manifest),
                                    '--output-dir', str(single), '--methods', 'loss'])
    run.main()
    independent = json.loads((single / 'baseline_metrics.json').read_text())
    assert independent['scores']['loss'] == report['scores']['loss']
    for field in ('target_sequences_per_record', 'tokens_per_record'):
        assert independent['costs']['loss'][field] == report['costs']['loss'][field]
    with np.load(output / 'baseline_scores.npz') as archive:
        assert archive['labels'].tolist() == [1] * 40 + [0] * 40
        for name in run.METHODS:
            assert np.isfinite(archive[name]).all()
    assert len(list(output.glob('executions/*/*.npz'))) == len(run.METHODS)
    from experiments.pretraining.cache import freeze_partitions
    from experiments.pretraining.evaluate_baselines import evaluate
    with np.load(output / 'baseline_scores.npz') as archive:
        freeze_partitions(archive['labels'], archive['record_ids'], output / 'partitions.json')
    matched = evaluate(manifest, output, output / 'partitions.json', output / 'matched.json')
    assert matched['methods']['loss']['test']['test_n_member'] == 8
    assert matched['methods']['loss']['test']['test_n_nonmember'] == 8


def test_pretrained_m1_extract_load_fit_and_tamper_detection(pretrained_fixture):
    root, manifest, _ = pretrained_fixture
    output = root / 'm1'
    extract(manifest, output, device=torch.device('cpu'), batch_size=8,
            selected_blocks=(0, 1, 2, 3))
    data = load_m1_data(output / 'features', output / 'probabilities', ROLE)
    assert data.q.shape == (240, 6)
    assert data.h.shape == (240, 40)
    assert data.lengths.tolist() == [3] * 80
    assert not data.eos_mask.any()
    partition_path = output / 'features/partition_manifest.json'
    partitions = make_partitions(data.labels, data.record_ids, frozen_manifest_path=partition_path)
    leaves = [partitions[name] for name in ('nuisance_location', 'nuisance_scale', 'detector_fit', 'validation', 'calibration', 'test')]
    assert sorted(np.concatenate(leaves).tolist()) == list(range(80))
    assert np.all(data.labels[partitions['nuisance_fit']] == 0)
    args = argparse.Namespace(feature_dir=output / 'features', probability_dir=output / 'probabilities',
        output_dir=output / 'fit', role=ROLE, device='cpu', partition_manifest=partition_path,
        conditional_family='linear', detector_families='logistic', activation_mode='real',
        seed=20260909, no_bootstrap=True, bootstrap_repeats=0)
    report = run_experiment(args)
    assert report['protocol']['role'] == ROLE
    assert (output / 'fit/m1_metrics.json').exists()
    feature_path = output / 'features/h.npy'
    h = np.load(feature_path)
    h[0, 0] += 1
    np.save(feature_path, h)
    with pytest.raises(RuntimeError, match='content hash mismatch'):
        load_m1_data(output / 'features', output / 'probabilities', ROLE)


def test_cross_label_duplicates_rejected(pretrained_fixture):
    root, _, models = pretrained_fixture
    with pytest.raises(ValueError, match='cross-label'):
        freeze(root / 'member.jsonl', root / 'member.jsonl', root / 'duplicate', source='fixture',
               split='test', n_per_class=1, n_aux=0, models=models)
