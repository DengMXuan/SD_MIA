"""CPU integration checks with real tiny Pythia/Qwen models; no network/GPU."""
import json
from pathlib import Path

import numpy as np
import pytest
import torch
from transformers import Qwen3Config, Qwen3ForCausalLM

from experiments.pretraining.data import load_evaluation
from experiments.pretraining.datasets import prepare_mimir
from experiments.pretraining.evaluation import evaluate_main
from experiments.shared.audit.artifacts import digest
from standalone.pretraining_baselines import evaluate as runner
from standalone.pretraining_baselines.contract import METHODS, MAIN, frozen_contract
from standalone.pretraining_baselines.reporting import checked_report
from standalone.pretraining_baselines.run import parse_args
from tests.pretraining.test_pretraining import pretrained_fixture

SIZES = dict(detector_train=3, detector_validation=1, calibration=2)


def small_manifest(pretrained_fixture, tmp_path, family='pythia', seed=1919):
    root, _, models = pretrained_fixture
    if family == 'qwen':
        tokenizer = load_evaluation(root / 'data/manifest.json').tokenizer
        models = dict(models)
        path = tmp_path / 'qwen-target'
        torch.manual_seed(12)
        config = Qwen3Config(vocab_size=104, hidden_size=16, intermediate_size=32,
            num_hidden_layers=1, num_attention_heads=2, num_key_value_heads=1, head_dim=8,
            max_position_embeddings=128, bos_token_id=1, eos_token_id=1, pad_token_id=0)
        Qwen3ForCausalLM(config).save_pretrained(path)
        tokenizer.save_pretrained(path)
        models['target'] = dict(repo_id=str(path), revision='local-test')
    return prepare_mimir(root / 'member.jsonl', root / 'nonmember.jsonl', tmp_path / 'data',
        source='fixture', split='test', seed=seed, models=models, n_per_class=4, n_aux=6)


@pytest.mark.parametrize('family,seed', [('pythia', 1919), ('qwen', 1949)])
def test_seven_methods_frozen_roles_and_resume(pretrained_fixture, tmp_path, monkeypatch, family, seed):
    manifest = small_manifest(pretrained_fixture, tmp_path, family, seed)
    output = tmp_path / 'baseline'
    loaded = []
    real_load = runner.load_model

    def track(spec, *args, **kwargs):
        model = real_load(spec, *args, **kwargs)
        loaded.append((spec, model, {k: v.clone() for k, v in model.state_dict().items()}))
        return model

    monkeypatch.setattr(runner, 'load_model', track)
    reports = runner.evaluate(manifest, output, seed=seed, device='cpu', **SIZES)
    assert tuple(reports) == METHODS
    assert len(loaded) == 1 and 'draft' not in loaded[0][0]['repo_id']
    _, model, weights = loaded[0]
    assert not model.training and all(not p.requires_grad for p in model.parameters())
    assert all(torch.equal(v, weights[k]) for k, v in model.state_dict().items())
    data, parts, *_ = frozen_contract(manifest, seed, **SIZES)
    assert json.loads((output / 'PARTITIONS.json').read_text()) == parts
    eval_data = load_evaluation(manifest)
    task = dict(manifest=str(manifest), seed=seed)
    for method, report in reports.items():
        assert report['settings']['audit_seed'] == seed
        assert report['metrics']['n_test_member'] == report['metrics']['n_test_nonmember'] == 4
        assert report['metrics']['n_calibration'] == 2
        used = report['reference_ids']
        assert set(used).issubset(parts['record_ids']['reference'])
        assert set(used).isdisjoint(parts['record_ids']['calibration'] + parts['record_ids']['test'])
        assert len(used) == (4 if method in ('petal', 'recall', 'icp_mia') else 0)
        assert report['cost']['draft_sequences'] == 0
        assert np.isfinite(report['metrics']['auc'])
        assert checked_report({**task, 'output': str(output)}, method, data, parts) == report
    # Check the real-token normalization/no-EOS contract against a direct loss.
    record = eval_data.members[0]
    tokens = torch.tensor([list(record.prompt_ids) + list(record.response_ids)])
    with torch.inference_mode():
        logp = model(input_ids=tokens).logits[0, :-1].float().log_softmax(-1)
        expected = logp.gather(-1, tokens[0, 1:, None]).mean().item()
    with np.load(output / 'loss/scores.npz') as scores:
        assert scores['scores'][scores['test'][0]] == pytest.approx(expected, abs=1e-6)
    monkeypatch.setattr(runner, 'load_model', lambda *a, **k: pytest.fail('reloaded completed model'))
    assert runner.evaluate(manifest, output, seed=seed, device='cpu', **SIZES) == reports
    with pytest.raises(ValueError, match='seed'):
        runner.evaluate(manifest, output, seed=seed + 1, device='cpu', **SIZES)
    with pytest.raises(ValueError, match='parameters/sources changed'):
        runner.evaluate(manifest, output, seed=seed, device='cpu',
                        detector_train=2, detector_validation=2, calibration=2)
    (output / 'loss/scores.npz').write_bytes(b'corrupt')
    with pytest.raises(ValueError, match='checksum'):
        runner.evaluate(manifest, output, seed=seed, device='cpu', **SIZES)


def test_partial_resume_and_exact_main_comparison(pretrained_fixture, tmp_path, monkeypatch):
    seed = 1978
    manifest = small_manifest(pretrained_fixture, tmp_path, seed=seed)
    output, main = tmp_path / 'baseline', tmp_path / 'main'
    main_report = evaluate_main(manifest, main, seed=seed, device='cpu', detector_epochs=1, **SIZES)
    real_score = runner._score_method

    def fail_once(task, sources, method, *args):
        if method == 'min_k_pp':
            raise RuntimeError('simulated interruption')
        return real_score(task, sources, method, *args)

    monkeypatch.setattr(runner, '_score_method', fail_once)
    with pytest.raises(RuntimeError, match='interruption'):
        runner.evaluate(manifest, output, seed=seed, device='cpu', main_dir=main, **SIZES)
    timestamps = [(output / m / 'scores.npz').stat().st_mtime_ns for m in METHODS[:2]]
    calls = []

    def track(task, sources, method, *args):
        calls.append(method)
        return real_score(task, sources, method, *args)

    monkeypatch.setattr(runner, '_score_method', track)
    runner.evaluate(manifest, output, seed=seed, device='cpu', main_dir=main, **SIZES)
    assert calls == list(METHODS[2:])
    assert timestamps == [(output / m / 'scores.npz').stat().st_mtime_ns for m in METHODS[:2]]
    assert json.loads((output / 'PARTITIONS.json').read_text()) == json.loads((main / 'PARTITIONS.json').read_text())
    data, partitions, *_ = frozen_contract(manifest, seed, **SIZES)
    task = dict(manifest=str(manifest), seed=seed, output=str(output), main_output=str(main))
    assert checked_report(task, MAIN, data, partitions) == main_report
    # A same-sized main split with changed IDs is rejected before inference.
    changed = json.loads((main / 'PARTITIONS.json').read_text())
    changed['record_ids']['test'].reverse()
    (main / 'PARTITIONS.json').write_text(json.dumps(changed))
    with pytest.raises(ValueError, match='partitions differ'):
        runner.evaluate(manifest, output, seed=seed, device='cpu', main_dir=main, **SIZES)


def test_protected_output(pretrained_fixture, tmp_path):
    manifest = small_manifest(pretrained_fixture, tmp_path)
    for path in (manifest.parent, tmp_path, tmp_path / 'main/child'):
        with pytest.raises(ValueError, match='overlap|separate'):
            runner.evaluate(manifest, path, seed=1919, device='cpu', main_dir=tmp_path / 'main', **SIZES)


@pytest.mark.parametrize('argv', [
    ['mimir', 'run', '--seeds', '1919', '1919'],
    ['mimir', 'run', '--variants', 'clean_only'],
    ['temporal', 'run', '--sources', 'github'],
    ['mimir', 'run', '--gpus', '0', '0'],
    ['mimir', 'run', '--output-root', '/home/mxd/lib/SD_MIA-pretraining-data'],
])
def test_invalid_cli_selections(argv):
    with pytest.raises(SystemExit):
        parse_args(argv)


def test_domain_and_seed_selection():
    args = parse_args(['mimir', 'dry-run', '--sources', 'github', 'wikipedia_(en)',
                      '--seeds', '1949', '--gpus', '0', '2'])
    assert args.sources == ['github', 'wikipedia_(en)'] and args.seeds == [1949]
    args = parse_args(['temporal', 'dry-run'])
    assert args.seeds == [1919, 1949, 1978] and args.variants == ['length_matched']
