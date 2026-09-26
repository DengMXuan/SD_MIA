"""Exercise the public pretraining interface with real, tiny frozen models."""
import json
from pathlib import Path
import shutil

import numpy as np
import pytest
import torch
from transformers import Qwen3Config, Qwen3ForCausalLM

from experiments.pretraining import evaluation
from experiments.pretraining.data import load_evaluation, sha256
from experiments.pretraining.datasets import prepare_mimir, prepare_temporal, wikitext_articles
from tests.pretraining.test_pretraining import pretrained_fixture


SIZES = dict(detector_train=3, detector_validation=1, calibration=2)


def _mimir(root, models, output, seed=1919):
    return prepare_mimir(root / 'member.jsonl', root / 'nonmember.jsonl', output,
                         source='fixture', split='test', seed=seed, models=models,
                         n_per_class=4, n_aux=6)


@pytest.mark.parametrize('family,seed', [('pythia', 1919), ('pythia', 1949), ('pythia', 1978), ('qwen3', 1978)])
def test_current_method_frozen_models_seed_and_resume(pretrained_fixture, tmp_path, monkeypatch, family, seed):
    root, _, models = pretrained_fixture
    if family == 'qwen3':
        tokenizer = load_evaluation(root / 'data/manifest.json').tokenizer
        models = {}
        for i, role in enumerate(('target', 'draft')):
            path = tmp_path / role
            torch.manual_seed(i + 12)
            cfg = Qwen3Config(vocab_size=104, hidden_size=16, intermediate_size=32,
                              num_hidden_layers=1, num_attention_heads=2, num_key_value_heads=1,
                              head_dim=8, max_position_embeddings=128,
                              eos_token_id=1, bos_token_id=1, pad_token_id=0)
            Qwen3ForCausalLM(cfg).save_pretrained(path)
            tokenizer.save_pretrained(path)
            models[role] = dict(repo_id=str(path), revision='local-test')
    manifest = _mimir(root, models, tmp_path / 'data', seed)
    output = tmp_path / 'audit'
    loaded, weights = [], {}
    load = evaluation.load_model

    def track(*args, **kwargs):
        model = load(*args, **kwargs)
        weights[id(model)] = {k: v.clone() for k, v in model.state_dict().items()}
        loaded.append(model)
        return model

    monkeypatch.setattr(evaluation, 'load_model', track)
    report = evaluation.evaluate_main(manifest, output, seed=seed, device='cpu', detector_epochs=2, **SIZES)
    assert len(loaded) == 2
    for model in loaded:
        assert not model.training and all(not p.requires_grad for p in model.parameters())
        assert all(torch.equal(v, weights[id(model)][k]) for k, v in model.state_dict().items())
    assert report['method'] == 'main_fixed_sparse_positive'
    assert report['training_member_count'] == 0
    assert report['evaluation_context']['language_model_finetuning'] is False
    assert report['settings']['audit_seed'] == seed
    metrics = report['metrics']
    assert (metrics['n_test_member'], metrics['n_test_nonmember'], metrics['n_calibration']) == (4, 4, 2)
    for field in ('auc', 'pauc_10_raw', 'pauc_10_normalized', 'roc_tpr_at_1pct_fpr', 'roc_tpr_at_10pct_fpr'):
        assert np.isfinite(metrics[field])
    parts = json.loads((output / 'PARTITIONS.json').read_text())['record_ids']
    leaves = [parts[k] for k in ('train', 'validation', 'calibration', 'test')]
    assert len(set(sum(leaves, []))) == sum(map(len, leaves)) == 14
    data = load_evaluation(manifest)
    auxiliary_ids = {r.record_id for r in data.auxiliary}
    assert set(parts['train'] + parts['validation'] + parts['calibration']) == auxiliary_ids
    with np.load(output / 'observations.npz') as archive:
        assert archive['lengths'].tolist() == [3] * 14
        assert 'logp' not in archive.files
        assert archive['features'].shape == (42, 6)
    # A completed call must not even load an LM again.
    monkeypatch.setattr(evaluation, 'load_model', lambda *a, **k: pytest.fail('loaded cached LM'))
    assert evaluation.evaluate_main(manifest, output, seed=seed, device='cpu', detector_epochs=2, **SIZES) == report
    with pytest.raises(ValueError, match='seed must match'):
        evaluation.evaluate_main(manifest, output, seed=seed + 1, device='cpu', **SIZES)
    with pytest.raises(ValueError, match='request or frozen sources changed'):
        evaluation.evaluate_main(manifest, output, seed=seed, device='cpu', detector_epochs=3, **SIZES)
    # Cold repeat: collection and detector fitting must reproduce scores.
    monkeypatch.setattr(evaluation, 'load_model', load)
    repeated = tmp_path / 'repeat'
    evaluation.evaluate_main(manifest, repeated, seed=seed, device='cpu', detector_epochs=2, **SIZES)
    with np.load(output / report['method'] / 'scores.npz') as a, np.load(repeated / report['method'] / 'scores.npz') as b:
        np.testing.assert_array_equal(a['scores'], b['scores'])
    (output / 'observations.npz').write_bytes(b'corrupt')
    with pytest.raises(ValueError, match='observation archive checksum'):
        evaluation.evaluate_main(manifest, output, seed=seed, device='cpu', detector_epochs=2, **SIZES)


def test_inputs_and_existing_outputs_are_protected(pretrained_fixture, tmp_path):
    root, _, models = pretrained_fixture
    manifest = _mimir(root, models, tmp_path / 'data')
    for protected in (manifest.parent, Path(models['target']['repo_id']) / 'audit'):
        with pytest.raises(ValueError, match='separate from data, models'):
            evaluation.evaluate_main(manifest, protected, seed=1919, device='cpu', **SIZES)
    output = tmp_path / 'unrelated'
    output.mkdir()
    (output / 'preserve.txt').write_text('do not overwrite')
    with pytest.raises(ValueError, match='unrelated experiment'):
        evaluation.evaluate_main(manifest, output, seed=1919, device='cpu', **SIZES)
    assert (output / 'preserve.txt').read_text() == 'do not overwrite'
    with pytest.raises(ValueError, match='exhaust'):
        evaluation.evaluate_main(manifest, tmp_path / 'wrong-count', seed=1919, device='cpu')
    with pytest.raises(ValueError, match='not enough disjoint'):
        prepare_mimir(root / 'member.jsonl', root / 'nonmember.jsonl', tmp_path / 'too-large',
                      source='fixture', split='test', seed=1919, models=models, n_per_class=40, n_aux=600)
    records = manifest.parent / 'records.jsonl'
    records.write_text(records.read_text() + '\n')
    with pytest.raises(ValueError, match='records hash mismatch'):
        evaluation.evaluate_main(manifest, tmp_path / 'tampered', seed=1919, device='cpu', **SIZES)


def _temporal_sources(tmp_path):
    import pyarrow as pa
    import pyarrow.parquet as pq
    lines = ['ignored continuation from another article\n']
    for i in range(10):
        lines += [f' = Title {i} = \n', f'document item{i} ', ' = = Section = = \n', 'alpha beta\n']
    shards = [tmp_path / f'train-{i}.parquet' for i in (0, 1)]
    # Split mid-article: the second shard does not start with a title.
    for path, rows in zip(shards, (lines[:3], lines[3:])):
        pq.write_table(pa.table({'text': rows}), path)
    pool = tmp_path / 'recent.jsonl'
    rows = [dict(record_id=f'wiki:{i}', source='en.wikipedia.org', title=f'Recent {i}',
                 creation_timestamp='2026-05-01T00:00:00Z', text=f'document item{i} alpha beta')
            for i in range(40, 60)]
    pool.write_text(''.join(json.dumps(r) + '\n' for r in rows))
    pool.with_suffix('.manifest.json').write_text(json.dumps(dict(jsonl_sha256=sha256(pool))))
    return shards, pool


def test_temporal_articles_dates_and_current_method(pretrained_fixture, tmp_path):
    _, _, models = pretrained_fixture
    shards, pool = _temporal_sources(tmp_path)
    articles = list(wikitext_articles(shards))
    assert len(articles) == 10
    assert articles[0]['text'] == 'document item0\nalpha beta'
    assert all('Section' not in r['text'] and 'ignored' not in r['text'] for r in articles)
    args = dict(seed=1919, n_per_class=4, n_aux=6, min_tokens=2, max_tokens=8, models=models)
    with pytest.raises(ValueError, match='documented publication'):
        prepare_temporal(shards, pool, tmp_path / 'missing-date', **args)
    args['historical_provenance'] = dict(published_before='2016-09-26', reference='fixture')
    manifest = prepare_temporal(shards, pool, tmp_path / 'temporal', **args)
    again = prepare_temporal(shards, pool, tmp_path / 'temporal-repeat', **args)
    assert (manifest.parent / 'records.jsonl').read_bytes() == (again.parent / 'records.jsonl').read_bytes()
    frozen = json.loads(manifest.read_text())
    assert frozen['membership_verified'] is False
    assert frozen['filtering']['member']['candidates'] == 10
    data = load_evaluation(manifest, verify_draft=True)
    assert len(data.members) == len(data.nonmembers) == 4 and len(data.auxiliary) == 6
    report = evaluation.evaluate_main(manifest, tmp_path / 'audit', seed=1919, device='cpu', detector_epochs=1, **SIZES)
    assert report['evaluation_context']['membership_verified'] is False
    assert report['evaluation_context']['model_release'] == '2025-04-29'
    with pytest.raises(FileExistsError):
        prepare_temporal(shards, pool, manifest.parent, **args)
    pool.write_text(pool.read_text().replace('2026-05-01', '2025-04-29'))
    pool.with_suffix('.manifest.json').write_text(json.dumps(dict(jsonl_sha256=sha256(pool))))
    with pytest.raises(ValueError, match='postdate'):
        prepare_temporal(shards, pool, tmp_path / 'old-negative', **args)


def test_cached_shard_symlinks_keep_logical_order(pretrained_fixture, tmp_path):
    _, _, models = pretrained_fixture
    shards, pool = _temporal_sources(tmp_path)
    # Actual Hub caches use content-hash blob names; those may sort backwards.
    for shard, blob in zip(shards, (tmp_path / 'z-blob', tmp_path / 'a-blob')):
        shard.rename(blob)
        shard.symlink_to(blob)
    manifest = prepare_temporal(shards[::-1], pool, tmp_path / 'data', seed=1919,
        n_per_class=10, n_aux=2, min_tokens=2, max_tokens=8, models=models,
        historical_provenance=dict(published_before='2016-09-26', reference='fixture'))
    rows = [json.loads(line) for line in (manifest.parent / 'records.jsonl').read_text().splitlines()]
    first = next(r for r in rows if r.get('temporal_metadata', {}).get('title') == 'Title 0')
    assert first['text'] == 'document item0\nalpha beta'


def test_historical_wiki_requires_old_revision_and_disjoint_content(pretrained_fixture, tmp_path):
    _, _, models = pretrained_fixture
    _, recent = _temporal_sources(tmp_path)
    historical = tmp_path / 'historical.jsonl'
    rows = [dict(record_id=f'old:{i}', source='en.wikipedia.org',
                 creation_timestamp='2023-01-01T00:00:00Z', snapshot_timestamp='2023-12-20T00:00:00Z',
                 snapshot_revision=99+i, text=f'document item{i} alpha beta') for i in range(10)]
    # Exact model-visible copy of a recent page: it must not cross roles.
    rows[0]['text'] = 'document item40 alpha beta'

    def save():
        historical.write_text(''.join(json.dumps(r)+'\n' for r in rows))
        historical.with_suffix('.manifest.json').write_text(json.dumps(dict(
            jsonl_sha256=sha256(historical), label_semantics='presumed_member_temporal_proxy')))

    save()
    args = dict(seed=1919, historical_format='historical_wiki', n_per_class=10, n_aux=6,
                min_tokens=2, max_tokens=8, models=models)
    manifest = prepare_temporal([historical], recent, tmp_path / 'data', **args)
    data = load_evaluation(manifest)
    tokens = [r.prompt_ids + r.response_ids for r in data.members + data.nonmembers + data.auxiliary]
    assert len(tokens) == len(set(tokens)) == 26
    rows[0]['snapshot_timestamp'] = '2026-01-01T00:00:00Z'
    save()
    with pytest.raises(ValueError, match='predate'):
        prepare_temporal([historical], recent, tmp_path / 'late-revision', **args)


def test_resume_rejects_changed_pretrained_weights(pretrained_fixture, tmp_path):
    root, _, original = pretrained_fixture
    models = {}
    for role, spec in original.items():
        dest = tmp_path / role
        shutil.copytree(spec['repo_id'], dest)
        models[role] = dict(repo_id=str(dest), revision='local-test')
    manifest = _mimir(root, models, tmp_path / 'data')
    evaluation.evaluate_main(manifest, tmp_path / 'audit', seed=1919, device='cpu', detector_epochs=1, **SIZES)
    weight = next((tmp_path / 'draft').glob('*.safetensors'))
    with weight.open('ab') as stream:
        stream.write(b'changed')
    with pytest.raises(ValueError, match='request or frozen sources changed'):
        evaluation.evaluate_main(manifest, tmp_path / 'audit', seed=1919, device='cpu', detector_epochs=1, **SIZES)
