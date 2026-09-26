"""Frozen split reuse must preserve identities, ordering and condition seeds."""
import hashlib
import json

import pytest

from experiments.pretraining.data import load_evaluation, sha256
from experiments.pretraining.datasets import prepare_temporal
from experiments.pretraining.temporal_reuse import inspect_inputs, prepare_from_shared_split
from tests.pretraining.test_current_main import _temporal_sources, SIZES
from tests.pretraining.test_pretraining import pretrained_fixture


def _inputs(tmp_path, models, seed):
    shards, pool = _temporal_sources(tmp_path)
    history = prepare_temporal(shards, pool, tmp_path / 'history', seed=seed,
        models=models, n_per_class=4, n_aux=6, min_tokens=2, max_tokens=8,
        historical_provenance=dict(published_before='2016-09-26', reference='fixture'))
    documents = [json.loads(line) for line in pool.read_text().splitlines()]
    # Frozen roles intentionally have a different order from the pool.
    entries = [dict(record_id=row['record_id'],
                    text_sha256=hashlib.sha256(row['text'].encode()).hexdigest())
               for row in reversed(documents)]
    shared = tmp_path / 'shared.json'
    shared.write_text(json.dumps(dict(schema_version=3, benchmark='wikitection', seed=seed,
        pool_sha256=sha256(pool), token_band=dict(min_tokens=2, max_tokens=8),
        splits=dict(member=entries[:4], nonmember=entries[4:8],
                    auxiliary=entries[8:14], audit_auxiliary=entries[14:]))))
    return history, shared, pool


@pytest.mark.parametrize('seed', (1919, 1949, 1978))
def test_exact_frozen_roles_order_and_resume(pretrained_fixture, tmp_path, seed):
    _, _, models = pretrained_fixture
    history, shared, pool = _inputs(tmp_path, models, seed)
    originals = {p: p.read_bytes() for p in (history, history.parent / 'records.jsonl', shared, pool)}
    _, split, contract = inspect_inputs(history, shared, pool, seed=seed)
    output = tmp_path / 'reused'
    assert not output.exists()
    manifest = prepare_from_shared_split(history, shared, pool, output, seed=seed)
    rows = [json.loads(line) for line in (output / 'records.jsonl').read_text().splitlines()]
    old = [json.loads(line) for line in originals[history.parent / 'records.jsonl'].splitlines()]
    assert [r for r in rows if r['group'] == 'member'] == [r for r in old if r['group'] == 'member']
    for group, role in (('nonmember', 'nonmember'), ('auxiliary', 'audit_auxiliary')):
        assert [r['source_record_id'] for r in rows if r['group'] == group] == [
            r['record_id'] for r in split['splits'][role]]
    saved = json.loads(manifest.read_text())
    assert saved['source_provenance']['reuse_contract'] == contract
    assert saved['selection_seed'] == seed and saved['membership_verified'] is False
    data = load_evaluation(manifest, verify_draft=True)
    assert (len(data.members), len(data.nonmembers), len(data.auxiliary)) == (4, 4, 6)
    hashes = {p: sha256(p) for p in (manifest, output / 'records.jsonl')}
    assert prepare_from_shared_split(history, shared, pool, output, seed=seed) == manifest
    assert all(sha256(p) == digest for p, digest in hashes.items())
    assert all(p.read_bytes() == content for p, content in originals.items())
    split['splits']['nonmember'].reverse()
    shared.write_text(json.dumps(split))
    with pytest.raises(ValueError, match='sources changed'):
        prepare_from_shared_split(history, shared, pool, output, seed=seed)


@pytest.mark.parametrize('fault,match', (
    ('seed', 'seeds must match'), ('overlap', 'roles overlap'),
    ('missing_role', 'all four'), ('pool', 'pool differ'),
    ('history', 'records checksum'), ('text', 'document text changed'),
))
def test_invalid_sources_fail_before_publishing(pretrained_fixture, tmp_path, fault, match):
    _, _, models = pretrained_fixture
    history, shared, pool = _inputs(tmp_path, models, 1919)
    split = json.loads(shared.read_text())
    if fault == 'seed':
        split['seed'] = 1949
    elif fault == 'overlap':
        split['splits']['nonmember'][0] = split['splits']['member'][0]
    elif fault == 'missing_role':
        del split['splits']['member']
    elif fault == 'pool':
        split['pool_sha256'] = 'wrong'
    elif fault == 'history':
        with (history.parent / 'records.jsonl').open('a') as stream:
            stream.write('\n')
    elif fault == 'text':
        split['splits']['nonmember'][0]['text_sha256'] = 'wrong'
    shared.write_text(json.dumps(split))
    with pytest.raises(ValueError, match=match):
        prepare_from_shared_split(history, shared, pool, tmp_path / 'out', seed=1919)
    assert not (tmp_path / 'out').exists()


def test_reused_manifest_runs_current_method_and_rejects_changed_seed(pretrained_fixture, tmp_path):
    from experiments.pretraining.evaluation import evaluate_main
    _, _, models = pretrained_fixture
    history, shared, pool = _inputs(tmp_path, models, 1949)
    manifest = prepare_from_shared_split(history, shared, pool, tmp_path / 'out', seed=1949)
    report = evaluate_main(manifest, tmp_path / 'audit', seed=1949, device='cpu', detector_epochs=1, **SIZES)
    assert report['settings']['audit_seed'] == 1949
    assert report['evaluation_context']['membership_verified'] is False
    assert report['metrics']['n_test_member'] == report['metrics']['n_test_nonmember'] == 4
    saved = json.loads(manifest.read_text())
    saved['selection_seed'] = 1919
    manifest.write_text(json.dumps(saved))
    with pytest.raises(ValueError, match='contract changed'):
        prepare_from_shared_split(history, shared, pool, manifest.parent, seed=1949)
