"""Information preservation, held-out isolation and real frozen-TCN replay."""
import json
from pathlib import Path

import numpy as np
import pytest
import torch

from experiments.pretraining import preserved_evidence as runner
from experiments.pretraining.frozen_audit import MODELS, identify_inputs, load_inputs, predict_frozen
from experiments.shared.audit.main import sparse_scores
from experiments.shared.audit.metrics import metrics
from experiments.shared.core.audit_runtime import _write_json
from experiments.shared.core.deployment_archive import sha256_file
from experiments.shared.methods.conditional_accept_only import ConditionalCountTCN
from experiments.shared.methods.preserved_accept_only import (
    OLD, PRIMARY, BASE_METHODS, METHODS, base_scores, fit_normalization, fuse_scores,
)
from experiments.shared.protocols.protocol_archive import save_archive


@pytest.fixture(autouse=True)
def single_thread():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


def test_shared_sparse_control_and_draft_information_are_preserved():
    rng = np.random.default_rng(42)
    lengths = np.array([3, 5, 7])
    counts = rng.integers(0, 3, lengths.sum())
    logq = -rng.uniform(.1, 4, lengths.sum())
    logpmf = np.log(rng.dirichlet(np.ones(3), lengths.sum()))
    base = base_scores(logpmf, counts, lengths, logq)
    data = dict(record_ids=np.arange(3), document_indices=np.arange(3), lengths=lengths, counts=counts)
    np.testing.assert_allclose(base[OLD], sparse_scores(data, logpmf, False), atol=1e-12)
    # Identical feedback can still differ through the draft-side information.
    reference = {name: np.array([-1., 0., 1.]) for name in BASE_METHODS}
    norm = fit_normalization(reference)
    identical = {name: np.zeros(2) for name in BASE_METHODS}
    identical['q_only'] = np.array([-1., 1.])
    scores = fuse_scores(identical, norm)
    assert scores[OLD][0] == scores[OLD][1]
    assert scores[PRIMARY][1] > scores[PRIMARY][0]


def test_fusion_does_not_normalize_using_other_test_or_calibration_records():
    reference = {name: np.array([0., 1., 2.]) for name in BASE_METHODS}
    norm = fit_normalization(reference)
    batch = {name: np.array([3., 4.]) for name in BASE_METHODS}
    before = fuse_scores(batch, norm)
    changed = {name: np.array([3., 1e8, -1e8]) for name in BASE_METHODS}
    after = fuse_scores(changed, norm)
    for name in METHODS:
        assert before[name][0] == after[name][0]
    assert norm == fit_normalization(reference)
    constant = fit_normalization({name: np.ones(3) for name in BASE_METHODS})
    assert np.array_equal(fuse_scores(changed, constant)[PRIMARY], np.zeros(3))


@pytest.fixture
def archive(tmp_path):
    root = tmp_path / 'legacy'
    folder = root / 'tasks/github/seed1919'
    score_dir = folder / OLD
    score_dir.mkdir(parents=True)
    torch.manual_seed(123)
    rng = np.random.default_rng(123)
    lengths = np.arange(3, 11)
    ids = np.array([f'doc{i}' for i in range(len(lengths))])
    labels = np.array([0, 0, 0, 0, 0, 1, 0, 1])
    features = np.zeros((lengths.sum(), 6), np.float32)
    features[:, 0] = -rng.uniform(.1, 3, len(features))
    features[:, 1:3] = rng.uniform(0, 1, (len(features), 2))
    features[:, 3] = .2
    features[:, 4] = np.concatenate([np.linspace(0, 1, n) for n in lengths])
    data = dict(features=features, counts=rng.integers(0, 3, len(features), dtype=np.uint8),
                lengths=lengths, record_ids=ids, labels=labels, document_indices=np.arange(len(ids)),
                start_indices=np.zeros(len(ids), int),
                record_roles=np.array(['audit_auxiliary']*5 + ['member', 'nonmember', 'member']))
    save_archive(folder / 'observations.npz', data, dict(protocol='fixed', starts=['fixed'], seed=1919),
                 [{} for _ in lengths])
    model = ConditionalCountTCN(5, 2)
    torch.save(dict(state_dict=model.state_dict(), mean=torch.zeros(5), scale=torch.ones(5)), folder / 'detector.pt')
    pmf = predict_frozen(folder / 'detector.pt', data)
    original = sparse_scores(data, pmf, False)[3:]
    saved = dict(record_ids=ids[3:], labels=labels[3:], scores=original,
                 calibration=np.array([0, 1]), test=np.array([2, 3, 4]))
    np.savez_compressed(score_dir / 'scores.npz', **saved)
    records = folder / 'records.jsonl'
    records.write_text(''.join(json.dumps(dict(record_id=key, label=int(y), token_ids=[1]*(int(n)+1)))+'\n'
                               for key, y, n in zip(ids, labels, lengths)))
    manifest = folder / 'manifest.json'
    _write_json(manifest, dict(models=MODELS, selection_seed=1919, benchmark='mimir/github/ngram_13_0.8',
                              kind='mimir_pretraining_v1', records_file=records.name, records_sha256=sha256_file(records)))
    _write_json(folder / 'PARTITIONS.json', dict(seed=1919, manifest_sha256=sha256_file(manifest), record_ids=dict(
        train=ids[:2].tolist(), validation=ids[2:3].tolist(), reference=ids[:3].tolist(),
        calibration=ids[3:5].tolist(), test=ids[5:].tolist())))
    measured = metrics(original, saved['labels'], saved['calibration'], saved['test'], bootstrap=0)
    _write_json(score_dir / 'REPORT.json', dict(method=OLD, training_member_count=0,
        condition=dict(benchmark='mimir/github/ngram_13_0.8', condition_seed=1919, models=MODELS),
        settings=dict(audit_seed=1919), evaluation_context=dict(data_manifest=str(manifest), training_regime='pretraining', membership_verified=True),
        observation_archive=dict(path=str(folder / 'observations.npz'), sha256=sha256_file(folder / 'observations.npz')),
        detector=dict(file='../detector.pt', sha256=sha256_file(folder / 'detector.pt')),
        scores_sha256=sha256_file(score_dir / 'scores.npz'), metrics=measured))
    return root


def args_for(root, output, command='run'):
    return [command, '--benchmark', 'mimir13', '--sources', 'github', '--input-root', str(root),
            '--output-root', str(output), '--bootstrap', '8']


def test_real_tcn_replay_preserves_inputs_and_independent_reference(archive, tmp_path, monkeypatch):
    before = {str(p): sha256_file(p) for p in archive.rglob('*') if p.is_file()}
    output = tmp_path / 'output'
    assert runner.main(args_for(archive, output)) == 0
    summary = json.loads((output / 'SUMMARY.json').read_text())
    assert len(summary['rows']) == len(METHODS) and not summary['missing']
    norm = json.loads((output / 'conditions/github/NORMALIZATION.json').read_text())
    assert norm['reference_ids'] == ['doc0', 'doc1', 'doc2']
    assert norm['calibration_used_for_fitting'] is False
    report = json.loads((output / 'conditions/github/REPORT.json').read_text())
    assert report['replay_max_abs_error'] < 1e-12
    assert report['language_model_queries'] == report['detector_training_steps'] == 0
    assert report['metrics'][OLD]['paired_delta_auc_vs_legacy_ci95'] == [0., 0.]
    assert before == {str(p): sha256_file(p) for p in archive.rglob('*') if p.is_file()}
    monkeypatch.setattr(runner, 'predict_frozen', lambda *a: pytest.fail('completed detector replayed'))
    assert runner.main(args_for(archive, output)) == 0
    with pytest.raises(ValueError, match='plan changed'):
        runner.main([*args_for(archive, output)[:-1], '9'])
    (output / 'conditions/github/_COMPLETE.json').write_text('{}')
    with pytest.raises(ValueError, match='completion marker'):
        runner.main(args_for(archive, output))


def test_dry_run_and_isolation(archive, tmp_path):
    output = tmp_path / 'dry'
    assert runner.main(args_for(archive, output, 'dry-run')) == 0
    assert not output.exists()
    with pytest.raises(ValueError, match='separate'):
        runner.main(args_for(archive, archive))
    with (archive / 'tasks/github/seed1919/detector.pt').open('ab') as handle:
        handle.write(b'corrupt')
    with pytest.raises(ValueError, match='checksum'):
        runner.main(args_for(archive, output))


def test_calibration_cannot_become_normalization_reference(archive):
    report = archive / 'tasks/github/seed1919' / OLD / 'REPORT.json'
    paths, _, old = identify_inputs(report, 'mimir/github/ngram_13_0.8', 1919)
    parts = json.loads(paths['partitions'].read_text())
    parts['record_ids']['reference'].append('doc3')
    _write_json(paths['partitions'], parts)
    with pytest.raises(ValueError, match='reference'):
        load_inputs(paths, old)


def test_changed_predictor_cannot_produce_new_metrics(archive, tmp_path, monkeypatch):
    original = runner.predict_frozen
    def changed(*args):
        pmf = original(*args)
        return np.log(np.full_like(pmf, 1/3))
    monkeypatch.setattr(runner, 'predict_frozen', changed)
    output = tmp_path / 'output'
    with pytest.raises(ValueError, match='legacy replay mismatch'):
        runner.main(args_for(archive, output))
    assert not (output / 'conditions/github/REPORT.json').exists()


def test_tied_feedback_keeps_zero_operating_point_and_paired_intervals():
    values, labels = np.ones(5), np.array([0, 0, 1, 0, 1])
    result = runner.measure({OLD: values, 'accept_rate': values, PRIMARY: values}, labels,
                            np.array([0, 1]), np.array([2, 3, 4]), 1919, 8)
    assert result[PRIMARY]['auc'] == .5
    assert result[PRIMARY]['roc_tpr_at_1pct_fpr'] == 0
    assert result[PRIMARY]['calibrated_tpr_at_1pct'] == 0
    assert result[PRIMARY]['paired_delta_auc_vs_accept_rate_ci95'] == [0., 0.]
