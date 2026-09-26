"""Standalone scoring, calibration and artifact-isolation regression checks."""
import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest
import torch
from scipy.special import logsumexp

spec = importlib.util.spec_from_file_location('pythia_evidence_standalone', Path(__file__).with_name('run.py'))
runner = importlib.util.module_from_spec(spec)
spec.loader.exec_module(runner)


@pytest.fixture(autouse=True)
def single_thread():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


def test_partial_correction_and_factorial_tilts_match_direct_definition():
    pmf = np.array([[.1, .2, .7], [.2, .5, .3], [.4, .1, .5], [.3, .3, .4], [.05, .1, .85]])
    counts, lengths = np.array([2, 1, 0, 2, 2]), np.array([2, 3])
    scores = runner.score_variants(np.log(pmf), counts, lengths)
    expected_mean = pmf @ np.arange(3) / 2
    accepted = np.array([counts[:2].mean(), counts[2:].mean()]) / 2
    predicted = np.array([expected_mean[:2].mean(), expected_mean[2:].mean()])
    np.testing.assert_allclose(scores['accept_rate'], accepted)
    np.testing.assert_allclose(scores['full_residual'], accepted - predicted)
    np.testing.assert_allclose(scores['partial_residual_050'], accepted - .5 * predicted)
    for method, settings in runner.METHODS.items():
        if settings['kind'] != 'tilt':
            continue
        expected = []
        for left, right in [(0, 2), (2, 5)]:
            components = []
            for rho in settings['rho']:
                for eta in settings['eta']:
                    tilted = np.exp(eta * counts[left:right]) / (pmf[left:right] @ np.exp(eta * np.arange(3)))
                    components.append(np.log((1 - rho) + rho * tilted).sum())
            expected.append(logsumexp(components) - np.log(len(components)))
        np.testing.assert_allclose(scores[method], expected, rtol=1e-12, atol=1e-12)


def test_ties_auc_and_independent_calibration_are_not_roc_thresholds():
    values = np.array([.1, .2, .9, .8])
    labels = np.array([0, 0, 1, 0])
    scores = {method: values.copy() for method in runner.METHODS}
    result = runner.measure(scores, labels, np.array([0, 1]), np.array([2, 3]), 1919, 8)
    for row in result.values():
        assert row['auc'] == row['roc_tpr_at_1pct_fpr'] == 1.
        assert row['calibrated_tpr_at_1pct'] == 0.  # Minimum p=1/3 with two calibration docs.
        assert row['paired_delta_auc_ci95'] == [0., 0.]
    tied = {method: np.ones(4) for method in runner.METHODS}
    result = runner.measure(tied, labels, np.array([0, 1]), np.array([2, 3]), 1919, 0)
    assert result[runner.OLD]['auc'] == .5
    assert result[runner.OLD]['roc_tpr_at_10pct_fpr'] == 0.
    assert result[runner.OLD]['calibrated_actual_fpr_at_10pct'] == 0.


def test_auc_matches_pairwise_definition():
    labels = np.array([1, 0, 1, 0, 1, 0])
    values = np.array([1, 1, 0, 2, 3, 0])
    diff = values[labels == 1][:, None] - values[labels == 0]
    expected = ((diff > 0) + .5 * (diff == 0)).mean()
    assert runner.auc(values, labels) == expected


def make_state():
    torch.manual_seed(1919)
    channels = 4
    state = {}
    for prefix, module in [('projection', torch.nn.Linear(5, channels)),
                           ('head', torch.nn.Linear(channels, 18))]:
        state.update({prefix + '.' + k: v for k, v in module.state_dict().items()})
    for i, dilation in enumerate((1, 2, 4, 8)):
        for prefix, module in [(f'convs.{i}', torch.nn.Conv1d(channels, channels, 3, padding=dilation, dilation=dilation)),
                               (f'norms.{i}', torch.nn.LayerNorm(channels))]:
            state.update({prefix + '.' + k: v for k, v in module.state_dict().items()})
    grid = torch.sigmoid(torch.linspace(-7., 7., 18)).double()
    state['log_kernel'] = torch.stack([2*torch.log1p(-grid), np.log(2.) + grid.log() + torch.log1p(-grid),
                                       2*grid.log()]).float()
    return state


@pytest.fixture
def archive(tmp_path):
    root = tmp_path / 'legacy'
    folder = root / 'tasks/github/seed1919'
    score_dir = folder / runner.OLD
    score_dir.mkdir(parents=True)
    ids = np.array([f'doc{i}' for i in range(6)])
    labels = np.array([0, 0, 0, 0, 1, 0])
    lengths = np.array([3, 4, 5, 3, 6, 4])
    features = np.zeros((lengths.sum(), 6), dtype=np.float32)
    features[:, 0] = -1
    features[:, 4] = .5
    counts = np.arange(lengths.sum(), dtype=np.uint8) % 3
    np.savez_compressed(folder / 'observations.npz', record_ids=ids, labels=labels, lengths=lengths,
                        features=features, counts=counts, document_indices=np.arange(6), start_indices=np.zeros(6, dtype=int),
                        record_roles=np.array(['audit_auxiliary'] * 4 + ['member', 'nonmember']))
    runner.write_json(folder / 'observations.npz.json', dict(schema='sd_mia_protocol_observations_v1', feature_names=runner.FEATURES,
        archive_sha256=runner.sha256(folder / 'observations.npz'), contract=dict(protocol='fixed', starts=['fixed'], seed=1919)))
    torch.save(dict(state_dict=make_state(), mean=torch.zeros(5), scale=torch.ones(5)), folder / 'detector.pt')
    pmf = runner.predict(folder / 'detector.pt', features, lengths)
    score = runner.score_variants(pmf, counts, lengths)[runner.OLD][2:]
    saved = dict(record_ids=ids[2:], labels=labels[2:], scores=score, calibration=np.array([0, 1]), test=np.array([2, 3]))
    np.savez_compressed(score_dir / 'scores.npz', **saved)
    records = folder / 'records.jsonl'
    records.write_text(''.join(json.dumps(dict(record_id=key, label=int(label), token_ids=[1] * (int(length) + 1))) + '\n'
                               for key, label, length in zip(ids, labels, lengths)))
    models = dict(target=dict(repo_id='EleutherAI/pythia-6.9b', revision='a'*40),
                  draft=dict(repo_id='EleutherAI/pythia-1.4b', revision='b'*40))
    manifest = folder / 'manifest.json'
    runner.write_json(manifest, dict(models=models, selection_seed=1919, records_file='records.jsonl',
                                    records_sha256=runner.sha256(records)))
    runner.write_json(folder / 'PARTITIONS.json', dict(seed=1919, manifest_sha256=runner.sha256(manifest),
        record_ids=dict(train=ids[:1].tolist(), validation=ids[1:2].tolist(), reference=ids[:2].tolist(),
                        calibration=ids[2:4].tolist(), test=ids[4:].tolist())))
    metrics = runner.measure({runner.OLD: score}, labels[2:], saved['calibration'], saved['test'], 1919, 0)[runner.OLD]
    runner.write_json(score_dir / 'REPORT.json', dict(method=runner.OLD,
        condition=dict(benchmark='mimir/github/ngram_13_0.8', condition_seed=1919, models=models),
        settings=dict(audit_seed=1919), training_member_count=0,
        evaluation_context=dict(data_manifest=str(manifest), membership_verified=True, training_regime='pretraining'),
        detector=dict(file='../detector.pt', sha256=runner.sha256(folder / 'detector.pt')),
        observation_archive=dict(path=str(folder / 'observations.npz'), sha256=runner.sha256(folder / 'observations.npz')),
        scores_sha256=runner.sha256(score_dir / 'scores.npz'), metrics=metrics))
    return root


def test_end_to_end_preserves_inputs_replays_control_and_resumes(archive, tmp_path, monkeypatch):
    output = tmp_path / 'new'
    before = {str(p): runner.sha256(p) for p in archive.rglob('*') if p.is_file()}
    args = ['run', '--input-root', str(archive), '--output-root', str(output), '--sources', 'github', '--seeds', '1919', '--bootstrap', '4']
    runner.main(args)
    summary = runner.read_json(output / 'SUMMARY.json')
    assert len(summary['rows']) == 7 and summary['missing'] == []
    report = runner.read_json(output / 'conditions/github/seed1919/REPORT.json')
    assert report['replay_max_abs_error'] == 0.
    assert report['language_model_queries'] == report['detector_training_steps'] == 0
    assert before == {str(p): runner.sha256(p) for p in archive.rglob('*') if p.is_file()}
    monkeypatch.setattr(runner, 'predict', lambda *a: pytest.fail('completed detector was replayed'))
    runner.main(args)
    with pytest.raises(ValueError, match='plan changed'):
        runner.main([*args[:-1], '5'])


def test_corrupted_input_is_rejected(archive, tmp_path):
    with (archive / 'tasks/github/seed1919/detector.pt').open('ab') as handle:
        handle.write(b'corrupted')
    with pytest.raises(ValueError, match='checksum'):
        runner.main(['run', '--input-root', str(archive), '--output-root', str(tmp_path / 'new'),
                     '--sources', 'github', '--seeds', '1919'])


def test_empty_completion_marker_cannot_bypass_output_verification(archive, tmp_path):
    output = tmp_path / 'new'
    args = ['run', '--input-root', str(archive), '--output-root', str(output),
            '--sources', 'github', '--seeds', '1919', '--bootstrap', '0']
    runner.main(args)
    runner.write_json(output / 'conditions/github/seed1919/_COMPLETE.json', {})
    with pytest.raises(ValueError, match='completion marker'):
        runner.main(args)


def test_legacy_replay_disagreement_blocks_new_metrics(archive, tmp_path, monkeypatch):
    original = runner.score_variants
    def altered(*args):
        result = original(*args)
        result[runner.OLD] += .1
        return result
    monkeypatch.setattr(runner, 'score_variants', altered)
    with pytest.raises(ValueError, match='legacy replay mismatch'):
        runner.main(['run', '--input-root', str(archive), '--output-root', str(tmp_path / 'new'),
                     '--sources', 'github', '--seeds', '1919'])
    assert not (tmp_path / 'new/conditions/github/seed1919/REPORT.json').exists()


def test_dry_run_and_dataset_selection_create_no_outputs(tmp_path, capsys):
    output = tmp_path / 'new'
    runner.main(['dry-run', '--input-root', str(tmp_path / 'missing'), '--output-root', str(output)])
    plan = json.loads(capsys.readouterr().out)
    assert plan['archives'] == 9 and plan['scorer_rows'] == 63
    assert not output.exists()
    runner.main(['dry-run', '--input-root', str(tmp_path / 'missing'), '--output-root', str(output),
                 '--sources', 'all', '--seeds', '1949'])
    plan = json.loads(capsys.readouterr().out)
    assert plan['archives'] == 8 and plan['scorer_rows'] == 56
    assert not output.exists()
    with pytest.raises(ValueError, match='separate'):
        runner.main(['run', '--input-root', str(tmp_path), '--output-root', str(output)])
