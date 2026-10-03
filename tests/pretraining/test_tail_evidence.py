"""Tail protection invariants and unchanged frozen controls through the real runner."""
import json

import numpy as np
import pytest

from experiments.pretraining import preserved_evidence as runner
from experiments.shared.methods import tail_accept_only as tail
from experiments.shared.methods.preserved_accept_only import OLD, PRIMARY as PREVIOUS
from tests.pretraining.test_preserved_evidence import archive, single_thread, args_for


def normalization():
    return tail.fit_reference({name: np.array([-1., 0., 1.]) for name in tail.CHANNELS})


def test_extreme_negative_conditions_cannot_penalize_anchor_and_bonuses_are_bounded():
    features = {name: np.array([-1e6, 0., 1e6]) for name in tail.CHANNELS}
    features['smoothed_reject_tail'] = np.array([-4., 0., 4.])
    result = tail.score(features, normalization())
    bonus = result[tail.PRIMARY] - features['smoothed_reject_tail']
    assert (bonus >= 0).all() and (bonus <= 1.25).all()
    assert bonus[0] == 0 and bonus[-1] == 1.25
    assert (np.diff(result[tail.PRIMARY]) > 0).all()
    # The unbounded control must actually exercise the unbounded failure mode.
    assert result['tail_guarded_unbounded'][-1] > 1e5
    assert result['tail_signed_fusion'][0] < -1e5


def test_zero_rejections_are_finite_and_short_evidence_is_smoothed():
    lengths = np.array([5, 50])
    counts = np.full(lengths.sum(), 2, dtype=int)
    base = {'accept_rate': np.ones(2), 'q_only': -np.ones(2),
            'dense_positive': np.zeros(2), OLD: np.zeros(2)}
    features = tail.components(base, counts, lengths)
    assert np.isfinite(features['smoothed_reject_tail']).all()
    assert features['smoothed_reject_tail'][1] > features['smoothed_reject_tail'][0]
    base['accept_rate'][0] = .5
    with pytest.raises(ValueError, match='disagrees'):
        tail.components(base, counts, lengths)


def test_acceptance_anchor_increases_with_feedback_at_fixed_length():
    lengths = np.full(4, 2)
    counts = np.array([0, 0, 0, 1, 1, 2, 2, 2])
    base = {'accept_rate': counts.reshape(4, 2).mean(1)/2, 'q_only': -np.ones(4),
            'dense_positive': np.zeros(4), OLD: np.zeros(4)}
    features = tail.components(base, counts, lengths)
    assert (np.diff(tail.score(features, normalization())[tail.PRIMARY]) > 0).all()


def test_frozen_map_is_batch_independent_and_constant_channels_are_disabled():
    batch = {name: np.array([1., 2.]) for name in tail.CHANNELS}
    changed = {name: np.array([1., -1e9, 1e9]) for name in tail.CHANNELS}
    before, after = tail.score(batch, normalization()), tail.score(changed, normalization())
    for name in tail.NEW_METHODS:
        assert before[name][0] == after[name][0]
    constant = tail.fit_reference({name: np.ones(4) for name in tail.CHANNELS})
    assert np.array_equal(tail.score(changed, constant)[tail.PRIMARY], np.zeros(3))


def test_tail_runner_replays_both_generations_and_keeps_calibration_separate(archive, tmp_path):
    preserved, output = tmp_path/'preserved', tmp_path/'tail'
    assert runner.main(args_for(archive, preserved)) == 0
    assert runner.main([*args_for(archive, output), '--suite', 'tail']) == 0
    with np.load(preserved/'conditions/github/scores.npz', allow_pickle=False) as old:
        with np.load(output/'conditions/github/scores.npz', allow_pickle=False) as new:
            for name in tail.CONTROL_METHODS:
                np.testing.assert_array_equal(old[name], new[name])
    ref = json.loads((output/'conditions/github/NORMALIZATION.json').read_text())
    assert ref['reference_ids'] == ['doc0', 'doc1', 'doc2']
    assert set(ref['tail_channels']) == set(tail.CHANNELS)
    assert not ref['calibration_used_for_fitting']
    report = json.loads((output/'conditions/github/REPORT.json').read_text())
    assert report['primary_method'] == tail.PRIMARY
    assert len(report['metrics']) == len(tail.METHODS)
    assert report['metrics'][PREVIOUS]['paired_delta_roc_tpr1_vs_previous_ci95'] == [0., 0.]
    assert runner.main([*args_for(archive, output), '--suite', 'tail']) == 0
    with pytest.raises(ValueError, match='plan changed'):
        runner.main(args_for(archive, output))


def test_paired_tail_intervals_preserve_ties_and_do_not_split_false_positives():
    values, labels = np.ones(5), np.array([0, 0, 1, 0, 1])
    variants = {m: values for m in (OLD, PREVIOUS, 'accept_rate', tail.PRIMARY)}
    result = runner.measure(variants, labels, np.array([0, 1]), np.array([2, 3, 4]),
                            1919, 8, compare_previous=True)
    row = result[tail.PRIMARY]
    assert row['auc'] == .5 and row['roc_tpr_at_1pct_fpr'] == 0
    assert row['paired_delta_roc_tpr1_vs_accept_rate_ci95'] == [0., 0.]
    assert row['paired_delta_auc_vs_previous_ci95'] == [0., 0.]
