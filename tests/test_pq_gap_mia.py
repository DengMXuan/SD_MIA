"""Unit tests for the acceptance-gap membership-audit helpers."""

import numpy as np

from experiments.sd_membership_sft.pq_gap_mia import (
    log_ratio_score,
    mean_alpha_score,
    prob_space_score,
    rank_auc,
    sampled_acceptance_score,
    tpr_at_fpr,
)


class TestRankAuc:
    def test_perfect_separation(self):
        member = np.array([5.0, 6.0, 7.0])
        nonmember = np.array([1.0, 2.0, 3.0])
        assert rank_auc(member, nonmember) == 1.0

    def test_reversed_separation(self):
        member = np.array([1.0, 2.0])
        nonmember = np.array([3.0, 4.0])
        assert rank_auc(member, nonmember) == 0.0

    def test_hand_computed_overlap_with_tie(self):
        # pairs: (2>1 yes), (2=2 tie .5), (3>1 yes), (3>2 yes) -> 3.5/4
        member = np.array([2.0, 3.0])
        nonmember = np.array([1.0, 2.0])
        assert rank_auc(member, nonmember) == np.float64(0.875)

    def test_identical_distributions_all_ties(self):
        values = np.arange(20, dtype=np.float64)
        assert rank_auc(values, values) == 0.5


class TestTprAtFpr:
    def test_hand_computed_threshold(self):
        # nonmember quantile(0.9) of {0..9} is 8.1; members above: {9.5} -> 0.5
        nonmember = np.arange(10, dtype=np.float64)
        member = np.array([8.0, 9.5])
        assert tpr_at_fpr(member, nonmember, 0.10) == 0.5

    def test_perfect_separation(self):
        nonmember = np.arange(10, dtype=np.float64)
        member = np.array([100.0, 200.0])
        assert tpr_at_fpr(member, nonmember, 0.01) == 1.0

    def test_all_below_threshold(self):
        nonmember = np.arange(10, dtype=np.float64)
        member = np.array([0.0, 1.0])
        assert tpr_at_fpr(member, nonmember, 0.10) == 0.0


class TestScores:
    def test_log_ratio_matches_manual(self):
        logp = np.log(np.array([0.5, 0.1]))
        logq = np.log(np.array([0.25, 0.4]))
        expected = np.mean(np.abs(np.log(np.array([2.0, 0.25]))))
        assert np.isclose(log_ratio_score(logp, logq), expected)

    def test_prob_space_matches_manual(self):
        logp = np.log(np.array([0.5, 0.1]))
        logq = np.log(np.array([0.25, 0.4]))
        assert np.isclose(prob_space_score(logp, logq), np.mean([0.25, 0.3]))

    def test_mean_alpha_and_sampled_censoring(self):
        rng = np.random.default_rng(0)
        # p >= q everywhere: acceptance pinned to 1 by the protocol
        logp = np.log(np.array([0.9, 0.8]))
        logq = np.log(np.array([0.5, 0.4]))
        assert mean_alpha_score(logp, logq) == 1.0
        assert sampled_acceptance_score(logp, logq, repeats=16, rng=rng) == 1.0

    def test_mean_alpha_matches_manual(self):
        logp = np.log(np.array([0.5, 0.2]))
        logq = np.log(np.array([0.25, 0.8]))
        # min(1, 2) = 1; min(1, 0.25) = 0.25
        assert np.isclose(mean_alpha_score(logp, logq), np.mean([1.0, 0.25]))

    def test_sampled_score_deterministic_under_seed(self):
        logp = np.log(np.linspace(0.05, 0.9, 64))
        logq = np.log(np.linspace(0.02, 0.95, 64))
        first = sampled_acceptance_score(logp, logq, 16, np.random.default_rng(7))
        second = sampled_acceptance_score(logp, logq, 16, np.random.default_rng(7))
        assert first == second
