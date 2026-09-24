import numpy as np
import torch

from experiments.sd_membership_sft.archive.active_protocol_design import GRID, METHODS, CachedVerifier, binary_js, candidate_proposal_weight, fit_prior, multiq_log_likelihood, replay_policy


def test_candidate_q_is_realizable_by_a_normalized_distribution():
    base = np.array([.1, .2, .7])
    desired = np.sqrt(base[1])
    weight = float(candidate_proposal_weight(np.array(base[1]), np.array(desired)))
    proposal = (1 - weight) * base + weight * np.array([0., 1., 0.])
    np.testing.assert_allclose(proposal.sum(), 1.)
    np.testing.assert_allclose(proposal[1], desired)
    assert np.all(proposal >= 0)


def test_saturated_observations_keep_exact_impossible_event_likelihood():
    logq = torch.full((1, 2), -1.)
    counts = torch.full((1, 2, 5), 2.)
    accepted = multiq_log_likelihood(logq, counts)
    assert torch.isfinite(accepted).all()
    counts[..., 0] = 1
    rejected = multiq_log_likelihood(logq, counts)
    assert torch.isneginf(rejected[..., -1]).all()
    assert torch.isfinite(rejected[..., 0]).all()


def test_all_policies_use_equal_budget_and_consistent_query_streams():
    q = np.full((3, 8), -2.)
    p = q + np.linspace(-1., 1., 8)
    prior = np.ones((3, 8, len(GRID))) / len(GRID)
    oracle = CachedVerifier(p, q, np.arange(3), 42)
    pilot = None
    for method in METHODS:
        first, cost = replay_policy(q, prior, oracle.query, method, budgets=(1, 2, 4))
        second, _ = replay_policy(q, prior, oracle.query, method, budgets=(1, 2, 4))
        np.testing.assert_array_equal(cost["action_counts"].sum((1, 2)), [32, 32, 32])
        assert cost["per_position_counts"].min() >= 1
        assert cost["per_position_counts"].max() <= 16
        if pilot is None:
            pilot = first[1]
        np.testing.assert_array_equal(first[1], pilot)
        for budget in first:
            assert np.isfinite(first[budget]).all()
            np.testing.assert_array_equal(first[budget], second[budget])


def test_js_has_no_value_for_identical_or_jointly_saturated_predictions():
    probabilities = np.array([0., .1, .5, .9, 1.])
    np.testing.assert_allclose(binary_js(probabilities, probabilities), 0., atol=1e-14)
    assert binary_js(np.array(.2), np.array(.8)) > 0


def test_nonmember_prior_fit_cannot_use_heldout_counts():
    previous_threads = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        rng = np.random.default_rng(1)
        q = -rng.uniform(.1, 4, (8, 8))
        counts = rng.integers(0, 3, (8, 8, 5))
        parts = {"train": np.array([0, 1, 2]), "validation": np.array([3, 4])}
        first = fit_prior(q, counts, parts, seed=17, epochs=2)
        changed = counts.copy()
        changed[5:] = 2 - changed[5:]
        second = fit_prior(q, changed, parts, seed=17, epochs=2)
        assert first[-2:] == second[-2:]
        for key, value in first[0].state_dict().items():
            torch.testing.assert_close(value, second[0].state_dict()[key])
    finally:
        torch.set_num_threads(previous_threads)
