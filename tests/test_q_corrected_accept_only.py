import math

import numpy as np

from experiments.sd_membership_sft.q_corrected_accept_only import (
    QBinomialObservation,
    acceptance_probability,
    corrected_delta0,
    estimate_joint,
    joint_binomial_log_likelihood,
    likelihood_ratio_interval,
    recover_from_exact_alpha,
    synthetic_invariance_report,
)


def test_q_active_equal_q0_is_exactly_the_fixed_q_likelihood() -> None:
    fixed = QBinomialObservation(q=0.6, accepts=37, trials=64)
    active_with_same_q = QBinomialObservation(q=0.6, accepts=37, trials=64)
    grid = np.linspace(0.001, 0.999, 1000)
    fixed_values = joint_binomial_log_likelihood(grid, (fixed,))
    active_values = joint_binomial_log_likelihood(grid, (active_with_same_q,))
    assert np.array_equal(fixed_values, active_values)
    assert estimate_joint(0.6, (fixed,)) == estimate_joint(0.6, (active_with_same_q,))


def test_exact_unsaturated_recovery_is_invariant_to_active_q() -> None:
    q0 = 0.6
    p = 0.8
    expected_delta0 = math.log(p / q0)
    for active_q in (0.85, 0.9, 0.95, 1.0):
        alpha = acceptance_probability(p, active_q)
        assert alpha < 1.0
        recovery = recover_from_exact_alpha(q0, active_q, alpha)
        assert recovery.point_identified
        assert math.isclose(recovery.p_lower, p, rel_tol=0.0, abs_tol=1e-15)
        assert math.isclose(
            corrected_delta0(q0, active_q, alpha), expected_delta0, abs_tol=1e-15
        )


def test_preregistered_p08_numerical_example_recovers_delta0() -> None:
    recovered = corrected_delta0(q0=0.6, q=0.9, alpha=0.8 / 0.9)
    assert math.isclose(recovered, math.log(0.8 / 0.6), abs_tol=1e-15)


def test_preregistered_p03_fixed_and_active_q_recover_same_delta0() -> None:
    fixed = corrected_delta0(q0=0.6, q=0.6, alpha=0.3 / 0.6)
    active = corrected_delta0(q0=0.6, q=0.9, alpha=0.3 / 0.9)
    assert math.isclose(fixed, math.log(0.3 / 0.6), abs_tol=1e-15)
    assert math.isclose(active, fixed, abs_tol=1e-15)


def test_saturated_exact_alpha_is_an_interval_not_a_smoothed_point() -> None:
    recovery = recover_from_exact_alpha(q0=0.6, q=0.6, alpha=1.0)
    assert not recovery.point_identified
    assert recovery.p_lower == 0.6
    assert recovery.p_upper == 1.0
    assert recovery.delta0_lower == 0.0
    assert math.isclose(recovery.delta0_upper, math.log(1.0 / 0.6))
    with np.testing.assert_raises_regex(ValueError, "saturated"):
        corrected_delta0(q0=0.6, q=0.6, alpha=1.0)


def test_larger_k_narrows_interval_without_changing_exact_ratio_mle() -> None:
    small = (QBinomialObservation(q=0.6, accepts=10, trials=20),)
    large = (QBinomialObservation(q=0.6, accepts=100, trials=200),)
    small_estimate = estimate_joint(0.6, small)
    large_estimate = estimate_joint(0.6, large)
    assert math.isclose(small_estimate.mle_p_lower, 0.3, abs_tol=2e-8)
    assert math.isclose(large_estimate.mle_p_lower, 0.3, abs_tol=2e-8)
    small_width = small_estimate.profile_p_upper - small_estimate.profile_p_lower
    large_width = large_estimate.profile_p_upper - large_estimate.profile_p_lower
    assert large_width < small_width


def test_all_accept_reports_likelihood_plateau_and_finite_score() -> None:
    observations = (
        QBinomialObservation(q=0.6, accepts=16, trials=16),
        QBinomialObservation(q=0.9, accepts=32, trials=32),
    )
    estimate = estimate_joint(0.6, observations)
    assert estimate.censoring == "lower"
    assert estimate.mle_p_lower == 0.9
    assert estimate.mle_p_upper == 1.0
    assert estimate.profile_p_upper == 1.0
    assert np.isfinite(estimate.posterior_delta0_median)
    assert np.isfinite(estimate.max_log_likelihood)


def test_all_reject_retains_binomial_information_and_finite_score() -> None:
    few = (QBinomialObservation(q=0.9, accepts=0, trials=4),)
    many = (QBinomialObservation(q=0.9, accepts=0, trials=64),)
    few_estimate = estimate_joint(0.6, few)
    many_estimate = estimate_joint(0.6, many)
    assert few_estimate.censoring == many_estimate.censoring == "upper"
    assert few_estimate.mle_p_lower == few_estimate.mle_p_upper == 0.0
    assert np.isfinite(few_estimate.posterior_delta0_median)
    assert np.isfinite(many_estimate.posterior_delta0_median)
    assert many_estimate.profile_p_upper < few_estimate.profile_p_upper


def test_joint_likelihood_contains_rejection_term() -> None:
    observation = (QBinomialObservation(q=0.6, accepts=1, trials=2),)
    p = 0.3
    expected = math.log(0.5) + math.log(1.0 - 0.5)
    assert math.isclose(joint_binomial_log_likelihood(p, observation), expected)
    assert joint_binomial_log_likelihood(0.6, observation) == -math.inf


def test_joint_multi_q_profile_interval_is_narrower_on_consistent_data() -> None:
    first = QBinomialObservation(q=0.6, accepts=40, trials=80)
    second = QBinomialObservation(q=0.9, accepts=40, trials=120)
    first_interval = likelihood_ratio_interval((first,))
    joint_interval = likelihood_ratio_interval((first, second))
    assert joint_interval[1] - joint_interval[0] < first_interval[1] - first_interval[0]


def test_joint_estimate_is_expressed_against_original_q0() -> None:
    observations = (QBinomialObservation(q=0.9, accepts=80, trials=90),)
    estimate = estimate_joint(q0=0.6, observations=observations)
    assert math.isclose(estimate.mle_p_lower, 0.8, abs_tol=2e-8)
    assert math.isclose(
        math.log(estimate.mle_p_lower / estimate.q0), math.log(0.8 / 0.6), abs_tol=3e-8
    )


def test_tiny_probabilities_use_relative_numerical_scale() -> None:
    q0 = 6e-12
    observation = (QBinomialObservation(q=9e-12, accepts=80, trials=90),)
    estimate = estimate_joint(q0, observation)
    assert math.isclose(estimate.mle_p_lower, 8e-12, rel_tol=1e-7)
    assert estimate.profile_p_lower < 8e-12 < estimate.profile_p_upper
    assert np.isfinite(estimate.posterior_delta0_median)


def test_synthetic_invariance_gate_passes() -> None:
    report = synthetic_invariance_report()
    assert report["status"] == "pass"
    assert all(case["max_abs_error"] < 1e-12 for case in report["cases"])


def test_invalid_counts_and_probabilities_are_rejected() -> None:
    with np.testing.assert_raises(ValueError):
        QBinomialObservation(q=0.0, accepts=0, trials=1)
    with np.testing.assert_raises(ValueError):
        QBinomialObservation(q=0.5, accepts=2, trials=1)
    with np.testing.assert_raises(ValueError):
        acceptance_probability(1.1, 0.5)
