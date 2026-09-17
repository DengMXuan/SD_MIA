import numpy as np

from experiments.sd_membership_sft.adaptive_window_accept_only import (
    coverage_constrained_schedule,
    evt_membership_metrics,
    fit_nonmember_model,
    raw_fragment_scores,
    token_features,
    window_priority,
)


def test_token_features_are_q_only_aligned_and_finite() -> None:
    logq = np.log(np.asarray([0.1, 0.2, 0.5, 0.9, 0.3]))
    features = token_features(logq, np.asarray([3, 2]))
    assert features.shape == (5, 8)
    assert np.all(np.isfinite(features))


def test_nonmember_model_learns_acceptance_direction() -> None:
    q = np.linspace(-4.0, 0.0, 200)
    features = np.column_stack([q, q**2])
    response = (q > -2.0).astype(float)
    model = fit_nonmember_model(
        features,
        response,
        np.ones(len(q), dtype=bool),
        trials_per_token=1,
        iterations=30,
    )
    prediction = model.predict(features)
    assert np.mean(prediction[-50:]) > np.mean(prediction[:50])


def test_raw_scores_include_lowq_and_windows() -> None:
    bits = np.asarray([1, 1, 0, 1, 0, 0], dtype=float)
    predicted = np.full(6, 0.5)
    logq = np.log(np.asarray([0.1, 0.2, 0.9, 0.8, 0.7, 0.6]))
    scores = raw_fragment_scores(bits, predicted, logq, np.asarray([6]), 1)
    assert scores["lowq_50"][0] == 2 / 3
    assert all(np.isfinite(value[0]) for value in scores.values())


def test_window_priority_prefers_supported_lowq_positions() -> None:
    pilot = np.asarray([1, 1, 1, 0, 0, 0], dtype=float)
    predicted = np.full(6, 0.5)
    logq = np.log(np.asarray([0.1, 0.2, 0.3, 0.8, 0.9, 0.95]))
    priority = window_priority(pilot, predicted, logq, width=2)
    assert np.mean(priority[:3]) > np.mean(priority[3:])


def test_coverage_schedule_has_exact_budget_and_normal_pilot() -> None:
    length, budget = 20, 8
    schedule = coverage_constrained_schedule(
        length,
        budget,
        np.linspace(0.0, 1.0, length),
        adaptive_fraction=0.5,
    )
    assert np.sum(schedule >= 0) == length * budget
    assert np.all(schedule[:, 0] == 0)
    assert np.all(np.sum(schedule >= 0, axis=1) >= 1)


def test_evt_thresholds_are_finite_and_ordered() -> None:
    rng = np.random.default_rng(7)
    calibration = rng.exponential(size=200)
    member = rng.exponential(scale=2.0, size=100)
    nonmember = rng.exponential(size=100)
    scores = np.r_[calibration, member, nonmember]
    labels = np.r_[np.zeros(200), np.ones(100), np.zeros(100)].astype(int)
    result = evt_membership_metrics(
        scores,
        labels,
        np.arange(200),
        np.arange(200, 400),
    )
    threshold_1 = result["tpr_at_fpr"]["1%"]["threshold"]
    threshold_10 = result["tpr_at_fpr"]["10%"]["threshold"]
    assert np.isfinite(threshold_1)
    assert threshold_1 > threshold_10
