import numpy as np

from experiments.sd_membership_sft.archive.adaptive_window_accept_only import membership_metrics
from experiments.sd_membership_sft.archive.analyze_dynamic_marginal_query import _metric
from experiments.sd_membership_sft.archive.dynamic_marginal_query import STATE_DIM, state_features, waterfill_counts


def test_waterfill_uses_exact_budget_and_unequal_counts() -> None:
    utility = np.asarray([10.0, 3.0, 1.0, 0.1])
    current = np.ones(4)
    counts = waterfill_counts(utility, current, 6, max_add=4)
    assert np.sum(counts) == 6
    assert np.all((0 <= counts) & (counts <= 4))
    assert counts[0] > counts[-1]


def test_waterfill_respects_variable_remaining_caps() -> None:
    utility = np.asarray([10.0, 4.0, 2.0, 1.0])
    current = np.ones(4)
    caps = np.asarray([0, 1, 3, 3])
    counts = waterfill_counts(
        utility,
        current,
        5,
        max_add=4,
        remaining_cap=caps,
        decay_power=1.0,
    )
    assert np.sum(counts) == 5
    assert np.all(counts <= caps)
    assert counts[0] == 0


def test_state_features_encode_variable_query_state() -> None:
    static = np.zeros((3, 8), dtype=np.float64)
    logq = np.log(np.asarray([0.2, 0.5, 0.8]))
    accepts = np.asarray(
        [[1, 0, 0, 0, 0], [0, 1, 0, 0, 0], [1, 0, 1, 0, 0]], dtype=np.int64
    )
    trials = np.asarray(
        [[1, 0, 0, 0, 0], [1, 1, 0, 0, 0], [1, 0, 1, 0, 0]], dtype=np.int64
    )
    features, delta, censoring = state_features(static, logq, accepts, trials)
    assert features.shape == (3, STATE_DIM)
    assert delta.shape == censoring.shape == (3,)
    assert np.all(np.isfinite(features))


def test_fast_bootstrap_metrics_match_canonical_metrics_with_ties() -> None:
    rng = np.random.default_rng(42)
    labels = np.r_[np.zeros(40, dtype=np.int64), np.ones(40, dtype=np.int64)]
    scores = rng.integers(0, 8, size=len(labels)).astype(np.float64)
    calibration = np.arange(20)
    test = np.arange(20, len(labels))
    expected = membership_metrics(scores, labels, calibration, test)
    actual = _metric(scores, labels, calibration, test)
    np.testing.assert_allclose(
        actual,
        [
            expected["auc"],
            expected["pauc_0_10"],
            expected["tpr_at_fpr"]["1%"]["tpr"],
            expected["tpr_at_fpr"]["10%"]["tpr"],
        ],
    )
