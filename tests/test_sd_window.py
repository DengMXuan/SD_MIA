from __future__ import annotations

import numpy as np
import pytest

from experiments.sd_membership_sft.sd_window import (
    WindowResult,
    adjacency_windows,
    block_windows,
    build_window_feature_families,
    evaluate_window_audit,
    geometric_windows,
    sign_scores,
    wilcoxon_scores,
)


def _positions(record_count: int, width: int) -> np.ndarray:
    return np.tile(np.arange(width), (record_count, 1))


def test_block_windows_groups_by_document_block() -> None:
    acceptance = np.array([[0.9, 0.1, 0.8, 0.2, 0.95, 0.85]])
    positions = np.array([[0, 1, 3, 7, 8, 9]])  # blocks 0,0,0,1,2,2 (block=4)
    result = block_windows(acceptance, None, positions, block_size=4)
    assert result.window_counts[0] == 3
    assert result.window_means[0, 0] == pytest.approx((0.9 + 0.1 + 0.8) / 3)
    assert result.window_means[0, 1] == pytest.approx(0.2)
    assert result.window_means[0, 2] == pytest.approx(0.9)


def test_block_windows_honours_visibility_mask() -> None:
    # blocks 0,0,1 for block_size 4: the visible block-0 member averages alone
    acceptance = np.array([[0.9, 0.9, 0.1]])
    visible = np.array([[True, False, True]])
    positions = np.array([[0, 1, 5]])
    result = block_windows(acceptance, visible, positions, block_size=4)
    assert result.window_counts[0] == 2
    assert result.window_means[0, 0] == pytest.approx(0.9)
    assert result.window_means[0, 1] == pytest.approx(0.1)


def test_adjacency_windows_clusters_gapped_positions() -> None:
    # document positions 0,1,2 cluster together; 50 and 51 form a second pair
    acceptance = np.array([[0.9, 0.8, 0.1, 0.6, 0.6]])
    positions = np.array([[0, 1, 2, 50, 51]])
    result = adjacency_windows(acceptance, None, positions, gap=4, min_size=2)
    assert result.window_counts[0] == 2
    assert result.window_means[0, 0] == pytest.approx((0.9 + 0.8 + 0.1) / 3)
    assert result.window_means[0, 1] == pytest.approx(0.6)


def test_adjacency_windows_drops_short_clusters() -> None:
    acceptance = np.array([[0.9, 0.8, 0.6]])
    positions = np.array([[0, 1, 40]])
    result = adjacency_windows(acceptance, None, positions, gap=4, min_size=2)
    assert result.window_counts[0] == 1  # only the (0,1) pair survives
    assert result.window_means[0, 0] == pytest.approx(0.85)


def test_geometric_windows_uses_probed_order() -> None:
    rng = np.random.default_rng(0)
    acceptance = rng.uniform(0.4, 0.6, size=(1, 12))
    positions = _positions(1, 12)
    result = geometric_windows(acceptance, None, positions, w_min=2, w_max=6, n_sizes=3)
    # counts now report VALID WINDOW MEANS (11+10+7 for widths 12, sizes 2/3/6),
    # keeping the sign statistic on the same [0, 1] scale as other definitions
    assert result.window_counts[0] == 28
    assert np.isfinite(result.window_means[0]).sum() == 28
    scores = sign_scores(result)
    assert 0.0 <= scores[0] <= 1.0


def test_geometric_sign_bounded_when_windows_missing() -> None:
    # some windows invisible -> NaN means; sign must stay in [0, 1]
    acceptance = np.full((2, 12), 0.9)
    acceptance[1] = 0.2
    visible = np.ones((2, 12), dtype=bool)
    visible[0, [0, 5, 11]] = False  # punch holes so short windows survive, long fail
    positions = _positions(2, 12)
    families = build_window_feature_families(acceptance, positions, visible)
    for key in ("sd_window_geometric_sign", "sd_window_geometric_wilcoxon"):
        assert np.all(families[key] >= -1.0) and np.all(families[key] <= 1.0)
    # record 0: 0.9 where visible -> every valid window > 0.5; record 1: 0.2 -> none
    assert families["sd_window_geometric_sign"][0] == pytest.approx(1.0)
    assert families["sd_window_geometric_sign"][1] == pytest.approx(0.0)


def test_sign_scores_saturate_for_member_like_and_nonmember_like() -> None:
    member = WindowResult(
        np.array([[0.9, 0.85, 0.8]]), np.array([3])
    )
    nonmember = WindowResult(
        np.array([[0.2, 0.3, 0.1]]), np.array([3])
    )
    assert sign_scores(member)[0] == pytest.approx(1.0)
    assert sign_scores(nonmember)[0] == pytest.approx(0.0)


def test_wilcoxon_scores_weight_by_deviation_magnitude() -> None:
    # same two-sided structure, opposite assignment of the large deviation:
    # the signed-rank statistic must follow where the big |deviation| votes
    large_positive = WindowResult(np.array([[0.95, 0.45]]), np.array([2]))
    large_negative = WindowResult(np.array([[0.55, 0.05]]), np.array([2]))
    assert wilcoxon_scores(large_positive)[0] > 0
    assert wilcoxon_scores(large_negative)[0] < 0
    assert (
        wilcoxon_scores(large_positive)[0] == -wilcoxon_scores(large_negative)[0]
    )


def test_build_feature_families_covers_matrix() -> None:
    rng = np.random.default_rng(1)
    acceptance = rng.uniform(0.2, 0.9, size=(6, 26))
    positions = _positions(6, 26)
    families = build_window_feature_families(acceptance, positions)
    for definition in ("block", "adjacency", "geometric"):
        for statistic in ("sign", "wilcoxon"):
            key = f"sd_window_{definition}_{statistic}"
            assert key in families
            assert families[key].shape == (6,)


def test_evaluate_window_audit_end_to_end() -> None:
    rng = np.random.default_rng(3)
    records = 40
    # members carry a uniform acceptance lift; nonmembers sit at the baseline
    # with occasional extreme positions (domain-adaptation style events)
    positions = np.tile(np.arange(24), (records, 1))
    labels = np.zeros(records, dtype=np.int64)
    acceptance = np.empty((records, 24))
    for row in range(records):
        if row % 2 == 0:
            labels[row] = 1
            acceptance[row] = rng.normal(0.55, 0.03, size=24).clip(0.01, 0.99)
        else:
            acceptance[row] = rng.normal(0.40, 0.03, size=24).clip(0.01, 0.99)
            extremes = rng.choice(24, size=2, replace=False)
            acceptance[row, extremes] = rng.uniform(0.9, 0.99, size=2)
    result = evaluate_window_audit(
        acceptance,
        positions,
        selected_activations=None,
        labels=labels,
        calibration=np.arange(0, 16),
        test=np.arange(16, 40),
        bootstrap_repeats=50,
        seed=11,
    )
    expected_keys = {
        "sd_window/acceptance_mean_reference",
        "sd_window/block_sign",
        "sd_window/block_wilcoxon",
        "sd_window/adjacency_sign",
        "sd_window/adjacency_wilcoxon",
        "sd_window/geometric_sign",
        "sd_window/geometric_wilcoxon",
    }
    assert expected_keys.issubset(result["metrics"])
    for key in expected_keys:
        row = result["metrics"][key]
        assert 0.0 <= row["auc"] <= 1.0
        assert np.isfinite(row["auc_ci95_low"])
    assert "sd_window/adjacency_sign minus unwindowed_mean" in result["paired_auc_deltas"]


def test_no_window_records_fall_back_to_own_mean_not_constant() -> None:
    # 4 records; only the first two have admissible adjacency clusters
    acceptance = np.array(
        [
            [0.9, 0.8, 0.7, np.nan],   # cluster (0,1,2) -> sign 3/3
            [0.1, 0.2, 0.3, np.nan],   # cluster (0,1,2) -> sign 0/3
            [0.6, np.nan, np.nan, np.nan],  # no cluster -> own mean 0.6
            [0.2, np.nan, np.nan, np.nan],  # no cluster -> own mean 0.2
        ]
    )
    positions = np.array([[0, 1, 2, 3], [0, 1, 2, 3], [0, 5, 9, 13], [0, 5, 9, 13]])
    families = build_window_feature_families(acceptance, positions)
    adjacency = families["sd_window_adjacency_sign"]
    # records 0/1 keep window votes; records 2/3 get their OWN means, not one constant
    assert adjacency[0] == pytest.approx(1.0)
    assert adjacency[1] == pytest.approx(0.0)
    assert adjacency[2] == pytest.approx(0.6)
    assert adjacency[3] == pytest.approx(0.2)


def test_window_weights_zero_outside_clusters_and_fallback() -> None:
    from experiments.sd_membership_sft.sd_window import _window_weights

    acceptance = np.array(
        [
            [0.9, 0.8, 0.7, 0.4, 0.4],   # cluster (0,1,2) -> weights 0.9/0.8/0.7, 0 elsewhere
            [0.5, np.nan, 0.5, 0.5, 0.5],  # no cluster -> fallback to visible acceptance
        ]
    )
    positions = np.array([[0, 1, 2, 40, 50], [0, 1, 2, 3, 4]])
    visible = ~np.isnan(acceptance)
    weights = _window_weights(acceptance, positions, visible, gap=4)
    assert weights[0, :3].tolist() == [0.9, 0.8, 0.7]
    assert weights[0, 3:].tolist() == [0.0, 0.0]
    assert weights[1, 0] == pytest.approx(0.5)  # NaN position excluded
    assert weights[1, 1] == 0.0
    assert weights[1, 2:].tolist() == [0.5, 0.5, 0.5]


def test_conditioned_activations_are_actually_window_conditioned() -> None:
    from experiments.sd_membership_sft.sd_window import (
        _window_conditioned_activations,
    )

    rng = np.random.default_rng(5)
    records, width, n_layers, n_stats = 2, 8, 3, 4
    activations = rng.normal(size=(records, width, n_layers, n_stats)).astype(np.float32)
    positions = _positions(records, width)
    # same global mean acceptance, different in-cluster structure
    acceptance = np.full((records, width), 0.5)
    acceptance[0, :4] = 0.95  # record 0: cluster carries all the lift
    acceptance[1, 4:] = 0.95  # record 1: lift entirely outside the cluster
    visible = np.ones((records, width), dtype=bool)
    out = _window_conditioned_activations(activations, acceptance, positions, visible, gap=2)
    assert out.shape[0] == records
    assert np.isfinite(out).all()
    # the accepted pool of record 0 must reflect in-cluster activations only,
    # so the two records' outputs cannot coincide
    assert not np.allclose(out[0], out[1])


def test_all_invisible_positions_degrade_to_baseline() -> None:
    # degenerate case: every record loses every window -> deterministic
    # baseline fallback (0.5), never a NaN or a crash
    from experiments.sd_membership_sft.sd_window import evaluate_window_audit

    rng = np.random.default_rng(9)
    records = 20
    acceptance = rng.uniform(0.3, 0.7, size=(records, 10))
    positions = _positions(records, 10)
    labels = (np.arange(records) % 2).astype(np.int64)
    result = evaluate_window_audit(
        acceptance,
        positions,
        selected_activations=None,
        labels=labels,
        calibration=np.arange(0, 8),
        test=np.arange(8, records),
        bootstrap_repeats=10,
        seed=1,
        visible=np.zeros((records, 10), dtype=bool),
    )
    for key, row in result["metrics"].items():
        if key.endswith("_sign"):
            assert row["auc"] == pytest.approx(0.5, abs=0.01)
