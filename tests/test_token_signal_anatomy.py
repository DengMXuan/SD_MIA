from pathlib import Path

import numpy as np

from experiments.sd_membership_sft.stat_delta_mia import DeltaData
from experiments.sd_membership_sft.full_delta_mia import partial_auc
from experiments.sd_membership_sft.token_signal_anatomy import (
    Roles,
    direction_scores,
    drop_final_cached_token,
    evaluate_score,
    fast_partial_auc,
    load_paired_logps,
    matching_global_baseline,
    paired_pauc_delta,
    signal_values,
    top_fraction_scores,
    window_scores,
)


def _small_data() -> DeltaData:
    lengths = np.asarray([5, 5, 5, 5], dtype=np.int64)
    values = np.asarray([
        -1, 0, 1, 2, 9,
        -2, -1, 0, 1, 9,
        -1, -1, 0, 0, 9,
        -2, -2, -1, 0, 9,
    ], dtype=np.float32)
    return DeltaData(
        labels=np.asarray([1, 1, 0, 0]),
        record_ids=np.asarray(["m0", "m1", "n0", "n1"]),
        lengths=lengths,
        offsets=np.r_[0, np.cumsum(lengths)],
        delta=values,
    )


def _roles() -> Roles:
    return Roles(
        n_ref=np.asarray([2]),
        m_diag=np.asarray([0]),
        n_cal=np.asarray([3]),
        t_member=np.asarray([1]),
        t_nonmember=np.asarray([2]),
        reserved_member=np.asarray([], dtype=np.int64),
    )


def test_signal_directions_have_explicit_meanings() -> None:
    delta = np.asarray([-2.0, 1.0, 3.0])
    assert signal_values(delta, "positive").tolist() == [0.0, 1.0, 3.0]
    assert signal_values(delta, "negative").tolist() == [2.0, 0.0, 0.0]
    assert signal_values(delta, "absolute").tolist() == [2.0, 1.0, 3.0]
    assert signal_values(delta, "positive_indicator").tolist() == [0.0, 1.0, 1.0]


def test_drop_final_token_drops_each_record_not_global_tail() -> None:
    data = drop_final_cached_token(_small_data())
    assert data.lengths.tolist() == [4, 4, 4, 4]
    assert data.offsets.tolist() == [0, 4, 8, 12, 16]
    assert 9 not in data.delta


def test_direction_score_definitions() -> None:
    result = direction_scores(_small_data())
    assert set(result) == {
        "mean_signed_delta", "mean_positive_delta", "mean_negative_delta",
        "mean_abs_delta", "positive_delta_fraction",
    }
    assert np.isclose(result["mean_signed_delta"][0], 11 / 5)
    assert np.isclose(result["mean_negative_delta"][0], 1 / 5)
    assert np.isclose(result["positive_delta_fraction"][0], 3 / 5)


def test_top_keep_drop_and_random_are_label_free_and_deterministic() -> None:
    first = top_fraction_scores(_small_data(), fractions=(0.4, 1.0), random_repeats=3, seed=7)
    second = top_fraction_scores(_small_data(), fractions=(0.4, 1.0), random_repeats=3, seed=7)
    assert first.keys() == second.keys()
    assert all(np.array_equal(first[key], second[key]) for key in first)
    assert "top_drop_absolute_40pct" in first
    assert "top_drop_absolute_100pct" not in first
    # Top 40% of abs([-1, 0, 1, 2, 9]) is {9, 2}.
    assert np.isclose(first["top_keep_absolute_40pct"][0], 5.5)
    assert np.isclose(first["top_drop_absolute_40pct"][0], 2 / 3)
    assert np.allclose(first["top_keep_absolute_100pct"], first["random_keep_absolute_100pct"])


def test_window_shuffle_is_deterministic_and_width_larger_than_record_is_valid() -> None:
    first = window_scores(_small_data(), windows=(2, 8), shuffle_repeats=3, seed=9)
    second = window_scores(_small_data(), windows=(2, 8), shuffle_repeats=3, seed=9)
    assert first.keys() == second.keys()
    assert all(np.array_equal(first[key], second[key]) for key in first)
    # A too-wide window reduces to the record mean and is order invariant.
    assert np.allclose(first["window_max_signed_w8"], first["window_max_signed_w8_shuffled"])


def test_metrics_use_explicit_roles_and_record_bootstrap() -> None:
    scores = np.asarray([4.0, 3.0, 1.0, 0.0])
    result = evaluate_score(scores, _roles(), repeats=5, seed=1)
    assert result["diagnostic_M_diag_vs_N_ref"]["auc"]["point"] == 1.0
    assert result["exploratory_T"]["auc"]["point"] == 1.0
    delta = paired_pauc_delta(scores, scores, _roles(), repeats=5, seed=2)
    assert delta["exploratory_T"]["point"] == 0.0


def test_fast_partial_auc_matches_repository_metric_with_ties() -> None:
    scores = np.asarray([3.0, 3.0, 2.0, 1.0, 1.0, 0.0])
    labels = np.asarray([1, 0, 1, 0, 1, 0])
    assert np.isclose(fast_partial_auc(scores, labels), partial_auc(scores, labels))


def test_sparse_statistics_use_like_for_like_global_baselines() -> None:
    assert matching_global_baseline("top_keep_positive_05pct") == "mean_positive_delta"
    assert matching_global_baseline("top_drop_negative_10pct") == "mean_negative_delta"
    assert matching_global_baseline("window_max_absolute_w8") == "mean_abs_delta"
    assert matching_global_baseline("window_max_signed_w16") == "mean_signed_delta"
    assert matching_global_baseline("window_positive_rate_w32") == "positive_delta_fraction"


def test_paired_logp_loader_validates_and_drops_each_final_token(tmp_path: Path) -> None:
    original = _small_data()
    target = original.delta + 0.25
    draft = np.full(len(original.delta), 0.25, dtype=np.float32)
    path = tmp_path / "pq.npz"
    np.savez(path, lengths=original.lengths, target=target, draft_auxiliary_distilled=draft)
    expected = drop_final_cached_token(original)
    loaded_target, loaded_draft = load_paired_logps(path, expected, drop_final=True)
    assert len(loaded_target) == len(expected.delta)
    assert np.allclose(loaded_target - loaded_draft, expected.delta)
