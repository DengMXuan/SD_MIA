import numpy as np

from experiments.sd_membership_sft.full_ai_query_allocation import (
    full_one_shot_counts,
    hybrid_counts,
    sequential_round_counts,
    uniform_counts,
)


def test_allocation_counts_use_exact_post_pilot_budget() -> None:
    priority = np.linspace(-1.0, 1.0, 21)
    for counts in (
        uniform_counts(len(priority)),
        hybrid_counts(priority),
        full_one_shot_counts(priority),
    ):
        assert counts.shape == priority.shape
        assert np.sum(counts) == 6 * len(priority)
        assert np.all(counts >= 0)


def test_full_one_shot_leaves_unselected_tokens_at_pilot_only() -> None:
    priority = np.linspace(0.0, 1.0, 20)
    counts = full_one_shot_counts(priority)
    assert np.sum(counts == 0) == 10
    assert np.all(counts[counts > 0] == 12)


def test_sequential_round_spends_one_length_of_queries() -> None:
    priority = np.linspace(0.0, 1.0, 23)
    counts = sequential_round_counts(priority, selected_fraction=0.25)
    assert np.sum(counts) == len(priority)
    assert np.sum(counts > 0) == int(np.ceil(0.25 * len(priority)))
