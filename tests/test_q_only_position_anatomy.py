import numpy as np

from experiments.sd_membership_sft.q_only_position_anatomy import (
    q_selected_positive_fraction_scores,
)


def test_q_only_selection_uses_low_and_high_draft_probability_positions() -> None:
    delta = np.asarray([1.0, -1.0, 2.0, -2.0, -1.0, -1.0, 3.0, 3.0])
    logq = np.asarray([-4.0, -3.0, -2.0, -1.0, -1.0, -2.0, -3.0, -4.0])
    scores = q_selected_positive_fraction_scores(
        delta, logq, np.asarray([4, 4]), fractions=(0.5,), random_repeats=2, seed=7
    )
    assert scores["lowq_positive_fraction_50pct"].tolist() == [0.5, 1.0]
    assert scores["highq_positive_fraction_50pct"].tolist() == [0.5, 0.0]
    assert np.all((scores["random_positive_fraction_50pct"] >= 0.0) & (scores["random_positive_fraction_50pct"] <= 1.0))
