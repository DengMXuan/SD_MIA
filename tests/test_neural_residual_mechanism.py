import numpy as np

from experiments.sd_membership_sft.neural_residual_mechanism import (
    banded_tie_break_score,
    lexicographic_score,
)


def test_lexicographic_score_never_reverses_strict_base_order() -> None:
    base = np.asarray([0.0, 0.0, 0.2, 1.0])
    neural = np.asarray([10.0, -10.0, -100.0, -1000.0])
    score = lexicographic_score(base, neural)
    assert score[0] > score[1]
    assert score[2] > score[0]
    assert score[3] > score[2]


def test_banded_tie_break_only_uses_neural_within_band() -> None:
    base = np.asarray([0.01, 0.09, 0.31, 0.39])
    neural = np.asarray([0.0, 1.0, 1.0, 0.0])
    score = banded_tie_break_score(base, neural, width=0.1)
    assert score[1] > score[0]
    assert score[2] > score[3]
    assert min(score[2:]) > max(score[:2])
