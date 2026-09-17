import numpy as np
import torch

from experiments.sd_membership_sft.interpretable_scale_gate import (
    ScaleGate,
    fragment_q_summaries,
    scale_accept_scores,
)


def test_scale_scores_follow_lowq_prefixes() -> None:
    logq = np.asarray([-5.0, -4.0, -3.0, -2.0, -1.0])
    accepted = np.asarray([1.0, 1.0, 0.0, 0.0, 0.0])
    scores = scale_accept_scores(accepted, logq, np.asarray([5]), (0.2, 0.4, 1.0))
    np.testing.assert_allclose(scores[0], [1.0, 1.0, 0.4])


def test_q_summaries_are_finite_and_record_aligned() -> None:
    values = np.linspace(-8.0, -0.1, 20)
    result = fragment_q_summaries(values, np.asarray([8, 12]))
    assert result.shape[0] == 2
    assert result.shape[1] >= 10
    assert np.all(np.isfinite(result))


def test_scale_gate_returns_simplex_weights() -> None:
    model = ScaleGate(summary_dim=12, scales=7, accept_aware=False)
    evidence = torch.randn(5, 7)
    summary = torch.randn(5, 12)
    score, weights = model(summary, evidence)
    assert score.shape == (5,)
    assert weights.shape == (5, 7)
    assert torch.all(weights >= 0)
    torch.testing.assert_close(weights.sum(dim=1), torch.ones(5))
