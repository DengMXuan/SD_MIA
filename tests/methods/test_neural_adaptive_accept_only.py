import numpy as np
import torch

from experiments.sd_membership_sft.archive.neural_adaptive_accept_only import EvidenceTCN, build_evidence_features, learned_pilot_schedule, pseudo_member_acceptance


def test_pseudo_member_acceptance_is_local_and_monotone() -> None:
    base = np.linspace(0.1, 0.9, 64)
    logq = np.linspace(-8.0, -0.1, 64)
    boosted, selected = pseudo_member_acceptance(base, logq, seed=11, family=1)
    assert boosted.shape == base.shape
    assert selected.shape == base.shape
    assert selected.dtype == bool
    assert 0 < np.sum(selected) < len(base)
    assert np.all(boosted >= base)
    assert np.all(boosted <= 1.0)
    np.testing.assert_allclose(boosted[~selected], base[~selected])
    assert np.all(boosted[selected] > base[selected])


def test_evidence_features_are_aligned_and_finite() -> None:
    static = np.arange(24, dtype=np.float64).reshape(4, 6)
    rate = np.asarray([0.0, 0.5, 1.0, 0.5])
    expected = np.asarray([0.2, 0.4, 0.8, 0.6])
    features = build_evidence_features(static, rate, expected, k=2)
    assert features.shape == (4, 12)
    assert np.all(np.isfinite(features))
    np.testing.assert_allclose(features[:, -6], rate)
    np.testing.assert_allclose(features[:, -4], expected)


def test_evidence_tcn_masks_padding_and_returns_token_values() -> None:
    model = EvidenceTCN(input_dim=12, channels=8, dropout=0.0)
    values = torch.randn(3, 9, 12)
    mask = torch.tensor(
        [
            [1, 1, 1, 1, 1, 1, 1, 1, 1],
            [1, 1, 1, 1, 1, 0, 0, 0, 0],
            [1, 1, 0, 0, 0, 0, 0, 0, 0],
        ],
        dtype=torch.bool,
    )
    fragment, token = model(values, mask)
    assert fragment.shape == (3,)
    assert token.shape == (3, 9)
    assert torch.all(torch.isfinite(fragment))
    assert torch.all(torch.isneginf(token[~mask]))


def test_learned_schedule_has_exact_budget_and_two_normal_pilots() -> None:
    priority = np.linspace(0.0, 1.0, 20)
    schedule = learned_pilot_schedule(priority, budget=8, selected_fraction=0.5)
    assert np.sum(schedule >= 0) == 8 * len(priority)
    assert np.all(schedule[:, :2] == 0)
    assert np.all(np.sum(schedule >= 0, axis=1) >= 6)
    assert np.all((schedule >= -1) & (schedule <= 4))
    high = np.argsort(-priority)[:10]
    low = np.argsort(priority)[:10]
    assert np.mean(np.sum(schedule[high] >= 0, axis=1)) > np.mean(
        np.sum(schedule[low] >= 0, axis=1)
    )
