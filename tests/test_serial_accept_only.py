from types import SimpleNamespace

import numpy as np
import pytest
import torch

from experiments.sd_membership_sft.serial_accept_only import (
    HazardGRU, collect_transcript, correction_distribution, hazard_features, hazard_evidence,
)


class ToyCache:
    def __init__(self):
        self.tokens = []

    def crop(self, length):
        self.tokens = self.tokens[:length]


class ToyLM(torch.nn.Module):
    def __init__(self, reverse=False):
        super().__init__()
        self.reverse = reverse
        self.last_cache = None

    def forward(self, input_ids, past_key_values=None, use_cache=True, logits_to_keep=0):
        cache = past_key_values if past_key_values is not None else ToyCache()
        rows = []
        for token in input_ids[0].tolist():
            cache.tokens.append(token)
            probabilities = torch.tensor([.1, .2, .7])
            if self.reverse:
                probabilities = probabilities.flip(0)
            rows.append(probabilities.roll(token % 3).log())
        self.last_cache = cache
        logits = torch.stack(rows)[None]
        if logits_to_keep:
            logits = logits[:, -logits_to_keep:]
        return SimpleNamespace(logits=logits, past_key_values=cache)


def test_residual_correction_recovers_target_marginal_exactly():
    p, q = torch.tensor([.2, .3, .5]), torch.tensor([.6, .3, .1])
    alpha = torch.minimum(torch.ones_like(q), p / q)
    corrected = correction_distribution(p, q)
    torch.testing.assert_close(q * alpha + (1 - (q * alpha).sum()) * corrected, p)
    with pytest.raises(ValueError, match="positive correction"):
        correction_distribution(p, p)


def test_hazard_alternatives_match_bernoulli_likelihoods_with_unsigned_bits():
    from scipy.special import expit

    logits = np.array([-.3, .7, 1.2])
    bits = np.array([1, 0, 1], dtype=np.uint8)
    null = expit(logits)
    base = np.prod(np.where(bits, null, 1 - null))
    ratios = []
    for sign in (1, -1):
        ratios.append(np.mean([np.prod(np.where(bits, expit(logits + sign * eta),
                                                expit(-logits - sign * eta))) / base
                               for eta in (.5, 1., 2.)]))
    np.testing.assert_allclose(hazard_evidence(logits, bits),
                               np.log([ratios[0], ratios[1], np.mean(ratios)]))
    flipped = hazard_evidence(-logits, 1 - bits)
    original = hazard_evidence(logits, bits)
    np.testing.assert_allclose(flipped, [original[1], original[0], original[2]])


def test_natural_sd_crops_both_caches_and_exports_only_reached_bits():
    target, draft = ToyLM(), ToyLM(reverse=True)
    trace = collect_transcript(target, draft, [0, 1], device="cpu", seed=77, rounds=10, gamma=4)
    assert target.last_cache.tokens == draft.last_cache.tokens
    assert len(target.last_cache.tokens) == 2 + trace["generated_tokens"]
    assert trace["generated_tokens"] == sum(trace["accepted_lengths"]) + trace["rounds"]
    assert trace["verified_candidate_positions"] >= trace["reached_decisions"]
    assert trace["features"].shape == (len(trace["bits"]), 4)
    assert 0 in trace["bits"]
    for round_id in np.unique(trace["round_ids"]):
        bits = trace["bits"][trace["round_ids"] == round_id]
        assert np.all(bits[:-1] == 1)
    assert not {"logp", "target", "correction_tokens", "token_ids"}.intersection(trace)


def test_identical_models_accept_every_proposal_and_handle_eos():
    trace = collect_transcript(ToyLM(), ToyLM(), [0], device="cpu", seed=5, rounds=3, gamma=4)
    assert np.all(trace["bits"] == 1)
    assert trace["generated_tokens"] == 15
    stopped = collect_transcript(ToyLM(), ToyLM(), [0], device="cpu", seed=5, rounds=20, gamma=4, eos_id=2)
    assert stopped["rounds"] < 20


def test_hazard_inputs_and_predictions_cannot_see_future_feedback():
    raw = np.zeros((8, 4), dtype=np.float32)
    bits = np.ones(8, dtype=np.uint8)
    features = hazard_features(raw, bits, np.array([4, 4]))
    assert features[0, -1] == features[4, -1] == 0
    altered = bits.copy()
    altered[2] = 0
    changed = hazard_features(raw, altered, np.array([4, 4]))
    np.testing.assert_array_equal(features[:3], changed[:3])
    model = HazardGRU().eval()
    x = torch.from_numpy(features[:4])[None]
    with torch.no_grad():
        prediction = model(x)
        x[:, 3] = 100
        updated = model(x)
    torch.testing.assert_close(prediction[:, :3], updated[:, :3])
