import numpy as np
import pytest
import torch

from experiments.baseline.methods import (
    geometric_windows,
    icp_score,
    mean_log_likelihood,
    min_k_plus_plus_score,
    min_k_prob_score,
    petal_score,
    rank_auc,
    recall_score,
    rouge1_recall,
    sead_score,
    upper_tail_tpr,
)
from experiments.baseline.run import TargetScorer


def test_min_k_scores_are_member_positive():
    member_logp = np.asarray([-0.1, -0.2, -0.3, -0.4])
    nonmember_logp = np.asarray([-3.0, -3.5, -4.0, -5.0])
    member_min_k = min_k_prob_score(member_logp, 50)
    nonmember_min_k = min_k_prob_score(nonmember_logp, 50)
    member_min_k_pp = min_k_plus_plus_score(
        member_logp, np.zeros(4), np.ones(4), 50
    )
    nonmember_min_k_pp = min_k_plus_plus_score(
        nonmember_logp, np.zeros(4), np.ones(4), 50
    )

    assert member_min_k == pytest.approx(-0.35)
    assert nonmember_min_k == pytest.approx(-4.5)
    assert member_min_k_pp == pytest.approx(-0.35)
    assert nonmember_min_k_pp == pytest.approx(-4.5)
    member_min_k_scores = [member_min_k, -0.45]
    nonmember_min_k_scores = [nonmember_min_k, -4.75]
    member_min_k_pp_scores = [member_min_k_pp, -0.45]
    nonmember_min_k_pp_scores = [nonmember_min_k_pp, -4.75]
    assert rank_auc(member_min_k_scores, nonmember_min_k_scores) == pytest.approx(1.0)
    assert rank_auc(member_min_k_pp_scores, nonmember_min_k_pp_scores) == pytest.approx(1.0)


def test_probability_style_baselines_keep_member_positive_pair_order():
    member_logp = np.asarray([-0.15, -0.20, -0.30, -0.40])
    nonmember_logp = np.asarray([-3.0, -3.5, -4.0, -5.0])

    member_scores = {
        "loss": mean_log_likelihood(member_logp),
        "min_k_prob": min_k_prob_score(member_logp, 50),
        "min_k_pp": min_k_plus_plus_score(
            member_logp, np.zeros(4), np.ones(4), 50
        ),
        "petal": petal_score(
            member_logp, member_logp, slope=1.0, intercept=0.0
        )[0],
    }
    nonmember_scores = {
        "loss": mean_log_likelihood(nonmember_logp),
        "min_k_prob": min_k_prob_score(nonmember_logp, 50),
        "min_k_pp": min_k_plus_plus_score(
            nonmember_logp, np.zeros(4), np.ones(4), 50
        ),
        "petal": petal_score(
            nonmember_logp, nonmember_logp, slope=1.0, intercept=0.0
        )[0],
    }

    assert set(member_scores) == set(nonmember_scores)
    assert all(
        member_scores[name] > nonmember_scores[name] for name in member_scores
    )
    assert all(
        rank_auc([member_scores[name]], [nonmember_scores[name]]) == pytest.approx(1.0)
        for name in member_scores
    )


def test_relative_scores_follow_paper_orientation():
    assert recall_score(-4.0, -5.0) == pytest.approx(1.25)
    assert icp_score(-4.0, -5.0) == pytest.approx(1.0)


def test_target_only_petal_calibration_and_density():
    score, slope, intercept = petal_score([-1.0, -2.0], [-2.0, -4.0])
    assert slope == pytest.approx(2.0)
    assert intercept == pytest.approx(0.0)
    assert score == pytest.approx(-3.0)

    member_score = petal_score(
        [-0.1, -0.2], [-0.2, -0.4], slope=1.0, intercept=0.0
    )[0]
    nonmember_score = petal_score(
        [-3.0, -4.0], [-6.0, -8.0], slope=1.0, intercept=0.0
    )[0]
    assert member_score > nonmember_score
    assert rank_auc([member_score, -0.25], [nonmember_score, -3.75]) == pytest.approx(1.0)

    samples = np.asarray([[1, 1, 2], [4, 5, 5]])
    log_density, lexical = sead_score(samples, [1, 5])
    assert lexical == pytest.approx((2 / 3 + 2 / 3) / 2)
    assert log_density < 0.0


def test_surface_similarity_and_metrics():
    assert rouge1_recall("a b b", "a b c") == pytest.approx(2 / 3)
    assert geometric_windows(2, 40, 10) == (2, 3, 4, 6, 9, 13, 18, 25, 32, 40)
    assert rank_auc([0.9, 0.8], [0.2, 0.3]) == pytest.approx(1.0)
    tpr, fpr, threshold = upper_tail_tpr([0.9, 0.8], [0.1, 0.2], 0.5)
    assert tpr == pytest.approx(1.0)
    assert fpr == pytest.approx(0.0)
    assert threshold == pytest.approx(0.2)


def test_generation_uses_transformers_compatible_local_rng():
    class Tokenizer:
        pad_token_id = 0

    class Model:
        def generate(self, **kwargs):
            assert "generator" not in kwargs
            input_ids = kwargs["input_ids"]
            extra = torch.zeros(
                (input_ids.shape[0], kwargs["max_new_tokens"]), dtype=torch.long
            )
            return torch.cat((input_ids, extra), dim=1)

    scorer = TargetScorer(
        Model(), Tokenizer(), torch.device("cpu"), sead_samples=2,
        sead_temperature=1.0, seed=7
    )
    generated = scorer.generate([4, 5], max_new_tokens=3, seed=9, sample=True)
    assert generated == [[0, 0, 0]]
