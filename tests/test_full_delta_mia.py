import numpy as np
import torch

from experiments.sd_membership_sft.full_delta_mia import (
    DeltaTransformer,
    DeltaTCN,
    TokenPoolNet,
    _metric_point,
    collate_sequences,
    method_scores_path,
    partial_auc,
    split_indices,
    transform_sequences,
)


def test_full_delta_split_is_fixed_and_disjoint():
    labels = np.asarray([1] * 2000 + [0] * 2000)
    first = split_indices(labels, 20260824)
    second = split_indices(labels, 20260824)
    assert all(np.array_equal(first[name], second[name]) for name in first)
    assert len(first["D"]) == 1600
    assert all(len(first[name]) == 800 for name in ("V", "C", "T"))
    assert set(np.concatenate(list(first.values()))) == set(range(4000))
    assert sum(int(np.sum(labels[first[name]] == 1)) for name in first) == 2000
    assert sum(int(np.sum(labels[first[name]] == 0)) for name in first) == 2000


def test_sequence_transformations_preserve_values_and_lengths():
    delta = np.arange(9, dtype=np.float32)
    lengths = np.asarray([4, 5])
    normal, normal_lengths = transform_sequences(delta, lengths, "normal")
    reverse, reverse_lengths = transform_sequences(delta, lengths, "reverse")
    shuffled, shuffled_lengths = transform_sequences(delta, lengths, "shuffle", seed=20260909)
    assert normal_lengths.tolist() == [4, 5]
    assert reverse_lengths.tolist() == [4, 5]
    assert shuffled_lengths.tolist() == [4, 5]
    assert reverse[0].tolist() == [3.0, 2.0, 1.0, 0.0]
    assert sorted(shuffled[1].tolist()) == sorted(normal[1].tolist())


def test_collate_sequences_masks_padding():
    values, mask, labels = collate_sequences(
        [(torch.tensor([1.0, 2.0]), 0), (torch.tensor([3.0]), 1)]
    )
    assert values.shape == (2, 2, 1)
    assert mask.tolist() == [[True, True], [True, False]]
    assert labels.tolist() == [0.0, 1.0]


def test_sequence_models_accept_variable_lengths():
    values, mask, _labels = collate_sequences(
        [(torch.tensor([1.0, 2.0, 3.0]), 0), (torch.tensor([4.0]), 1)]
    )
    for model in (
        TokenPoolNet(8, 0.0, False, False),
        TokenPoolNet(8, 0.0, True, True),
        DeltaTCN(8, 3, 0.0, False),
        DeltaTransformer(8, 2, 0.0, True, max_length=16),
    ):
        output = model(values, mask)
        assert output.shape == (2,)
        assert torch.isfinite(output).all()


def test_partial_auc_has_expected_bounds_and_order():
    labels = np.asarray([1, 1, 0, 0])
    assert partial_auc(np.asarray([4.0, 3.0, 2.0, 1.0]), labels) == 1.0
    assert 0.0 <= partial_auc(np.asarray([1.0, 2.0, 3.0, 4.0]), labels) <= 1.0


def test_metric_point_uses_scores_not_record_indices_for_thresholds():
    labels = np.asarray([1, 1] + [0] * 202)
    scores = np.zeros(len(labels), dtype=np.float64)
    scores[:2] = [201.0, 202.0]
    scores[2:202] = np.arange(200, dtype=np.float64)
    scores[202:] = [0.0, 1.0]
    partitions = {
        "D": np.asarray([0, 1]),
        "V": np.asarray([0, 1, 202, 203]),
        "C": np.arange(2, 202),
        "T": np.asarray([0, 1, 202, 203]),
    }
    operating = _metric_point(scores, labels, partitions)["test"]["tpr_at_fpr"]["1%"]
    assert operating["tpr"] == 1.0
    assert operating["actual_fpr"] == 0.0


def test_method_score_paths_are_distinct():
    from pathlib import Path

    assert method_scores_path(Path("/tmp/run"), "delta-tcn").name == "scores_delta_tcn.npz"
