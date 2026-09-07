from __future__ import annotations

import numpy as np

from experiments.sd_membership_sft.audit import make_audit_split


def test_make_audit_split_is_deterministic_and_disjoint() -> None:
    train_a, test_a = make_audit_split(20, 20, 5, seed=123)
    train_b, test_b = make_audit_split(20, 20, 5, seed=123)
    assert np.array_equal(train_a, train_b)
    assert np.array_equal(test_a, test_b)
    assert set(train_a).isdisjoint(set(test_a))
    assert len(train_a) == 10 and len(test_a) == 30
    # class balance: first per_class entries of each permutation go to train
    train_labels = [0] * 20 + [1] * 20
    ones = sum(1 for index in train_a if train_labels[index] == 1)
    assert ones == 5
