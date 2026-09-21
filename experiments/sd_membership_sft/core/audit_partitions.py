"""Data contracts for current deployment evaluation and frozen legacy studies."""
from __future__ import annotations

import numpy as np

from experiments.sd_membership_sft.core.audit_runtime import SPLIT_SEED, _deterministic_subset, split_indices
from experiments.sd_membership_sft.core.data_contract import DEFAULT_DATA_CONTRACT, ControlledDataContract


TRAIN_NONMEMBERS = DEFAULT_DATA_CONTRACT.detector_train
VALIDATION_NONMEMBERS = DEFAULT_DATA_CONTRACT.detector_validation
CALIBRATION_NONMEMBERS = DEFAULT_DATA_CONTRACT.calibration
TEST_NONMEMBERS = DEFAULT_DATA_CONTRACT.nonmembers
TEST_MEMBERS = DEFAULT_DATA_CONTRACT.members


def _fit_and_calibration(
    labels: np.ndarray,
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray], np.ndarray]:
    """Reproduce the registered 320/80 fit and 200 calibration records exactly."""
    base = split_indices(labels, SPLIT_SEED)
    reference = _deterministic_subset(
        base["D"][labels[base["D"]] == 0],
        TRAIN_NONMEMBERS + VALIDATION_NONMEMBERS,
        SPLIT_SEED + 400,
    )
    calibration = _deterministic_subset(
        base["C"][labels[base["C"]] == 0], CALIBRATION_NONMEMBERS, SPLIT_SEED + 1200,
    )
    shuffled = np.random.default_rng(SPLIT_SEED).permutation(reference)
    fit = {
        "train": np.sort(shuffled[:TRAIN_NONMEMBERS]),
        "validation": np.sort(shuffled[TRAIN_NONMEMBERS:]),
        "reference": reference,
        "calibration": calibration,
    }
    return fit, base, np.unique(np.concatenate(tuple(base.values())))


def deployment_partitions(
    labels: np.ndarray,
    record_ids: np.ndarray,
    record_roles: np.ndarray,
    contract: ControlledDataContract = DEFAULT_DATA_CONTRACT,
) -> dict[str, np.ndarray]:
    """Partition an explicit 600+2000+2000 four-role observation archive."""
    labels = np.asarray(labels, dtype=np.int64)
    record_ids = np.asarray(record_ids)
    roles = np.asarray(record_roles).astype(str)
    if not (len(labels) == len(record_ids) == len(roles)):
        raise ValueError("labels, record_ids, and record_roles must align")
    if len(np.unique(record_ids)) != len(record_ids):
        raise ValueError("record IDs must be unique")

    auxiliary = np.flatnonzero(roles == "audit_auxiliary")
    test_members = np.flatnonzero(roles == "member")
    test_nonmembers = np.flatnonzero(roles == "nonmember")
    expected = (contract.audit_auxiliary, contract.members, contract.nonmembers)
    if (len(auxiliary), len(test_members), len(test_nonmembers)) != expected:
        raise ValueError(
            "deployment archive requires "
            f"{contract.audit_auxiliary} audit auxiliaries, "
            f"{contract.members} members, and {contract.nonmembers} nonmembers"
        )
    if np.any(labels[auxiliary] != 0) or np.any(labels[test_nonmembers] != 0):
        raise ValueError("auxiliary and nonmember roles must have label 0")
    if np.any(labels[test_members] != 1):
        raise ValueError("member roles must have label 1")

    shuffled = np.random.default_rng(SPLIT_SEED).permutation(auxiliary)
    train_end = contract.detector_train
    validation_end = train_end + contract.detector_validation
    train = np.sort(shuffled[:train_end])
    validation = np.sort(
        shuffled[train_end:validation_end]
    )
    calibration = np.sort(shuffled[validation_end:])
    reference = np.sort(np.concatenate((train, validation)))
    result = {
        "train": train,
        "validation": validation,
        "reference": reference,
        "calibration": calibration,
        "test": np.sort(np.concatenate((test_members, test_nonmembers))),
    }
    used = np.concatenate((train, validation, calibration, result["test"]))
    if len(np.unique(used)) != len(used) or len(used) != len(labels):
        raise ValueError("deployment roles must be exhaustive and disjoint")
    return result


def legacy_partitions(labels: np.ndarray, record_ids: np.ndarray) -> dict[str, np.ndarray]:
    """Original 320/80/200 fit/calibration and 400+400 test contract."""
    labels = np.asarray(labels, dtype=np.int64)
    fit, base, _ = _fit_and_calibration(labels)
    return {**fit, "test": base["T"]}
