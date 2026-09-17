"""Fit and evaluate the preregistered M1 conditional-calibration models.

This module is intentionally self-contained after the model-facing cache has
been produced by :mod:`m1_extract`.  It never reads the calibration or test
labels while fitting a conditional model or selecting a detector.  The
``detector_fit`` partition is the only supervised training partition; V is
used for the preregistered model choice, C supplies conformal tail p-values,
and T is reported once at the end.

The implementation uses NumPy/SciPy for the convex linear fits and PyTorch for
the fixed one-hidden-layer confirmation models.  No sklearn dependency is
introduced.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import platform
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterable

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.optimize import minimize

from ..audit_metrics import (conformal_tail_pvalues)
from ..directional_mia import (rank_auc, split_indices, threshold_metrics)
from ..m1_features import (ACTIVATION_FEATURE_NAMES, AGGREGATE_FEATURE_NAMES, Q_FEATURE_NAMES, aggregate_matrix, document_mean_std_matrix)
from ..scoring_common import (ROOT, resolve_run_dir)


SPLIT_SEED = 20260824
NUISANCE_SEED = 20260909
DETECTOR_SEEDS = (20260909, 20260910, 20260911)
CONDITIONAL_LINEAR_L2 = (1e-3, 1e-2, 1e-1)
CONDITIONAL_MLP_WEIGHT_DECAYS = (1e-3, 1e-2)
DETECTOR_LOGISTIC_L2 = (1e-3, 1e-2, 1e-1, 1.0)
DETECTOR_MLP_WEIGHT_DECAYS = (1e-3, 1e-2)
CALIBRATION_RATES = (0.10, 0.05, 0.01)
EPSILON = 1e-8
FIT_VERSION = "m1-fit-v2-frozen-id-provenance-v-validation"


def _jsonable(value: Any) -> Any:
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_jsonable(item) for item in value]
    return value


def _softplus_np(value: np.ndarray) -> np.ndarray:
    return np.logaddexp(0.0, value)


def _sigmoid_np(value: np.ndarray) -> np.ndarray:
    output = np.empty_like(value, dtype=np.float64)
    positive = value >= 0
    output[positive] = 1.0 / (1.0 + np.exp(-value[positive]))
    exp_value = np.exp(value[~positive])
    output[~positive] = exp_value / (1.0 + exp_value)
    return output


def _record_offsets(lengths: np.ndarray) -> np.ndarray:
    lengths = np.asarray(lengths, dtype=np.int64)
    if lengths.ndim != 1 or np.any(lengths <= 0):
        raise ValueError("record lengths must be positive")
    return np.concatenate(([0], np.cumsum(lengths, dtype=np.int64)))


def _trim_appended_eos(values: np.ndarray, full_lengths: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    pieces: list[np.ndarray] = []
    offset = 0
    for length in full_lengths:
        count = int(length)
        if count <= 1:
            raise RuntimeError("cannot remove EOS from a record with no response token")
        pieces.append(values[offset : offset + count - 1])
        offset += count
    return np.concatenate(pieces), np.asarray(full_lengths - 1, dtype=np.int64)


@dataclass
class M1Data:
    feature_dir: Path
    probability_dir: Path
    role: str
    labels: np.ndarray
    record_ids: np.ndarray
    lengths: np.ndarray
    offsets: np.ndarray
    q: np.ndarray
    h: np.ndarray
    target_logp: np.ndarray
    draft_logq: np.ndarray
    delta: np.ndarray
    f19: np.ndarray
    f19_names: tuple[str, ...]
    eos_mask: np.ndarray
    feature_manifest: dict[str, Any]
    probability_manifest: dict[str, Any] | None


def _load_optional_json(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def _sha256_file(path: Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _record_ids_sha256(record_ids: np.ndarray) -> str:
    return hashlib.sha256(
        "\n".join(str(value) for value in record_ids).encode("utf-8")
    ).hexdigest()


def _validate_record_ids(record_ids: np.ndarray, expected_length: int | None = None) -> np.ndarray:
    values = np.asarray(record_ids)
    if values.ndim != 1:
        raise RuntimeError("record_ids must be one-dimensional")
    if expected_length is not None and len(values) != expected_length:
        raise RuntimeError("record_ids and labels disagree")
    normalized = [str(value) for value in values]
    if len(set(normalized)) != len(normalized):
        raise RuntimeError("record_ids must be unique")
    return values


def _validate_probability_provenance(
    feature_dir: Path,
    probability_dir: Path,
    feature_manifest: dict[str, Any],
    probability_manifest: dict[str, Any] | None,
    role: str,
) -> None:
    """Verify the cache source chain instead of trusting IDs and lengths alone."""

    if feature_manifest.get("training_regime") == "pretraining":
        from ...pretraining.cache import validate_cache_provenance
        validate_cache_provenance(feature_dir, probability_dir, feature_manifest, probability_manifest, role)
        return

    required_feature = (
        "benchmark",
        "epoch",
        "role",
        "run_dir",
        "records",
        "total_tokens",
        "record_ids_sha256",
    )
    missing = [key for key in required_feature if key not in feature_manifest]
    if missing:
        raise RuntimeError(f"feature manifest {feature_dir} lacks provenance fields: {missing}")
    if str(feature_manifest["role"]) != role:
        raise RuntimeError("feature manifest role disagrees with requested role")
    if probability_manifest is None:
        raise RuntimeError(f"probability cache has no provenance manifest: {probability_dir}")

    feature_probability = feature_manifest.get("probability_cache")
    if not isinstance(feature_probability, dict):
        raise RuntimeError("feature manifest lacks probability_cache provenance")
    feature_alignment = feature_manifest.get("probability_cache_alignment", {})
    token_path = probability_dir / "pq_gap_token_logps.npz"
    scores_path = probability_dir / "pq_gap_scores.npz"
    if not token_path.exists() or not scores_path.exists():
        raise RuntimeError(f"probability cache is incomplete under {probability_dir}")
    token_sha256 = _sha256_file(token_path)
    scores_sha256 = _sha256_file(scores_path)
    probability_kind = probability_manifest.get("kind")

    if probability_kind == "reconciled_q_cache":
        expected_reconciled = feature_alignment.get("reconciled_probability_dir")
        if not expected_reconciled:
            raise RuntimeError("reconciled probability cache is not recorded by the feature manifest")
        if Path(str(expected_reconciled)).resolve() != probability_dir.resolve():
            raise RuntimeError("probability cache is not the reconciled cache used by extraction")
        if str(probability_manifest.get("q_role")) != role:
            raise RuntimeError("reconciled probability cache role disagrees with feature cache")
        source_dir = Path(str(probability_manifest.get("source_probability_dir", ""))).resolve()
        expected_source = Path(str(feature_probability.get("path", ""))).resolve().parent
        if source_dir != expected_source:
            raise RuntimeError("reconciled cache source differs from feature manifest source")
        if str(probability_manifest.get("source_probability_sha256")) != str(feature_probability.get("sha256")):
            raise RuntimeError("reconciled cache source token file hash differs from feature manifest")
        if str(probability_manifest.get("source_scores_sha256")) != str(feature_probability.get("scores_sha256")):
            raise RuntimeError("reconciled cache source score file hash differs from feature manifest")
        source_token_path = source_dir / "pq_gap_token_logps.npz"
        source_scores_path = source_dir / "pq_gap_scores.npz"
        if not source_token_path.exists() or not source_scores_path.exists():
            raise RuntimeError("reconciled cache source files are missing")
        if _sha256_file(source_token_path) != str(feature_probability.get("sha256")):
            raise RuntimeError("source probability token file has changed since extraction")
        expected_scores_sha256 = feature_probability.get("scores_sha256")
        if expected_scores_sha256 is not None and _sha256_file(source_scores_path) != str(expected_scores_sha256):
            raise RuntimeError("source probability score file has changed since extraction")
        source_manifest = _load_optional_json(source_dir / "pq_gap_provenance.json")
        if source_manifest is None:
            raise RuntimeError("reconciled cache source has no provenance manifest")
        _validate_historical_probability_provenance(
            source_manifest,
            feature_manifest,
            role,
            source_dir,
            expected_tokens=(
                int(feature_manifest["total_tokens"])
                + (0 if bool(feature_manifest.get("eos_included", True)) else int(feature_manifest["records"]))
            ),
        )
        recorded_token_sha256 = probability_manifest.get("reconciled_token_logps_sha256")
        recorded_scores_sha256 = probability_manifest.get("reconciled_scores_sha256")
        if recorded_token_sha256 is None or str(recorded_token_sha256) != token_sha256:
            raise RuntimeError("reconciled token cache content hash does not match provenance")
        if recorded_scores_sha256 is None or str(recorded_scores_sha256) != scores_sha256:
            raise RuntimeError("reconciled score cache content hash does not match provenance")
        if "eos_included" in probability_manifest and bool(probability_manifest["eos_included"]) != bool(feature_manifest.get("eos_included", True)):
            raise RuntimeError("reconciled cache EOS contract disagrees with feature cache")
        return

    expected_token_path = Path(str(feature_probability.get("path", ""))).resolve()
    expected_scores_path = Path(str(feature_probability.get("scores_path", ""))).resolve()
    if expected_token_path != token_path.resolve() or expected_scores_path != scores_path.resolve():
        raise RuntimeError("probability cache path differs from the cache recorded by extraction")
    if token_sha256 != str(feature_probability.get("sha256")):
        raise RuntimeError("probability token cache content hash differs from extraction provenance")
    if scores_sha256 != str(feature_probability.get("scores_sha256")):
        raise RuntimeError("probability score cache content hash differs from extraction provenance")
    # Historical caches include appended EOS unless explicitly marked otherwise.
    # Validate the source count before load_m1_data trims the response arrays.
    source_includes_eos = probability_manifest.get("eos_included") is not False
    feature_includes_eos = bool(feature_manifest.get("eos_included", True))
    expected_tokens = int(feature_manifest["total_tokens"])
    if source_includes_eos and not feature_includes_eos:
        expected_tokens += int(feature_manifest["records"])
    _validate_historical_probability_provenance(
        probability_manifest, feature_manifest, role, probability_dir,
        expected_tokens=expected_tokens,
    )


def _validate_historical_probability_provenance(
    probability_manifest: dict[str, Any],
    feature_manifest: dict[str, Any],
    role: str,
    probability_dir: Path,
    expected_tokens: int | None = None,
) -> None:
    if str(probability_manifest.get("benchmark")) != str(feature_manifest["benchmark"]):
        raise RuntimeError("probability cache benchmark disagrees with feature cache")
    if int(probability_manifest.get("epoch", -1)) != int(feature_manifest["epoch"]):
        raise RuntimeError("probability cache epoch disagrees with feature cache")
    if int(probability_manifest.get("records", -1)) != int(feature_manifest["records"]):
        raise RuntimeError("probability cache record count disagrees with feature cache")
    if expected_tokens is None:
        expected_tokens = int(feature_manifest["total_tokens"])
    if int(probability_manifest.get("tokens", -1)) != expected_tokens:
        raise RuntimeError("probability cache token count disagrees with feature cache")
    if str(probability_manifest.get("run_dir", "")).rstrip("/") != str(feature_manifest["run_dir"]).rstrip("/"):
        raise RuntimeError("probability cache run differs from feature cache run")
    role_provenance = probability_manifest.get("roles", {}).get(role)
    if not isinstance(role_provenance, dict):
        raise RuntimeError(f"probability cache lacks provenance for role {role!r}: {probability_dir}")
    if int(role_provenance.get("epoch", -1)) != int(feature_manifest["epoch"]):
        raise RuntimeError("probability role epoch disagrees with feature cache")
    if str(role_provenance.get("run_dir", "")).rstrip("/") != str(feature_manifest["run_dir"]).rstrip("/"):
        raise RuntimeError("probability role run differs from feature cache run")
    if str(role_provenance.get("role")) != role:
        raise RuntimeError("probability role provenance has the wrong role")


def load_m1_data(feature_dir: Path, probability_dir: Path, role: str) -> M1Data:
    """Load and cross-check one extracted Q/H cache and its p/q archive."""

    feature_dir = feature_dir.resolve()
    probability_dir = probability_dir.resolve()
    feature_manifest_path = feature_dir / "feature_manifest.json"
    if not feature_manifest_path.exists():
        raise FileNotFoundError(feature_manifest_path)
    feature_manifest = json.loads(feature_manifest_path.read_text(encoding="utf-8"))
    if str(feature_manifest.get("role")) != role:
        raise RuntimeError(
            f"feature cache role {feature_manifest.get('role')!r} != requested {role!r}"
        )
    q = np.load(feature_dir / "q.npy", mmap_mode="r")
    h = np.load(feature_dir / "h.npy", mmap_mode="r")
    labels = np.asarray(np.load(feature_dir / "labels.npy"), dtype=np.int64)
    record_ids = _validate_record_ids(np.asarray(np.load(feature_dir / "record_ids.npy")), len(labels))
    lengths = np.asarray(np.load(feature_dir / "lengths.npy"), dtype=np.int64)
    offsets = np.asarray(np.load(feature_dir / "offsets.npy"), dtype=np.int64)
    eos_mask = np.asarray(np.load(feature_dir / "eos_mask.npy"), dtype=bool)
    if q.ndim != 2 or q.shape[1] != len(Q_FEATURE_NAMES):
        raise RuntimeError(f"Q feature array must be [tokens, 6], got {q.shape}")
    if h.ndim != 2 or h.shape[1] != len(ACTIVATION_FEATURE_NAMES):
        raise RuntimeError(f"H feature array must be [tokens, 40], got {h.shape}")
    if len(labels) != len(record_ids) or len(labels) != len(lengths):
        raise RuntimeError("feature cache record arrays disagree")
    if _record_ids_sha256(record_ids) != str(feature_manifest.get("record_ids_sha256")):
        raise RuntimeError("feature record_ids content differs from feature manifest provenance")
    expected_offsets = _record_offsets(lengths)
    if not np.array_equal(offsets, expected_offsets):
        raise RuntimeError("feature cache offsets do not match lengths")
    if len(eos_mask) != int(offsets[-1]) or len(q) != int(offsets[-1]) or len(h) != int(offsets[-1]):
        raise RuntimeError("feature cache token arrays disagree with offsets")

    scores_path = probability_dir / "pq_gap_scores.npz"
    token_path = probability_dir / "pq_gap_token_logps.npz"
    if not scores_path.exists() or not token_path.exists():
        raise FileNotFoundError(f"probability cache is incomplete under {probability_dir}")
    score_data = np.load(scores_path, allow_pickle=False)
    token_data = np.load(token_path, allow_pickle=False)
    cached_labels = np.asarray(score_data["labels"], dtype=np.int64)
    cached_ids = np.asarray(score_data["record_ids"])
    if not np.array_equal(cached_labels, labels) or not np.array_equal(cached_ids, record_ids):
        raise RuntimeError("feature and probability caches have different record identity/order")
    full_lengths = np.asarray(token_data["lengths"], dtype=np.int64)
    full_target = np.asarray(token_data["target"], dtype=np.float32)
    full_draft = np.asarray(token_data[role], dtype=np.float32)
    if len(full_lengths) != len(lengths) or int(full_lengths.sum()) != len(full_target):
        raise RuntimeError("target probability cache has invalid lengths")
    if len(full_draft) != len(full_target):
        raise RuntimeError("draft probability cache is not aligned with target")
    probability_manifest = _load_optional_json(probability_dir / "pq_gap_provenance.json")
    _validate_probability_provenance(
        feature_dir,
        probability_dir,
        feature_manifest,
        probability_manifest,
        role,
    )
    feature_eos_included = bool(feature_manifest.get("eos_included", True))
    probability_eos_included = (
        probability_manifest.get("eos_included")
        if probability_manifest is not None
        else None
    )
    if feature_eos_included:
        if not np.array_equal(full_lengths, lengths):
            raise RuntimeError("EOS-inclusive feature lengths disagree with probability cache")
        target_logp, draft_logq = full_target, full_draft
    elif probability_eos_included is False:
        # A reconciled no-EOS cache already contains trimmed p/q arrays.  Do
        # not remove a second token from every record.
        if not np.array_equal(full_lengths, lengths):
            raise RuntimeError("EOS-excluded feature lengths disagree with trimmed probability cache")
        target_logp, draft_logq = full_target, full_draft
    else:
        draft_logq, expected_lengths = _trim_appended_eos(full_draft, full_lengths)
        target_logp, target_lengths = _trim_appended_eos(full_target, full_lengths)
        if not np.array_equal(expected_lengths, lengths) or not np.array_equal(target_lengths, lengths):
            raise RuntimeError("EOS-excluded feature lengths disagree with probability cache")
    if not np.allclose(np.asarray(q[:, 0]), draft_logq, atol=1e-4, rtol=1e-5):
        difference = np.abs(np.asarray(q[:, 0], dtype=np.float64) - draft_logq)
        raise RuntimeError(
            "Q cache log_q does not match the probability cache: "
            f"max_abs_error={float(difference.max()):.6g}"
        )
    offsets = _record_offsets(lengths)
    f19_rows: list[dict[str, float]] = []
    from ..directional_mia import (record_features)

    for start, end in zip(offsets[:-1], offsets[1:]):
        f19_rows.append(record_features(target_logp[int(start) : int(end)], draft_logq[int(start) : int(end)]))
    f19_names = tuple(f19_rows[0]) if f19_rows else ()
    f19 = np.asarray([[row[name] for name in f19_names] for row in f19_rows], dtype=np.float64)
    return M1Data(
        feature_dir=feature_dir,
        probability_dir=probability_dir,
        role=role,
        labels=labels,
        record_ids=record_ids,
        lengths=lengths,
        offsets=offsets,
        q=q,
        h=h,
        target_logp=target_logp.astype(np.float64, copy=False),
        draft_logq=draft_logq.astype(np.float64, copy=False),
        delta=(target_logp - draft_logq).astype(np.float64, copy=False),
        f19=f19,
        f19_names=f19_names,
        eos_mask=eos_mask,
        feature_manifest=feature_manifest,
        probability_manifest=probability_manifest,
    )


@dataclass
class Partitions:
    indices: dict[str, np.ndarray]
    partition_by_index: np.ndarray

    def __getitem__(self, name: str) -> np.ndarray:
        return self.indices[name]


PARTITION_LEAVES = (
    "nuisance_location",
    "nuisance_scale",
    "detector_fit",
    "validation",
    "calibration",
    "test",
)


def validation_nonmember_indices(labels: np.ndarray, partitions: Partitions) -> np.ndarray:
    """Return only V records allowed for conditional-model selection."""

    labels = np.asarray(labels, dtype=np.int64)
    validation = partitions["validation"]
    selected = validation[labels[validation] == 0]
    if len(selected) == 0:
        raise RuntimeError("conditional model selection requires V nonmember records")
    return selected


def _partitions_from_frozen_manifest(
    labels: np.ndarray,
    record_ids: np.ndarray,
    manifest: dict[str, Any],
) -> Partitions:
    """Materialize a frozen ID mapping in the current array order."""

    record_ids = _validate_record_ids(record_ids, len(labels))
    current_index = {str(record_id): index for index, record_id in enumerate(record_ids)}
    entries = manifest.get("record_id_to_partition")
    if not isinstance(entries, list):
        raise RuntimeError("frozen partition manifest lacks record_id_to_partition")
    entry_by_id: dict[str, dict[str, Any]] = {}
    for entry in entries:
        if not isinstance(entry, dict) or "record_id" not in entry or "partition" not in entry:
            raise RuntimeError("invalid record_id_to_partition entry")
        record_id = str(entry["record_id"])
        if record_id in entry_by_id:
            raise RuntimeError(f"frozen partition manifest contains duplicate record_id {record_id!r}")
        entry_by_id[record_id] = entry
    current_ids = set(current_index)
    frozen_ids = set(entry_by_id)
    if current_ids != frozen_ids:
        missing = sorted(frozen_ids - current_ids)[:5]
        extra = sorted(current_ids - frozen_ids)[:5]
        raise RuntimeError(
            "current records do not match frozen partition IDs: "
            f"missing={missing}, extra={extra}"
        )
    for record_id, entry in entry_by_id.items():
        index = current_index[record_id]
        if "label" in entry and int(entry["label"]) != int(labels[index]):
            raise RuntimeError(f"record label changed for frozen record_id {record_id!r}")

    manifest_partitions = manifest.get("partitions")
    if not isinstance(manifest_partitions, dict):
        raise RuntimeError("frozen partition manifest lacks partitions")
    indices: dict[str, np.ndarray] = {}
    for name in ("nuisance_fit", *PARTITION_LEAVES):
        partition = manifest_partitions.get(name)
        if not isinstance(partition, dict) or not isinstance(partition.get("record_ids"), list):
            raise RuntimeError(f"frozen partition manifest lacks record IDs for {name}")
        values: list[int] = []
        seen: set[str] = set()
        for raw_id in partition["record_ids"]:
            record_id = str(raw_id)
            if record_id in seen:
                raise RuntimeError(f"frozen partition {name} contains duplicate record_id {record_id!r}")
            if record_id not in current_index:
                raise RuntimeError(f"frozen partition {name} references unknown record_id {record_id!r}")
            seen.add(record_id)
            values.append(current_index[record_id])
        indices[name] = np.asarray(values, dtype=np.int64)

    owner = np.full(len(labels), "", dtype="U32")
    for name in PARTITION_LEAVES:
        values = indices[name]
        if np.any(owner[values] != ""):
            raise RuntimeError(f"record appears in more than one frozen M1 partition: {name}")
        owner[values] = name
    if np.any(owner == ""):
        raise RuntimeError("frozen M1 partitions do not cover every record exactly once")
    if np.any(labels[indices["nuisance_fit"]] != 0):
        raise RuntimeError("frozen nuisance-fit records must be nonmembers")
    return Partitions(indices=indices, partition_by_index=owner)


def make_partitions(
    labels: np.ndarray,
    record_ids: np.ndarray | None = None,
    split_seed: int = SPLIT_SEED,
    nuisance_seed: int = NUISANCE_SEED,
    frozen_manifest: dict[str, Any] | None = None,
    frozen_manifest_path: Path | None = None,
) -> Partitions:
    """Create or materialize the explicit Nμ/Ns/D/V/C/T record partition.

    Production fits must provide a frozen manifest.  The seed-based branch is
    retained for creating the initial manifest and for small numerical tests.
    """

    labels = np.asarray(labels, dtype=np.int64)
    if labels.ndim != 1:
        raise RuntimeError("labels must be one-dimensional")
    if record_ids is not None:
        record_ids = _validate_record_ids(record_ids, len(labels))
    if frozen_manifest is not None and frozen_manifest_path is not None:
        raise ValueError("provide frozen_manifest or frozen_manifest_path, not both")
    if frozen_manifest_path is not None:
        frozen_manifest = _load_optional_json(Path(frozen_manifest_path).resolve())
        if frozen_manifest is None:
            raise FileNotFoundError(frozen_manifest_path)
    if frozen_manifest is not None:
        if record_ids is None:
            raise RuntimeError("frozen partition materialization requires record_ids")
        return _partitions_from_frozen_manifest(labels, record_ids, frozen_manifest)
    base = split_indices(labels, split_seed)
    train_nonmember = base["train"][labels[base["train"]] == 0]
    train_member = base["train"][labels[base["train"]] == 1]
    if len(train_nonmember) != 800 or len(train_member) != 800:
        raise RuntimeError(
            "M1 requires 800 member and 800 nonmember records in the historical train partition"
        )
    rng = np.random.default_rng(nuisance_seed)
    nonmember_order = rng.permutation(train_nonmember)
    nuisance_fit = nonmember_order[:400]
    detector_nonmember = nonmember_order[400:]
    nuisance_order = rng.permutation(nuisance_fit)
    nuisance_location = nuisance_order[:300]
    nuisance_scale = nuisance_order[300:]
    indices = {
        "nuisance_fit": np.sort(nuisance_fit),
        "nuisance_location": np.sort(nuisance_location),
        "nuisance_scale": np.sort(nuisance_scale),
        "detector_fit": np.sort(np.concatenate((train_member, detector_nonmember))),
        "validation": np.asarray(base["validation"], dtype=np.int64),
        "calibration": np.asarray(base["calibration"], dtype=np.int64),
        "test": np.asarray(base["test"], dtype=np.int64),
    }
    owner = np.full(len(labels), "", dtype="U32")
    # ``nuisance_fit`` is the N0 superset and intentionally overlaps its two
    # fitting subsets.  The record-level owner map uses the disjoint leaves
    # Nμ/Ns/D/V/C/T instead.
    for name, values in indices.items():
        if name == "nuisance_fit":
            continue
        if np.any(owner[values] != ""):
            raise RuntimeError(f"record appears in more than one M1 partition: {name}")
        owner[values] = name
    if np.any(owner == ""):
        raise RuntimeError("M1 partitions do not cover every record exactly once")
    if np.any(labels[indices["nuisance_fit"]] != 0):
        raise RuntimeError("nuisance-fit records must be nonmembers")
    return Partitions(indices=indices, partition_by_index=owner)


def partition_manifest(
    partitions: Partitions,
    labels: np.ndarray,
    record_ids: np.ndarray,
    split_seed: int = SPLIT_SEED,
    nuisance_seed: int = NUISANCE_SEED,
) -> dict[str, Any]:
    return {
        "split_seed": split_seed,
        "nuisance_seed": nuisance_seed,
        "split_indices_compatibility": "frozen record IDs; counts recorded below",
        "nuisance_rule": "disjoint nonmember Nmu/Ns subsets of nuisance_fit; exact IDs/counts recorded below",
        "partitions": {
            name: {
                "indices": values.tolist(),
                "record_ids": [str(record_ids[index]) for index in values],
                "n": int(len(values)),
                "members": int(np.sum(labels[values] == 1)),
                "nonmembers": int(np.sum(labels[values] == 0)),
            }
            for name, values in partitions.indices.items()
        },
        "record_id_to_partition": [
            {
                "index": int(index),
                "record_id": str(record_ids[index]),
                "label": int(labels[index]),
                "partition": str(partitions.partition_by_index[index]),
            }
            for index in range(len(labels))
        ],
    }


def _resolve_frozen_partition_path(
    feature_dir: Path,
    output_dir: Path,
    requested: Path | None,
) -> Path:
    if requested is not None:
        path = requested.resolve()
        if not path.exists():
            raise FileNotFoundError(path)
        return path
    candidates = (
        feature_dir / "partition_manifest.json",
        output_dir / "partition_manifest.json",
    )
    for path in candidates:
        if path.exists():
            return path.resolve()
    raise RuntimeError(
        "M1 requires a frozen partition manifest; pass --partition-manifest "
        "or create feature_dir/partition_manifest.json from the canonical record order"
    )


def sample_token_indices(
    lengths: np.ndarray,
    record_indices: np.ndarray,
    seed: int,
    max_tokens: int = 256,
) -> tuple[np.ndarray, np.ndarray]:
    """Sample consecutive-token rows while giving each document equal weight."""

    if max_tokens <= 0:
        raise ValueError("max_tokens must be positive")
    offsets = _record_offsets(lengths)
    rng = np.random.default_rng(seed)
    flat: list[np.ndarray] = []
    docs: list[np.ndarray] = []
    for record_index in np.asarray(record_indices, dtype=np.int64):
        start, end = int(offsets[record_index]), int(offsets[record_index + 1])
        count = end - start
        take = min(count, max_tokens)
        local = np.arange(count, dtype=np.int64) if count <= max_tokens else np.sort(rng.choice(count, take, replace=False))
        flat.append(start + local)
        docs.append(np.full(len(local), int(record_index), dtype=np.int64))
    if not flat:
        return np.empty(0, dtype=np.int64), np.empty(0, dtype=np.int64)
    return np.concatenate(flat), np.concatenate(docs)


@dataclass
class Standardizer:
    mean: np.ndarray
    scale: np.ndarray
    zero_variance: np.ndarray

    def transform(self, values: np.ndarray) -> np.ndarray:
        array = np.asarray(values, dtype=np.float32)
        result = (array - self.mean.astype(np.float32)) / self.scale.astype(np.float32)
        if np.any(self.zero_variance):
            result[:, self.zero_variance] = 0.0
        return result.astype(np.float32, copy=False)


def fit_token_standardizer(
    values: np.ndarray,
    lengths: np.ndarray,
    record_indices: np.ndarray,
    seed: int,
    max_tokens: int = 256,
) -> Standardizer:
    sampled, docs = sample_token_indices(lengths, record_indices, seed, max_tokens)
    if len(sampled) == 0:
        raise ValueError("cannot standardize an empty nuisance partition")
    selected = np.asarray(values[sampled], dtype=np.float64)
    unique_docs = np.asarray(record_indices, dtype=np.int64)
    doc_means = np.empty((len(unique_docs), selected.shape[1]), dtype=np.float64)
    doc_second = np.empty_like(doc_means)
    for row, record_index in enumerate(unique_docs):
        own = docs == record_index
        doc_values = selected[own]
        doc_means[row] = doc_values.mean(axis=0)
        doc_second[row] = np.square(doc_values).mean(axis=0)
    mean = doc_means.mean(axis=0)
    variance = np.maximum(doc_second.mean(axis=0) - np.square(mean), 0.0)
    scale = np.sqrt(variance)
    zero = ~np.isfinite(scale) | (scale < EPSILON)
    scale[zero] = 1.0
    mean[~np.isfinite(mean)] = 0.0
    return Standardizer(mean=mean, scale=scale, zero_variance=zero)


def fit_row_standardizer(values: np.ndarray) -> Standardizer:
    array = np.asarray(values, dtype=np.float64)
    if array.ndim != 2 or len(array) == 0:
        raise ValueError("row standardizer requires a non-empty [records, dimensions] matrix")
    mean = array.mean(axis=0)
    scale = array.std(axis=0, ddof=0)
    zero = ~np.isfinite(scale) | (scale < EPSILON)
    scale[zero] = 1.0
    mean[~np.isfinite(mean)] = 0.0
    return Standardizer(mean=mean, scale=scale, zero_variance=zero)


def _document_means(values: np.ndarray, lengths: np.ndarray, record_indices: np.ndarray) -> np.ndarray:
    offsets = _record_offsets(lengths)
    return np.asarray(
        [
            np.asarray(values[int(offsets[index]) : int(offsets[index + 1])], dtype=np.float64).mean()
            for index in np.asarray(record_indices, dtype=np.int64)
        ],
        dtype=np.float64,
    )


def _document_metric(
    values: np.ndarray,
    lengths: np.ndarray,
    record_indices: np.ndarray,
    metric: Callable[[np.ndarray], float],
) -> float:
    offsets = _record_offsets(lengths)
    rows = [
        metric(np.asarray(values[int(offsets[index]) : int(offsets[index + 1])], dtype=np.float64))
        for index in np.asarray(record_indices, dtype=np.int64)
    ]
    return float(np.mean(rows))


class Regressor:
    feature_dim: int
    positive: bool

    def predict(self, values: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    def parameter_count(self) -> int:
        raise NotImplementedError


@dataclass
class LinearRegressor(Regressor):
    weight: np.ndarray
    bias: float
    positive: bool = False

    @property
    def feature_dim(self) -> int:
        return int(len(self.weight))

    def predict(self, values: np.ndarray) -> np.ndarray:
        raw = np.asarray(values, dtype=np.float64) @ self.weight + float(self.bias)
        return _softplus_np(raw) + 1e-6 if self.positive else raw

    def parameter_count(self) -> int:
        return len(self.weight) + 1


class _TorchMLP(nn.Module):
    def __init__(self, feature_dim: int) -> None:
        super().__init__()
        self.network = nn.Sequential(nn.Linear(feature_dim, 64), nn.GELU(), nn.Linear(64, 1))

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        return self.network(values).squeeze(-1)


@dataclass
class MLPRegressor(Regressor):
    module: _TorchMLP
    positive: bool = False
    device: torch.device = torch.device("cpu")

    @property
    def feature_dim(self) -> int:
        return int(self.module.network[0].in_features)

    def predict(self, values: np.ndarray) -> np.ndarray:
        array = np.asarray(values, dtype=np.float32)
        outputs: list[np.ndarray] = []
        self.module.eval()
        with torch.inference_mode():
            for start in range(0, len(array), 65_536):
                batch = torch.from_numpy(array[start : start + 65_536]).to(self.device)
                raw = self.module(batch).cpu().numpy()
                outputs.append(raw)
        result = np.concatenate(outputs).astype(np.float64) if outputs else np.empty(0, dtype=np.float64)
        return _softplus_np(result) + 1e-6 if self.positive else result

    def parameter_count(self) -> int:
        return sum(int(parameter.numel()) for parameter in self.module.parameters())


def _doc_weights(doc_ids: np.ndarray) -> tuple[np.ndarray, int]:
    unique, counts = np.unique(doc_ids, return_counts=True)
    lookup = {int(doc): int(count) for doc, count in zip(unique, counts)}
    weights = np.asarray([1.0 / lookup[int(doc)] for doc in doc_ids], dtype=np.float64)
    return weights, len(unique)


def _fit_linear_regressor(
    values: np.ndarray,
    targets: np.ndarray,
    doc_ids: np.ndarray,
    l2: float,
    positive: bool,
) -> tuple[LinearRegressor, dict[str, Any]]:
    x = np.asarray(values, dtype=np.float64)
    y = np.asarray(targets, dtype=np.float64)
    weights, n_docs = _doc_weights(doc_ids)
    dimension = x.shape[1]
    initial = np.zeros(dimension + 1, dtype=np.float64)
    if positive:
        mean_target = max(float(np.mean(y)), 1e-6)
        initial[-1] = float(np.log(np.expm1(mean_target))) if mean_target < 30 else mean_target

    def objective(parameters: np.ndarray) -> tuple[float, np.ndarray]:
        weight = parameters[:-1]
        bias = parameters[-1]
        raw = x @ weight + bias
        if positive:
            prediction = _softplus_np(raw) + 1e-6
            residual = prediction - y
            data_loss = np.sum(weights * np.square(residual)) / n_docs
            derivative = 2.0 * residual * _sigmoid_np(raw)
        else:
            residual = raw - y
            absolute = np.abs(residual)
            huber = np.where(absolute <= 1.0, 0.5 * np.square(residual), absolute - 0.5)
            data_loss = np.sum(weights * huber) / n_docs
            derivative = np.where(absolute <= 1.0, residual, np.sign(residual))
        gradient_weight = (x.T @ (weights * derivative)) / n_docs
        gradient_bias = float(np.sum(weights * derivative) / n_docs)
        value = float(data_loss + l2 * np.dot(weight, weight))
        gradient = np.concatenate((gradient_weight + 2.0 * l2 * weight, [gradient_bias]))
        return value, gradient

    result = minimize(
        lambda parameters: objective(parameters),
        initial,
        jac=True,
        method="L-BFGS-B",
        options={"maxiter": 500, "ftol": 1e-12, "gtol": 1e-8, "maxls": 40},
    )
    model = LinearRegressor(result.x[:-1].copy(), float(result.x[-1]), positive=positive)
    return model, {
        "optimizer": "scipy.optimize.minimize:L-BFGS-B",
        "success": bool(result.success),
        "message": str(result.message),
        "iterations": int(result.nit),
        "objective": float(result.fun),
    }


def _fit_mlp_regressor(
    values: np.ndarray,
    targets: np.ndarray,
    doc_ids: np.ndarray,
    validation_values: np.ndarray,
    validation_targets: np.ndarray,
    validation_doc_ids: np.ndarray | None,
    weight_decay: float,
    seed: int,
    positive: bool,
    max_epochs: int = 100,
    patience: int = 10,
    device: torch.device = torch.device("cpu"),
) -> tuple[MLPRegressor, dict[str, Any]]:
    torch.manual_seed(seed)
    x = torch.from_numpy(np.asarray(values, dtype=np.float32)).to(device)
    y = torch.from_numpy(np.asarray(targets, dtype=np.float32)).to(device)
    weights, n_docs = _doc_weights(doc_ids)
    token_weights = torch.from_numpy(weights.astype(np.float32)).to(device)
    model = _TorchMLP(x.shape[1]).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=weight_decay)
    validation_x = torch.from_numpy(np.asarray(validation_values, dtype=np.float32)).to(device)
    validation_y = torch.from_numpy(np.asarray(validation_targets, dtype=np.float32)).to(device)
    if validation_doc_ids is None:
        validation_weights = None
        validation_doc_count = 1
    else:
        validation_weight_values, validation_doc_count = _doc_weights(validation_doc_ids)
        validation_weights = torch.from_numpy(validation_weight_values.astype(np.float32)).to(device)
    best_state: dict[str, torch.Tensor] | None = None
    best_loss = float("inf")
    stale = 0
    epochs_run = 0
    for epoch in range(max_epochs):
        model.train()
        raw = model(x)
        prediction = F.softplus(raw) + 1e-6 if positive else raw
        if positive:
            loss_rows = torch.square(prediction - y)
        else:
            loss_rows = F.huber_loss(prediction, y, delta=1.0, reduction="none")
        loss = (loss_rows * token_weights).sum() / float(n_docs)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        model.eval()
        with torch.inference_mode():
            val_raw = model(validation_x)
            val_prediction = F.softplus(val_raw) + 1e-6 if positive else val_raw
            if positive:
                val_rows = torch.square(val_prediction - validation_y)
            else:
                val_rows = F.huber_loss(val_prediction, validation_y, delta=1.0, reduction="none")
            if validation_weights is None:
                val_loss = val_rows.mean()
            else:
                val_loss = (val_rows * validation_weights).sum() / float(validation_doc_count)
        value = float(val_loss.cpu())
        epochs_run = epoch + 1
        if value < best_loss - 1e-10:
            best_loss = value
            best_state = {key: tensor.detach().cpu().clone() for key, tensor in model.state_dict().items()}
            stale = 0
        else:
            stale += 1
            if stale >= patience:
                break
    if best_state is not None:
        model.load_state_dict(best_state)
    return MLPRegressor(model, positive=positive, device=device), {
        "optimizer": "torch.optim.AdamW",
        "learning_rate": 1e-3,
        "weight_decay": weight_decay,
        "max_epochs": max_epochs,
        "epochs_run": epochs_run,
        "patience": patience,
        "best_validation_loss": best_loss,
        "seed": seed,
    }


def _fit_regressor(
    family: str,
    values: np.ndarray,
    targets: np.ndarray,
    doc_ids: np.ndarray,
    l2: float,
    positive: bool,
    validation_values: np.ndarray | None = None,
    validation_targets: np.ndarray | None = None,
    validation_doc_ids: np.ndarray | None = None,
    seed: int = 0,
    device: torch.device = torch.device("cpu"),
) -> tuple[Regressor, dict[str, Any]]:
    if family == "linear":
        return _fit_linear_regressor(values, targets, doc_ids, l2, positive)
    if family == "mlp":
        if validation_values is None or validation_targets is None:
            raise ValueError("MLP fitting requires validation values and targets")
        return _fit_mlp_regressor(
            values,
            targets,
            doc_ids,
            validation_values,
            validation_targets,
            validation_doc_ids,
            l2,
            seed,
            positive,
            device=device,
        )
    raise ValueError(f"unknown conditional family {family!r}")


@dataclass
class ConditionalCalibration:
    name: str
    family: str
    scaler: Standardizer
    target_center: float
    target_scale: float
    location: Regressor
    scale: Regressor
    mean_abs_residual_ns: float
    location_candidates: list[dict[str, Any]]
    scale_candidates: list[dict[str, Any]]
    location_l2: float
    scale_l2: float

    def transform(self, values: np.ndarray) -> np.ndarray:
        return self.scaler.transform(values)

    def predict_mu(self, transformed_values: np.ndarray) -> np.ndarray:
        normalized = self.location.predict(transformed_values)
        return normalized * self.target_scale + self.target_center

    def predict_scale(self, transformed_values: np.ndarray) -> np.ndarray:
        return np.maximum(self.scale.predict(transformed_values), 1e-6)

    def residual(
        self,
        transformed_values: np.ndarray,
        delta: np.ndarray,
        floor_fraction: float,
    ) -> tuple[np.ndarray, dict[str, float]]:
        mu = self.predict_mu(transformed_values)
        scale = self.predict_scale(transformed_values)
        floor = max(1e-3, floor_fraction * self.mean_abs_residual_ns)
        denominator = np.maximum(scale, floor)
        residual = (np.asarray(delta, dtype=np.float64) - mu) / denominator
        return residual, {
            "floor_fraction": floor_fraction,
            "s_min": float(floor),
            "scale_floor_fraction": float(np.mean(scale < floor)),
            "residual_q10": float(np.quantile(residual, 0.10)),
            "residual_q50": float(np.quantile(residual, 0.50)),
            "residual_q90": float(np.quantile(residual, 0.90)),
        }


def _conditional_candidates(family: str) -> tuple[float, ...]:
    return CONDITIONAL_LINEAR_L2 if family == "linear" else CONDITIONAL_MLP_WEIGHT_DECAYS


def fit_conditional_calibration(
    name: str,
    values: np.ndarray,
    data: M1Data,
    partitions: Partitions,
    family: str,
    seed: int,
    device: torch.device = torch.device("cpu"),
) -> ConditionalCalibration:
    """Fit μ on Nμ and positive scale on Ns, selecting only with V."""

    nmu = partitions["nuisance_location"]
    ns = partitions["nuisance_scale"]
    validation_nonmember = validation_nonmember_indices(data.labels, partitions)
    scaler = fit_token_standardizer(values, data.lengths, nmu, seed=seed)
    transformed = scaler.transform(values)
    target_means = _document_means(data.delta, data.lengths, nmu)
    target_center = float(target_means.mean())
    target_scale = float(target_means.std(ddof=0))
    if not np.isfinite(target_scale) or target_scale < EPSILON:
        target_scale = 1.0
    sampled_location, location_docs = sample_token_indices(data.lengths, nmu, seed + 11)
    location_x = transformed[sampled_location]
    location_y = ((data.delta[sampled_location] - target_center) / target_scale).astype(np.float64)
    validation_offsets = data.offsets
    validation_tokens = np.concatenate(
        [np.arange(int(validation_offsets[index]), int(validation_offsets[index + 1])) for index in validation_nonmember]
    )
    validation_x = transformed[validation_tokens]
    validation_y = ((data.delta[validation_tokens] - target_center) / target_scale).astype(np.float64)
    validation_doc_ids = np.concatenate(
        [np.full(int(data.lengths[index]), int(index), dtype=np.int64) for index in validation_nonmember]
    )
    location_candidates: list[dict[str, Any]] = []
    fitted_locations: list[tuple[float, Regressor, dict[str, Any]]] = []
    for candidate_index, regularization in enumerate(_conditional_candidates(family)):
        model, optimizer_info = _fit_regressor(
            family,
            location_x,
            location_y,
            location_docs,
            regularization,
            positive=False,
            validation_values=validation_x,
            validation_targets=validation_y,
            validation_doc_ids=validation_doc_ids,
            seed=seed + candidate_index,
            device=device,
        )
        predicted = model.predict(validation_x)
        # Candidate selection is explicitly document-weighted.  The flat
        # validation token arrays above are in record order, so use a full
        # record-shaped prediction for the metric rather than weighting long
        # documents more heavily.
        validation_prediction_all = model.predict(transformed)
        validation_residual_all = np.empty_like(data.delta)
        validation_residual_all[:] = 0.0
        validation_residual_all[validation_tokens] = (
            validation_prediction_all[validation_tokens]
            - ((data.delta[validation_tokens] - target_center) / target_scale)
        )
        validation_huber_all = np.where(
            np.abs(validation_residual_all) <= 1.0,
            0.5 * np.square(validation_residual_all),
            np.abs(validation_residual_all) - 0.5,
        )
        loss = _document_metric(
            validation_huber_all,
            data.lengths,
            validation_nonmember,
            lambda row: float(np.mean(row)),
        )
        candidate = {
            "regularization": regularization,
            "validation_huber_token_mean": loss,
            "parameter_count": model.parameter_count(),
            "optimizer": optimizer_info,
        }
        location_candidates.append(candidate)
        fitted_locations.append((regularization, model, optimizer_info))
    best_location_index = min(
        range(len(location_candidates)),
        key=lambda index: (location_candidates[index]["validation_huber_token_mean"], -float(location_candidates[index]["regularization"])),
    )
    location_l2, location, _ = fitted_locations[best_location_index]

    sampled_scale, scale_docs = sample_token_indices(data.lengths, ns, seed + 101)
    scale_x = transformed[sampled_scale]
    location_ns = location.predict(transformed[sampled_scale]) * target_scale + target_center
    scale_y = np.abs(data.delta[sampled_scale] - location_ns)
    validation_mu = location.predict(validation_x) * target_scale + target_center
    validation_scale_tokens = validation_tokens
    validation_scale_x = transformed[validation_scale_tokens]
    validation_scale_y = np.abs(data.delta[validation_scale_tokens] - validation_mu)
    scale_candidates: list[dict[str, Any]] = []
    fitted_scales: list[tuple[float, Regressor, dict[str, Any]]] = []
    for candidate_index, regularization in enumerate(_conditional_candidates(family)):
        model, optimizer_info = _fit_regressor(
            family,
            scale_x,
            scale_y,
            scale_docs,
            regularization,
            positive=True,
            validation_values=validation_scale_x,
            validation_targets=validation_scale_y,
            validation_doc_ids=validation_doc_ids,
            seed=seed + 201 + candidate_index,
            device=device,
        )
        predicted = model.predict(validation_scale_x)
        validation_scale_all = model.predict(transformed)
        validation_scale_error_all = np.zeros_like(data.delta)
        validation_scale_error_all[validation_tokens] = (
            validation_scale_all[validation_tokens]
            - np.abs(data.delta[validation_tokens] - validation_mu)
        )
        loss = _document_metric(
            np.square(validation_scale_error_all),
            data.lengths,
            validation_nonmember,
            lambda row: float(np.mean(row)),
        )
        candidate = {
            "regularization": regularization,
            "validation_scale_mse_token_mean": loss,
            "parameter_count": model.parameter_count(),
            "optimizer": optimizer_info,
        }
        scale_candidates.append(candidate)
        fitted_scales.append((regularization, model, optimizer_info))
    best_scale_index = min(
        range(len(scale_candidates)),
        key=lambda index: (scale_candidates[index]["validation_scale_mse_token_mean"], -float(scale_candidates[index]["regularization"])),
    )
    scale_l2, scale, _ = fitted_scales[best_scale_index]
    nmu_transformed = transformed
    ns_mu = location.predict(nmu_transformed) * target_scale + target_center
    ns_abs_residual = np.abs(data.delta - ns_mu)
    mean_abs_residual_ns = _document_metric(
        ns_abs_residual,
        data.lengths,
        ns,
        lambda row: float(np.mean(row)),
    )
    return ConditionalCalibration(
        name=name,
        family=family,
        scaler=scaler,
        target_center=target_center,
        target_scale=target_scale,
        location=location,
        scale=scale,
        mean_abs_residual_ns=mean_abs_residual_ns,
        location_candidates=location_candidates,
        scale_candidates=scale_candidates,
        location_l2=float(location_l2),
        scale_l2=float(scale_l2),
    )


def conditional_dev_metrics(
    calibration: ConditionalCalibration,
    values: np.ndarray,
    data: M1Data,
    partitions: Partitions,
) -> dict[str, Any]:
    transformed = calibration.transform(values)
    validation_nonmember = validation_nonmember_indices(data.labels, partitions)
    mu_all = calibration.predict_mu(transformed)
    scale_all = calibration.predict_scale(transformed)
    location_residual_all = data.delta - mu_all
    scale_abs_error_all = np.abs(np.abs(location_residual_all) - scale_all)
    return {
        "validation_nonmember_records": int(len(validation_nonmember)),
        "location_mae": _document_metric(
            location_residual_all,
            data.lengths,
            validation_nonmember,
            lambda row: float(np.mean(np.abs(row))),
        ),
        "location_bias": _document_metric(
            location_residual_all,
            data.lengths,
            validation_nonmember,
            lambda row: float(np.mean(row)),
        ),
        "scale_mae": _document_metric(
            scale_abs_error_all,
            data.lengths,
            validation_nonmember,
            lambda row: float(np.mean(row)),
        ),
        "validation_residual_abs_q50": float(
            np.quantile(
                np.concatenate(
                    [
                        np.abs(location_residual_all[int(data.offsets[index]) : int(data.offsets[index + 1])])
                        for index in validation_nonmember
                    ]
                ),
                0.50,
            )
        ),
        "validation_residual_abs_q90": float(
            np.quantile(
                np.concatenate(
                    [
                        np.abs(location_residual_all[int(data.offsets[index]) : int(data.offsets[index + 1])])
                        for index in validation_nonmember
                    ]
                ),
                0.90,
            )
        ),
    }


def _token_groups_by_record(lengths: np.ndarray) -> np.ndarray:
    return np.repeat(np.arange(len(lengths), dtype=np.int64), lengths)


def _activation_noise_or_permutation(data: M1Data, mode: str, seed: int, partitions: Partitions) -> np.ndarray:
    if mode == "real":
        return np.asarray(data.h)
    if mode == "noise":
        return np.random.default_rng(seed).normal(0.0, 1.0, size=data.h.shape).astype(np.float32)
    if mode != "within_bucket_permute":
        raise ValueError(f"unknown activation mode {mode!r}")
    h = np.asarray(data.h).copy()
    q_values = np.asarray(data.q[:, 0], dtype=np.float64)
    nmu = partitions["nuisance_location"]
    nmu_tokens = np.concatenate(
        [np.arange(int(data.offsets[index]), int(data.offsets[index + 1])) for index in nmu]
    )
    q_edges = np.quantile(q_values[nmu_tokens], (1.0 / 3.0, 2.0 / 3.0))
    length_edges = np.quantile(data.lengths[nmu], 0.50)
    q_bin = np.digitize(q_values, q_edges, right=False)
    length_bin = np.digitize(data.lengths, [length_edges], right=False)
    record_bucket = length_bin[_token_groups_by_record(data.lengths)]
    eos_bucket = data.eos_mask.astype(np.int64)
    token_bucket = q_bin + 3 * record_bucket + 6 * eos_bucket
    record_owner = _token_groups_by_record(data.lengths)
    rng = np.random.default_rng(seed)
    owner_names = partitions.partition_by_index
    for partition_name in (
        "nuisance_location",
        "nuisance_scale",
        "detector_fit",
        "validation",
        "calibration",
        "test",
    ):
        partition_records = set(int(value) for value in partitions[partition_name])
        token_indices = np.flatnonzero(np.isin(record_owner, list(partition_records)))
        for bucket in np.unique(token_bucket[token_indices]):
            group = token_indices[token_bucket[token_indices] == bucket]
            by_record: dict[int, np.ndarray] = {}
            for record in np.unique(record_owner[group]):
                by_record[int(record)] = group[record_owner[group] == record]
            if len(by_record) < 2:
                continue
            records = list(by_record)
            donor_order = list(rng.permutation(records))
            for destination_index, destination_record in enumerate(records):
                donor_record = donor_order[(donor_order.index(destination_record) + 1) % len(donor_order)]
                destination = by_record[destination_record]
                source = by_record[donor_record]
                source = source[rng.permutation(len(source))]
                source = source[np.arange(len(destination)) % len(source)]
                h[destination] = np.asarray(data.h)[source]
    return h


def _roc_points(scores: np.ndarray, labels: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    scores = np.asarray(scores, dtype=np.float64)
    labels = np.asarray(labels, dtype=np.int64)
    positives = int(np.sum(labels == 1))
    negatives = int(np.sum(labels == 0))
    if positives == 0 or negatives == 0:
        raise ValueError("ROC requires both member and nonmember records")
    order = np.argsort(-scores, kind="mergesort")
    ordered_scores = scores[order]
    ordered_labels = labels[order]
    change = np.r_[True, ordered_scores[1:] != ordered_scores[:-1]]
    starts = np.flatnonzero(change)
    ends = np.r_[starts[1:], len(scores)]
    tpr = [0.0]
    fpr = [0.0]
    tp = 0
    fp = 0
    for start, end in zip(starts, ends):
        group = ordered_labels[start:end]
        tp += int(np.sum(group == 1))
        fp += int(np.sum(group == 0))
        tpr.append(tp / positives)
        fpr.append(fp / negatives)
    return np.asarray(fpr, dtype=np.float64), np.asarray(tpr, dtype=np.float64)


def partial_auc(scores: np.ndarray, labels: np.ndarray, max_fpr: float = 0.05) -> float:
    """The registered ``integral TPR(f) df / max_fpr`` with tied thresholds."""

    if not 0.0 < max_fpr <= 1.0:
        raise ValueError("max_fpr must be in (0, 1]")
    fpr, tpr = _roc_points(scores, labels)
    area = 0.0
    boundary = float(max_fpr)
    for left in range(len(fpr) - 1):
        x0, x1 = float(fpr[left]), float(fpr[left + 1])
        y0, y1 = float(tpr[left]), float(tpr[left + 1])
        if x0 >= boundary:
            break
        right = min(x1, boundary)
        if right <= x0:
            continue
        if x1 <= x0:
            y_right = y1
        else:
            fraction = (right - x0) / (x1 - x0)
            y_right = y0 + fraction * (y1 - y0)
        area += (right - x0) * (y0 + y_right) / 2.0
        if x1 >= boundary:
            break
    return float(area / boundary)


def _wilson_interval(successes: int, total: int, confidence: float = 0.95) -> tuple[float, float]:
    if total <= 0:
        return float("nan"), float("nan")
    z = 1.959963984540054
    p = successes / total
    denominator = 1.0 + z * z / total
    centre = (p + z * z / (2.0 * total)) / denominator
    radius = z * math.sqrt(p * (1.0 - p) / total + z * z / (4.0 * total * total)) / denominator
    return max(0.0, centre - radius), min(1.0, centre + radius)


@dataclass
class BootstrapIndices:
    member: np.ndarray
    nonmember: np.ndarray
    calibration: np.ndarray


def make_bootstrap_indices(
    n_member: int,
    n_nonmember: int,
    n_calibration: int,
    repeats: int,
    seed: int,
) -> BootstrapIndices:
    rng = np.random.default_rng(seed)
    return BootstrapIndices(
        member=rng.integers(0, n_member, size=(repeats, n_member), dtype=np.int64),
        nonmember=rng.integers(0, n_nonmember, size=(repeats, n_nonmember), dtype=np.int64),
        calibration=rng.integers(0, n_calibration, size=(repeats, n_calibration), dtype=np.int64),
    )


def _bootstrap_metric_ci(
    scores: np.ndarray,
    labels: np.ndarray,
    partitions: Partitions,
    bootstrap: BootstrapIndices,
) -> dict[str, dict[str, float]]:
    test = partitions["test"]
    calibration = partitions["calibration"]
    test_member = test[labels[test] == 1]
    test_nonmember = test[labels[test] == 0]
    calibration_nonmember = calibration[labels[calibration] == 0]
    members = scores[test_member]
    nonmembers = scores[test_nonmember]
    calibration_values = scores[calibration_nonmember]
    auc_values = np.empty(len(bootstrap.member), dtype=np.float64)
    pauc_values = np.empty_like(auc_values)
    tpr_values = {rate: np.empty_like(auc_values) for rate in CALIBRATION_RATES}
    fpr_values = {rate: np.empty_like(auc_values) for rate in CALIBRATION_RATES}
    for repeat in range(len(auc_values)):
        member = members[bootstrap.member[repeat]]
        nonmember = nonmembers[bootstrap.nonmember[repeat]]
        cal = calibration_values[bootstrap.calibration[repeat]]
        auc_values[repeat] = rank_auc(member, nonmember)
        pauc_values[repeat] = partial_auc(
            np.concatenate((member, nonmember)),
            np.concatenate((np.ones(len(member), dtype=np.int64), np.zeros(len(nonmember), dtype=np.int64))),
        )
        for rate in CALIBRATION_RATES:
            point = threshold_metrics(member, nonmember, rate, calibration_nonmember=cal)
            tpr_values[rate][repeat] = point["tpr"]
            fpr_values[rate][repeat] = point["fpr"]
    result = {
        "auc": {
            "ci95_low": float(np.quantile(auc_values, 0.025)),
            "ci95_high": float(np.quantile(auc_values, 0.975)),
        },
        "pauc_0_05": {
            "ci95_low": float(np.quantile(pauc_values, 0.025)),
            "ci95_high": float(np.quantile(pauc_values, 0.975)),
        },
    }
    for rate in CALIBRATION_RATES:
        key = f"{int(rate * 100)}%"
        result[key] = {
            "tpr_ci95_low": float(np.quantile(tpr_values[rate], 0.025)),
            "tpr_ci95_high": float(np.quantile(tpr_values[rate], 0.975)),
            "fpr_ci95_low": float(np.quantile(fpr_values[rate], 0.025)),
            "fpr_ci95_high": float(np.quantile(fpr_values[rate], 0.975)),
        }
    return result


def evaluate_score_vector(
    scores: np.ndarray,
    data: M1Data,
    partitions: Partitions,
    bootstrap: BootstrapIndices | None,
    bootstrap_repeats: int,
) -> dict[str, Any]:
    labels = data.labels
    validation = partitions["validation"]
    validation_member = validation[labels[validation] == 1]
    validation_nonmember = validation[labels[validation] == 0]
    test = partitions["test"]
    test_member = test[labels[test] == 1]
    test_nonmember = test[labels[test] == 0]
    calibration = partitions["calibration"]
    calibration_nonmember = calibration[labels[calibration] == 0]
    validation_pauc = partial_auc(
        np.concatenate((scores[validation_member], scores[validation_nonmember])),
        np.concatenate((np.ones(len(validation_member), dtype=np.int64), np.zeros(len(validation_nonmember), dtype=np.int64))),
    )
    validation_dev = threshold_metrics(
        scores[validation_member],
        scores[validation_nonmember],
        0.01,
        calibration_nonmember=scores[validation_nonmember],
    )
    test_auc = rank_auc(scores[test_member], scores[test_nonmember])
    test_pauc = partial_auc(
        np.concatenate((scores[test_member], scores[test_nonmember])),
        np.concatenate((np.ones(len(test_member), dtype=np.int64), np.zeros(len(test_nonmember), dtype=np.int64))),
    )
    calibrated: dict[str, Any] = {}
    for rate in CALIBRATION_RATES:
        point = threshold_metrics(
            scores[test_member], scores[test_nonmember], rate, calibration_nonmember=scores[calibration_nonmember]
        )
        tpr_low, tpr_high = _wilson_interval(point["member_hits"], point["member_n"])
        fpr_low, fpr_high = _wilson_interval(point["nonmember_hits"], point["nonmember_n"])
        point.update(
            {
                "tpr_ci95_low_binomial": tpr_low,
                "tpr_ci95_high_binomial": tpr_high,
                "fpr_ci95_low_binomial": fpr_low,
                "fpr_ci95_high_binomial": fpr_high,
                "fpr_one_sided95_upper_if_zero": (
                    float(1.0 - 0.05 ** (1.0 / point["nonmember_n"]))
                    if point["nonmember_hits"] == 0
                    else None
                ),
            }
        )
        calibrated[f"{int(rate * 100)}%"] = point
    result: dict[str, Any] = {
        "validation": {
            "pauc_0_05": validation_pauc,
            "tpr_at_nominal_1pct": validation_dev["tpr"],
            "fpr_at_nominal_1pct": validation_dev["fpr"],
            "member_hits": validation_dev["member_hits"],
            "nonmember_hits": validation_dev["nonmember_hits"],
        },
        "test": {
            "auc": {"point": test_auc},
            "pauc_0_05": {"point": test_pauc},
            "calibrated": calibrated,
            "test_n_member": int(len(test_member)),
            "test_n_nonmember": int(len(test_nonmember)),
            "calibration_n_nonmember": int(len(calibration_nonmember)),
        },
    }
    if bootstrap is not None and bootstrap_repeats > 0:
        result["test"]["bootstrap_95"] = _bootstrap_metric_ci(scores, labels, partitions, bootstrap)
    return result


def paired_method_delta(
    left: np.ndarray,
    right: np.ndarray,
    data: M1Data,
    partitions: Partitions,
    bootstrap: BootstrapIndices | None,
) -> dict[str, Any]:
    labels = data.labels
    test = partitions["test"]
    tm = test[labels[test] == 1]
    tn = test[labels[test] == 0]
    cal = partitions["calibration"]
    cn = cal[labels[cal] == 0]
    left_auc = rank_auc(left[tm], left[tn])
    right_auc = rank_auc(right[tm], right[tn])
    left_pauc = partial_auc(
        np.concatenate((left[tm], left[tn])),
        np.concatenate((np.ones(len(tm), dtype=np.int64), np.zeros(len(tn), dtype=np.int64))),
    )
    right_pauc = partial_auc(
        np.concatenate((right[tm], right[tn])),
        np.concatenate((np.ones(len(tm), dtype=np.int64), np.zeros(len(tn), dtype=np.int64))),
    )
    result: dict[str, Any] = {
        "auc": {"point": left_auc - right_auc},
        "pauc_0_05": {"point": left_pauc - right_pauc},
        "calibrated": {},
    }
    for rate in CALIBRATION_RATES:
        left_point = threshold_metrics(left[tm], left[tn], rate, calibration_nonmember=left[cn])
        right_point = threshold_metrics(right[tm], right[tn], rate, calibration_nonmember=right[cn])
        result["calibrated"][f"{int(rate * 100)}%"] = {
            "tpr_delta": left_point["tpr"] - right_point["tpr"],
            "fpr_delta": left_point["fpr"] - right_point["fpr"],
        }
    if bootstrap is not None:
        auc_values = np.empty(len(bootstrap.member), dtype=np.float64)
        pauc_values = np.empty_like(auc_values)
        tpr_values = {rate: np.empty_like(auc_values) for rate in CALIBRATION_RATES}
        fpr_values = {rate: np.empty_like(auc_values) for rate in CALIBRATION_RATES}
        for repeat in range(len(auc_values)):
            mi, ni, ci = bootstrap.member[repeat], bootstrap.nonmember[repeat], bootstrap.calibration[repeat]
            left_m, right_m = left[tm][mi], right[tm][mi]
            left_n, right_n = left[tn][ni], right[tn][ni]
            left_c, right_c = left[cn][ci], right[cn][ci]
            auc_values[repeat] = rank_auc(left_m, left_n) - rank_auc(right_m, right_n)
            pauc_values[repeat] = partial_auc(
                np.concatenate((left_m, left_n)),
                np.concatenate((np.ones(len(left_m), dtype=np.int64), np.zeros(len(left_n), dtype=np.int64))),
            ) - partial_auc(
                np.concatenate((right_m, right_n)),
                np.concatenate((np.ones(len(right_m), dtype=np.int64), np.zeros(len(right_n), dtype=np.int64))),
            )
            for rate in CALIBRATION_RATES:
                left_point = threshold_metrics(left_m, left_n, rate, calibration_nonmember=left_c)
                right_point = threshold_metrics(right_m, right_n, rate, calibration_nonmember=right_c)
                tpr_values[rate][repeat] = left_point["tpr"] - right_point["tpr"]
                fpr_values[rate][repeat] = left_point["fpr"] - right_point["fpr"]
        result["auc"].update(
            {"ci95_low": float(np.quantile(auc_values, 0.025)), "ci95_high": float(np.quantile(auc_values, 0.975))}
        )
        result["pauc_0_05"].update(
            {"ci95_low": float(np.quantile(pauc_values, 0.025)), "ci95_high": float(np.quantile(pauc_values, 0.975))}
        )
        for rate in CALIBRATION_RATES:
            result["calibrated"][f"{int(rate * 100)}%"].update(
                {
                    "tpr_delta_ci95_low": float(np.quantile(tpr_values[rate], 0.025)),
                    "tpr_delta_ci95_high": float(np.quantile(tpr_values[rate], 0.975)),
                    "fpr_delta_ci95_low": float(np.quantile(fpr_values[rate], 0.025)),
                    "fpr_delta_ci95_high": float(np.quantile(fpr_values[rate], 0.975)),
                }
            )
    return result


def _fit_logistic(
    values: np.ndarray,
    labels: np.ndarray,
    l2: float,
) -> tuple[LinearRegressor, dict[str, Any]]:
    x = np.asarray(values, dtype=np.float64)
    y = np.asarray(labels, dtype=np.float64)
    class_counts = {0: int(np.sum(y == 0)), 1: int(np.sum(y == 1))}
    weights = np.asarray([0.5 / class_counts[int(label)] for label in y], dtype=np.float64)
    initial = np.zeros(x.shape[1] + 1, dtype=np.float64)

    def objective(parameters: np.ndarray) -> tuple[float, np.ndarray]:
        w, b = parameters[:-1], parameters[-1]
        logits = x @ w + b
        loss = np.logaddexp(0.0, logits) - y * logits
        probability = _sigmoid_np(logits)
        value = float(np.sum(weights * loss) + l2 * np.dot(w, w))
        gradient = np.concatenate((x.T @ (weights * (probability - y)) + 2.0 * l2 * w, [float(np.sum(weights * (probability - y)))]))
        return value, gradient

    result = minimize(
        lambda parameters: objective(parameters),
        initial,
        jac=True,
        method="L-BFGS-B",
        options={"maxiter": 500, "ftol": 1e-12, "gtol": 1e-8, "maxls": 40},
    )
    return LinearRegressor(result.x[:-1].copy(), float(result.x[-1]), positive=False), {
        "optimizer": "scipy.optimize.minimize:L-BFGS-B",
        "success": bool(result.success),
        "message": str(result.message),
        "iterations": int(result.nit),
    }


class _TorchDetector(nn.Module):
    def __init__(self, feature_dim: int) -> None:
        super().__init__()
        self.network = nn.Sequential(nn.Linear(feature_dim, 64), nn.GELU(), nn.Linear(64, 1))

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        return self.network(values).squeeze(-1)


@dataclass
class MLPDetector:
    module: _TorchDetector
    device: torch.device = torch.device("cpu")

    def predict(self, values: np.ndarray) -> np.ndarray:
        array = np.asarray(values, dtype=np.float32)
        output: list[np.ndarray] = []
        self.module.eval()
        with torch.inference_mode():
            for start in range(0, len(array), 65_536):
                batch = torch.from_numpy(array[start : start + 65_536]).to(self.device)
                output.append(self.module(batch).cpu().numpy())
        return np.concatenate(output).astype(np.float64) if output else np.empty(0, dtype=np.float64)

    def parameter_count(self) -> int:
        return sum(int(parameter.numel()) for parameter in self.module.parameters())


def _fit_mlp_detector(
    values: np.ndarray,
    labels: np.ndarray,
    validation_values: np.ndarray,
    validation_labels: np.ndarray,
    weight_decay: float,
    seed: int,
    max_epochs: int = 100,
    patience: int = 10,
    device: torch.device = torch.device("cpu"),
) -> tuple[MLPDetector, dict[str, Any]]:
    torch.manual_seed(seed)
    x = torch.from_numpy(np.asarray(values, dtype=np.float32)).to(device)
    y = torch.from_numpy(np.asarray(labels, dtype=np.float32)).to(device)
    class_counts = {0: max(1, int(np.sum(labels == 0))), 1: max(1, int(np.sum(labels == 1)))}
    weights = torch.from_numpy(np.asarray([0.5 / class_counts[int(label)] for label in labels], dtype=np.float32)).to(device)
    validation_x = torch.from_numpy(np.asarray(validation_values, dtype=np.float32)).to(device)
    validation_y = np.asarray(validation_labels, dtype=np.int64)
    model = _TorchDetector(x.shape[1]).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=weight_decay)
    best_state: dict[str, torch.Tensor] | None = None
    best_pauc = -float("inf")
    stale = 0
    epochs_run = 0
    for epoch in range(max_epochs):
        model.train()
        logits = model(x)
        loss = (F.binary_cross_entropy_with_logits(logits, y, reduction="none") * weights).sum()
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        model.eval()
        with torch.inference_mode():
            validation_scores = model(validation_x).cpu().numpy()
        value = partial_auc(validation_scores, validation_y)
        epochs_run = epoch + 1
        if value > best_pauc + 1e-10:
            best_pauc = value
            best_state = {key: tensor.detach().cpu().clone() for key, tensor in model.state_dict().items()}
            stale = 0
        else:
            stale += 1
            if stale >= patience:
                break
    if best_state is not None:
        model.load_state_dict(best_state)
    return MLPDetector(model, device=device), {
        "optimizer": "torch.optim.AdamW",
        "learning_rate": 1e-3,
        "weight_decay": weight_decay,
        "max_epochs": max_epochs,
        "epochs_run": epochs_run,
        "patience": patience,
        "best_validation_pauc_0_05": best_pauc,
        "seed": seed,
    }


@dataclass
class DetectorFit:
    method: str
    family: str
    regularization: float
    scaler: Standardizer
    predictors: dict[int, Regressor | MLPDetector]
    selection: list[dict[str, Any]]
    parameter_count: int

    def scores(self, values: np.ndarray, seed: int) -> np.ndarray:
        return self.predictors[seed].predict(self.scaler.transform(values))


def _select_detector_candidate(selection: list[dict[str, Any]]) -> int:
    """Select by V pAUC, then stronger regularization, then fewer parameters."""

    if not selection:
        raise RuntimeError("detector candidate selection received no candidates")
    return max(
        range(len(selection)),
        key=lambda index: (
            selection[index]["validation_pauc_0_05"],
            float(selection[index]["regularization"]),
            -int(selection[index]["parameter_count"]),
        ),
    )


def fit_detector(
    method: str,
    values: np.ndarray,
    data: M1Data,
    partitions: Partitions,
    family: str,
    detector_seeds: tuple[int, ...] = DETECTOR_SEEDS,
    device: torch.device = torch.device("cpu"),
) -> DetectorFit:
    detector_indices = partitions["detector_fit"]
    validation_indices = partitions["validation"]
    scaler = fit_row_standardizer(values[detector_indices])
    transformed = scaler.transform(values)
    x_train = transformed[detector_indices]
    y_train = data.labels[detector_indices]
    x_validation = transformed[validation_indices]
    y_validation = data.labels[validation_indices]
    candidates = DETECTOR_LOGISTIC_L2 if family == "logistic" else DETECTOR_MLP_WEIGHT_DECAYS
    selection: list[dict[str, Any]] = []
    for index, regularization in enumerate(candidates):
        if family == "logistic":
            predictor, optimizer_info = _fit_logistic(x_train, y_train, regularization)
        else:
            predictor, optimizer_info = _fit_mlp_detector(
                x_train,
                y_train,
                x_validation,
                y_validation,
                regularization,
                DETECTOR_SEEDS[0] + index,
                device=device,
            )
        validation_scores = predictor.predict(x_validation)
        validation_pauc = partial_auc(validation_scores, y_validation)
        selection.append(
            {
                "regularization": regularization,
                "validation_pauc_0_05": validation_pauc,
                "parameter_count": predictor.parameter_count(),
                "optimizer": optimizer_info,
            }
        )
    selected_index = _select_detector_candidate(selection)
    regularization = float(selection[selected_index]["regularization"])
    predictors: dict[int, Regressor | MLPDetector] = {}
    for seed in detector_seeds:
        if family == "logistic":
            predictor, _ = _fit_logistic(x_train, y_train, regularization)
        else:
            predictor, _ = _fit_mlp_detector(
                x_train,
                y_train,
                x_validation,
                y_validation,
                regularization,
                seed,
                device=device,
            )
        predictors[int(seed)] = predictor
    return DetectorFit(
        method=method,
        family=family,
        regularization=regularization,
        scaler=scaler,
        predictors=predictors,
        selection=selection,
        parameter_count=predictors[detector_seeds[0]].parameter_count(),
    )


def fixed_probability_baseline_scores(
    f19: np.ndarray,
    f19_names: Iterable[str],
) -> dict[str, np.ndarray]:
    """Build the fixed probability-only baseline scores from F19."""

    lookup = {name: index for index, name in enumerate(f19_names)}
    required = ("p_mean_logp", "mean_abs_delta", "window_sign_16", "window_sign_multiscale")
    missing = [name for name in required if name not in lookup]
    if missing:
        raise RuntimeError(f"F19 is missing B0 fields: {missing}")
    return {
        "B0-p_mean_logp": f19[:, lookup["p_mean_logp"]],
        "B0-mean_abs_delta": f19[:, lookup["mean_abs_delta"]],
        "B0-S16": f19[:, lookup["window_sign_16"]],
        "B0-multiscale": f19[:, lookup["window_sign_multiscale"]],
    }


def probability_b2_values(
    f19: np.ndarray,
    aggregate_delta: np.ndarray,
) -> np.ndarray:
    """Build B2 with the registered continuous delta aggregator."""

    return np.concatenate((f19, aggregate_delta), axis=1)


def _fixed_baseline_scores(data: M1Data) -> dict[str, np.ndarray]:
    return fixed_probability_baseline_scores(data.f19, data.f19_names)


def _aggregate_cache(values: np.ndarray, lengths: np.ndarray) -> np.ndarray:
    return aggregate_matrix(values, lengths).astype(np.float64)


def build_method_inputs(
    data: M1Data,
    residual_q: np.ndarray,
    residual_qh: np.ndarray,
    mu_q: np.ndarray,
    mu_qh: np.ndarray,
) -> dict[str, np.ndarray]:
    aggregate_delta = _aggregate_cache(data.delta, data.lengths)
    aggregate_rq = _aggregate_cache(residual_q, data.lengths)
    aggregate_rqh = _aggregate_cache(residual_qh, data.lengths)
    qh = np.concatenate((np.asarray(data.q), np.asarray(data.h)), axis=1)
    return {
        "B1": data.f19,
        "B2": probability_b2_values(data.f19, aggregate_delta),
        "MQ": np.concatenate((data.f19, aggregate_rq), axis=1),
        "MQH": np.concatenate((data.f19, aggregate_rqh), axis=1),
        "Q-direct": np.concatenate((data.f19, document_mean_std_matrix(np.asarray(data.q), data.lengths)), axis=1),
        "QH-direct": np.concatenate((data.f19, document_mean_std_matrix(qh, data.lengths)), axis=1),
        "MQ-A-only": aggregate_rq,
        "MQH-A-only": aggregate_rqh,
        "MQ-no-scale": np.concatenate((data.f19, _aggregate_cache(data.delta - mu_q, data.lengths)), axis=1),
        "MQH-no-scale": np.concatenate((data.f19, _aggregate_cache(data.delta - mu_qh, data.lengths)), axis=1),
    }


def _predictor_summary(predictor: Regressor | MLPDetector) -> dict[str, Any]:
    if isinstance(predictor, LinearRegressor):
        return {
            "type": "linear",
            "positive": predictor.positive,
            "parameter_count": predictor.parameter_count(),
            "weight": predictor.weight.tolist(),
            "bias": predictor.bias,
        }
    if isinstance(predictor, (MLPRegressor, MLPDetector)):
        return {
            "type": "mlp",
            "positive": bool(getattr(predictor, "positive", False)),
            "parameter_count": predictor.parameter_count(),
            "state_dict": {key: value.detach().cpu().numpy().tolist() for key, value in predictor.module.state_dict().items()},
        }
    return {"type": type(predictor).__name__, "parameter_count": predictor.parameter_count()}


def save_model_artifacts(
    output_dir: Path,
    conditional: dict[str, ConditionalCalibration],
    detector_fits: dict[str, DetectorFit],
) -> None:
    model_dir = output_dir / "models"
    model_dir.mkdir(parents=True, exist_ok=True)
    conditional_manifest: dict[str, Any] = {}
    for name, fit in conditional.items():
        payload = {
            "name": fit.name,
            "family": fit.family,
            "target_center": fit.target_center,
            "target_scale": fit.target_scale,
            "mean_abs_residual_ns": fit.mean_abs_residual_ns,
            "location_l2": fit.location_l2,
            "scale_l2": fit.scale_l2,
            "scaler": {
                "mean": fit.scaler.mean.tolist(),
                "scale": fit.scaler.scale.tolist(),
                "zero_variance": fit.scaler.zero_variance.tolist(),
            },
            "location": _predictor_summary(fit.location),
            "scale": _predictor_summary(fit.scale),
        }
        path = model_dir / f"conditional_{name}.json"
        path.write_text(json.dumps(_jsonable(payload), indent=2), encoding="utf-8")
        conditional_manifest[name] = str(path.relative_to(output_dir))
    detector_manifest: dict[str, Any] = {}
    for name, fit in detector_fits.items():
        payload = {
            "method": fit.method,
            "family": fit.family,
            "regularization": fit.regularization,
            "parameter_count": fit.parameter_count,
            "scaler": {
                "mean": fit.scaler.mean.tolist(),
                "scale": fit.scaler.scale.tolist(),
                "zero_variance": fit.scaler.zero_variance.tolist(),
            },
            "selection": fit.selection,
            "predictors": {str(seed): _predictor_summary(predictor) for seed, predictor in fit.predictors.items()},
        }
        path = model_dir / f"detector_{name.replace('/', '_')}.json"
        path.write_text(json.dumps(_jsonable(payload), indent=2), encoding="utf-8")
        detector_manifest[name] = str(path.relative_to(output_dir))
    (model_dir / "model_manifest.json").write_text(
        json.dumps({"conditional": conditional_manifest, "detectors": detector_manifest}, indent=2),
        encoding="utf-8",
    )


def run_experiment(args: argparse.Namespace) -> dict[str, Any]:
    started = time.perf_counter()
    feature_dir = Path(args.feature_dir).resolve()
    probability_dir = Path(args.probability_dir).resolve()
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)
    if device.type == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("requested CUDA for M1 fitting, but CUDA is unavailable")
        torch.cuda.set_device(device)
    data = load_m1_data(feature_dir, probability_dir, args.role)
    frozen_partition_path = _resolve_frozen_partition_path(
        feature_dir,
        output_dir,
        args.partition_manifest,
    )
    frozen_settings = json.loads(frozen_partition_path.read_text(encoding="utf-8"))
    partitions = make_partitions(
        data.labels,
        data.record_ids,
        frozen_manifest_path=frozen_partition_path,
    )
    frozen_root = feature_dir / "partition_manifest.json"
    if not frozen_root.exists():
        frozen_root.write_text(
            json.dumps(
                partition_manifest(partitions, data.labels, data.record_ids),
                indent=2,
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )
    (output_dir / "partition_manifest.json").write_text(
        json.dumps(partition_manifest(partitions, data.labels, data.record_ids), indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    h_variant = _activation_noise_or_permutation(data, args.activation_mode, args.seed, partitions)
    q_values = np.asarray(data.q)
    qh_values = np.concatenate((q_values, h_variant), axis=1)
    conditional_q = fit_conditional_calibration(
        "Q", q_values, data, partitions, args.conditional_family, args.seed, device=device
    )
    conditional_qh = fit_conditional_calibration(
        "QH", qh_values, data, partitions, args.conditional_family, args.seed, device=device
    )
    transformed_q = conditional_q.transform(q_values)
    transformed_qh = conditional_qh.transform(qh_values)
    residual_options: dict[str, dict[float, np.ndarray]] = {"Q": {}, "QH": {}}
    residual_diagnostics: dict[str, dict[str, Any]] = {}
    for name, fit, transformed in (("Q", conditional_q, transformed_q), ("QH", conditional_qh, transformed_qh)):
        for fraction in (0.10, 0.25):
            residual_options[name][fraction], diagnostics = fit.residual(transformed, data.delta, fraction)
            residual_diagnostics[f"{name}_c{fraction}"] = diagnostics
    # The conditional-model development table is kept separate from detector V
    # selection so the result makes the H1 path auditable.
    conditional_report = {
        "Q": {
            "family": conditional_q.family,
            "location_l2": conditional_q.location_l2,
            "scale_l2": conditional_q.scale_l2,
            "location_candidates": conditional_q.location_candidates,
            "scale_candidates": conditional_q.scale_candidates,
            "target_center": conditional_q.target_center,
            "target_scale": conditional_q.target_scale,
            "mean_abs_residual_ns": conditional_q.mean_abs_residual_ns,
            "development": conditional_dev_metrics(conditional_q, q_values, data, partitions),
        },
        "QH": {
            "family": conditional_qh.family,
            "location_l2": conditional_qh.location_l2,
            "scale_l2": conditional_qh.scale_l2,
            "location_candidates": conditional_qh.location_candidates,
            "scale_candidates": conditional_qh.scale_candidates,
            "target_center": conditional_qh.target_center,
            "target_scale": conditional_qh.target_scale,
            "mean_abs_residual_ns": conditional_qh.mean_abs_residual_ns,
            "development": conditional_dev_metrics(conditional_qh, qh_values, data, partitions),
        },
    }
    method_inputs: dict[str, np.ndarray] = {}
    # B2 and direct baselines do not depend on the floor choice. MQ/MQH are
    # selected over c on V using the same detector learner/search as the final
    # method, with C/T withheld until after that choice.
    fixed_baselines = _fixed_baseline_scores(data)
    base_inputs = build_method_inputs(
        data,
        residual_options["Q"][0.10],
        residual_options["QH"][0.10],
        conditional_q.predict_mu(transformed_q),
        conditional_qh.predict_mu(transformed_qh),
    )
    method_inputs.update({name: base_inputs[name] for name in ("B1", "B2", "Q-direct", "QH-direct", "MQ-A-only", "MQH-A-only", "MQ-no-scale", "MQH-no-scale")})
    for method, matrix in base_inputs.items():
        if method in {"MQ", "MQH"}:
            continue
        if method not in method_inputs:
            method_inputs[method] = matrix
    c_candidates = (0.10, 0.25)
    for method_name, condition_name in (("MQ", "Q"), ("MQH", "QH")):
        # Store candidate matrices; fit_detector below performs the fixed
        # detector search and lets V choose c and the regularization together.
        for fraction in c_candidates:
            residual = residual_options[condition_name][fraction]
            method_inputs[f"{method_name}@c{fraction}"] = np.concatenate(
                (
                    data.f19,
                    _aggregate_cache(residual, data.lengths),
                ),
                axis=1,
            )

    method_names = [*fixed_baselines.keys()]
    for name in ("B1", "B2", "Q-direct", "QH-direct", "MQ-A-only", "MQH-A-only", "MQ-no-scale", "MQH-no-scale"):
        method_names.append(f"{name}/{args.detector_families.split(',')[0]}")
    for name in ("MQ", "MQH"):
        method_names.append(f"{name}/{args.detector_families.split(',')[0]}")
    detector_fits: dict[str, DetectorFit] = {}
    primary_scores: dict[str, np.ndarray] = dict(fixed_baselines)
    all_seed_scores: dict[str, dict[int, np.ndarray]] = {}
    detector_reports: dict[str, Any] = {}
    selected_c: dict[str, Any] = {}
    requested_families = tuple(value.strip() for value in args.detector_families.split(",") if value.strip())
    if not requested_families or any(value not in {"logistic", "mlp"} for value in requested_families):
        raise ValueError("--detector-families must contain logistic and/or mlp")
    bootstrap = None if args.no_bootstrap else make_bootstrap_indices(
        int(np.sum(data.labels[partitions["test"]] == 1)),
        int(np.sum(data.labels[partitions["test"]] == 0)),
        int(np.sum(data.labels[partitions["calibration"]] == 0)),
        args.bootstrap_repeats,
        args.seed + 50_000,
    )
    method_result_scores: dict[str, np.ndarray] = {}
    method_results: dict[str, Any] = {}
    for family in requested_families:
        learned_method_names = ("B1", "B2", "Q-direct", "QH-direct", "MQ-A-only", "MQH-A-only", "MQ-no-scale", "MQH-no-scale")
        for method in learned_method_names:
            detector_name = f"{method}/{family}"
            fit = fit_detector(
                detector_name, method_inputs[method], data, partitions, family, device=device
            )
            detector_fits[detector_name] = fit
            seed_scores = {seed: fit.scores(method_inputs[method], seed) for seed in DETECTOR_SEEDS}
            all_seed_scores[detector_name] = seed_scores
            primary_scores[detector_name] = seed_scores[DETECTOR_SEEDS[0]]
            method_result_scores[detector_name] = primary_scores[detector_name]
            detector_reports[detector_name] = {
                "family": family,
                "regularization": fit.regularization,
                "parameter_count": fit.parameter_count,
                "selection": fit.selection,
                "training_seeds": list(DETECTOR_SEEDS),
                "seed_validation_and_test": {
                    str(seed): evaluate_score_vector(scores, data, partitions, None, 0)
                    for seed, scores in seed_scores.items()
                },
            }
        for method in ("MQ", "MQH"):
            candidates: list[tuple[float, DetectorFit, dict[str, Any], dict[int, np.ndarray]]] = []
            for fraction in c_candidates:
                candidate_name = f"{method}@c{fraction}/{family}"
                fit = fit_detector(
                    candidate_name,
                    method_inputs[f"{method}@c{fraction}"],
                    data,
                    partitions,
                    family,
                    device=device,
                )
                primary_candidate_scores = fit.scores(method_inputs[f"{method}@c{fraction}"], DETECTOR_SEEDS[0])
                candidate_validation = evaluate_score_vector(primary_candidate_scores, data, partitions, None, 0)["validation"]["pauc_0_05"]
                candidates.append((fraction, fit, {"validation_pauc_0_05": candidate_validation}, {seed: fit.scores(method_inputs[f"{method}@c{fraction}"], seed) for seed in DETECTOR_SEEDS}))
            selected_fraction, selected_fit, selected_meta, selected_seed_scores = max(
                candidates,
                key=lambda item: (item[2]["validation_pauc_0_05"], -item[0]),
            )
            public_name = f"{method}/{family}"
            selected_fit.method = public_name
            detector_fits[public_name] = selected_fit
            all_seed_scores[public_name] = selected_seed_scores
            primary_scores[public_name] = selected_seed_scores[DETECTOR_SEEDS[0]]
            method_result_scores[public_name] = primary_scores[public_name]
            selected_c[public_name] = {
                "selected_c": selected_fraction,
                "c_candidates": [
                    {"c": fraction, **metadata} for fraction, _fit, metadata, _scores in candidates
                ],
                "floor_diagnostics": residual_diagnostics[f"{method.replace('M', '') if False else ('Q' if method == 'MQ' else 'QH')}_c{selected_fraction}"],
            }
            detector_reports[public_name] = {
                "family": family,
                "regularization": selected_fit.regularization,
                "parameter_count": selected_fit.parameter_count,
                "selection": selected_fit.selection,
                "training_seeds": list(DETECTOR_SEEDS),
                "selected_c": selected_fraction,
                "c_candidates": [
                    {"c": fraction, **metadata} for fraction, _fit, metadata, _scores in candidates
                ],
                "seed_validation_and_test": {
                    str(seed): evaluate_score_vector(scores, data, partitions, None, 0)
                    for seed, scores in selected_seed_scores.items()
                },
            }
    # Fixed B0 scores are not learned and therefore have no detector seed.
    for name, scores in fixed_baselines.items():
        method_result_scores[name] = scores
    for name, scores in method_result_scores.items():
        method_results[name] = evaluate_score_vector(scores, data, partitions, bootstrap, args.bootstrap_repeats)
    paired: dict[str, Any] = {}
    for family in requested_families:
        for left, right in (("MQH", "MQ"), ("MQH", "B2"), ("MQH", "B1")):
            left_name = f"{left}/{family}"
            right_name = f"{right}/{family}"
            if left_name in method_result_scores and right_name in method_result_scores:
                paired[f"{left_name} - {right_name}"] = paired_method_delta(
                    method_result_scores[left_name],
                    method_result_scores[right_name],
                    data,
                    partitions,
                    bootstrap,
                )
    score_names = list(method_result_scores)
    primary_matrix = np.column_stack([method_result_scores[name] for name in score_names])
    partition_codes = np.asarray([partitions.partition_by_index[index] for index in range(len(data.labels))])
    pvalues = np.empty((len(data.labels), len(score_names), len(CALIBRATION_RATES)), dtype=np.float32)
    calibration_nonmember = partitions["calibration"][data.labels[partitions["calibration"]] == 0]
    for column, name in enumerate(score_names):
        for rate_index, _rate in enumerate(CALIBRATION_RATES):
            pvalues[:, column, rate_index] = conformal_tail_pvalues(
                method_result_scores[name], method_result_scores[name][calibration_nonmember]
            ).astype(np.float32)
    np.savez_compressed(
        output_dir / "frozen_scores.npz",
        method_names=np.asarray(score_names),
        labels=data.labels,
        record_ids=data.record_ids,
        partition=partition_codes,
        scores=primary_matrix.astype(np.float32),
        conformal_pvalues=pvalues,
        calibration_rates=np.asarray(CALIBRATION_RATES, dtype=np.float32),
        primary_seed=np.asarray(DETECTOR_SEEDS[0], dtype=np.int64),
    )
    for detector_name, seed_scores in all_seed_scores.items():
        np.savez_compressed(
            output_dir / f"scores_{detector_name.replace('/', '_')}_seeds.npz",
            method=np.asarray(detector_name),
            seeds=np.asarray(DETECTOR_SEEDS, dtype=np.int64),
            scores=np.column_stack([seed_scores[seed] for seed in DETECTOR_SEEDS]).astype(np.float32),
        )
    save_model_artifacts(output_dir, {"Q": conditional_q, "QH": conditional_qh}, detector_fits)
    report = {
        "protocol": {
            "fit_version": FIT_VERSION,
            "training_regime": data.feature_manifest.get("training_regime", "controlled_sft"),
            "models": data.feature_manifest.get("models"),
            "token_contract": data.feature_manifest.get("token_contract"),
            "dataset_manifest": data.feature_manifest.get("dataset_manifest"),
            "selected_blocks_zero_based": data.feature_manifest.get("selected_blocks_zero_based"),
            "role": data.role,
            "feature_dir": str(feature_dir),
            "probability_dir": str(probability_dir),
            "benchmark": data.feature_manifest.get("benchmark"),
            "epoch": data.feature_manifest.get("epoch"),
            "records": int(len(data.labels)),
            "tokens": int(len(data.delta)),
            "eos_included": bool(data.feature_manifest.get("eos_included", True)),
            "activation_mode": args.activation_mode,
            "conditional_family": args.conditional_family,
            "detector_families": list(requested_families),
            "split_seed": frozen_settings.get("split_seed", SPLIT_SEED),
            "nuisance_seed": frozen_settings.get("nuisance_seed", NUISANCE_SEED),
            "detector_seeds": list(DETECTOR_SEEDS),
            "bootstrap_repeats": 0 if args.no_bootstrap else args.bootstrap_repeats,
            "fit_device": str(device),
            "fit_device_name": (
                torch.cuda.get_device_name(device) if device.type == "cuda" else "cpu"
            ),
            "threshold_rule": "(1 + count(calibration_score >= score))/(n+1) <= eta; inclusive ties",
            "pauc_definition": "integral TPR(f) df from 0 to 0.05 divided by 0.05; tied scores are thresholded as one group",
            "parameter_standardization": "detector rows standardized on D only; conditional token features standardized on Nmu only; zero-variance dimensions set to zero",
        },
        "partitions": {name: {"n": int(len(values)), "members": int(np.sum(data.labels[values] == 1)), "nonmembers": int(np.sum(data.labels[values] == 0))} for name, values in partitions.indices.items()},
        "conditional_models": conditional_report,
        "residual_diagnostics": residual_diagnostics,
        "detector_models": detector_reports,
        "selected_c": selected_c,
        "methods": method_results,
        "paired_deltas": paired,
        "score_archive": {
            "path": str((output_dir / "frozen_scores.npz").resolve()),
            "method_names": score_names,
            "calibration_rates": list(CALIBRATION_RATES),
        },
        "runtime": {
            "seconds": time.perf_counter() - started,
            "python": platform.python_version(),
            "numpy": np.__version__,
            "torch": torch.__version__,
        },
    }
    (output_dir / "m1_metrics.json").write_text(
        json.dumps(_jsonable(report), indent=2, ensure_ascii=False), encoding="utf-8"
    )
    _write_m1_report(output_dir, report)
    return report


def _fmt_interval(row: dict[str, Any], value_key: str = "point") -> str:
    point = row.get(value_key)
    if point is None:
        return "n/a"
    low = row.get("ci95_low")
    high = row.get("ci95_high")
    return f"{point:.4f} [{low:.4f}, {high:.4f}]" if low is not None and high is not None else f"{point:.4f}"


def _write_m1_report(output_dir: Path, report: dict[str, Any]) -> None:
    lines = [
        "# M1 conditional-calibration experiment",
        "",
        f"- Benchmark/epoch: `{report['protocol'].get('benchmark')}` / `{report['protocol'].get('epoch')}`",
        f"- Role: `{report['protocol']['role']}`; activation mode: `{report['protocol']['activation_mode']}`",
        "- Main evaluation uses the frozen C/T split. D trains the detector, V selects conditional/detector settings, and C supplies only nonmember conformal tails.",
        "- pAUC is the registered integral definition, not a library standardized partial-AUC score.",
        "",
        "## Primary detector results",
        "",
        "| Method | Test AUC | pAUC[0,.05] | TPR@1% | actual FPR@1% |",
        "|---|---:|---:|---:|---:|",
    ]
    for name, result in report["methods"].items():
        test = result["test"]
        one = test["calibrated"]["1%"]
        lines.append(
            f"| `{name}` | {_fmt_interval(test['auc'])} | {_fmt_interval(test['pauc_0_05'])} | "
            f"{one['tpr']:.4f} ({one['member_hits']}/{one['member_n']}) | "
            f"{one['fpr']:.4f} ({one['nonmember_hits']}/{one['nonmember_n']}) |"
        )
    lines.extend(["", "## Conditional development (V nonmembers)", "", "| Model | location MAE | location bias | scale MAE |", "|---|---:|---:|---:|"])
    for name, result in report["conditional_models"].items():
        development = result["development"]
        lines.append(
            f"| `{name}` | {development['location_mae']:.6f} | {development['location_bias']:.6f} | {development['scale_mae']:.6f} |"
        )
    lines.extend(["", "## Paired primary deltas", "", "| Comparison | ΔAUC | ΔpAUC | ΔTPR@1% | ΔFPR@1% |", "|---|---:|---:|---:|---:|"])
    for name, result in report["paired_deltas"].items():
        one = result["calibrated"]["1%"]
        lines.append(
            f"| `{name}` | {_fmt_interval(result['auc'])} | {_fmt_interval(result['pauc_0_05'])} | "
            f"{one['tpr_delta']:.4f} | {one['fpr_delta']:.4f} |"
        )
    lines.extend(
        [
            "",
            "Zero-FP test rows include a one-sided 95% upper bound in `m1_metrics.json`; zero observed false positives is not zero risk.",
            "",
        ]
    )
    (output_dir / "M1_RESULTS.md").write_text("\n".join(lines), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--feature-dir", type=Path, required=True)
    parser.add_argument("--probability-dir", type=Path, required=True)
    parser.add_argument("--role", required=True, choices=("draft_auxiliary_distilled", "draft_member_sft", "draft_pretrained"))
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--partition-manifest",
        type=Path,
        default=None,
        help="Frozen record_id partition manifest; required unless one already exists in the feature/output directory.",
    )
    parser.add_argument("--conditional-family", choices=("linear", "mlp"), default="linear")
    parser.add_argument("--detector-families", default="logistic,mlp")
    parser.add_argument("--activation-mode", choices=("real", "noise", "within_bucket_permute"), default="real")
    parser.add_argument("--bootstrap-repeats", type=int, default=2000)
    parser.add_argument("--no-bootstrap", action="store_true")
    parser.add_argument("--seed", type=int, default=NUISANCE_SEED)
    parser.add_argument(
        "--device",
        default="cpu",
        help="Torch device for MLP conditional/detector fits (e.g. cuda:0); linear/SciPy fits remain CPU-bound.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_experiment(args)
    print(json.dumps({"output_dir": str(Path(args.output_dir).resolve())}, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
