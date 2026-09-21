"""Frozen cache loading for the offline verifier simulator.

Exact p/delta stay on the simulator side; detector observations are defined in
conditional_accept_only. Cache validation and EOS removal match the original
registered replay, including its historical p/q alignment checks.
"""
from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
import numpy as np

@dataclass(frozen=True)
class DeltaData:
    labels: np.ndarray
    record_ids: np.ndarray
    lengths: np.ndarray
    offsets: np.ndarray
    delta: np.ndarray


def load_delta_data(path: Path) -> DeltaData:
    path = path.resolve()
    with np.load(path, allow_pickle=False) as data:
        required = ("labels", "record_ids", "lengths", "offsets", "delta")
        missing = [key for key in required if key not in data.files]
        if missing:
            raise ValueError(f"{path} is missing {missing}")
        labels = np.asarray(data["labels"], dtype=np.int64)
        record_ids = np.asarray(data["record_ids"])
        lengths = np.asarray(data["lengths"], dtype=np.int64)
        offsets = np.asarray(data["offsets"], dtype=np.int64)
        delta = np.asarray(data["delta"], dtype=np.float32)
    if labels.ndim != 1 or record_ids.ndim != 1 or len(labels) != len(record_ids):
        raise ValueError("labels and record_ids must be aligned vectors")
    if len(lengths) != len(labels) or len(offsets) != len(labels) + 1:
        raise ValueError("lengths/offsets do not align with records")
    if np.any(lengths <= 0) or int(offsets[-1]) != len(delta):
        raise ValueError("invalid lengths/offsets")
    if not np.all(np.isfinite(delta)):
        raise ValueError("delta contains non-finite values")
    if len(set(str(value) for value in record_ids)) != len(record_ids):
        raise ValueError("record_ids must be unique")
    return DeltaData(labels, record_ids, lengths, offsets, delta)


def sliding_means(values: np.ndarray, width: int) -> np.ndarray:
    if width <= 0:
        raise ValueError("width must be positive")
    if len(values) == 0:
        return np.zeros(1, dtype=np.float64)
    if len(values) < width:
        return np.asarray([float(np.mean(values))], dtype=np.float64)
    cumulative = np.concatenate(([0.0], np.cumsum(values, dtype=np.float64)))
    return (cumulative[width:] - cumulative[:-width]) / width


def drop_final_cached_token(data: DeltaData) -> DeltaData:
    """Remove the producer-appended final token without changing fragment scope."""
    if np.any(data.lengths <= 1):
        raise ValueError("cannot drop the final token from a one-token record")
    pieces = [
        data.delta[int(start) : int(end) - 1]
        for start, end in zip(data.offsets[:-1], data.offsets[1:])
    ]
    lengths = data.lengths - 1
    offsets = np.r_[0, np.cumsum(lengths, dtype=np.int64)]
    return DeltaData(
        labels=data.labels,
        record_ids=data.record_ids,
        lengths=lengths,
        offsets=offsets,
        delta=np.concatenate(pieces).astype(np.float32, copy=False),
    )


def load_paired_logps(path: Path, expected: DeltaData, drop_final: bool) -> tuple[np.ndarray, np.ndarray]:
    with np.load(path, allow_pickle=False) as archive:
        for key in ("lengths", "target", "draft_auxiliary_distilled"):
            if key not in archive.files:
                raise ValueError(f"{path} is missing {key}")
        source_lengths = np.asarray(archive["lengths"], dtype=np.int64)
        target = np.asarray(archive["target"], dtype=np.float32)
        draft = np.asarray(archive["draft_auxiliary_distilled"], dtype=np.float32)
    if len(source_lengths) != len(expected.labels):
        raise ValueError("paired p/q cache record count is not aligned")
    if drop_final:
        pieces_target, pieces_draft = [], []
        offsets = np.r_[0, np.cumsum(source_lengths, dtype=np.int64)]
        for start, end in zip(offsets[:-1], offsets[1:]):
            pieces_target.append(target[int(start) : int(end) - 1])
            pieces_draft.append(draft[int(start) : int(end) - 1])
        target, draft = np.concatenate(pieces_target), np.concatenate(pieces_draft)
        source_lengths = source_lengths - 1
    if not np.array_equal(source_lengths, expected.lengths):
        raise ValueError("paired p/q cache lengths are not aligned")
    if len(target) != len(expected.delta) or len(draft) != len(target):
        raise ValueError("paired p/q token arrays are not aligned")
    if not np.allclose(target - draft, expected.delta, atol=2e-6, rtol=1e-6):
        raise ValueError("paired p/q cache disagrees with delta cache")
    return target, draft


@dataclass(frozen=True)
class ReplayData:
    labels: np.ndarray
    record_ids: np.ndarray
    lengths: np.ndarray
    offsets: np.ndarray
    logp: np.ndarray
    logq0: np.ndarray

    @property
    def delta0(self) -> np.ndarray:
        return self.logp - self.logq0


def load_replay_data(full_delta_path: Path, pq_path: Path) -> ReplayData:
    """Load aligned exact caches and drop each record's final cached token."""
    original = load_delta_data(full_delta_path)
    without_final = drop_final_cached_token(original)
    logp, logq0 = load_paired_logps(pq_path, without_final, drop_final=True)
    if np.any(logp > 1e-6) or np.any(logq0 > 1e-6):
        raise ValueError("cached log probabilities must not be positive")
    return ReplayData(
        labels=without_final.labels,
        record_ids=without_final.record_ids,
        lengths=without_final.lengths,
        offsets=without_final.offsets,
        logp=np.asarray(logp, dtype=np.float64),
        logq0=np.asarray(logq0, dtype=np.float64),
    )

