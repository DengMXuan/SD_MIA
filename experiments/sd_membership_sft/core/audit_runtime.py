"""Shared paths, frozen split constants and atomic JSON for SD experiments."""
from __future__ import annotations
import json
import os
import tempfile
from pathlib import Path
from typing import Any
import numpy as np

from experiments.paths import ROOT
SPLIT_SEED = 20260824
DEFAULT_SPLIT_SEED = SPLIT_SEED
BENCHMARKS = ("wikitection", "newstection", "arxivtection")
EPOCHS = (1, 3)
REPLAY_SEEDS = (20260914, 20260915, 20260916)
N_REF = 400
N_CAL = 200

def _jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_jsonable(item) for item in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, Path):
        return str(value)
    return value


def _write_json(path: Path, value: Any) -> None:
    path = path.resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(_jsonable(value), handle, indent=2, ensure_ascii=False)
            handle.write("\n")
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def _paths(benchmark: str, epoch: int) -> tuple[Path, Path]:
    full = ROOT / "experiments/results/sft_runs/full_delta" / f"{benchmark}_epoch{epoch}" / "draft_auxiliary_distilled/full_delta.npz"
    pq = ROOT / "experiments/results/sft_runs/pq_directional" / f"{benchmark}_epoch{epoch}" / "pq_gap_token_logps.npz"
    return full, pq


def split_indices(labels: np.ndarray, seed: int = DEFAULT_SPLIT_SEED) -> dict[str, np.ndarray]:
    """Return the registered per-class D/V/C/T split (800/400/400/400)."""

    labels = np.asarray(labels, dtype=np.int64)
    rng = np.random.default_rng(seed)
    result: dict[str, list[int]] = {name: [] for name in ("D", "V", "C", "T")}
    for label in (1, 0):
        indices = np.flatnonzero(labels == label)
        if len(indices) < 2000:
            raise ValueError("Full-Delta requires at least 2,000 records per class")
        indices = rng.permutation(indices)[:2000]
        start = 0
        for name, count in zip(result, (800, 400, 400, 400)):
            result[name].extend(indices[start : start + count].tolist())
            start += count
    return {name: np.asarray(sorted(values), dtype=np.int64) for name, values in result.items()}


def _deterministic_subset(values: np.ndarray, count: int, seed: int) -> np.ndarray:
    values = np.asarray(values, dtype=np.int64)
    if count > len(values):
        raise ValueError("requested subset is too large")
    permutation = np.random.default_rng(seed).permutation(values)
    return np.sort(permutation[:count])


def _record_uniforms(seed: int, record_index: int, length: int, width: int) -> np.ndarray:
    return np.random.default_rng(np.random.SeedSequence([seed, record_index])).random(
        (length, width), dtype=np.float64
    )
