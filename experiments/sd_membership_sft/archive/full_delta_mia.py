"""Full-Delta membership audit.

This module implements the first information-permission experiment from the
2026-09-12 protocol.  It consumes the already materialized teacher-forced
target/draft log-probabilities, derives ``delta = logp - logq`` and trains
detectors whose sequence input is delta only.  The legacy B2 statistic is kept
as an explicitly labelled P/Q-stat control.

The command is intentionally splittable by ``--method`` so independent
benchmark/method jobs can share the seven available GPUs.  Candidate
hyperparameters are selected on V using the mean validation pAUC over the
three registered detector seeds.  C is used only for threshold calibration and
T is held out for the final report.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import itertools
import json
import math
import os
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import torch
from scipy.optimize import minimize
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset, Subset

from experiments.sd_membership_sft.core.audit_metrics import (conformal_tail_pvalues, order_statistic_threshold)
from experiments.sd_membership_sft.analysis.directional_mia import (record_features)
from experiments.sd_membership_sft.analysis.m1_features import (aggregate_matrix)

from experiments.sd_membership_sft.core.audit_runtime import (DEFAULT_SPLIT_SEED, split_indices)
from experiments.sd_membership_sft.core.audit_metrics import (_roc_points, rank_auc, partial_auc)


from experiments.paths import ROOT
DEFAULT_TRAINING_SEEDS = (20260909, 20260910, 20260911)
DEFAULT_WINDOWS = (4, 8, 16, 32, 64)
METHODS = (
    "b2",
    "delta-mean",
    "delta-uniformnet",
    "delta-attn",
    "delta-tcn",
    "delta-transformer",
)
LEARNED_METHODS = METHODS[2:]
RATES = (0.10, 0.01)


def method_scores_path(output_dir: Path, method: str) -> Path:
    """Return a method-specific score path so parallel methods cannot collide."""

    return output_dir / f"scores_{method.replace('-', '_').lower()}.npz"


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


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_json_atomic(path: Path, value: Any) -> None:
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


@dataclass(frozen=True)
class DeltaArchive:
    labels: np.ndarray
    record_ids: np.ndarray
    lengths: np.ndarray
    offsets: np.ndarray
    delta: np.ndarray
    target: np.ndarray | None
    draft: np.ndarray | None
    input_path: Path
    scores_path: Path


def _validate_ids(record_ids: np.ndarray, n: int) -> np.ndarray:
    values = np.asarray(record_ids)
    if values.ndim != 1 or len(values) != n:
        raise ValueError("record_ids must be one-dimensional and aligned with labels")
    normalized = [str(value) for value in values]
    if len(set(normalized)) != len(normalized):
        raise ValueError("record_ids must be unique")
    return values


def load_delta_archive(
    input_path: Path,
    scores_path: Path | None = None,
    role: str = "draft_auxiliary_distilled",
) -> DeltaArchive:
    """Load and validate one saved p/q archive without using activation/H."""

    input_path = input_path.resolve()
    if scores_path is None:
        scores_path = input_path.with_name("pq_gap_scores.npz")
    scores_path = scores_path.resolve()
    data = np.load(input_path, allow_pickle=False)
    score_data = np.load(scores_path, allow_pickle=False)
    for key in ("lengths", "target", role):
        if key not in data.files:
            raise ValueError(f"{input_path} is missing {key}")
    for key in ("labels", "record_ids"):
        if key not in score_data.files:
            raise ValueError(f"{scores_path} is missing {key}")
    labels = np.asarray(score_data["labels"], dtype=np.int64)
    record_ids = _validate_ids(np.asarray(score_data["record_ids"]), len(labels))
    lengths = np.asarray(data["lengths"], dtype=np.int64)
    if lengths.ndim != 1 or len(lengths) != len(labels) or np.any(lengths <= 0):
        raise ValueError("lengths must be positive and aligned with labels")
    offsets = np.concatenate(([0], np.cumsum(lengths, dtype=np.int64)))
    target = np.asarray(data["target"], dtype=np.float32)
    draft = np.asarray(data[role], dtype=np.float32)
    if len(target) != int(offsets[-1]) or len(draft) != len(target):
        raise ValueError("token arrays do not match lengths")
    delta = target.astype(np.float32) - draft.astype(np.float32)
    if not np.all(np.isfinite(delta)):
        raise ValueError("delta contains non-finite values")
    manifest_path = input_path.with_name("pq_gap_provenance.json")
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if int(manifest.get("records", len(labels))) != len(labels):
            raise ValueError("p/q provenance record count disagrees with cache")
        if int(manifest.get("tokens", len(delta))) != len(delta):
            raise ValueError("p/q provenance token count disagrees with cache")
    return DeltaArchive(
        labels=labels,
        record_ids=record_ids,
        lengths=lengths,
        offsets=offsets,
        delta=delta,
        target=target,
        draft=draft,
        input_path=input_path,
        scores_path=scores_path,
    )




def partition_manifest(
    labels: np.ndarray,
    record_ids: np.ndarray,
    partitions: dict[str, np.ndarray],
    seed: int,
) -> dict[str, Any]:
    owner = np.full(len(labels), "", dtype=object)
    for name, indices in partitions.items():
        if np.any(owner[indices] != ""):
            raise ValueError("partitions overlap")
        owner[indices] = name
    if np.any(owner == ""):
        raise ValueError("partitions do not cover every record")
    return {
        "split_seed": seed,
        "split_unit": "record",
        "counts": {
            name: {
                "member": int(np.sum(labels[indices] == 1)),
                "nonmember": int(np.sum(labels[indices] == 0)),
            }
            for name, indices in partitions.items()
        },
        "partitions": {
            name: [str(record_ids[index]) for index in indices]
            for name, indices in partitions.items()
        },
    }


def materialize_full_delta(
    output_dir: Path,
    archive: DeltaArchive,
    partitions: dict[str, np.ndarray],
    split_seed: int,
    eos_policy: str = "as cached; historical cache EOS flag unspecified",
) -> None:
    """Persist the full-delta contract and the shared record-ID split."""

    output_dir.mkdir(parents=True, exist_ok=True)
    archive_path = output_dir / "full_delta.npz"
    if not archive_path.exists():
        fd, temporary = tempfile.mkstemp(prefix=".full_delta.", suffix=".npz", dir=output_dir)
        os.close(fd)
        try:
            np.savez_compressed(
                temporary,
                labels=archive.labels,
                record_ids=archive.record_ids,
                lengths=archive.lengths,
                offsets=archive.offsets,
                delta=archive.delta,
            )
            os.replace(temporary, archive_path)
        finally:
            if os.path.exists(temporary):
                os.unlink(temporary)
    _write_json_atomic(
        output_dir / "full_delta_protocol.json",
        {
            "source_input": str(archive.input_path),
            "source_input_sha256": _sha256_file(archive.input_path),
            "source_scores": str(archive.scores_path),
            "source_scores_sha256": _sha256_file(archive.scores_path),
            "records": int(len(archive.labels)),
            "tokens": int(len(archive.delta)),
            "input_contract": "delta only; no p, q, token ID or activation/H",
            "delta_definition": "target_logp - draft_logq",
            "eos_policy": eos_policy,
            "partition": partition_manifest(archive.labels, archive.record_ids, partitions, split_seed),
        },
    )


def drop_final_token(archive: DeltaArchive) -> DeltaArchive:
    """Return a copy without each record's final cached token.

    The historical probability cache does not record EOS metadata. Its
    producer appends EOS to every response, so this is the registered no-EOS
    sensitivity check, explicitly described as dropping the final token.
    """

    if np.any(archive.lengths <= 1):
        raise ValueError("cannot drop the final token from a one-token record")
    pieces_delta: list[np.ndarray] = []
    pieces_target: list[np.ndarray] = []
    pieces_draft: list[np.ndarray] = []
    for start, end in zip(archive.offsets[:-1], archive.offsets[1:]):
        start_i, end_i = int(start), int(end)
        pieces_delta.append(archive.delta[start_i : end_i - 1])
        if archive.target is not None:
            pieces_target.append(archive.target[start_i : end_i - 1])
        if archive.draft is not None:
            pieces_draft.append(archive.draft[start_i : end_i - 1])
    lengths = archive.lengths - 1
    offsets = np.concatenate(([0], np.cumsum(lengths, dtype=np.int64)))
    return DeltaArchive(
        labels=archive.labels,
        record_ids=archive.record_ids,
        lengths=lengths,
        offsets=offsets,
        delta=np.concatenate(pieces_delta).astype(np.float32, copy=False),
        target=np.concatenate(pieces_target).astype(np.float32, copy=False) if pieces_target else None,
        draft=np.concatenate(pieces_draft).astype(np.float32, copy=False) if pieces_draft else None,
        input_path=archive.input_path,
        scores_path=archive.scores_path,
    )








def _operating_point(member: np.ndarray, nonmember: np.ndarray, calibration: np.ndarray, rate: float) -> dict[str, Any]:
    threshold = order_statistic_threshold(calibration, rate)
    member_p = conformal_tail_pvalues(member, calibration)
    nonmember_p = conformal_tail_pvalues(nonmember, calibration)
    member_hits = int(np.sum(member_p <= rate))
    nonmember_hits = int(np.sum(nonmember_p <= rate))
    return {
        "threshold": float(threshold),
        "tpr": float(member_hits / len(member)),
        "actual_fpr": float(nonmember_hits / len(nonmember)),
        "member_hits": member_hits,
        "nonmember_hits": nonmember_hits,
        "member_n": int(len(member)),
        "nonmember_n": int(len(nonmember)),
        "calibration_n_nonmember": int(len(calibration)),
    }


def _metric_point(scores: np.ndarray, labels: np.ndarray, partitions: dict[str, np.ndarray]) -> dict[str, Any]:
    validation = partitions["V"]
    test = partitions["T"]
    calibration_nonmember = partitions["C"][labels[partitions["C"]] == 0]
    validation_member = validation[labels[validation] == 1]
    validation_nonmember = validation[labels[validation] == 0]
    test_member = test[labels[test] == 1]
    test_nonmember = test[labels[test] == 0]
    test_labels = np.r_[np.ones(len(test_member), dtype=np.int64), np.zeros(len(test_nonmember), dtype=np.int64)]
    test_scores = np.r_[scores[test_member], scores[test_nonmember]]
    return {
        "validation_pauc_0_10": partial_auc(
            np.r_[scores[validation_member], scores[validation_nonmember]],
            np.r_[np.ones(len(validation_member), dtype=np.int64), np.zeros(len(validation_nonmember), dtype=np.int64)],
        ),
        "test": {
            "auc": rank_auc(scores[test_member], scores[test_nonmember]),
            "pauc_0_10": partial_auc(test_scores, test_labels),
            "tpr_at_fpr": {f"{int(rate * 100)}%": _operating_point(scores[test_member], scores[test_nonmember], scores[calibration_nonmember], rate) for rate in RATES},
        },
    }


def _bootstrap_metrics(
    scores: np.ndarray,
    labels: np.ndarray,
    partitions: dict[str, np.ndarray],
    repeats: int,
    seed: int,
) -> dict[str, Any]:
    test = partitions["T"]
    calibration = partitions["C"]
    members = scores[test[labels[test] == 1]]
    nonmembers = scores[test[labels[test] == 0]]
    calibration = scores[calibration[labels[calibration] == 0]]
    rng = np.random.default_rng(seed)
    values: dict[str, list[float]] = {"auc": [], "pauc_0_10": []}
    values.update({f"tpr_{int(rate * 100)}": [] for rate in RATES})
    values.update({f"fpr_{int(rate * 100)}": [] for rate in RATES})
    for _ in range(repeats):
        member = members[rng.integers(0, len(members), len(members))]
        nonmember = nonmembers[rng.integers(0, len(nonmembers), len(nonmembers))]
        cal = calibration[rng.integers(0, len(calibration), len(calibration))]
        values["auc"].append(rank_auc(member, nonmember))
        values["pauc_0_10"].append(partial_auc(np.r_[member, nonmember], np.r_[np.ones(len(member)), np.zeros(len(nonmember))]))
        for rate in RATES:
            point = _operating_point(member, nonmember, cal, rate)
            values[f"tpr_{int(rate * 100)}"].append(point["tpr"])
            values[f"fpr_{int(rate * 100)}"].append(point["actual_fpr"])
    point = _metric_point(scores, labels, partitions)
    result: dict[str, Any] = {
        "auc": {"point": point["test"]["auc"]},
        "pauc_0_10": {"point": point["test"]["pauc_0_10"]},
        "tpr_at_fpr": {},
    }
    for name in ("auc", "pauc_0_10"):
        result[name].update({"ci95_low": float(np.quantile(values[name], 0.025)), "ci95_high": float(np.quantile(values[name], 0.975))})
    for rate in RATES:
        key = f"{int(rate * 100)}%"
        result["tpr_at_fpr"][key] = {
            "tpr": {"point": point["test"]["tpr_at_fpr"][key]["tpr"], "ci95_low": float(np.quantile(values[f"tpr_{int(rate * 100)}"], 0.025)), "ci95_high": float(np.quantile(values[f"tpr_{int(rate * 100)}"], 0.975))},
            "actual_fpr": {"point": point["test"]["tpr_at_fpr"][key]["actual_fpr"], "ci95_low": float(np.quantile(values[f"fpr_{int(rate * 100)}"], 0.025)), "ci95_high": float(np.quantile(values[f"fpr_{int(rate * 100)}"], 0.975))},
        }
    return {"validation_pauc_0_10": point["validation_pauc_0_10"], "test": result}


class SequenceDataset(Dataset[tuple[torch.Tensor, int]]):
    def __init__(self, sequences: list[np.ndarray], labels: np.ndarray) -> None:
        self.sequences = sequences
        self.labels = np.asarray(labels, dtype=np.int64)

    def __len__(self) -> int:
        return len(self.sequences)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, int]:
        return torch.from_numpy(self.sequences[index]), int(self.labels[index])


def collate_sequences(batch: list[tuple[torch.Tensor, int]]) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if not batch:
        raise ValueError("cannot collate an empty batch")
    order = sorted(range(len(batch)), key=lambda index: batch[index][0].numel(), reverse=True)
    values = [batch[index][0] for index in order]
    labels = torch.tensor([batch[index][1] for index in order], dtype=torch.float32)
    lengths = torch.tensor([len(value) for value in values], dtype=torch.long)
    padded = torch.zeros((len(values), int(lengths.max())), dtype=torch.float32)
    mask = torch.zeros_like(padded, dtype=torch.bool)
    for row, value in enumerate(values):
        padded[row, : len(value)] = value
        mask[row, : len(value)] = True
    return padded.unsqueeze(-1), mask, labels


def transform_sequences(
    delta: np.ndarray,
    lengths: np.ndarray,
    mode: str = "normal",
    seed: int = DEFAULT_TRAINING_SEEDS[0],
) -> tuple[list[np.ndarray], np.ndarray]:
    offsets = np.concatenate(([0], np.cumsum(lengths, dtype=np.int64)))
    result: list[np.ndarray] = []
    transformed_lengths: list[int] = []
    for index, (start, end) in enumerate(zip(offsets[:-1], offsets[1:])):
        value = np.asarray(delta[int(start) : int(end)], dtype=np.float32).copy()
        if mode == "shuffle":
            np.random.default_rng([seed, index]).shuffle(value)
        elif mode == "reverse":
            value = value[::-1].copy()
        elif mode in ("first_half", "second_half"):
            midpoint = max(1, len(value) // 2)
            value = value[:midpoint] if mode == "first_half" else value[-midpoint:]
            value = value.copy()
        elif mode != "normal":
            raise ValueError(f"unknown sequence transform {mode!r}")
        result.append(value)
        transformed_lengths.append(len(value))
    return result, np.asarray(transformed_lengths, dtype=np.int64)


def fit_token_standardizer(sequences: list[np.ndarray], indices: np.ndarray) -> tuple[float, float]:
    values = np.concatenate([sequences[int(index)] for index in indices]).astype(np.float64)
    mean, std = float(values.mean()), float(values.std(ddof=0))
    return mean, max(std, 1e-8)


def standardize_sequences(sequences: list[np.ndarray], mean: float, std: float, mode: str) -> list[np.ndarray]:
    if mode == "raw":
        return [np.asarray(value, dtype=np.float32) for value in sequences]
    if mode != "d":
        raise ValueError(f"unknown standardization mode {mode!r}")
    return [((value.astype(np.float64) - mean) / std).astype(np.float32) for value in sequences]


def masked_mean(values: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    weights = mask.unsqueeze(-1).to(values.dtype)
    return (values * weights).sum(dim=1) / weights.sum(dim=1).clamp_min(1.0)


class TokenPoolNet(nn.Module):
    def __init__(self, hidden: int, dropout: float, attention: bool, include_length: bool) -> None:
        super().__init__()
        self.token = nn.Sequential(nn.Linear(1, hidden), nn.GELU(), nn.Dropout(dropout))
        self.attention = nn.Linear(hidden, 1) if attention else None
        self.head = nn.Linear(hidden + int(include_length), 1)
        self.include_length = include_length

    def forward(self, values: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        hidden = self.token(values)
        if self.attention is None:
            pooled = masked_mean(hidden, mask)
        else:
            weights = self.attention(hidden).squeeze(-1).masked_fill(~mask, -torch.inf)
            weights = torch.softmax(weights, dim=1).unsqueeze(-1)
            pooled = (hidden * weights).sum(dim=1)
        if self.include_length:
            length = mask.sum(dim=1, keepdim=True).float().log1p()
            pooled = torch.cat((pooled, length), dim=1)
        return self.head(pooled).squeeze(-1)


class TCNBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel: int, dilation: int, dropout: float) -> None:
        super().__init__()
        padding = dilation * (kernel - 1) // 2
        self.conv = nn.Conv1d(in_channels, out_channels, kernel, padding=padding, dilation=dilation)
        self.norm = nn.GroupNorm(1, out_channels)
        self.dropout = nn.Dropout(dropout)
        self.residual = nn.Conv1d(in_channels, out_channels, 1) if in_channels != out_channels else nn.Identity()

    def forward(self, values: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        output = self.conv(values)
        output = F.gelu(self.norm(output))
        output = self.dropout(output)
        output = output + self.residual(values)
        return output * mask.unsqueeze(1).to(output.dtype)


class DeltaTCN(nn.Module):
    def __init__(self, channels: int, kernel: int, dropout: float, include_length: bool) -> None:
        super().__init__()
        blocks: list[nn.Module] = []
        in_channels = 1
        for dilation in (1, 2, 4, 8):
            blocks.append(TCNBlock(in_channels, channels, kernel, dilation, dropout))
            in_channels = channels
        self.blocks = nn.ModuleList(blocks)
        self.head = nn.Linear(channels * 2 + int(include_length), 1)
        self.include_length = include_length

    def forward(self, values: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        hidden = values.transpose(1, 2)
        for block in self.blocks:
            hidden = block(hidden, mask)
        valid = mask.unsqueeze(1).to(hidden.dtype)
        mean = (hidden * valid).sum(dim=2) / valid.sum(dim=2).clamp_min(1.0)
        maximum = hidden.masked_fill(~mask.unsqueeze(1), -torch.inf).amax(dim=2)
        pooled = torch.cat((mean, maximum), dim=1)
        if self.include_length:
            pooled = torch.cat((pooled, mask.sum(dim=1, keepdim=True).float().log1p()), dim=1)
        return self.head(pooled).squeeze(-1)


def sinusoidal_positions(length: int, hidden: int) -> torch.Tensor:
    position = torch.arange(length, dtype=torch.float32).unsqueeze(1)
    divisor = torch.exp(torch.arange(0, hidden, 2, dtype=torch.float32) * (-math.log(10000.0) / hidden))
    result = torch.zeros((length, hidden), dtype=torch.float32)
    result[:, 0::2] = torch.sin(position * divisor)
    result[:, 1::2] = torch.cos(position * divisor[: result[:, 1::2].shape[1]])
    return result


class DeltaTransformer(nn.Module):
    def __init__(self, hidden: int, layers: int, dropout: float, include_length: bool, max_length: int = 4096) -> None:
        super().__init__()
        self.projection = nn.Linear(1, hidden)
        layer = nn.TransformerEncoderLayer(
            d_model=hidden,
            nhead=4,
            dim_feedforward=hidden * 4,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=layers)
        self.register_buffer("positions", sinusoidal_positions(max_length, hidden), persistent=False)
        self.head = nn.Linear(hidden + int(include_length), 1)
        self.include_length = include_length

    def forward(self, values: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        if values.shape[1] > self.positions.shape[0]:
            raise ValueError("sequence exceeds transformer positional buffer")
        hidden = self.projection(values) + self.positions[: values.shape[1]].unsqueeze(0)
        hidden = self.encoder(hidden, src_key_padding_mask=~mask)
        pooled = masked_mean(hidden, mask)
        if self.include_length:
            pooled = torch.cat((pooled, mask.sum(dim=1, keepdim=True).float().log1p()), dim=1)
        return self.head(pooled).squeeze(-1)


def make_model(method: str, spec: dict[str, Any], include_length: bool, max_length: int) -> nn.Module:
    if method == "delta-uniformnet":
        return TokenPoolNet(int(spec["hidden"]), float(spec["dropout"]), False, include_length)
    if method == "delta-attn":
        return TokenPoolNet(int(spec["hidden"]), float(spec["dropout"]), True, include_length)
    if method == "delta-tcn":
        return DeltaTCN(int(spec["channels"]), int(spec["kernel"]), float(spec["dropout"]), include_length)
    if method == "delta-transformer":
        return DeltaTransformer(int(spec["hidden"]), int(spec["layers"]), float(spec["dropout"]), include_length, max_length=max_length)
    raise ValueError(f"no neural model for {method!r}")


def candidate_specs(method: str) -> list[dict[str, Any]]:
    common = {
        "lr": (1e-3, 3e-4),
        "weight_decay": (1e-3, 1e-2),
        "dropout": (0.0, 0.1),
    }
    if method in ("delta-uniformnet", "delta-attn"):
        grid = {**common, "hidden": (32, 64)}
    elif method == "delta-tcn":
        grid = {**common, "channels": (32, 64), "kernel": (3, 5)}
    elif method == "delta-transformer":
        grid = {**common, "hidden": (64, 128), "layers": (2, 3)}
    else:
        raise ValueError(f"unknown learned method {method!r}")
    keys = tuple(grid)
    return [dict(zip(keys, values)) for values in itertools.product(*(grid[key] for key in keys))]


def set_seed(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _loader(dataset: Dataset, indices: np.ndarray, batch_size: int, shuffle: bool, seed: int, device: torch.device) -> DataLoader:
    generator = torch.Generator()
    generator.manual_seed(seed)
    return DataLoader(
        Subset(dataset, [int(index) for index in indices]),
        batch_size=batch_size,
        shuffle=shuffle,
        generator=generator,
        collate_fn=collate_sequences,
        num_workers=0,
        # These batches are small relative to an A100.  Pinning every
        # variable-length sample was more expensive than the host-to-device
        # copy and caused severe CPU oversubscription when seven jobs ran.
        pin_memory=False,
    )


@torch.inference_mode()
def predict(model: nn.Module, dataset: Dataset, indices: np.ndarray, batch_size: int, device: torch.device) -> np.ndarray:
    model.eval()
    values: list[np.ndarray] = []
    for batch_values, mask, _labels in _loader(dataset, indices, batch_size, False, 0, device):
        values.append(model(batch_values.to(device, non_blocking=True), mask.to(device, non_blocking=True)).cpu().numpy())
    return np.concatenate(values) if values else np.empty(0, dtype=np.float64)


def fit_neural_candidate(
    method: str,
    spec: dict[str, Any],
    dataset: SequenceDataset,
    labels: np.ndarray,
    train_indices: np.ndarray,
    validation_indices: np.ndarray,
    seed: int,
    device: torch.device,
    batch_size: int,
    max_epochs: int,
    patience: int,
    include_length: bool,
    max_length: int,
) -> tuple[nn.Module, float, int]:
    set_seed(seed)
    model = make_model(method, spec, include_length, max_length).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(spec["lr"]), weight_decay=float(spec["weight_decay"]))
    train_labels = labels[train_indices]
    class_counts = {0: max(1, int(np.sum(train_labels == 0))), 1: max(1, int(np.sum(train_labels == 1)))}
    weights = {label: 1.0 / (2.0 * count) for label, count in class_counts.items()}
    loader = _loader(dataset, train_indices, batch_size, True, seed, device)
    best_score = -np.inf
    best_epoch = 0
    best_state: dict[str, torch.Tensor] | None = None
    stale = 0
    for epoch in range(1, max_epochs + 1):
        model.train()
        for values, mask, batch_labels in loader:
            values = values.to(device, non_blocking=True)
            mask = mask.to(device, non_blocking=True)
            batch_labels = batch_labels.to(device, non_blocking=True)
            logits = model(values, mask)
            per_sample = nn.functional.binary_cross_entropy_with_logits(logits, batch_labels, reduction="none")
            batch_weights = torch.tensor([weights[int(label)] for label in batch_labels.cpu().numpy()], device=device, dtype=per_sample.dtype)
            loss = (per_sample * batch_weights).sum() / batch_weights.sum().clamp_min(1e-12)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
        validation_scores = predict(model, dataset, validation_indices, batch_size, device)
        validation_pauc = partial_auc(validation_scores, labels[validation_indices])
        if validation_pauc > best_score + 1e-10:
            best_score = validation_pauc
            best_epoch = epoch
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
            stale = 0
        else:
            stale += 1
            if stale >= patience:
                break
    if best_state is None:
        raise RuntimeError("neural candidate did not produce a validation checkpoint")
    model.load_state_dict(best_state)
    model.to(device)
    return model, float(best_score), best_epoch


@dataclass
class Standardizer:
    mean: np.ndarray
    std: np.ndarray

    def transform(self, values: np.ndarray) -> np.ndarray:
        return (np.asarray(values, dtype=np.float64) - self.mean) / self.std


def fit_row_standardizer(values: np.ndarray, indices: np.ndarray) -> Standardizer:
    matrix = np.asarray(values, dtype=np.float64)
    mean = matrix[indices].mean(axis=0)
    std = matrix[indices].std(axis=0, ddof=0)
    return Standardizer(mean=mean, std=np.where(std < 1e-8, 1.0, std))


def fit_logistic(values: np.ndarray, labels: np.ndarray, indices: np.ndarray, l2: float) -> tuple[np.ndarray, float]:
    x = np.asarray(values[indices], dtype=np.float64)
    y = np.asarray(labels[indices], dtype=np.float64)
    weights = np.where(y == 1.0, 1.0 / (2.0 * max(1, int(np.sum(y == 1)))), 1.0 / (2.0 * max(1, int(np.sum(y == 0)))))
    def sigmoid(value: np.ndarray) -> np.ndarray:
        return 1.0 / (1.0 + np.exp(-np.clip(value, -60.0, 60.0)))
    def objective(params: np.ndarray) -> tuple[float, np.ndarray]:
        score = x @ params[:-1] + params[-1]
        probability = sigmoid(score)
        gradient_score = weights * (probability - y)
        loss = float(np.sum(weights * np.logaddexp(0.0, score) - weights * y * score) + l2 * np.sum(params[:-1] ** 2))
        gradient = np.r_[x.T @ gradient_score + 2.0 * l2 * params[:-1], np.sum(gradient_score)]
        return loss, gradient
    result = minimize(objective, np.zeros(x.shape[1] + 1), jac=True, method="L-BFGS-B", options={"maxiter": 500, "ftol": 1e-12, "gtol": 1e-8})
    if not result.success:
        raise RuntimeError(f"logistic fit failed: {result.message}")
    return result.x[:-1], float(result.x[-1])


def b2_values(archive: DeltaArchive) -> np.ndarray:
    if archive.target is None or archive.draft is None:
        raise ValueError("B2 requires target and draft probabilities")
    rows = []
    for start, end in zip(archive.offsets[:-1], archive.offsets[1:]):
        rows.append(record_features(archive.target[int(start):int(end)], archive.draft[int(start):int(end)]))
    names = tuple(rows[0])
    f19 = np.asarray([[row[name] for name in names] for row in rows], dtype=np.float64)
    return np.concatenate((f19, aggregate_matrix(archive.delta, archive.lengths)), axis=1)


def run_b2(archive: DeltaArchive, partitions: dict[str, np.ndarray], bootstrap_repeats: int, seed: int) -> dict[str, Any]:
    values = b2_values(archive)
    scaler = fit_row_standardizer(values, partitions["D"])
    standardized = scaler.transform(values)
    candidates = []
    for l2 in (1e-3, 1e-2, 1e-1, 1.0):
        weight, bias = fit_logistic(standardized, archive.labels, partitions["D"], l2)
        scores = standardized @ weight + bias
        validation = _metric_point(scores, archive.labels, partitions)["validation_pauc_0_10"]
        candidates.append({"l2": l2, "validation_pauc_0_10": validation, "weight": weight, "bias": bias, "scores": scores})
    selected = max(candidates, key=lambda item: (item["validation_pauc_0_10"], -float(item["l2"])))
    metrics = _bootstrap_metrics(selected["scores"], archive.labels, partitions, bootstrap_repeats, seed + 3000)
    return {
        "method": "B2",
        "permission_label": "legacy P/Q-stat control; not delta-only",
        "selected_config": {"l2": selected["l2"]},
        "candidate_table": [{"l2": row["l2"], "validation_pauc_0_10": row["validation_pauc_0_10"]} for row in candidates],
        "standardizer": {"mean": scaler.mean, "std": scaler.std},
        "scores": selected["scores"],
        "metrics": metrics,
        "parameter_count": int(values.shape[1] + 1),
    }


def run_delta_mean(archive: DeltaArchive, partitions: dict[str, np.ndarray], bootstrap_repeats: int, seed: int) -> dict[str, Any]:
    offsets = archive.offsets
    scores = np.asarray([archive.delta[int(start):int(end)].mean() for start, end in zip(offsets[:-1], offsets[1:])], dtype=np.float64)
    return {
        "method": "Delta-Mean",
        "permission_label": "delta-only",
        "selected_config": {},
        "candidate_table": [],
        "scores": scores,
        "metrics": _bootstrap_metrics(scores, archive.labels, partitions, bootstrap_repeats, seed + 3001),
        "parameter_count": 0,
    }


def run_neural(
    method: str,
    archive: DeltaArchive,
    partitions: dict[str, np.ndarray],
    output_dir: Path,
    training_seeds: tuple[int, ...],
    standardization: str,
    transform: str,
    include_length: bool,
    device: torch.device,
    batch_size: int,
    max_epochs: int,
    patience: int,
    bootstrap_repeats: int,
    split_seed: int,
    candidate_indices: Iterable[int] | None = None,
    candidate_only: bool = False,
    fixed_config: dict[str, Any] | None = None,
) -> dict[str, Any]:
    raw_sequences, transformed_lengths = transform_sequences(archive.delta, archive.lengths, transform, seed=training_seeds[0])
    if standardization == "d":
        mean, std = fit_token_standardizer(raw_sequences, partitions["D"])
    else:
        mean, std = 0.0, 1.0
    sequences = standardize_sequences(raw_sequences, mean, std, standardization)
    dataset = SequenceDataset(sequences, archive.labels)
    max_length = int(max(transformed_lengths))
    candidates = [dict(fixed_config)] if fixed_config is not None else candidate_specs(method)
    candidate_table: list[dict[str, Any]] = []
    candidate_states: dict[int, dict[int, dict[str, torch.Tensor]]] = {}
    selected_indices = list(range(len(candidates))) if candidate_indices is None else [int(index) for index in candidate_indices]
    if not selected_indices or any(index < 0 or index >= len(candidates) for index in selected_indices):
        raise ValueError("candidate_indices must select at least one valid candidate")
    for candidate_index in selected_indices:
        spec = candidates[candidate_index]
        seed_results = []
        candidate_states[candidate_index] = {}
        for train_seed in training_seeds:
            started = time.perf_counter()
            model, validation_pauc, best_epoch = fit_neural_candidate(
                method,
                spec,
                dataset,
                archive.labels,
                partitions["D"],
                partitions["V"],
                train_seed,
                device,
                batch_size,
                max_epochs,
                patience,
                include_length,
                max_length,
            )
            candidate_states[candidate_index][train_seed] = {
                key: value.detach().cpu().clone() for key, value in model.state_dict().items()
            }
            seed_results.append({"seed": train_seed, "validation_pauc_0_10": validation_pauc, "best_epoch": best_epoch, "seconds": time.perf_counter() - started})
            del model
            if device.type == "cuda":
                torch.cuda.empty_cache()
        mean_validation = float(np.mean([row["validation_pauc_0_10"] for row in seed_results]))
        std_validation = float(np.std([row["validation_pauc_0_10"] for row in seed_results], ddof=0))
        candidate_table.append({"candidate_index": candidate_index, "config": spec, "seed_results": seed_results, "validation_pauc_mean": mean_validation, "validation_pauc_std": std_validation})
        print(f"{method} candidate {candidate_index + 1}/{len(candidates)} V-pAUC={mean_validation:.4f}±{std_validation:.4f}", flush=True)
    if candidate_only:
        candidate_model_dir = output_dir / "candidate_models"
        candidate_model_dir.mkdir(parents=True, exist_ok=True)
        for row in candidate_table:
            paths: dict[str, str] = {}
            for train_seed in training_seeds:
                path = candidate_model_dir / f"{method.replace('-', '_')}_candidate{row['candidate_index']}_seed{train_seed}.pt"
                torch.save(
                    {
                        "method": method,
                        "config": row["config"],
                        "seed": train_seed,
                        "standardization": {"mode": standardization, "mean": mean, "std": std},
                        "transform": transform,
                        "include_length": include_length,
                        "state_dict": candidate_states[row["candidate_index"]][train_seed],
                    },
                    path,
                )
                paths[str(train_seed)] = str(path.resolve())
            row["checkpoint_paths"] = paths
        start, end = min(selected_indices), max(selected_indices) + 1
        shard_path = output_dir / "methods" / f"candidate_shard_{start}_{end}.json"
        return {
            "method": method,
            "permission_label": "delta-only",
            "candidate_start": start,
            "candidate_end": end,
            "candidate_indices": selected_indices,
            "candidate_table": candidate_table,
            "standardization": {"mode": standardization, "mean": mean, "std": std},
            "sequence_transform": transform,
            "include_length": include_length,
            "training_seeds": list(training_seeds),
            "shard_path": str(shard_path.resolve()),
            "_report_name": shard_path.name,
        }
    selected_row = max(candidate_table, key=lambda row: (row["validation_pauc_mean"], -row["validation_pauc_std"], -sum(int(value) for value in row["config"].values() if isinstance(value, int))))
    selected_spec = dict(selected_row["config"])
    score_columns: list[np.ndarray] = []
    seed_metrics: dict[str, Any] = {}
    model_dir = output_dir / "models"
    model_dir.mkdir(parents=True, exist_ok=True)
    for train_seed in training_seeds:
        model = make_model(method, selected_spec, include_length, max_length).to(device)
        model.load_state_dict(candidate_states[int(selected_row["candidate_index"])][train_seed])
        validation_pauc = float(next(item["validation_pauc_0_10"] for item in selected_row["seed_results"] if item["seed"] == train_seed))
        best_epoch = int(next(item["best_epoch"] for item in selected_row["seed_results"] if item["seed"] == train_seed))
        scores = predict(model, dataset, np.arange(len(archive.labels), dtype=np.int64), batch_size, device).astype(np.float64)
        score_columns.append(scores)
        seed_metrics[str(train_seed)] = {
            "validation_pauc_0_10": validation_pauc,
            "best_epoch": best_epoch,
            "metrics": _bootstrap_metrics(scores, archive.labels, partitions, bootstrap_repeats, split_seed + 4000 + train_seed % 1000),
            "parameter_count": int(sum(parameter.numel() for parameter in model.parameters())),
        }
        torch.save({"method": method, "config": selected_spec, "seed": train_seed, "standardization": {"mode": standardization, "mean": mean, "std": std}, "transform": transform, "include_length": include_length, "state_dict": model.state_dict()}, model_dir / f"{method.replace('-', '_')}_seed{train_seed}.pt")
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()
    score_matrix = np.column_stack(score_columns)
    scores_path = method_scores_path(output_dir, method)
    np.savez_compressed(scores_path, labels=archive.labels, record_ids=archive.record_ids, seeds=np.asarray(training_seeds, dtype=np.int64), scores=score_matrix)
    return {
        "method": method,
        "permission_label": "delta-only",
        "selected_config": selected_spec,
        "candidate_count": len(candidate_table),
        "candidate_table": candidate_table,
        "seed_metrics": seed_metrics,
        "scores_path": str(scores_path),
        "standardization": {"mode": standardization, "mean": mean, "std": std},
        "sequence_transform": transform,
        "transformed_length_min": int(transformed_lengths.min()),
        "transformed_length_max": int(transformed_lengths.max()),
        "include_length": include_length,
        "training_seeds": list(training_seeds),
    }


def finalize_neural(
    method: str,
    archive: DeltaArchive,
    partitions: dict[str, np.ndarray],
    output_dir: Path,
    training_seeds: tuple[int, ...],
    standardization: str,
    transform: str,
    include_length: bool,
    device: torch.device,
    batch_size: int,
    bootstrap_repeats: int,
    split_seed: int,
) -> dict[str, Any]:
    """Select across candidate shards and evaluate the selected models."""

    candidates = candidate_specs(method)
    shard_paths = sorted((output_dir / "methods").glob("candidate_shard_*.json"))
    if not shard_paths:
        raise FileNotFoundError(f"no candidate shards under {output_dir / 'methods'}")
    candidate_table: list[dict[str, Any]] = []
    for path in shard_paths:
        shard = json.loads(path.read_text(encoding="utf-8"))
        if shard.get("method") != method:
            continue
        candidate_table.extend(shard.get("candidate_table", []))
    candidate_table.sort(key=lambda row: int(row["candidate_index"]))
    indices = [int(row["candidate_index"]) for row in candidate_table]
    if indices != list(range(len(candidates))):
        raise RuntimeError(f"candidate shards do not cover exactly {len(candidates)} candidates: {indices}")
    selected_row = max(
        candidate_table,
        key=lambda row: (
            row["validation_pauc_mean"],
            -row["validation_pauc_std"],
            -sum(int(value) for value in row["config"].values() if isinstance(value, int)),
        ),
    )
    raw_sequences, transformed_lengths = transform_sequences(
        archive.delta, archive.lengths, transform, seed=training_seeds[0]
    )
    if standardization == "d":
        mean, std = fit_token_standardizer(raw_sequences, partitions["D"])
    else:
        mean, std = 0.0, 1.0
    sequences = standardize_sequences(raw_sequences, mean, std, standardization)
    dataset = SequenceDataset(sequences, archive.labels)
    max_length = int(max(transformed_lengths))
    score_columns: list[np.ndarray] = []
    seed_metrics: dict[str, Any] = {}
    model_dir = output_dir / "models"
    model_dir.mkdir(parents=True, exist_ok=True)
    for train_seed in training_seeds:
        checkpoint = Path(selected_row["checkpoint_paths"][str(train_seed)])
        payload = torch.load(checkpoint, map_location="cpu")
        model = make_model(method, dict(selected_row["config"]), include_length, max_length).to(device)
        model.load_state_dict(payload["state_dict"])
        scores = predict(
            model,
            dataset,
            np.arange(len(archive.labels), dtype=np.int64),
            batch_size,
            device,
        ).astype(np.float64)
        score_columns.append(scores)
        validation_pauc = float(
            next(item["validation_pauc_0_10"] for item in selected_row["seed_results"] if item["seed"] == train_seed)
        )
        best_epoch = int(next(item["best_epoch"] for item in selected_row["seed_results"] if item["seed"] == train_seed))
        seed_metrics[str(train_seed)] = {
            "validation_pauc_0_10": validation_pauc,
            "best_epoch": best_epoch,
            "metrics": _bootstrap_metrics(
                scores,
                archive.labels,
                partitions,
                bootstrap_repeats,
                split_seed + 4000 + train_seed % 1000,
            ),
            "parameter_count": int(sum(parameter.numel() for parameter in model.parameters())),
        }
        torch.save(
            {
                "method": method,
                "config": selected_row["config"],
                "seed": train_seed,
                "standardization": {"mode": standardization, "mean": mean, "std": std},
                "transform": transform,
                "include_length": include_length,
                "state_dict": model.state_dict(),
            },
            model_dir / f"{method.replace('-', '_')}_seed{train_seed}.pt",
        )
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()
    scores_path = method_scores_path(output_dir, method)
    np.savez_compressed(
        scores_path,
        labels=archive.labels,
        record_ids=archive.record_ids,
        seeds=np.asarray(training_seeds, dtype=np.int64),
        scores=np.column_stack(score_columns),
    )
    return {
        "method": method,
        "permission_label": "delta-only",
        "selected_config": dict(selected_row["config"]),
        "candidate_count": len(candidate_table),
        "candidate_table": candidate_table,
        "seed_metrics": seed_metrics,
        "scores_path": str(scores_path),
        "standardization": {"mode": standardization, "mean": mean, "std": std},
        "sequence_transform": transform,
        "transformed_length_min": int(transformed_lengths.min()),
        "transformed_length_max": int(transformed_lengths.max()),
        "include_length": include_length,
        "training_seeds": list(training_seeds),
        "selection_source": [str(path.resolve()) for path in shard_paths],
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--benchmark", required=True, choices=("wikitection", "newstection", "arxivtection"))
    parser.add_argument("--epoch", type=int, required=True)
    parser.add_argument("--role", default="draft_auxiliary_distilled", choices=("draft_auxiliary_distilled", "draft_member_sft"))
    parser.add_argument("--method", required=True, choices=METHODS)
    parser.add_argument("--input", type=Path)
    parser.add_argument("--scores", type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--split-seed", type=int, default=DEFAULT_SPLIT_SEED)
    parser.add_argument("--training-seeds", default=",".join(str(seed) for seed in DEFAULT_TRAINING_SEEDS))
    parser.add_argument("--standardization", choices=("d", "raw"), default="d")
    parser.add_argument("--transform", choices=("normal", "shuffle", "reverse", "first_half", "second_half"), default="normal")
    parser.add_argument("--include-length", action="store_true")
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--cpu", action="store_true")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--max-epochs", type=int, default=100)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--bootstrap-repeats", type=int, default=1000)
    parser.add_argument("--candidate-start", type=int, default=None)
    parser.add_argument("--candidate-end", type=int, default=None)
    parser.add_argument("--candidate-only", action="store_true")
    parser.add_argument("--finalize", action="store_true")
    parser.add_argument(
        "--fixed-config",
        help="JSON hyperparameter object for a selected-config ablation; skips the V grid search",
    )
    parser.add_argument(
        "--drop-eos",
        action="store_true",
        help="drop each record's final cached token for the no-EOS sensitivity check",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    # One CPU thread per job leaves the host available for seven independent
    # GPU workers; all heavy tensor work remains on the selected A100.
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    input_path = args.input or Path(f"experiments/results/sft_runs/pq_directional/{args.benchmark}_epoch{args.epoch}/pq_gap_token_logps.npz")
    input_path = input_path if input_path.is_absolute() else ROOT / input_path
    scores_path = args.scores
    if scores_path is not None and not scores_path.is_absolute():
        scores_path = ROOT / scores_path
    archive = load_delta_archive(input_path, scores_path, args.role)
    eos_policy = "as cached; historical cache EOS flag unspecified"
    if args.drop_eos:
        archive = drop_final_token(archive)
        eos_policy = "drop final cached token (registered no-EOS proxy; cache EOS flag unspecified)"
    partitions = split_indices(archive.labels, args.split_seed)
    output_dir = args.output_dir or Path(f"experiments/results/sft_runs/full_delta/{args.benchmark}_epoch{args.epoch}/{args.role}")
    output_dir = output_dir if output_dir.is_absolute() else ROOT / output_dir
    materialize_full_delta(output_dir, archive, partitions, args.split_seed, eos_policy)
    training_seeds = tuple(int(value) for value in args.training_seeds.split(",") if value.strip())
    if not training_seeds:
        raise ValueError("at least one training seed is required")
    if args.method in ("b2", "delta-mean"):
        result = run_b2(archive, partitions, args.bootstrap_repeats, args.split_seed) if args.method == "b2" else run_delta_mean(archive, partitions, args.bootstrap_repeats, args.split_seed)
    else:
        device = torch.device("cpu" if args.cpu or not torch.cuda.is_available() else f"cuda:{args.gpu}")
        if device.type == "cuda":
            torch.cuda.set_device(device)
        fixed_config = None
        if args.fixed_config:
            parsed_config = json.loads(args.fixed_config)
            if not isinstance(parsed_config, dict):
                raise ValueError("--fixed-config must be a JSON object")
            fixed_config = parsed_config
        if args.finalize:
            result = finalize_neural(args.method, archive, partitions, output_dir, training_seeds, args.standardization, args.transform, args.include_length, device, args.batch_size, args.bootstrap_repeats, args.split_seed)
        else:
            candidate_indices = None
            if args.candidate_only:
                if args.candidate_start is None or args.candidate_end is None or args.candidate_end <= args.candidate_start:
                    raise ValueError("candidate-only requires candidate-start < candidate-end")
                candidate_indices = range(args.candidate_start, args.candidate_end)
            result = run_neural(args.method, archive, partitions, output_dir, training_seeds, args.standardization, args.transform, args.include_length, device, args.batch_size, args.max_epochs, args.patience, args.bootstrap_repeats, args.split_seed, candidate_indices=candidate_indices, candidate_only=args.candidate_only, fixed_config=fixed_config)
    scores = result.pop("scores", None)
    if scores is not None:
        scores_path = method_scores_path(output_dir, args.method)
        np.savez_compressed(scores_path, labels=archive.labels, record_ids=archive.record_ids, seeds=np.asarray([DEFAULT_TRAINING_SEEDS[0]], dtype=np.int64), scores=np.asarray(scores)[:, None])
        result["scores_path"] = str(scores_path)
    result.update({
        "protocol": {
            "benchmark": args.benchmark,
            "target_epochs": args.epoch,
            "role": args.role,
            "split_seed": args.split_seed,
            "training_seeds": list(training_seeds),
            "bootstrap_repeats": args.bootstrap_repeats,
            "selection_metric": "mean validation pAUC[0,0.1] over registered detector seeds",
            "test_usage": "T is read only after configuration selection; C supplies thresholds",
        },
        "source_input": str(input_path.resolve()),
        "completed_at_unix": time.time(),
    })
    report_name = str(result.pop("_report_name", f"{args.method}.json"))
    _write_json_atomic(output_dir / "methods" / report_name, result)
    print(json.dumps({"method": args.method, "output_dir": str(output_dir), "report": str(output_dir / "methods" / f"{args.method}.json")}, indent=2), flush=True)


if __name__ == "__main__":
    main()
