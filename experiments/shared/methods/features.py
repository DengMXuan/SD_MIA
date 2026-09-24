"""Feature definitions shared by the M1 conditional-calibration pipeline.

The functions in this module are deliberately free of model loading and data
partitioning.  Keeping the numerical contract here makes it possible to test
the feature order and the short-record/window edge cases without touching a
Qwen checkpoint.
"""

from __future__ import annotations

from typing import Iterable

import numpy as np
import torch


Q_FEATURE_NAMES: tuple[str, ...] = (
    "log_q",
    "q_entropy_norm",
    "q_rank_norm",
    "q_top1_margin",
    "relative_position",
    "log_length",
)

ACTIVATION_STAT_NAMES: tuple[str, ...] = (
    "mean",
    "population_std",
    "rms",
    "mean_abs",
    "min",
    "max",
    "q10",
    "q50",
    "q90",
    "positive_fraction",
)

SELECTED_BLOCKS: tuple[int, ...] = (6, 13, 20, 27)
ACTIVATION_FEATURE_NAMES: tuple[str, ...] = tuple(
    f"block{block + 1}_{stat}"
    for block in SELECTED_BLOCKS
    for stat in ACTIVATION_STAT_NAMES
)
M1_FEATURE_NAMES: tuple[str, ...] = Q_FEATURE_NAMES + ACTIVATION_FEATURE_NAMES

AGGREGATE_FEATURE_NAMES: tuple[str, ...] = (
    "mean",
    "population_std",
    "mean_abs",
    "mean_positive_part",
    "mean_negative_part",
    "positive_fraction",
    "negative_fraction",
    "q10",
    "q25",
    "q50",
    "q75",
    "q90",
    "window4_q10",
    "window4_q90",
    "window8_q10",
    "window8_q90",
    "window16_q10",
    "window16_q90",
    "window32_q10",
    "window32_q90",
    "window64_q10",
    "window64_q90",
)
AGGREGATE_WINDOWS: tuple[int, ...] = (4, 8, 16, 32, 64)


def _as_float_tensor(value: torch.Tensor) -> torch.Tensor:
    """Convert a tensor to FP32 without changing its device."""

    return value if value.dtype == torch.float32 else value.float()


@torch.inference_mode()
def q_features_from_logits(
    logits: torch.Tensor,
    token_ids: torch.Tensor,
    relative_position: torch.Tensor,
    log_length: torch.Tensor,
    row_chunk: int = 128,
) -> torch.Tensor:
    """Return the six prescribed Q features for selected candidate tokens.

    ``logits`` is a two-dimensional tensor containing only prediction rows.
    Entropy and all vocabulary comparisons are performed in FP32.  Rows are
    processed in chunks so a long ArXiv record does not require a second full
    FP32 vocabulary tensor on the GPU.
    """

    if logits.ndim != 2:
        raise ValueError(f"logits must have shape [tokens, vocab], got {tuple(logits.shape)}")
    if token_ids.ndim != 1 or len(token_ids) != logits.shape[0]:
        raise ValueError("token_ids must be one-dimensional and aligned with logits")
    if relative_position.ndim != 1 or len(relative_position) != logits.shape[0]:
        raise ValueError("relative_position must be aligned with logits")
    if log_length.ndim != 1 or len(log_length) != logits.shape[0]:
        raise ValueError("log_length must be aligned with logits")
    if row_chunk <= 0:
        raise ValueError("row_chunk must be positive")
    vocab_size = int(logits.shape[-1])
    if vocab_size < 2:
        raise ValueError("q features require a vocabulary with at least two entries")
    if token_ids.numel() and (
        int(token_ids.min()) < 0 or int(token_ids.max()) >= vocab_size
    ):
        raise ValueError("candidate token id is outside the logits vocabulary")

    rows = _as_float_tensor(logits)
    ids = token_ids.to(device=rows.device, dtype=torch.long)
    output = torch.empty(
        (rows.shape[0], len(Q_FEATURE_NAMES)), dtype=torch.float32, device=rows.device
    )
    log_vocab = float(np.log(vocab_size))
    for start in range(0, rows.shape[0], row_chunk):
        end = min(start + row_chunk, rows.shape[0])
        chunk = rows[start:end]
        candidate = chunk.gather(1, ids[start:end, None]).squeeze(1)
        log_z = torch.logsumexp(chunk, dim=-1)
        log_probs = chunk - log_z[:, None]
        probs = torch.exp(log_probs)
        entropy = -(probs * log_probs).sum(dim=-1) / log_vocab
        rank = torch.sum(chunk > candidate[:, None], dim=-1).float()
        top2 = torch.topk(chunk, k=2, dim=-1, largest=True, sorted=True).values
        output[start:end, 0] = candidate - log_z
        output[start:end, 1] = entropy
        output[start:end, 2] = torch.log1p(rank) / log_vocab
        output[start:end, 3] = top2[:, 0] - top2[:, 1]
        output[start:end, 4] = relative_position[start:end].float()
        output[start:end, 5] = log_length[start:end].float()
    return output


@torch.inference_mode()
def activation_statistics(hidden_states: torch.Tensor) -> torch.Tensor:
    """Summarize a ``[tokens, hidden_size]`` residual stream in FP32."""

    if hidden_states.ndim != 2:
        raise ValueError(
            "hidden_states must contain selected prediction positions with shape "
            f"[tokens, hidden], got {tuple(hidden_states.shape)}"
        )
    if hidden_states.shape[0] == 0:
        raise ValueError("cannot summarize an empty set of activation positions")
    values = hidden_states.float()
    return torch.stack(
        (
            values.mean(dim=-1),
            values.std(dim=-1, unbiased=False),
            torch.sqrt(torch.mean(values.square(), dim=-1)),
            values.abs().mean(dim=-1),
            values.min(dim=-1).values,
            values.max(dim=-1).values,
            torch.quantile(values, 0.10, dim=-1),
            torch.quantile(values, 0.50, dim=-1),
            torch.quantile(values, 0.90, dim=-1),
            (values > 0.0).float().mean(dim=-1),
        ),
        dim=-1,
    )


def aggregate_values(
    values: np.ndarray | Iterable[float],
    windows: tuple[int, ...] = AGGREGATE_WINDOWS,
) -> np.ndarray:
    """Apply the fixed 22-dimensional continuous document aggregator ``A``.

    Window means are computed from the complete consecutive token sequence.
    For a record shorter than a window, the record mean is the sole window
    value, so the two corresponding quantiles are equal to that mean.
    """

    vector = np.asarray(values, dtype=np.float64)
    if vector.ndim != 1 or vector.size == 0:
        raise ValueError("values must be a non-empty one-dimensional array")
    positive = np.maximum(vector, 0.0)
    negative = np.minimum(vector, 0.0)
    row = [
        float(vector.mean()),
        float(vector.std(ddof=0)),
        float(np.abs(vector).mean()),
        float(positive.mean()),
        float(negative.mean()),
        float(np.mean(vector > 0.0)),
        float(np.mean(vector < 0.0)),
        *[float(np.quantile(vector, q)) for q in (0.10, 0.25, 0.50, 0.75, 0.90)],
    ]
    for width in windows:
        if width <= 0:
            raise ValueError("window widths must be positive")
        if vector.size < width:
            window_means = np.asarray([vector.mean()], dtype=np.float64)
        else:
            # A prefix sum is algebraically identical to a stride-1 moving
            # mean, but avoids the O(n * width) repeated convolution that
            # dominates the multi-million-token M1 caches.
            prefix = np.concatenate(
                (np.zeros(1, dtype=np.float64), np.cumsum(vector, dtype=np.float64))
            )
            window_means = (prefix[width:] - prefix[:-width]) / float(width)
        row.extend(
            [
                float(np.quantile(window_means, 0.10)),
                float(np.quantile(window_means, 0.90)),
            ]
        )
    result = np.asarray(row, dtype=np.float64)
    expected = len(AGGREGATE_FEATURE_NAMES)
    if len(windows) != len(AGGREGATE_WINDOWS):
        expected = 12 + 2 * len(windows)
    if result.size != expected:
        raise AssertionError(f"aggregate feature count {result.size} != {expected}")
    return result


def aggregate_matrix(
    token_values: np.ndarray,
    lengths: np.ndarray,
    windows: tuple[int, ...] = AGGREGATE_WINDOWS,
) -> np.ndarray:
    """Aggregate a flat token vector into one row per record."""

    values = np.asarray(token_values)
    record_lengths = np.asarray(lengths, dtype=np.int64)
    if values.ndim != 1 or record_lengths.ndim != 1:
        raise ValueError("token_values and lengths must be one-dimensional")
    if np.any(record_lengths <= 0) or int(record_lengths.sum()) != len(values):
        raise ValueError("lengths do not describe token_values")
    offsets = np.concatenate(([0], np.cumsum(record_lengths)))
    return np.vstack(
        [
            aggregate_values(values[int(start) : int(end)], windows=windows)
            for start, end in zip(offsets[:-1], offsets[1:])
        ]
    )


def document_mean_std_matrix(token_matrix: np.ndarray, lengths: np.ndarray) -> np.ndarray:
    """Return per-document mean and population std for each token feature."""

    matrix = np.asarray(token_matrix, dtype=np.float64)
    record_lengths = np.asarray(lengths, dtype=np.int64)
    if matrix.ndim != 2 or record_lengths.ndim != 1:
        raise ValueError("token_matrix must be [tokens, dimensions]")
    if np.any(record_lengths <= 0) or int(record_lengths.sum()) != len(matrix):
        raise ValueError("lengths do not describe token_matrix")
    offsets = np.concatenate(([0], np.cumsum(record_lengths)))
    rows: list[np.ndarray] = []
    for start, end in zip(offsets[:-1], offsets[1:]):
        row = matrix[int(start) : int(end)]
        rows.append(np.concatenate((row.mean(axis=0), row.std(axis=0, ddof=0))))
    return np.vstack(rows)
