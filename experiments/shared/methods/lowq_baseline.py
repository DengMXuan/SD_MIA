"""Fixed low-q baseline; no learned window or active-query policy."""
from __future__ import annotations
import math
from typing import Iterable
import numpy as np

LOWQ_FRACTIONS = (0.10, 0.20, 0.50)


def lowq_fragment_scores(all_accept, logq0, lengths):
    result = {f"lowq_{int(100*fraction)}": np.empty(len(lengths)) for fraction in LOWQ_FRACTIONS}
    offset = 0
    for record_index, length_value in enumerate(lengths):
        length = int(length_value)
        end = offset + length
        q_order = np.argsort(logq0[offset:end], kind="stable")
        bits = all_accept[offset:end]
        for fraction in LOWQ_FRACTIONS:
            count = max(1, int(math.ceil(fraction * length)))
            result[f"lowq_{int(100*fraction)}"][record_index] = np.mean(bits[q_order[:count]])
        offset = end
    return result


def standardized_max(raw: dict[str, np.ndarray], names: Iterable[str], reference: np.ndarray) -> np.ndarray:
    names = tuple(names)
    matrix = np.column_stack([np.asarray(raw[name], dtype=np.float64) for name in names])
    fit = matrix[np.asarray(reference, dtype=np.int64)]
    center, scale = np.mean(fit, axis=0), np.std(fit, axis=0)
    scale = np.where(scale < 1e-8, 1.0, scale)
    return np.max((matrix - center) / scale, axis=1)

