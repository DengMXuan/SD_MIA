"""Baseline record and token-statistic contracts."""
from __future__ import annotations
from dataclasses import dataclass
import numpy as np
from experiments.shared.data.data import SFTRecord

@dataclass(frozen=True)
class AuditRecord:
    record: SFTRecord
    label: int



@dataclass
class TokenStats:
    token_ids: np.ndarray
    token_logp: np.ndarray
    expected_logp: np.ndarray
    variance_logp: np.ndarray
    log_similarity: np.ndarray | None = None
    sampled_token_ids: np.ndarray | None = None
    sead_log_density: float | None = None
    sead_lexical_density: float | None = None
