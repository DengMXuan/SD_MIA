"""Shared setup and provenance helpers for model-facing scoring passes."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from transformers import AutoTokenizer

from .generalization import load_run_config
from .logpq_distribution import verify_shared_tokenizer, verify_split_against_run
from .splits import build_split, pool_path

ROOT = Path(__file__).resolve().parents[2]
ROLES = ("target", "draft_auxiliary_distilled", "draft_member_sft")


@dataclass(frozen=True)
class ScoringRecords:
    """The tokenizer and fixed record order shared by all model roles."""

    tokenizer: Any
    members: list[Any]
    nonmembers: list[Any]
    records: list[Any]
    labels: np.ndarray
    record_ids: np.ndarray


def resolve_run_dir(path: Path) -> Path:
    return path if path.is_absolute() else ROOT / path


def prepare_scoring_records(
    run_dir: Path,
    cfg: Any | None = None,
    pool_override: Path | None = None,
) -> tuple[Any, ScoringRecords]:
    """Build the fixed split and verify it against the saved SFT run."""
    run_dir = resolve_run_dir(run_dir).resolve()
    cfg = load_run_config(run_dir) if cfg is None else cfg
    tokenizer = AutoTokenizer.from_pretrained(cfg.draft_model)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    verify_shared_tokenizer(cfg.target_model, cfg.draft_model)

    pool = pool_override if pool_override is not None else cfg.pool_path
    if pool is None:
        pool = pool_path(cfg.benchmark)
    pool = pool if pool.is_absolute() else ROOT / pool
    members, nonmembers, _auxiliary, _metadata = build_split(
        cfg.benchmark, pool, tokenizer, cfg.n_per_class, cfg.n_aux, cfg.data_seed
    )
    verify_split_against_run(members, nonmembers, run_dir)
    records = members + nonmembers
    labels = np.asarray([1] * len(members) + [0] * len(nonmembers), dtype=np.int64)
    record_ids = np.asarray([record.record_id for record in records])
    return cfg, ScoringRecords(
        tokenizer=tokenizer,
        members=members,
        nonmembers=nonmembers,
        records=records,
        labels=labels,
        record_ids=record_ids,
    )


def resolve_checkpoint_path(run_dir: Path, role: str) -> Path:
    """Resolve the exact saved full checkpoint or adapter used for ``role``."""
    run_dir = resolve_run_dir(run_dir).resolve()
    if role not in ROLES:
        raise ValueError(f"Unsupported scoring role: {role}")
    for directory in ("checkpoints", "adapters"):
        path = run_dir / directory / role
        if path.exists():
            return path.resolve()
    raise FileNotFoundError(f"No {role} checkpoint under {run_dir}")


def role_provenance(cfg: Any, run_dir: Path, role: str) -> dict[str, str | int]:
    """Return metadata that proves which model source was scored."""
    checkpoint_path = resolve_checkpoint_path(run_dir, role)
    is_target = role == "target"
    return {
        "run_dir": str(resolve_run_dir(run_dir).resolve()),
        "benchmark": str(cfg.benchmark),
        "epoch": int(cfg.target_epochs),
        "role": role,
        "model_source": "target_checkpoint" if is_target else f"{role}_checkpoint",
        "base_model": str(cfg.target_model if is_target else cfg.draft_model),
        "checkpoint_path": str(checkpoint_path),
        "checkpoint_kind": checkpoint_path.parent.name,
    }
