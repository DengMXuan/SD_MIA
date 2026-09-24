"""Reconstruct attested baseline inputs without launching inference."""
from __future__ import annotations
from pathlib import Path
from typing import Any
from transformers import AutoTokenizer
from experiments.shared.data.data import SFTRecord
from experiments.shared.data.validation import verify_split_against_run
from experiments.shared.data.splits import build_split, pool_path

from experiments.paths import ROOT
from .types import AuditRecord

def _resolve(path: Path) -> Path:
    return path if path.is_absolute() else ROOT / path



def _target_tokenizer(run_dir: Path, cfg: Any) -> Any:
    """Prefer tokenizer files saved with the target checkpoint when present."""

    for directory in ("checkpoints", "adapters"):
        target = run_dir / directory / "target"
        if target.exists() and (target / "tokenizer_config.json").exists():
            tokenizer = AutoTokenizer.from_pretrained(str(target))
            break
    else:
        # Full target checkpoints from ``drafts.plain`` do not copy tokenizer
        # files.  The configured target/draft pairs share a tokenizer, so the
        # target tokenizer reproduces the frozen split without touching any
        # draft artifact.  ``verify_split_against_run`` below fails loudly if
        # a custom run violates that repository invariant.
        tokenizer = AutoTokenizer.from_pretrained(cfg.target_model)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    return tokenizer



def load_audit_records(
    run_dir: Path, cfg: Any, tokenizer: Any, pool_override: Path | None
) -> tuple[list[AuditRecord], list[AuditRecord], list[SFTRecord], dict[str, Any]]:
    pool = pool_override if pool_override is not None else cfg.pool_path
    if pool is None:
        pool = pool_path(cfg.benchmark)
    pool = _resolve(Path(pool))
    members, nonmembers, auxiliary, metadata = build_split(
        cfg.benchmark,
        pool,
        tokenizer,
        cfg.n_per_class,
        cfg.n_aux,
        cfg.data_seed,
    )
    verify_split_against_run(members, nonmembers, run_dir)
    return (
        [AuditRecord(record, 1) for record in members],
        [AuditRecord(record, 0) for record in nonmembers],
        auxiliary,
        metadata,
    )
