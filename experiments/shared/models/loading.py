"""Load frozen protocol pairs and reconstruct their attested four-role records."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
from transformers import AutoTokenizer

from experiments.shared.core.audit_runtime import ROOT
from experiments.shared.training.generalization import load_run_config, load_finetuned_model, load_draft_model
from experiments.shared.core.scoring_common import DeploymentScoringRecords, prepare_deployment_scoring_records, resolve_checkpoint_path
from experiments.shared.core.deployment_archive import checkpoint_fingerprint, sha256_file
from .adapters import family_for


def local_tokenizer(checkpoint: Path, model: str, revision: str | None):
    """Older complete weight checkpoints did not copy tokenizer assets."""
    if (checkpoint / "tokenizer_config.json").is_file():
        return AutoTokenizer.from_pretrained(str(checkpoint), local_files_only=True)
    return AutoTokenizer.from_pretrained(model, revision=revision, local_files_only=True)


DRAFT_ROLES = ("draft_auxiliary_distilled", "draft_member_sft", "auxiliary_head", "member_head")
HEAD_NAMES = {"draft_auxiliary_distilled": "auxiliary_head", "draft_member_sft": "member_head",
              "auxiliary_head": "auxiliary_head", "member_head": "member_head"}


def checkpoint_paths(run_dir: Path, kind: str, draft_role: str = DRAFT_ROLES[0]) -> tuple[Path, Path]:
    family = family_for(kind)
    # Historical role names map to the same explicitly selected head.
    role = HEAD_NAMES.get(draft_role, draft_role) if family.uses_head else draft_role
    if role not in family.roles:
        raise ValueError("unsupported draft role for this adapter")
    target = resolve_checkpoint_path(run_dir, "target")
    draft = (run_dir / family.checkpoint_directory / role if family.uses_head
             else resolve_checkpoint_path(run_dir, role))
    if not draft.is_dir():
        raise FileNotFoundError(f"matching {role} checkpoint not ready: {draft}")
    if family.uses_head:
        from experiments.shared.drafts.common import checkpoint_complete
        if not checkpoint_complete(target) or not checkpoint_complete(draft):
            raise ValueError(f"head collection requires completed target and {role} stages")
    return target, draft


def prepare_records(run_dir: Path, kind: str, draft_role: str | None = DRAFT_ROLES[0]):
    if not family_for(kind).uses_head:
        cfg, records = prepare_deployment_scoring_records(run_dir)
        return cfg, records
    from experiments.shared.drafts.common import PAIR_MODELS, tokenizer_for, tokenizer_source_for
    from experiments.shared.data.splits import build_controlled_split_from_shared_manifest, pool_path

    cfg = load_run_config(run_dir)
    artifact = json.loads((run_dir / "results.json").read_text())
    track = artifact["protocol_track"]
    pair = track["pair"]
    if PAIR_MODELS[pair]["kind"] != kind or not track["target_frozen_before_heads"]:
        raise ValueError("requested adapter disagrees with frozen-head training passport")
    spec = PAIR_MODELS[pair]
    if cfg.target_model != spec["target"] or cfg.target_revision != spec["target_revision"]:
        raise ValueError("training tokenizer revision differs from registered head pair")
    tokenizer = tokenizer_for(pair)
    manifest = Path(track["shared_raw_split"])
    manifest = manifest if manifest.is_absolute() else ROOT / manifest
    pool = cfg.pool_path or pool_path(cfg.benchmark)
    pool = pool if pool.is_absolute() else ROOT / pool
    split = build_controlled_split_from_shared_manifest(
        cfg.benchmark, pool, tokenizer, manifest, tokenizer_source_for(pair),
    )
    metadata = split.metadata
    audit = json.loads(manifest.with_suffix(".audit.json").read_text())
    attestation = audit["tokenizers"][tokenizer_source_for(pair)]
    if (attestation["shared_split_sha256"] != metadata["shared_split_sha256"]
            or attestation["cross_split_ngram_audit"]["gate"] != "PASS"
            or metadata["split_seed"] != cfg.data_seed):
        raise ValueError("shared split audit is stale or mismatched")
    stages = ["target"]
    if draft_role is not None:
        if draft_role not in HEAD_NAMES:
            raise ValueError("unknown head role")
        stages.append("aux_head" if HEAD_NAMES[draft_role] == "auxiliary_head" else "member_head")
    for stage in stages:
        saved = artifact["stages"][stage]["data"]
        if saved["shared_split_sha256"] != metadata["shared_split_sha256"]:
            raise ValueError(f"{stage} used a different shared split")
    rows = split.audit_auxiliary + split.members + split.nonmembers
    roles = np.asarray(["audit_auxiliary"] * len(split.audit_auxiliary)
                       + ["member"] * len(split.members) + ["nonmember"] * len(split.nonmembers))
    return cfg, DeploymentScoringRecords(
        tokenizer, split.audit_auxiliary, split.members, split.nonmembers,
        rows, (roles == "member").astype(np.int64),
        np.asarray([r.record_id for r in rows]), roles,
    )


def load_adapter(run_dir: Path, kind: str, device: str, draft_role: str = DRAFT_ROLES[0]):
    cfg = load_run_config(run_dir)
    target_path, draft_path = checkpoint_paths(run_dir, kind, draft_role)
    device = torch.device(device)
    family = family_for(kind)
    if family.shared_tokenizer:
        # Full token-ID equality, not just equal vocabulary widths or probe strings.
        target_tok = local_tokenizer(target_path, cfg.target_model, cfg.target_revision)
        draft_tok = local_tokenizer(draft_path, cfg.draft_model, cfg.draft_revision)
        if target_tok.get_vocab() != draft_tok.get_vocab():
            raise ValueError("plain target/draft token-ID vocabularies differ")
    target = load_finetuned_model(run_dir, cfg.target_model, device, attn_implementation="sdpa")
    draft = family.load_draft(run_dir, cfg, draft_path, target_path, draft_role, device)
    return family.protocol_factory(target, draft, device)


def source_contract(run_dir: Path, kind: str, draft_role: str = DRAFT_ROLES[0]) -> dict:
    paths = checkpoint_paths(run_dir, kind, draft_role)
    artifact = json.loads((run_dir / "results.json").read_text())
    files = [run_dir / "results.json"]
    manifest = artifact.get("protocol_track", {}).get("shared_raw_split") or artifact.get("data", {}).get("shared_split_manifest")
    if manifest:
        path = Path(manifest)
        path = path if path.is_absolute() else ROOT / path
        files.extend([path, path.with_suffix(".audit.json")])
    return {
        "files": [{"path": str(p.resolve()), "sha256": sha256_file(p)} for p in files],
        "checkpoints": [{"path": str(p.resolve()), "sha256": checkpoint_fingerprint(p)} for p in paths],
    }
