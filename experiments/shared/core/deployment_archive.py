"""Schema and provenance checks for accept-only deployment observations."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any

import numpy as np


ARCHIVE_SCHEMA = "sd_mia_accept_only_deployment_v1"
ARCHIVE_PRODUCER = "experiments.shared.protocols.collect_deployment_observations"
PROTOCOL = "fixed_candidate_accept_only_b2"
DRAFT_FEATURE_NAMES = (
    "q_entropy_norm",
    "q_rank_norm",
    "q_top1_margin",
)
_HEX_DIGITS = frozenset("0123456789abcdef")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while chunk := handle.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def is_runtime_bytecode(path: Path) -> bool:
    """Only import-generated caches; standalone bytecode may be model code."""
    path = Path(path)
    return '__pycache__' in path.parts and path.suffix in ('.pyc', '.pyo')


def checkpoint_files(path: Path) -> list[Path]:
    """Model assets include remote source, weights, configs and tokenizer files.

    Importing checkpoint-owned Python may create disposable bytecode alongside
    it. Inventory and content hashing must use this same asset boundary.
    """
    path = Path(path)
    if not path.is_dir():
        raise FileNotFoundError(path)
    return sorted(item for item in path.rglob('*')
                  if item.is_file() and not is_runtime_bytecode(item.relative_to(path)))


def checkpoint_fingerprint(path: Path) -> str:
    """Hash checkpoint asset paths and bytes so provenance survives relocation."""
    path = Path(path)
    files = checkpoint_files(path)
    if not files:
        raise RuntimeError(f"Checkpoint has no files: {path}")
    digest = hashlib.sha256()
    for item in files:
        relative = item.relative_to(path).as_posix().encode("utf-8")
        digest.update(len(relative).to_bytes(4, "big"))
        digest.update(relative)
        with item.open("rb") as handle:
            while chunk := handle.read(8 * 1024 * 1024):
                digest.update(chunk)
    return digest.hexdigest()


def _sidecar_path(path: Path) -> Path:
    return path.with_suffix(path.suffix + ".json")


def _scalar_text(archive: Any, name: str) -> str:
    value = np.asarray(archive[name])
    if value.ndim != 0:
        raise ValueError(f"{name} must be a scalar string")
    return str(value.item())


def _valid_sha256(value: str) -> bool:
    return len(value) == 64 and set(value).issubset(_HEX_DIGITS)


def write_deployment_archive(
    path: Path,
    *,
    logq: np.ndarray,
    bits: np.ndarray,
    lengths: np.ndarray,
    labels: np.ndarray,
    record_ids: np.ndarray,
    record_roles: np.ndarray,
    draft_features: np.ndarray,
    provenance: dict[str, Any],
) -> dict[str, Any]:
    """Write the observable archive and a hash-anchored provenance sidecar."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    logq = np.asarray(logq, dtype=np.float32)
    bits = np.asarray(bits, dtype=np.uint8)
    lengths = np.asarray(lengths, dtype=np.int64)
    labels = np.asarray(labels, dtype=np.int64)
    record_ids = np.asarray(record_ids)
    record_roles = np.asarray(record_roles)
    draft_features = np.asarray(draft_features, dtype=np.float32)
    if lengths.ndim != 1 or np.any(lengths <= 0):
        raise ValueError("deployment lengths must be positive record lengths")
    tokens = int(lengths.sum())
    if logq.shape != (tokens, 1) or np.any(~np.isfinite(logq)) or np.any(logq > 1e-6):
        raise ValueError("deployment logq must be a finite token-by-one log probability")
    if bits.shape != (tokens, 1, 2) or np.any((bits != 0) & (bits != 1)):
        raise ValueError("deployment bits must contain exactly two binary decisions")
    if draft_features.shape != (tokens, len(DRAFT_FEATURE_NAMES)):
        raise ValueError("deployment draft features have the wrong shape")
    if not np.isfinite(draft_features).all():
        raise ValueError("deployment draft features contain nonfinite values")
    if not (
        labels.shape == record_ids.shape == record_roles.shape == lengths.shape
    ):
        raise ValueError("deployment record metadata is not aligned")
    if np.any((labels != 0) & (labels != 1)) or len(np.unique(record_ids)) != len(record_ids):
        raise ValueError("deployment records require binary labels and unique IDs")
    required_provenance = {
        "run_manifest_sha256",
        "benchmark",
        "target_epochs",
        "draft_checkpoint",
        "target_checkpoint",
        "pool_sha256",
        "split_seed",
        "acceptance_seed",
        "language_models_frozen",
        "query_budget",
    }
    missing = required_provenance.difference(provenance)
    if missing:
        raise ValueError(f"deployment provenance lacks {sorted(missing)}")
    draft_fingerprint = str(provenance["draft_checkpoint"]["fingerprint"])
    target_fingerprint = str(provenance["target_checkpoint"]["fingerprint"])
    run_manifest_sha = str(provenance["run_manifest_sha256"])
    pool_sha = str(provenance["pool_sha256"])
    if not all(
        _valid_sha256(value)
        for value in (
            draft_fingerprint,
            target_fingerprint,
            run_manifest_sha,
            pool_sha,
        )
    ):
        raise ValueError("pool, checkpoint, and run-manifest fingerprints must be SHA-256")
    if provenance["language_models_frozen"] is not True:
        raise ValueError("deployment collection requires frozen language models")
    if int(provenance["query_budget"]) != 2:
        raise ValueError("deployment archive requires exactly two accept decisions")
    if not str(provenance["benchmark"]) or int(provenance["target_epochs"]) <= 0:
        raise ValueError("deployment archive requires a benchmark and target epoch")
    if provenance["draft_checkpoint"].get("role") != "draft_auxiliary_distilled":
        raise ValueError("deployment archive requires the auxiliary-data draft")
    if provenance["target_checkpoint"].get("role") != "target_verifier":
        raise ValueError("deployment archive requires a target verifier checkpoint")

    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    with temporary.open("wb") as handle:
        np.savez_compressed(
            handle,
            logq=logq,
            bits=bits,
            lengths=lengths,
            view_names=np.asarray(("original",)),
            labels=labels,
            record_ids=record_ids,
            record_roles=record_roles,
            draft_features=draft_features,
            draft_feature_names=np.asarray(DRAFT_FEATURE_NAMES),
            archive_schema=np.asarray(ARCHIVE_SCHEMA),
            producer=np.asarray(ARCHIVE_PRODUCER),
            protocol=np.asarray(PROTOCOL),
            run_manifest_sha256=np.asarray(run_manifest_sha),
            draft_checkpoint_fingerprint=np.asarray(draft_fingerprint),
            target_checkpoint_fingerprint=np.asarray(target_fingerprint),
        )
    os.replace(temporary, path)
    archive_sha = sha256_file(path)
    sidecar = {
        "archive_schema": ARCHIVE_SCHEMA,
        "archive_sha256": archive_sha,
        "producer": ARCHIVE_PRODUCER,
        "protocol": PROTOCOL,
        "detector_inputs": "draft q statistics and verifier accept bits only",
        "target_values_persisted": False,
        "draft_feature_names": list(DRAFT_FEATURE_NAMES),
        "language_models_frozen": True,
        "query_budget": 2,
        **provenance,
    }
    sidecar_path = _sidecar_path(path)
    sidecar_temporary = sidecar_path.with_name(
        f".{sidecar_path.name}.tmp.{os.getpid()}"
    )
    sidecar_temporary.write_text(
        json.dumps(sidecar, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    os.replace(sidecar_temporary, sidecar_path)
    return sidecar


def validate_deployment_archive(path: Path) -> dict[str, Any]:
    """Reject archives that do not attest the exact observable feature contract."""
    path = Path(path)
    sidecar_path = _sidecar_path(path)
    if not sidecar_path.is_file():
        raise ValueError(f"deployment archive lacks provenance sidecar: {sidecar_path}")
    sidecar = json.loads(sidecar_path.read_text(encoding="utf-8"))
    actual_sha = sha256_file(path)
    if sidecar.get("archive_sha256") != actual_sha:
        raise ValueError("deployment archive hash does not match its provenance sidecar")
    expected_sidecar = {
        "archive_schema": ARCHIVE_SCHEMA,
        "producer": ARCHIVE_PRODUCER,
        "protocol": PROTOCOL,
        "target_values_persisted": False,
        "draft_feature_names": list(DRAFT_FEATURE_NAMES),
    }
    for key, expected in expected_sidecar.items():
        if sidecar.get(key) != expected:
            raise ValueError(f"invalid deployment provenance field {key!r}")
    if sidecar.get("detector_inputs") != "draft q statistics and verifier accept bits only":
        raise ValueError("deployment provenance does not attest accept-only inputs")
    for name in ("run_manifest_sha256", "pool_sha256"):
        if not _valid_sha256(str(sidecar.get(name, ""))):
            raise ValueError(f"invalid SHA-256 provenance in {name}")
    if sidecar.get("draft_checkpoint", {}).get("role") != "draft_auxiliary_distilled":
        raise ValueError("deployment provenance names the wrong draft role")
    if sidecar.get("target_checkpoint", {}).get("role") != "target_verifier":
        raise ValueError("deployment provenance names the wrong verifier role")

    with np.load(path, allow_pickle=False) as archive:
        required = {
            "logq",
            "bits",
            "lengths",
            "view_names",
            "labels",
            "record_ids",
            "record_roles",
            "draft_features",
            "draft_feature_names",
            "archive_schema",
            "producer",
            "protocol",
            "run_manifest_sha256",
            "draft_checkpoint_fingerprint",
            "target_checkpoint_fingerprint",
        }
        missing = required.difference(archive.files)
        if missing:
            raise ValueError(f"deployment archive lacks {sorted(missing)}")
        unexpected = set(archive.files).difference(required)
        if unexpected:
            raise ValueError(
                f"deployment archive contains unapproved fields {sorted(unexpected)}"
            )
        feature_names = tuple(np.asarray(archive["draft_feature_names"]).astype(str))
        if feature_names != DRAFT_FEATURE_NAMES:
            raise ValueError("deployment archive has an unapproved draft feature schema")
        scalar_contract = {
            "archive_schema": ARCHIVE_SCHEMA,
            "producer": ARCHIVE_PRODUCER,
            "protocol": PROTOCOL,
            "run_manifest_sha256": str(sidecar.get("run_manifest_sha256", "")),
            "draft_checkpoint_fingerprint": str(
                sidecar.get("draft_checkpoint", {}).get("fingerprint", "")
            ),
            "target_checkpoint_fingerprint": str(
                sidecar.get("target_checkpoint", {}).get("fingerprint", "")
            ),
        }
        for name, expected in scalar_contract.items():
            value = _scalar_text(archive, name)
            if value != expected:
                raise ValueError(f"deployment archive metadata mismatch for {name}")
            if name.endswith("sha256") or name.endswith("fingerprint"):
                if not _valid_sha256(value):
                    raise ValueError(f"invalid SHA-256 provenance in {name}")
        features = np.asarray(archive["draft_features"])
        if features.ndim != 2 or features.shape[1] != len(DRAFT_FEATURE_NAMES):
            raise ValueError("draft_features do not match the approved feature schema")
    return {
        **sidecar,
        "archive_sha256": actual_sha,
        "sidecar": str(sidecar_path),
        "sidecar_sha256": sha256_file(sidecar_path),
    }
