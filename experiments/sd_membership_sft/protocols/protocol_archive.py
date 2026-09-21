"""Strict observable-only archives for head probes and natural SD trajectories."""
from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np

from experiments.sd_membership_sft.core.audit_runtime import _write_json
from experiments.sd_membership_sft.core.deployment_archive import sha256_file, checkpoint_fingerprint
from experiments.sd_membership_sft.protocols.sd_protocol import FEATURE_NAMES

SCHEMA = "sd_mia_protocol_observations_v1"
FIELDS = {
    "features", "counts", "lengths", "document_indices", "start_indices",
    "record_ids", "record_roles", "labels",
}


def atomic_npz(path: Path, values: dict) -> None:
    path = path.resolve()
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp.npz")
    np.savez_compressed(temporary, **values)
    temporary.replace(path)


def validate_arrays(data: dict, contract: dict) -> None:
    if set(data) != FIELDS:
        raise ValueError(f"unexpected/missing observation fields: {set(data) ^ FIELDS}")
    lengths = data["lengths"]
    ids, roles, labels = data["record_ids"], data["record_roles"], data["labels"]
    if (ids.ndim != 1 or not len(ids) or len(np.unique(ids)) != len(ids)
            or labels.shape != ids.shape or roles.shape != ids.shape
            or not np.isin(labels, [0, 1]).all()):
        raise ValueError("invalid record identity/label arrays")
    if not np.isin(roles, ["audit_auxiliary", "member", "nonmember", "smoke"]).all():
        raise ValueError("unknown record role")
    if not np.array_equal(labels, (roles == "member").astype(int)):
        raise ValueError("record labels disagree with roles")
    if (lengths.ndim != 1 or not len(lengths) or (lengths <= 0).any()
            or not np.issubdtype(lengths.dtype, np.integer)):
        raise ValueError("invalid trajectory lengths")
    docs, starts = data["document_indices"], data["start_indices"]
    nstarts = len(contract["starts"])
    if (docs.shape != lengths.shape or starts.shape != lengths.shape
            or not np.issubdtype(docs.dtype, np.integer)
            or not np.issubdtype(starts.dtype, np.integer)
            or (docs < 0).any() or (docs >= len(ids)).any()
            or (starts < 0).any() or (starts >= nstarts).any()):
        raise ValueError("invalid trajectory ownership")
    pairs = list(zip(docs.tolist(), starts.tolist()))
    if len(set(pairs)) != len(pairs) or len(pairs) != len(ids) * nstarts:
        raise ValueError("every document must have exactly one trajectory per start")
    x, counts = data["features"], data["counts"]
    if (x.shape != (int(lengths.sum()), len(FEATURE_NAMES)) or not np.isfinite(x).all()
            or (x[:, 0] > 1e-5).any() or (x[:, 1:3] < -1e-5).any()
            or (x[:, 1:3] > 1 + 1e-5).any() or (x[:, 3:] < 0).any()
            or (x[:, 4:] > 1).any()):
        raise ValueError("invalid observable draft features")
    protocol = contract["protocol"]
    if protocol not in ("natural", "fixed"):
        raise ValueError("unknown protocol")
    k = 1 if protocol == "natural" else 2
    if (counts.shape != (len(x),) or not np.issubdtype(counts.dtype, np.integer)
            or (counts < 0).any() or (counts > k).any()):
        raise ValueError("invalid acceptance counts")
    if protocol == "natural":
        if contract["rounds_per_start"] < 1 or (lengths > contract["rounds_per_start"]).any():
            raise ValueError("trajectory exceeds round cap")
    elif contract["starts"] != ["fixed"]:
        raise ValueError("fixed probes have one document trajectory")


def verify_sources(sources: dict) -> None:
    for source in sources.get("files", []):
        if sha256_file(Path(source["path"])) != source["sha256"]:
            raise ValueError(f"source file changed: {source['path']}")
    for source in sources.get("checkpoints", []):
        if checkpoint_fingerprint(Path(source["path"])) != source["sha256"]:
            raise ValueError(f"checkpoint changed: {source['path']}")


def save_archive(path: Path, data: dict, contract: dict, costs: list[dict]) -> None:
    validate_arrays(data, contract)
    if len(costs) != len(data["lengths"]):
        raise ValueError("cost/trajectory alignment failed")
    path.parent.mkdir(parents=True, exist_ok=True)
    atomic_npz(path, data)
    _write_json(path.with_suffix(path.suffix + ".json"), {
        "schema": SCHEMA, "feature_names": list(FEATURE_NAMES),
        "archive_sha256": sha256_file(path), "contract": contract, "costs": costs,
    })


def load_archive(path: Path, *, check_sources: bool = True):
    sidecar = path.with_suffix(path.suffix + ".json")
    envelope = json.loads(sidecar.read_text())
    if (envelope["schema"] != SCHEMA or envelope["feature_names"] != list(FEATURE_NAMES)
            or envelope["archive_sha256"] != sha256_file(path)):
        raise ValueError("archive schema, feature names or hash mismatch")
    with np.load(path, allow_pickle=False) as source:
        data = dict(source)
    validate_arrays(data, envelope["contract"])
    if len(envelope["costs"]) != len(data["lengths"]):
        raise ValueError("cost/trajectory alignment failed")
    if check_sources:
        verify_sources(envelope["contract"]["sources"])
    envelope["sidecar_sha256"] = sha256_file(sidecar)
    return data, envelope
