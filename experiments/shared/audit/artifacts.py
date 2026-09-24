"""Atomic method results and provenance checks for matrix recovery."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np

from experiments.shared.core.audit_runtime import ROOT, _write_json
from experiments.shared.core.deployment_archive import checkpoint_fingerprint, sha256_file
from experiments.shared.protocols.protocol_archive import atomic_npz


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def checkpoint_inventory(path):
    return [[str(p.relative_to(path)), p.stat().st_size, p.stat().st_mtime_ns]
            for p in sorted(Path(path).rglob("*")) if p.is_file()]


def runtime_files():
    roots = (ROOT / "experiments/shared", ROOT / "experiments/sd_membership_sft", ROOT / "experiments/baseline")
    return sorted([ROOT / "experiments/paths.py", ROOT / "experiments/MODULE_ALIASES.json", ROOT / "experiments/_compat.py", ROOT / "experiments/shared/models/model_pairs.json", *(p for root in roots for p in root.rglob("*.py")
                  if "tests" not in p.parts and "archive" not in p.parts)])


def sources_for(run_dir, checkpoint_roles):
    artifact = json.loads((run_dir / "results.json").read_text())
    files = [run_dir / "results.json", *runtime_files()]
    manifest = Path(artifact["data"]["shared_split_manifest"])
    manifest = manifest if manifest.is_absolute() else ROOT / manifest
    files += [manifest, manifest.with_suffix(".audit.json")]
    checkpoints = []
    for role in checkpoint_roles:
        path = (run_dir / "checkpoints" / role).resolve()
        before = checkpoint_inventory(path)
        fingerprint = checkpoint_fingerprint(path)
        if before != checkpoint_inventory(path):
            raise ValueError(f"checkpoint changed during fingerprinting: {path}")
        checkpoints.append({"path": str(path), "sha256": fingerprint, "inventory": before})
    return {"files": [{"path": str(p.resolve()), "sha256": sha256_file(p)} for p in files],
            "checkpoints": checkpoints}


def check_sources_light(sources):
    """Status/summary check; workers recompute full weight hashes before reuse."""
    for source in sources["files"]:
        if sha256_file(Path(source["path"])) != source["sha256"]:
            raise ValueError(f"source changed: {source['path']}")
    for source in sources["checkpoints"]:
        current = [list(row) for row in checkpoint_inventory(Path(source["path"]))]
        if current != [list(row) for row in source["inventory"]]:
            raise ValueError(f"checkpoint inventory changed: {source['path']}")


def read_result(folder, request_key=None, source_digest=None, *, check_sources=True):
    report = json.loads((folder / "REPORT.json").read_text())
    if report.get("schema") != "qwen_audit_method_v1":
        raise ValueError("unknown result schema")
    if request_key is not None and report["request_key"] != request_key:
        raise ValueError("method parameters changed")
    if source_digest is not None and digest(report["sources"]) != source_digest:
        raise ValueError("method source hashes changed")
    if sha256_file(folder / "scores.npz") != report["scores_sha256"]:
        raise ValueError("method score checksum failed")
    if report.get("detector"):
        detector = folder / report["detector"]["file"]
        if sha256_file(detector) != report["detector"]["sha256"]:
            raise ValueError("detector checksum failed")
    if report.get("observation_archive"):
        archive = report["observation_archive"]
        if sha256_file(Path(archive["path"])) != archive["sha256"]:
            raise ValueError("observation archive checksum failed")
    if check_sources:
        check_sources_light(report["sources"])
    with np.load(folder / "scores.npz", allow_pickle=False) as data:
        if set(data.files) != {"record_ids", "labels", "scores", "calibration", "test"}:
            raise ValueError("unexpected score fields")
        ids, labels, scores = data["record_ids"], data["labels"], data["scores"]
        cal, test = data["calibration"], data["test"]
        if (ids.ndim != 1 or labels.shape != ids.shape or scores.shape != ids.shape
                or cal.ndim != 1 or test.ndim != 1 or not len(cal) or not len(test)
                or cal.dtype.kind not in "iu" or test.dtype.kind not in "iu"
                or len(np.unique(ids)) != len(ids) or not np.isfinite(scores).all()
                or not np.isin(labels, [0, 1]).all()
                or sorted(np.r_[cal, test].tolist()) != list(range(len(ids)))
                or (labels[cal] != 0).any() or set(np.unique(labels[test])) != {0, 1}):
            raise ValueError("invalid score record/partition contract")
        if digest(ids.tolist()) != report["record_ids_sha256"]:
            raise ValueError("score IDs differ from report")
    return report


def save_result(folder, *, record_ids, labels, scores, calibration, test, report):
    folder.mkdir(parents=True, exist_ok=True)
    atomic_npz(folder / "scores.npz", dict(record_ids=record_ids, labels=labels, scores=scores,
                                          calibration=calibration, test=test))
    _write_json(folder / "REPORT.json", {
        **report, "schema": "qwen_audit_method_v1",
        "scores_sha256": sha256_file(folder / "scores.npz"),
        "record_ids_sha256": digest(record_ids.tolist()),
    })


def already_complete(folder, request_key, sources):
    if not (folder / "REPORT.json").exists():
        return False
    read_result(folder, request_key, digest(sources), check_sources=False)
    return True
