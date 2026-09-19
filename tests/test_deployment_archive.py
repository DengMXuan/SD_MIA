from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from experiments.sd_membership_sft.deployment_accept_only import (
    _load_archive,
    _report_matches_source,
)
from experiments.sd_membership_sft.deployment_archive import (
    DRAFT_FEATURE_NAMES,
    checkpoint_fingerprint,
    sha256_file,
    validate_deployment_archive,
    write_deployment_archive,
)
from experiments.sd_membership_sft import scoring_common


def _provenance(tmp_path: Path) -> dict[str, object]:
    draft = tmp_path / "draft"
    target = tmp_path / "target"
    draft.mkdir(exist_ok=True)
    target.mkdir(exist_ok=True)
    (draft / "weights.bin").write_bytes(b"draft")
    (target / "weights.bin").write_bytes(b"target")
    return {
        "run_manifest_sha256": "a" * 64,
        "benchmark": "wikitection",
        "target_epochs": 1,
        "pool_sha256": "b" * 64,
        "split_seed": 7,
        "acceptance_seed": 11,
        "language_models_frozen": True,
        "query_budget": 2,
        "draft_checkpoint": {
            "path": str(draft),
            "role": "draft_auxiliary_distilled",
            "fingerprint": checkpoint_fingerprint(draft),
        },
        "target_checkpoint": {
            "path": str(target),
            "role": "target_verifier",
            "fingerprint": checkpoint_fingerprint(target),
        },
    }


def _write_valid(path: Path, provenance: dict[str, object]) -> None:
    write_deployment_archive(
        path,
        logq=np.asarray([[-1.0], [-2.0]], dtype=np.float32),
        bits=np.asarray([[[1, 0]], [[1, 1]]], dtype=np.uint8),
        lengths=np.asarray([2], dtype=np.int64),
        labels=np.asarray([0], dtype=np.int64),
        record_ids=np.asarray(["record-0"]),
        record_roles=np.asarray(["audit_auxiliary"]),
        draft_features=np.asarray([[0.1, 0.2, 0.3], [0.2, 0.3, 0.4]]),
        provenance=provenance,
    )


def _rewrite_archive(path: Path, **updates: np.ndarray) -> None:
    with np.load(path, allow_pickle=False) as archive:
        values = {name: np.asarray(archive[name]) for name in archive.files}
    values.update(updates)
    with path.open("wb") as handle:
        np.savez_compressed(handle, **values)
    sidecar_path = path.with_suffix(path.suffix + ".json")
    sidecar = json.loads(sidecar_path.read_text())
    sidecar["archive_sha256"] = sha256_file(path)
    sidecar_path.write_text(json.dumps(sidecar))


def test_valid_deployment_archive_has_hash_anchored_observable_contract(
    tmp_path: Path,
) -> None:
    path = tmp_path / "observations.npz"
    _write_valid(path, _provenance(tmp_path))

    provenance = validate_deployment_archive(path)
    observations, labels, ids, roles, features, loaded = _load_archive(path)

    assert provenance["archive_sha256"] == sha256_file(path)
    assert loaded["archive_sha256"] == provenance["archive_sha256"]
    assert observations.bits.shape == (2, 1, 2)
    assert labels.tolist() == [0]
    assert ids.tolist() == ["record-0"]
    assert roles.tolist() == ["audit_auxiliary"]
    assert features.shape == (2, len(DRAFT_FEATURE_NAMES))


def test_deployment_archive_rejects_disguised_or_extra_target_features(
    tmp_path: Path,
) -> None:
    path = tmp_path / "observations.npz"
    _write_valid(path, _provenance(tmp_path))
    _rewrite_archive(path, draft_feature_names=np.asarray(("logp", "x", "y")))
    with pytest.raises(ValueError, match="unapproved draft feature schema"):
        validate_deployment_archive(path)

    path = tmp_path / "extra.npz"
    _write_valid(path, _provenance(tmp_path))
    _rewrite_archive(path, target_logp=np.asarray([-1.0, -2.0]))
    with pytest.raises(ValueError, match="unapproved fields"):
        validate_deployment_archive(path)


def test_report_reuse_requires_archive_and_sidecar_hashes(tmp_path: Path) -> None:
    output = tmp_path / "result"
    output.mkdir()
    (output / "REPORT.json").write_text(
        json.dumps({"source": {"sha256": "a", "sidecar_sha256": "b"}})
    )
    assert _report_matches_source(
        output, {"archive_sha256": "a", "sidecar_sha256": "b"}
    )
    assert not _report_matches_source(
        output, {"archive_sha256": "changed", "sidecar_sha256": "b"}
    )


def test_deployment_scoring_records_include_independent_audit_role(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def record(name: str):
        return SimpleNamespace(record_id=name, response_hash=f"hash-{name}")

    split = SimpleNamespace(
        audit_auxiliary=[record("audit")],
        members=[record("member")],
        nonmembers=[record("nonmember")],
        draft_auxiliary=[record("draft")],
    )
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    artifact = {
        "data": {"pool_sha256": "b" * 64},
        "records": {
            "members": [{"response_hash": "hash-member"}],
            "nonmembers": [{"response_hash": "hash-nonmember"}],
            "auxiliary": [{"response_hash": "hash-draft"}],
            "audit_auxiliary": [{"response_hash": "hash-audit"}],
        },
    }
    (run_dir / "results.json").write_text(json.dumps(artifact))
    cfg = SimpleNamespace(
        draft_model="draft",
        draft_revision="revision",
        target_model="target",
        benchmark="wikitection",
        pool_path=tmp_path / "pool.jsonl",
        n_per_class=1,
        n_aux=1,
        n_audit_aux=1,
        data_seed=7,
    )
    tokenizer = SimpleNamespace(pad_token_id=0, eos_token_id=0)
    monkeypatch.setattr(
        scoring_common.AutoTokenizer,
        "from_pretrained",
        lambda *args, **kwargs: tokenizer,
    )
    monkeypatch.setattr(scoring_common, "verify_shared_tokenizer", lambda *args: None)
    monkeypatch.setattr(scoring_common, "build_controlled_split", lambda *args, **kwargs: split)

    _cfg, prepared = scoring_common.prepare_deployment_scoring_records(
        run_dir, cfg
    )

    assert prepared.record_ids.tolist() == ["audit", "member", "nonmember"]
    assert prepared.record_roles.tolist() == [
        "audit_auxiliary",
        "member",
        "nonmember",
    ]
    assert prepared.labels.tolist() == [0, 1, 0]
