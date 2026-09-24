"""Read-only validation of completed head conditions, including relocated callers."""
from __future__ import annotations

import json
from pathlib import Path

from experiments.shared.core.deployment_archive import sha256_file
from experiments.shared.drafts.common import PAIR_MODELS, CHECKPOINT_MARKER

HEAD_ROLES = {"draft_auxiliary_distilled": ("aux_head", "auxiliary_head", "aux"),
              "draft_member_sft": ("member_head", "member_head", "member")}


class PendingHeadRun(ValueError):
    """Training has not atomically published every required stage yet."""


def validate_checkpoint(path):
    from safetensors import safe_open
    if not (path / CHECKPOINT_MARKER).is_file() or not (path / "config.json").is_file():
        raise PendingHeadRun(f"checkpoint incomplete: {path}")
    marker = json.loads((path / CHECKPOINT_MARKER).read_text())
    if marker.get("status") != "complete":
        raise PendingHeadRun(f"checkpoint not complete: {path}")
    index = path / "model.safetensors.index.json"
    names = set(json.loads(index.read_text())["weight_map"].values()) if index.exists() else {"model.safetensors"}
    if not names:
        raise ValueError("empty checkpoint index")
    for name in names:
        if Path(name).name != name:
            raise ValueError("invalid checkpoint shard path")
        if not (path / name).is_file():
            raise PendingHeadRun(f"missing shard: {path / name}")
        with safe_open(path / name, framework="np") as weights:
            if not list(weights.keys()):
                raise ValueError("empty checkpoint shard")
    return marker


def inspect_head_run(run_dir, *, kind=None):
    """Require all three stages so an active trainer cannot rewrite our passport."""
    run_dir = Path(run_dir).resolve()
    passport = run_dir / "results.json"
    if not passport.exists():
        raise PendingHeadRun(f"training passport missing: {passport}")
    artifact = json.loads(passport.read_text())
    track, cfg = artifact["protocol_track"], artifact["config"]
    pair = track["pair"]
    if pair not in PAIR_MODELS or (kind is not None and PAIR_MODELS[pair]["kind"] != kind):
        raise ValueError("head pair/adapter mismatch")
    spec = PAIR_MODELS[pair]
    if (not track.get("target_frozen_before_heads")
            or cfg["target_model"] != spec["target"] or cfg["target_revision"] != spec["target_revision"]
            or cfg["seed"] != cfg["data_seed"]):
        raise ValueError("head target identity/freeze/seed mismatch")
    stages = artifact.get("stages", {})
    if not {"target", "aux_head", "member_head"}.issubset(stages):
        raise PendingHeadRun("target/auxiliary/member training passport is incomplete")
    # Stage metadata already records absolute paths, unlike old protocol_track.
    data = stages["target"]["data"]
    manifest = Path(data["shared_split_manifest"])
    pool = Path(data["pool_path"])
    if not manifest.is_absolute() or not pool.is_absolute():
        raise ValueError("head passport must supply absolute split and pool paths")
    sha = sha256_file(manifest)
    audit_path = manifest.with_suffix(".audit.json")
    audit = json.loads(audit_path.read_text())
    token_source = f"{spec['target']}@{spec['target_revision']}"
    attestation = audit["tokenizers"][token_source]
    if (data["shared_split_sha256"] != sha or attestation["shared_split_sha256"] != sha
            or attestation["cross_split_ngram_audit"]["gate"] != "PASS"):
        raise ValueError("head shared split/audit hash mismatch")
    expected = dict(member=2000, nonmember=2000, auxiliary=2000, audit_auxiliary=600)
    for stage in ("target", "aux_head", "member_head"):
        row = stages[stage]["data"]
        if (row["shared_split_sha256"] != sha or row["split_seed"] != cfg["seed"]
                or row["counts"] != expected or row["pool_sha256"] != data["pool_sha256"]
                or row["tokenizer_source"] != token_source):
            raise ValueError(f"{stage} training data contract differs")
    target = run_dir / "checkpoints/target"
    marker = validate_checkpoint(target)
    if (marker.get("stage") != "target" or marker.get("pair") != pair
            or marker.get("seed") != cfg["seed"] or marker.get("epochs") != cfg["target_epochs"]
            or marker.get("base_model") != spec["target"]
            or marker.get("data_seed") != cfg["data_seed"]
            or marker.get("base_revision") != spec["target_revision"]
            or not marker.get("full_parameter_sft")):
        raise ValueError("target completion marker differs from passport")
    paths = {"target": target}
    for role, (stage, folder, variant) in HEAD_ROLES.items():
        path = run_dir / "heads" / folder
        marker = validate_checkpoint(path)
        objective = "native-mtp-cross-entropy" if spec["kind"] == "mtp" and variant == "member" else "temperature-kl"
        if (marker.get("stage") != stage or marker.get("variant") != variant
                or marker.get("seed") != cfg["seed"] or not marker.get("target_frozen")
                or Path(marker["target_checkpoint"]).resolve() != target.resolve()
                or marker.get("initialized_from_revision") != spec["speculator_revision"]
                or marker.get("objective") != objective
                or marker.get("initialized_from") != spec["speculator"]
                or marker.get("optimizer_updates") != 384
                or marker.get("effective_batch_size") != 16
                or marker.get("learning_rate") != 2e-5
                or marker.get("temperature") != (None if spec["kind"] == "mtp" and variant == "member" else 2.)
                or (spec["kind"] == "mtp" and not marker.get("verifier_owned_weights_from_target"))):
            raise ValueError(f"{role} completion marker/source mismatch")
        paths[role] = path
    return dict(artifact=artifact, pair=pair, kind=spec["kind"], paths=paths,
                manifest=manifest, audit_path=audit_path, pool=pool, data=data)
