"""Owned output directories and atomic, provenance-bound DP stages."""
from __future__ import annotations

from contextlib import contextmanager
import fcntl
import json
from pathlib import Path
import shutil

from experiments.sd_membership_sft.audit_runtime import _write_json
from experiments.sd_membership_sft.deployment_archive import sha256_file, checkpoint_fingerprint
from experiments.sd_membership_sft.matrix_artifacts import digest, runtime_files

ROLES = ("target", "draft_auxiliary_distilled", "draft_member_sft")


def dp_runtime_files():
    return sorted(Path(__file__).parent.glob("*.py"))


def code_sources():
    from experiments.cross_model_audit.artifacts import runtime_files as audit_runtime_files
    return [{"path": str(p.resolve()), "sha256": sha256_file(p)}
            for p in [*audit_runtime_files(), *dp_runtime_files()]]


@contextmanager
def owned_run(output: Path, request: dict):
    """Never adopt an existing non-DP directory or silently replace a request."""
    output = output.resolve()
    output.mkdir(parents=True, exist_ok=True, mode=0o700)
    with (output / ".dp.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        path = output / "DP_REQUEST.json"
        if path.exists():
            if json.loads(path.read_text()) != request:
                raise ValueError("DP request/source changed; use a new output directory")
        else:
            if any(p.name != ".dp.lock" for p in output.iterdir()):
                raise ValueError("refusing to adopt a nonempty directory without DP provenance")
            _write_json(path, request)
        yield output


def validate_weights(path: Path):
    from safetensors import safe_open
    if not (path / "config.json").is_file() or (path / "adapter_config.json").exists():
        raise ValueError("DP stage must contain a full model checkpoint")
    index = path / "model.safetensors.index.json"
    if index.exists():
        names = set(json.loads(index.read_text())["weight_map"].values())
        if not names:
            raise ValueError("empty checkpoint shard index")
        for name in names:
            if Path(name).name != name:
                raise ValueError("invalid checkpoint shard path")
    else:
        names = {"model.safetensors"}
    for name in names:
        with safe_open(path / name, framework="np") as handle:
            if not list(handle.keys()):
                raise ValueError("empty model shard")


def stage_key(request, role, teacher_sha=None):
    if role not in ROLES:
        raise ValueError("unknown DP stage")
    return digest({"request": request, "role": role, "teacher_sha256": teacher_sha})


def stage_directory(output, role):
    output = Path(output)
    path = output / "DP_REQUEST.json"
    request = json.loads(path.read_text()) if path.exists() else {}
    if role not in ROLES:
        raise ValueError("unknown DP stage")
    if request.get("head_pair") and role != "target":
        folder = "auxiliary_head" if role == "draft_auxiliary_distilled" else "member_head"
        return output / "heads" / folder
    return output / "checkpoints" / role


def read_stage(output, role, key):
    folder = stage_directory(output, role)
    if not folder.exists():
        return None
    marker = folder / "DP_STAGE.json"
    if not marker.exists():
        raise ValueError("checkpoint lacks DP completion/provenance marker")
    metadata = json.loads(marker.read_text())
    if metadata["key"] != key or metadata["role"] != role:
        raise ValueError("DP stage request or teacher mismatch")
    validate_weights(folder)
    expected = metadata["files"]
    actual = {str(p.relative_to(folder)): sha256_file(p)
              for p in folder.rglob("*") if p.is_file() and p != marker}
    if actual != expected:
        raise ValueError("DP checkpoint checksum mismatch")
    return {**metadata, "checkpoint_sha256": checkpoint_fingerprint(folder)}


def save_stage(output, role, key, model, tokenizer, privacy, *, marker=None, implementation=None):
    folder = stage_directory(output, role)
    if folder.exists():
        raise ValueError("refusing to overwrite a published DP stage")
    # Only this owned staging path may be replaced after a failed save. No
    # intermediate model is treated as a released/completed checkpoint.
    temporary = output / ".pending" / role
    if temporary.exists():
        shutil.rmtree(temporary)
    temporary.mkdir(parents=True)
    model.save_pretrained(temporary, safe_serialization=True)
    tokenizer.save_pretrained(temporary)
    if marker is not None:
        _write_json(temporary / "_COMPLETE.json", {"status": "complete", **marker})
    if implementation is not None:
        shutil.copyfile(implementation, temporary / "eagle3.py")
    validate_weights(temporary)
    files = {str(p.relative_to(temporary)): sha256_file(p)
             for p in temporary.rglob("*") if p.is_file()}
    _write_json(temporary / "DP_STAGE.json", {
        "schema": "sd_mia_dp_stage_v1", "key": key, "role": role,
        "files": files, "privacy": privacy,
    })
    folder.parent.mkdir(exist_ok=True)
    temporary.replace(folder)
    return read_stage(output, role, key)


def verify_run(output: Path):
    """Audit refuses relabeled ordinary checkpoints and incomplete DP training."""
    request = json.loads((output / "DP_REQUEST.json").read_text())
    artifact = json.loads((output / "results.json").read_text())
    if artifact["privacy"]["request_key"] != digest(request):
        raise ValueError("DP passport request mismatch")
    for source in request["sources"]:
        if sha256_file(Path(source["path"])) != source["sha256"]:
            raise ValueError(f"DP training source changed: {source['path']}")
    if request.get("source_head"):
        if checkpoint_fingerprint(Path(request["source_head"])) != request["source_head_sha256"]:
            raise ValueError("initial native MTP source changed")
    stages = {}
    for role in ROLES:
        teacher = (stages["target"]["checkpoint_sha256"]
                   if role == "draft_auxiliary_distilled" or (request.get("head_pair") and role == "draft_member_sft")
                   else None)
        stage = read_stage(output, role, stage_key(request, role, teacher))
        if stage is None or stage != artifact["privacy"]["stages"][role]:
            raise ValueError("DP stage missing or differs from training passport")
        stages[role] = stage
    from .accounting import epsilon_for, pair_budgets
    for role in ("target", "draft_member_sft"):
        value = stages[role]["privacy"]
        if any(value.get(k) != v for k, v in request["plans"][role].items()):
            raise ValueError("completed DP stage differs from planned mechanism")
        if value["completed_steps"] != value["steps"]:
            raise ValueError("DP stage incomplete")
        actual = epsilon_for(value["noise_multiplier"], value["sample_rate"], value["steps"], value["delta"])
        if abs(actual - value["accounted_epsilon"]) > 1e-8 or actual > value["epsilon"]:
            raise ValueError("DP accounting result does not match mechanism")
    pairs = pair_budgets(stages["target"]["privacy"], stages["draft_member_sft"]["privacy"])
    if pairs != artifact["privacy"]["pairs"]:
        raise ValueError("DP deployment-pair accounting mismatch")
    return artifact
