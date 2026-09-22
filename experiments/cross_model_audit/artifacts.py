"""Cross-model provenance includes shared algorithms and this opt-in runner."""
import json
from pathlib import Path
from experiments.paths import ROOT
from experiments.sd_membership_sft.audit.matrix_artifacts import (
    digest, checkpoint_inventory, read_result, save_result, already_complete,
    runtime_files as shared_runtime_files,
)
from experiments.sd_membership_sft.core.deployment_archive import checkpoint_fingerprint, sha256_file


def runtime_files():
    extra = [p for p in Path(__file__).parent.rglob('*.py') if 'tests' not in p.parts]
    return sorted([*shared_runtime_files(), *extra,
                   ROOT / 'experiments/sd_membership_sft/scripts/model_pair_revisions.env'])


def sources_for(run_dir, checkpoint_roles, *, adapter="plain"):
    artifact = json.loads((run_dir / "results.json").read_text())
    files = [run_dir / "results.json", *runtime_files()]
    from experiments.cross_model_audit.model_registry import split_manifest
    manifest = split_manifest(artifact)
    manifest = manifest if manifest.is_absolute() else ROOT / manifest
    files += [manifest, manifest.with_suffix(".audit.json")]
    checkpoints = []
    for role in checkpoint_roles:
        if adapter != "plain" and role != "target":
            from experiments.cross_model_audit.models import checkpoint_paths
            path = checkpoint_paths(run_dir, adapter, role)[1].resolve()
        else:
            path = (run_dir / "checkpoints" / role).resolve()
        before = checkpoint_inventory(path)
        fingerprint = checkpoint_fingerprint(path)
        if before != checkpoint_inventory(path):
            raise ValueError(f"checkpoint changed during fingerprinting: {path}")
        checkpoints.append({"path": str(path), "sha256": fingerprint, "inventory": before})
    return {"files": [{"path": str(p.resolve()), "sha256": sha256_file(p)} for p in files],
            "checkpoints": checkpoints}

