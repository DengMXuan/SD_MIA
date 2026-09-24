"""Read-only model/passport preflight shared by matrix and single-condition audits."""
import json
from pathlib import Path
from safetensors import SafetensorError
from experiments.paths import ROOT
from experiments.shared.core.deployment_archive import sha256_file
from .registry import pair_for, validate_head_passport, validate_identity

def ready(task):
    """Read-only preflight; complete passports, shards and shared split required."""
    run = Path(task["run_dir"])
    if not (run / "results.json").is_file():
        return False, "pending: training passport is not ready"
    try:
        artifact = json.loads((run / "results.json").read_text())
        cfg = artifact["config"]
        condition = task["condition"]
        spec = pair_for(task)
        if spec.is_head:
            return validate_head_passport(task, artifact)
        validate_identity(task, artifact)
        if artifact["material_passport"]["status"] != "COMPLETED":
            return False, "pending: training has not completed"
        expected = dict(benchmark=condition["benchmark"], target_epochs=condition["epoch"],
                        seed=condition["condition_seed"], data_seed=condition["condition_seed"],
                        target_model=spec.target, draft_model=spec.draft)
        if any(cfg.get(key) != value for key, value in expected.items()):
            raise ValueError("training condition/model identity mismatch")
        records = artifact["records"]
        for name, count in (("members", 2000), ("nonmembers", 2000), ("auxiliary", 2000), ("audit_auxiliary", 600)):
            if len(records[name]) != count:
                raise ValueError(f"wrong {name} count")
        ids = [row["record_id"] for rows in records.values() for row in rows]
        if len(set(ids)) != len(ids):
            raise ValueError("training/audit record IDs overlap")
        manifest = Path(artifact["data"]["shared_split_manifest"])
        manifest = manifest if manifest.is_absolute() else ROOT / manifest
        checksum = sha256_file(manifest)
        if checksum != artifact["data"]["shared_split_sha256"]:
            raise ValueError("shared split differs from training")
        audit = json.loads(manifest.with_suffix(".audit.json").read_text())
        token_source = f"{cfg['draft_model']}@{cfg['draft_revision']}"
        attestation = audit["tokenizers"][token_source]
        if attestation["shared_split_sha256"] != checksum or attestation["cross_split_ngram_audit"]["gate"] != "PASS":
            raise ValueError("shared split tokenizer audit is stale")
        roles = ("target",) if task["kind"] == "baseline" else ("target", task["draft_role"])
        from safetensors import safe_open
        for role in roles:
            folder = run / "checkpoints" / role
            if not (folder / "config.json").is_file():
                return False, f"pending: {role} config missing"
            shards = list(folder.glob("*.safetensors"))
            if not shards:
                return False, f"pending: {role} weights missing"
            index = folder / "model.safetensors.index.json"
            if index.exists():
                expected_shards = set(json.loads(index.read_text())["weight_map"].values())
                if not expected_shards.issubset({p.name for p in shards}):
                    return False, f"pending: {role} shards incomplete"
            for shard in shards:
                with safe_open(shard, framework="np") as weights:
                    if not list(weights.keys()):
                        raise ValueError(f"empty checkpoint {shard}")
        return True, "ready"
    except FileNotFoundError as error:
        return False, f"pending: {error}"
    except (OSError, ValueError, KeyError, TypeError, SafetensorError) as error:
        return False, f"invalid: {error}"
