"""Merge three isolated role archives into the standard p/q archive format."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

from experiments.sd_membership_sft.finetune.generalization import load_run_config
from experiments.sd_membership_sft.core.scoring_common import resolve_checkpoint_path

from experiments.paths import ROOT
ROLES = ("target", "draft_auxiliary_distilled", "draft_member_sft")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--target", type=Path, required=True)
    parser.add_argument("--draft-aux", type=Path, required=True)
    parser.add_argument("--draft-member", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def resolve(path: Path) -> Path:
    return path if path.is_absolute() else ROOT / path


def _scalar_string(value: np.ndarray, name: str, role: str) -> str:
    array = np.asarray(value)
    if array.ndim != 0:
        raise RuntimeError(f"{name} for {role} must be a scalar")
    return str(array.item())


def _load_sidecar(path: Path, role: str) -> dict[str, Any]:
    sidecar = path.with_suffix(path.suffix + ".json")
    if not sidecar.exists():
        raise RuntimeError(f"Missing provenance sidecar for {role}: {sidecar}")
    try:
        metadata = json.loads(sidecar.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"Invalid provenance sidecar for {role}: {sidecar}") from exc
    required = {
        "run_dir",
        "benchmark",
        "epoch",
        "role",
        "model_source",
        "base_model",
        "checkpoint_path",
        "checkpoint_kind",
        "records",
        "tokens",
    }
    missing = sorted(required - metadata.keys())
    if missing:
        raise RuntimeError(f"Provenance sidecar for {role} is missing {missing}")
    return metadata


def _validate_archive(role: str, path: Path, archive: Any) -> dict[str, Any]:
    required = {"role", "labels", "record_ids", "lengths", "logp"}
    missing = sorted(required - set(archive.files))
    if missing:
        raise RuntimeError(f"Archive for {role} is missing {missing}: {path}")
    actual_role = _scalar_string(archive["role"], "role", role)
    if actual_role != role:
        raise RuntimeError(
            f"Archive role mismatch for {path}: declared {actual_role!r}, expected {role!r}"
        )
    metadata = _load_sidecar(path, role)
    if metadata["role"] != role:
        raise RuntimeError(
            f"Sidecar role mismatch for {path}: declared {metadata['role']!r}, expected {role!r}"
        )
    expected_source = "target_checkpoint" if role == "target" else f"{role}_checkpoint"
    if metadata["model_source"] != expected_source:
        raise RuntimeError(
            f"Sidecar model source mismatch for {role}: {metadata['model_source']!r}"
        )
    checkpoint = Path(str(metadata["checkpoint_path"])).resolve()
    if not checkpoint.exists() or checkpoint.name != role:
        raise RuntimeError(
            f"Sidecar checkpoint path for {role} is not the expected existing role checkpoint: "
            f"{checkpoint}"
        )
    if metadata["checkpoint_kind"] not in {"checkpoints", "adapters"}:
        raise RuntimeError(f"Unexpected checkpoint kind for {role}: {metadata['checkpoint_kind']!r}")
    if checkpoint.parent.name != metadata["checkpoint_kind"]:
        raise RuntimeError(f"Checkpoint kind/path mismatch for {role}: {checkpoint}")
    if int(metadata["records"]) != len(archive["labels"]):
        raise RuntimeError(f"Record count mismatch between archive and sidecar for {role}")
    if int(metadata["tokens"]) != len(archive["logp"]):
        raise RuntimeError(f"Token count mismatch between archive and sidecar for {role}")
    if len(archive["lengths"]) != len(archive["labels"]):
        raise RuntimeError(f"lengths and labels disagree for {role}")
    if int(np.sum(archive["lengths"])) != len(archive["logp"]):
        raise RuntimeError(f"lengths and logp disagree for {role}")
    return metadata


def main() -> None:
    args = parse_args()
    paths = {
        "target": resolve(args.target),
        "draft_auxiliary_distilled": resolve(args.draft_aux),
        "draft_member_sft": resolve(args.draft_member),
    }
    archives = {role: np.load(path, allow_pickle=False) for role, path in paths.items()}
    metadata = {
        role: _validate_archive(role, paths[role], archive)
        for role, archive in archives.items()
    }
    reference = archives["target"]
    reference_metadata = metadata["target"]
    for role, archive in archives.items():
        for key in ("labels", "record_ids", "lengths"):
            if not np.array_equal(archive[key], reference[key]):
                raise RuntimeError(f"{key} differs for {role}")
        for key in ("run_dir", "benchmark", "epoch", "checkpoint_kind"):
            if metadata[role][key] != reference_metadata[key]:
                raise RuntimeError(
                    f"{key} differs between target and {role}: "
                    f"{reference_metadata[key]!r} != {metadata[role][key]!r}"
                )
    run_dir = Path(str(reference_metadata["run_dir"])).resolve()
    cfg = load_run_config(run_dir)
    if str(cfg.benchmark) != str(reference_metadata["benchmark"]):
        raise RuntimeError(
            f"Run config benchmark {cfg.benchmark!r} disagrees with archive provenance "
            f"{reference_metadata['benchmark']!r}"
        )
    if int(cfg.target_epochs) != int(reference_metadata["epoch"]):
        raise RuntimeError(
            f"Run config epoch {cfg.target_epochs} disagrees with archive provenance "
            f"{reference_metadata['epoch']}"
        )
    for role, role_metadata in metadata.items():
        checkpoint = Path(str(role_metadata["checkpoint_path"])).resolve()
        try:
            checkpoint.relative_to(run_dir)
        except ValueError as exc:
            raise RuntimeError(
                f"Checkpoint for {role} is outside the declared run_dir: {checkpoint}"
            ) from exc
        expected_checkpoint = resolve_checkpoint_path(run_dir, role)
        if checkpoint != expected_checkpoint:
            raise RuntimeError(
                f"Checkpoint provenance for {role} does not match the run config: "
                f"{checkpoint} != {expected_checkpoint}"
            )
        expected_base_model = cfg.target_model if role == "target" else cfg.draft_model
        if str(role_metadata["base_model"]) != str(expected_base_model):
            raise RuntimeError(
                f"Base model provenance for {role} does not match run config: "
                f"{role_metadata['base_model']!r} != {expected_base_model!r}"
            )
    output_dir = resolve(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    values = {role: np.asarray(archive["logp"], dtype=np.float32) for role, archive in archives.items()}
    np.savez_compressed(
        output_dir / "pq_gap_token_logps.npz",
        lengths=np.asarray(reference["lengths"], dtype=np.int64),
        **values,
    )
    np.savez_compressed(
        output_dir / "pq_gap_scores.npz",
        labels=np.asarray(reference["labels"], dtype=np.int64),
        record_ids=np.asarray(reference["record_ids"]),
    )
    (output_dir / "pq_gap_provenance.json").write_text(
        json.dumps(
            {
                "run_dir": str(run_dir),
                "benchmark": reference_metadata["benchmark"],
                "epoch": int(reference_metadata["epoch"]),
                "roles": metadata,
                "records": len(reference["labels"]),
                "tokens": int(np.sum(reference["lengths"])),
            },
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    print(f"merged {len(reference['labels'])} records into {output_dir}")


if __name__ == "__main__":
    main()
