"""Shared helpers for the head-based draft approaches (EAGLE-3 / MTP).

``drafts.plain`` is self-contained; the two head-based approaches share
the model-pair registry, the split loader, and the run-config writer
here. Results stay under ``artifacts/training/protocol_ft/runs`` for
continuity with the existing run artifacts.
"""

from __future__ import annotations

import argparse
import gc
import json
import os
import tempfile
import time
from pathlib import Path
from typing import Any, Callable

import torch
from transformers import AutoTokenizer

from experiments.shared.training.config import Config
from experiments.shared.core.data_contract import DEFAULT_DATA_CONTRACT
from experiments.shared.data.splits import CONTROLLED_SPLIT_SCHEMA_VERSION, build_controlled_split_from_shared_manifest, pool_path
from experiments.shared.training.training import set_seed

from experiments.paths import ROOT
DATA_ROOT = Path("artifacts/data/pools")
RESULTS_ROOT = Path("artifacts/training/protocol_ft/runs")
KD_TEMPERATURE = 2.0
KD_STEPS = 384
KD_LR = 2e-5
EFFECTIVE_BATCH_SIZE = 16
CHECKPOINT_MARKER = "_COMPLETE.json"


from experiments.shared.models.catalog import HEAD_PAIRS as PAIR_MODELS


def run_dir_for(args: Any) -> Path:
    from experiments.paths import prepare_training_storage
    path = args.output_dir if args.output_dir.is_absolute() else ROOT / args.output_dir
    prepare_training_storage(path)
    return path


def load_split(tokenizer: Any, args: Any):
    pool = pool_path(args.benchmark)
    manifest = args.split_manifest
    if not manifest.is_absolute():
        manifest = ROOT / manifest
    split = build_controlled_split_from_shared_manifest(
        args.benchmark,
        ROOT / pool,
        tokenizer,
        manifest,
        tokenizer_source_for(args.pair),
    )
    metadata = split.metadata
    audit_path = manifest.with_suffix(".audit.json")
    if not audit_path.is_file():
        raise RuntimeError(f"Shared split has no preflight audit: {audit_path}")
    audit = json.loads(audit_path.read_text(encoding="utf-8"))
    source = tokenizer_source_for(args.pair)
    tokenizer_audit = audit.get("tokenizers", {}).get(source)
    if (
        audit.get("shared_split_schema_version") != CONTROLLED_SPLIT_SCHEMA_VERSION
        or tokenizer_audit is None
        or tokenizer_audit.get("shared_split_sha256")
        != metadata["shared_split_sha256"]
        or tokenizer_audit.get("cross_split_ngram_audit", {}).get("gate") != "PASS"
    ):
        raise RuntimeError(
            f"Shared split audit is missing or stale for {source}: {audit_path}"
        )
    if metadata["split_seed"] != args.data_seed:
        raise RuntimeError(
            f"Shared split seed {metadata['split_seed']} does not match "
            f"data_seed {args.data_seed}"
        )
    expected = {
        "member": args.n_per_class,
        "nonmember": args.n_per_class,
        "auxiliary": args.n_aux,
        "audit_auxiliary": args.n_audit_aux,
    }
    if metadata["counts"] != expected:
        raise RuntimeError(
            f"Shared split counts {metadata['counts']} do not match {expected}"
        )
    # Model training deliberately receives only the three model-facing roles.
    # The independent audit auxiliary is required and included in the split
    # audit above, but it is reserved for downstream detector fitting.
    return split.members, split.nonmembers, split.draft_auxiliary, metadata


def device_for(args: Any) -> torch.device:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available")
    device = torch.device(f"cuda:{args.gpu}")
    torch.cuda.set_device(device)
    set_seed(args.seed)
    return device


def tokenizer_for(pair: str) -> Any:
    model = PAIR_MODELS[pair]
    tokenizer = AutoTokenizer.from_pretrained(
        model["target"],
        revision=model["target_revision"],
        local_files_only=True,
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    return tokenizer


def tokenizer_source_for(pair: str) -> str:
    model = PAIR_MODELS[pair]
    return f"{model['target']}@{model['target_revision']}"


def cached_snapshot(model_id: str, revision: str) -> Path:
    from huggingface_hub import snapshot_download

    return Path(
        snapshot_download(
            repo_id=model_id,
            revision=revision,
            local_files_only=True,
        )
    )


def build_parser(description: str, pairs: list[str]) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--pair", required=True, choices=pairs)
    parser.add_argument(
        "--benchmark",
        choices=["newstection", "wikitection", "arxivtection"],
        default="newstection",
    )
    parser.add_argument("--epochs", type=int, default=3, help="target SFT epochs")
    parser.add_argument("--data-seed", type=int, default=20260824)
    parser.add_argument("--seed", type=int, default=20260824)
    parser.add_argument(
        "--n-per-class", type=int, default=DEFAULT_DATA_CONTRACT.members
    )
    parser.add_argument(
        "--n-aux", type=int, default=DEFAULT_DATA_CONTRACT.draft_auxiliary
    )
    parser.add_argument(
        "--n-audit-aux", type=int, default=DEFAULT_DATA_CONTRACT.audit_auxiliary
    )
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--grad-accum", type=int, default=8)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument(
        "--head-updates",
        "--kd-steps",
        dest="head_updates",
        type=int,
        default=KD_STEPS,
        help="optimizer updates per head branch (not micro-batches)",
    )
    parser.add_argument(
        "--head-lr", "--kd-lr", dest="head_lr", type=float, default=KD_LR
    )
    parser.add_argument(
        "--head-batch-size",
        "--kd-batch-size",
        dest="head_batch_size",
        type=int,
        default=2,
    )
    parser.add_argument(
        "--head-grad-accum",
        "--kd-grad-accum",
        dest="head_grad_accum",
        type=int,
        default=8,
    )
    parser.add_argument("--kd-temperature", type=float, default=KD_TEMPERATURE)
    parser.add_argument("--split-manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=None)
    return parser


def resolve_output_dir(args: argparse.Namespace) -> None:
    if args.output_dir is None:
        tag = "" if args.lr == 2e-5 else f"_lr{args.lr:g}"
        bench = "" if args.benchmark == "newstection" else f"_{args.benchmark}"
        args.output_dir = (
            RESULTS_ROOT
            / args.pair
            / f"{args.pair}{bench}{tag}_epoch{args.epochs}_seed{args.seed}"
        )


def validate_training_contract(args: argparse.Namespace) -> None:
    if args.data_seed != args.seed:
        raise ValueError("data_seed and seed must be identical for this matrix")
    if args.batch_size * args.grad_accum != EFFECTIVE_BATCH_SIZE:
        raise ValueError(
            "target batch_size * grad_accum must equal "
            f"{EFFECTIVE_BATCH_SIZE}"
        )
    if args.head_batch_size * args.head_grad_accum != EFFECTIVE_BATCH_SIZE:
        raise ValueError(
            "head batch_size * head_grad_accum must equal "
            f"{EFFECTIVE_BATCH_SIZE}"
        )
    if args.head_updates <= 0:
        raise ValueError("head_updates must be positive")
    if args.kd_temperature <= 0:
        raise ValueError("kd_temperature must be positive")


def run_command(args: argparse.Namespace, handlers: dict[str, Callable[[Any], None]]) -> None:
    started = time.time()
    validate_training_contract(args)
    handler = handlers.get(args.command)
    if handler is None:
        raise ValueError(args.command)
    handler(args)
    gc.collect()
    torch.cuda.empty_cache()
    print(f"[{args.command}] total {time.time() - started:.0f}s", flush=True)


def write_run_config(args: Any, extra: dict[str, Any]) -> None:
    """results.json whose config block matches Config for generalization.py."""
    run_dir = run_dir_for(args)
    cfg = Config().as_dict()
    cfg.update(
        {
            "gpu": args.gpu,
            "seed": args.seed,
            "data_seed": args.data_seed,
            "target_model": PAIR_MODELS[args.pair]["target"],
            "target_revision": PAIR_MODELS[args.pair]["target_revision"],
            "draft_model": PAIR_MODELS[args.pair]["speculator"],
            "draft_revision": PAIR_MODELS[args.pair]["speculator_revision"],
            "benchmark": args.benchmark,
            "pool_path": ROOT / DATA_ROOT / args.benchmark / "pool.jsonl",
            "n_per_class": args.n_per_class,
            "n_aux": args.n_aux,
            "n_audit_aux": args.n_audit_aux,
            "target_epochs": args.epochs,
            "target_batch_size": args.batch_size,
            "target_grad_accum": args.grad_accum,
            "target_lr": args.lr,
            "draft_batch_size": args.head_batch_size,
            "draft_grad_accum": args.head_grad_accum,
            "draft_lr": args.head_lr,
            "distill_steps": args.head_updates,
            "distill_temperature": args.kd_temperature,
            "output_dir": run_dir,
        }
    )
    result_path = run_dir / "results.json"
    artifact: dict[str, Any] = {}
    if result_path.is_file():
        artifact = json.loads(result_path.read_text(encoding="utf-8"))
    artifact["protocol_track"] = {
        "pair": args.pair,
        "target_frozen_before_heads": True,
        "shared_raw_split": str(args.split_manifest),
    }
    artifact["config"] = cfg
    stage_name = str(extra.get("stage", args.command))
    artifact.setdefault("stages", {})[stage_name] = extra
    run_dir.mkdir(parents=True, exist_ok=True)
    temporary = result_path.with_name(f".{result_path.name}.tmp.{os.getpid()}")
    temporary.write_text(
        json.dumps(artifact, indent=2, ensure_ascii=False, default=str),
        encoding="utf-8",
    )
    os.replace(temporary, result_path)


def checkpoint_complete(path: Path) -> bool:
    """Return true only for a finalized checkpoint with config and weights."""
    if not (path / CHECKPOINT_MARKER).is_file() or not (path / "config.json").is_file():
        return False
    patterns = (
        "*.safetensors",
        "pytorch_model.bin",
        "pytorch_model.bin.index.json",
    )
    return any(any(path.glob(pattern)) for pattern in patterns)


def partial_checkpoint_path(destination: Path) -> Path:
    destination.parent.mkdir(parents=True, exist_ok=True)
    candidate = tempfile.mkdtemp(
        prefix=f".{destination.name}.partial.", dir=destination.parent
    )
    path = Path(candidate)
    path.rmdir()
    return path


def promote_checkpoint(
    temporary: Path, destination: Path, metadata: dict[str, Any]
) -> None:
    """Validate and atomically promote a stage checkpoint.

    An older incomplete destination is preserved under a timestamped hidden
    name rather than being overwritten. The completion marker is written into
    the temporary directory before the single rename that publishes it.
    """
    if not (temporary / "config.json").is_file():
        raise RuntimeError(f"Checkpoint has no config.json: {temporary}")
    json.loads((temporary / "config.json").read_text(encoding="utf-8"))
    weight_patterns = (
        "*.safetensors",
        "pytorch_model.bin",
        "pytorch_model.bin.index.json",
    )
    if not any(any(temporary.glob(pattern)) for pattern in weight_patterns):
        raise RuntimeError(f"Checkpoint has no model weights: {temporary}")
    for index_name in (
        "model.safetensors.index.json",
        "pytorch_model.bin.index.json",
    ):
        index_path = temporary / index_name
        if not index_path.is_file():
            continue
        index = json.loads(index_path.read_text(encoding="utf-8"))
        missing_shards = sorted(
            {
                shard
                for shard in index.get("weight_map", {}).values()
                if not (temporary / shard).is_file()
            }
        )
        if missing_shards:
            raise RuntimeError(
                f"Checkpoint index {index_path} references missing shards: "
                f"{missing_shards}"
            )
    marker = {
        "status": "complete",
        "completed_unix_time": time.time(),
        **metadata,
    }
    (temporary / CHECKPOINT_MARKER).write_text(
        json.dumps(marker, indent=2, ensure_ascii=False, default=str) + "\n",
        encoding="utf-8",
    )
    if destination.exists():
        if checkpoint_complete(destination):
            raise FileExistsError(f"Complete checkpoint already exists: {destination}")
        archived = destination.with_name(
            f".{destination.name}.incomplete.{int(time.time())}.{os.getpid()}"
        )
        os.replace(destination, archived)
    os.replace(temporary, destination)
    if not checkpoint_complete(destination):
        raise RuntimeError(f"Promoted checkpoint failed validation: {destination}")


def save_pretrained_atomically(
    destination: Path,
    writer: Callable[[Path], None],
    metadata: dict[str, Any],
) -> None:
    temporary = partial_checkpoint_path(destination)
    writer(temporary)
    promote_checkpoint(temporary, destination, metadata)
