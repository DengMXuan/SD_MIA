from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
from huggingface_hub import snapshot_download
from peft import PeftModel
from transformers import AutoTokenizer

from .draver_activation import (
    evaluate_activation_audit,
    extract_draft_activation_outputs,
    extract_target_token_outputs,
    make_audit_split,
)
from .public_data import build_public_snapshot_split
from .training import load_causal_lm, set_seed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Re-evaluate an independent-draft deployment endpoint from saved "
            "adapters after a feature-definition correction."
        )
    )
    parser.add_argument("--gpu", type=int, required=True)
    parser.add_argument("--source-results", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--cpu-threads", type=int, default=8)
    return parser.parse_args()


def _resolve(root: Path, path: Path) -> Path:
    return path if path.is_absolute() else root / path


def _load_tokenizer(model: str, revision: str) -> Any:
    snapshot = snapshot_download(
        repo_id=model, revision=revision, local_files_only=True
    )
    tokenizer = AutoTokenizer.from_pretrained(snapshot, local_files_only=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    return tokenizer


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable")
    torch.set_num_threads(args.cpu_threads)
    torch.set_num_interop_threads(min(4, args.cpu_threads))
    root = Path(__file__).resolve().parents[2]
    source_path = _resolve(root, args.source_results)
    output_dir = _resolve(root, args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=False)
    source = json.loads(source_path.read_text(encoding="utf-8"))
    config = source["config"]
    if config["primary_deployment_endpoint"] != "deployment_aligned_aux_distilled":
        raise ValueError("source primary endpoint must be auxiliary-distilled")
    source_dir = source_path.parent
    target_adapter = source_dir / source["adapter_paths"]["target"]
    draft_adapter = source_dir / source["adapter_paths"]["draft_aux_distilled"]

    device = torch.device(f"cuda:{args.gpu}")
    torch.cuda.set_device(device)
    torch.cuda.reset_peak_memory_stats(device)
    set_seed(int(config["seed"]))
    tokenizer = _load_tokenizer(config["draft_model"], config["draft_revision"])
    dataset = _resolve(root, Path(config["dataset"]))
    members, nonmembers, auxiliary, data_metadata = build_public_snapshot_split(
        dataset,
        tokenizer,
        int(config["response_tokens"]),
        int(config["n_per_class"]),
        int(config["n_aux"]),
        int(config["resolved_split_seed"]),
    )
    candidates = members + nonmembers
    labels = np.concatenate(
        [
            np.ones(len(members), dtype=np.int64),
            np.zeros(len(nonmembers), dtype=np.int64),
        ]
    )
    calibration, test = make_audit_split(
        len(members),
        len(nonmembers),
        int(config["audit_train_per_class"]),
        int(config["resolved_split_seed"]) + 30,
    )
    started = time.time()
    target = PeftModel.from_pretrained(
        load_causal_lm(
            config["target_model"],
            device,
            revision=config["target_revision"],
            local_files_only=True,
        ),
        str(target_adapter),
    ).to(device)
    draft = PeftModel.from_pretrained(
        load_causal_lm(
            config["draft_model"],
            device,
            revision=config["draft_revision"],
            local_files_only=True,
        ),
        str(draft_adapter),
    ).to(device)
    draft_outputs = extract_draft_activation_outputs(
        draft,
        candidates,
        tokenizer,
        device,
        int(config["draft_batch_size"]),
    )
    target_outputs = extract_target_token_outputs(
        target,
        candidates,
        tokenizer,
        device,
        int(config["target_batch_size"]),
    )
    endpoint = evaluate_activation_audit(
        draft_outputs,
        target_outputs,
        labels,
        calibration,
        test,
        float(config["min_k_fraction"]),
        int(config["transcript_repeats"]),
        int(config["bootstrap_repeats"]),
        int(config["detector_seeds"]),
        int(config["seed"]) + 12000,
    )
    artifact = {
        "material_passport": {
            "experiment_id": source["material_passport"]["experiment_id"]
            + "-corrected-transcript",
            "status": "COMPLETED",
            "verification_status": "ANALYZED_SAVED_ADAPTER_REMEASUREMENT",
        },
        "source_results": str(source_path),
        "feature_correction": (
            "transcript-only excludes q_logp; draft difficulty remains available "
            "to the nuisance fit and white-box feature families"
        ),
        "config": config,
        "data": data_metadata,
        "training": {
            "reused_adapters": True,
            "target_adapter": str(target_adapter),
            "draft_adapter": str(draft_adapter),
            "source_training": source["training"],
        },
        "alignment_recovery": source["alignment_recovery"],
        "pair_alignment": source["pair_alignment"],
        "endpoint": endpoint,
        "runtime": {
            "device": torch.cuda.get_device_name(device),
            "visible_cuda_devices": os.environ.get("CUDA_VISIBLE_DEVICES", ""),
            "duration_seconds": time.time() - started,
            "peak_gpu_memory_gib": torch.cuda.max_memory_allocated(device)
            / (1024**3),
        },
    }
    (output_dir / "results.json").write_text(
        json.dumps(artifact, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(
        json.dumps(
            {
                "output_dir": str(output_dir),
                "runtime": artifact["runtime"],
                "paired_auc_deltas": endpoint["paired_auc_deltas"],
            },
            indent=2,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
