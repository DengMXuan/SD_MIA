"""Score one saved Qwen3 SFT role in an isolated process.

Keeping one role per process avoids the allocator peak caused by loading the
8B target and both 1.7B draft checkpoints sequentially in one Python process.
The output is a compact, role-specific teacher-forced log-probability archive
that can be merged without loading a model again.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch

from experiments.shared.training.generalization import load_draft_model, load_finetuned_model, load_run_config
from experiments.shared.models.token_scores import record_logprobabilities
from experiments.shared.core.scoring_common import prepare_scoring_records, role_provenance
from experiments.shared.training.training import set_seed

from experiments.paths import ROOT
ROLES = ("target", "draft_auxiliary_distilled", "draft_member_sft")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--role", choices=ROLES, required=True)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--empty-cache-each-batch", action="store_true")
    parser.add_argument(
        "--attn-implementation",
        choices=("eager", "sdpa"),
        default="eager",
    )
    parser.add_argument("--seed", type=int, default=20260824)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available")
    run_dir = args.run_dir if args.run_dir.is_absolute() else ROOT / args.run_dir
    output = args.output if args.output.is_absolute() else ROOT / args.output
    output.parent.mkdir(parents=True, exist_ok=True)
    cfg = load_run_config(run_dir)
    device = torch.device(f"cuda:{args.gpu}")
    torch.cuda.set_device(device)
    set_seed(args.seed)

    cfg, prepared = prepare_scoring_records(run_dir, cfg)
    tokenizer = prepared.tokenizer
    records, labels = prepared.records, prepared.labels
    record_ids = prepared.record_ids

    if args.role == "target":
        model = load_finetuned_model(
            run_dir,
            cfg.target_model,
            device,
            attn_implementation=args.attn_implementation,
        )
    else:
        model = load_draft_model(
            run_dir,
            cfg.draft_model,
            args.role,
            device,
            attn_implementation=args.attn_implementation,
        )
    started = time.perf_counter()
    try:
        logps = record_logprobabilities(
            model,
            records,
            tokenizer,
            device,
            args.batch_size,
            empty_cache_each_batch=args.empty_cache_each_batch,
        )
    finally:
        del model
        torch.cuda.empty_cache()
    lengths = np.asarray([len(row) for row in logps], dtype=np.int64)
    values = np.concatenate(logps).astype(np.float32)
    np.savez_compressed(
        output,
        role=np.asarray(args.role),
        labels=labels,
        record_ids=record_ids,
        lengths=lengths,
        logp=values,
    )
    (output.with_suffix(output.suffix + ".json")).write_text(
        json.dumps(
            {
                **role_provenance(cfg, run_dir, args.role),
                "records": len(records),
                "tokens": int(values.size),
                "seconds": time.perf_counter() - started,
                "scoring_runtime": {
                    "attention_implementation": args.attn_implementation,
                    "batch_size": args.batch_size,
                    "model_dtype": "bfloat16",
                    "vocabulary_normalizer_dtype": "float32",
                    "empty_cache_each_batch": args.empty_cache_each_batch,
                    "seed": args.seed,
                },
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(json.dumps({"output": str(output), "tokens": int(values.size)}), flush=True)


if __name__ == "__main__":
    main()
