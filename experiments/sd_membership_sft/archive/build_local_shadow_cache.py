"""Build a legitimate local shadow-IN/OUT cache from trusted nonmembers."""

from __future__ import annotations

import argparse
import gc
import json
from pathlib import Path

import numpy as np
import torch

from experiments.sd_membership_sft.core.audit_runtime import (_deterministic_subset)
from experiments.sd_membership_sft.core.audit_runtime import (split_indices)
from experiments.sd_membership_sft.finetune.generalization import (load_run_config)
from experiments.sd_membership_sft.core.audit_runtime import (BENCHMARKS, N_REF, ROOT, SPLIT_SEED)
from experiments.sd_membership_sft.analysis.pq_gap_mia import (record_logprobabilities)
from experiments.sd_membership_sft.core.scoring_common import (prepare_scoring_records)
from experiments.sd_membership_sft.finetune.training import (add_lora, load_causal_lm, set_seed, sft_train)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--benchmark", choices=BENCHMARKS, required=True)
    parser.add_argument("--gpu", type=int, required=True)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--seed", type=int, default=20260921)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "experiments/results/sft_runs/accept_only_active_v2/local_shadow",
    )
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for local shadow construction")
    run_dir = (
        ROOT
        / "experiments/results/sft_runs"
        / f"{args.benchmark}_qwen3_8b_epoch1"
    )
    cfg = load_run_config(run_dir)
    cfg, prepared = prepare_scoring_records(run_dir, cfg)
    labels = prepared.labels
    partitions = split_indices(labels, SPLIT_SEED)
    d_nm = partitions["D"][labels[partitions["D"]] == 0]
    reference = _deterministic_subset(d_nm, N_REF, SPLIT_SEED + N_REF)
    shuffled = np.random.default_rng(args.seed).permutation(reference)
    shadow_in = np.sort(shuffled[: N_REF // 2])
    shadow_out = np.sort(shuffled[N_REF // 2 :])
    selected = np.r_[shadow_in, shadow_out]
    shadow_labels = np.r_[
        np.ones(len(shadow_in), dtype=np.int64),
        np.zeros(len(shadow_out), dtype=np.int64),
    ]
    records = [prepared.records[int(index)] for index in selected]
    train_records = [prepared.records[int(index)] for index in shadow_in]
    train_batch_size = 1 if args.benchmark == "arxivtection" else 2
    grad_accum = 16 if args.benchmark == "arxivtection" else 8
    device = torch.device(f"cuda:{args.gpu}")
    torch.cuda.set_device(device)
    set_seed(args.seed)
    # SDPA avoids materializing the quadratic eager-attention matrix for the
    # native 1024--2048-token ArXiv records.
    model = load_causal_lm(cfg.draft_model, device, attn_implementation="sdpa")
    model.eval()
    q_rows = record_logprobabilities(
        model, records, prepared.tokenizer, device, batch_size=2, empty_cache_each_batch=False
    )
    model = add_lora(model, cfg.lora_r, cfg.lora_alpha, cfg.lora_dropout)
    history = sft_train(
        model,
        train_records,
        prepared.tokenizer,
        device,
        args.epochs,
        batch_size=train_batch_size,
        grad_accum=grad_accum,
        lr=2e-4,
        seed=args.seed + 1,
        label=f"{args.benchmark} local shadow",
        optimizer_name="adamw8bit",
    )
    p_rows = record_logprobabilities(
        model, records, prepared.tokenizer, device, batch_size=2, empty_cache_each_batch=False
    )
    q_lengths = np.asarray([len(row) for row in q_rows], dtype=np.int64)
    p_lengths = np.asarray([len(row) for row in p_rows], dtype=np.int64)
    if not np.array_equal(q_lengths, p_lengths):
        raise RuntimeError("shadow p/q token lengths differ")
    output = args.output_dir.resolve() / args.benchmark
    output.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output / "shadow_pq.npz",
        labels=shadow_labels,
        record_ids=prepared.record_ids[selected],
        source_indices=selected,
        lengths=q_lengths,
        logp=np.concatenate(p_rows).astype(np.float32),
        logq0=np.concatenate(q_rows).astype(np.float32),
    )
    (output / "metadata.json").write_text(
        json.dumps(
            {
                "benchmark": args.benchmark,
                "base_model": cfg.draft_model,
                "source": "400 trusted target-nonmembers only",
                "shadow_in": len(shadow_in),
                "shadow_out": len(shadow_out),
                "target_member_labels_used": 0,
                "epochs": args.epochs,
                "lr": 2e-4,
                "batch_size": train_batch_size,
                "gradient_accumulation": grad_accum,
                "seed": args.seed,
                "training_loss": history,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    del model
    gc.collect()
    torch.cuda.empty_cache()
    print(json.dumps({"output": str(output), "records": len(records)}), flush=True)


if __name__ == "__main__":
    main()
