"""Collect paired original/truncated-prefix accept-only replay observations.

Candidates are the same final response tokens in both views, without EOS.
The fixed instruction stays unchanged; the counterfactual drops distant
response context while retaining its last --keep-context tokens. This is a
prefix-length intervention, not a guaranteed semantic-preserving rewrite.

Local saved models serve as an offline verifier simulator. Target logp is
used transiently to generate independent Bernoulli bits, never exported.
This is not a live SD endpoint or a natural serial decoding trajectory.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np
import torch

from experiments.shared.methods.conditional_accept_only import Observations, save_observations
from experiments.shared.data.data import SFTRecord, _hash_ids, prompt_prefix_ids
from experiments.shared.training.generalization import load_draft_model, load_finetuned_model
from experiments.shared.core.audit_runtime import _write_json
from experiments.shared.models.token_scores import record_logprobabilities
from experiments.shared.core.scoring_common import prepare_scoring_records, resolve_run_dir, role_provenance


def paired_records(
    records: list[SFTRecord], tokenizer: Any, *, candidate_tokens: int = 64, keep_context: int = 32,
) -> tuple[list[SFTRecord], list[SFTRecord]]:
    if candidate_tokens < 1 or keep_context < 1:
        raise ValueError("candidate and retained context lengths must be positive")
    original, truncated = [], []
    for record in records:
        if len(record.response_ids) <= candidate_tokens + keep_context:
            raise ValueError(f"record {record.record_id} has insufficient context for a genuine intervention")
        response = list(record.response_ids)
        context, candidates = response[:-candidate_tokens], response[-candidate_tokens:]
        instruction = prompt_prefix_ids(record, tokenizer)
        for output, retained in ((original, context), (truncated, context[-keep_context:])):
            prefix = instruction + retained
            output.append(replace(record, response_ids=tuple(candidates), prompt_ids=tuple(prefix),
                                  response_hash=_hash_ids(candidates), prompt_hash=_hash_ids(prefix), append_eos=False))
    return original, truncated


def simulate_paired_bits(logp: np.ndarray, logq: np.ndarray, lengths: np.ndarray, *, repeats: int, seed: int) -> Observations:
    """The sole boundary where paired target probabilities are consumed."""
    if repeats < 1 or logp.shape != logq.shape or logq.ndim != 2 or logq.shape[1] != 2:
        raise ValueError("paired log probabilities must be token by two views")
    if not np.all(np.isfinite(logp)) or np.any(logp > 1e-6):
        raise ValueError("invalid target log probabilities")
    bits = np.empty((*logq.shape, repeats), dtype=np.uint8)
    offsets = np.r_[0, np.cumsum(lengths)]
    if offsets[-1] != len(logq):
        raise ValueError("unaligned lengths")
    for index, (start, end) in enumerate(zip(offsets[:-1], offsets[1:])):
        for view in range(2):
            rng = np.random.default_rng(np.random.SeedSequence([seed, index, view, 1709]))
            # Repeat-major draws make lower budgets stable prefixes of higher
            # budgets, while views have independent verifier randomness.
            uniforms = rng.random((repeats, end - start)).T
            alpha = np.exp(np.minimum(0.0, logp[start:end, view] - logq[start:end, view]))
            bits[start:end, view] = uniforms < alpha[:, None]
    return Observations(logq, bits, lengths, ("original", "truncated_context"))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--candidate-tokens", type=int, default=64)
    parser.add_argument("--keep-context", type=int, default=32)
    parser.add_argument("--repeats", type=int, default=2, help="stored repeats per view, including original-only budget control")
    parser.add_argument("--seed", type=int, default=20260914)
    parser.add_argument("--replay-seeds", help="comma-separated seeds; reuse frozen-model forward passes and append _seedN to output stems")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    if min(args.repeats, args.batch_size, args.candidate_tokens, args.keep_context) < 1:
        parser.error("all size parameters must be positive")
    if args.output.suffix != ".npz":
        parser.error("--output must end in .npz")
    seeds = [int(value) for value in args.replay_seeds.split(",")] if args.replay_seeds else [args.seed]
    outputs = [args.output.with_name(f"{args.output.stem}_seed{seed}.npz") if args.replay_seeds else args.output for seed in seeds]
    if len(set(seeds)) != len(seeds) or any(path.exists() for path in outputs):
        parser.error("duplicate seeds or output already exists; choose new archive paths")
    torch.set_num_threads(4)
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable; paired extraction requires model inference. No observations were generated.")
    if device.type == "cuda":
        torch.cuda.set_device(device)
    run_dir = resolve_run_dir(args.run_dir)
    cfg, prepared = prepare_scoring_records(run_dir)
    original, truncated = paired_records(prepared.records, prepared.tokenizer,
                                         candidate_tokens=args.candidate_tokens, keep_context=args.keep_context)
    combined = original + truncated
    n = len(original)
    role_values = {}
    provenance = {}
    # Sequential role loading avoids simultaneously resident 8B + 1.7B models.
    for role in ("draft_auxiliary_distilled", "target"):
        print(f"Scoring {role}: {len(combined)} paired record views", flush=True)
        provenance[role] = role_provenance(cfg, run_dir, role)
        if role == "target":
            model = load_finetuned_model(run_dir, cfg.target_model, device, attn_implementation="sdpa")
        else:
            model = load_draft_model(run_dir, cfg.draft_model, role, device, attn_implementation="sdpa")
        model.requires_grad_(False)
        try:
            values = record_logprobabilities(model, combined, prepared.tokenizer, device, args.batch_size)
            if any(len(value) != args.candidate_tokens for value in values):
                raise RuntimeError("candidate scoring alignment failed")
            role_values[role] = np.column_stack((np.concatenate(values[:n]), np.concatenate(values[n:])))
        finally:
            del model
            gc.collect()
            if device.type == "cuda":
                torch.cuda.empty_cache()
    lengths = np.full(n, args.candidate_tokens, dtype=np.int64)
    manifest = {
        "protocol": "position-locked offline verifier simulation; no target probabilities exported",
        "intervention": "truncate response prefix, retain instruction and immediate response context",
        "candidate_tokens": args.candidate_tokens, "keep_context": args.keep_context,
        "repeats_per_view": args.repeats,
        "evaluation_cost": "original-only B versus B/2 original + B/2 truncated; unused stored bits excluded",
        "provenance": provenance, "models_frozen": True,
        "records": [{"record_id": item.record_id, "candidate_hash": item.response_hash,
                     "original_prefix_hash": item.prompt_hash, "truncated_prefix_hash": other.prompt_hash,
                     "original_prefix_length": len(item.prompt_ids), "truncated_prefix_length": len(other.prompt_ids)}
                    for item, other in zip(original, truncated)],
    }
    for seed, output in zip(seeds, outputs):
        obs = simulate_paired_bits(role_values["target"], role_values["draft_auxiliary_distilled"], lengths,
                                   repeats=args.repeats, seed=seed)
        save_observations(output, obs, prepared.labels, prepared.record_ids)
        _write_json(output.with_suffix(".json"), {**manifest, "seed": seed,
                    "collection_verifier_decisions": int(obs.bits.size),
                    "archive_sha256": hashlib.sha256(output.read_bytes()).hexdigest()})
        print(f"Saved observable-only archive: {output}", flush=True)


if __name__ == "__main__":
    main()
