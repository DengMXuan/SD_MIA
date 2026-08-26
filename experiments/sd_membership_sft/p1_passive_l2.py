"""Run the passive L2 speculative-decoding membership experiment.

The draft model samples an unconstrained four-token block.  The target model
then applies the standard speculative-sampling acceptance rule
``min(1, p / q)`` and samples a rejection correction from ``(p - q)+``.
This is deliberately separate from the earlier tomography pilot, which chose
candidate tokens instead of observing naturally sampled draft tokens.
"""

from __future__ import annotations

import argparse
import gc
import json
import math
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
from peft import PeftModel
from transformers import AutoTokenizer

from .audit import fit_logistic, metric_row
from .data import (
    SFTRecord,
    build_controlled_split,
    prompt_prefix_ids,
    records_metadata,
)
from .training import load_causal_lm, set_seed


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT_ROOT = ROOT / "experiments" / "results" / "qwen3_sft" / "p1_passive_l2"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--seed", type=int, default=20260824)
    parser.add_argument("--data-seed", type=int, default=20260824)
    parser.add_argument("--audit-seed", type=int, default=20260824)
    parser.add_argument("--target-model", default="Qwen/Qwen3-8B-Base")
    parser.add_argument("--draft-model", default="Qwen/Qwen3-1.7B-Base")
    parser.add_argument(
        "--target-adapter",
        type=Path,
        required=True,
        help="LoRA adapter produced by target SFT",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--n-per-class", type=int, default=160)
    parser.add_argument("--audit-train-per-class", type=int, default=48)
    parser.add_argument("--response-tokens", type=int, default=64)
    parser.add_argument("--context-tokens", type=int, default=48)
    parser.add_argument("--block-size", type=int, default=4)
    parser.add_argument("--rounds", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--bootstrap-repeats", type=int, default=200)
    return parser.parse_args()


def _pad_sequences(
    sequences: list[list[int]], pad_token_id: int, device: torch.device
) -> tuple[torch.Tensor, torch.Tensor, list[int]]:
    lengths = [len(sequence) for sequence in sequences]
    width = max(lengths)
    input_ids = torch.full(
        (len(sequences), width), pad_token_id, dtype=torch.long, device=device
    )
    attention_mask = torch.zeros(
        (len(sequences), width), dtype=torch.long, device=device
    )
    for row, sequence in enumerate(sequences):
        length = lengths[row]
        input_ids[row, :length] = torch.tensor(sequence, dtype=torch.long, device=device)
        attention_mask[row, :length] = 1
    return input_ids, attention_mask, lengths


def _load_target_with_adapter(
    model_id: str, adapter: Path, device: torch.device
) -> PeftModel:
    base = load_causal_lm(model_id, device)
    target = PeftModel.from_pretrained(base, str(adapter), is_trainable=False)
    target.eval()
    return target


def _natural_block_batch(
    draft: torch.nn.Module,
    target: torch.nn.Module,
    contexts: list[list[int]],
    pad_token_id: int,
    block_size: int,
    temperature: float,
    rng: np.random.Generator,
    torch_generator: torch.Generator,
    device: torch.device,
) -> list[dict[str, float]]:
    """Sample and verify one independent natural block per context."""

    batch_size = len(contexts)
    draft_sequences = [list(sequence) for sequence in contexts]
    q_log_probs: list[torch.Tensor] = []
    candidates: list[list[int]] = [[] for _ in contexts]

    # The q distribution at each position is retained only until the target
    # verifies this block.  Keeping a batch of distributions is inexpensive
    # for the configured batch size and is needed for the residual correction.
    with torch.inference_mode():
        for _ in range(block_size):
            input_ids, attention_mask, lengths = _pad_sequences(
                draft_sequences, pad_token_id, device
            )
            output = draft(
                input_ids=input_ids,
                attention_mask=attention_mask,
                use_cache=False,
            )
            last = output.logits[
                torch.arange(batch_size, device=device),
                torch.tensor(lengths, device=device) - 1,
            ].float()
            q_logp = torch.log_softmax(last / temperature, dim=-1)
            sampled = torch.multinomial(
                q_logp.exp(), 1, generator=torch_generator
            ).squeeze(1)
            sampled_ids = sampled.tolist()
            q_log_probs.append(q_logp.detach())
            for row, token_id in enumerate(sampled_ids):
                candidates[row].append(int(token_id))
                draft_sequences[row].append(int(token_id))
            del output, input_ids, attention_mask, last, q_logp, sampled

        full_sequences = [context + candidate for context, candidate in zip(contexts, candidates)]
        input_ids, attention_mask, _ = _pad_sequences(
            full_sequences, pad_token_id, device
        )
        output = target(
            input_ids=input_ids,
            attention_mask=attention_mask,
            use_cache=False,
        )
        target_log_probs: list[torch.Tensor] = []
        for position in range(block_size + 1):
            logits_position = torch.tensor(
                [len(contexts[row]) - 1 + position for row in range(batch_size)],
                device=device,
            )
            logits = output.logits[
                torch.arange(batch_size, device=device), logits_position
            ].float()
            target_log_probs.append(torch.log_softmax(logits / temperature, dim=-1))
        del output, input_ids, attention_mask

    rows: list[dict[str, float]] = []
    for row in range(batch_size):
        candidate_q_logp: list[float] = []
        candidate_p_logp: list[float] = []
        accepted = 0
        correction = 0
        correction_target_logp = math.nan
        correction_q_logp = math.nan
        correction_residual_mass = math.nan
        for position, token_id in enumerate(candidates[row]):
            q_distribution = q_log_probs[position][row]
            p_distribution = target_log_probs[position][row]
            q_lp = float(q_distribution[token_id].item())
            p_lp = float(p_distribution[token_id].item())
            candidate_q_logp.append(q_lp)
            candidate_p_logp.append(p_lp)
            acceptance_probability = min(1.0, math.exp(min(0.0, p_lp - q_lp)))
            if float(rng.random()) < acceptance_probability:
                accepted += 1
                continue

            correction = 1
            residual = torch.clamp(p_distribution.exp() - q_distribution.exp(), min=0.0)
            residual_mass = float(residual.sum().item())
            if residual_mass > 1e-12:
                correction_id = int(
                    torch.multinomial(
                        residual / residual_mass, 1, generator=torch_generator
                    ).item()
                )
                correction_residual_mass = residual_mass
            else:
                correction_id = int(
                    torch.multinomial(
                        p_distribution.exp(), 1, generator=torch_generator
                    ).item()
                )
                correction_residual_mass = 0.0
            correction_target_logp = float(p_distribution[correction_id].item())
            correction_q_logp = float(q_distribution[correction_id].item())
            break

        full_block = float(accepted == block_size)
        bonus_sampled = 0.0
        bonus_logp = math.nan
        if full_block:
            bonus_id = int(
                torch.multinomial(
                    target_log_probs[-1][row].exp(), 1, generator=torch_generator
                ).item()
            )
            bonus_sampled = 1.0
            bonus_logp = float(target_log_probs[-1][row, bonus_id].item())
        rows.append(
            {
                "accepted_tokens": float(accepted),
                "accepted_fraction": float(accepted / block_size),
                "full_block_accept": full_block,
                "correction": float(correction),
                "candidate_q_logp": float(np.mean(candidate_q_logp)),
                "candidate_target_logp": float(np.mean(candidate_p_logp)),
                "candidate_target_minus_q": float(
                    np.mean(np.asarray(candidate_p_logp) - np.asarray(candidate_q_logp))
                ),
                "correction_target_logp": correction_target_logp,
                "correction_q_logp": correction_q_logp,
                "correction_residual_mass": correction_residual_mass,
                "bonus_sampled": bonus_sampled,
                "bonus_logp": bonus_logp,
            }
        )
    del q_log_probs, target_log_probs
    return rows


def run_passive_l2(
    draft: torch.nn.Module,
    target: torch.nn.Module,
    records: list[SFTRecord],
    tokenizer: Any,
    config: argparse.Namespace,
    device: torch.device,
) -> list[dict[str, Any]]:
    if config.context_tokens + config.block_size > config.response_tokens:
        raise ValueError("context_tokens + block_size must not exceed response_tokens")
    contexts = [
        prompt_prefix_ids(record, tokenizer) + list(record.response_ids[: config.context_tokens])
        for record in records
    ]
    rng = np.random.default_rng(config.seed + 701)
    torch_generator = torch.Generator(device=device)
    torch_generator.manual_seed(config.seed + 702)
    all_rows: list[dict[str, Any]] = []
    for start in range(0, len(records), config.batch_size):
        batch_records = records[start : start + config.batch_size]
        batch_contexts = contexts[start : start + config.batch_size]
        aggregate: list[list[dict[str, float]]] = [[] for _ in batch_records]
        for _ in range(config.rounds):
            round_rows = _natural_block_batch(
                draft,
                target,
                batch_contexts,
                int(tokenizer.pad_token_id),
                config.block_size,
                config.temperature,
                rng,
                torch_generator,
                device,
            )
            for row, values in enumerate(round_rows):
                aggregate[row].append(values)
        for record, repetitions in zip(batch_records, aggregate):
            names = repetitions[0].keys()
            summarized: dict[str, float] = {}
            for name in names:
                values = np.asarray([item[name] for item in repetitions], dtype=np.float64)
                summarized[name] = (
                    math.nan if np.all(np.isnan(values)) else float(np.nanmean(values))
                )
            summarized["correction_observed_rounds"] = float(
                np.sum([item["correction"] for item in repetitions])
            )
            all_rows.append(
                {
                    "record_id": record.record_id,
                    "source": record.source,
                    "response_hash": record.response_hash,
                    "metrics": summarized,
                }
            )
        print(
            f"P1 processed {min(start + config.batch_size, len(records))}/{len(records)} records",
            flush=True,
        )
    return all_rows


def _feature_vector(rows: list[dict[str, Any]]) -> dict[str, np.ndarray]:
    values = {name: [] for name in rows[0]["metrics"]}
    for row in rows:
        for name, value in row["metrics"].items():
            values[name].append(value)
    return {name: np.asarray(items, dtype=np.float64) for name, items in values.items()}


def _audit(
    members: list[SFTRecord],
    nonmembers: list[SFTRecord],
    member_rows: list[dict[str, Any]],
    nonmember_rows: list[dict[str, Any]],
    config: argparse.Namespace,
) -> dict[str, Any]:
    labels = np.concatenate(
        [np.ones(len(members), dtype=np.int64), np.zeros(len(nonmembers), dtype=np.int64)]
    )
    features = _feature_vector(member_rows + nonmember_rows)
    rng = np.random.default_rng(config.audit_seed + 30)
    positive = rng.permutation(len(members))
    negative = rng.permutation(len(nonmembers)) + len(members)
    train_idx = np.concatenate(
        [positive[: config.audit_train_per_class], negative[: config.audit_train_per_class]]
    )
    test_idx = np.concatenate(
        [positive[config.audit_train_per_class :], negative[config.audit_train_per_class :]]
    )
    rng.shuffle(train_idx)
    rng.shuffle(test_idx)

    scores: dict[str, np.ndarray] = {
        "acceptance_rate": features["accepted_fraction"],
        "accepted_prefix_length": features["accepted_tokens"],
        "full_block_acceptance": features["full_block_accept"],
        "negative_correction_rate": -features["correction"],
        "candidate_q_logp": features["candidate_q_logp"],
        "candidate_target_logp": features["candidate_target_logp"],
        "candidate_target_minus_q": features["candidate_target_minus_q"],
    }
    results: dict[str, dict[str, float]] = {}
    for index, (name, score) in enumerate(scores.items()):
        results[name] = metric_row(
            labels[test_idx], score[test_idx], config.bootstrap_repeats, config.audit_seed + index
        )

    joint_names = [
        "accepted_fraction",
        "accepted_tokens",
        "full_block_accept",
        "correction",
        "candidate_q_logp",
        "candidate_target_logp",
        "candidate_target_minus_q",
    ]
    joint = np.column_stack([features[name] for name in joint_names])
    joint_score = fit_logistic(joint[train_idx], labels[train_idx], joint[test_idx])
    results["joint_passive_l2"] = metric_row(
        labels[test_idx], joint_score, config.bootstrap_repeats, config.audit_seed + 100
    )

    return {
        "metrics": results,
        "split": {
            "audit_train_per_class": config.audit_train_per_class,
            "audit_train_size": int(len(train_idx)),
            "audit_test_size": int(len(test_idx)),
            "audit_seed": config.audit_seed,
        },
        "record_features": member_rows + nonmember_rows,
    }


def render_summary(artifact: dict[str, Any]) -> str:
    cfg = artifact["config"]
    lines = [
        "# P1 Passive L2 Speculative-Decoding Membership Audit",
        "",
        "## Material Passport",
        "",
        "- Status: COMPLETED",
        "- Protocol: naturally sampled q block; target acceptance `min(1, p/q)`; residual correction `(p-q)+`",
        "- Membership target: adapter-based Qwen3 SFT, not Qwen3 pretraining membership",
        "",
        "## Protocol",
        "",
        f"- Target adapter: `{cfg['target_adapter']}`",
        f"- Context: first {cfg['context_tokens']} known response tokens",
        f"- Draft block: {cfg['block_size']} naturally sampled tokens; {cfg['rounds']} independent rounds/record",
        f"- Temperature: {cfg['temperature']}; records: {cfg['n_per_class']} per class",
        "- Candidate tokens were not fixed, optimized, or selected by the auditor",
        "",
        "## Held-out results",
        "",
        "| Signal | AUC (95% bootstrap CI) | TPR@1%FPR | TPR@5%FPR |",
        "|---|---:|---:|---:|",
    ]
    for name, row in artifact["audit"]["metrics"].items():
        lines.append(
            f"| `{name}` | {row['auc']:.3f} [{row['auc_ci95_low']:.3f}, {row['auc_ci95_high']:.3f}] "
            f"| {row['tpr_at_1pct_fpr']:.3f} | {row['tpr_at_5pct_fpr']:.3f} |"
        )
    lines.extend(
        [
            "",
            "## Interpretation boundary",
            "",
            "This run tests the passive L2 protocol under a local, honest target simulation.",
            "It does not establish leakage for an arbitrary API, a malicious/adaptive client, or pretraining data.",
            "The 160/class condition is a protocol-validity experiment; it is not sized for a low-FPR claim.",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> None:
    config = parse_args()
    if config.audit_train_per_class >= config.n_per_class:
        raise ValueError("audit_train_per_class must be smaller than n_per_class")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available; run with the approved host GPU access")
    if not config.target_adapter.is_absolute():
        config.target_adapter = ROOT / config.target_adapter
    if not config.output_dir.is_absolute():
        config.output_dir = ROOT / config.output_dir
    config.output_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device(f"cuda:{config.gpu}")
    torch.cuda.set_device(device)
    torch.cuda.reset_peak_memory_stats(device)
    set_seed(config.seed)
    tokenizer = AutoTokenizer.from_pretrained(config.draft_model)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    members, nonmembers, _, data_metadata = build_controlled_split(
        ROOT,
        tokenizer,
        config.response_tokens,
        config.n_per_class,
        0,
        config.data_seed,
    )
    print(f"device={torch.cuda.get_device_name(device)}", flush=True)
    started = time.time()
    target = _load_target_with_adapter(config.target_model, config.target_adapter, device)
    draft = load_causal_lm(config.draft_model, device)
    draft.eval()
    member_rows = run_passive_l2(draft, target, members, tokenizer, config, device)
    nonmember_rows = run_passive_l2(draft, target, nonmembers, tokenizer, config, device)
    audit = _audit(members, nonmembers, member_rows, nonmember_rows, config)
    artifact = {
        "material_passport": {
            "experiment_id": f"qwen3-passive-l2-{config.seed}",
            "status": "COMPLETED",
            "verification_status": "ANALYZED_PROTOCOL_VALIDITY",
        },
        "config": vars(config),
        "models": {"target": config.target_model, "draft": config.draft_model},
        "data": data_metadata,
        "records": {
            "members": records_metadata(members),
            "nonmembers": records_metadata(nonmembers),
        },
        "audit": audit,
        "runtime": {
            "duration_seconds": time.time() - started,
            "device": torch.cuda.get_device_name(device),
            "peak_gpu_memory_gib": torch.cuda.max_memory_allocated(device) / (1024**3),
        },
    }
    artifact["config"] = {key: str(value) if isinstance(value, Path) else value for key, value in artifact["config"].items()}
    (config.output_dir / "results.json").write_text(
        json.dumps(artifact, indent=2, ensure_ascii=False, allow_nan=True), encoding="utf-8"
    )
    (config.output_dir / "RESULTS.md").write_text(render_summary(artifact), encoding="utf-8")
    print(json.dumps({"output_dir": str(config.output_dir), "audit": audit["metrics"], "runtime": artifact["runtime"]}, indent=2, ensure_ascii=False), flush=True)
    del target, draft
    gc.collect()
    torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
