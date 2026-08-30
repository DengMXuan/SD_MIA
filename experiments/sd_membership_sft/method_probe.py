"""Minimal method-optimization probes on saved public-benchmark adapters.

Three preregistered checks, all re-scoring a completed run's saved adapters
without any retraining:

``qcond`` (B)
    Extend the transcript-only feature summary with q-decile conditional
    acceptance rates and bottom-k% / top-k% alpha aggregations, then test
    whether the extended detector beats plain transcript-only at weak memory
    and low feedback budgets.

``paraphrase`` (A)
    Rewrite every audit record with a local instruct model, re-score the
    adapted pair on the paraphrased responses, and compare per-family AUC
    degradation against the original responses. The preregistered question is
    whether the activation branch degrades more slowly than the transcript.

``fe`` (D)
    Rebuild the same detectors with histogram and random-projection
    per-layer activation features instead of StatFE-lite, including raw
    terminal-token features, to test whether feature compression is the
    bottleneck of the activation branch.
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
from huggingface_hub import snapshot_download
from peft import PeftModel
from transformers import AutoTokenizer

from .audit import bottom_k_indices, metric_row
from .data import SFTRecord
from .draver_activation import (
    _conditioned_activation_features,
    _loader,
    _transcript_summary,
    build_activation_feature_families,
    cross_fitted_verifier_residual,
    extract_draft_activation_outputs,
    extract_target_token_outputs,
    gather_tokens,
    make_audit_split,
    paired_bootstrap_delta,
    q_stratified_shuffle,
    triplet_scores_by_seed,
)
from .public_budget_sweep import _nested_acceptance, _select_draft_endpoint, _validate_records
from .public_data import _hash_ids, build_public_snapshot_split
from .training import load_causal_lm, set_seed

PARAPHRASE_INSTRUCTION = (
    "Rewrite the following passage using different wording and sentence "
    "structure while strictly preserving its meaning, facts, names, and the "
    "order of ideas. Output only the rewritten passage without any preamble, "
    "heading, or commentary.\n\nPassage:\n"
)
QCOND_BINS = 10
QCOND_K_FRACTIONS = (0.1, 0.2, 0.3, 0.4, 0.5)
HIST_BINS = 32
HIST_RANGE = (-6.0, 6.0)
PROJECTION_DIM = 64
DEGENERATE_TOKENS = 48


# ---------------------------------------------------------------------------
# shared loading helpers


def _resolve(root: Path, path: Path) -> Path:
    return path if path.is_absolute() else root / path


def _load_source(source_path: Path) -> dict[str, Any]:
    source = json.loads(source_path.read_text(encoding="utf-8"))
    config = source["config"]
    required = (
        "dataset",
        "response_tokens",
        "n_per_class",
        "n_aux",
        "seed",
        "audit_train_per_class",
        "min_k_fraction",
        "target_model",
        "target_revision",
        "draft_model",
        "draft_revision",
    )
    missing = sorted(set(required) - set(config))
    if missing:
        raise ValueError(f"source results are missing config keys: {missing}")
    return source


def _reconstruct(
    source: dict[str, Any], source_path: Path
) -> tuple[Any, list[SFTRecord], list[SFTRecord], np.ndarray, np.ndarray, np.ndarray, dict[str, Any]]:
    config = source["config"]
    root = Path(__file__).resolve().parents[2]
    tokenizer_snapshot = snapshot_download(
        repo_id=config["draft_model"],
        revision=config["draft_revision"],
        local_files_only=True,
    )
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_snapshot, local_files_only=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    dataset_path = _resolve(root, Path(config["dataset"]))
    members, nonmembers, _, data_metadata = build_public_snapshot_split(
        dataset_path,
        tokenizer,
        int(config["response_tokens"]),
        int(config["n_per_class"]),
        int(config["n_aux"]),
        int(config["seed"]),
    )
    _validate_records(source, members, nonmembers)
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
        int(config["seed"]) + 30,
    )
    return tokenizer, members, nonmembers, labels, calibration, test, data_metadata


def _load_endpoint_draft(
    config: dict[str, Any],
    source_path: Path,
    device: torch.device,
) -> tuple[str, torch.nn.Module]:
    source_stub: dict[str, Any] = {"config": config}
    endpoint, adapter_path = _select_draft_endpoint(source_stub, source_path, "auto")
    model = load_causal_lm(
        config["draft_model"],
        device,
        revision=config["draft_revision"],
        local_files_only=True,
    )
    if adapter_path is not None:
        model = PeftModel.from_pretrained(model, adapter_path, is_trainable=False)
    return endpoint, model


def _load_adapted_target(
    config: dict[str, Any],
    source_path: Path,
    device: torch.device,
) -> torch.nn.Module:
    target = load_causal_lm(
        config["target_model"],
        device,
        revision=config["target_revision"],
        local_files_only=True,
    )
    return PeftModel.from_pretrained(
        target, source_path.parent / "adapter_target", is_trainable=False
    )


@torch.no_grad()
def extract_draft_position_states(
    model: torch.nn.Module,
    records: list[SFTRecord],
    tokenizer: Any,
    device: torch.device,
    batch_size: int,
    positions: np.ndarray,
) -> np.ndarray:
    """Gather raw all-layer hidden states at fixed response positions.

    ``positions`` maps each record to response-position indices (0-based,
    clipped to the record's valid response length). The result is a CPU float32
    array of shape ``(records, positions, layers, hidden_dim)``.
    """
    model.eval()
    total = len(records)
    layers = None
    hidden_dim = None
    result: np.ndarray | None = None
    start = 0
    for batch in _loader(records, tokenizer, batch_size):
        batch = {key: value.to(device) for key, value in batch.items()}
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            output = model(
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"],
                output_hidden_states=True,
                use_cache=False,
            )
        if result is None:
            layers = len(output.hidden_states) - 1
            hidden_dim = output.hidden_states[0].shape[-1]
            result = np.full(
                (total, positions.shape[1], layers, hidden_dim),
                np.nan,
                dtype=np.float32,
            )
        stacked = torch.stack(
            [hidden[:, :-1].float() for hidden in output.hidden_states[1:]], dim=1
        )
        for row in range(len(batch["input_ids"])):
            picked = np.clip(positions[start + row], 0, stacked.shape[2] - 1)
            result[start + row] = (
                stacked[row][:, picked, :].permute(1, 0, 2).cpu().numpy()
            )
        start += len(batch["input_ids"])
        del output, stacked
    assert result is not None
    return result


# ---------------------------------------------------------------------------
# shared evaluation helpers


def _score_families(
    families: dict[str, np.ndarray],
    labels: np.ndarray,
    calibration: np.ndarray,
    test: np.ndarray,
    detector_seeds: int,
    bootstrap_repeats: int,
    seed: int,
    baselines: list[tuple[str, str]],
) -> dict[str, Any]:
    seeds = tuple(seed + 1000 + offset for offset in range(detector_seeds))
    test_labels = labels[test]
    scores: dict[str, np.ndarray] = {}
    metrics: dict[str, dict[str, float]] = {}
    stability: dict[str, dict[str, float]] = {}
    for offset, (name, features) in enumerate(families.items()):
        seed_scores = triplet_scores_by_seed(features, labels, calibration, test, seeds)
        scores[name] = np.mean(seed_scores, axis=0)
        metrics[name] = metric_row(
            test_labels, scores[name], bootstrap_repeats, seed + 2000 + offset
        )
        seed_aucs = [float(auc) for auc in _seed_aucs(test_labels, seed_scores)]
        stability[name] = {
            "auc_by_seed": seed_aucs,
            "auc_mean": float(np.mean(seed_aucs)),
            "auc_std": float(np.std(seed_aucs, ddof=1)) if len(seed_aucs) > 1 else 0.0,
        }
    deltas = {
        f"{left}_minus_{right}": paired_bootstrap_delta(
            test_labels, scores[left], scores[right], bootstrap_repeats, seed + 3000 + offset
        )
        for offset, (left, right) in enumerate(baselines)
    }
    return {
        "metrics": metrics,
        "scores": {
            name: [float(value) for value in score] for name, score in scores.items()
        },
        "detector_stability": stability,
        "paired_auc_deltas": deltas,
    }


def _seed_aucs(test_labels: np.ndarray, seed_scores: np.ndarray) -> list[float]:
    from .audit import auc_rank

    return [auc_rank(test_labels, values) for values in seed_scores]


def _write_artifact(
    output_dir: Path, artifact: dict[str, Any], markdown: str
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "results.json").write_text(
        json.dumps(artifact, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    (output_dir / "RESULTS.md").write_text(markdown, encoding="utf-8")


def _metric_lines(metrics: dict[str, dict[str, float]]) -> list[str]:
    lines = [
        "| Signal | AUC (95% CI) | TPR@1%FPR | TPR@5%FPR |",
        "|---|---:|---:|---:|",
    ]
    for name, row in metrics.items():
        lines.append(
            f"| `{name}` | {row['auc']:.4f} "
            f"[{row['auc_ci95_low']:.4f}, {row['auc_ci95_high']:.4f}] "
            f"| {row.get('tpr_at_1pct_fpr', float('nan')):.4f} "
            f"| {row.get('tpr_at_5pct_fpr', float('nan')):.4f} |"
        )
    return lines


def _delta_lines(deltas: dict[str, dict[str, float]]) -> list[str]:
    lines = [
        "| Comparison | Delta AUC (95% CI) |",
        "|---|---:|",
    ]
    for name, row in deltas.items():
        lines.append(
            f"| `{name}` | {row['delta_auc']:+.4f} "
            f"[{row['ci95_low']:+.4f}, {row['ci95_high']:+.4f}] |"
        )
    return lines


# ---------------------------------------------------------------------------
# probe B: q-conditional transcript features


def bottom_k_fraction_means(values: np.ndarray, fractions: tuple[float, ...]) -> np.ndarray:
    """Mean of the smallest k% values per row for each fraction."""
    ordered = np.sort(values, axis=1)
    columns = []
    width = values.shape[1]
    for fraction in fractions:
        k = max(1, min(width, int(math.ceil(width * fraction))))
        columns.append(ordered[:, :k].mean(axis=1))
    return np.column_stack(columns)


def top_k_fraction_means(values: np.ndarray, fractions: tuple[float, ...]) -> np.ndarray:
    ordered = np.sort(values, axis=1)
    columns = []
    width = values.shape[1]
    for fraction in fractions:
        k = max(1, min(width, int(math.ceil(width * fraction))))
        columns.append(ordered[:, -k:].mean(axis=1))
    return np.column_stack(columns)


def q_conditional_transcript(
    acceptance: np.ndarray,
    q_logp: np.ndarray,
    calibration: np.ndarray,
    test: np.ndarray,
    bins: int = QCOND_BINS,
    k_fractions: tuple[float, ...] = QCOND_K_FRACTIONS,
) -> np.ndarray:
    """Transcript summary plus q-bin conditional acceptance features.

    The q-bin edges come from the pooled calibration q log-probabilities only.
    Each additional feature is the mean observed acceptance of the record's
    selected positions falling inside one q bin, plus bottom-k% / top-k%
    aggregations of the acceptance values themselves.
    """
    edges = np.quantile(
        q_logp[np.sort(calibration)].reshape(-1), np.linspace(0.0, 1.0, bins + 1)
    )
    edges = np.unique(edges)
    if len(edges) < 2:
        raise ValueError("degenerate q-bin edges from calibration")
    records, width = acceptance.shape
    n_bins = len(edges) - 1
    bin_index = np.digitize(q_logp.reshape(-1), edges[1:-1])
    offsets = bin_index.reshape(records, width) + np.arange(records)[:, None] * n_bins
    counts = np.bincount(
        offsets.reshape(-1), minlength=records * n_bins
    ).reshape(records, n_bins)
    sums = np.bincount(
        offsets.reshape(-1),
        weights=acceptance.reshape(-1).astype(np.float64),
        minlength=records * n_bins,
    ).reshape(records, n_bins)
    conditional = np.where(counts > 0, sums / np.maximum(counts, 1), 0.0)
    return np.column_stack(
        [
            _transcript_summary(acceptance),
            conditional.astype(np.float32),
            bottom_k_fraction_means(acceptance, k_fractions),
            top_k_fraction_means(acceptance, k_fractions),
        ]
    ).astype(np.float32)


def run_qcond(args: argparse.Namespace) -> None:
    if args.cpu_threads < 1:
        raise ValueError("cpu-threads must be positive")
    torch.set_num_threads(args.cpu_threads)
    torch.set_num_interop_threads(min(4, args.cpu_threads))
    root = Path(__file__).resolve().parents[2]
    source_path = _resolve(root, args.source_results)
    source = _load_source(source_path)
    config = source["config"]
    output_dir = _resolve(root, args.output_dir)
    device = torch.device(f"cuda:{args.gpu}")
    torch.cuda.set_device(device)
    seed = int(config["seed"])
    set_seed(seed)
    started = time.time()

    tokenizer, members, nonmembers, labels, calibration, test, data_metadata = (
        _reconstruct(source, source_path)
    )
    candidates = members + nonmembers
    endpoint, draft = _load_endpoint_draft(config, source_path, device)
    draft_outputs = extract_draft_activation_outputs(
        draft, candidates, tokenizer, device, int(config.get("draft_batch_size", 4))
    )
    del draft
    gc.collect()
    torch.cuda.empty_cache()
    target = _load_adapted_target(config, source_path, device)
    target_outputs = extract_target_token_outputs(
        target, candidates, tokenizer, device, int(config.get("target_batch_size", 4))
    )
    del target
    gc.collect()
    torch.cuda.empty_cache()

    selected = bottom_k_indices(draft_outputs["token_logp"], float(config["min_k_fraction"]))
    q_selected = gather_tokens(draft_outputs["token_logp"], selected)
    budgets = sorted(set(args.repeats))
    observed, exact_alpha = _nested_acceptance(
        target_outputs["token_logp"],
        draft_outputs["token_logp"],
        selected,
        budgets,
        seed + 50000,
    )

    budgets_result: dict[str, Any] = {}
    for repeats in budgets:
        acceptance = observed[repeats]
        families = {
            "transcript_only_triplet": _transcript_summary(acceptance),
            "transcript_qcond_triplet": q_conditional_transcript(
                acceptance, q_selected, calibration, test
            ),
        }
        shuffled = q_stratified_shuffle(
            acceptance, q_selected, calibration, test, seed + 2, bins=QCOND_BINS
        )
        families["control_qbin_shuffled_transcript_qcond_triplet"] = (
            q_conditional_transcript(shuffled, q_selected, calibration, test)
        )
        evaluation = _score_families(
            families,
            labels,
            calibration,
            test,
            args.detector_seeds,
            args.bootstrap_repeats,
            seed + repeats,
            baselines=[
                ("transcript_qcond_triplet", "transcript_only_triplet"),
                (
                    "transcript_qcond_triplet",
                    "control_qbin_shuffled_transcript_qcond_triplet",
                ),
            ],
        )
        test_labels = labels[test]
        evaluation["direct_scores"] = {
            "verifier_mean_acceptance": metric_row(
                test_labels,
                acceptance.mean(axis=1)[test],
                args.bootstrap_repeats,
                seed + 7000 + repeats,
            ),
            "oracle_exact_mean_acceptance": metric_row(
                test_labels,
                exact_alpha.mean(axis=1)[test],
                args.bootstrap_repeats,
                seed + 8000 + repeats,
            ),
        }
        budgets_result[str(repeats)] = evaluation

    artifact = {
        "material_passport": {
            "status": "COMPLETED",
            "verification_status": "ANALYZED_QCOND_TRANSCRIPT_PROBE",
        },
        "source_experiment": str(source_path.relative_to(root)),
        "config": {
            "qcond_bins": QCOND_BINS,
            "qcond_k_fractions": list(QCOND_K_FRACTIONS),
            "repeats": budgets,
            "detector_seeds": args.detector_seeds,
            "bootstrap_repeats": args.bootstrap_repeats,
            "draft_endpoint": endpoint,
        },
        "data": data_metadata,
        "runtime": {
            "device": torch.cuda.get_device_name(device),
            "duration_seconds": time.time() - started,
            "peak_gpu_memory_gib": torch.cuda.max_memory_allocated(device) / (1024**3),
        },
        "budgets": budgets_result,
    }
    _write_artifact(output_dir, artifact, _render_qcond(artifact))


def _render_qcond(artifact: dict[str, Any]) -> str:
    lines = [
        "# q-conditional transcript probe (B)",
        "",
        f"- Source: `{artifact['source_experiment']}`",
        f"- Draft endpoint: `{artifact['config']['draft_endpoint']}`",
        "- Features: plain transcript summary vs the same summary plus q-decile",
        "  conditional acceptance, bottom-k% and top-k% aggregations.",
        "",
    ]
    for repeats, result in artifact["budgets"].items():
        lines.append(f"## Repeats per token: {repeats}")
        lines.append("")
        lines.extend(_metric_lines(result["metrics"]))
        lines.append("")
        lines.extend(_delta_lines(result["paired_auc_deltas"]))
        lines.append("")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# probe A: paraphrase robustness


def _resolve_paraphrase_snapshot(model_id: str) -> tuple[str, str]:
    path = Path(snapshot_download(repo_id=model_id, local_files_only=True))
    return str(path), path.name


@torch.no_grad()
def generate_paraphrases(
    model: torch.nn.Module,
    tokenizer: Any,
    records: list[SFTRecord],
    device: torch.device,
    batch_size: int,
    max_new_tokens: int,
    temperature: float,
    top_p: float,
    seed: int,
) -> list[str]:
    """Rewrite each record's response with the local instruct model.

    Returns raw rewritten texts; callers must re-tokenize them with the audit
    pair's own tokenizer before scoring.
    """
    tokenizer.padding_side = "left"
    texts = [
        tokenizer.decode(record.response_ids, skip_special_tokens=True)
        for record in records
    ]
    generated: list[str] = []
    for start in range(0, len(texts), batch_size):
        chunk = texts[start : start + batch_size]
        prompts = []
        for text in chunk:
            message = [{"role": "user", "content": PARAPHRASE_INSTRUCTION + text}]
            try:
                prompts.append(
                    tokenizer.apply_chat_template(
                        message,
                        tokenize=False,
                        add_generation_prompt=True,
                        enable_thinking=False,
                    )
                )
            except (TypeError, ValueError):
                prompts.append(
                    tokenizer.apply_chat_template(
                        message, tokenize=False, add_generation_prompt=True
                    )
                )
        encoded = tokenizer(
            prompts, return_tensors="pt", padding=True, add_special_tokens=False
        ).to(device)
        torch.manual_seed(seed + start)
        output = model.generate(
            **encoded,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=temperature,
            top_p=top_p,
            pad_token_id=tokenizer.eos_token_id,
        )
        continuation = output[:, encoded.input_ids.shape[1]:]
        generated.extend(
            tokenizer.batch_decode(continuation, skip_special_tokens=True)
        )
        del output, continuation, encoded

    tokenized = [
        tokenizer(text.strip(), add_special_tokens=False)["input_ids"]
        for text in generated
    ]
    short = [index for index, ids in enumerate(tokenized) if len(ids) < DEGENERATE_TOKENS]
    if short:
        retries = {}
        for start in range(0, len(short), batch_size):
            chunk = short[start : start + batch_size]
            prompts = []
            for index in chunk:
                message = [
                    {
                        "role": "user",
                        "content": PARAPHRASE_INSTRUCTION + texts[index],
                    }
                ]
                try:
                    prompts.append(
                        tokenizer.apply_chat_template(
                            message,
                            tokenize=False,
                            add_generation_prompt=True,
                            enable_thinking=False,
                        )
                    )
                except (TypeError, ValueError):
                    prompts.append(
                        tokenizer.apply_chat_template(
                            message, tokenize=False, add_generation_prompt=True
                        )
                    )
            encoded = tokenizer(
                prompts, return_tensors="pt", padding=True, add_special_tokens=False
            ).to(device)
            torch.manual_seed(seed + 100000 + start)
            output = model.generate(
                **encoded,
                max_new_tokens=max_new_tokens,
                do_sample=True,
                temperature=1.0,
                top_p=top_p,
                pad_token_id=tokenizer.eos_token_id,
            )
            continuation = output[:, encoded.input_ids.shape[1]:]
            decoded = tokenizer.batch_decode(continuation, skip_special_tokens=True)
            for position, index in enumerate(chunk):
                retries[index] = tokenizer(
                    decoded[position].strip(), add_special_tokens=False
                )["input_ids"]
        for index, ids in retries.items():
            if len(ids) >= len(tokenized[index]):
                tokenized[index] = ids
    return [generated[index].strip() for index in range(len(generated))]


def build_paraphrased_records(
    records: list[SFTRecord],
    paraphrase_texts: list[str],
    tokenizer: Any,
    max_response_tokens: int,
) -> tuple[list[SFTRecord], dict[str, Any]]:
    """Re-tokenize rewritten texts with the audit pair's tokenizer.

    The paraphrase model has its own vocabulary; record ids must always be
    rebuilt with the draft/target tokenizer that will score them.
    """
    if len(paraphrase_texts) != len(records):
        raise ValueError("paraphrase texts must align with records")
    tokenized = [
        tokenizer(text, add_special_tokens=False)["input_ids"]
        for text in paraphrase_texts
    ]
    rebuilt: list[SFTRecord] = []
    for record, ids in zip(records, tokenized):
        if len(ids) < 1:
            raise RuntimeError(f"empty paraphrase for record {record.record_id}")
        truncated = [int(value) for value in ids[:max_response_tokens]]
        rebuilt.append(
            SFTRecord(
                record_id=record.record_id,
                source=record.source,
                response_ids=tuple(truncated),
                response_hash=_hash_ids(truncated),
                prompt_ids=record.prompt_ids,
                prompt_hash=record.prompt_hash,
                prompt_text=record.prompt_text,
                topic=record.topic,
                source_char_count=record.source_char_count,
                source_timestamp=record.source_timestamp,
                source_revision=record.source_revision,
            )
        )
    lengths = {
        name: {
            "mean": float(np.mean([len(record.response_ids) for record in subset])),
            "min": int(min(len(record.response_ids) for record in subset)),
            "max": int(max(len(record.response_ids) for record in subset)),
        }
        for name, subset in (("members", rebuilt[: len(rebuilt) // 2]), ("nonmembers", rebuilt[len(rebuilt) // 2 :]))
    }
    original_lengths = {
        name: float(np.mean([len(record.response_ids) for record in subset]))
        for name, subset in (("members", records[: len(records) // 2]), ("nonmembers", records[len(records) // 2 :]))
    }
    report = {
        "paraphrased_token_counts": lengths,
        "original_mean_token_counts": original_lengths,
        "records_truncated": int(
            sum(1 for ids in tokenized if len(ids) > max_response_tokens)
        ),
    }
    return rebuilt, report


def _audit_outputs(
    draft: torch.nn.Module,
    target: torch.nn.Module,
    candidates: list[SFTRecord],
    tokenizer: Any,
    device: torch.device,
    config: dict[str, Any],
    repeats: int,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    draft_outputs = extract_draft_activation_outputs(
        draft, candidates, tokenizer, device, int(config.get("draft_batch_size", 4))
    )
    target_outputs = extract_target_token_outputs(
        target, candidates, tokenizer, device, int(config.get("target_batch_size", 4))
    )
    selected = bottom_k_indices(draft_outputs["token_logp"], float(config["min_k_fraction"]))
    observed, exact_alpha = _nested_acceptance(
        target_outputs["token_logp"],
        draft_outputs["token_logp"],
        selected,
        [repeats],
        seed,
    )
    return draft_outputs, target_outputs, observed[repeats]


def restrict_to_long_responses(
    candidates: list[SFTRecord],
    paraphrased: list[SFTRecord],
    labels: np.ndarray,
    calibration: np.ndarray,
    test: np.ndarray,
    min_tokens: int,
) -> tuple[list[SFTRecord], list[SFTRecord], np.ndarray, np.ndarray, np.ndarray, dict[str, int]]:
    """Drop still-degenerate paraphrases from both sides, preserving pairing.

    Records whose paraphrased response cannot supply the audit protocol's
    selected positions are removed from the original and paraphrased sets
    together, and the audit-split index arrays are remapped.
    """
    keep = np.array(
        [len(record.response_ids) >= min_tokens for record in paraphrased]
    )
    report = {
        "records_dropped_short": int((~keep).sum()),
        "dropped_members": int(((~keep) & (labels == 1)).sum()),
        "dropped_nonmembers": int(((~keep) & (labels == 0)).sum()),
    }
    if keep.all():
        return candidates, paraphrased, labels, calibration, test, report
    new_index = np.cumsum(keep) - 1
    candidates = [record for record, kept in zip(candidates, keep) if kept]
    paraphrased = [record for record, kept in zip(paraphrased, keep) if kept]
    labels = labels[keep]
    calibration = new_index[calibration[keep[calibration]]]
    test = new_index[test[keep[test]]]
    return candidates, paraphrased, labels, calibration, test, report


def run_paraphrase(args: argparse.Namespace) -> None:
    if args.cpu_threads < 1:
        raise ValueError("cpu-threads must be positive")
    torch.set_num_threads(args.cpu_threads)
    torch.set_num_interop_threads(min(4, args.cpu_threads))
    root = Path(__file__).resolve().parents[2]
    source_path = _resolve(root, args.source_results)
    source = _load_source(source_path)
    config = source["config"]
    output_dir = _resolve(root, args.output_dir)
    device = torch.device(f"cuda:{args.gpu}")
    torch.cuda.set_device(device)
    seed = int(config["seed"])
    set_seed(seed)
    started = time.time()

    tokenizer, members, nonmembers, labels, calibration, test, data_metadata = (
        _reconstruct(source, source_path)
    )
    candidates = members + nonmembers
    output_dir = _resolve(root, args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    cache_path = output_dir / "paraphrase_texts.json"
    snapshot_path, snapshot_commit = _resolve_paraphrase_snapshot(args.paraphrase_model)
    if cache_path.exists():
        paraphrase_texts = json.loads(cache_path.read_text(encoding="utf-8"))
        paraphrased, length_report = build_paraphrased_records(
            candidates, paraphrase_texts, tokenizer, int(config["response_tokens"])
        )
    else:
        paraphrase_model = load_causal_lm(
            args.paraphrase_model, device, revision=None, local_files_only=True
        )
        paraphrase_tokenizer = AutoTokenizer.from_pretrained(
            snapshot_path, local_files_only=True
        )
        if paraphrase_tokenizer.pad_token_id is None:
            paraphrase_tokenizer.pad_token = paraphrase_tokenizer.eos_token
        paraphrase_texts = generate_paraphrases(
            paraphrase_model,
            paraphrase_tokenizer,
            candidates,
            device,
            args.paraphrase_batch_size,
            args.max_new_tokens,
            args.temperature,
            args.top_p,
            seed + 90000,
        )
        paraphrased, length_report = build_paraphrased_records(
            candidates, paraphrase_texts, tokenizer, int(config["response_tokens"])
        )
        min_tokens = (
            math.ceil(float(config["min_k_fraction"]) * int(config["response_tokens"]))
            + 4
        )
        for attempt, temperature in enumerate((1.0, 1.3)):
            short = [
                index
                for index, record in enumerate(paraphrased)
                if len(record.response_ids) < min_tokens
            ]
            if not short:
                break
            regenerated = generate_paraphrases(
                paraphrase_model,
                paraphrase_tokenizer,
                [candidates[index] for index in short],
                device,
                args.paraphrase_batch_size,
                args.max_new_tokens,
                temperature,
                args.top_p,
                seed + 91000 + attempt,
            )
            for position, index in enumerate(short):
                paraphrase_texts[index] = regenerated[position]
            paraphrased, length_report = build_paraphrased_records(
                candidates, paraphrase_texts, tokenizer, int(config["response_tokens"])
            )
            length_report["short_retries"] = attempt + 1
        del paraphrase_model
        gc.collect()
        torch.cuda.empty_cache()
        cache_path.write_text(
            json.dumps(paraphrase_texts, ensure_ascii=False), encoding="utf-8"
        )
    candidates, paraphrased, labels, calibration, test, drop_report = (
        restrict_to_long_responses(
            candidates,
            paraphrased,
            labels,
            calibration,
            test,
            math.ceil(float(config["min_k_fraction"]) * int(config["response_tokens"]))
            + 4,
        )
    )
    length_report.update(drop_report)
    if len(test) < 8 or len(calibration) < 4:
        raise RuntimeError("too many degenerate paraphrases to run the audit")

    endpoint, draft = _load_endpoint_draft(config, source_path, device)
    target = _load_adapted_target(config, source_path, device)
    draft_original, target_original, acceptance_original = _audit_outputs(
        draft, target, candidates, tokenizer, device, config, args.repeats, seed + 50000
    )
    draft_para, target_para, acceptance_para = _audit_outputs(
        draft, target, paraphrased, tokenizer, device, config, args.repeats, seed + 50000
    )
    del draft, target
    gc.collect()
    torch.cuda.empty_cache()

    families_original = build_activation_feature_families(
        draft_original,
        target_original,
        bottom_k_indices(draft_original["token_logp"], float(config["min_k_fraction"])),
        acceptance_original,
        calibration,
        test,
        seed + 10,
    )
    selected_para = bottom_k_indices(
        draft_para["token_logp"], float(config["min_k_fraction"])
    )
    families_paraphrased = build_activation_feature_families(
        draft_para,
        target_para,
        selected_para,
        acceptance_para,
        calibration,
        test,
        seed + 10,
    )

    seeds = tuple(seed + 1000 + offset for offset in range(args.detector_seeds))
    test_labels = labels[test]
    metrics: dict[str, dict[str, Any]] = {}
    deltas: dict[str, dict[str, float]] = {}
    scores_original: dict[str, np.ndarray] = {}
    scores_paraphrased: dict[str, np.ndarray] = {}
    for offset, name in enumerate(families_original):
        original_scores = np.mean(
            triplet_scores_by_seed(
                families_original[name], labels, calibration, test, seeds
            ),
            axis=0,
        )
        paraphrased_scores = np.mean(
            triplet_scores_by_seed(
                families_paraphrased[name], labels, calibration, test, seeds
            ),
            axis=0,
        )
        scores_original[name] = original_scores
        scores_paraphrased[name] = paraphrased_scores
        metrics[name] = {
            "original": metric_row(
                test_labels, original_scores, args.bootstrap_repeats, seed + 2000 + offset
            ),
            "paraphrased": metric_row(
                test_labels,
                paraphrased_scores,
                args.bootstrap_repeats,
                seed + 3000 + offset,
            ),
        }
        deltas[f"{name}_paraphrased_minus_original"] = paired_bootstrap_delta(
            test_labels,
            paraphrased_scores,
            original_scores,
            args.bootstrap_repeats,
            seed + 4000 + offset,
        )

    direct = {
        "verifier_mean_acceptance": {
            "original": float(acceptance_original.mean()),
            "paraphrased": float(acceptance_para.mean()),
        }
    }
    artifact = {
        "material_passport": {
            "status": "COMPLETED",
            "verification_status": "ANALYZED_PARAPHRASE_PROBE",
        },
        "source_experiment": str(source_path.relative_to(root)),
        "config": {
            "paraphrase_model": args.paraphrase_model,
            "paraphrase_snapshot_commit": snapshot_commit,
            "temperature": args.temperature,
            "top_p": args.top_p,
            "max_new_tokens": args.max_new_tokens,
            "repeats": args.repeats,
            "draft_endpoint": endpoint,
        },
        "data": {**data_metadata, "paraphrase_length_report": length_report},
        "runtime": {
            "device": torch.cuda.get_device_name(device),
            "duration_seconds": time.time() - started,
            "peak_gpu_memory_gib": torch.cuda.max_memory_allocated(device) / (1024**3),
        },
        "metrics": metrics,
        "paired_paraphrased_minus_original": deltas,
        "direct_mean_acceptance": direct,
    }
    _write_artifact(output_dir, artifact, _render_paraphrase(artifact))


def _render_paraphrase(artifact: dict[str, Any]) -> str:
    lines = [
        "# Paraphrase robustness probe (A)",
        "",
        f"- Source: `{artifact['source_experiment']}`",
        f"- Paraphrase model: `{artifact['config']['paraphrase_model']}` "
        f"@ {artifact['config']['paraphrase_snapshot_commit']}",
        f"- Temperature {artifact['config']['temperature']}, top-p {artifact['config']['top_p']}",
        "",
    ]
    report = artifact["data"]["paraphrase_length_report"]
    lines.append(
        f"- Paraphrased mean tokens: member {report['paraphrased_token_counts']['members']['mean']:.1f}, "
        f"nonmember {report['paraphrased_token_counts']['nonmembers']['mean']:.1f} "
        f"(original: member {report['original_mean_token_counts']['members']:.1f}, "
        f"nonmember {report['original_mean_token_counts']['nonmembers']:.1f})"
    )
    lines.append("")
    lines.append("| Family | Original AUC | Paraphrased AUC | Delta (95% CI) |")
    lines.append("|---|---:|---:|---:|")
    for name, rows in artifact["metrics"].items():
        delta = artifact["paired_paraphrased_minus_original"][
            f"{name}_paraphrased_minus_original"
        ]
        lines.append(
            f"| `{name}` | {rows['original']['auc']:.4f} | {rows['paraphrased']['auc']:.4f} "
            f"| {delta['delta_auc']:+.4f} [{delta['ci95_low']:+.4f}, {delta['ci95_high']:+.4f}] |"
        )
    lines.append("")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# probe D: activation feature-extraction variants


def calibration_layer_statistics(
    raw_states: np.ndarray, calibration: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Per-layer mean/std over calibration records and pooled positions."""
    subset = raw_states[np.sort(calibration)]
    mean = subset.mean(axis=(0, 1), keepdims=True)
    std = subset.std(axis=(0, 1), keepdims=True) + 1e-6
    return mean.astype(np.float32), std.astype(np.float32)


def histogram_features(
    standardized: np.ndarray, bins: int = HIST_BINS, bounds: tuple[float, float] = HIST_RANGE
) -> np.ndarray:
    """Per-cell histogram counts of standardized activations.

    Input is ``(records, positions, layers, hidden)``; output is
    ``(records, positions, layers, bins)`` with counts normalized to sum to
    one per cell. Values outside ``bounds`` are clipped into the edge bins.
    """
    low, high = bounds
    scaled = (standardized - low) / (high - low)
    index = np.clip(np.floor(scaled * bins), 0, bins - 1).astype(np.int64)
    records, positions, layers, hidden = index.shape
    chunk = max(1, int(2**26 // (positions * layers * hidden)))
    output = np.empty((records, positions, layers, bins), dtype=np.float32)
    for start in range(0, records, chunk):
        stop = min(records, start + chunk)
        flat_index = index[start:stop].reshape(-1)
        cells = flat_index.size // hidden
        offsets = np.repeat(np.arange(cells, dtype=np.int64) * bins, hidden)
        counts = np.bincount(flat_index + offsets, minlength=cells * bins)
        output[start:stop] = (
            counts.reshape(-1, bins).astype(np.float32) / float(hidden)
        ).reshape(stop - start, positions, layers, bins)
    return output


def projection_features(
    standardized: np.ndarray, dim: int = PROJECTION_DIM, seed: int = 0
) -> np.ndarray:
    """Fixed random projection of the hidden dimension, per position and layer."""
    generator = np.random.default_rng(seed)
    matrix = generator.standard_normal(
        (standardized.shape[-1], dim), dtype=np.float32
    ) / math.sqrt(standardized.shape[-1])
    records, positions, layers, hidden = standardized.shape
    flat = standardized.reshape(-1, hidden)
    projected = flat @ matrix
    return projected.reshape(records, positions, layers, dim)


def run_fe(args: argparse.Namespace) -> None:
    if args.cpu_threads < 1:
        raise ValueError("cpu-threads must be positive")
    torch.set_num_threads(args.cpu_threads)
    torch.set_num_interop_threads(min(4, args.cpu_threads))
    root = Path(__file__).resolve().parents[2]
    source_path = _resolve(root, args.source_results)
    source = _load_source(source_path)
    config = source["config"]
    output_dir = _resolve(root, args.output_dir)
    device = torch.device(f"cuda:{args.gpu}")
    torch.cuda.set_device(device)
    seed = int(config["seed"])
    set_seed(seed)
    started = time.time()

    tokenizer, members, nonmembers, labels, calibration, test, data_metadata = (
        _reconstruct(source, source_path)
    )
    candidates = members + nonmembers
    endpoint, draft = _load_endpoint_draft(config, source_path, device)
    draft_outputs = extract_draft_activation_outputs(
        draft, candidates, tokenizer, device, int(config.get("draft_batch_size", 4))
    )
    selected = bottom_k_indices(draft_outputs["token_logp"], float(config["min_k_fraction"]))
    valid_count = np.isfinite(draft_outputs["token_logp"]).sum(axis=1)
    terminal = (valid_count - 1)[:, None]
    positions = np.concatenate([selected, terminal], axis=1)
    raw_states = extract_draft_position_states(
        draft, candidates, tokenizer, device, int(config.get("draft_batch_size", 4)), positions
    )
    del draft
    gc.collect()
    torch.cuda.empty_cache()
    target = _load_adapted_target(config, source_path, device)
    target_outputs = extract_target_token_outputs(
        target, candidates, tokenizer, device, int(config.get("target_batch_size", 4))
    )
    del target
    gc.collect()
    torch.cuda.empty_cache()

    observed, exact_alpha = _nested_acceptance(
        target_outputs["token_logp"],
        draft_outputs["token_logp"],
        selected,
        [args.repeats],
        seed + 50000,
    )
    acceptance = observed[args.repeats]
    q_selected = gather_tokens(draft_outputs["token_logp"], selected)
    entropy_selected = gather_tokens(draft_outputs["entropy"], selected)
    transcript = _transcript_summary(acceptance)
    residual = cross_fitted_verifier_residual(
        acceptance, q_selected, entropy_selected, calibration, test, seed + 1
    )
    shuffled_acceptance = q_stratified_shuffle(
        acceptance, q_selected, calibration, test, seed + 2
    )
    shuffled_transcript = _transcript_summary(shuffled_acceptance)
    shuffled_residual = cross_fitted_verifier_residual(
        shuffled_acceptance, q_selected, entropy_selected, calibration, test, seed + 3
    )

    mean, std = calibration_layer_statistics(raw_states, calibration)
    standardized = np.clip((raw_states - mean) / std, -8.0, 8.0)
    hist = histogram_features(standardized)
    projected = projection_features(standardized, seed=seed + 60000)
    n_selected = selected.shape[1]

    stat_selected = gather_tokens(draft_outputs["activation_stats"], selected)
    stat_all = np.concatenate(
        [stat_selected, draft_outputs["activation_stats"][np.arange(len(candidates)), terminal[:, 0]][:, None]], axis=1
    )

    def _pool(block: np.ndarray, positions_slice: slice) -> np.ndarray:
        selected_part = block[:, positions_slice]
        return selected_part.mean(axis=1).reshape(len(block), -1)

    families: dict[str, np.ndarray] = {
        "transcript_only_triplet": transcript,
        "standalone_statfe_selected": _pool(stat_all, slice(0, n_selected)),
        "standalone_statfe_terminal": _pool(stat_all, slice(n_selected, None)),
        "standalone_hist_selected": _pool(hist, slice(0, n_selected)),
        "standalone_hist_terminal": _pool(hist, slice(n_selected, None)),
        "standalone_rawproj_selected": _pool(projected, slice(0, n_selected)),
        "standalone_rawproj_terminal": _pool(projected, slice(n_selected, None)),
        "draver_act_statfe_residual_triplet": _conditioned_activation_features(
            stat_all[:, :n_selected], acceptance, transcript, residual=residual
        ),
        "draver_act_hist_residual": _conditioned_activation_features(
            hist[:, :n_selected], acceptance, transcript, residual=residual
        ),
        "draver_act_rawproj_residual": _conditioned_activation_features(
            projected[:, :n_selected], acceptance, transcript, residual=residual
        ),
        "control_qbin_shuffled_draver_act_statfe": _conditioned_activation_features(
            stat_all[:, :n_selected],
            shuffled_acceptance,
            shuffled_transcript,
            residual=shuffled_residual,
        ),
    }
    evaluation = _score_families(
        families,
        labels,
        calibration,
        test,
        args.detector_seeds,
        args.bootstrap_repeats,
        seed,
        baselines=[
            ("draver_act_statfe_residual_triplet", "transcript_only_triplet"),
            ("draver_act_hist_residual", "transcript_only_triplet"),
            ("draver_act_rawproj_residual", "transcript_only_triplet"),
            ("draver_act_hist_residual", "draver_act_statfe_residual_triplet"),
            ("draver_act_rawproj_residual", "draver_act_statfe_residual_triplet"),
            ("standalone_hist_selected", "standalone_statfe_selected"),
            ("standalone_rawproj_selected", "standalone_statfe_selected"),
            ("standalone_hist_terminal", "standalone_statfe_terminal"),
            ("standalone_rawproj_terminal", "standalone_statfe_terminal"),
        ],
    )
    artifact = {
        "material_passport": {
            "status": "COMPLETED",
            "verification_status": "ANALYZED_ACTIVATION_FE_PROBE",
        },
        "source_experiment": str(source_path.relative_to(root)),
        "config": {
            "repeats": args.repeats,
            "hist_bins": HIST_BINS,
            "hist_range": list(HIST_RANGE),
            "projection_dim": PROJECTION_DIM,
            "selected_tokens_per_record": int(n_selected),
            "layers": int(raw_states.shape[2]),
            "hidden_dim": int(raw_states.shape[3]),
            "detector_seeds": args.detector_seeds,
            "draft_endpoint": endpoint,
        },
        "data": data_metadata,
        "runtime": {
            "device": torch.cuda.get_device_name(device),
            "duration_seconds": time.time() - started,
            "peak_gpu_memory_gib": torch.cuda.max_memory_allocated(device) / (1024**3),
        },
        "evaluation": evaluation,
        "protocol": {
            "bits_per_record": int(n_selected * args.repeats),
            "mean_observed_acceptance": float(acceptance.mean()),
            "mean_exact_acceptance_probability": float(exact_alpha.mean()),
        },
    }
    _write_artifact(output_dir, artifact, _render_fe(artifact))


def _render_fe(artifact: dict[str, Any]) -> str:
    evaluation = artifact["evaluation"]
    lines = [
        "# Activation feature-extraction probe (D)",
        "",
        f"- Source: `{artifact['source_experiment']}`",
        f"- Draft endpoint: `{artifact['config']['draft_endpoint']}`",
        f"- Selected positions {artifact['config']['selected_tokens_per_record']}, "
        f"layers {artifact['config']['layers']}, hist bins {artifact['config']['hist_bins']}, "
        f"projection dim {artifact['config']['projection_dim']}",
        "",
    ]
    lines.extend(_metric_lines(evaluation["metrics"]))
    lines.append("")
    lines.extend(_delta_lines(evaluation["paired_auc_deltas"]))
    lines.append("")
    return "\n".join(lines)


# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Minimal method-optimization probes on saved public adapters."
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    qcond = subparsers.add_parser("qcond", help="q-conditional transcript features (B)")
    qcond.add_argument("--gpu", type=int, required=True)
    qcond.add_argument("--source-results", type=Path, required=True)
    qcond.add_argument("--output-dir", type=Path, required=True)
    qcond.add_argument("--repeats", type=int, nargs="+", default=[4, 24])
    qcond.add_argument("--bootstrap-repeats", type=int, default=500)
    qcond.add_argument("--detector-seeds", type=int, default=3)
    qcond.add_argument("--cpu-threads", type=int, default=8)
    qcond.set_defaults(func=run_qcond)

    paraphrase = subparsers.add_parser("paraphrase", help="paraphrase robustness (A)")
    paraphrase.add_argument("--gpu", type=int, required=True)
    paraphrase.add_argument("--source-results", type=Path, required=True)
    paraphrase.add_argument("--output-dir", type=Path, required=True)
    paraphrase.add_argument("--paraphrase-model", default="Qwen/Qwen3-8B")
    paraphrase.add_argument("--repeats", type=int, default=24)
    paraphrase.add_argument("--paraphrase-batch-size", type=int, default=8)
    paraphrase.add_argument("--max-new-tokens", type=int, default=192)
    paraphrase.add_argument("--temperature", type=float, default=0.7)
    paraphrase.add_argument("--top-p", type=float, default=0.9)
    paraphrase.add_argument("--bootstrap-repeats", type=int, default=500)
    paraphrase.add_argument("--detector-seeds", type=int, default=3)
    paraphrase.add_argument("--cpu-threads", type=int, default=8)
    paraphrase.set_defaults(func=run_paraphrase)

    fe = subparsers.add_parser("fe", help="activation feature-extraction variants (D)")
    fe.add_argument("--gpu", type=int, required=True)
    fe.add_argument("--source-results", type=Path, required=True)
    fe.add_argument("--output-dir", type=Path, required=True)
    fe.add_argument("--repeats", type=int, default=24)
    fe.add_argument("--bootstrap-repeats", type=int, default=500)
    fe.add_argument("--detector-seeds", type=int, default=3)
    fe.add_argument("--cpu-threads", type=int, default=8)
    fe.set_defaults(func=run_fe)

    args = parser.parse_args()
    args.func(args)
    return args


if __name__ == "__main__":
    parse_args()
