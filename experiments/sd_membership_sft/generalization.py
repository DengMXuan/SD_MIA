"""Generalization check for a fine-tuned run.

Replays the run's data split (same pool, tokenizer, sizes, and seed as
training) and measures generation quality on document continuations:

1. No-overfitting protocol: the fine-tuned target model generates
   continuations for member and nonmember documents under the fixed
   instruction prompt; BLEU-4 / ROUGE-1 / ROUGE-L are compared between the
   two classes. Member-vs-nonmember differences below 0.03 count as stable
   generation quality without overfitting; this module computes that
   comparison with bootstrap CIs and a soft gate.
2. Base-vs-fine-tuned comparison: the same samples are scored with the
   pre-fine-tuning base target, so quality degradation caused by fine-tuning
   is visible directly (reported with CIs and relative drop; no hard gate).

Optionally the same evaluation is repeated for the draft variants
(``--include-drafts``), which matters for speculative-decoding acceptance.

Outputs ``generalization.json`` and ``GENERALIZATION.md`` in the run
directory. Generation uses greedy decoding: context = first
``--context-tokens`` tokens of the document, reference = the true following
``--gen-tokens`` tokens.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import sacrebleu
import torch
from rouge_score import rouge_scorer
from transformers import AutoTokenizer

from .data import SFTRecord
from .splits import build_split, pool_path
from .training import load_causal_lm, set_seed


ROOT = Path(__file__).resolve().parents[2]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--pool-path", type=Path)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--samples", type=int, default=500)
    parser.add_argument("--context-tokens", type=int, default=256)
    parser.add_argument("--gen-tokens", type=int, default=128)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--bootstrap-repeats", type=int, default=500)
    parser.add_argument("--gap-threshold", type=float, default=0.03)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--include-drafts", action="store_true", default=False)
    return parser.parse_args()


def load_run_config(run_dir: Path) -> Any:
    import dataclasses

    from .config import Config

    artifact = json.loads((run_dir / "results.json").read_text(encoding="utf-8"))
    values = dict(artifact["config"])
    # Older run dirs predate config slimming and may carry removed fields.
    known = {field.name for field in dataclasses.fields(Config)}
    values = {key: value for key, value in values.items() if key in known}
    if values.get("pool_path"):
        values["pool_path"] = Path(values["pool_path"])
    cfg = Config(**values)
    if cfg.pool_path is not None and not Path(cfg.pool_path).exists():
        # Older run dirs record pool paths from before the pools directory
        # was renamed; fall back to the current default pool.
        cfg.pool_path = pool_path(cfg.benchmark)
    return cfg


def load_finetuned_model(
    run_dir: Path,
    model_id: str,
    device: torch.device,
    attn_implementation: str = "eager",
) -> Any:
    """Load the fine-tuned target, transparently handling LoRA and full runs."""
    for directory in ("checkpoints", "adapters"):
        path = run_dir / directory / "target"
        if not path.exists():
            continue
        if (path / "adapter_config.json").exists():
            from peft import PeftModel

            base = load_causal_lm(model_id, device, attn_implementation=attn_implementation)
            model = PeftModel.from_pretrained(base, str(path), is_trainable=False)
        else:
            model = load_causal_lm(
                str(path), device, attn_implementation=attn_implementation
            )
        model.eval()
        model.config.use_cache = True
        return model
    raise FileNotFoundError(f"No fine-tuned target checkpoint under {run_dir}")


def load_draft_model(
    run_dir: Path,
    model_id: str,
    name: str,
    device: torch.device,
    attn_implementation: str = "eager",
) -> Any:
    """Load a saved draft variant, transparently handling LoRA and full runs."""
    for directory in ("checkpoints", "adapters"):
        path = run_dir / directory / name
        if not path.exists():
            continue
        if (path / "adapter_config.json").exists():
            from peft import PeftModel

            base = load_causal_lm(model_id, device, attn_implementation=attn_implementation)
            model = PeftModel.from_pretrained(base, str(path), is_trainable=False)
        else:
            model = load_causal_lm(
                str(path), device, attn_implementation=attn_implementation
            )
        model.eval()
        model.config.use_cache = True
        return model
    raise FileNotFoundError(f"No {name} checkpoint under {run_dir}")


def build_eval_samples(
    records: list[SFTRecord],
    tokenizer: Any,
    samples: int,
    context_tokens: int,
    gen_tokens: int,
    seed: int,
) -> list[dict[str, Any]]:
    rng = np.random.default_rng(seed)
    indices = rng.permutation(len(records))[:samples]
    samples_out: list[dict[str, Any]] = []
    for index in indices:
        record = records[int(index)]
        context_ids = list(record.response_ids[:context_tokens])
        reference_ids = list(
            record.response_ids[context_tokens : context_tokens + gen_tokens]
        )
        if len(reference_ids) < gen_tokens or len(context_ids) < context_tokens:
            continue
        samples_out.append(
            {
                "record_id": record.record_id,
                "prompt_ids": list(record.prompt_ids or ()) + context_ids,
                "reference": tokenizer.decode(
                    reference_ids, skip_special_tokens=True
                ),
            }
        )
    return samples_out


@torch.inference_mode()
def generate_continuations(
    model: Any,
    samples: list[dict[str, Any]],
    tokenizer: Any,
    device: torch.device,
    gen_tokens: int,
    batch_size: int,
) -> list[str]:
    previous_side = tokenizer.padding_side
    tokenizer.padding_side = "left"
    hypotheses: list[str] = []
    try:
        for start in range(0, len(samples), batch_size):
            batch = samples[start : start + batch_size]
            lengths = [len(sample["prompt_ids"]) for sample in batch]
            width = max(lengths)
            input_ids = torch.full(
                (len(batch), width), int(tokenizer.pad_token_id), dtype=torch.long
            )
            attention_mask = torch.zeros((len(batch), width), dtype=torch.long)
            for row, sample in enumerate(batch):
                offset = width - len(sample["prompt_ids"])
                input_ids[row, offset:] = torch.tensor(sample["prompt_ids"])
                attention_mask[row, offset:] = 1
            output = model.generate(
                input_ids=input_ids.to(device),
                attention_mask=attention_mask.to(device),
                max_new_tokens=gen_tokens,
                do_sample=False,
                num_beams=1,
                pad_token_id=int(tokenizer.pad_token_id),
            )
            generated = output[:, width:]
            for row in range(len(batch)):
                hypotheses.append(
                    tokenizer.decode(
                        generated[row], skip_special_tokens=True
                    ).strip()
                )
    finally:
        tokenizer.padding_side = previous_side
    return hypotheses


class GenerationQualityScorer:
    """Per-sample BLEU-4, ROUGE-1, and ROUGE-L F1 scores."""

    METRICS = ("bleu4", "rouge1", "rougeL")

    def __init__(self) -> None:
        self._rouge = rouge_scorer.RougeScorer(
            ["rouge1", "rougeL"], use_stemmer=True
        )

    def score(self, hypothesis: str, reference: str) -> dict[str, float]:
        bleu = sacrebleu.sentence_bleu(hypothesis, [reference]).score / 100.0
        rouge = self._rouge.score(reference, hypothesis)
        return {
            "bleu4": bleu,
            "rouge1": rouge["rouge1"].fmeasure,
            "rougeL": rouge["rougeL"].fmeasure,
        }


def score_samples(
    hypotheses: list[str], samples: list[dict[str, Any]]
) -> dict[str, np.ndarray]:
    scorer = GenerationQualityScorer()
    values: dict[str, list[float]] = {name: [] for name in scorer.METRICS}
    for hypothesis, sample in zip(hypotheses, samples):
        row = scorer.score(hypothesis, sample["reference"])
        for name in scorer.METRICS:
            values[name].append(row[name])
    return {
        name: np.asarray(scores, dtype=np.float64)
        for name, scores in values.items()
    }


def paired_bootstrap_delta(
    left: np.ndarray,
    right: np.ndarray,
    repeats: int,
    seed: int,
) -> dict[str, float]:
    """CI for mean(left) - mean(right).

    Equal-length inputs are resampled with shared indices (paired); unequal
    lengths (e.g. member vs nonmember after per-class length filtering)
    are resampled independently.
    """
    rng = np.random.default_rng(seed)
    deltas = np.empty(repeats, dtype=np.float64)
    if len(left) == len(right):
        count = len(left)
        for repeat in range(repeats):
            index = rng.integers(0, count, size=count)
            deltas[repeat] = left[index].mean() - right[index].mean()
    else:
        for repeat in range(repeats):
            left_index = rng.integers(0, len(left), size=len(left))
            right_index = rng.integers(0, len(right), size=len(right))
            deltas[repeat] = left[left_index].mean() - right[right_index].mean()
    return {
        "delta": float(left.mean() - right.mean()),
        "ci95_low": float(np.quantile(deltas, 0.025)),
        "ci95_high": float(np.quantile(deltas, 0.975)),
    }


def evaluate_model(
    model: Any,
    member_samples: list[dict[str, Any]],
    nonmember_samples: list[dict[str, Any]],
    tokenizer: Any,
    device: torch.device,
    gen_tokens: int,
    batch_size: int,
) -> dict[str, dict[str, np.ndarray]]:
    output: dict[str, dict[str, np.ndarray]] = {}
    for class_name, samples in (
        ("member", member_samples),
        ("nonmember", nonmember_samples),
    ):
        hypotheses = generate_continuations(
            model, samples, tokenizer, device, gen_tokens, batch_size
        )
        output[class_name] = score_samples(hypotheses, samples)
    return output


def summarize_model_scores(
    tuned: dict[str, dict[str, np.ndarray]],
    base: dict[str, dict[str, np.ndarray]],
    metrics: tuple[str, ...],
    bootstrap_repeats: int,
    seed: int,
    gap_threshold: float,
) -> dict[str, Any]:
    """Build the no-overfitting gaps and the base-vs-tuned deltas."""
    gaps: dict[str, Any] = {}
    gate_pass = True
    for metric in metrics:
        gap = paired_bootstrap_delta(
            tuned["member"][metric],
            tuned["nonmember"][metric],
            bootstrap_repeats,
            seed,
        )
        passed = abs(gap["delta"]) < gap_threshold
        gate_pass = gate_pass and passed
        gaps[metric] = {
            **gap,
            "member_mean": float(tuned["member"][metric].mean()),
            "nonmember_mean": float(tuned["nonmember"][metric].mean()),
            "threshold": gap_threshold,
            "gate": "PASS" if passed else "FAIL",
        }

    degradation: dict[str, Any] = {}
    for class_name in ("member", "nonmember"):
        for metric in metrics:
            delta = paired_bootstrap_delta(
                base[class_name][metric],
                tuned[class_name][metric],
                bootstrap_repeats,
                seed + 1,
            )
            base_mean = float(base[class_name][metric].mean())
            tuned_mean = float(tuned[class_name][metric].mean())
            degradation[f"{class_name}/{metric}"] = {
                **delta,
                "base_mean": base_mean,
                "tuned_mean": tuned_mean,
                "relative_drop": (
                    (base_mean - tuned_mean) / base_mean if base_mean > 1e-9 else 0.0
                ),
            }

    return {
        "member_minus_nonmember": gaps,
        "overfitting_gate": "PASS" if gate_pass else "FAIL",
        "base_minus_tuned": degradation,
    }


def render_markdown(
    run_dir: Path,
    model_name: str,
    summary: dict[str, Any],
    protocol: dict[str, Any],
) -> str:
    metrics = ("bleu4", "rouge1", "rougeL")
    lines = [
        "# Generalization Check (generation-quality protocol)",
        "",
        f"- Run directory: `{run_dir}`",
        f"- Evaluated model: `{model_name}`",
        f"- Protocol: {protocol['samples_per_class']} member + "
        f"{protocol['samples_per_class']} nonmember samples; context "
        f"{protocol['context_tokens']} tokens, greedy generation "
        f"{protocol['gen_tokens']} tokens; benchmark "
        f"{protocol['benchmark']}",
        "",
        "## Fine-tuned model: member vs nonmember generation quality",
        "",
        "Member/nonmember differences < 0.03 count as no-overfitting "
        "evidence; the gate here is soft (advisory).",
        "",
        "| Metric | Member | Nonmember | Gap (95% CI) | Gate |",
        "|---|---:|---:|---:|---|",
    ]
    for metric in metrics:
        row = summary["member_minus_nonmember"][metric]
        lines.append(
            f"| {metric} | {row['member_mean']:.4f} | {row['nonmember_mean']:.4f} "
            f"| {row['delta']:+.4f} [{row['ci95_low']:+.4f}, {row['ci95_high']:+.4f}] "
            f"| {row['gate']} |"
        )
    lines.extend(
        [
            "",
            f"Overall no-overfitting gate: **{summary['overfitting_gate']}**",
            "",
            "## Base vs fine-tuned generation quality",
            "",
            "Deltas are base minus fine-tuned on the same samples: a positive "
            "delta means the base model scored higher, i.e. fine-tuning "
            "degraded that metric; ``relative drop`` scales it by the base "
            "score. No hard gate is applied.",
            "",
            "| Class | Metric | Base | Fine-tuned | Delta (95% CI) | Relative drop |",
            "|---|---|---:|---:|---:|---:|",
        ]
    )
    for class_name in ("member", "nonmember"):
        for metric in metrics:
            row = summary["base_minus_tuned"][f"{class_name}/{metric}"]
            lines.append(
                f"| {class_name} | {metric} | {row['base_mean']:.4f} "
                f"| {row['tuned_mean']:.4f} "
                f"| {row['delta']:+.4f} [{row['ci95_low']:+.4f}, {row['ci95_high']:+.4f}] "
                f"| {row['relative_drop']:+.2%} |"
            )
    lines.append("")
    return "\n".join(lines)


def main() -> None:
    args = parse_args()
    run_dir = args.run_dir if args.run_dir.is_absolute() else ROOT / args.run_dir
    cfg = load_run_config(run_dir)
    if cfg.benchmark == "legacy":
        raise RuntimeError(
            "The generalization check requires a pool-benchmark run, not legacy PDF data"
        )
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available; run with the approved host GPU access")

    seed = args.seed if args.seed is not None else cfg.data_seed + 7
    device = torch.device(f"cuda:{args.gpu}")
    torch.cuda.set_device(device)
    set_seed(seed)

    tokenizer = AutoTokenizer.from_pretrained(cfg.draft_model)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    pool = (
        args.pool_path
        if args.pool_path is not None
        else (cfg.pool_path if cfg.pool_path is not None else pool_path(cfg.benchmark))
    )
    if not pool.is_absolute():
        pool = ROOT / pool
    members, nonmembers, _auxiliary, split_metadata = build_split(
        cfg.benchmark,
        pool,
        tokenizer,
        cfg.n_per_class,
        cfg.n_aux,
        cfg.data_seed,
    )
    member_samples = build_eval_samples(
        members, tokenizer, args.samples, args.context_tokens, args.gen_tokens, seed
    )
    nonmember_samples = build_eval_samples(
        nonmembers, tokenizer, args.samples, args.context_tokens, args.gen_tokens, seed
    )

    tuned_target = load_finetuned_model(run_dir, cfg.target_model, device)
    base_target = load_causal_lm(cfg.target_model, device)
    base_target.config.use_cache = True

    tuned_scores = evaluate_model(
        tuned_target,
        member_samples,
        nonmember_samples,
        tokenizer,
        device,
        args.gen_tokens,
        args.batch_size,
    )
    base_scores = evaluate_model(
        base_target,
        member_samples,
        nonmember_samples,
        tokenizer,
        device,
        args.gen_tokens,
        args.batch_size,
    )
    summary = summarize_model_scores(
        tuned_scores,
        base_scores,
        GenerationQualityScorer.METRICS,
        args.bootstrap_repeats,
        seed,
        args.gap_threshold,
    )

    drafts: dict[str, Any] = {}
    if args.include_drafts:
        for name, model_id in (
            ("base_draft", cfg.draft_model),
            ("draft_auxiliary_distilled", cfg.draft_model),
            ("draft_member_sft", cfg.draft_model),
        ):
            if name == "base_draft":
                model = load_causal_lm(model_id, device)
            else:
                model = load_draft_model(run_dir, model_id, name, device)
            model.config.use_cache = True
            draft_scores = evaluate_model(
                model,
                member_samples,
                nonmember_samples,
                tokenizer,
                device,
                args.gen_tokens,
                args.batch_size,
            )
            drafts[name] = {
                class_name: {
                    metric: float(draft_scores[class_name][metric].mean())
                    for metric in GenerationQualityScorer.METRICS
                }
                for class_name in ("member", "nonmember")
            }

    protocol = {
        "benchmark": cfg.benchmark,
        "pool_path": str(pool),
        "pool_sha256": split_metadata["pool_sha256"],
        "split_seed": cfg.data_seed,
        "samples_per_class": len(member_samples),
        "context_tokens": args.context_tokens,
        "gen_tokens": args.gen_tokens,
        "decoding": "greedy",
        "gap_threshold": args.gap_threshold,
        "bootstrap_repeats": args.bootstrap_repeats,
        "include_drafts": args.include_drafts,
    }
    artifact = {
        "run_dir": str(run_dir),
        "protocol": protocol,
        "target_summary": summary,
        "drafts": drafts,
    }
    (run_dir / "generalization.json").write_text(
        json.dumps(artifact, indent=2), encoding="utf-8"
    )
    (run_dir / "GENERALIZATION.md").write_text(
        render_markdown(run_dir, cfg.target_model, summary, protocol),
        encoding="utf-8",
    )
    print(json.dumps({"overfitting_gate": summary["overfitting_gate"]}, indent=2))


if __name__ == "__main__":
    main()
