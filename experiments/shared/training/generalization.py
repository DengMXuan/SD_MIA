"""Target-only continuation quality on a condition's immutable shared split.

The matrix entry point is experiments.model_quality.cli. This module retains
its single-run CLI and reusable generation/scoring helpers.
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
from experiments.shared.data.data import SFTRecord
from experiments.shared.data.splits import pool_path
from experiments.shared.training.training import load_causal_lm
from experiments.shared.models.precision import inference_attention


from experiments.paths import ROOT


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--pool-path", type=Path)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--samples", type=int, default=500)
    parser.add_argument("--context-tokens", type=int, default=256)
    parser.add_argument("--gen-tokens", type=int, default=128)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--bootstrap-repeats", type=int, default=1000)
    parser.add_argument("--gap-threshold", type=float, default=0.03)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--output-dir", type=Path, help="separate evaluation output directory")
    return parser.parse_args()


def load_run_config(run_dir: Path) -> Any:
    import dataclasses

    from experiments.shared.training.config import Config

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
    if samples < 1 or samples > len(records) or context_tokens < 1 or gen_tokens < 1:
        raise ValueError("positive budgets and enough distinct records required")
    rng = np.random.default_rng(seed)
    indices = rng.permutation(len(records))[:samples]
    samples_out: list[dict[str, Any]] = []
    for index in indices:
        record = records[int(index)]
        length = len(record.response_ids)
        if length < 2:
            raise ValueError(f"record {record.record_id} cannot be split into context and reference")
        context_length = context_tokens
        reference_length = gen_tokens
        if length < context_tokens + gen_tokens:
            context_length = max(1, length * context_tokens // (context_tokens + gen_tokens))
            reference_length = length - context_length
        context_ids = list(record.response_ids[:context_length])
        reference_ids = list(record.response_ids[context_length:context_length + reference_length])
        samples_out.append(
            {
                "record_id": record.record_id,
                "context_tokens": context_length,
                "reference_tokens": reference_length,
                "response_hash": record.response_hash,
                "reference_ids": reference_ids,
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
    if batch_size < 1:
        raise ValueError("batch size must be positive")
    # Bucket by the per-record budget: short references never receive 128 tokens.
    groups: dict[int, list[int]] = {}
    for index, sample in enumerate(samples):
        groups.setdefault(sample.get("reference_tokens", gen_tokens), []).append(index)
    previous_side = tokenizer.padding_side
    tokenizer.padding_side = "left"
    hypotheses = [""] * len(samples)
    try:
        for budget, indices in groups.items():
            for start in range(0, len(indices), batch_size):
                selected = indices[start:start + batch_size]
                batch = [samples[i] for i in selected]
                width = max(len(sample["prompt_ids"]) for sample in batch)
                input_ids = torch.full((len(batch), width), int(tokenizer.pad_token_id), dtype=torch.long)
                attention_mask = torch.zeros_like(input_ids)
                for row, sample in enumerate(batch):
                    offset = width - len(sample["prompt_ids"])
                    input_ids[row, offset:] = torch.tensor(sample["prompt_ids"])
                    attention_mask[row, offset:] = 1
                with inference_attention(model):
                    output = model.generate(
                        input_ids=input_ids.to(device), attention_mask=attention_mask.to(device),
                        max_new_tokens=budget, min_new_tokens=0, do_sample=False, num_beams=1,
                        pad_token_id=int(tokenizer.pad_token_id),
                    )
                for row, index in enumerate(selected):
                    hypotheses[index] = tokenizer.decode(output[row, width:], skip_special_tokens=True).strip()
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
    if len(hypotheses) != len(samples):
        raise ValueError("predictions and references must align")
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
    *,
    paired: bool = True,
) -> dict[str, float]:
    """CI for mean(left) - mean(right).

    Pairing is a property of record identity, never inferred from array length.
    """
    left, right = np.asarray(left), np.asarray(right)
    if (left.ndim != 1 or right.ndim != 1 or not len(left) or not len(right)
            or not np.isfinite(left).all() or not np.isfinite(right).all() or repeats < 1):
        raise ValueError("finite nonempty score vectors and positive bootstrap repeats required")
    if paired and len(left) != len(right):
        raise ValueError("paired bootstrap requires aligned records")
    rng = np.random.default_rng(seed)
    deltas = np.empty(repeats, dtype=np.float64)
    if paired:
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
            paired=False,
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
                seed,
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
        f"up to {protocol['gen_tokens']} tokens (short records use proportional context/reference lengths); benchmark "
        f"{protocol['benchmark']}",
        "",
        "## Fine-tuned model: member vs nonmember generation quality",
        "",
        f"Absolute mean gaps < {protocol['gap_threshold']} pass an advisory threshold; "
        "this does not establish absence of overfitting or general-purpose capability.",
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
    from experiments.shared.evaluation.quality import evaluate_quality, make_task

    args = parse_args()
    task = make_task(args.run_dir, "generalization", output=args.output_dir,
                     samples=args.samples, context_tokens=args.context_tokens,
                     gen_tokens=args.gen_tokens, batch_size=args.batch_size,
                     bootstrap_repeats=args.bootstrap_repeats, gap_threshold=args.gap_threshold)
    if args.pool_path is not None:
        raise ValueError("evaluation uses the training passport's frozen pool; omit --pool-path")
    if args.seed is not None and args.seed != task["condition"]["condition_seed"]:
        raise ValueError("evaluation seed must equal the condition seed")
    report = evaluate_quality(task, device=f"cuda:{args.gpu}")
    print(json.dumps({"output": task["output"], "summary": report["summary"]}))


if __name__ == "__main__":
    main()
