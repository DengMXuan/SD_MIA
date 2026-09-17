"""Acceptance-gap membership audit: |logp - logq| for edge-cloud SD.

Threat model (SD-protocol conformance without simulating communication):
the client holds the fine-tuned draft ``q`` white-box; the cloud deploys the
fine-tuned target ``p``. The audited record's response tokens are the fixed
speculative-decoding candidate tokens, and the verification quantity
``logp_i - logq_i`` — the logarithm of the acceptance ratio that governs
``A_i ~ Bernoulli(min(1, p_i/q_i))`` — is the signal source. Teacher-forced
log-probabilities are the exact expected-value realization of that protocol
quantity (infinitely many verification rounds, no censoring).

The audit score aggregates the per-token gap over a record's response tokens.
Two-sided scores read the full verification quantity (they include the
``p >= q`` side that finite accept bits censor away — members memorize, so
``p`` sits above ``q`` there):

- ``log_ratio`` (main): ``mean_i |logp_i - logq_i|``
- ``prob_space`` (control): ``mean_i |p_i - q_i|`` in probability space

One-sided protocol-observable scores use only what finite verification
rounds reveal (member-positive orientation: memorization pushes the target
above the draft, so members are accepted *more*):

- ``mean_alpha``: ``mean_i min(1, p_i/q_i)`` — the exact expected acceptance
- ``sampled_acceptance``: ``mean_i r_hat_i`` with
  ``r_hat_i ~ Binomial(R, min(1, p_i/q_i)) / R`` — the finite-R observable

Draft variants: ``draft_auxiliary_distilled`` (deployment-aligned mainline,
member-blind) and ``draft_member_sft`` (boundary condition).

Metrics: rank-AUC (member positive), TPR@10%FPR and TPR@1%FPR with
empirical nonmember-quantile thresholds, 95% bootstrap CIs over records.

Usage (from the repository root):

    CUDA_VISIBLE_DEVICES=3 uv run --no-sync python -m \
        experiments.sd_membership_sft.pq_gap_mia \
        --run-dir experiments/results/sft_runs/newstection_qwen3_8b_epoch3 \
        --gpu 0 \
        --output-dir experiments/results/sft_runs/pq_gap_mia_newstection_epoch3
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
from .data import (SFTRecord, collate_sft, make_sft_example)
from .generalization import (load_draft_model, load_finetuned_model, load_run_config)
from .scoring_common import (prepare_scoring_records)
from .audit_metrics import (order_statistic_threshold)
from .training import (set_seed)

ROOT = Path(__file__).resolve().parents[2]

DRAFT_ROLES: tuple[tuple[str, str], ...] = (
    ("draft_auxiliary_distilled", "q aux-distilled (deployment mainline)"),
    ("draft_member_sft", "q member-SFT (boundary)"),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--pool-path", type=Path)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--repeats", type=int, default=16)
    parser.add_argument("--bootstrap-repeats", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=20260824)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Defaults to results/sft_runs/pq_gap_mia_<benchmark>_<condition>",
    )
    return parser.parse_args()


# ---------------------------------------------------------------- scoring ---

def log_ratio_score(logp: np.ndarray, logq: np.ndarray) -> float:
    """Main score: mean |logp - logq| over the record's response tokens."""
    return float(np.mean(np.abs(logp - logq)))


def prob_space_score(logp: np.ndarray, logq: np.ndarray) -> float:
    """Control score: mean |p - q| in probability space (the plotted gap)."""
    return float(np.mean(np.abs(np.exp(logp) - np.exp(logq))))


def mean_alpha_score(logp: np.ndarray, logq: np.ndarray) -> float:
    """Exact one-sided observable: mean expected acceptance min(1, p/q)."""
    return float(np.mean(np.clip(np.exp(logp - logq), 0.0, 1.0)))


def sampled_acceptance_score(
    logp: np.ndarray,
    logq: np.ndarray,
    repeats: int,
    rng: np.random.Generator,
) -> float:
    """Finite-R protocol-observable acceptance: mean observed accept rate."""
    alpha = np.clip(np.exp(logp - logq), 0.0, 1.0)
    return float(np.mean(rng.binomial(repeats, alpha) / repeats))


SCORE_NAMES = ("log_ratio", "prob_space", "mean_alpha", "sampled_acceptance")
LOGSUMEXP_SEQUENCE_CHUNK = 64


def score_functions(repeats: int, rng: np.random.Generator) -> dict[str, Any]:
    return {
        "log_ratio": log_ratio_score,
        "prob_space": prob_space_score,
        "mean_alpha": mean_alpha_score,
        "sampled_acceptance": lambda logp, logq: sampled_acceptance_score(
            logp, logq, repeats, rng
        ),
    }


def selected_token_logprobs(
    logits: torch.Tensor,
    labels: torch.Tensor,
    sequence_chunk: int = LOGSUMEXP_SEQUENCE_CHUNK,
) -> torch.Tensor:
    """Compute selected-token log-probabilities with an FP32 normalizer.

    Qwen checkpoints are normally loaded in BF16.  Calling ``logsumexp`` on
    BF16 logits makes the vocabulary normalizer itself low precision, which is
    especially harmful for the p-q difference used by this experiment.  The
    full vocabulary is converted in sequence chunks so the FP32 operation does
    not require an additional full-length FP32 logits tensor.
    """
    if sequence_chunk <= 0:
        raise ValueError("sequence_chunk must be positive")
    selected = logits.gather(
        -1, labels.clamp_min(0).unsqueeze(-1)
    ).squeeze(-1).float()
    normalizer = torch.empty_like(selected, dtype=torch.float32)
    for start in range(0, logits.shape[1], sequence_chunk):
        end = min(start + sequence_chunk, logits.shape[1])
        normalizer[:, start:end] = torch.logsumexp(
            logits[:, start:end, :].float(), dim=-1
        )
    return selected - normalizer


# --------------------------------------------------------------- metrics ---

def rank_auc(member: np.ndarray, nonmember: np.ndarray) -> float:
    """AUC with member as the positive class (Mann-Whitney rank statistic)."""
    combined = np.concatenate([member, nonmember])
    # average ranks inside tie groups so ties contribute 0.5
    _, inverse, counts = np.unique(combined, return_inverse=True, return_counts=True)
    ranks = np.cumsum(counts) - (counts - 1) / 2.0
    ranks = ranks[inverse]
    member_rank_sum = ranks[: len(member)].sum()
    return float(
        (member_rank_sum - len(member) * (len(member) + 1) / 2.0)
        / (len(member) * len(nonmember))
    )


def tpr_at_fpr(member: np.ndarray, nonmember: np.ndarray, fpr: float) -> float:
    """TPR at an empirical FPR using a discrete upper-tail threshold."""
    threshold = order_statistic_threshold(nonmember, fpr)
    return float(np.mean(member > threshold))


def bootstrap_metrics(
    member: np.ndarray,
    nonmember: np.ndarray,
    repeats: int,
    seed: int,
) -> dict[str, dict[str, float]]:
    """Record-level bootstrap CIs for AUC and the two TPR@FPR points."""
    rng = np.random.default_rng(seed)
    auc = np.empty(repeats)
    tpr10 = np.empty(repeats)
    tpr01 = np.empty(repeats)
    for repeat in range(repeats):
        member_sample = member[rng.integers(0, len(member), size=len(member))]
        nonmember_sample = nonmember[
            rng.integers(0, len(nonmember), size=len(nonmember))
        ]
        auc[repeat] = rank_auc(member_sample, nonmember_sample)
        tpr10[repeat] = tpr_at_fpr(member_sample, nonmember_sample, 0.10)
        tpr01[repeat] = tpr_at_fpr(member_sample, nonmember_sample, 0.01)
    return {
        name: {
            "point": point,
            "ci95_low": float(np.quantile(values, 0.025)),
            "ci95_high": float(np.quantile(values, 0.975)),
        }
        for name, point, values in (
            ("auc", rank_auc(member, nonmember), auc),
            ("tpr@10%fpr", tpr_at_fpr(member, nonmember, 0.10), tpr10),
            ("tpr@1%fpr", tpr_at_fpr(member, nonmember, 0.01), tpr01),
        )
    }


# ------------------------------------------------------------- forward -----

@torch.inference_mode()
def record_logprobabilities(
    model: Any,
    records: list[SFTRecord],
    tokenizer: Any,
    device: torch.device,
    batch_size: int,
    empty_cache_each_batch: bool = False,
    progress: Any = None,
) -> list[np.ndarray]:
    """Per-record log-probability arrays over response tokens (+ EOS)."""
    model.eval()
    examples = [make_sft_example(record, tokenizer) for record in records]
    order = sorted(
        range(len(examples)), key=lambda index: len(examples[index]["input_ids"])
    )
    outputs: list[np.ndarray | None] = [None] * len(examples)
    starts = list(range(0, len(order), batch_size))
    if progress is not None:
        starts = progress.track(starts, "token probability scoring", unit="batches")
    for start in starts:
        indices = order[start : start + batch_size]
        rows = [examples[index] for index in indices]
        batch = {
            key: value.to(device)
            for key, value in collate_sft(rows, int(tokenizer.pad_token_id)).items()
        }
        logits = model(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
            use_cache=False,
        ).logits[:, :-1]
        labels = batch["labels"][:, 1:]
        valid = labels.ne(-100)
        # Keep the vocabulary normalizer in FP32.  It is computed in sequence
        # chunks to avoid materializing a full FP32 vocabulary tensor.
        logp = selected_token_logprobs(logits, labels)
        for row, index in enumerate(indices):
            outputs[index] = logp[row][valid[row]].to(torch.float32).cpu().numpy()
        del logits, logp, labels, valid, batch
        if empty_cache_each_batch and device.type == "cuda":
            torch.cuda.empty_cache()
    if any(output is None for output in outputs):
        raise RuntimeError("Forward pass left records unscored")
    return [output for output in outputs if output is not None]


# ------------------------------------------------------------- reporting ---

def render_histogram(
    member: np.ndarray,
    nonmember: np.ndarray,
    score_name: str,
    draft_label: str,
    output_dir: Path,
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figure, axis = plt.subplots(figsize=(7.0, 4.6))
    axis.hist(
        nonmember,
        bins=60,
        density=True,
        alpha=0.55,
        label="nonmember",
        color="tab:blue",
    )
    axis.hist(
        member,
        bins=60,
        density=True,
        alpha=0.55,
        label="member",
        color="tab:red",
    )
    axis.set_xlabel(f"score ({score_name})", fontsize=10)
    axis.set_ylabel("density", fontsize=10)
    axis.set_title(f"Per-record scores: {draft_label}", fontsize=11)
    axis.legend(fontsize=9, frameon=False)
    figure.tight_layout()
    figure.savefig(output_dir / "score_histogram.png", dpi=200)
    plt.close(figure)


def render_markdown(
    run_dir: Path,
    protocol: dict[str, Any],
    table: dict[str, dict[str, dict[str, dict[str, float]]]],
) -> str:
    lines = [
        "# Acceptance-gap membership audit (|logp − logq|)",
        "",
        f"- Run: `{run_dir}`",
        "- Threat model: see `THREAT_MODEL.md` in this directory",
        f"- Records: {protocol['n_member']} member + {protocol['n_nonmember']} "
        "nonmember, full classes, per-record token-level aggregation",
        "- Signal: teacher-forced logp (fine-tuned target 8B) and logq "
        "(fine-tuned draft 1.7B) over the record's response tokens — the "
        "exact expected value of the SD verification quantity "
        "logp − logq (acceptance-ratio log); no communication simulated.",
        "- Orientation: two-sided scores (log_ratio, prob_space) are larger "
        "for members; one-sided observables (mean_alpha, sampled_acceptance) "
        "are member-positive as acceptance — memorization pushes the target "
        "above the draft, so members are accepted more. The two-sided main "
        "score includes the p ≥ q side that finite accept bits censor away.",
        f"- Sampled ablation: R = {protocol['repeats']} Bernoulli verification "
        "rounds per token, score = mean observed accept rate.",
        "- Thresholds: TPR@FPR thresholds use discrete nonmember order statistics; "
        f"CIs are {protocol['bootstrap_repeats']} record-level bootstrap "
        "(95%).",
        "",
        "| Draft | Score | AUC | TPR@10%FPR | TPR@1%FPR |",
        "|---|---|---|---|---|",
    ]
    for draft_label, draft_table in table.items():
        for score_name in SCORE_NAMES:
            row = draft_table[score_name]
            auc, tpr10, tpr01 = row["auc"], row["tpr@10%fpr"], row["tpr@1%fpr"]
            lines.append(
                f"| {draft_label} | {score_name} "
                f"| {auc['point']:.4f} [{auc['ci95_low']:.4f}, {auc['ci95_high']:.4f}] "
                f"| {tpr10['point']:.4f} [{tpr10['ci95_low']:.4f}, {tpr10['ci95_high']:.4f}] "
                f"| {tpr01['point']:.4f} [{tpr01['ci95_low']:.4f}, {tpr01['ci95_high']:.4f}] |"
            )
    lines.extend(
        [
            "",
            "Per-record scores and per-token log-probabilities are persisted in "
            "`pq_gap_scores.npz` and `pq_gap_token_logps.npz`; the main-score "
            "member/nonmember histogram is `score_histogram.png`.",
            "",
        ]
    )
    return "\n".join(lines)


# ------------------------------------------------------------------ main ---

def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available; run with the approved host GPU access")
    device = torch.device(f"cuda:{args.gpu}")
    torch.cuda.set_device(device)
    set_seed(args.seed)

    run_dir = args.run_dir if args.run_dir.is_absolute() else ROOT / args.run_dir
    cfg = load_run_config(run_dir)
    output_dir = (
        args.output_dir
        if args.output_dir is not None
        else Path(
            f"experiments/results/sft_runs/pq_gap_mia_{cfg.benchmark}"
            f"{'_epoch' + str(cfg.target_epochs) if cfg.target_epochs else ''}"
        )
    )
    output_dir = output_dir if output_dir.is_absolute() else ROOT / output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    cfg, prepared = prepare_scoring_records(run_dir, cfg, args.pool_path)
    tokenizer = prepared.tokenizer
    members, nonmembers = prepared.members, prepared.nonmembers
    records, labels = prepared.records, prepared.labels

    logp_by_role: dict[str, list[np.ndarray]] = {}
    for role in ("target", *(role for role, _label in DRAFT_ROLES)):
        if role == "target":
            model = load_finetuned_model(run_dir, cfg.target_model, device)
        else:
            model = load_draft_model(run_dir, cfg.draft_model, role, device)
        try:
            started = time.perf_counter()
            logp_by_role[role] = record_logprobabilities(
                model, records, tokenizer, device, args.batch_size
            )
            print(
                f"{role}: scored {len(records)} records "
                f"({time.perf_counter() - started:.1f}s)",
                flush=True,
            )
        finally:
            del model
            torch.cuda.empty_cache()

    lengths = np.array([len(arr) for arr in logp_by_role["target"]], dtype=np.int64)
    rng = np.random.default_rng(args.seed)
    scorers = score_functions(args.repeats, rng)
    scores: dict[str, np.ndarray] = {}
    table: dict[str, dict[str, dict[str, dict[str, float]]]] = {}
    for draft_role, draft_label in DRAFT_ROLES:
        table[draft_label] = {}
        for score_name in SCORE_NAMES:
            scorer = scorers[score_name]
            values = np.array(
                [
                    scorer(logp, logq)
                    for logp, logq in zip(
                        logp_by_role["target"], logp_by_role[draft_role]
                    )
                ],
                dtype=np.float64,
            )
            key = f"{draft_role}__{score_name}"
            scores[key] = values
            table[draft_label][score_name] = bootstrap_metrics(
                values[labels == 1], values[labels == 0], args.bootstrap_repeats, args.seed
            )
            print(
                f"{key}: AUC = {table[draft_label][score_name]['auc']['point']:.4f}, "
                f"TPR@10% = {table[draft_label][score_name]['tpr@10%fpr']['point']:.4f}, "
                f"TPR@1% = {table[draft_label][score_name]['tpr@1%fpr']['point']:.4f}",
                flush=True,
            )

    np.savez_compressed(
        output_dir / "pq_gap_scores.npz",
        labels=labels,
        record_ids=np.array([record.record_id for record in records]),
        **scores,
    )
    np.savez_compressed(
        output_dir / "pq_gap_token_logps.npz",
        lengths=lengths,
        target=np.concatenate(logp_by_role["target"]),
        **{role: np.concatenate(logp_by_role[role]) for role, _ in DRAFT_ROLES},
    )
    mainline_key = f"{DRAFT_ROLES[0][0]}__log_ratio"
    render_histogram(
        scores[mainline_key][labels == 1],
        scores[mainline_key][labels == 0],
        "log_ratio",
        DRAFT_ROLES[0][1],
        output_dir,
    )
    protocol = {
        "run_dir": str(run_dir),
        "benchmark": cfg.benchmark,
        "target_epochs": cfg.target_epochs,
        "n_member": len(members),
        "n_nonmember": len(nonmembers),
        "batch_size": args.batch_size,
        "repeats": args.repeats,
        "bootstrap_repeats": args.bootstrap_repeats,
        "seed": args.seed,
    }
    (output_dir / "pq_gap_protocol.json").write_text(
        json.dumps(protocol, indent=2), encoding="utf-8"
    )
    (output_dir / "RESULTS.md").write_text(
        render_markdown(run_dir, protocol, table), encoding="utf-8"
    )
    print(json.dumps({"output_dir": str(output_dir)}, indent=2))


if __name__ == "__main__":
    main()
