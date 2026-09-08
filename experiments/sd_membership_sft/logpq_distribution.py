"""Log-probability distributions of the fine-tuned SD pair on audit records.

Teacher-forced protocol (the repo's established audit convention): the audited
record's document-continuation tokens are the fixed speculative-decoding
candidate tokens; both the fine-tuned target (cloud ``p``) and the fine-tuned
draft (edge ``q``) assign each of those tokens a log-probability under the
run's training prompt (prompt masked, response scored exactly as in SFT loss).
This module pools the token log-probabilities assigned by p and q over every
response token of the full member and nonmember classes and renders one
comparison plot per (run condition, class): p, auxiliary-distilled q, and
member-SFT q as three density curves in the same axes. The four plots are
saved both as individual PNGs and as one 2x2 overview matrix (rows = run
conditions, columns = member / nonmember).

Usage (from the repository root):

    CUDA_VISIBLE_DEVICES=3 uv run --no-sync python -m \
        experiments.sd_membership_sft.logpq_distribution \
        --run-dir experiments/results/sft_runs/newstection_qwen3_8b_epoch1 \
        --run-dir experiments/results/sft_runs/newstection_qwen3_8b_epoch3 \
        --gpu 0 \
        --output-dir experiments/results/sft_runs/logpq_distribution_newstection

Outputs into ``--output-dir``:

- ``logpq_distribution.png`` / ``.pdf``  2x2 overview of logp/logq KDEs: one
  comparison plot per (condition, class), three curves each (p, q
  aux-distilled, q member-SFT)
- ``logpq_<condition>_<class>.png``       the four individual plots
- ``logpq_token_probabilities.npz``       pooled token probabilities per
  (run condition, model role, class) so figures can be restyled offline
- ``RESULTS.md``                          protocol, split verification, stats
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
from transformers import AutoTokenizer

from .data import SFTRecord, collate_sft, make_sft_example
from .generalization import load_draft_model, load_finetuned_model, load_run_config
from .splits import build_split, pool_path
from .training import set_seed

ROOT = Path(__file__).resolve().parents[2]

MODEL_ROLES: tuple[tuple[str, str], ...] = (
    ("target", "p — target Qwen3-8B (member SFT)"),
    ("draft_auxiliary_distilled", "q — draft 1.7B (auxiliary-distilled KD)"),
    ("draft_member_sft", "q — draft 1.7B (member SFT)"),
)

CLASS_NAMES = ("member", "nonmember")

ROLE_COLORS = {
    "target": "tab:blue",
    "draft_auxiliary_distilled": "tab:orange",
    "draft_member_sft": "tab:green",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--run-dir",
        type=Path,
        action="append",
        required=True,
        help="Run directory of a plain-draft SFT run; repeat per epoch condition",
    )
    parser.add_argument("--pool-path", type=Path)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--kde-grid", type=int, default=256)
    parser.add_argument(
        "--kde-subsample",
        type=int,
        default=100_000,
        help="Tokens per curve used to fit the KDE (curves only; summary stats use all tokens)",
    )
    parser.add_argument("--seed", type=int, default=20260824)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("experiments/results/sft_runs/logpq_distribution_newstection"),
    )
    return parser.parse_args()


def verify_split_against_run(
    members: list[SFTRecord],
    nonmembers: list[SFTRecord],
    run_dir: Path,
) -> None:
    """Fail loud if the rebuilt split disagrees with the run's data passport.

    Identity is the response hash (the token-sequence digest); record ids
    carry a scheme prefix that was renamed (``nart:`` -> ``sft:``) after
    these runs were trained, so the prefix is not stable across commits.
    """
    artifact = json.loads((run_dir / "results.json").read_text(encoding="utf-8"))
    stored = artifact["records"]
    for class_name, rebuilt in (
        ("members", members),
        ("nonmembers", nonmembers),
    ):
        stored_hashes = [record["response_hash"] for record in stored[class_name]]
        rebuilt_hashes = [record.response_hash for record in rebuilt]
        if stored_hashes != rebuilt_hashes:
            raise RuntimeError(
                f"Rebuilt {class_name} split does not match the run passport in {run_dir}"
            )


def verify_shared_tokenizer(target_id: str, draft_id: str) -> None:
    """The teacher-forced audit assumes one tokenization for p and q."""
    target_tokenizer = AutoTokenizer.from_pretrained(target_id)
    draft_tokenizer = AutoTokenizer.from_pretrained(draft_id)
    probe = "edge-cloud speculative decoding membership audit probe 0123"
    if (
        target_tokenizer(probe).input_ids != draft_tokenizer(probe).input_ids
        or target_tokenizer.vocab_size != draft_tokenizer.vocab_size
    ):
        raise RuntimeError(
            f"Target {target_id} and draft {draft_id} tokenizers disagree; "
            "the shared-tokenization audit protocol does not hold"
        )


@torch.inference_mode()
def token_logprobabilities(
    model: Any,
    records: list[SFTRecord],
    tokenizer: Any,
    device: torch.device,
    batch_size: int,
) -> np.ndarray:
    """log-probability of every response token (prompt masked, EOS included)."""
    model.eval()
    examples = [make_sft_example(record, tokenizer) for record in records]
    order = sorted(range(len(examples)), key=lambda index: len(examples[index]["input_ids"]))
    pooled: list[np.ndarray] = []
    for start in range(0, len(order), batch_size):
        rows = [examples[index] for index in order[start : start + batch_size]]
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
        logp = (
            logits.float()
            .log_softmax(dim=-1)
            .gather(-1, labels.clamp_min(0).unsqueeze(-1))
            .squeeze(-1)
        )
        pooled.append(logp[valid].to(torch.float32).cpu().numpy())
    return np.concatenate(pooled)


def density_curve(
    values: np.ndarray,
    grid: np.ndarray,
    subsample: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """Evaluate a Gaussian KDE on ``grid`` and renormalize on that grid."""
    from scipy.stats import gaussian_kde

    if len(values) > subsample:
        values = rng.choice(values, size=subsample, replace=False)
    density = gaussian_kde(values)(grid)
    total = np.trapezoid(density, grid)
    if total <= 0:
        raise RuntimeError("KDE density has no mass on the requested grid")
    return density / total


def summarize(probabilities: np.ndarray) -> dict[str, float]:
    quantiles = np.quantile(probabilities, (0.10, 0.50, 0.90))
    return {
        "n_tokens": int(probabilities.size),
        "mean": float(probabilities.mean()),
        "q10": float(quantiles[0]),
        "q50": float(quantiles[1]),
        "q90": float(quantiles[2]),
    }


def _plot_comparison(
    axis: Any,
    curves_for_class: dict[str, np.ndarray],
    grid: np.ndarray,
) -> None:
    """Three log-probability density curves in one axes."""
    for role, role_title in MODEL_ROLES:
        density = curves_for_class[role]
        axis.fill_between(grid, density, color=ROLE_COLORS[role], alpha=0.12)
        axis.plot(
            grid,
            density,
            color=ROLE_COLORS[role],
            linewidth=1.8,
            label=role_title,
        )
    axis.set_xlim(float(grid[0]), float(grid[-1]))
    axis.grid(True, alpha=0.25)


def _shared_ceiling(curves: dict[str, dict[str, dict[str, np.ndarray]]]) -> float:
    return float(
        max(
            np.max(curves[row_label][class_name][role])
            for row_label in curves
            for role, _ in MODEL_ROLES
            for class_name in CLASS_NAMES
        )
    )


def render_figure(
    curves: dict[str, dict[str, dict[str, np.ndarray]]],
    row_labels: list[str],
    grid: np.ndarray,
    benchmark: str,
    output_dir: Path,
    space: str = "prob",
    output_suffix: str | None = None,
) -> None:
    """One comparison plot per (condition, class) plus a 2x2 overview matrix.

    ``space`` selects the axis semantics: ``prob`` (default) renders token
    probabilities on [0,1]; ``log`` renders natural-log token probabilities
    (the grid bounds are the caller's choice, e.g. [-16, 0]).
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    if space not in ("prob", "log"):
        raise ValueError(f"Unknown space {space!r}")
    suffix = (
        ("" if space == "prob" else "_logspace")
        if output_suffix is None
        else output_suffix
    )
    axis_label = "token probability" if space == "prob" else "logp / logq (nats)"
    quantity = "token-probability" if space == "prob" else "log-probability"

    ceiling = _shared_ceiling(curves) * 1.05

    for row_label in row_labels:
        for class_name in CLASS_NAMES:
            figure, axis = plt.subplots(figsize=(7.0, 4.6))
            _plot_comparison(axis, curves[row_label][class_name], grid)
            axis.set_ylim(0.0, ceiling)
            axis.set_xlabel(axis_label, fontsize=10)
            axis.set_ylabel("density", fontsize=10)
            axis.set_title(
                f"{benchmark.capitalize()} {row_label}, {class_name}: "
                f"p vs q {quantity} distributions",
                fontsize=11,
            )
            axis.legend(fontsize=9, frameon=False)
            figure.tight_layout()
            figure.savefig(output_dir / f"logpq_{row_label}_{class_name}{suffix}.png", dpi=200)
            figure.savefig(output_dir / f"logpq_{row_label}_{class_name}{suffix}.pdf")
            plt.close(figure)

    figure, axes = plt.subplots(
        len(row_labels),
        len(CLASS_NAMES),
        figsize=(12.5, 8.2),
        sharex=True,
        sharey=True,
        squeeze=False,
    )
    for row, row_label in enumerate(row_labels):
        for column, class_name in enumerate(CLASS_NAMES):
            axis = axes[row, column]
            _plot_comparison(axis, curves[row_label][class_name], grid)
            axis.set_ylim(0.0, ceiling)
            axis.set_title(f"{class_name}", fontsize=11)
            if row == len(row_labels) - 1:
                axis.set_xlabel(axis_label, fontsize=9)
            if column == 0:
                axis.set_ylabel(f"{row_label}\ndensity", fontsize=9)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    figure.legend(handles, labels, loc="upper center", ncol=3, frameon=False)
    figure.suptitle(
        f"{benchmark.capitalize()}: p vs q {quantity} distributions "
        "(teacher-forced response tokens)",
        fontsize=12,
    )
    figure.tight_layout(rect=(0, 0, 1, 0.92))
    figure.savefig(output_dir / f"logpq_distribution{suffix}.png", dpi=200)
    figure.savefig(output_dir / f"logpq_distribution{suffix}.pdf")
    plt.close(figure)


def render_markdown(
    benchmark: str,
    run_dirs: list[Path],
    row_labels: list[str],
    stats: dict[str, dict[str, dict[str, dict[str, float]]]],
    protocol: dict[str, Any],
) -> str:
    lines = [
        "# logp/logq distributions (member vs nonmember)",
        "",
        f"- Benchmark: `{benchmark}`",
        "- Run conditions: "
        + ", ".join(f"`{label}` → `{path}`" for label, path in zip(row_labels, run_dirs)),
        "- Protocol: teacher-forced response tokens under each run's fixed SFT "
        "prompt (prompt masked, appended EOS included, exactly the membership "
        "loss token set); p from the fine-tuned target, q from the fine-tuned "
        "drafts; the KDEs use natural log probabilities, while the raw pooled "
        "values and summary statistics remain probabilities = exp(logp), "
        "exp(logq).",
        f"- Records: {protocol['n_per_class']} member + {protocol['n_per_class']} "
        "nonmember (full classes, token-level pooling)",
        f"- Split verification: rebuilt member/nonmember response hashes match "
        "each run's results.json passport",
        f"- KDE: Gaussian, fitted on up to {protocol['kde_subsample']} tokens per "
        f"curve (seed {protocol['seed']}), fitted to logp/logq in "
        f"[{protocol['log_grid_min']}, {protocol['log_grid_max']}] nats and "
        "renormalized on that range",
        "",
        "## Summary statistics (all tokens, not the KDE subsample)",
        "",
    ]
    for role, role_title in MODEL_ROLES:
        lines.append(f"### {role_title}")
        lines.append("")
        lines.append("| Condition | Class | Tokens | Mean | q10 | q50 | q90 |")
        lines.append("|---|---|---:|---:|---:|---:|---:|")
        for row_label in row_labels:
            for class_name in ("member", "nonmember"):
                row = stats[row_label][role][class_name]
                lines.append(
                    f"| {row_label} | {class_name} | {row['n_tokens']} "
                    f"| {row['mean']:.4f} | {row['q10']:.4f} | {row['q50']:.4f} "
                    f"| {row['q90']:.4f} |"
                )
        gaps = [
            stats[row_label][role]["member"]["mean"]
            - stats[row_label][role]["nonmember"]["mean"]
            for row_label in row_labels
        ]
        lines.append("")
        lines.append(
            "member − nonmember mean gap: "
            + ", ".join(
                f"{row_label} {gap:+.4f}"
                for row_label, gap in zip(row_labels, gaps)
            )
        )
        lines.append("")
    lines.append(
        "Raw pooled token probabilities: `logpq_token_probabilities.npz` "
        "(keys `<condition>__<role>__<class>`); the `grid` entry is the "
        "logp/logq KDE grid."
    )
    lines.append("")
    return "\n".join(lines)


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available; run with the approved host GPU access")
    device = torch.device(f"cuda:{args.gpu}")
    torch.cuda.set_device(device)
    set_seed(args.seed)
    rng = np.random.default_rng(args.seed)

    run_dirs = [
        run_dir if run_dir.is_absolute() else ROOT / run_dir for run_dir in args.run_dir
    ]
    configs = [load_run_config(run_dir) for run_dir in run_dirs]
    benchmarks = {cfg.benchmark for cfg in configs}
    if len(benchmarks) != 1:
        raise RuntimeError(f"All run dirs must share one benchmark, got {sorted(benchmarks)}")
    benchmark = benchmarks.pop()
    row_labels = [f"epoch{cfg.target_epochs}" for cfg in configs]

    output_dir = args.output_dir if args.output_dir.is_absolute() else ROOT / args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    reference = configs[0]
    tokenizer = AutoTokenizer.from_pretrained(reference.draft_model)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    verify_shared_tokenizer(reference.target_model, reference.draft_model)

    pool = (
        args.pool_path
        if args.pool_path is not None
        else (
            reference.pool_path
            if reference.pool_path is not None
            else pool_path(benchmark)
        )
    )
    if not pool.is_absolute():
        pool = ROOT / pool
    members, nonmembers, _auxiliary, _metadata = build_split(
        benchmark,
        pool,
        tokenizer,
        reference.n_per_class,
        reference.n_aux,
        reference.data_seed,
    )
    for run_dir, cfg in zip(run_dirs, configs):
        if (cfg.n_per_class, cfg.n_aux, cfg.data_seed) != (
            reference.n_per_class,
            reference.n_aux,
            reference.data_seed,
        ):
            raise RuntimeError(
                f"{run_dir} uses a different split configuration than the first run dir"
            )
        verify_split_against_run(members, nonmembers, run_dir)

    # p and q are probabilities in (0, 1], while the figure compares their
    # natural logarithms.  A shared [-16, 0] range keeps the four panels
    # readable while retaining virtually all of the observed token mass.
    grid = np.linspace(-16.0, 0.0, args.kde_grid)
    curves: dict[str, dict[str, dict[str, np.ndarray]]] = {}
    stats: dict[str, dict[str, dict[str, dict[str, float]]]] = {}
    pooled: dict[str, np.ndarray] = {}
    for row_label, run_dir, cfg in zip(row_labels, run_dirs, configs):
        curves[row_label] = {class_name: {} for class_name in CLASS_NAMES}
        stats[row_label] = {}
        for role, _role_title in MODEL_ROLES:
            stats[row_label][role] = {}
            if role == "target":
                model = load_finetuned_model(run_dir, cfg.target_model, device)
            else:
                model = load_draft_model(run_dir, cfg.draft_model, role, device)
            try:
                for class_name, records in (
                    ("member", members),
                    ("nonmember", nonmembers),
                ):
                    started = time.perf_counter()
                    logp = token_logprobabilities(
                        model, records, tokenizer, device, args.batch_size
                    )
                    probabilities = np.exp(logp.astype(np.float64))
                    log_values = logp.astype(np.float64)
                    key = f"{row_label}__{role}__{class_name}"
                    pooled[key] = probabilities.astype(np.float32)
                    curves[row_label][class_name][role] = density_curve(
                        log_values, grid, args.kde_subsample, rng
                    )
                    stats[row_label][role][class_name] = summarize(probabilities)
                    print(
                        f"{key}: {probabilities.size} tokens, "
                        f"mean p = {probabilities.mean():.4f} "
                        f"({time.perf_counter() - started:.1f}s)",
                        flush=True,
                    )
            finally:
                del model
                torch.cuda.empty_cache()

    np.savez_compressed(
        output_dir / "logpq_token_probabilities.npz",
        grid=grid.astype(np.float32),
        **pooled,
    )
    render_figure(
        curves,
        row_labels,
        grid,
        benchmark,
        output_dir,
        space="log",
        output_suffix="",
    )
    protocol = {
        "benchmark": benchmark,
        "n_per_class": reference.n_per_class,
        "batch_size": args.batch_size,
        "kde_grid": args.kde_grid,
        "kde_subsample": args.kde_subsample,
        "log_grid_min": float(grid[0]),
        "log_grid_max": float(grid[-1]),
        "seed": args.seed,
        "tokenization": "draft tokenizer, shared with target (verified)",
        "eos_included": True,
    }
    (output_dir / "logpq_protocol.json").write_text(
        json.dumps(protocol, indent=2), encoding="utf-8"
    )
    (output_dir / "RESULTS.md").write_text(
        render_markdown(benchmark, run_dirs, row_labels, stats, protocol),
        encoding="utf-8",
    )
    print(json.dumps({"output_dir": str(output_dir)}, indent=2))


if __name__ == "__main__":
    main()
