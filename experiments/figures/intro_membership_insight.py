"""Seaborn introduction figures from aligned, real teacher-forced p/q caches.

No language model is loaded. Panel (b) is the theoretical acceptance probability,
not observed verifier feedback. See the generated README and provenance.json.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import MultipleLocator, PercentFormatter
import numpy as np
import pandas as pd
import seaborn as sns


ROOT = Path(__file__).resolve().parents[2]
NAMES = {1: "Member", 0: "Non-member"}
COLORS = {"Member": "#C45A27", "Non-member": "#2378A5"}
STYLES = {"Member": "-", "Non-member": "--"}


def sha256(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def load_records(target_path, draft_path, partition_path, benchmark="wikitection"):
    """Check identity, labels, lengths and provenance before pairing tokens."""
    with np.load(target_path, allow_pickle=False) as archive:
        target = dict(archive)
    with np.load(draft_path, allow_pickle=False) as archive:
        draft = dict(archive)
    for key in ("record_ids", "labels", "lengths"):
        if not np.array_equal(target[key], draft[key]):
            raise ValueError(f"Target and draft disagree on {key}")
    if str(target["role"].item()) != "target" or str(draft["role"].item()) != "draft_auxiliary_distilled":
        raise ValueError("Expected target and auxiliary-distilled draft caches")
    ids, labels, lengths = (target[key] for key in ("record_ids", "labels", "lengths"))
    if (ids.ndim != 1 or labels.shape != ids.shape or lengths.shape != ids.shape
            or len(np.unique(ids)) != len(ids) or not np.isin(labels, [0, 1]).all()
            or not np.issubdtype(lengths.dtype, np.integer) or (lengths <= 1).any()):
        raise ValueError("Invalid record metadata")
    for data in (target, draft):
        if (data["logp"].shape != (int(lengths.sum()),)
                or not np.isfinite(data["logp"]).all() or (data["logp"] > 1e-6).any()):
            raise ValueError("Invalid log-probability array")
    source_meta = [json.loads(path.with_suffix(".npz.json").read_text())
                   for path in (target_path, draft_path)]
    for key in ("run_dir", "benchmark", "epoch", "records", "tokens"):
        if source_meta[0][key] != source_meta[1][key]:
            raise ValueError(f"Source metadata differ on {key}")
    if source_meta[0]["benchmark"] != benchmark or source_meta[0]["epoch"] != 3:
        raise ValueError(f"Expected {benchmark}, epoch 3")
    if (source_meta[0]["base_model"] != "Qwen/Qwen3-8B-Base"
            or source_meta[1]["base_model"] != "Qwen/Qwen3-1.7B-Base"):
        raise ValueError("Unexpected model pair")
    partition = json.loads(partition_path.read_text())["partition"]
    diagnostic_ids = set(partition["partitions"]["D"])
    if not diagnostic_ids.issubset(set(ids.tolist())):
        raise ValueError("Diagnostic partition contains missing record IDs")
    offsets = np.r_[0, lengths.cumsum()]
    records = []
    for index, record_id in enumerate(ids):
        if record_id not in diagnostic_ids:
            continue
        # The historical producer appends a final EOS. Follow the existing
        # replay cache's removal rule, preserving all original response tokens.
        start, end = int(offsets[index]), int(offsets[index + 1]) - 1
        records.append({
            "record_id": str(record_id), "label": int(labels[index]),
            "logp": target["logp"][start:end].astype(float),
            "logq": draft["logp"][start:end].astype(float),
        })
    for label, name in NAMES.items():
        expected = partition["counts"]["D"]["member" if label else "nonmember"]
        if sum(r["label"] == label for r in records) != expected:
            raise ValueError(f"Unexpected diagnostic count for {name}")
    return records, source_meta


def kde_sample(records, tokens_per_record, rng):
    frames = []
    for record in records:
        count = min(tokens_per_record, len(record["logp"]))
        index = rng.choice(len(record["logp"]), count, replace=False)
        frames.append(pd.DataFrame({
            "logp": record["logp"][index], "logq": record["logq"][index],
            "membership": NAMES[record["label"]], "weight": 1.0 / count,
            "record_id": record["record_id"], "token_position": index,
        }))
    return pd.concat(frames, ignore_index=True)


def conditional_summary(records, edges, bootstrap, min_documents, rng):
    """Average within each document/bin, then across documents; cluster bootstrap."""
    n_bins = len(edges) - 1
    rows, document_rows = [], []
    for label, membership in NAMES.items():
        group = [r for r in records if r["label"] == label]
        means = np.full((len(group), n_bins), np.nan)
        counts = np.zeros((len(group), n_bins), dtype=int)
        for i, record in enumerate(group):
            logq = record["logq"]
            alpha = np.exp(np.minimum(0.0, record["logp"] - logq))
            assignments = np.searchsorted(edges, logq, side="right") - 1
            assignments[logq == edges[-1]] = n_bins - 1
            for j in range(n_bins):
                select = assignments == j
                counts[i, j] = select.sum()
                if counts[i, j]:
                    means[i, j] = alpha[select].mean()
                    document_rows.append({
                        "record_id": record["record_id"], "membership": membership,
                        "bin": j, "tokens": int(counts[i, j]),
                        "mean_theoretical_acceptance": means[i, j],
                    })
        valid = np.isfinite(means)
        denominator = valid.sum(axis=0)
        center = np.divide(np.nansum(means, axis=0), denominator,
                           out=np.full(n_bins, np.nan), where=denominator > 0)
        samples = np.full((bootstrap, n_bins), np.nan)
        # One resampled set of documents is shared across all bins in a replicate.
        for start in range(0, bootstrap, 100):
            weights = rng.multinomial(len(group), np.full(len(group), 1 / len(group)),
                                      size=min(100, bootstrap - start))
            den = weights @ valid.astype(float)
            np.divide(weights @ np.nan_to_num(means), den,
                      out=samples[start:start + len(weights)], where=den > 0)
        for j in range(n_bins):
            supported = denominator[j] >= min_documents
            bounds = np.nanquantile(samples[:, j], [0.025, 0.975]) if supported else [np.nan, np.nan]
            rows.append({
                "membership": membership, "bin": j, "left": edges[j],
                "right": edges[j + 1], "logq_center": (edges[j] + edges[j + 1]) / 2,
                "documents": int(denominator[j]), "tokens": int(counts[:, j].sum()),
                "mean": center[j], "lower": bounds[0], "upper": bounds[1],
                "plotted": bool(supported),
            })
    return pd.DataFrame(rows), pd.DataFrame(document_rows)


def theme():
    sns.set_theme(context="paper", style="ticks", font="DejaVu Serif", rc={
        "font.size": 10, "axes.labelsize": 11, "axes.titlesize": 11,
        "xtick.labelsize": 9, "ytick.labelsize": 9, "legend.fontsize": 9,
        "axes.linewidth": 0.8, "lines.linewidth": 1.7,
        "mathtext.fontset": "dejavuserif", "pdf.fonttype": 42,
        "ps.fonttype": 42, "svg.fonttype": "none", "savefig.facecolor": "white",
    })


def plot_density(ax, samples, log_min):
    for name in ("Non-member", "Member"):
        group = samples[samples.membership == name]
        sns.kdeplot(data=group, x="logp", y="logq", weights="weight", ax=ax,
                    color=COLORS[name], linestyles=STYLES[name], linewidths=1.4,
                    levels=[0.05, 0.2, 0.5, 0.8], bw_adjust=1.0, gridsize=160,
                    clip=((log_min, 0), (log_min, 0)), cut=0, fill=False)
    ax.plot([log_min, 0], [log_min, 0], color="0.5", linestyle=":", linewidth=1, zorder=0)
    ax.text(log_min + 0.7, log_min + 0.8, r"$p=q$", color="0.4", fontsize=9,
            rotation=45, rotation_mode="anchor")
    ax.set(xlim=(log_min, 0), ylim=(log_min, 0),
           xlabel=r"Target log-probability, $\log p$", ylabel=r"Draft log-probability, $\log q$")
    ax.set_aspect("equal", adjustable="box")
    ax.xaxis.set_major_locator(MultipleLocator(4))
    ax.yaxis.set_major_locator(MultipleLocator(4))
    ax.legend(handles=[Line2D([], [], color=COLORS[name], ls=STYLES[name], label=name)
                       for name in NAMES.values()], loc="upper left", frameon=False)
    ax.set_title("(a) Joint probability structure", loc="left", pad=12)
    sns.despine(ax=ax)


def plot_acceptance(ax, summary, log_min):
    for name in NAMES.values():
        group = summary[summary.membership == name].sort_values("bin")
        # Separate contiguous supported runs so omitted bins are never bridged.
        supported = group.plotted.to_numpy()
        starts = np.flatnonzero(supported & ~np.r_[False, supported[:-1]])
        ends = np.flatnonzero(supported & ~np.r_[supported[1:], False]) + 1
        for run_index, (start, end) in enumerate(zip(starts, ends)):
            run = group.iloc[start:end]
            sns.lineplot(data=run, x="logq_center", y="mean", estimator=None,
                         errorbar=None, ax=ax, color=COLORS[name], linestyle=STYLES[name],
                         marker="o" if name == "Member" else "s", markersize=4.5,
                         label=name if run_index == 0 else None)
            ax.fill_between(run.logq_center.to_numpy(), run.lower.to_numpy(), run.upper.to_numpy(),
                            color=COLORS[name], alpha=0.16, linewidth=0)
    visible = summary[summary.plotted]
    lower_limit = max(0.0, np.floor((visible.lower.min() - 0.025) / 0.05) * 0.05)
    ax.set(xlim=(log_min, 0), ylim=(lower_limit, 1.01), xlabel=r"Draft log-probability, $\log q$",
           ylabel="Mean theoretical acceptance probability")
    ax.xaxis.set_major_locator(MultipleLocator(4))
    ax.yaxis.set_major_formatter(PercentFormatter(1))
    ax.yaxis.set_major_locator(MultipleLocator(0.05 if lower_limit >= 0.6 else 0.2))
    ax.grid(axis="y", linewidth=0.5, color="0.9")
    ax.legend(loc="lower right", frameon=False)
    ax.set_title("(b) Acceptance conditional on draft probability", loc="left", pad=12)
    ax.text(0.035, 0.045, r"$\alpha=\min(1,p/q)$" + "\n95% document-bootstrap CI",
            transform=ax.transAxes, fontsize=8.5, color="0.35", va="bottom")
    sns.despine(ax=ax)


def save_figure(fig, output, stem):
    for extension in ("pdf", "png", "svg"):
        fig.savefig(output / f"{stem}.{extension}", dpi=400, bbox_inches="tight", pad_inches=0.05)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--benchmark", choices=("wikitection", "newstection"), default="wikitection")
    parser.add_argument("--cache-dir", type=Path)
    parser.add_argument("--partition", type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--seed", type=int, default=20260922)
    parser.add_argument("--tokens-per-record", type=int, default=32)
    parser.add_argument("--bins", type=int, default=12)
    parser.add_argument("--bootstrap", type=int, default=2000)
    parser.add_argument("--min-documents", type=int, default=30)
    parser.add_argument("--log-min", type=float, default=-16)
    args = parser.parse_args()
    condition = f"{args.benchmark}_epoch3"
    args.cache_dir = args.cache_dir or ROOT / f"artifacts/archive/sft_runs/pq_directional/{condition}"
    args.partition = args.partition or ROOT / f"artifacts/archive/sft_runs/full_delta/{condition}/draft_auxiliary_distilled/full_delta_protocol.json"
    args.output_dir = args.output_dir or ROOT / f"artifacts/reports/figures/introduction/qwen3_{condition}_auxiliary"
    if (args.tokens_per_record < 2 or args.bins < 2 or args.bootstrap < 100
            or args.min_documents < 2 or not np.isfinite(args.log_min) or args.log_min >= 0):
        parser.error("Invalid sampling, binning, bootstrap or axis settings")
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    target = args.cache_dir / "target.npz"
    draft = args.cache_dir / "draft_auxiliary_distilled.npz"
    records, source_meta = load_records(target, draft, args.partition, args.benchmark)
    print(f"Loaded {len(records)} diagnostic records with aligned target/draft tokens", flush=True)
    sample_seed, bootstrap_seed = np.random.SeedSequence(args.seed).spawn(2)
    samples = kde_sample(records, args.tokens_per_record, np.random.default_rng(sample_seed))
    edges = np.linspace(args.log_min, 0, args.bins + 1)
    summary, document_summary = conditional_summary(records, edges, args.bootstrap,
                                                    args.min_documents, np.random.default_rng(bootstrap_seed))
    plotted = summary[summary.plotted]
    if (not np.isfinite(plotted[["mean", "lower", "upper"]]).all().all()
            or (plotted.lower > plotted.upper).any()
            or (plotted.lower < 0).any() or (plotted.upper > 1).any()):
        raise ValueError("Invalid bootstrap summary")
    summary.to_csv(output / "conditional_acceptance.csv", index=False)
    document_summary.to_csv(output / "document_bin_statistics.csv", index=False)
    samples.to_csv(output / "density_samples.csv.gz", index=False, compression={"method": "gzip", "mtime": 0})
    diagnostics = {}
    for label, name in NAMES.items():
        group = [r for r in records if r["label"] == label]
        diagnostics[name] = {
            "documents": len(group), "tokens_without_eos": sum(len(r["logp"]) for r in group),
            "density_samples": int((samples.membership == name).sum()),
            "document_weighted_joint_mass_outside_view": float(np.mean([
                np.mean((r["logp"] < args.log_min) | (r["logq"] < args.log_min)) for r in group])),
            "document_weighted_q_mass_outside_bins": float(np.mean([
                np.mean(r["logq"] < args.log_min) for r in group])),
            "document_weighted_mean_alpha": float(np.mean([
                np.exp(np.minimum(0, r["logp"] - r["logq"])).mean() for r in group])),
        }
    inputs = [target, draft, target.with_suffix(".npz.json"), draft.with_suffix(".npz.json"), args.partition]
    report = {
        "source": "Historical Qwen3 teacher-forced caches; not controlled_sft_v2 observations",
        "source_metadata": source_meta,
        "inputs": [{"path": str(p.resolve()), "sha256": sha256(p)} for p in inputs],
        "script_sha256": sha256(__file__),
        "parameters": {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
        "libraries": {"seaborn": sns.__version__, "matplotlib": matplotlib.__version__,
                      "numpy": np.__version__, "pandas": pd.__version__},
        "partition": "Existing D diagnostic partition; no V/C/T records",
        "token_scope": "Original response candidates at original prefixes; appended final EOS excluded",
        "acceptance": "Theoretical min(1,p/q); not observed bits, not a natural SD trajectory",
        "density": "Uniform sampling without replacement per document; per-token weight 1/sample_count",
        "density_levels": [0.05, 0.2, 0.5, 0.8],
        "density_level_scope": "Seaborn iso-proportion contours normalized on its clipped evaluation grid",
        "conditional_estimator": "Mean within each document/bin, then equal-weight mean of contributing documents",
        "uncertainty": "Pointwise percentile 95% CI; documents resampled within each membership class; fixed bins",
        "diagnostics": diagnostics,
        "selected_records": [{"record_id": r["record_id"], "label": r["label"]} for r in records],
    }
    (output / "provenance.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    (output / "README.md").write_text(
        f"# Introduction figures: {args.benchmark}, 3 epochs\n\n"
        "Qwen3-8B-Base target / Qwen3-1.7B-Base auxiliary-distilled draft. "
        "Historical teacher-forced caches, existing D diagnostic partition, appended EOS excluded.\n\n"
        "Panel (a): document-balanced Seaborn KDE of log p versus log q. "
        "Panel (b): document-balanced theoretical min(1, p/q) within shared log q bins; "
        f"pointwise 95% confidence intervals from {args.bootstrap:,} document bootstrap replicates. "
        "Panel (b) does not report measured acceptance frequencies.\n\n"
        "Exports: joint_logprob_density, conditional_acceptance, and intro_membership_insight "
        "(combined); each has PDF, 400-DPI PNG, and SVG versions. "
        "CSV files contain the numerical summaries; provenance.json records input hashes, "
        "selected records, sampling parameters, and software versions.\n\n"
        "From the repository root, using the existing Seaborn-enabled Python environment:\n\n"
        "```bash\nPYTHONNOUSERSITE=1 MPLCONFIGDIR=/tmp/mpl-intro OPENBLAS_NUM_THREADS=2 "
        f"python experiments/figures/intro_membership_insight.py --benchmark {args.benchmark}\n```\n",
        encoding="utf-8",
    )
    theme()
    print("Rendering joint density and conditional acceptance figures with Seaborn", flush=True)
    fig, ax = plt.subplots(figsize=(4.1, 3.85), layout="constrained")
    plot_density(ax, samples, args.log_min)
    save_figure(fig, output, "joint_logprob_density")
    fig, ax = plt.subplots(figsize=(4.8, 3.85), layout="constrained")
    plot_acceptance(ax, summary, args.log_min)
    save_figure(fig, output, "conditional_acceptance")
    fig, axes = plt.subplots(1, 2, figsize=(9.0, 3.8), layout="constrained",
                             gridspec_kw={"width_ratios": [1, 1.12]})
    plot_density(axes[0], samples, args.log_min)
    plot_acceptance(axes[1], summary, args.log_min)
    save_figure(fig, output, "intro_membership_insight")
    print(json.dumps({"output": str(output), "diagnostics": diagnostics}, indent=2), flush=True)


if __name__ == "__main__":
    main()
