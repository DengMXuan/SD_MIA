"""Alternative introduction figures: full-support probability heatmaps and bin estimates."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from matplotlib.ticker import PercentFormatter, MultipleLocator
import numpy as np
import pandas as pd
import seaborn as sns

from intro_membership_insight import (
    ROOT, NAMES, COLORS, load_records, conditional_summary, save_figure, sha256, theme,
)


def histogram(records, bins=40):
    edges = np.linspace(0, 1, bins + 1)
    result = {}
    for label, name in NAMES.items():
        group = [r for r in records if r["label"] == label]
        mass = np.zeros((bins, bins))
        for r in group:
            h, _, _ = np.histogram2d(np.exp(r["logp"]), np.exp(r["logq"]), bins=(edges, edges))
            mass += h / len(r["logp"]) / len(group)
        if not np.isclose(mass.sum(), 1):
            raise ValueError("Probability histogram lost token mass")
        result[name] = mass
    return result


def plot_histograms(axes, cax, matrices):
    positive = np.concatenate([m[m > 0] for m in matrices.values()])
    norm = LogNorm(vmin=positive.min(), vmax=positive.max())
    cmap = sns.color_palette("light:#315D7F", as_cmap=True)
    for index, (name, ax) in enumerate(zip(NAMES.values(), axes)):
        mass = matrices[name]
        sns.heatmap(mass.T, mask=mass.T == 0, cmap=cmap, norm=norm, ax=ax,
                    cbar=index == 1, cbar_ax=cax if index == 1 else None,
                    cbar_kws={"label": "Probability mass per cell"},
                    square=True, linewidths=0, rasterized=True)
        n = len(mass)
        ax.invert_yaxis()
        ticks = np.linspace(0, n, 5)
        labels = ["0", "0.25", "0.50", "0.75", "1"]
        ax.set_xticks(ticks, labels, rotation=0)
        ax.set_yticks(ticks, labels, rotation=0)
        ax.plot([0, n], [0, n], ls="--", lw=0.9, color="0.35")
        ax.set(xlabel=r"Target probability, $p$", ylabel=r"Draft probability, $q$")
        ax.set_title(name, color=COLORS[name], pad=10, fontweight="medium")
        ax.tick_params(length=3, width=0.8, pad=4)
    return norm


def plot_conditionals(ax, summary):
    for name in NAMES.values():
        rows = summary[(summary.membership == name) & summary.plotted]
        ax.errorbar(rows.q_center, rows["mean"],
                    yerr=np.vstack([rows["mean"] - rows.lower, rows.upper - rows["mean"]]),
                    fmt="none", ecolor=COLORS[name], elinewidth=1.1, capsize=2.2)
    sns.scatterplot(data=summary[summary.plotted], x="q_center", y="mean", hue="membership",
                    style="membership", hue_order=list(NAMES.values()), style_order=list(NAMES.values()),
                    palette=COLORS, markers={"Member": "o", "Non-member": "s"},
                    s=33, linewidth=0.6, edgecolor="white", ax=ax, zorder=3)
    lower = max(0, np.floor((summary.loc[summary.plotted, "lower"].min() - 0.01) / .05) * .05)
    ax.set(xlim=(0, 1), ylim=(lower, 1.01), xlabel=r"Draft probability, $q$",
           ylabel="Theoretical acceptance probability")
    ax.xaxis.set_major_locator(MultipleLocator(.2))
    ax.yaxis.set_major_locator(MultipleLocator(.05))
    ax.yaxis.set_major_formatter(PercentFormatter(1, decimals=0))
    ax.grid(axis="y", lw=.5, color="0.9")
    ax.legend(title=None, loc="upper left", frameon=False)
    sns.despine(ax=ax)


def plot_relative_preference(ax, scores):
    for name in NAMES.values():
        sns.kdeplot(data=scores[scores.membership == name], x="mean_log_ratio", ax=ax,
                    color=COLORS[name], fill=True, alpha=.18, linewidth=1.6,
                    linestyle="-" if name == "Member" else "--", bw_adjust=1.0,
                    cut=0, gridsize=256, label=name)
    ax.axvline(0, color=".55", lw=.8, ls=":")
    ax.set(xlabel=r"Mean log probability ratio, $\overline{\Delta}$", ylabel="Density",
           xlim=(min(-.1, scores.mean_log_ratio.min() - .1), scores.mean_log_ratio.max() + .1))
    ax.xaxis.set_major_locator(MultipleLocator(.5))
    ax.legend(frameon=False, title=None, loc="upper right")
    sns.despine(ax=ax)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--benchmark", choices=("wikitection", "newstection"), default="newstection")
    args = parser.parse_args()
    condition = f"{args.benchmark}_epoch3"
    cache = ROOT / f"artifacts/archive/sft_runs/pq_directional/{condition}"
    partition = ROOT / f"artifacts/archive/sft_runs/full_delta/{condition}/draft_auxiliary_distilled/full_delta_protocol.json"
    output = ROOT / f"artifacts/reports/figures/introduction/qwen3_{condition}_auxiliary/alternatives"
    output.mkdir(parents=True, exist_ok=True)
    records, metadata = load_records(cache / "target.npz", cache / "draft_auxiliary_distilled.npz", partition, args.benchmark)
    matrices = histogram(records)
    q_edges = np.linspace(0, 1, 21)
    log_edges = np.r_[-np.inf, np.log(q_edges[1:])]
    summary, document_stats = conditional_summary(records, log_edges, 2000, 30, np.random.default_rng(20260923))
    summary["q_left"] = q_edges[summary.bin]
    summary["q_right"] = q_edges[summary.bin + 1]
    summary["q_center"] = (summary.q_left + summary.q_right) / 2
    summary = summary.drop(columns="logq_center")
    scores = pd.DataFrame([{
        "record_id": r["record_id"], "membership": NAMES[r["label"]],
        "mean_log_ratio": float(np.mean(r["logp"] - r["logq"])),
    } for r in records])
    scores.to_csv(output / "document_mean_log_ratio.csv", index=False)
    summary.to_csv(output / "acceptance_by_probability.csv", index=False)
    document_stats.to_csv(output / "document_bin_statistics.csv", index=False)
    np.savez_compressed(output / "probability_histograms.npz", edges=np.linspace(0, 1, 41),
                        member=matrices["Member"], nonmember=matrices["Non-member"])
    theme()
    fig = plt.figure(figsize=(7.3, 3.25), layout="constrained")
    grid = fig.add_gridspec(1, 3, width_ratios=[1, 1, .045], wspace=.1)
    plot_histograms([fig.add_subplot(grid[0]), fig.add_subplot(grid[1])], fig.add_subplot(grid[2]), matrices)
    save_figure(fig, output, "joint_probability_heatmaps")
    fig, ax = plt.subplots(figsize=(5.7, 3.45), layout="constrained")
    plot_conditionals(ax, summary)
    save_figure(fig, output, "conditional_acceptance_points")
    fig, ax = plt.subplots(figsize=(4.7, 3.45), layout="constrained")
    plot_relative_preference(ax, scores)
    save_figure(fig, output, "relative_preference_distribution")
    fig, axes = plt.subplots(1, 2, figsize=(9.5, 3.55), layout="constrained")
    plot_relative_preference(axes[0], scores)
    plot_conditionals(axes[1], summary)
    axes[0].set_title("(a) Relative target–draft preference", loc="left", pad=10)
    axes[1].set_title("(b) Acceptance at matched draft probabilities", loc="left", pad=10)
    save_figure(fig, output, "recommended_intro_alternative")
    inputs = [cache / "target.npz", cache / "draft_auxiliary_distilled.npz", partition]
    (output / "provenance.json").write_text(json.dumps({
        "benchmark": args.benchmark, "epoch": 3, "source_metadata": metadata,
        "inputs": [{"path": str(p), "sha256": sha256(p)} for p in inputs],
        "code": [{"path": str(p), "sha256": sha256(p)} for p in
                 (Path(__file__), Path(__file__).with_name("intro_membership_insight.py"))],
        "density": "40 x 40 bins on p,q in [0,1]; all response tokens; document-balanced mass; common log color scale; zero cells blank",
        "acceptance": "Theoretical alpha=min(1,p/q), not observed; 20 equal-width raw-q bins; no connecting lines",
        "relative_preference": "One mean(log p - log q) per document, all response tokens; no label-driven feature selection; Seaborn KDE, Scott bandwidth x 1, cut=0",
        "estimator": "Mean within document/bin, then mean over contributing documents",
        "uncertainty": "2000 document bootstrap replicates, pointwise percentile 95% CI",
        "seed": 20260923, "minimum_documents_per_bin": 30,
        "document_counts": {name: sum(r['label'] == label for r in records) for label, name in NAMES.items()},
        "tokens": sum(len(r['logp']) for r in records), "eos": "Appended final EOS removed",
        "selected_records": [{"record_id": r["record_id"], "label": r["label"]} for r in records],
    }, indent=2) + "\n")
    print(json.dumps({"output": str(output), "bins": len(summary),
                      "min_documents": int(summary.documents.min()),
                      "histogram_sums": {k: float(v.sum()) for k, v in matrices.items()}}, indent=2))


if __name__ == "__main__":
    main()
