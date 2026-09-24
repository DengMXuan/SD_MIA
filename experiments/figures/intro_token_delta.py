"""Token-level delta distributions; no within-document aggregation or absolute value."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import numpy as np
import pandas as pd
import seaborn as sns

from intro_membership_insight import ROOT, NAMES, COLORS, load_records, save_figure, sha256, theme


def decorate(ax, limits):
    ax.axvline(0, color=".5", lw=.9, ls=":", zorder=0)
    ax.set(xlim=limits, ylim=(0, None), xlabel=r"Token log probability ratio, $\delta=\log p-\log q$",
           ylabel="Density")
    ax.xaxis.set_major_locator(MaxNLocator(nbins=6, integer=True))
    ax.legend(frameon=False, loc="upper right")
    sns.despine(ax=ax)


def plot_histogram(ax, values, edges, limits):
    for name in NAMES.values():
        # Count every token. Density normalization uses the entire class, so
        # truncating the VIEW does not renormalize the displayed central mass.
        frequency, _ = np.histogram(values[name], bins=edges)
        density = frequency / len(values[name]) / np.diff(edges)
        sns.histplot(x=(edges[:-1] + edges[1:]) / 2, weights=density,
                     bins=edges.tolist(), stat="count", element="step", fill=True,
                     alpha=.16, color=COLORS[name], linewidth=1.2, label=name, ax=ax)
    decorate(ax, limits)


def plot_kde(ax, sample, limits):
    for name in NAMES.values():
        sns.kdeplot(data=sample[sample.membership == name], x="delta", color=COLORS[name],
                    fill=True, alpha=.17, linewidth=1.7,
                    linestyle="-" if name == "Member" else "--",
                    bw_adjust=1, cut=0, gridsize=1024, label=name, ax=ax)
    decorate(ax, limits)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--benchmark", choices=("newstection", "wikitection"), default="newstection")
    parser.add_argument("--kde-tokens-per-class", type=int, default=100000)
    parser.add_argument("--seed", type=int, default=20260923)
    args = parser.parse_args()
    if args.kde_tokens_per_class < 2:
        parser.error("At least two KDE samples are required")
    condition = f"{args.benchmark}_epoch3"
    cache = ROOT / f"artifacts/archive/sft_runs/pq_directional/{condition}"
    partition = ROOT / f"artifacts/archive/sft_runs/full_delta/{condition}/draft_auxiliary_distilled/full_delta_protocol.json"
    output = ROOT / f"artifacts/reports/figures/introduction/qwen3_{condition}_auxiliary/token_delta"
    records, source_meta = load_records(cache / "target.npz", cache / "draft_auxiliary_distilled.npz",
                                        partition, args.benchmark)
    lengths = np.array([len(r["logp"]) for r in records])
    labels = np.array([r["label"] for r in records])
    deltas = np.concatenate([r["logp"] - r["logq"] for r in records])
    token_labels = np.repeat(labels, lengths)
    values = {name: deltas[token_labels == label] for label, name in NAMES.items()}
    low, high = np.quantile(deltas, [.005, .995])
    limits = (float(np.floor(low)), float(np.ceil(high)))
    full_limits = (float(np.floor(deltas.min())), float(np.ceil(deltas.max())))
    edges = np.linspace(*full_limits, int((full_limits[1] - full_limits[0]) / .15) + 1)
    rng = np.random.default_rng(args.seed)
    sample_parts, sample_indices = [], {}
    for label, name in NAMES.items():
        indices = np.flatnonzero(token_labels == label)
        selected = rng.choice(indices, min(args.kde_tokens_per_class, len(indices)), replace=False)
        sample_indices[name] = selected
        sample_parts.append(pd.DataFrame({"delta": deltas[selected], "membership": name}))
    sample = pd.concat(sample_parts, ignore_index=True)
    output.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output / "token_deltas.npz", delta=deltas,
                        lengths=lengths, offsets=np.r_[0, lengths.cumsum()], labels=labels,
                        record_ids=np.array([r["record_id"] for r in records]),
                        kde_member_indices=sample_indices["Member"],
                        kde_nonmember_indices=sample_indices["Non-member"])
    summary = {}
    histogram_rows = []
    for name, delta in values.items():
        count, _ = np.histogram(delta, bins=edges)
        for j in range(len(count)):
            histogram_rows.append({"membership": name, "left": edges[j], "right": edges[j+1],
                                   "count": count[j], "density": count[j] / len(delta) / (edges[j+1] - edges[j])})
        summary[name] = {
            "tokens": len(delta), "kde_samples": len(sample_indices[name]),
            "fraction_below_zero": float(np.mean(delta < 0)),
            "fraction_equal_zero": float(np.mean(delta == 0)),
            "fraction_above_zero": float(np.mean(delta > 0)),
            "fraction_outside_focus": float(np.mean((delta < limits[0]) | (delta > limits[1]))),
            "min": float(delta.min()), "max": float(delta.max()), "median": float(np.median(delta)),
        }
    pd.DataFrame(histogram_rows).to_csv(output / "histogram_bins.csv", index=False)
    theme()
    for stem, kind in (("token_delta_kde", "kde"), ("token_delta_histogram", "histogram")):
        fig, ax = plt.subplots(figsize=(5.4, 3.55), layout="constrained")
        if kind == "kde":
            plot_kde(ax, sample, limits)
        else:
            plot_histogram(ax, values, edges, limits)
        save_figure(fig, output, stem)
    fig, axes = plt.subplots(1, 2, figsize=(9.5, 3.55), layout="constrained")
    plot_histogram(axes[0], values, edges, limits)
    plot_kde(axes[1], sample, limits)
    axes[0].set_title("(a) Token-level histogram", loc="left", pad=10)
    axes[1].set_title("(b) Token-level density", loc="left", pad=10)
    save_figure(fig, output, "token_delta_comparison")
    fig, ax = plt.subplots(figsize=(6.2, 3.55), layout="constrained")
    plot_kde(ax, sample, full_limits)
    save_figure(fig, output, "token_delta_kde_full_range")
    inputs = [cache / "target.npz", cache / "draft_auxiliary_distilled.npz", partition]
    (output / "provenance.json").write_text(json.dumps({
        "source_metadata": source_meta,
        "inputs": [{"path": str(p), "sha256": sha256(p)} for p in inputs],
        "script_sha256": sha256(__file__),
        "loader_sha256": sha256(Path(__file__).with_name("intro_membership_insight.py")),
        "parameters": vars(args), "summary": summary,
        "statistical_unit": "One token: signed log p - log q, no absolute value, no document-level averaging",
        "scope": "Existing D diagnostic subset, 800 member and 800 nonmember records; original response tokens; appended EOS excluded",
        "weighting": "Every token equally weighted; each class density normalized separately",
        "histogram": "All tokens, shared bins, normalization by complete class token count and bin width",
        "kde": "Uniform token subsample per class without replacement; Scott bandwidth x 1; Seaborn KDE; cut=0; no focus-range renormalization",
        "focus_limits": limits, "focus_rule": "Round pooled token 0.5% and 99.5% quantiles outward to integers",
        "full_limits": full_limits,
        "libraries": {"numpy": np.__version__, "seaborn": sns.__version__, "matplotlib": matplotlib.__version__},
    }, indent=2) + "\n")
    (output / "README.md").write_text(
        "# Token-level delta distributions\n\n"
        f"Qwen3 / {args.benchmark} / 3 epochs / auxiliary-distilled draft. "
        "Same historical D diagnostic subset: 800 member and 800 nonmember records. "
        "Each statistical sample is one token's signed delta=log(p)-log(q). "
        "There is no absolute value, within-document averaging, or pooling of document scores. "
        "All original response tokens are retained except the appended EOS. "
        "The previous scatter's [-8,0] filter and 600-token sample are not used here.\n\n"
        f"Histogram: {len(deltas):,} real tokens, normalized separately by class. "
        f"KDE: uniform sample of up to {args.kde_tokens_per_class:,} tokens per class, without replacement. "
        "Tokens are equally weighted; longer documents contribute more tokens. "
        "KDE is a smoothed estimate, including smoothing of the small exact-zero mass.\n\n"
        f"Main display limits {limits} use pooled 0.5%/99.5% token quantiles rounded outward; "
        "outside-view fractions are reported in provenance.json. The viewport is not used to select fitting data or renormalize densities. "
        "The full-range KDE is also exported.\n\n"
        "p/q=exp(delta), so delta=0 corresponds to p/q=1; delta>0 to p>q. "
        "The density axis is density per unit delta, not density per unit p/q.\n\n"
        "Each plot has PDF, 400-DPI PNG and SVG exports; token_deltas.npz preserves exact token values and document offsets, "
        "and histogram_bins.csv preserves counts and normalized densities.\n\n"
        "```bash\nPYTHONNOUSERSITE=1 MPLCONFIGDIR=/tmp/mpl-intro OPENBLAS_NUM_THREADS=2 "
        f"python experiments/figures/intro_token_delta.py --benchmark {args.benchmark}\n```\n",
        encoding="utf-8")
    print(json.dumps({"output": str(output), "summary": summary, "limits": limits}, indent=2))


if __name__ == "__main__":
    main()
