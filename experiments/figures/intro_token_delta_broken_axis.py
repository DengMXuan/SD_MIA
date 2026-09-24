"""Reframe the existing token KDE with x in [-2,2] and a broken density axis."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import MultipleLocator, FuncFormatter
import numpy as np
import pandas as pd
import seaborn as sns

from intro_membership_insight import ROOT, NAMES, COLORS, save_figure, sha256, theme


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--benchmark", choices=("newstection", "wikitection"), default="newstection")
    args = parser.parse_args()
    source_dir = ROOT / f"artifacts/reports/figures/introduction/qwen3_{args.benchmark}_epoch3_auxiliary/token_delta"
    source = source_dir / "token_deltas.npz"
    output = source_dir / "focused"
    with np.load(source, allow_pickle=False) as archive:
        samples = {
            "Member": archive["delta"][archive["kde_member_indices"]],
            "Non-member": archive["delta"][archive["kde_nonmember_indices"]],
        }
    theme()
    curves = {}
    # Fit on the EXACT previous sample and full support, with the original KDE
    # parameters. Axis changes neither select observations nor renormalize KDEs.
    scratch, axis = plt.subplots()
    for name, sample in samples.items():
        sns.kdeplot(x=sample, bw_adjust=1, cut=0, gridsize=1024,
                    fill=False, color=COLORS[name], ax=axis)
        x, y = axis.lines[-1].get_data()
        curves[name] = (x.copy(), y.copy())
    plt.close(scratch)
    if any(not np.isfinite(y).all() or y.max() > 1.1 for _, y in curves.values()):
        raise ValueError("Requested upper density band does not contain the KDE peak")
    fig, (top, bottom) = plt.subplots(2, 1, sharex=True, figsize=(5.5, 4.65),
                                    gridspec_kw={"height_ratios": [1, 1]})
    fig.subplots_adjust(left=.145, right=.975, bottom=.145, top=.97, hspace=.13)
    for ax, limits in ((top, (.9, 1.1)), (bottom, (0, .2))):
        for name in NAMES.values():
            x, y = curves[name]
            sns.lineplot(x=x, y=y, estimator=None, errorbar=None, ax=ax,
                         color=COLORS[name], linewidth=1.7,
                         linestyle="-" if name == "Member" else "--")
            ax.fill_between(x, 0, y, color=COLORS[name], alpha=.17, linewidth=0)
        ax.axvline(0, color=".5", lw=.9, ls=":", zorder=0)
        ax.set(xlim=(-2, 2), ylim=limits)
        ax.yaxis.set_major_locator(MultipleLocator(.05))
        ax.yaxis.set_major_formatter(FuncFormatter(lambda value, _pos: f"{value:.2f}"))
        ax.xaxis.set_major_locator(MultipleLocator(.5))
        sns.despine(ax=ax)
    top.spines["bottom"].set_visible(False)
    top.tick_params(axis="x", which="both", bottom=False, labelbottom=False)
    # Broken-axis slashes on the sole visible vertical spine.
    slash = .012
    top.plot((-slash, slash), (-slash, slash), transform=top.transAxes,
             color=".25", lw=1, clip_on=False)
    bottom.plot((-slash, slash), (1-slash, 1+slash), transform=bottom.transAxes,
                color=".25", lw=1, clip_on=False)
    top.legend(handles=[Line2D([], [], color=COLORS[name], lw=1.7,
                              ls="-" if name == "Member" else "--", label=name)
                        for name in NAMES.values()], loc="upper right", frameon=False)
    bottom.set_xlabel(r"Token log probability ratio, $\delta=\log p-\log q$", labelpad=8)
    fig.supylabel("Density", x=.025, fontsize=11)
    gap_y = (top.get_position().y0 + bottom.get_position().y1) / 2
    fig.text(.97, gap_y, "0.20–0.90 omitted", ha="right", va="center", fontsize=8, color=".45")
    output.mkdir(parents=True, exist_ok=True)
    save_figure(fig, output, "token_delta_kde_broken_axis")
    pd.concat([pd.DataFrame({"membership": name, "delta": x, "density": y})
               for name, (x, y) in curves.items()], ignore_index=True).to_csv(output / "kde_curves.csv", index=False)
    report = {
        "source": {"path": str(source), "sha256": sha256(source)},
        "source_provenance": {"path": str(source_dir / "provenance.json"),
                              "sha256": sha256(source_dir / "provenance.json")},
        "script_sha256": sha256(__file__),
        "benchmark": args.benchmark, "epoch": 3, "draft": "auxiliary-distilled",
        "statistical_unit": "Token; signed delta=log p-log q; no document averaging",
        "x_limits": [-2, 2], "y_bands": [[0, .2], [.9, 1.1]], "omitted_y_interval": [.2, .9],
        "equal_vertical_scale_for_both_bands": True,
        "kde": "Identical saved samples; Scott bandwidth x 1; cut=0; gridsize=1024; full-support fitting",
        "view_only_change": True, "resampling": False, "renormalization": False,
        "sample_counts": {name: len(sample) for name, sample in samples.items()},
        "peaks": {name: float(y.max()) for name, (_, y) in curves.items()},
        "seaborn": sns.__version__, "matplotlib": matplotlib.__version__,
    }
    (output / "provenance.json").write_text(json.dumps(report, indent=2) + "\n")
    (output / "README.md").write_text(
        "# Token KDE with broken density axis\n\n"
        "Only the former panel B is shown. The x axis spans [-2,2]. "
        "The density axis displays [0,0.2] and [0.9,1.1] with equal physical heights and equal vertical scales. "
        "The omitted (0.2,0.9) band is explicitly marked with break slashes and a label.\n\n"
        "All sample indices and KDE settings are reused from the preceding token-level plot. "
        "No samples are removed before fitting, and densities are not renormalized to [-2,2]. "
        "Each sample is one token's signed log(p)-log(q), without document aggregation.\n\n"
        "PDF, 400-DPI PNG and SVG exports contain the same figure. "
        "kde_curves.csv preserves full-support curve values; provenance.json records sources and display settings.\n\n"
        "```bash\nPYTHONNOUSERSITE=1 MPLCONFIGDIR=/tmp/mpl-intro OPENBLAS_NUM_THREADS=2 "
        f"python experiments/figures/intro_token_delta_broken_axis.py --benchmark {args.benchmark}\n```\n",
        encoding="utf-8")
    print(json.dumps({"output": str(output), "peaks": report["peaks"]}, indent=2))


if __name__ == "__main__":
    main()
