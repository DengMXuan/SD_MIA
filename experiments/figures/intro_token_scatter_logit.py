"""Render the frozen token scatter sample in logit coordinates, without resampling."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import MultipleLocator
import numpy as np
import pandas as pd
import seaborn as sns

from intro_membership_insight import ROOT, NAMES, COLORS, save_figure, sha256, theme


def stable_logit(log_probability, epsilon):
    """Use log1mexp for stability, applying the same upper probability cap to p/q."""
    ceiling = np.log1p(-epsilon)
    clipped = log_probability > ceiling
    bounded = np.minimum(log_probability, ceiling)
    return bounded - np.log(-np.expm1(bounded)), clipped


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--benchmark", choices=("wikitection", "newstection"), default="newstection")
    parser.add_argument("--epsilon", type=float, default=1e-6)
    args = parser.parse_args()
    if not 0 < args.epsilon < .01:
        parser.error("epsilon must lie in (0, 0.01)")
    base = ROOT / f"artifacts/reports/figures/introduction/qwen3_{args.benchmark}_epoch3_auxiliary/token_scatter"
    source = base / "revised/sampled_tokens.csv"
    source_manifest = base / "revised/provenance.json"
    output = base / "logit"
    points = pd.read_csv(source)
    if (not np.isfinite(points[["logp", "logq"]]).all().all()
            or (points[["logp", "logq"]] > 0).any().any()
            or not points.membership.isin(NAMES.values()).all()
            or points.duplicated(["record_id", "token_position"]).any()):
        raise ValueError("Invalid frozen sample")
    if points.membership.value_counts().nunique() != 1:
        raise ValueError("Classes must have equal sample sizes")
    for suffix in ("p", "q"):
        points[f"logit_{suffix}"], points[f"capped_{suffix}"] = stable_logit(points[f"log{suffix}"].to_numpy(), args.epsilon)
    values = points[["logit_p", "logit_q"]].to_numpy()
    if not np.isfinite(values).all():
        raise ValueError("Nonfinite logit coordinates")
    if not np.array_equal(np.sign(points.logp - points.logq), np.sign(points.logit_p - points.logit_q)):
        raise ValueError("Chosen probability cap changed the side of p=q for a token")
    lower = float(np.floor((values.min() - .6) / 2) * 2)
    upper = float(np.ceil((values.max() + .6) / 2) * 2)
    theme()
    fig, ax = plt.subplots(figsize=(5.1, 4.8), layout="constrained")
    sns.scatterplot(data=points, x="logit_p", y="logit_q", hue="membership",
                    hue_order=list(NAMES.values()), palette=COLORS,
                    s=24, alpha=1.0, linewidth=0, legend=False, ax=ax)
    ax.plot([lower, upper], [lower, upper], color=".35", ls="--", lw=1.0)
    span = upper - lower
    ax.text(lower + .84 * span, lower + .91 * span, r"$p=q$", color=".35", fontsize=10, rotation=45)
    ax.text(.16, .59, r"$p<q$", transform=ax.transAxes, color=".4", fontsize=11, ha="center")
    ax.text(.74, .16, r"$p>q$", transform=ax.transAxes, color=".4", fontsize=11, ha="center")
    ax.legend(handles=[Line2D([], [], ls="none", marker="o", markersize=4.5,
                             color=COLORS[name], label=name) for name in NAMES.values()],
              loc="upper left", frameon=False)
    ax.set(xlim=(lower, upper), ylim=(lower, upper),
           xlabel=r"Target log-odds, $\log\frac{p}{1-p}$",
           ylabel=r"Draft log-odds, $\log\frac{q}{1-q}$")
    ax.set_aspect("equal", adjustable="box")
    ax.xaxis.set_major_locator(MultipleLocator(4))
    ax.yaxis.set_major_locator(MultipleLocator(4))
    sns.despine(ax=ax)
    output.mkdir(parents=True, exist_ok=True)
    save_figure(fig, output, "token_logit_scatter")
    points.to_csv(output / "sampled_tokens_logit.csv", index=False)
    metadata = {
        "condition": f"Qwen3 / {args.benchmark} / epoch3 / auxiliary-distilled draft",
        "source_sample": {"path": str(source), "sha256": sha256(source)},
        "source_provenance": {"path": str(source_manifest), "sha256": sha256(source_manifest)},
        "code": [{"path": str(p), "sha256": sha256(p)} for p in
                 (Path(__file__), Path(__file__).with_name("intro_membership_insight.py"))],
        "statistical_unit": "One original response token; same rows and drawing order as previous scatter",
        "resampled": False, "original_sample_filter": "Both original log probabilities in [-8,0]",
        "coordinates": "log(probability/(1-probability)); computed stably from cached log probability",
        "upper_probability_cap": 1 - args.epsilon, "epsilon": args.epsilon,
        "capped_p": int(points.capped_p.sum()), "capped_q": int(points.capped_q.sum()),
        "capped_tokens_either_axis": int((points.capped_p | points.capped_q).sum()),
        "cap_changed_diagonal_side": False,
        "counts": points.membership.value_counts().to_dict(),
        "axis_limits": [lower, upper], "marker_area_points_squared": 24, "marker_alpha": 1.0,
        "seaborn": sns.__version__, "matplotlib": matplotlib.__version__, "numpy": np.__version__,
    }
    (output / "provenance.json").write_text(json.dumps(metadata, indent=2) + "\n")
    (output / "README.md").write_text(
        "# Token-level logit scatter\n\n"
        "Same 300 member and 300 non-member tokens and drawing order as the preceding NewsTection scatter "
        "(or the benchmark selected at execution). No resampling, aggregation or jitter. "
        "The original sample was restricted to log p,log q in [-8,0]; the logit transformation does not restore excluded tokens.\n\n"
        "Axes show log[p/(1-p)] and log[q/(1-q)], with identical scales. The p=q diagonal and p<q/p>q regions are preserved. "
        "Orange and blue markers remain fully opaque and 24 pt^2. All 600 points are within the exported axes.\n\n"
        f"Probabilities above 1-{args.epsilon:g} are capped at that value on both axes. "
        f"This affects {metadata['capped_p']} target values and {metadata['capped_q']} draft values "
        f"across {metadata['capped_tokens_either_axis']} tokens. The cap preserves the original diagonal side for every point. "
        "Capped positions are display conventions, not recovered higher-precision probabilities. "
        "Original logs, transformed coordinates and cap flags are saved in sampled_tokens_logit.csv.\n\n"
        "```bash\nPYTHONNOUSERSITE=1 MPLCONFIGDIR=/tmp/mpl-intro OPENBLAS_NUM_THREADS=2 "
        f"python experiments/figures/intro_token_scatter_logit.py --benchmark {args.benchmark}\n```\n",
        encoding="utf-8",
    )
    print(json.dumps({"output": str(output), "counts": metadata["counts"],
                      "capped_p": metadata["capped_p"], "capped_q": metadata["capped_q"],
                      "axis_limits": metadata["axis_limits"]}, indent=2))


if __name__ == "__main__":
    main()
