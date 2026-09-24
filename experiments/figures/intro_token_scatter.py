"""Token-level paired log p/log q scatter with document-balanced sampling."""
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

from intro_membership_insight import ROOT, NAMES, COLORS, load_records, save_figure, sha256, theme


def render(points, lower, output, stem):
    fig, ax = plt.subplots(figsize=(5.1, 4.8), layout="constrained")
    # A single collection in shuffled order avoids consistently putting one
    # membership class above the other. Every plotted point is an actual token.
    sns.scatterplot(data=points, x="logp", y="logq", hue="membership",
                    hue_order=list(NAMES.values()), palette=COLORS,
                    s=24, alpha=1.0, linewidth=0, legend=False, ax=ax)
    ax.plot([lower, 0], [lower, 0], color=".35", ls="--", lw=1.0)
    ax.text(lower * .74, lower * .79, r"$p=q$", color=".35", fontsize=10, rotation=45)
    ax.text(.16, .59, r"$p<q$", transform=ax.transAxes,
            color=".4", fontsize=11, ha="center")
    ax.text(.74, .16, r"$p>q$", transform=ax.transAxes,
            color=".4", fontsize=11, ha="center")
    ax.legend(handles=[Line2D([], [], ls="none", marker="o", markersize=4.5,
                             color=COLORS[name], label=name) for name in NAMES.values()],
              loc="upper left", frameon=False)
    ax.set(xlim=(lower, 0), ylim=(lower, 0),
           xlabel=r"Target log-probability, $\log p$",
           ylabel=r"Draft log-probability, $\log q$")
    ax.set_aspect("equal", adjustable="box")
    ax.xaxis.set_major_locator(MultipleLocator(2))
    ax.yaxis.set_major_locator(MultipleLocator(2))
    sns.despine(ax=ax)
    save_figure(fig, output, stem)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--benchmark", choices=("wikitection", "newstection"), default="newstection")
    parser.add_argument("--tokens-per-record", type=int, default=1)
    parser.add_argument("--points-per-class", type=int, default=300)
    parser.add_argument("--seed", type=int, default=20260923)
    args = parser.parse_args()
    if args.tokens_per_record < 1 or args.points_per_class < 1:
        parser.error("Sampling counts must be positive")
    condition = f"{args.benchmark}_epoch3"
    cache = ROOT / f"artifacts/archive/sft_runs/pq_directional/{condition}"
    partition = ROOT / f"artifacts/archive/sft_runs/full_delta/{condition}/draft_auxiliary_distilled/full_delta_protocol.json"
    output = ROOT / f"artifacts/reports/figures/introduction/qwen3_{condition}_auxiliary/token_scatter/revised"
    output.mkdir(parents=True, exist_ok=True)
    records, source_meta = load_records(cache / "target.npz", cache / "draft_auxiliary_distilled.npz",
                                        partition, args.benchmark)
    rng = np.random.default_rng(args.seed)
    rows = []
    for record in records:
        if len(record["logp"]) < args.tokens_per_record:
            raise ValueError("A record is too short for equal per-record sampling")
        positions = rng.choice(len(record["logp"]), args.tokens_per_record, replace=False)
        for position in positions:
            rows.append({"record_id": record["record_id"], "token_position": int(position),
                         "membership": NAMES[record["label"]],
                         "logp": float(record["logp"][position]), "logq": float(record["logq"][position])})
    candidates = pd.DataFrame(rows)
    in_range = candidates.logp.between(-8, 0) & candidates.logq.between(-8, 0)
    visible = candidates[in_range]
    selected = []
    for name in NAMES.values():
        group = visible[visible.membership == name]
        if len(group) < args.points_per_class:
            raise ValueError(f"Not enough in-range token candidates for {name}")
        selected.append(group.iloc[rng.choice(len(group), args.points_per_class, replace=False)])
    points = pd.concat(selected, ignore_index=True)
    points = points.iloc[rng.permutation(len(points))].reset_index(drop=True)
    if points.membership.value_counts().nunique() != 1:
        raise ValueError("Membership classes must have equal sampled token counts")
    points.to_csv(output / "sampled_tokens.csv", index=False)
    counts = {}
    for name in NAMES.values():
        group = points[points.membership == name]
        counts[name] = {"documents": group.record_id.nunique(), "tokens": len(group),
                        "candidate_tokens_before_range_filter": int((candidates.membership == name).sum()),
                        "candidate_tokens_in_range": int((visible.membership == name).sum())}
    theme()
    render(points, -8, output, "token_logprob_scatter")
    sources = [cache / "target.npz", cache / "draft_auxiliary_distilled.npz", partition,
               cache / "target.npz.json", cache / "draft_auxiliary_distilled.npz.json"]
    manifest = {
        "source_metadata": source_meta, "parameters": vars(args), "counts": counts,
        "statistical_unit": "One paired target/draft probability at one original response-token position",
        "sampling": "Uniform tokens per document, then filter both log probabilities to [-8,0], then uniformly subsample the same count per class; shuffled drawing order",
        "partition": "Frozen D diagnostic subset, 800 member and 800 nonmember documents",
        "token_scope": "Original response candidates under original prefixes; appended EOS removed; no aggregation or jitter",
        "axis_limits": [-8, 0], "marker_area_points_squared": 24, "marker_alpha": 1.0,
        "inputs": [{"path": str(path), "sha256": sha256(path)} for path in sources],
        "code": [{"path": str(path), "sha256": sha256(path)} for path in
                 (Path(__file__), Path(__file__).with_name("intro_membership_insight.py"))],
        "seaborn": sns.__version__, "numpy": np.__version__, "matplotlib": matplotlib.__version__,
    }
    (output / "provenance.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n")
    (output / "README.md").write_text(
        f"# Token-level scatter: Qwen3 / {args.benchmark} / 3 epochs / auxiliary-distilled draft\n\n"
        f"Each dot is a real paired (log p, log q) for a single token. Uniformly sample {args.tokens_per_record} "
        "distinct response tokens per diagnostic record, retain candidates with both log probabilities "
        f"in [-8,0], and uniformly subsample {args.points_per_class} candidates per class; no record-level averaging. "
        "The drawing order is shuffled so neither class is systematically painted last. "
        "The appended final EOS is excluded.\n\n"
        "The dashed diagonal is p=q; the two sides are labeled p<q and p>q only. "
        "Colors indicate membership of the containing text, not independent token-level training labels.\n\n"
        "Both axes span exactly [-8,0]. All exported sample points are in this range. "
        "Markers retain the original orange/blue colors, are fully opaque, and have area 24 pt^2. "
        "All exports include PDF, 400-DPI PNG and SVG. sampled_tokens.csv preserves exact values and positions.\n\n"
        "Historical teacher-forced caches; not natural SD trajectories or a new controlled_sft_v2 audit.\n\n"
        "```bash\nPYTHONNOUSERSITE=1 MPLCONFIGDIR=/tmp/mpl-intro OPENBLAS_NUM_THREADS=2 "
        f"python experiments/figures/intro_token_scatter.py --benchmark {args.benchmark}\n```\n",
        encoding="utf-8",
    )
    print(json.dumps({"output": str(output), "counts": counts}, indent=2))


if __name__ == "__main__":
    main()
