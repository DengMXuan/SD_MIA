"""E1: dissect where exact target/draft delta carries membership signal.

This is a CPU-only analysis of the already materialized teacher-forced cache.
It does not train an attack or select a statistic on member performance.  The
registered statistics are evaluated twice: ``M_diag`` versus ``N_ref`` is a
mechanism diagnostic, while the unchanged statistic is also reported on the
exploratory ``T_member``/``T_nonmember`` split.  ``N_cal`` is used only for
conformal operating points.

Example::

    uv run python -m experiments.sd_membership_sft.token_signal_anatomy \
      --benchmark wikitection --epoch 1
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from .directional_mia import conformal_tail_pvalues
from .full_delta_mia import rank_auc, split_indices
from .stat_delta_mia import DeltaData, load_delta_data, sliding_means


ROOT = Path(__file__).resolve().parents[2]
BENCHMARKS = ("wikitection", "newstection", "arxivtection")
EPOCHS = (1, 3)
TOP_FRACTIONS = (0.01, 0.02, 0.05, 0.10, 0.20, 0.50, 1.00)
WINDOWS = (4, 8, 16, 32, 64)
RATES = (0.01, 0.05, 0.10)
SIGNALS = ("positive", "negative", "absolute")


@dataclass(frozen=True)
class Roles:
    n_ref: np.ndarray
    m_diag: np.ndarray
    n_cal: np.ndarray
    t_member: np.ndarray
    t_nonmember: np.ndarray
    reserved_member: np.ndarray


def _jsonable(value: Any) -> Any:
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_jsonable(item) for item in value]
    return value


def _write_json_atomic(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(_jsonable(value), handle, indent=2, ensure_ascii=False)
            handle.write("\n")
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def build_roles(labels: np.ndarray, split_seed: int) -> Roles:
    """Build exactly the roles registered in section 4.3 of the protocol."""
    labels = np.asarray(labels, dtype=np.int64)
    partitions = split_indices(labels, split_seed)

    def select(partition: str, label: int) -> np.ndarray:
        indices = partitions[partition]
        return indices[labels[indices] == label]

    roles = Roles(
        n_ref=select("D", 0),
        m_diag=select("D", 1),
        n_cal=select("C", 0),
        t_member=np.sort(np.r_[select("V", 1), select("T", 1)]),
        t_nonmember=np.sort(np.r_[select("V", 0), select("T", 0)]),
        reserved_member=select("C", 1),
    )
    expected = {
        "n_ref": 800,
        "m_diag": 800,
        "n_cal": 400,
        "t_member": 800,
        "t_nonmember": 800,
        "reserved_member": 400,
    }
    for name, count in expected.items():
        if len(getattr(roles, name)) != count:
            raise ValueError(f"role {name} has {len(getattr(roles, name))} records, expected {count}")
    all_indices = np.concatenate([getattr(roles, name) for name in expected])
    if len(np.unique(all_indices)) != len(all_indices):
        raise ValueError("registered roles overlap")
    return roles


def drop_final_cached_token(data: DeltaData) -> DeltaData:
    """Remove the producer-appended final token without changing fragment scope."""
    if np.any(data.lengths <= 1):
        raise ValueError("cannot drop the final token from a one-token record")
    pieces = [
        data.delta[int(start) : int(end) - 1]
        for start, end in zip(data.offsets[:-1], data.offsets[1:])
    ]
    lengths = data.lengths - 1
    offsets = np.r_[0, np.cumsum(lengths, dtype=np.int64)]
    return DeltaData(
        labels=data.labels,
        record_ids=data.record_ids,
        lengths=lengths,
        offsets=offsets,
        delta=np.concatenate(pieces).astype(np.float32, copy=False),
    )


def _record_values(data: DeltaData, index: int) -> np.ndarray:
    start, end = int(data.offsets[index]), int(data.offsets[index + 1])
    return np.asarray(data.delta[start:end], dtype=np.float64)


def signal_values(delta: np.ndarray, signal: str) -> np.ndarray:
    delta = np.asarray(delta, dtype=np.float64)
    if signal == "signed":
        return delta
    if signal == "positive":
        return np.maximum(delta, 0.0)
    if signal == "negative":
        return np.maximum(-delta, 0.0)
    if signal == "absolute":
        return np.abs(delta)
    if signal == "positive_indicator":
        return (delta > 0.0).astype(np.float64)
    raise ValueError(f"unknown signal {signal!r}")


def direction_scores(data: DeltaData) -> dict[str, np.ndarray]:
    result = {
        "mean_signed_delta": np.empty(len(data.labels)),
        "mean_positive_delta": np.empty(len(data.labels)),
        "mean_negative_delta": np.empty(len(data.labels)),
        "mean_abs_delta": np.empty(len(data.labels)),
        "positive_delta_fraction": np.empty(len(data.labels)),
    }
    mapping = {
        "mean_signed_delta": "signed",
        "mean_positive_delta": "positive",
        "mean_negative_delta": "negative",
        "mean_abs_delta": "absolute",
        "positive_delta_fraction": "positive_indicator",
    }
    for index in range(len(data.labels)):
        delta = _record_values(data, index)
        for name, signal in mapping.items():
            result[name][index] = float(np.mean(signal_values(delta, signal)))
    return result


def top_fraction_scores(
    data: DeltaData,
    fractions: tuple[float, ...] = TOP_FRACTIONS,
    random_repeats: int = 5,
    seed: int = 20260914,
) -> dict[str, np.ndarray]:
    """Return fixed keep/drop/random curves for three directional evidences.

    Ranking and aggregation use the same non-negative evidence.  Random keeps
    are averaged over registered deterministic draws to make the control less
    dependent on one lucky subset.  No labels are read by this function.
    """
    if random_repeats <= 0:
        raise ValueError("random_repeats must be positive")
    if any(not 0.0 < fraction <= 1.0 for fraction in fractions):
        raise ValueError("fractions must lie in (0, 1]")
    output: dict[str, np.ndarray] = {}
    for signal in SIGNALS:
        for fraction in fractions:
            tag = f"{round(100 * fraction):02d}pct"
            output[f"top_keep_{signal}_{tag}"] = np.empty(len(data.labels))
            output[f"random_keep_{signal}_{tag}"] = np.empty(len(data.labels))
            if fraction < 1.0:
                output[f"top_drop_{signal}_{tag}"] = np.empty(len(data.labels))

    for index in range(len(data.labels)):
        delta = _record_values(data, index)
        for signal_index, signal in enumerate(SIGNALS):
            evidence = signal_values(delta, signal)
            order = np.argsort(-evidence, kind="stable")
            counts = [
                min(len(evidence), max(1, int(math.ceil(fraction * len(evidence)))))
                for fraction in fractions
            ]
            # One random permutation supplies a uniform subset for every
            # registered size through its prefixes.  Reusing prefixes is both
            # unbiased for each size and far cheaper than drawing 21 separate
            # subsets per record.
            random_sums = np.zeros(len(fractions), dtype=np.float64)
            for repeat in range(random_repeats):
                rng = np.random.default_rng(
                    np.random.SeedSequence([seed, index, signal_index, repeat])
                )
                permutation = rng.permutation(len(evidence))
                prefix = np.cumsum(evidence[permutation], dtype=np.float64)
                random_sums += np.asarray(
                    [prefix[count - 1] / count for count in counts], dtype=np.float64
                )
            for fraction_index, fraction in enumerate(fractions):
                count = counts[fraction_index]
                tag = f"{round(100 * fraction):02d}pct"
                output[f"top_keep_{signal}_{tag}"][index] = float(np.mean(evidence[order[:count]]))
                if count < len(evidence):
                    output[f"top_drop_{signal}_{tag}"][index] = float(np.mean(evidence[order[count:]]))
                output[f"random_keep_{signal}_{tag}"][index] = random_sums[fraction_index] / random_repeats
    return output


def window_scores(
    data: DeltaData,
    windows: tuple[int, ...] = WINDOWS,
    shuffle_repeats: int = 5,
    seed: int = 20260915,
) -> dict[str, np.ndarray]:
    """Return strongest-window and order-destruction controls."""
    if shuffle_repeats <= 0 or any(width <= 0 for width in windows):
        raise ValueError("window widths and shuffle_repeats must be positive")
    output: dict[str, np.ndarray] = {}
    for signal in ("signed",) + SIGNALS:
        for width in windows:
            output[f"window_max_{signal}_w{width}"] = np.empty(len(data.labels))
            output[f"window_max_{signal}_w{width}_shuffled"] = np.empty(len(data.labels))
    for width in windows:
        output[f"window_positive_rate_w{width}"] = np.empty(len(data.labels))
        output[f"window_positive_rate_w{width}_shuffled"] = np.empty(len(data.labels))

    for index in range(len(data.labels)):
        delta = _record_values(data, index)
        shuffled = []
        for repeat in range(shuffle_repeats):
            values = delta.copy()
            np.random.default_rng(np.random.SeedSequence([seed, index, repeat])).shuffle(values)
            shuffled.append(values)
        for signal in ("signed",) + SIGNALS:
            values = signal_values(delta, signal)
            shuffled_values = [signal_values(values_, signal) for values_ in shuffled]
            for width in windows:
                output[f"window_max_{signal}_w{width}"][index] = float(np.max(sliding_means(values, width)))
                output[f"window_max_{signal}_w{width}_shuffled"][index] = float(
                    np.mean([np.max(sliding_means(values_, width)) for values_ in shuffled_values])
                )
        for width in windows:
            output[f"window_positive_rate_w{width}"][index] = float(
                np.mean(sliding_means(delta, width) > 0.0)
            )
            output[f"window_positive_rate_w{width}_shuffled"][index] = float(
                np.mean([np.mean(sliding_means(values, width) > 0.0) for values in shuffled])
            )
    return output


def _point_metrics(member: np.ndarray, nonmember: np.ndarray) -> dict[str, float]:
    labels = np.r_[np.ones(len(member), dtype=np.int64), np.zeros(len(nonmember), dtype=np.int64)]
    values = np.r_[member, nonmember]
    return {
        "auc": rank_auc(member, nonmember),
        "pauc_0_10": fast_partial_auc(values, labels, max_fpr=0.10),
    }


def fast_partial_auc(scores: np.ndarray, labels: np.ndarray, max_fpr: float = 0.10) -> float:
    """Vectorized equivalent of the repository's tie-aware partial AUC."""
    scores = np.asarray(scores, dtype=np.float64)
    labels = np.asarray(labels, dtype=np.int64)
    if scores.ndim != 1 or labels.shape != scores.shape or not 0.0 < max_fpr <= 1.0:
        raise ValueError("invalid scores, labels, or max_fpr")
    positives = int(np.sum(labels == 1))
    negatives = int(np.sum(labels == 0))
    if positives == 0 or negatives == 0:
        raise ValueError("partial AUC requires both classes")
    order = np.argsort(-scores, kind="stable")
    ordered_scores = scores[order]
    ordered_labels = labels[order]
    ends = np.flatnonzero(np.r_[ordered_scores[1:] != ordered_scores[:-1], True])
    cumulative_positive = np.cumsum(ordered_labels == 1)[ends]
    cumulative_negative = np.cumsum(ordered_labels == 0)[ends]
    fpr = np.r_[0.0, cumulative_negative / negatives]
    tpr = np.r_[0.0, cumulative_positive / positives]
    below = fpr < max_fpr
    x = fpr[below]
    y = tpr[below]
    if fpr[-1] >= max_fpr:
        right = int(np.searchsorted(fpr, max_fpr, side="left"))
        if fpr[right] == max_fpr:
            y_at = tpr[right]
        else:
            left = right - 1
            fraction = (max_fpr - fpr[left]) / (fpr[right] - fpr[left])
            y_at = tpr[left] + fraction * (tpr[right] - tpr[left])
        x = np.r_[x, max_fpr]
        y = np.r_[y, y_at]
    return float(np.trapezoid(y, x) / max_fpr)


def _interval(point: float, samples: np.ndarray) -> dict[str, float]:
    return {
        "point": float(point),
        "ci95_low": float(np.quantile(samples, 0.025)),
        "ci95_high": float(np.quantile(samples, 0.975)),
    }


def evaluate_score(
    scores: np.ndarray,
    roles: Roles,
    repeats: int,
    seed: int,
) -> dict[str, Any]:
    """Evaluate with record-level bootstrap; tokens are never resampled."""
    if repeats <= 0:
        raise ValueError("bootstrap repeats must be positive")
    scores = np.asarray(scores, dtype=np.float64)
    diagnostic = _point_metrics(scores[roles.m_diag], scores[roles.n_ref])
    test = _point_metrics(scores[roles.t_member], scores[roles.t_nonmember])
    rng = np.random.default_rng(seed)
    diagnostic_samples = {"auc": np.empty(repeats), "pauc_0_10": np.empty(repeats)}
    test_samples = {"auc": np.empty(repeats), "pauc_0_10": np.empty(repeats)}
    rate_samples = {rate: {"tpr": np.empty(repeats), "fpr": np.empty(repeats)} for rate in RATES}

    def operating(member: np.ndarray, nonmember: np.ndarray, calibration: np.ndarray, rate: float) -> tuple[float, float]:
        member_p = conformal_tail_pvalues(member, calibration)
        nonmember_p = conformal_tail_pvalues(nonmember, calibration)
        return float(np.mean(member_p <= rate)), float(np.mean(nonmember_p <= rate))

    point_rates = {
        rate: operating(scores[roles.t_member], scores[roles.t_nonmember], scores[roles.n_cal], rate)
        for rate in RATES
    }
    for repeat in range(repeats):
        md = roles.m_diag[rng.integers(0, len(roles.m_diag), len(roles.m_diag))]
        nr = roles.n_ref[rng.integers(0, len(roles.n_ref), len(roles.n_ref))]
        tm = roles.t_member[rng.integers(0, len(roles.t_member), len(roles.t_member))]
        tn = roles.t_nonmember[rng.integers(0, len(roles.t_nonmember), len(roles.t_nonmember))]
        nc = roles.n_cal[rng.integers(0, len(roles.n_cal), len(roles.n_cal))]
        for metric, value in _point_metrics(scores[md], scores[nr]).items():
            diagnostic_samples[metric][repeat] = value
        for metric, value in _point_metrics(scores[tm], scores[tn]).items():
            test_samples[metric][repeat] = value
        for rate in RATES:
            tpr, fpr = operating(scores[tm], scores[tn], scores[nc], rate)
            rate_samples[rate]["tpr"][repeat] = tpr
            rate_samples[rate]["fpr"][repeat] = fpr
    return {
        "diagnostic_M_diag_vs_N_ref": {
            metric: _interval(value, diagnostic_samples[metric])
            for metric, value in diagnostic.items()
        },
        "exploratory_T": {
            **{metric: _interval(value, test_samples[metric]) for metric, value in test.items()},
            "tpr_at_fpr": {
                f"{int(rate * 100)}%": {
                    "tpr": _interval(point_rates[rate][0], rate_samples[rate]["tpr"]),
                    "actual_fpr": _interval(point_rates[rate][1], rate_samples[rate]["fpr"]),
                    "calibration_n_nonmember": len(roles.n_cal),
                }
                for rate in RATES
            },
        },
    }


def paired_pauc_delta(
    scores: np.ndarray,
    baseline: np.ndarray,
    roles: Roles,
    repeats: int,
    seed: int,
) -> dict[str, dict[str, float]]:
    """Paired record bootstrap for the pAUC difference of two scores."""
    rng = np.random.default_rng(seed)

    def difference(member: np.ndarray, nonmember: np.ndarray) -> float:
        labels = np.r_[np.ones(len(member), dtype=np.int64), np.zeros(len(nonmember), dtype=np.int64)]
        return fast_partial_auc(np.r_[scores[member], scores[nonmember]], labels) - fast_partial_auc(
            np.r_[baseline[member], baseline[nonmember]], labels
        )

    result: dict[str, dict[str, float]] = {}
    for name, member, nonmember in (
        ("diagnostic_M_diag_vs_N_ref", roles.m_diag, roles.n_ref),
        ("exploratory_T", roles.t_member, roles.t_nonmember),
    ):
        samples = np.empty(repeats)
        for repeat in range(repeats):
            m = member[rng.integers(0, len(member), len(member))]
            n = nonmember[rng.integers(0, len(nonmember), len(nonmember))]
            samples[repeat] = difference(m, n)
        result[name] = _interval(difference(member, nonmember), samples)
    return result


def matching_global_baseline(method: str) -> str:
    """Return the like-for-like full-fragment mean for a sparse statistic."""
    if method.startswith("window_positive_rate_"):
        return "positive_delta_fraction"
    for signal, baseline in (
        ("positive", "mean_positive_delta"),
        ("negative", "mean_negative_delta"),
        ("absolute", "mean_abs_delta"),
        ("signed", "mean_signed_delta"),
    ):
        if f"_{signal}_" in method:
            return baseline
    return "mean_abs_delta"


def load_paired_logps(path: Path, expected: DeltaData, drop_final: bool) -> tuple[np.ndarray, np.ndarray]:
    with np.load(path, allow_pickle=False) as archive:
        for key in ("lengths", "target", "draft_auxiliary_distilled"):
            if key not in archive.files:
                raise ValueError(f"{path} is missing {key}")
        source_lengths = np.asarray(archive["lengths"], dtype=np.int64)
        target = np.asarray(archive["target"], dtype=np.float32)
        draft = np.asarray(archive["draft_auxiliary_distilled"], dtype=np.float32)
    if len(source_lengths) != len(expected.labels):
        raise ValueError("paired p/q cache record count is not aligned")
    if drop_final:
        pieces_target, pieces_draft = [], []
        offsets = np.r_[0, np.cumsum(source_lengths, dtype=np.int64)]
        for start, end in zip(offsets[:-1], offsets[1:]):
            pieces_target.append(target[int(start) : int(end) - 1])
            pieces_draft.append(draft[int(start) : int(end) - 1])
        target, draft = np.concatenate(pieces_target), np.concatenate(pieces_draft)
        source_lengths = source_lengths - 1
    if not np.array_equal(source_lengths, expected.lengths):
        raise ValueError("paired p/q cache lengths are not aligned")
    if len(target) != len(expected.delta) or len(draft) != len(target):
        raise ValueError("paired p/q token arrays are not aligned")
    if not np.allclose(target - draft, expected.delta, atol=2e-6, rtol=1e-6):
        raise ValueError("paired p/q cache disagrees with delta cache")
    return target, draft


def q_bin_analysis(
    data: DeltaData,
    draft: np.ndarray,
    roles: Roles,
    bins: int = 10,
    repeats: int = 200,
    seed: int = 20260916,
) -> dict[str, Any]:
    """Compare per-record delta within q bins, with record-level CIs."""
    if bins < 2:
        raise ValueError("bins must be at least two")
    ref_q = np.concatenate([
        draft[int(data.offsets[index]) : int(data.offsets[index + 1])]
        for index in roles.n_ref
    ])
    edges = np.unique(np.quantile(ref_q, np.linspace(0.0, 1.0, bins + 1)))
    if len(edges) < 3:
        raise ValueError("draft log-probabilities do not support at least two distinct q bins")
    # Keep JSON bounds finite while including every observed endpoint.
    edges[0] = np.nextafter(edges[0], -np.inf)
    edges[-1] = np.nextafter(edges[-1], np.inf)
    actual_bins = len(edges) - 1
    per_record = np.full((len(data.labels), actual_bins), np.nan)
    for index in range(len(data.labels)):
        start, end = int(data.offsets[index]), int(data.offsets[index + 1])
        q = draft[start:end]
        delta = data.delta[start:end]
        assignment = np.searchsorted(edges[1:-1], q, side="right")
        for bin_index in range(actual_bins):
            chosen = delta[assignment == bin_index]
            if len(chosen):
                per_record[index, bin_index] = float(np.mean(chosen))
    rng = np.random.default_rng(seed)
    rows = []
    for bin_index in range(actual_bins):
        member = per_record[roles.m_diag, bin_index]
        nonmember = per_record[roles.n_ref, bin_index]
        member, nonmember = member[np.isfinite(member)], nonmember[np.isfinite(nonmember)]
        difference = float(np.mean(member) - np.mean(nonmember))
        samples = np.empty(repeats)
        for repeat in range(repeats):
            m = member[rng.integers(0, len(member), len(member))]
            n = nonmember[rng.integers(0, len(nonmember), len(nonmember))]
            samples[repeat] = float(np.mean(m) - np.mean(n))
        rows.append({
            "bin": bin_index,
            "logq_lower": float(edges[bin_index]),
            "logq_upper": float(edges[bin_index + 1]),
            "member_records": len(member),
            "nonmember_records": len(nonmember),
            "mean_delta_member_minus_nonmember": _interval(difference, samples),
        })
    return {"edge_source": "N_ref tokens only", "bins": rows, "per_record": per_record}


def _role_manifest(data: DeltaData, roles: Roles, split_seed: int) -> dict[str, Any]:
    explanations = {
        "n_ref": "D nonmembers; reference/diagnostic only",
        "m_diag": "D members; E1 diagnostic/oracle only",
        "n_cal": "C nonmembers; conformal thresholds only",
        "t_member": "V+T members; exploratory evaluation only",
        "t_nonmember": "V+T nonmembers; exploratory evaluation only",
        "reserved_member": "C members; deliberately unused",
    }
    return {
        "split_seed": split_seed,
        "split_unit": "record",
        "roles": {
            name: {
                "description": explanations[name],
                "count": len(getattr(roles, name)),
                "record_ids": [str(data.record_ids[index]) for index in getattr(roles, name)],
            }
            for name in explanations
        },
    }


def _plot_outputs(
    output_dir: Path,
    report: dict[str, Any],
    data: DeltaData,
    roles: Roles,
    target: np.ndarray | None,
    draft: np.ndarray | None,
    q_bins: dict[str, Any] | None,
) -> list[str]:
    os.environ.setdefault("MPLCONFIGDIR", str(output_dir / ".matplotlib_cache"))
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    files: list[str] = []
    methods = report["methods"]

    direction_names = [
        "mean_signed_delta", "mean_positive_delta", "mean_negative_delta",
        "mean_abs_delta", "positive_delta_fraction",
    ]
    fig, ax = plt.subplots(figsize=(9, 4.5))
    x = np.arange(len(direction_names))
    ax.bar(x, [methods[name]["exploratory_T"]["pauc_0_10"]["point"] for name in direction_names])
    ax.set_xticks(x, [name.replace("_delta", "").replace("_", "\n") for name in direction_names])
    ax.set_ylabel("Exploratory T pAUC@0.10")
    ax.set_title("Exact-delta direction ablation")
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    path = output_dir / "direction_pauc.png"
    fig.savefig(path, dpi=180)
    plt.close(fig)
    files.append(path.name)

    fig, axes = plt.subplots(1, 3, figsize=(14, 4.2), sharey=True)
    for ax, signal in zip(axes, SIGNALS):
        percentages, keep, random, drop = [], [], [], []
        for fraction in TOP_FRACTIONS:
            tag = f"{round(100 * fraction):02d}pct"
            percentages.append(100 * fraction)
            keep.append(methods[f"top_keep_{signal}_{tag}"]["exploratory_T"]["pauc_0_10"]["point"])
            random.append(methods[f"random_keep_{signal}_{tag}"]["exploratory_T"]["pauc_0_10"]["point"])
            if fraction < 1.0:
                drop.append(methods[f"top_drop_{signal}_{tag}"]["exploratory_T"]["pauc_0_10"]["point"])
        ax.plot(percentages, keep, marker="o", label="top keep")
        ax.plot(percentages, random, marker="o", label="random keep")
        ax.plot(percentages[:-1], drop, marker="o", label="top drop")
        baseline = matching_global_baseline(f"top_keep_{signal}_01pct")
        ax.axhline(methods[baseline]["exploratory_T"]["pauc_0_10"]["point"], color="black", ls="--", lw=1, label=baseline)
        ax.set_xscale("log")
        ax.set_title(signal)
        ax.set_xlabel("token fraction (%)")
        ax.grid(alpha=0.25)
    axes[0].set_ylabel("Exploratory T pAUC@0.10")
    axes[-1].legend(fontsize=8)
    fig.tight_layout()
    path = output_dir / "top_fraction_curves.png"
    fig.savefig(path, dpi=180)
    plt.close(fig)
    files.append(path.name)

    fig, axes = plt.subplots(1, 3, figsize=(14, 4.2), sharey=True)
    for ax, signal in zip(axes, SIGNALS):
        original = [methods[f"window_max_{signal}_w{width}"]["exploratory_T"]["pauc_0_10"]["point"] for width in WINDOWS]
        shuffled = [methods[f"window_max_{signal}_w{width}_shuffled"]["exploratory_T"]["pauc_0_10"]["point"] for width in WINDOWS]
        ax.plot(WINDOWS, original, marker="o", label="original order")
        ax.plot(WINDOWS, shuffled, marker="o", label="shuffled")
        ax.set_title(signal)
        ax.set_xlabel("window width")
        ax.grid(alpha=0.25)
    axes[0].set_ylabel("Exploratory T pAUC@0.10")
    axes[-1].legend(fontsize=8)
    fig.tight_layout()
    path = output_dir / "window_contiguity.png"
    fig.savefig(path, dpi=180)
    plt.close(fig)
    files.append(path.name)

    if target is not None and draft is not None:
        rng = np.random.default_rng(20260914)
        sampled: dict[str, np.ndarray] = {}
        for role_name, indices in (("member", roles.m_diag), ("nonmember", roles.n_ref)):
            positions = np.concatenate([
                np.arange(int(data.offsets[index]), int(data.offsets[index + 1]))
                for index in indices
            ])
            if len(positions) > 100_000:
                positions = rng.choice(positions, size=100_000, replace=False)
            sampled[role_name] = positions
        fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
        for role_name, color in (("nonmember", "tab:blue"), ("member", "tab:red")):
            pos = sampled[role_name]
            axes[0].scatter(draft[pos], target[pos], s=2, alpha=0.08, c=color, label=role_name, rasterized=True)
            axes[1].scatter(draft[pos], data.delta[pos], s=2, alpha=0.08, c=color, label=role_name, rasterized=True)
        axes[0].set(xlabel="log q", ylabel="log p", title="Paired target/draft log-probability")
        axes[1].set(xlabel="log q", ylabel="delta = log p - log q", title="Delta conditional on draft log-probability")
        axes[0].legend(markerscale=4)
        all_q = np.r_[draft[sampled["member"]], draft[sampled["nonmember"]]]
        all_d = np.r_[data.delta[sampled["member"]], data.delta[sampled["nonmember"]]]
        q_range = np.quantile(all_q, (0.005, 0.995))
        d_range = np.quantile(all_d, (0.005, 0.995))
        hist = {}
        for role_name in ("member", "nonmember"):
            pos = sampled[role_name]
            hist[role_name], xedges, yedges = np.histogram2d(draft[pos], data.delta[pos], bins=60, range=(q_range, d_range), density=True)
        difference = hist["member"] - hist["nonmember"]
        limit = float(np.quantile(np.abs(difference), 0.99)) or 1.0
        image = axes[2].imshow(difference.T, origin="lower", aspect="auto", extent=(xedges[0], xedges[-1], yedges[0], yedges[-1]), cmap="coolwarm", vmin=-limit, vmax=limit)
        axes[2].set(xlabel="log q", ylabel="delta", title="Member - nonmember density")
        fig.colorbar(image, ax=axes[2], fraction=0.046)
        fig.tight_layout()
        path = output_dir / "paired_logp_logq.png"
        fig.savefig(path, dpi=180)
        plt.close(fig)
        files.append(path.name)

    if q_bins is not None:
        rows = q_bins["bins"]
        points = [row["mean_delta_member_minus_nonmember"]["point"] for row in rows]
        low = [row["mean_delta_member_minus_nonmember"]["ci95_low"] for row in rows]
        high = [row["mean_delta_member_minus_nonmember"]["ci95_high"] for row in rows]
        fig, ax = plt.subplots(figsize=(8, 4.5))
        x = np.arange(len(rows))
        lower_error = np.maximum(0.0, np.asarray(points) - np.asarray(low))
        upper_error = np.maximum(0.0, np.asarray(high) - np.asarray(points))
        ax.errorbar(x, points, yerr=[lower_error, upper_error], marker="o", capsize=3)
        ax.axhline(0.0, color="black", lw=1)
        ax.set(xlabel="N_ref log-q decile", ylabel="mean delta: member - nonmember", title="Conditional delta separation by draft-probability bin")
        ax.grid(alpha=0.25)
        fig.tight_layout()
        path = output_dir / "q_bin_delta.png"
        fig.savefig(path, dpi=180)
        plt.close(fig)
        files.append(path.name)
    return files


def _write_markdown(path: Path, report: dict[str, Any]) -> None:
    methods = report["methods"]
    lines = [
        f"# E1 Token Signal Anatomy: {report['protocol']['condition']}",
        "",
        "> This is a diagnostic/oracle analysis of exact delta. T is exploratory; no statistic was selected on T.",
        "",
        "## Data contract",
        "",
        f"- Native fragment policy: {report['protocol']['fragment_policy']}",
        f"- EOS policy: {report['protocol']['eos_policy']}",
        f"- Length range after EOS policy: {report['protocol']['length_range']}",
        "- Roles: N_ref=800, M_diag=800, N_cal=400, T_member=800, T_nonmember=800, reserved C member=400.",
        "",
        "## Direction ablation",
        "",
        "| Statistic | M_diag/N_ref pAUC | Exploratory T pAUC | T AUC |",
        "|---|---:|---:|---:|",
    ]
    for name in ("mean_signed_delta", "mean_positive_delta", "mean_negative_delta", "mean_abs_delta", "positive_delta_fraction"):
        row = methods[name]
        lines.append(
            f"| `{name}` | {row['diagnostic_M_diag_vs_N_ref']['pauc_0_10']['point']:.4f} | "
            f"{row['exploratory_T']['pauc_0_10']['point']:.4f} | {row['exploratory_T']['auc']['point']:.4f} |"
        )
    lines.extend(["", "## Exploratory sparse/window ranking", "", "> This table is descriptive only; Gate 1 is evaluated jointly across the four short-fragment conditions, not by selecting the best row on this condition's T split.", "", "| Statistic | Exploratory T pAUC | Matching global mean | Paired pAUC delta |", "|---|---:|---|---:|"])
    candidates = [name for name in methods if name.startswith("top_keep_") or (name.startswith("window_max_") and not name.endswith("_shuffled"))]
    candidates.sort(key=lambda name: methods[name]["exploratory_T"]["pauc_0_10"]["point"], reverse=True)
    for name in candidates[:15]:
        lines.append(
            f"| `{name}` | {methods[name]['exploratory_T']['pauc_0_10']['point']:.4f} | "
            f"`{report['comparison_baseline'][name]}` | "
            f"{report['paired_delta_vs_matching_global'][name]['exploratory_T']['point']:+.4f} |"
        )
    lines.extend(["", "## Interpretation constraints", "", "- Member labels are used only in M_diag mechanism plots, paired uncertainty estimates, and exploratory final evaluation.", "- N_cal is used only for conformal thresholds; its members are unused.", "- Entropy, rank, margin, token type, and repetition features are not present in the current cache and are explicitly deferred to draft-only extraction.", "- ArXiv records retain their full cached 1024–2048-token scope; this program never truncates them.", ""])
    path.write_text("\n".join(lines), encoding="utf-8")


def run_condition(
    benchmark: str,
    epoch: int,
    input_path: Path,
    pq_path: Path | None,
    output_dir: Path,
    split_seed: int = 20260824,
    bootstrap_repeats: int = 1000,
    random_repeats: int = 5,
    include_final_token: bool = False,
) -> dict[str, Any]:
    original = load_delta_data(input_path)
    data = original if include_final_token else drop_final_cached_token(original)
    roles = build_roles(data.labels, split_seed)
    score_map = direction_scores(data)
    score_map.update(top_fraction_scores(data, random_repeats=random_repeats))
    score_map.update(window_scores(data, shuffle_repeats=random_repeats))

    target = draft = None
    q_bins_public = None
    q_bins_private = None
    if pq_path is not None and pq_path.exists():
        target, draft = load_paired_logps(pq_path, data, drop_final=not include_final_token)
        q_bins_private = q_bin_analysis(data, draft, roles, repeats=bootstrap_repeats)
        q_bins_public = {key: value for key, value in q_bins_private.items() if key != "per_record"}

    methods = {}
    paired = {}
    comparison_baseline = {}
    for method_index, (name, scores) in enumerate(score_map.items()):
        methods[name] = evaluate_score(scores, roles, bootstrap_repeats, split_seed + 1000 + method_index)
        baseline_name = matching_global_baseline(name)
        comparison_baseline[name] = baseline_name
        paired[name] = paired_pauc_delta(
            scores,
            score_map[baseline_name],
            roles,
            bootstrap_repeats,
            split_seed + 5000 + method_index,
        )

    role_manifest = _role_manifest(data, roles, split_seed)
    output_dir.mkdir(parents=True, exist_ok=True)
    _write_json_atomic(output_dir / "role_manifest.json", role_manifest)
    report: dict[str, Any] = {
        "experiment": "E1 exact-delta token signal anatomy",
        "protocol": {
            "condition": f"{benchmark}_epoch{epoch}",
            "benchmark": benchmark,
            "epoch": epoch,
            "input": str(input_path.resolve()),
            "input_sha256": _sha256(input_path),
            "paired_pq_input": str(pq_path.resolve()) if pq_path is not None and pq_path.exists() else None,
            "paired_pq_sha256": _sha256(pq_path) if pq_path is not None and pq_path.exists() else None,
            "fragment_policy": "one complete native benchmark record; never truncate or split",
            "eos_policy": "include final cached token" if include_final_token else "drop producer-appended final token",
            "length_range": [int(np.min(data.lengths)), int(np.max(data.lengths))],
            "records": len(data.labels),
            "tokens": len(data.delta),
            "split_seed": split_seed,
            "bootstrap_unit": "record",
            "bootstrap_repeats": bootstrap_repeats,
            "top_fractions": list(TOP_FRACTIONS),
            "windows": list(WINDOWS),
            "random_control_repeats": random_repeats,
            "selection_policy": "all preregistered statistics reported; no member-based method selection",
            "test_status": "exploratory because V/T were inspected in prior work",
            "unavailable_cache_features": ["draft_entropy", "candidate_rank", "top1_top2_margin", "token_id/type", "repetition"],
        },
        "role_counts": {name: len(getattr(roles, name)) for name in Roles.__dataclass_fields__},
        "methods": methods,
        "comparison_baseline": comparison_baseline,
        "paired_delta_vs_matching_global": paired,
        "q_bin_analysis": q_bins_public,
    }
    report["figures"] = _plot_outputs(output_dir, report, data, roles, target, draft, q_bins_private)
    _write_json_atomic(output_dir / "E1_REPORT.json", report)
    _write_markdown(output_dir / "E1_REPORT.md", report)
    archive = {"labels": data.labels, "record_ids": data.record_ids, "lengths": data.lengths}
    archive.update({name: values.astype(np.float32) for name, values in score_map.items()})
    np.savez_compressed(output_dir / "scores.npz", **archive)
    return report


def _default_input(benchmark: str, epoch: int) -> Path:
    return ROOT / "experiments/results/sft_runs/full_delta" / f"{benchmark}_epoch{epoch}" / "draft_auxiliary_distilled/full_delta.npz"


def _default_pq(benchmark: str, epoch: int) -> Path:
    return ROOT / "experiments/results/sft_runs/pq_directional" / f"{benchmark}_epoch{epoch}" / "pq_gap_token_logps.npz"


def _default_output(benchmark: str, epoch: int) -> Path:
    return ROOT / "experiments/results/sft_runs/accept_only_active_v2" / "e1_token_anatomy" / f"{benchmark}_epoch{epoch}"


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--benchmark", choices=BENCHMARKS, required=True)
    parser.add_argument("--epoch", choices=EPOCHS, type=int, required=True)
    parser.add_argument("--input", type=Path)
    parser.add_argument("--pq-input", type=Path)
    parser.add_argument("--no-pq", action="store_true", help="Skip paired p/q and q-bin plots.")
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--split-seed", type=int, default=20260824)
    parser.add_argument("--bootstrap-repeats", type=int, default=1000)
    parser.add_argument("--random-repeats", type=int, default=5)
    parser.add_argument("--include-final-token", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    input_path = (args.input or _default_input(args.benchmark, args.epoch)).resolve()
    pq_path = None if args.no_pq else (args.pq_input or _default_pq(args.benchmark, args.epoch)).resolve()
    output_dir = (args.output_dir or _default_output(args.benchmark, args.epoch)).resolve()
    report = run_condition(
        benchmark=args.benchmark,
        epoch=args.epoch,
        input_path=input_path,
        pq_path=pq_path,
        output_dir=output_dir,
        split_seed=args.split_seed,
        bootstrap_repeats=args.bootstrap_repeats,
        random_repeats=args.random_repeats,
        include_final_token=args.include_final_token,
    )
    print(json.dumps({
        "condition": report["protocol"]["condition"],
        "output": str(output_dir),
        "records": report["protocol"]["records"],
        "tokens": report["protocol"]["tokens"],
        "figures": report["figures"],
    }, indent=2))


if __name__ == "__main__":
    main()
