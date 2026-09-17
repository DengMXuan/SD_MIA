"""Nonmember-only conditional count likelihoods for accept-only replay.

The detector sees local q sequences and verifier bits, never target p/delta.
A neural finite mixture of binomials predicts the complete count distribution;
its atom at acceptance probability one represents saturation without claiming
to recover p. Fixed positive exponential tilts and a two-state span prior
produce directional scores. These are model-based scores, NOT e-values:
held-out nonmember conformal calibration handles model misspecification.

With a paired observation archive, compare original-only B queries against
B/2 original plus B/2 truncated-context queries on exactly the same tokens.
No member or synthetic-member examples train or select either predictor.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from scipy.special import logsumexp
from torch import nn
from torch.nn import functional as F

from .audit_metrics import membership_metrics
from .audit_runtime import (
    ROOT, SPLIT_SEED, _deterministic_subset, _paths, _write_json, split_indices,
)
from .lowq_baseline import lowq_fragment_scores, standardized_max


@dataclass(frozen=True)
class Observations:
    """Observable-only boundary. Axes: token, view, repeat (bits only)."""

    logq: np.ndarray
    bits: np.ndarray
    lengths: np.ndarray
    view_names: tuple[str, ...] = ("original",)

    def __post_init__(self) -> None:
        if self.lengths.ndim != 1 or not len(self.lengths):
            raise ValueError("lengths must be a nonempty vector")
        if not np.issubdtype(self.lengths.dtype, np.integer) or np.any(self.lengths <= 0):
            raise ValueError("lengths must be positive integers")
        if self.logq.ndim != 2 or self.logq.shape != (int(self.lengths.sum()), len(self.view_names)):
            raise ValueError("q/view/length alignment failed")
        if not np.all(np.isfinite(self.logq)) or np.any(self.logq > 1e-6):
            raise ValueError("q must contain finite log probabilities")
        if self.bits.ndim != 3 or self.bits.shape[:2] != self.logq.shape or self.bits.shape[2] < 1:
            raise ValueError("bits must have token/view/repeat axes")
        if not np.all((self.bits == 0) | (self.bits == 1)):
            raise ValueError("only binary verifier outcomes are permitted")
        if not self.view_names or self.view_names[0] != "original" or len(set(self.view_names)) != len(self.view_names):
            raise ValueError("views must be unique and begin with original")

    @property
    def offsets(self) -> np.ndarray:
        return np.r_[0, np.cumsum(self.lengths)]


def save_observations(
    path: Path, observations: Observations, labels: np.ndarray, record_ids: np.ndarray,
) -> None:
    validate_records(observations, labels, record_ids)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path, logq=observations.logq, bits=observations.bits, lengths=observations.lengths,
        view_names=np.asarray(observations.view_names), labels=labels, record_ids=record_ids,
    )


def validate_records(obs: Observations, labels: np.ndarray, record_ids: np.ndarray) -> None:
    if labels.shape != obs.lengths.shape or record_ids.shape != labels.shape:
        raise ValueError("record metadata is misaligned")
    if not np.all((labels == 0) | (labels == 1)) or len(np.unique(record_ids)) != len(labels):
        raise ValueError("binary labels and unique record IDs are required")


def load_observations(path: Path) -> tuple[Observations, np.ndarray, np.ndarray]:
    with np.load(path, allow_pickle=False) as data:
        forbidden = {"logp", "p", "delta", "target", "target_logp"}.intersection(data.files)
        if forbidden:
            raise ValueError(f"target values are forbidden in an observation archive: {forbidden}")
        obs = Observations(data["logq"], data["bits"], data["lengths"], tuple(data["view_names"].tolist()))
        labels, ids = data["labels"], data["record_ids"]
    validate_records(obs, labels, ids)
    return obs, labels, ids


def replay_observations(benchmark: str, epoch: int, budget: int, seed: int):
    """Simulator boundary: exact target values are discarded before fitting."""
    from .replay_cache import (load_replay_data)
    from .audit_runtime import (_record_uniforms)

    full, pq = _paths(benchmark, epoch)
    data = load_replay_data(full, pq)
    bits = np.empty((len(data.logq0), 1, budget), dtype=np.uint8)
    for record, (start, end) in enumerate(zip(data.offsets[:-1], data.offsets[1:])):
        alpha = np.exp(np.minimum(0.0, data.logp[start:end] - data.logq0[start:end]))
        bits[start:end, 0] = _record_uniforms(seed, record, end - start, budget) < alpha[:, None]
    return Observations(data.logq0[:, None], bits, data.lengths), data.labels, data.record_ids


def record_partitions(labels: np.ndarray, record_ids: np.ndarray) -> dict[str, np.ndarray]:
    """Reuse the registered split; reference=400 NM, calibration=200 NM."""
    parts = split_indices(labels, SPLIT_SEED)
    reference = _deterministic_subset(parts["D"][labels[parts["D"]] == 0], 400, SPLIT_SEED + 400)
    calibration = _deterministic_subset(parts["C"][labels[parts["C"]] == 0], 200, SPLIT_SEED + 1200)
    shuffled = np.random.default_rng(SPLIT_SEED).permutation(reference)
    result = {"train": np.sort(shuffled[:320]), "validation": np.sort(shuffled[320:]),
              "reference": reference, "calibration": calibration, "test": parts["T"]}
    assert_partition_contract(labels, record_ids, result)
    return result


def assert_partition_contract(labels: np.ndarray, ids: np.ndarray, parts: dict[str, np.ndarray]) -> None:
    used: set[str] = set()
    for name in ("train", "validation", "calibration", "test"):
        indices = parts[name]
        if not len(indices) or np.any(indices < 0) or np.any(indices >= len(labels)):
            raise ValueError(f"invalid {name} partition")
        if name != "test" and np.any(labels[indices] != 0):
            raise ValueError(f"member labels are forbidden in {name}")
        local = set(ids[indices].tolist())
        if len(local) != len(indices) or local.intersection(used):
            raise ValueError("record partitions overlap")
        used.update(local)


def observable_inputs(obs: Observations, budget: int, paired: bool) -> tuple[np.ndarray, np.ndarray, int]:
    if budget <= 0 or budget > obs.bits.shape[2]:
        raise ValueError("budget exceeds available original-query bits")
    if paired and (budget % 2 or obs.logq.shape[1] != 2):
        raise ValueError("paired mode needs an even budget and exactly two views")
    k = budget // 2 if paired else budget
    position = np.concatenate([np.linspace(0.0, 1.0, int(n)) for n in obs.lengths])
    channels = [obs.logq[:, 0], position]
    if paired:
        channels.extend([obs.logq[:, 1], obs.bits[:, 1, :k].mean(axis=1)])
    # Original outcomes are never predictor inputs, including neighboring ones.
    return np.column_stack(channels).astype(np.float32), obs.bits[:, 0, :k].sum(axis=1), k


class ConditionalCountTCN(nn.Module):
    """Predict count PMFs from q context, optionally paired-view feedback."""

    def __init__(self, input_dim: int, k: int, channels: int = 24) -> None:
        super().__init__()
        if k <= 0 or input_dim <= 0 or channels <= 0:
            raise ValueError("positive model dimensions required")
        self.k = k
        self.projection = nn.Linear(input_dim, channels)
        self.convs = nn.ModuleList([nn.Conv1d(channels, channels, 3, padding=d, dilation=d) for d in (1, 2, 4, 8)])
        self.norms = nn.ModuleList([nn.LayerNorm(channels) for _ in self.convs])
        grid = torch.cat((torch.sigmoid(torch.linspace(-7.0, 7.0, 17)), torch.ones(1))).double()
        count = torch.arange(k + 1).double()[:, None]
        rejects = k - count
        log_kernel = (torch.lgamma(torch.tensor(float(k + 1), dtype=torch.float64)) - torch.lgamma(count + 1)
                      - torch.lgamma(rejects + 1) + count * torch.log(grid)[None, :]
                      + torch.where(rejects == 0, 0.0, rejects * torch.log1p(-grid)[None, :]))
        self.register_buffer("log_kernel", log_kernel.float())
        self.head = nn.Linear(channels, len(grid))

    def mixture_log_weights(self, features: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        valid = mask.unsqueeze(-1).to(features.dtype)
        hidden = F.gelu(self.projection(features * valid)) * valid
        for conv, norm in zip(self.convs, self.norms):
            update = conv(hidden.transpose(1, 2)).transpose(1, 2)
            hidden = (hidden + F.gelu(norm(update))) * valid
        return F.log_softmax(self.head(hidden), dim=-1)

    def forward(self, features: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        weights = self.mixture_log_weights(features, mask)
        pmf = torch.logsumexp(weights.unsqueeze(-2) + self.log_kernel, dim=-1)
        return pmf


def count_nll(logpmf: torch.Tensor, counts: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    losses = -logpmf.gather(-1, counts.long().unsqueeze(-1)).squeeze(-1)
    # Equal record weights; long ArXiv records cannot dominate the objective.
    return ((losses * mask).sum(dim=1) / mask.sum(dim=1)).mean()


def make_batch(features, counts, offsets, indices, device):
    width = int(max(offsets[i + 1] - offsets[i] for i in indices))
    x = np.zeros((len(indices), width, features.shape[1]), dtype=np.float32)
    y = np.zeros((len(indices), width), dtype=np.int64)
    mask = np.zeros((len(indices), width), dtype=bool)
    for row, index in enumerate(indices):
        start, end = offsets[index:index + 2]
        x[row, :end - start] = features[start:end]
        y[row, :end - start] = counts[start:end]
        mask[row, :end - start] = True
    return tuple(torch.as_tensor(array, device=device) for array in (x, y, mask))


@dataclass
class NullFit:
    model: ConditionalCountTCN
    mean: np.ndarray
    scale: np.ndarray
    history: list[dict[str, float]]
    best_epoch: int


def fit_null(
    obs: Observations, train: np.ndarray, validation: np.ndarray, *, budget: int,
    paired: bool, seed: int, device: torch.device, epochs: int = 30,
    patience: int = 5, channels: int = 24, batch_size: int = 16,
) -> NullFit:
    """Labels and exact target probabilities are absent from the fit API."""
    if epochs < 1 or patience < 1 or batch_size < 1:
        raise ValueError("positive training settings required")
    for indices in (train, validation):
        if not len(indices) or np.any(indices < 0) or np.any(indices >= len(obs.lengths)):
            raise ValueError("invalid fit records")
    if np.intersect1d(train, validation).size:
        raise ValueError("fit and validation records overlap")
    features, counts, k = observable_inputs(obs, budget, paired)
    offsets = obs.offsets
    train_tokens = np.concatenate([np.arange(offsets[i], offsets[i + 1]) for i in train])
    mean = features[train_tokens].mean(axis=0)
    scale = features[train_tokens].std(axis=0)
    scale = np.where(scale < 1e-6, 1.0, scale)
    features = (features - mean) / scale
    torch.manual_seed(seed)
    model = ConditionalCountTCN(features.shape[1], k, channels).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-3)
    rng = np.random.default_rng(seed)
    best_loss, best_epoch, best_state = float("inf"), 0, None
    history = []
    for epoch in range(1, epochs + 1):
        totals = {}
        for name, records in (("train", rng.permutation(train)), ("validation", validation)):
            model.train(name == "train")
            total = 0.0
            with torch.set_grad_enabled(name == "train"):
                for start in range(0, len(records), batch_size):
                    indices = records[start:start + batch_size]
                    x, y, mask = make_batch(features, counts, offsets, indices, device)
                    loss = count_nll(model(x, mask), y, mask)
                    if not torch.isfinite(loss):
                        raise RuntimeError("nonfinite count likelihood")
                    if name == "train":
                        optimizer.zero_grad()
                        loss.backward()
                        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                        optimizer.step()
                    total += float(loss.detach()) * len(indices)
            totals[f"{name}_nll"] = total / len(records)
        history.append({"epoch": epoch, **totals})
        if totals["validation_nll"] < best_loss - 1e-5:
            best_loss, best_epoch = totals["validation_nll"], epoch
            best_state = copy.deepcopy(model.state_dict())
        print(json.dumps({"paired": paired, "budget": budget, **history[-1]}), flush=True)
        if epoch - best_epoch >= patience:
            break
    model.load_state_dict(best_state)
    model.eval()
    return NullFit(model, mean, scale, history, best_epoch)


def directional_scores(logpmf: np.ndarray, counts: np.ndarray) -> tuple[float, float]:
    """Fixed positive tilts, mixed over global and Markov-local alternatives.

    No alternate model is fit on members. The Markov transition probabilities
    and tilts are preregistered assumptions, not learned membership truth.
    """
    support = np.arange(logpmf.shape[1], dtype=np.float64)
    tilts = np.asarray([0.5, 1.0, 2.0])
    llr = counts[:, None] * tilts - logsumexp(logpmf[:, :, None] + support[None, :, None] * tilts, axis=1)
    global_score = logsumexp(llr.sum(axis=0)) - np.log(len(tilts))
    enter, leave = 1.0 / 64.0, 1.0 / 8.0
    prior = enter / (enter + leave)
    inactive = np.full(len(tilts), np.log1p(-prior))
    active = np.log(prior) + llr[0]
    for emission in llr[1:]:
        inactive, active = (
            np.logaddexp(inactive + np.log1p(-enter), active + np.log(leave)),
            np.logaddexp(inactive + np.log(enter), active + np.log1p(-leave)) + emission,
        )
    span_score = logsumexp(np.logaddexp(inactive, active)) - np.log(len(tilts))
    return float(global_score), float(span_score)


@torch.inference_mode()
def predict_scores(fit: NullFit, obs: Observations, budget: int, paired: bool, device: torch.device):
    features, counts, _ = observable_inputs(obs, budget, paired)
    features = (features - fit.mean) / fit.scale
    offsets = obs.offsets
    scores = {name: np.empty(len(obs.lengths)) for name in ("global", "span", "nll")}
    for start in range(0, len(obs.lengths), 16):
        indices = np.arange(start, min(start + 16, len(obs.lengths)))
        x, y, mask = make_batch(features, counts, offsets, indices, device)
        predicted = fit.model(x, mask).cpu().numpy().astype(np.float64)
        for row, index in enumerate(indices):
            local = predicted[row, :obs.lengths[index]]
            observed = counts[offsets[index]:offsets[index + 1]]
            scores["global"][index], scores["span"][index] = directional_scores(local, observed)
            scores["nll"][index] = -np.mean(local[np.arange(len(observed)), observed])
    return scores


def _standardize(values: np.ndarray, reference: np.ndarray) -> np.ndarray:
    return (values - values[reference].mean()) / max(float(values[reference].std()), 1e-6)


def lowq_score(obs: Observations, k: int, reference: np.ndarray) -> np.ndarray:
    if k < 1 or k > obs.bits.shape[2]:
        raise ValueError("invalid low-q query budget")
    all_accept = np.all(obs.bits[:, 0, :k], axis=1).astype(float)
    raw = lowq_fragment_scores(all_accept, obs.logq[:, 0], obs.lengths)
    return standardized_max(raw, ("lowq_10", "lowq_20", "lowq_50"), reference)


def evaluate(
    obs: Observations, labels: np.ndarray, ids: np.ndarray, *, budget: int,
    output: Path, device: torch.device, seed: int, epochs: int, patience: int,
    channels: int = 24,
) -> dict[str, Any]:
    validate_records(obs, labels, ids)
    observable_inputs(obs, budget, False)
    if obs.logq.shape[1] > 1:
        observable_inputs(obs, budget, True)
    parts = record_partitions(labels, ids)
    output.mkdir(parents=True, exist_ok=True)
    lowq = lowq_score(obs, budget, parts["reference"])
    scores = {"lowq": lowq}
    offsets = obs.offsets
    scores["q_only"] = np.asarray([obs.logq[s:e, 0].mean() for s, e in zip(offsets[:-1], offsets[1:])])
    training = {}
    for paired in ((False, True) if obs.logq.shape[1] == 2 else (False,)):
        name = "paired" if paired else "original"
        fit = fit_null(obs, parts["train"], parts["validation"], budget=budget, paired=paired,
                       seed=seed, device=device, epochs=epochs, patience=patience, channels=channels)
        predicted = predict_scores(fit, obs, budget, paired, device)
        scores.update({f"{name}_{key}": value for key, value in predicted.items()})
        # Fixed weight; NEVER select it with member labels or test scores.
        fusion_base = lowq
        if paired:
            # Reuse ONLY the original half of the paired budget. Using the
            # full-budget original baseline here would silently cost 1.5 B.
            fusion_base = lowq_score(obs, budget // 2, parts["reference"])
        scores[f"{name}_fusion"] = (_standardize(fusion_base, parts["validation"])
                                       + 0.25 * _standardize(predicted["span"], parts["validation"]))
        training[name] = {"best_epoch": fit.best_epoch, "history": fit.history,
                          "original_queries": budget // 2 if paired else budget,
                          "counterfactual_queries": budget // 2 if paired else 0}
        torch.save({"state_dict": {k: v.cpu() for k, v in fit.model.state_dict().items()},
                    "mean": torch.from_numpy(fit.mean), "scale": torch.from_numpy(fit.scale),
                    "input_dim": len(fit.mean), "k": fit.model.k, "channels": channels,
                    "budget": budget, "paired": paired}, output / f"{name}.pt")
    # This is the first and only consumption of test membership for performance.
    metrics = {name: membership_metrics(value, labels, parts["calibration"], parts["test"])
               for name, value in scores.items()}
    report = {
        "experiment": "nonmember-only conditional accept-count likelihood",
        "seed": seed, "split_seed": SPLIT_SEED, "budget_per_candidate_token": budget,
        "training_member_count": 0, "synthetic_member_count": 0,
        "model_selection": "minimum held-out nonmember count NLL",
        "observations": "local logq and verifier bits; no target p/delta",
        "likelihood_scope": "conditional token-count model; cross-token independence is approximate",
        "protocol": "position-locked offline verifier replay, not a serial SD trajectory",
        "alternative": {"positive_count_tilts": [0.5, 1.0, 2.0], "enter": 1 / 64, "leave": 1 / 8},
        "fusion_weight": 0.25, "training": training, "metrics": metrics,
        "partitions": {name: ids[value].tolist() for name, value in parts.items()},
        "query_cost": {name: int(budget * obs.lengths[value].sum()) for name, value in parts.items() if name != "reference"},
        "calibration_min_pvalue": 1 / (len(parts["calibration"]) + 1),
    }
    np.savez_compressed(output / "scores.npz", labels=labels, record_ids=ids, **parts, **scores)
    _write_json(output / "REPORT.json", report)
    lines = ["# Conditional accept-only pilot", "", "Nonmember-only fit and selection; cached/position-locked replay.",
             "NLL is a two-sided diagnostic; global/span/fusion are member-positive hypotheses.", "",
             "| Method | AUC | pAUC@10% | TPR@1% | Actual FPR |", "|---|---:|---:|---:|---:|"]
    for name, metric in metrics.items():
        tail = metric["tpr_at_fpr"]["1%"]
        lines.append(f"| {name} | {metric['auc']:.4f} | {metric['pauc_0_10']:.4f} | {tail['tpr']:.4f} | {tail['actual_fpr']:.4f} |")
    (output / "REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--observations", type=Path, help="observable-only paired archive; otherwise replay existing original caches")
    parser.add_argument("--benchmark", choices=("wikitection", "newstection", "arxivtection"), default="wikitection")
    parser.add_argument("--epoch", type=int, choices=(1, 3), default=1)
    parser.add_argument("--budget", type=int, default=2)
    parser.add_argument("--seed", type=int, default=20260914)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--channels", type=int, default=24)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()
    if min(args.budget, args.threads, args.epochs, args.patience, args.channels) < 1:
        parser.error("budget, threads and training settings must be positive")
    torch.set_num_threads(args.threads)
    if args.observations:
        obs, labels, ids = load_observations(args.observations)
        scope_hash = hashlib.sha256()
        scope_hash.update(json.dumps(ids.tolist(), ensure_ascii=False).encode())
        scope_hash.update(np.asarray(obs.lengths, dtype="<i8").tobytes())
        scope_hash.update(np.asarray(obs.logq, dtype="<f8").tobytes())
        source = {"observations": str(args.observations.resolve()),
                  "sha256": hashlib.sha256(args.observations.read_bytes()).hexdigest(),
                  "candidate_scope_sha256": scope_hash.hexdigest(),
                  "seed_scope": "training only; supplied verifier bits are frozen"}
        condition = args.observations.stem
    else:
        obs, labels, ids = replay_observations(args.benchmark, args.epoch, args.budget, args.seed)
        source = {"benchmark": args.benchmark, "epoch": args.epoch, "replay_seed": args.seed,
                  "caches": {str(path): hashlib.sha256(path.read_bytes()).hexdigest() for path in _paths(args.benchmark, args.epoch)}}
        condition = f"{args.benchmark}_epoch{args.epoch}"
    # Validate budgets before creating artifacts or training either comparison.
    observable_inputs(obs, args.budget, False)
    if obs.logq.shape[1] > 1:
        observable_inputs(obs, args.budget, True)
    output = args.output_dir or ROOT / "experiments/results/sft_runs/conditional_accept_only" / condition / f"b{args.budget}_seed{args.seed}"
    report = evaluate(obs, labels, ids, budget=args.budget, output=output, device=torch.device(args.device),
                      seed=args.seed, epochs=args.epochs, patience=args.patience, channels=args.channels)
    _write_json(output / "SOURCE.json", source)
    print(json.dumps({"output": str(output), "metrics": report["metrics"]}), flush=True)


if __name__ == "__main__":
    main()
