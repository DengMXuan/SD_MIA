"""Evaluate interpretable q-only and accept-aware Low-q scale gates."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from experiments.sd_membership_sft.core.replay_cache import (load_replay_data)
from experiments.sd_membership_sft.core.audit_runtime import (_deterministic_subset)
from experiments.sd_membership_sft.archive.adaptive_window_accept_only import (_token_mask, fit_nonmember_model, fixed_q_observations, token_features)
from experiments.sd_membership_sft.core.audit_metrics import (membership_metrics)
from experiments.sd_membership_sft.core.audit_metrics import (rank_auc)
from experiments.sd_membership_sft.core.audit_runtime import (split_indices)
from experiments.sd_membership_sft.core.audit_runtime import (BENCHMARKS, EPOCHS, N_CAL, N_REF, REPLAY_SEEDS, ROOT, SPLIT_SEED, _paths, _write_json)
from experiments.sd_membership_sft.archive.neural_adaptive_accept_only import (pseudo_member_acceptance)


FRACTIONS = (0.05, 0.10, 0.15, 0.20, 0.30, 0.40, 0.50)


def scale_accept_scores(
    all_accept: np.ndarray,
    logq0: np.ndarray,
    lengths: np.ndarray,
    fractions: tuple[float, ...] = FRACTIONS,
) -> np.ndarray:
    all_accept = np.asarray(all_accept, dtype=np.float64)
    logq0 = np.asarray(logq0, dtype=np.float64)
    lengths = np.asarray(lengths, dtype=np.int64)
    if all_accept.shape != logq0.shape or len(all_accept) != int(np.sum(lengths)):
        raise ValueError("token arrays are not aligned with lengths")
    output = np.empty((len(lengths), len(fractions)), dtype=np.float64)
    offset = 0
    for record, length_value in enumerate(lengths):
        length = int(length_value)
        end = offset + length
        order = np.argsort(logq0[offset:end], kind="stable")
        for column, fraction in enumerate(fractions):
            count = max(1, int(math.ceil(fraction * length)))
            output[record, column] = float(np.mean(all_accept[offset:end][order[:count]]))
        offset = end
    return output


def fragment_q_summaries(logq0: np.ndarray, lengths: np.ndarray) -> np.ndarray:
    logq0 = np.asarray(logq0, dtype=np.float64)
    lengths = np.asarray(lengths, dtype=np.int64)
    rows = []
    offset = 0
    for length_value in lengths:
        length = int(length_value)
        end = offset + length
        surprise = np.clip(-logq0[offset:end], 0.0, 30.0)
        quantiles = np.quantile(surprise, (0.0, 0.10, 0.25, 0.50, 0.75, 0.90, 1.0))
        centered = surprise - float(np.mean(surprise))
        autocorrelation = 0.0
        if length > 1 and float(np.sum(centered * centered)) > 1e-12:
            autocorrelation = float(
                np.sum(centered[:-1] * centered[1:]) / np.sum(centered * centered)
            )
        threshold = float(np.quantile(surprise, 0.80))
        lowq = surprise >= threshold
        longest = current = 0
        for value in lowq:
            current = current + 1 if value else 0
            longest = max(longest, current)
        rows.append(
            np.r_[
                math.log1p(length),
                quantiles,
                float(np.mean(surprise)),
                float(np.std(surprise)),
                autocorrelation,
                longest / length,
            ]
        )
        offset = end
    return np.asarray(rows, dtype=np.float32)


class ScaleGate(nn.Module):
    """Sparse convex scale mixture; q-only mode cannot inspect accept evidence."""

    def __init__(self, summary_dim: int, scales: int, accept_aware: bool) -> None:
        super().__init__()
        input_dim = summary_dim + (scales if accept_aware else 0)
        self.accept_aware = accept_aware
        self.network = nn.Sequential(
            nn.Linear(input_dim, 24), nn.GELU(), nn.Linear(24, scales)
        )
        self.gain_raw = nn.Parameter(torch.tensor(0.5))
        self.bias = nn.Parameter(torch.zeros(()))

    def forward(
        self, summary: torch.Tensor, evidence: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        gate_input = torch.cat((summary, evidence), dim=1) if self.accept_aware else summary
        weights = torch.softmax(self.network(gate_input) / 0.35, dim=1)
        combined = torch.sum(weights * evidence, dim=1)
        score = F.softplus(self.gain_raw) * combined + self.bias
        return score, weights


def _simulate_examples(
    records: np.ndarray,
    q_summary: np.ndarray,
    expected: np.ndarray,
    logq0: np.ndarray,
    offsets: np.ndarray,
    actual_all: np.ndarray,
    centers: np.ndarray,
    scales: np.ndarray,
    *,
    k: int,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    summaries, evidence, labels = [], [], []
    for record_value in records:
        record = int(record_value)
        start, end = int(offsets[record]), int(offsets[record + 1])
        order = np.argsort(logq0[start:end], kind="stable")
        for variant in range(5):
            if variant == 0:
                all_accept = actual_all[start:end]
                label = 0.0
            else:
                example_seed = int(
                    np.random.SeedSequence([seed, record, variant]).generate_state(1)[0]
                )
                alpha = expected[start:end]
                label = float(variant >= 2)
                if variant >= 2:
                    alpha, _ = pseudo_member_acceptance(
                        alpha,
                        logq0[start:end],
                        seed=example_seed,
                        family=variant - 2,
                    )
                rng = np.random.default_rng(example_seed + 29)
                all_accept = np.all(
                    rng.random((end - start, k)) < alpha[:, None], axis=1
                ).astype(np.float64)
            raw = []
            for fraction in FRACTIONS:
                count = max(1, int(math.ceil(fraction * (end - start))))
                raw.append(float(np.mean(all_accept[order[:count]])))
            summaries.append(q_summary[record])
            evidence.append((np.asarray(raw) - centers) / scales)
            labels.append(label)
    return (
        np.asarray(summaries, dtype=np.float32),
        np.asarray(evidence, dtype=np.float32),
        np.asarray(labels, dtype=np.float32),
    )


def _fit_gate(
    train: tuple[np.ndarray, np.ndarray, np.ndarray],
    validation: tuple[np.ndarray, np.ndarray, np.ndarray],
    *,
    accept_aware: bool,
    seed: int,
    device: torch.device,
) -> tuple[ScaleGate, dict[str, float | int]]:
    torch.manual_seed(seed)
    model = ScaleGate(train[0].shape[1], train[1].shape[1], accept_aware).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-3, weight_decay=1e-3)
    train_tensors = tuple(torch.from_numpy(value).to(device) for value in train)
    validation_tensors = tuple(torch.from_numpy(value).to(device) for value in validation)
    best_auc, best_epoch, stale, best_state = -np.inf, 0, 0, None
    rng = np.random.default_rng(seed)
    for epoch in range(1, 101):
        model.train()
        permutation = rng.permutation(len(train[2]))
        for start in range(0, len(permutation), 128):
            rows = torch.as_tensor(permutation[start : start + 128], device=device)
            score, weights = model(train_tensors[0][rows], train_tensors[1][rows])
            classification = F.binary_cross_entropy_with_logits(score, train_tensors[2][rows])
            entropy = -torch.sum(weights * torch.log(weights.clamp_min(1e-8)), dim=1).mean()
            loss = classification + 0.01 * entropy
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
        model.eval()
        with torch.inference_mode():
            score, _ = model(validation_tensors[0], validation_tensors[1])
        values = score.cpu().numpy()
        labels = validation[2].astype(np.int64)
        auc = rank_auc(values[labels == 1], values[labels == 0])
        if auc > best_auc + 1e-4:
            best_auc, best_epoch, stale = auc, epoch, 0
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
        else:
            stale += 1
            if stale >= 12:
                break
    if best_state is None:
        raise RuntimeError("scale gate did not train")
    model.load_state_dict(best_state)
    model.eval()
    return model, {"best_epoch": best_epoch, "synthetic_validation_auc": float(best_auc)}


@torch.inference_mode()
def _predict_gate(
    model: ScaleGate,
    summary: np.ndarray,
    evidence: np.ndarray,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray]:
    score, weights = model(
        torch.from_numpy(np.asarray(summary, dtype=np.float32)).to(device),
        torch.from_numpy(np.asarray(evidence, dtype=np.float32)).to(device),
    )
    return score.cpu().numpy(), weights.cpu().numpy()


def evaluate_condition_seed(
    benchmark: str, epoch: int, seed: int, device: torch.device
) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    full, pq = _paths(benchmark, epoch)
    data = load_replay_data(full, pq)
    partitions = split_indices(data.labels, SPLIT_SEED)
    d_nm = partitions["D"][data.labels[partitions["D"]] == 0]
    c_nm = partitions["C"][data.labels[partitions["C"]] == 0]
    reference = _deterministic_subset(d_nm, N_REF, SPLIT_SEED + N_REF)
    calibration = _deterministic_subset(c_nm, N_CAL, SPLIT_SEED + N_CAL + 1000)
    shuffled = np.random.default_rng(SPLIT_SEED + seed).permutation(reference)
    train_records, validation_records = np.sort(shuffled[:320]), np.sort(shuffled[320:])
    rates, all_accept = fixed_q_observations(data, 2, seed)
    static = token_features(data.logq0, data.lengths)
    null_model = fit_nonmember_model(
        static,
        rates,
        _token_mask(train_records, data.offsets),
        trials_per_token=2,
        seed=seed + 2,
    )
    expected = null_model.predict(static)
    raw = scale_accept_scores(all_accept, data.logq0, data.lengths)
    centers = np.mean(raw[train_records], axis=0)
    scale = np.std(raw[train_records], axis=0)
    scale = np.where(scale < 1e-6, 1.0, scale)
    evidence = (raw - centers) / scale
    summary = fragment_q_summaries(data.logq0, data.lengths)
    summary_mean, summary_scale = np.mean(summary[train_records], axis=0), np.std(
        summary[train_records], axis=0
    )
    summary_scale = np.where(summary_scale < 1e-6, 1.0, summary_scale)
    summary = np.asarray((summary - summary_mean) / summary_scale, dtype=np.float32)
    train = _simulate_examples(
        train_records,
        summary,
        expected,
        data.logq0,
        data.offsets,
        all_accept,
        centers,
        scale,
        k=2,
        seed=seed + 100,
    )
    validation = _simulate_examples(
        validation_records,
        summary,
        expected,
        data.logq0,
        data.offsets,
        all_accept,
        centers,
        scale,
        k=2,
        seed=seed + 1000,
    )
    q_model, q_meta = _fit_gate(
        train, validation, accept_aware=False, seed=seed + 10, device=device
    )
    accept_model, accept_meta = _fit_gate(
        train, validation, accept_aware=True, seed=seed + 20, device=device
    )
    q_score, q_weights = _predict_gate(q_model, summary, evidence, device)
    accept_score, accept_weights = _predict_gate(accept_model, summary, evidence, device)
    legacy_columns = [FRACTIONS.index(value) for value in (0.10, 0.20, 0.50)]
    scores = {
        "legacy_max_10_20_50": np.max(evidence[:, legacy_columns], axis=1),
        "dense_max": np.max(evidence, axis=1),
        "q_only_sparse_gate": q_score,
        "accept_aware_gate": accept_score,
    }
    metrics = {
        name: membership_metrics(value, data.labels, calibration, partitions["T"])
        for name, value in scores.items()
    }

    def weight_summary(weights: np.ndarray) -> dict[str, Any]:
        fit = weights[reference]
        entropy = -np.sum(fit * np.log(np.clip(fit, 1e-8, 1.0)), axis=1)
        return {
            "mean_weights": np.mean(fit, axis=0),
            "mean_max_weight": float(np.mean(np.max(fit, axis=1))),
            "mean_effective_scales": float(np.mean(np.exp(entropy))),
        }

    row = {
        "benchmark": benchmark,
        "epoch": epoch,
        "seed": seed,
        "metrics": metrics,
        "training": {"q_only": q_meta, "accept_aware": accept_meta},
        "weights": {
            "q_only": weight_summary(q_weights),
            "accept_aware": weight_summary(accept_weights),
        },
    }
    return row, {
        "labels": data.labels,
        "record_ids": data.record_ids,
        **scores,
        "q_only_weights": q_weights,
        "accept_aware_weights": accept_weights,
    }


def aggregate(rows: list[dict[str, Any]]) -> dict[str, Any]:
    methods = tuple(rows[0]["metrics"])
    metrics = {
        method: {
            "auc": float(np.mean([row["metrics"][method]["auc"] for row in rows])),
            "pauc_0_10": float(
                np.mean([row["metrics"][method]["pauc_0_10"] for row in rows])
            ),
            "tpr_1": float(
                np.mean(
                    [row["metrics"][method]["tpr_at_fpr"]["1%"]["tpr"] for row in rows]
                )
            ),
            "actual_fpr_1": float(
                np.mean(
                    [
                        row["metrics"][method]["tpr_at_fpr"]["1%"]["actual_fpr"]
                        for row in rows
                    ]
                )
            ),
        }
        for method in methods
    }
    return {"experiment": "interpretable scale gate", "rows": len(rows), "metrics": metrics}


def write_markdown(summary: dict[str, Any], path: Path) -> None:
    lines = [
        "# Interpretable Low-q Scale Gate",
        "",
        "| Method | AUC | pAUC | TPR@1% / FPR |",
        "|---|---:|---:|---:|",
    ]
    for method, metric in summary["metrics"].items():
        lines.append(
            f"| `{method}` | {metric['auc']:.4f} | {metric['pauc_0_10']:.4f} | "
            f"{metric['tpr_1']:.4f} / {metric['actual_fpr_1']:.4f} |"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--benchmark", choices=BENCHMARKS)
    parser.add_argument("--epoch", choices=EPOCHS, type=int)
    parser.add_argument("--seeds", nargs="+", type=int, default=list(REPLAY_SEEDS))
    parser.add_argument("--device", default="auto")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "experiments/results/sft_runs/accept_only_active_v2/scale_gate",
    )
    parser.add_argument("--aggregate-existing", action="store_true")
    args = parser.parse_args()
    output = args.output_dir.resolve()
    if args.aggregate_existing:
        paths = sorted((output / "conditions").glob("*/RAW_RESULTS.json"))
        rows = [row for path in paths for row in json.loads(path.read_text())["rows"]]
        summary = aggregate(rows)
        _write_json(output / "AGGREGATE.json", summary)
        write_markdown(summary, output / "AGGREGATE.md")
        return
    if args.benchmark is None or args.epoch is None:
        parser.error("--benchmark and --epoch are required")
    device = torch.device("cuda" if args.device == "auto" and torch.cuda.is_available() else args.device)
    condition = output / "conditions" / f"{args.benchmark}_epoch{args.epoch}"
    rows = []
    for seed in args.seeds:
        row, arrays = evaluate_condition_seed(args.benchmark, args.epoch, seed, device)
        rows.append(row)
        condition.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(condition / f"scores_seed_{seed}.npz", **arrays)
        print(json.dumps({"condition": condition.name, "seed": seed}), flush=True)
    report = {"experiment": "interpretable scale gate", "rows": rows}
    _write_json(condition / "RAW_RESULTS.json", report)
    summary = aggregate(rows)
    _write_json(condition / "AGGREGATE.json", summary)
    write_markdown(summary, condition / "AGGREGATE.md")


if __name__ == "__main__":
    main()
