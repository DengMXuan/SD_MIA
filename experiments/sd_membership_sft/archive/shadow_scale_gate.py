"""Compare pseudo, legitimate local-shadow, mixed, and one-class scale gates."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F

from experiments.shared.core.replay_cache import load_replay_data
from experiments.shared.core.audit_runtime import _deterministic_subset
from experiments.sd_membership_sft.archive.adaptive_window_accept_only import _token_mask, fit_nonmember_model, fixed_q_observations, token_features
from experiments.shared.core.audit_metrics import membership_metrics
from experiments.shared.core.audit_runtime import split_indices
from experiments.sd_membership_sft.archive.interpretable_scale_gate import FRACTIONS, ScaleGate, _fit_gate, _predict_gate, _simulate_examples, fragment_q_summaries, scale_accept_scores
from experiments.shared.core.audit_runtime import BENCHMARKS, EPOCHS, N_CAL, N_REF, REPLAY_SEEDS, ROOT, SPLIT_SEED, _paths, _write_json


def _shadow_examples(
    path: Path, seed: int
) -> tuple[
    tuple[np.ndarray, np.ndarray, np.ndarray],
    tuple[np.ndarray, np.ndarray, np.ndarray],
]:
    with np.load(path, allow_pickle=False) as archive:
        labels = np.asarray(archive["labels"], dtype=np.int64)
        lengths = np.asarray(archive["lengths"], dtype=np.int64)
        logp = np.asarray(archive["logp"], dtype=np.float64)
        logq0 = np.asarray(archive["logq0"], dtype=np.float64)
    alpha = np.exp(np.minimum(0.0, logp - logq0))
    rng = np.random.default_rng(seed)
    all_accept = np.all(rng.random((len(alpha), 2)) < alpha[:, None], axis=1).astype(float)
    raw = scale_accept_scores(all_accept, logq0, lengths)
    indices = []
    for label in (0, 1):
        values = rng.permutation(np.flatnonzero(labels == label))
        indices.append(values)
    train_indices = np.sort(np.r_[indices[0][:160], indices[1][:160]])
    validation_indices = np.sort(np.r_[indices[0][160:], indices[1][160:]])
    nm_train = train_indices[labels[train_indices] == 0]
    center, scale = np.mean(raw[nm_train], axis=0), np.std(raw[nm_train], axis=0)
    scale = np.where(scale < 1e-6, 1.0, scale)
    evidence = np.asarray((raw - center) / scale, dtype=np.float32)
    summary = fragment_q_summaries(logq0, lengths)
    summary_mean, summary_scale = np.mean(summary[train_indices], axis=0), np.std(
        summary[train_indices], axis=0
    )
    summary_scale = np.where(summary_scale < 1e-6, 1.0, summary_scale)
    summary = np.asarray((summary - summary_mean) / summary_scale, dtype=np.float32)

    def take(rows: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        return summary[rows], evidence[rows], labels[rows].astype(np.float32)

    return take(train_indices), take(validation_indices)


def _fit_one_class(
    train_summary: np.ndarray,
    train_evidence: np.ndarray,
    validation_summary: np.ndarray,
    validation_evidence: np.ndarray,
    *,
    seed: int,
    device: torch.device,
) -> ScaleGate:
    torch.manual_seed(seed)
    model = ScaleGate(train_summary.shape[1], train_evidence.shape[1], False).to(device)
    model.gain_raw.requires_grad_(False)
    model.bias.requires_grad_(False)
    optimizer = torch.optim.AdamW(model.network.parameters(), lr=2e-3, weight_decay=1e-3)
    summary = torch.from_numpy(train_summary).to(device)
    evidence = torch.from_numpy(train_evidence).to(device)
    validation_summary_t = torch.from_numpy(validation_summary).to(device)
    validation_evidence_t = torch.from_numpy(validation_evidence).to(device)
    best_loss, best_state, stale = np.inf, None, 0
    for _ in range(100):
        model.train()
        _, weights = model(summary, evidence)
        combined = torch.sum(weights * evidence, dim=1)
        entropy = -torch.sum(weights * torch.log(weights.clamp_min(1e-8)), dim=1).mean()
        loss = torch.mean(torch.square(combined)) + 0.01 * entropy
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        model.eval()
        with torch.inference_mode():
            _, val_weights = model(validation_summary_t, validation_evidence_t)
            val_combined = torch.sum(val_weights * validation_evidence_t, dim=1)
            value = float(torch.mean(torch.square(val_combined)).cpu())
        if value < best_loss - 1e-5:
            best_loss, stale = value, 0
            best_state = {key: tensor.detach().cpu().clone() for key, tensor in model.state_dict().items()}
        else:
            stale += 1
            if stale >= 12:
                break
    if best_state is None:
        raise RuntimeError("one-class scale gate failed")
    model.load_state_dict(best_state)
    model.eval()
    return model


def evaluate_condition_seed(
    benchmark: str,
    epoch: int,
    seed: int,
    device: torch.device,
    shadow_root: Path,
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
    expected = fit_nonmember_model(
        static,
        rates,
        _token_mask(train_records, data.offsets),
        trials_per_token=2,
        seed=seed + 2,
    ).predict(static)
    raw = scale_accept_scores(all_accept, data.logq0, data.lengths)
    center, scale = np.mean(raw[train_records], axis=0), np.std(raw[train_records], axis=0)
    scale = np.where(scale < 1e-6, 1.0, scale)
    evidence = np.asarray((raw - center) / scale, dtype=np.float32)
    summary = fragment_q_summaries(data.logq0, data.lengths)
    summary_mean, summary_scale = np.mean(summary[train_records], axis=0), np.std(
        summary[train_records], axis=0
    )
    summary_scale = np.where(summary_scale < 1e-6, 1.0, summary_scale)
    summary = np.asarray((summary - summary_mean) / summary_scale, dtype=np.float32)
    pseudo_train = _simulate_examples(
        train_records,
        summary,
        expected,
        data.logq0,
        data.offsets,
        all_accept,
        center,
        scale,
        k=2,
        seed=seed + 100,
    )
    pseudo_validation = _simulate_examples(
        validation_records,
        summary,
        expected,
        data.logq0,
        data.offsets,
        all_accept,
        center,
        scale,
        k=2,
        seed=seed + 1000,
    )
    shadow_train, shadow_validation = _shadow_examples(
        shadow_root / benchmark / "shadow_pq.npz", seed + 2000
    )
    mixed_train = tuple(
        np.concatenate((pseudo_train[index], shadow_train[index]), axis=0)
        for index in range(3)
    )
    mixed_validation = tuple(
        np.concatenate((pseudo_validation[index], shadow_validation[index]), axis=0)
        for index in range(3)
    )
    pseudo_model, _ = _fit_gate(
        pseudo_train, pseudo_validation, accept_aware=False, seed=seed + 10, device=device
    )
    shadow_model, _ = _fit_gate(
        shadow_train, shadow_validation, accept_aware=False, seed=seed + 20, device=device
    )
    mixed_model, _ = _fit_gate(
        mixed_train, mixed_validation, accept_aware=False, seed=seed + 30, device=device
    )
    one_class_model = _fit_one_class(
        summary[train_records],
        evidence[train_records],
        summary[validation_records],
        evidence[validation_records],
        seed=seed + 40,
        device=device,
    )
    scores = {"dense_max": np.max(evidence, axis=1)}
    weights = {}
    for name, model in (
        ("pseudo_gate", pseudo_model),
        ("shadow_gate", shadow_model),
        ("mixed_gate", mixed_model),
        ("one_class_gate", one_class_model),
    ):
        scores[name], weights[name] = _predict_gate(model, summary, evidence, device)
    metrics = {
        name: membership_metrics(value, data.labels, calibration, partitions["T"])
        for name, value in scores.items()
    }
    return (
        {"benchmark": benchmark, "epoch": epoch, "seed": seed, "metrics": metrics},
        {
            "labels": data.labels,
            "record_ids": data.record_ids,
            **scores,
            **{f"{name}_weights": value for name, value in weights.items()},
        },
    )


def aggregate(rows: list[dict[str, Any]]) -> dict[str, Any]:
    methods = tuple(rows[0]["metrics"])
    result = {
        "experiment": "scale gate supervision comparison",
        "rows": len(rows),
        "metrics": {
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
        },
    }
    baseline = "dense_max"
    result["paired_point_deltas_vs_dense_max"] = {
        method: {
            "auc": float(
                np.mean(
                    [
                        row["metrics"][method]["auc"]
                        - row["metrics"][baseline]["auc"]
                        for row in rows
                    ]
                )
            ),
            "pauc_0_10": float(
                np.mean(
                    [
                        row["metrics"][method]["pauc_0_10"]
                        - row["metrics"][baseline]["pauc_0_10"]
                        for row in rows
                    ]
                )
            ),
            "pauc_condition_seed_wins": int(
                np.sum(
                    [
                        row["metrics"][method]["pauc_0_10"]
                        > row["metrics"][baseline]["pauc_0_10"]
                        for row in rows
                    ]
                )
            ),
            "condition_seed_total": len(rows),
        }
        for method in methods
        if method != baseline
    }
    return result


def write_markdown(summary: dict[str, Any], path: Path) -> None:
    lines = [
        "# Scale-Gate Supervision Comparison",
        "",
        "| Method | AUC | pAUC | TPR@1% / FPR |",
        "|---|---:|---:|---:|",
    ]
    for method, metric in summary["metrics"].items():
        lines.append(
            f"| `{method}` | {metric['auc']:.4f} | {metric['pauc_0_10']:.4f} | "
            f"{metric['tpr_1']:.4f} / {metric['actual_fpr_1']:.4f} |"
        )
    lines.extend(
        [
            "",
            "| Method vs dense-max | Delta AUC | Delta pAUC | pAUC wins |",
            "|---|---:|---:|---:|",
        ]
    )
    for method, value in summary["paired_point_deltas_vs_dense_max"].items():
        lines.append(
            f"| `{method}` | {value['auc']:+.4f} | {value['pauc_0_10']:+.4f} | "
            f"{value['pauc_condition_seed_wins']}/{value['condition_seed_total']} |"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--benchmark", choices=BENCHMARKS)
    parser.add_argument("--epoch", choices=EPOCHS, type=int)
    parser.add_argument("--seeds", nargs="+", type=int, default=list(REPLAY_SEEDS))
    parser.add_argument("--device", default="auto")
    parser.add_argument(
        "--shadow-root",
        type=Path,
        default=ROOT / "experiments/results/sft_runs/accept_only_active_v2/local_shadow",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "experiments/results/sft_runs/accept_only_active_v2/shadow_scale_gate",
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
        row, arrays = evaluate_condition_seed(
            args.benchmark, args.epoch, seed, device, args.shadow_root.resolve()
        )
        rows.append(row)
        condition.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(condition / f"scores_seed_{seed}.npz", **arrays)
        print(json.dumps({"condition": condition.name, "seed": seed}), flush=True)
    report = {"experiment": "scale gate supervision comparison", "rows": rows}
    _write_json(condition / "RAW_RESULTS.json", report)
    summary = aggregate(rows)
    _write_json(condition / "AGGREGATE.json", summary)
    write_markdown(summary, condition / "AGGREGATE.md")


if __name__ == "__main__":
    main()
