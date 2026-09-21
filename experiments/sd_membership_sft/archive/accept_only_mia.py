"""Strict Accept-Only membership audit on a fixed-candidate SD replay.

Only Bernoulli accept/reject observations are exposed to the detector.  The
saved full-delta cache is used internally as a protocol simulator to draw
``Bernoulli(min(1, exp(delta)))``; p, q and delta never enter the detector
features or fitted models.  This is the registered position-locked replay:
the same record prefix/candidate is re-queried for every repetition.  A
natural speculative trajectory must be reported separately because a reject
changes the later context.
"""

from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn

from experiments.sd_membership_sft.core.audit_runtime import (DEFAULT_SPLIT_SEED, split_indices)
from experiments.sd_membership_sft.archive.full_delta_mia import (DEFAULT_TRAINING_SEEDS, _bootstrap_metrics)
from experiments.sd_membership_sft.core.audit_metrics import (partial_auc)
from experiments.sd_membership_sft.archive.stat_delta_mia import (fit_logistic)

from experiments.paths import ROOT
BENCHMARKS = ("wikitection", "newstection", "arxivtection")
EPOCHS = (1, 3)
QUERY_COUNTS = (1, 4, 16, 64)
WINDOWS = (4, 8, 16)
BIT_STAT_NAMES = (
    "accept_rate",
    "transition_rate",
    "accept_run_mean",
    "accept_run_q90",
    "accept_run_max",
    "reject_run_mean",
    "reject_run_q90",
    "reject_run_max",
    "segment_1_accept_rate",
    "segment_2_accept_rate",
    "segment_3_accept_rate",
    "window_4_rate_q10",
    "window_4_rate_q90",
    "window_8_rate_q10",
    "window_8_rate_q90",
    "window_16_rate_q10",
    "window_16_rate_q90",
)


def _load(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    with np.load(path, allow_pickle=False) as data:
        required = ("labels", "record_ids", "lengths", "offsets", "delta")
        missing = [key for key in required if key not in data.files]
        if missing:
            raise ValueError(f"{path} is missing {missing}")
        labels = np.asarray(data["labels"], dtype=np.int64)
        record_ids = np.asarray(data["record_ids"])
        lengths = np.asarray(data["lengths"], dtype=np.int64)
        offsets = np.asarray(data["offsets"], dtype=np.int64)
        delta = np.asarray(data["delta"], dtype=np.float32)
    if len(lengths) != len(labels) or len(offsets) != len(labels) + 1:
        raise ValueError("invalid accept-only archive alignment")
    if int(offsets[-1]) != len(delta) or np.any(lengths <= 0):
        raise ValueError("invalid accept-only offsets")
    return labels, record_ids, lengths, offsets, delta


def run_lengths(bits: np.ndarray, value: int) -> np.ndarray:
    values = np.asarray(bits, dtype=np.uint8)
    if len(values) == 0:
        return np.zeros(0, dtype=np.float64)
    changes = np.flatnonzero(np.r_[True, values[1:] != values[:-1], True])
    lengths = np.diff(changes).astype(np.float64)
    starts = values[changes[:-1]]
    return lengths[starts == value]


def _window_rates(bits: np.ndarray, width: int) -> np.ndarray:
    if len(bits) < width:
        return np.asarray([float(np.mean(bits))], dtype=np.float64)
    cumulative = np.concatenate(([0.0], np.cumsum(bits, dtype=np.float64)))
    return (cumulative[width:] - cumulative[:-width]) / width


def bit_statistics(bits: np.ndarray) -> np.ndarray:
    bits = np.asarray(bits, dtype=np.uint8)
    accept_runs = run_lengths(bits, 1)
    reject_runs = run_lengths(bits, 0)
    segment = np.array_split(bits, 3)
    row: list[float] = [
        float(np.mean(bits)),
        float(np.mean(bits[1:] != bits[:-1])) if len(bits) > 1 else 0.0,
        float(np.mean(accept_runs)) if len(accept_runs) else 0.0,
        float(np.quantile(accept_runs, 0.90)) if len(accept_runs) else 0.0,
        float(np.max(accept_runs)) if len(accept_runs) else 0.0,
        float(np.mean(reject_runs)) if len(reject_runs) else 0.0,
        float(np.quantile(reject_runs, 0.90)) if len(reject_runs) else 0.0,
        float(np.max(reject_runs)) if len(reject_runs) else 0.0,
        *[float(np.mean(piece)) if len(piece) else 0.0 for piece in segment],
    ]
    for width in WINDOWS:
        rates = _window_rates(bits, width)
        row.extend((float(np.quantile(rates, 0.10)), float(np.quantile(rates, 0.90))))
    result = np.asarray(row, dtype=np.float64)
    if result.shape != (17,) or not np.all(np.isfinite(result)):
        raise RuntimeError("accept-only statistic construction failed")
    return result


def fixed_position_bins(bits: np.ndarray, bins: int) -> np.ndarray:
    if bins <= 0 or len(bits) < bins:
        raise ValueError(f"record length {len(bits)} is shorter than fixed bin count {bins}")
    pieces = np.array_split(np.asarray(bits, dtype=np.uint8), bins)
    return np.asarray([float(np.mean(piece)) for piece in pieces], dtype=np.float32)


def simulate_replay(
    delta: np.ndarray,
    lengths: np.ndarray,
    offsets: np.ndarray,
    queries: int,
    query_seed: int,
    bins: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Return [N,K,17] stats and [N,K,B] fixed-position bit rates."""
    stats = np.empty((len(lengths), queries, len(BIT_STAT_NAMES)), dtype=np.float32)
    position = np.empty((len(lengths), queries, bins), dtype=np.float32)
    for index, (start, end) in enumerate(zip(offsets[:-1], offsets[1:])):
        values = np.asarray(delta[int(start) : int(end)], dtype=np.float64)
        alpha = np.minimum(1.0, np.exp(np.clip(values, -80.0, 80.0)))
        rng = np.random.default_rng(np.random.SeedSequence([query_seed, index]))
        bits = rng.random((queries, len(values))) < alpha[None, :]
        for query in range(queries):
            row = bits[query].astype(np.uint8)
            stats[index, query] = bit_statistics(row)
            position[index, query] = fixed_position_bins(row, bins)
    return stats, position


def shuffle_replay(stats: np.ndarray, position_source: np.ndarray, seed: int) -> tuple[np.ndarray, np.ndarray]:
    """Shuffle each observed bit track while preserving its number of accepts."""
    # The raw bits are not retained after feature extraction.  A Bernoulli
    # reconstruction at the observed per-track rate is deliberately used only
    # for the negative-control dataset; it cannot improve on the real bit
    # sequence and is marked as a synthetic control in the report.
    rng = np.random.default_rng(seed)
    synthetic = rng.binomial(1, np.clip(position_source.mean(axis=2, keepdims=True), 0.0, 1.0), size=position_source.shape).astype(np.float32)
    shuffled = np.empty_like(synthetic)
    for index in range(len(synthetic)):
        for query in range(synthetic.shape[1]):
            shuffled[index, query] = rng.permutation(synthetic[index, query])
    stats_out = np.asarray(
        [[bit_statistics(np.rint(row).astype(np.uint8)) for row in record] for record in shuffled],
        dtype=np.float32,
    )
    position_out = np.asarray(
        [[fixed_position_bins(np.rint(row).astype(np.uint8), position_source.shape[2]) for row in record] for record in shuffled],
        dtype=np.float32,
    )
    return stats_out, position_out


class BitTCN(nn.Module):
    def __init__(self, channels: int, kernel: int, dropout: float) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(1, channels, kernel, padding=kernel // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Conv1d(channels, channels, kernel, padding=kernel // 2),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.head = nn.Linear(channels * 2, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        hidden = self.net(x.unsqueeze(1))
        return self.head(torch.cat((hidden.mean(dim=2), hidden.amax(dim=2)), dim=1)).squeeze(-1)


class BitTransformer(nn.Module):
    def __init__(self, hidden: int, layers: int, dropout: float) -> None:
        super().__init__()
        self.projection = nn.Linear(1, hidden)
        block = nn.TransformerEncoderLayer(
            d_model=hidden, nhead=4, dim_feedforward=hidden * 2,
            dropout=dropout, activation="gelu", batch_first=True, norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(block, num_layers=layers)
        self.head = nn.Linear(hidden, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.encoder(self.projection(x.unsqueeze(-1))).mean(dim=1)).squeeze(-1)


class QueryTransformer(nn.Module):
    def __init__(self, bins: int, hidden: int, layers: int, dropout: float) -> None:
        super().__init__()
        self.projection = nn.Linear(bins, hidden)
        block = nn.TransformerEncoderLayer(
            d_model=hidden, nhead=4, dim_feedforward=hidden * 2,
            dropout=dropout, activation="gelu", batch_first=True, norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(block, num_layers=layers)
        self.head = nn.Linear(hidden, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.encoder(self.projection(x)).mean(dim=1)).squeeze(-1)


def _seed(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)


def _sequence_candidates(kind: str) -> list[dict[str, Any]]:
    if kind == "tcn":
        return [{"channels": c, "kernel": k, "dropout": d, "lr": lr, "weight_decay": wd}
                for c in (16, 32) for k in (3, 5) for d in (0.0, 0.1)
                for lr in (3e-4, 1e-3) for wd in (1e-4, 1e-3)]
    return [{"hidden": h, "layers": layer, "dropout": d, "lr": lr, "weight_decay": wd}
            for h in (32, 64) for layer in (1, 2) for d in (0.0, 0.1)
            for lr in (3e-4, 1e-3) for wd in (1e-4, 1e-3)]


def _fit_sequence(
    values: np.ndarray,
    labels: np.ndarray,
    partitions: dict[str, np.ndarray],
    kind: str,
    seeds: tuple[int, ...],
    device: torch.device,
    max_epochs: int,
    patience: int,
) -> tuple[np.ndarray, dict[str, Any]]:
    candidates = _sequence_candidates(kind)
    train, validation = partitions["D"], partitions["V"]
    mean = values[train].reshape(-1, values.shape[-1]).mean(axis=0)
    std = values[train].reshape(-1, values.shape[-1]).std(axis=0)
    std = np.where(std < 1e-8, 1.0, std)
    values = (values - mean) / std
    table: list[dict[str, Any]] = []
    score_cache: dict[int, np.ndarray] = {}
    x = torch.from_numpy(values.astype(np.float32)).to(device)
    y = torch.from_numpy(labels.astype(np.float32)).to(device)
    for index, spec in enumerate(candidates):
        seed_results: list[dict[str, Any]] = []
        columns: list[np.ndarray] = []
        for seed in seeds:
            _seed(seed)
            if kind == "tcn":
                model: nn.Module = BitTCN(int(spec["channels"]), int(spec["kernel"]), float(spec["dropout"])) if values.ndim == 2 else BitTCN(int(spec["channels"]), int(spec["kernel"]), float(spec["dropout"]))
            elif kind == "bit-transformer":
                model = BitTransformer(int(spec["hidden"]), int(spec["layers"]), float(spec["dropout"]))
            elif kind == "query-transformer":
                model = QueryTransformer(values.shape[-1], int(spec["hidden"]), int(spec["layers"]), float(spec["dropout"]))
            else:
                raise ValueError(kind)
            model = model.to(device)
            optimizer = torch.optim.AdamW(model.parameters(), lr=float(spec["lr"]), weight_decay=float(spec["weight_decay"]))
            best = -np.inf
            best_state: dict[str, torch.Tensor] | None = None
            stale = 0
            best_epoch = 0
            for epoch in range(1, max_epochs + 1):
                model.train()
                logits = model(x[train])
                loss = nn.functional.binary_cross_entropy_with_logits(logits, y[train])
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                model.eval()
                with torch.inference_mode():
                    v = model(x[validation]).cpu().numpy()
                score = partial_auc(v, labels[validation])
                if score > best + 1e-10:
                    best = score
                    best_epoch = epoch
                    stale = 0
                    best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
                else:
                    stale += 1
                    if stale >= patience:
                        break
            if best_state is None:
                raise RuntimeError("accept-only sequence model did not checkpoint")
            model.load_state_dict(best_state)
            model.eval()
            with torch.inference_mode():
                scores = model(x).cpu().numpy().astype(np.float64)
            columns.append(scores)
            seed_results.append({"seed": seed, "validation_pauc_0_10": float(best), "best_epoch": best_epoch})
            del model
        mean_val = float(np.mean([row["validation_pauc_0_10"] for row in seed_results]))
        std_val = float(np.std([row["validation_pauc_0_10"] for row in seed_results]))
        table.append({"candidate_index": index, "config": spec, "seed_results": seed_results, "validation_pauc_mean": mean_val, "validation_pauc_std": std_val})
        score_cache[index] = np.column_stack(columns)
        print(f"{kind} candidate {index + 1}/{len(candidates)} V-pAUC={mean_val:.4f}±{std_val:.4f}", flush=True)
    selected = max(table, key=lambda row: (row["validation_pauc_mean"], -row["validation_pauc_std"], -row["candidate_index"]))
    return score_cache[int(selected["candidate_index"])], {"selected_config": selected["config"], "candidate_count": len(candidates), "candidate_table": table, "standardizer": {"mean": mean.tolist(), "std": std.tolist(), "fit_split": "D"}}


def _metric_summary(
    scores: np.ndarray,
    labels: np.ndarray,
    partitions: dict[str, np.ndarray],
    seeds: tuple[int, ...],
    bootstrap_repeats: int,
) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for column, seed in enumerate(seeds):
        result[str(seed)] = {"metrics": _bootstrap_metrics(scores[:, column], labels, partitions, bootstrap_repeats, 20261000 + seed % 1000)}
    return result


def _default_input(benchmark: str, epoch: int) -> Path:
    return ROOT / "experiments/results/sft_runs/full_delta" / f"{benchmark}_epoch{epoch}" / "draft_auxiliary_distilled" / "full_delta.npz"


def _default_output(benchmark: str, epoch: int, eos: str) -> Path:
    suffix = "" if eos == "with_eos" else "_no_eos"
    return ROOT / "experiments/results/sft_runs/accept_only" / f"{benchmark}_epoch{epoch}" / f"draft_auxiliary_distilled{suffix}"


def _args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--benchmark", choices=BENCHMARKS, required=True)
    parser.add_argument("--epoch", choices=EPOCHS, type=int, required=True)
    parser.add_argument("--input", type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--split-seed", type=int, default=DEFAULT_SPLIT_SEED)
    parser.add_argument("--training-seeds", default=",".join(str(v) for v in DEFAULT_TRAINING_SEEDS))
    parser.add_argument("--query-seed", type=int, default=20260909)
    parser.add_argument("--bootstrap-repeats", type=int, default=1000)
    parser.add_argument("--bins", type=int, default=64)
    parser.add_argument("--max-epochs", type=int, default=60)
    parser.add_argument("--patience", type=int, default=8)
    parser.add_argument("--drop-eos", action="store_true")
    parser.add_argument("--skip-sequence", action="store_true")
    return parser.parse_args()


def _drop_eos(labels: np.ndarray, ids: np.ndarray, lengths: np.ndarray, delta: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    if np.any(lengths <= 1):
        raise ValueError("cannot drop EOS from a one-token record")
    offsets = np.concatenate(([0], np.cumsum(lengths, dtype=np.int64)))
    pieces = [delta[int(start) : int(end) - 1] for start, end in zip(offsets[:-1], offsets[1:])]
    new_lengths = lengths - 1
    new_offsets = np.concatenate(([0], np.cumsum(new_lengths, dtype=np.int64)))
    return labels, ids, new_lengths, new_offsets, np.concatenate(pieces).astype(np.float32)


def main() -> None:
    args = _args()
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    seeds = tuple(int(value) for value in args.training_seeds.split(",") if value.strip())
    input_path = args.input or _default_input(args.benchmark, args.epoch)
    input_path = input_path if input_path.is_absolute() else ROOT / input_path
    labels, record_ids, lengths, offsets, delta = _load(input_path)
    if args.drop_eos:
        labels, record_ids, lengths, offsets, delta = _drop_eos(labels, record_ids, lengths, delta)
    if int(lengths.min()) < args.bins:
        raise ValueError(f"--bins={args.bins} exceeds minimum response length {int(lengths.min())}; choose the pre-registered smaller B")
    partitions = split_indices(labels, args.split_seed)
    output_dir = args.output_dir or _default_output(args.benchmark, args.epoch, "no_eos" if args.drop_eos else "with_eos")
    output_dir = output_dir if output_dir.is_absolute() else ROOT / output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)
    if device.type == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA requested but unavailable")
        torch.cuda.set_device(device)

    methods: dict[str, dict[str, Any]] = {}
    feature_manifest: dict[str, Any] = {}
    for queries in QUERY_COUNTS:
        started = time.perf_counter()
        stats, position = simulate_replay(delta, lengths, offsets, queries, args.query_seed, args.bins)
        np.savez_compressed(output_dir / f"accept_features_k{queries}.npz", labels=labels, record_ids=record_ids, stats=stats, position_bins=position)
        feature_manifest[f"K{queries}"] = {
            "stats_dim": list(stats.shape[1:]),
            "position_dim": list(position.shape[1:]),
            "query_count": queries,
            "query_seed": args.query_seed,
            "detector_input": "accept/reject-derived bits only",
            "seconds": time.perf_counter() - started,
        }
        stats_input = stats.mean(axis=1) if queries == 1 else np.concatenate((stats.mean(axis=1), stats.std(axis=1)), axis=1)
        # Logistic and MLP are the statistics baselines.  A fixed D-only
        # standardizer is fitted inside _run_stat_method below.
        for family in ("logistic", "mlp"):
            name = f"Accept-K{queries}-stat-{family}"
            from experiments.sd_membership_sft.archive.stat_delta_mia import (_run_method)
            result = _run_method(name, stats_input, labels, partitions, family, seeds, device, args.max_epochs, args.patience)
            scores = result.pop("scores")
            result["metrics"] = result["seed_metrics"][str(seeds[0])]["metrics"]
            result["query_count"] = queries
            result["detector_input"] = "accept/reject statistics only"
            methods[name] = result
            score_path = output_dir / f"scores_{name.lower().replace('-', '_')}.npz"
            np.savez_compressed(score_path, labels=labels, record_ids=record_ids, seeds=np.asarray(seeds), scores=scores)
            result["scores_path"] = str(score_path.resolve())

        if not args.skip_sequence:
            for kind, sequence in (("tcn", position[:, 0, :]), ("bit-transformer", position[:, 0, :])):
                name = f"Accept-K{queries}-bit-{kind}"
                scores, meta = _fit_sequence(sequence, labels, partitions, kind, seeds, device, args.max_epochs, args.patience)
                seed_metrics = _metric_summary(scores, labels, partitions, seeds, args.bootstrap_repeats)
                result = {"method": name, "permission_label": "strict accept-only", "family": kind, **meta, "query_count": queries, "detector_input": "fixed-position accept/reject bit rates only", "seed_metrics": seed_metrics, "metrics": seed_metrics[str(seeds[0])]["metrics"]}
                methods[name] = result
                score_path = output_dir / f"scores_{name.lower().replace('-', '_')}.npz"
                np.savez_compressed(score_path, labels=labels, record_ids=record_ids, seeds=np.asarray(seeds), scores=scores)
                result["scores_path"] = str(score_path.resolve())
            if queries > 1:
                name = f"Accept-K{queries}-query-transformer"
                scores, meta = _fit_sequence(position, labels, partitions, "query-transformer", seeds, device, args.max_epochs, args.patience)
                seed_metrics = _metric_summary(scores, labels, partitions, seeds, args.bootstrap_repeats)
                result = {"method": name, "permission_label": "strict accept-only", "family": "query-transformer", **meta, "query_count": queries, "detector_input": "repeated accept/reject matrix only", "seed_metrics": seed_metrics, "metrics": seed_metrics[str(seeds[0])]["metrics"]}
                methods[name] = result
                score_path = output_dir / f"scores_{name.lower().replace('-', '_')}.npz"
                np.savez_compressed(score_path, labels=labels, record_ids=record_ids, seeds=np.asarray(seeds), scores=scores)
                result["scores_path"] = str(score_path.resolve())

    report = {
        "protocol": {
            "benchmark": args.benchmark,
            "epoch": args.epoch,
            "role": "draft_auxiliary_distilled",
            "split_seed": args.split_seed,
            "training_seeds": list(seeds),
            "query_seed": args.query_seed,
            "bootstrap_repeats": args.bootstrap_repeats,
            "protocol": "position-locked fixed-candidate replay",
            "detector_contract": "accept/reject bits only; no p/q/delta/activation/token-id/length",
            "active_status": "not run: current cache lacks full logits/candidate proposal interface; no invalid approximation used",
            "natural_trajectory_status": "not part of the position-locked main result",
            "bins": args.bins,
            "eos_policy": "drop final cached token" if args.drop_eos else "as cached (main)",
        },
        "partitions": {name: [str(record_ids[index]) for index in indices] for name, indices in partitions.items()},
        "features": feature_manifest,
        "methods": methods,
    }
    (output_dir / "ACCEPT_ONLY_REPORT.json").write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    lines = [
        "# Accept-Only membership audit",
        "",
        f"- Condition: `{args.benchmark}_epoch{args.epoch}`",
        "- Protocol: position-locked fixed-candidate replay.",
        "- Detector input: accept/reject bits only; no p/q/delta/activation/token-id/length.",
        "- Active temperature/top-k/top-p: not run because the current cache lacks a full-logit candidate interface.",
        "",
        "| Method | V pAUC | T AUC | T pAUC | TPR@1% | TPR@10% |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for name, result in methods.items():
        metric = result["metrics"]
        test = metric["test"]
        lines.append(
            f"| {name} | {metric['validation_pauc_0_10']:.4f} | {test['auc']['point']:.4f} | {test['pauc_0_10']['point']:.4f} | {test['tpr_at_fpr']['1%']['tpr']['point']:.4f} | {test['tpr_at_fpr']['10%']['tpr']['point']:.4f} |"
        )
    (output_dir / "ACCEPT_ONLY_REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"output_dir": str(output_dir), "methods": len(methods)}, indent=2), flush=True)


if __name__ == "__main__":
    main()
