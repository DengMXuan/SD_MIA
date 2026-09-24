"""Fit the preregistered M2 method matrix on one condition.

Reads the shared M1 Q/H cache and probability archive, builds the M2
features, trains every registered method on the frozen D partition with V
pAUC selection, and freezes per-record logits for the conformal evaluation
in :mod:`m2_evaluate`.  C and T are never touched here.

Registered scope (M2 plan sections 5-9, 13):

- doc-level logistic family: B1, B2, BQ, BQ_noM, H_only, Direct, M2_F and
  the fixed ablations M2_F_nodiff / M2_F_plus / M2_F_minus / M2_F_noM;
- doc MLP: B2_mlp;
- token family: P_G_zero, P_G_wide, M2_U, M2_G, M2_G_T2 (4-config grid each);
- mechanism diagnostics for M2-F: paired (b+, b-) shuffle, within-doc H
  shuffle and 40-dim noise H, five fixed seeds each;
- stability seeds re-use the primary seed's selected hyperparameters.

Usage (one free GPU):

    CUDA_VISIBLE_DEVICES=2 uv run --no-sync python -m \
        experiments.sd_membership_sft.m2_fit \
        --benchmark wikitection --epoch 1 --role draft_auxiliary_distilled --device cuda:0
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any, Callable

import numpy as np
import torch
import torch.nn.functional as F

from experiments.sd_membership_sft.archive.m1_fit import partial_auc
from experiments.sd_membership_sft.archive.m2_features import M2Features, _u_pool, _w_pool, build_m2_features
from experiments.sd_membership_sft.archive.m2_models import DOC_BATCH, LOGISTIC_L2_GRID, NET_DROPOUTS, NET_WEIGHT_DECAYS, TokenBatcher, TokenGatedNet, DocMLP, fit_doc_standardizer, fit_logistic_grid, search_small_net, train_small_net
from experiments.sd_membership_sft.archive.m1_fit import load_m1_data, make_partitions
from experiments.shared.core.scoring_common import ROOT
from experiments.shared.training.training import set_seed

LOGISTIC_METHODS = (
    "B1",
    "B2",
    "BQ",
    "BQ_noM",
    "H_only",
    "Direct",
    "M2_F",
    "M2_F_nodiff",
    "M2_F_plus",
    "M2_F_minus",
    "M2_F_noM",
)
B0_COLUMNS = ("p_mean_logp", "mean_abs_delta", "window_sign_16", "window_sign_multiscale")
TOKEN_SPECS: dict[str, dict[str, Any]] = {
    "P_G_zero": {"hidden": 64, "use_token_h": True, "uniform_attention": False, "temperature": 1.0, "zero_h": True},
    "P_G_wide": {"hidden": 96, "use_token_h": False, "uniform_attention": False, "temperature": 1.0, "zero_h": False},
    "M2_U": {"hidden": 64, "use_token_h": True, "uniform_attention": True, "temperature": 1.0, "zero_h": False},
    "M2_G": {"hidden": 64, "use_token_h": True, "uniform_attention": False, "temperature": 1.0, "zero_h": False},
    "M2_G_T2": {"hidden": 64, "use_token_h": True, "uniform_attention": False, "temperature": 2.0, "zero_h": False},
}
SHUFFLE_SEEDS = (20260909, 20260910, 20260911, 20260912, 20260913)
DIFFICULTY_BINS = 3


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--benchmark", required=True)
    parser.add_argument("--epoch", type=int, required=True)
    parser.add_argument("--role", default="draft_auxiliary_distilled")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--seed", type=int, default=20260909)
    parser.add_argument("--stability-seeds", default="20260910,20260911")
    parser.add_argument("--skip-shuffles", action="store_true")
    parser.add_argument("--output-root", type=Path, default=None)
    return parser.parse_args()


def default_paths(benchmark: str, epoch: int, role: str) -> tuple[Path, Path, Path]:
    condition = f"{benchmark}_epoch{epoch}"
    feature_dir = ROOT / "experiments/results/sft_runs/m1_conditional" / condition / role
    probability_dir = ROOT / "experiments/results/sft_runs/pq_directional" / condition
    output_dir = (
        ROOT / "experiments/results/sft_runs/m2_activation_pooling" / condition / role
    )
    return feature_dir, probability_dir, output_dir


def difficulty_bucket_edges(draft_logq: np.ndarray, lengths: np.ndarray, detector_fit: np.ndarray) -> np.ndarray:
    """Tercile edges of D-token log q; EOS positions never swap buckets."""

    offsets = np.concatenate(([0], np.cumsum(lengths)))
    pieces = []
    for doc in detector_fit:
        start, end = int(offsets[doc]), int(offsets[doc + 1])
        pieces.append(draft_logq[start:end])
    values = np.concatenate(pieces)
    quantiles = np.linspace(0.0, 100.0, DIFFICULTY_BINS + 1)[1:-1]
    return np.percentile(values, quantiles)


def paired_weight_shuffle(
    features: M2Features,
    data: Any,
    edges: np.ndarray,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, dict[str, float]]:
    """Within-doc, within-difficulty-bucket paired (b+, b-) permutation."""

    lengths = features.lengths
    offsets = features.offsets
    eos_mask = np.asarray(data.eos_mask, dtype=bool)
    draft_logq = np.asarray(data.draft_logq, dtype=np.float64)
    b_plus = features.token["b_plus"].astype(np.float64).copy()
    b_minus = features.token["b_minus"].astype(np.float64).copy()
    swapped = 0
    total = 0
    for doc in range(len(lengths)):
        rng = np.random.default_rng([seed, doc])
        start, end = int(offsets[doc]), int(offsets[doc + 1])
        swappable = np.flatnonzero(~eos_mask[start:end])
        total += len(swappable)
        buckets = np.digitize(draft_logq[start + swappable], edges, right=False)
        for bucket in np.unique(buckets):
            members = swappable[buckets == bucket]
            if len(members) < 2:
                continue
            order = rng.permutation(len(members))
            b_plus[start + members] = features.token["b_plus"][start + members[order]]
            b_minus[start + members] = features.token["b_minus"][start + members[order]]
            swapped += len(members)
    diagnostics = {"swapped_token_fraction": swapped / max(total, 1)}
    return b_plus.astype(np.float32), b_minus.astype(np.float32), diagnostics


def within_doc_h_shuffle(
    features: M2Features,
    data: Any,
    edges: np.ndarray,
    seed: int,
) -> tuple[np.ndarray, dict[str, float]]:
    """Permute standardized H rows within doc and difficulty bucket."""

    lengths = features.lengths
    offsets = features.offsets
    eos_mask = np.asarray(data.eos_mask, dtype=bool)
    draft_logq = np.asarray(data.draft_logq, dtype=np.float64)
    h_source = features.token["h_std"]
    h_shuffled = h_source.copy()
    swapped = 0
    total = 0
    for doc in range(len(lengths)):
        rng = np.random.default_rng([seed + 1_000_000, doc])
        start, end = int(offsets[doc]), int(offsets[doc + 1])
        swappable = np.flatnonzero(~eos_mask[start:end])
        total += len(swappable)
        buckets = np.digitize(draft_logq[start + swappable], edges, right=False)
        for bucket in np.unique(buckets):
            members = swappable[buckets == bucket]
            if len(members) < 2:
                continue
            order = rng.permutation(len(members))
            h_shuffled[start + members] = h_source[start + members[order]]
            swapped += len(members)
    return h_shuffled, {"swapped_token_fraction": swapped / max(total, 1)}


def repool_w(
    features: M2Features,
    h_values: np.ndarray,
    b_plus: np.ndarray | None = None,
    b_minus: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Recompute U(H)/W(H) after an in-place token transformation."""

    offsets = features.offsets
    n_docs = len(features.lengths)
    if b_plus is None:
        b_plus = features.token["b_plus"]
    if b_minus is None:
        b_minus = features.token["b_minus"]
    u_h = np.empty((n_docs, 2 * h_values.shape[1]), dtype=np.float64)
    w_h = np.empty((n_docs, 3 * h_values.shape[1]), dtype=np.float64)
    for doc in range(n_docs):
        start, end = int(offsets[doc]), int(offsets[doc + 1])
        u_h[doc] = _u_pool(h_values[start:end])
        w_h[doc] = _w_pool(
            h_values[start:end].astype(np.float64),
            b_plus[start:end].astype(np.float64),
            b_minus[start:end].astype(np.float64),
        )
    return u_h, w_h


def fit_logistic_method(
    name: str,
    matrix: np.ndarray,
    labels: np.ndarray,
    detector_fit: np.ndarray,
    validation: np.ndarray,
) -> tuple[np.ndarray, dict[str, Any], np.ndarray, np.ndarray]:
    started = time.perf_counter()
    mean, std = fit_doc_standardizer(matrix, labels, detector_fit)
    standardized = (matrix - mean) / std
    model, search = fit_logistic_grid(standardized, labels, detector_fit, validation)
    scores = standardized @ model.weight + model.bias
    metadata = {
        "family": "logistic",
        "selected": search[
            max(
                range(len(search)),
                key=lambda i: (search[i]["validation_pauc_0_05"], search[i]["l2"]),
            )
        ],
        "search": search,
        "feature_dim": int(matrix.shape[1]),
        "seconds": time.perf_counter() - started,
    }
    return scores, metadata, mean, std


def make_doc_loss(
    standardized: np.ndarray, labels: np.ndarray, device: torch.device
) -> Callable[[Any, np.ndarray, np.ndarray], float]:
    matrix = torch.from_numpy(standardized.astype(np.float32)).to(device)
    targets = torch.from_numpy(labels.astype(np.float32)).to(device)

    def loss_batch(model: Any, member_docs: np.ndarray, nonmember_docs: np.ndarray) -> float:
        docs = np.concatenate((member_docs, nonmember_docs))
        logits = model(matrix[docs])
        is_member = targets[docs] == 1
        loss = 0.5 * F.binary_cross_entropy_with_logits(
            logits[is_member], torch.ones(int(is_member.sum()), device=device)
        ) + 0.5 * F.binary_cross_entropy_with_logits(
            logits[~is_member], torch.zeros(int((~is_member).sum()), device=device)
        )
        loss.backward()
        return float(loss.detach())

    return loss_batch


def make_doc_scorer(
    standardized: np.ndarray, device: torch.device, batch: int = 4096
) -> Callable[[Any, np.ndarray], np.ndarray]:
    matrix = torch.from_numpy(standardized.astype(np.float32)).to(device)

    def score(model: Any, docs: np.ndarray) -> np.ndarray:
        model.eval()
        outputs = []
        with torch.inference_mode():
            for start in range(0, len(docs), batch):
                chunk = np.asarray(docs)[start : start + batch]
                outputs.append(model(matrix[chunk]).cpu().numpy())
        return np.concatenate(outputs).astype(np.float64)

    return score


def make_token_loss(
    batcher: TokenBatcher,
    bq_standardized: np.ndarray,
    labels: np.ndarray,
    device: torch.device,
) -> Callable[[Any, np.ndarray, np.ndarray], float]:
    bq = torch.from_numpy(bq_standardized.astype(np.float32)).to(device)
    targets = torch.from_numpy(labels.astype(np.float32)).to(device)

    def loss_batch(model: Any, member_docs: np.ndarray, nonmember_docs: np.ndarray) -> float:
        docs = np.concatenate((member_docs, nonmember_docs))
        inputs_np, mask_np = batcher.encode(docs)
        inputs = torch.from_numpy(inputs_np).to(device)
        mask = torch.from_numpy(mask_np).to(device)
        logits = model(inputs, mask, bq[docs])
        is_member = targets[docs] == 1
        loss = 0.5 * F.binary_cross_entropy_with_logits(
            logits[is_member], torch.ones(int(is_member.sum()), device=device)
        ) + 0.5 * F.binary_cross_entropy_with_logits(
            logits[~is_member], torch.zeros(int((~is_member).sum()), device=device)
        )
        loss.backward()
        del inputs, mask, logits
        return float(loss.detach())

    return loss_batch


def make_token_scorer(
    batcher: TokenBatcher,
    bq_standardized: np.ndarray,
    device: torch.device,
    batch: int = 32,
) -> Callable[[Any, np.ndarray], np.ndarray]:
    bq = torch.from_numpy(bq_standardized.astype(np.float32)).to(device)

    def score(model: Any, docs: np.ndarray) -> np.ndarray:
        model.eval()
        docs = np.asarray(docs, dtype=np.int64)
        order = np.argsort(-batcher.lengths[docs], kind="mergesort")
        outputs = np.empty(len(docs), dtype=np.float64)
        with torch.inference_mode():
            for start in range(0, len(order), batch):
                chunk = order[start : start + batch]
                doc_chunk = docs[chunk]
                inputs_np, mask_np = batcher.encode(doc_chunk)
                logits = model(
                    torch.from_numpy(inputs_np).to(device),
                    torch.from_numpy(mask_np).to(device),
                    bq[doc_chunk],
                )
                outputs[chunk] = logits.cpu().numpy()
        return outputs

    return score


def collect_attention_stats(
    model: Any,
    batcher: TokenBatcher,
    docs: np.ndarray,
    device: torch.device,
    batch: int = 32,
) -> dict[str, np.ndarray]:
    """Batched attention-concentration diagnostics over the given documents."""

    docs = np.asarray(docs, dtype=np.int64)
    order = np.argsort(-batcher.lengths[docs], kind="mergesort")
    collected: dict[str, list[np.ndarray]] = {}
    with torch.inference_mode():
        for start in range(0, len(order), batch):
            chunk = order[start : start + batch]
            doc_chunk = docs[chunk]
            inputs_np, mask_np = batcher.encode(doc_chunk)
            stats = model.attention_stats(
                torch.from_numpy(inputs_np).to(device),
                torch.from_numpy(mask_np).to(device),
            )
            for key, values in stats.items():
                collected.setdefault(key, []).append(values)
    return {key: np.concatenate(values) for key, values in collected.items()}


def main() -> None:
    args = parse_args()
    feature_dir, probability_dir, output_dir = default_paths(
        args.benchmark, args.epoch, args.role
    )
    if args.output_root is not None:
        root = args.output_root if args.output_root.is_absolute() else ROOT / args.output_root
        output_dir = root / f"{args.benchmark}_epoch{args.epoch}" / args.role
    output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)
    if device.type == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA requested for M2 fitting but unavailable")
        torch.cuda.set_device(device)

    set_seed(args.seed)
    data = load_m1_data(feature_dir, probability_dir, args.role)
    frozen_manifest = feature_dir / "partition_manifest.json"
    partitions = make_partitions(
        data.labels, data.record_ids, frozen_manifest_path=frozen_manifest
    )
    detector_fit = np.asarray(partitions["detector_fit"], dtype=np.int64)
    validation = np.asarray(partitions["validation"], dtype=np.int64)
    labels = np.asarray(data.labels, dtype=np.int64)

    started = time.perf_counter()
    features = build_m2_features(data, partitions)
    feature_seconds = time.perf_counter() - started

    frozen: dict[str, np.ndarray] = {}
    metadata: dict[str, Any] = {"methods": {}, "diagnostics": {}, "timing": {}}

    lookup = {name: index for index, name in enumerate(features.f19_names)}
    for column in B0_COLUMNS:
        frozen[f"B0__{column}"] = features.f19[:, lookup[column]].astype(np.float64)
    metadata["methods"]["B0"] = {"family": "fixed_scores", "columns": list(B0_COLUMNS)}

    for name in LOGISTIC_METHODS:
        scores, method_meta, _mean, _std = fit_logistic_method(
            name, features.doc_matrix(name), labels, detector_fit, validation
        )
        frozen[name] = scores
        metadata["methods"][name] = method_meta
        print(
            f"{name}: V pAUC {method_meta['selected']['validation_pauc_0_05']:.4f} "
            f"(l2={method_meta['selected']['l2']})",
            flush=True,
        )

    bq_mean, bq_std = fit_doc_standardizer(features.bq, labels, detector_fit)
    bq_standardized = (features.bq - bq_mean) / bq_std

    # Doc MLP on F41.
    f41_mean, f41_std = fit_doc_standardizer(features.f41, labels, detector_fit)
    f41_standardized = (features.f41 - f41_mean) / f41_std
    started = time.perf_counter()
    model, selected, search = search_small_net(
        lambda config: DocMLP(features.f41.shape[1], config["dropout"]),
        make_doc_loss(f41_standardized, labels, device),
        make_doc_scorer(f41_standardized, device),
        detector_fit,
        labels,
        validation,
        labels[validation],
        device,
        args.seed,
        [
            {"weight_decay": wd, "dropout": do}
            for wd in NET_WEIGHT_DECAYS
            for do in NET_DROPOUTS
        ],
    )
    metadata["timing"]["B2_mlp_seconds"] = time.perf_counter() - started
    frozen["B2_mlp"] = make_doc_scorer(f41_standardized, device)(model, np.arange(len(labels)))
    metadata["methods"]["B2_mlp"] = {
        "family": "doc_mlp",
        "selected": selected,
        "search": search,
        "feature_dim": int(features.f41.shape[1]),
    }
    print(f"B2_mlp: V pAUC {selected['validation_pauc_0_05']:.4f}", flush=True)

    # Token family.
    batchers = {
        "full": TokenBatcher(
            lengths=features.lengths,
            offsets=features.offsets,
            q_std=features.token["q_std"],
            h_std=features.token["h_std"],
            l_std=features.token["l_std"],
            evidence=features.token["e"],
            b_plus=features.token["b_plus"],
            b_minus=features.token["b_minus"],
            use_token_h=True,
            zero_h=False,
        ),
        "zero": TokenBatcher(
            lengths=features.lengths,
            offsets=features.offsets,
            q_std=features.token["q_std"],
            h_std=features.token["h_std"],
            l_std=features.token["l_std"],
            evidence=features.token["e"],
            b_plus=features.token["b_plus"],
            b_minus=features.token["b_minus"],
            use_token_h=True,
            zero_h=True,
        ),
        "prob": TokenBatcher(
            lengths=features.lengths,
            offsets=features.offsets,
            q_std=features.token["q_std"],
            h_std=features.token["h_std"],
            l_std=features.token["l_std"],
            evidence=features.token["e"],
            b_plus=features.token["b_plus"],
            b_minus=features.token["b_minus"],
            use_token_h=False,
            zero_h=False,
        ),
    }
    token_loss = make_token_loss(batchers["full"], bq_standardized, labels, device)
    token_scorer = make_token_scorer(batchers["full"], bq_standardized, device)
    zero_loss = make_token_loss(batchers["zero"], bq_standardized, labels, device)
    zero_scorer = make_token_scorer(batchers["zero"], bq_standardized, device)
    prob_loss = make_token_loss(batchers["prob"], bq_standardized, labels, device)
    prob_scorer = make_token_scorer(batchers["prob"], bq_standardized, device)
    config_grid = [
        {"weight_decay": wd, "dropout": do}
        for wd in NET_WEIGHT_DECAYS
        for do in NET_DROPOUTS
    ]
    all_docs = np.arange(len(labels))
    for name, spec in TOKEN_SPECS.items():
        started = time.perf_counter()
        if spec["zero_h"]:
            loss_batch, score_fn = zero_loss, zero_scorer
        elif not spec["use_token_h"]:
            loss_batch, score_fn = prob_loss, prob_scorer
        else:
            loss_batch, score_fn = token_loss, token_scorer
        model, selected, search = search_small_net(
            lambda config, spec=spec: TokenGatedNet(
                bq_dim=features.bq.shape[1],
                hidden=spec["hidden"],
                use_token_h=spec["use_token_h"],
                uniform_attention=spec["uniform_attention"],
                temperature=spec["temperature"],
                dropout=config["dropout"],
            ),
            loss_batch,
            score_fn,
            detector_fit,
            labels,
            validation,
            labels[validation],
            device,
            args.seed,
            config_grid,
        )
        metadata["timing"][f"{name}_seconds"] = time.perf_counter() - started
        frozen[name] = score_fn(model, all_docs)
        method_meta = {
            "family": "token_gated",
            "selected": selected,
            "search": search,
            "hidden": spec["hidden"],
            "temperature": spec["temperature"],
            "uniform_attention": spec["uniform_attention"],
        }
        if name in ("M2_G", "M2_G_T2"):
            stats = collect_attention_stats(
                model, batchers["full"], np.asarray(partitions["test"], dtype=np.int64), device
            )
            method_meta["attention_diagnostics"] = {
                key: {
                    "mean": float(np.mean(values)),
                    "q10": float(np.quantile(values, 0.10)),
                    "q90": float(np.quantile(values, 0.90)),
                }
                for key, values in stats.items()
            }
        metadata["methods"][name] = method_meta
        print(
            f"{name}: V pAUC {selected['validation_pauc_0_05']:.4f} "
            f"(wd={selected['weight_decay']}, dropout={selected['dropout']})",
            flush=True,
        )

    # Stability seeds re-use the primary seed's selected hyperparameters.
    stability_seeds = [int(value) for value in args.stability_seeds.split(",") if value.strip()]
    for seed in stability_seeds:
        set_seed(seed)
        model = DocMLP(features.f41.shape[1], metadata["methods"]["B2_mlp"]["selected"]["dropout"])
        summary = train_small_net(
            model,
            make_doc_loss(f41_standardized, labels, device),
            make_doc_scorer(f41_standardized, device),
            detector_fit,
            labels,
            validation,
            labels[validation],
            device,
            seed,
            weight_decay=metadata["methods"]["B2_mlp"]["selected"]["weight_decay"],
        )
        frozen[f"B2_mlp__seed{seed}"] = make_doc_scorer(f41_standardized, device)(model, all_docs)
        for name, spec in TOKEN_SPECS.items():
            if spec["zero_h"]:
                loss_batch, score_fn = zero_loss, zero_scorer
            elif not spec["use_token_h"]:
                loss_batch, score_fn = prob_loss, prob_scorer
            else:
                loss_batch, score_fn = token_loss, token_scorer
            model = TokenGatedNet(
                bq_dim=features.bq.shape[1],
                hidden=spec["hidden"],
                use_token_h=spec["use_token_h"],
                uniform_attention=spec["uniform_attention"],
                temperature=spec["temperature"],
                dropout=metadata["methods"][name]["selected"]["dropout"],
            )
            train_small_net(
                model,
                loss_batch,
                score_fn,
                detector_fit,
                labels,
                validation,
                labels[validation],
                device,
                seed,
                weight_decay=metadata["methods"][name]["selected"]["weight_decay"],
            )
            frozen[f"{name}__seed{seed}"] = score_fn(model, all_docs)
        print(f"stability seed {seed}: done", flush=True)

    # Mechanism diagnostics for M2-F (cheap logistic refits).
    if not args.skip_shuffles:
        edges = difficulty_bucket_edges(np.asarray(data.draft_logq, dtype=np.float64), features.lengths, detector_fit)
        shuffle_results: dict[str, Any] = {}
        for seed in SHUFFLE_SEEDS:
            # 9.2 paired (b+, b-) shuffle.
            b_plus_s, b_minus_s, diag = paired_weight_shuffle(features, data, edges, seed)
            _u_h_keep, w_h_shuffled = repool_w(features, features.token["h_std"].astype(np.float64), b_plus_s, b_minus_s)
            matrix = np.concatenate((features.bq, features.u_h, w_h_shuffled), axis=1)
            scores, method_meta, _m, _s = fit_logistic_method(
                f"M2_F_wshuffle_s{seed}", matrix, labels, detector_fit, validation
            )
            frozen[f"M2_F_wshuffle__seed{seed}"] = scores
            shuffle_results[f"weight_shuffle_seed{seed}"] = {"diag": diag, "v_pauc": method_meta["selected"]["validation_pauc_0_05"]}
            # 9.3 within-doc H shuffle.
            h_shuffled, diag_h = within_doc_h_shuffle(features, data, edges, seed)
            _u_h_s, w_h_h = repool_w(features, h_shuffled.astype(np.float64))
            matrix = np.concatenate((features.bq, features.u_h, w_h_h), axis=1)
            scores, method_meta, _m, _s = fit_logistic_method(
                f"M2_F_hshuffle_s{seed}", matrix, labels, detector_fit, validation
            )
            frozen[f"M2_F_hshuffle__seed{seed}"] = scores
            shuffle_results[f"h_shuffle_seed{seed}"] = {"diag": diag_h, "v_pauc": method_meta["selected"]["validation_pauc_0_05"]}
            # 9.3 noise H.
            rng = np.random.default_rng([seed + 2_000_000, 7])
            h_noise = rng.standard_normal(features.token["h_std"].shape, dtype=np.float32)
            u_h_noise, w_h_noise = repool_w(features, h_noise.astype(np.float64))
            matrix = np.concatenate((features.bq, u_h_noise, w_h_noise), axis=1)
            scores, method_meta, _m, _s = fit_logistic_method(
                f"M2_F_noise_s{seed}", matrix, labels, detector_fit, validation
            )
            frozen[f"M2_F_noise__seed{seed}"] = scores
            shuffle_results[f"noise_seed{seed}"] = {"v_pauc": method_meta["selected"]["validation_pauc_0_05"]}
            print(f"shuffle seed {seed}: done", flush=True)
        metadata["diagnostics"]["m2_f_mechanism"] = shuffle_results
        metadata["diagnostics"]["difficulty_bucket_edges"] = edges.tolist()

    metadata["timing"]["feature_seconds"] = feature_seconds
    metadata["timing"]["total_seconds"] = sum(
        value for value in metadata["timing"].values() if isinstance(value, (int, float))
    )
    metadata["protocol"] = {
        "benchmark": args.benchmark,
        "epoch": args.epoch,
        "role": args.role,
        "feature_dir": str(feature_dir),
        "probability_dir": str(probability_dir),
        "frozen_partition_manifest": str(frozen_manifest),
        "seed": args.seed,
        "stability_seeds": stability_seeds,
        "shuffle_seeds": list(SHUFFLE_SEEDS),
        "logistic_l2_grid": list(LOGISTIC_L2_GRID),
        "net_grid": {"weight_decay": list(NET_WEIGHT_DECAYS), "dropout": list(NET_DROPOUTS)},
        "doc_batch": DOC_BATCH,
        "delta_scale": features.diagnostics["delta_scale"],
        "feature_diagnostics": features.diagnostics,
        "n_member": int(np.sum(labels == 1)),
        "n_nonmember": int(np.sum(labels == 0)),
    }

    np.savez_compressed(
        output_dir / "frozen_scores.npz",
        labels=labels,
        record_ids=np.asarray(data.record_ids),
        **{key: value.astype(np.float64) for key, value in frozen.items()},
    )
    (output_dir / "m2_fit_artifacts.json").write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    (output_dir / "experiment_spec.json").write_text(
        json.dumps(
            {
                "plan": "research/M2概率证据引导激活聚合_Qwen3已微调模型实验方案_2026-09-09.md",
                "scope": "P0 conditions, q_aux role, independent M2-F/M2-G with mechanism diagnostics",
                "not_run_this_round": [
                    "q_member boundary role",
                    "M1 residual combination (section 10)",
                    "epoch transfer and projection extensions (section 12)",
                    "gated-family noise/shuffle retrains",
                ],
                **metadata["protocol"],
            },
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    print(json.dumps({"output_dir": str(output_dir)}, indent=2))


if __name__ == "__main__":
    main()
