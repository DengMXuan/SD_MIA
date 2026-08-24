#!/usr/bin/env python3
"""Minimal, controlled membership-audit pilot for edge-cloud speculative decoding.

The cloud verifier is emulated locally, but transcript-derived audit scores only
consume accept/reject bits plus quantities available from the client-side draft.
No timing, packet size, or other side-channel signal is used.
"""

from __future__ import annotations

import argparse
import copy
import gc
import hashlib
import json
import math
import random
import re
import subprocess
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer, GPT2LMHeadModel


MODEL_SNAPSHOT = Path(
    "/home/mxd/.cache/huggingface/hub/"
    "models--openai-community--gpt2-xl/snapshots/"
    "15ea56dee5df4983c59b2538573817e1667135e2"
)


@dataclass
class Config:
    seed: int = 20260824
    gpu: int = 1
    seq_len: int = 64
    n_per_class: int = 160
    n_aux: int = 160
    audit_train_per_class: int = 48
    target_epochs: int = 8
    target_batch_size: int = 8
    target_lr: float = 5e-5
    draft_layers: int = 8
    distill_steps: int = 80
    distill_batch_size: int = 8
    distill_lr: float = 8e-5
    distill_temperature: float = 2.0
    min_k_fraction: float = 0.20
    transcript_repeats: int = 24
    transcript_levels: int = 5
    bootstrap_repeats: int = 500


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--gpu", type=int, default=1)
    p.add_argument("--seed", type=int, default=20260824)
    p.add_argument("--seq-len", type=int, default=64)
    p.add_argument("--n-per-class", type=int, default=160)
    p.add_argument("--n-aux", type=int, default=160)
    p.add_argument("--audit-train-per-class", type=int, default=48)
    p.add_argument("--target-epochs", type=int, default=8)
    p.add_argument("--target-batch-size", type=int, default=8)
    p.add_argument("--target-lr", type=float, default=5e-5)
    p.add_argument("--draft-layers", type=int, default=8)
    p.add_argument("--distill-steps", type=int, default=80)
    p.add_argument("--distill-batch-size", type=int, default=8)
    p.add_argument("--distill-lr", type=float, default=8e-5)
    p.add_argument("--distill-temperature", type=float, default=2.0)
    p.add_argument("--min-k-fraction", type=float, default=0.20)
    p.add_argument("--transcript-repeats", type=int, default=24)
    p.add_argument("--transcript-levels", type=int, default=5)
    p.add_argument("--bootstrap-repeats", type=int, default=500)
    p.add_argument("--output-dir", type=Path, default=Path("experiments/results/pilot"))
    return p.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def clean_pdf_text(raw: str) -> list[str]:
    raw = raw.replace("\f", "\n\n")
    raw = re.sub(r"-\n(?=[a-z])", "", raw)
    raw = re.sub(r"(?<!\n)\n(?!\n)", " ", raw)
    paragraphs = re.split(r"\n\s*\n+", raw)
    keep: list[str] = []
    for para in paragraphs:
        para = re.sub(r"\s+", " ", para).strip()
        if not (260 <= len(para) <= 5000):
            continue
        printable = sum(ch.isalpha() or ch.isspace() or ch in ".,;:()[]-'" for ch in para)
        if printable / max(1, len(para)) < 0.78:
            continue
        if para.lower().startswith(("references ", "acknowledg", "appendix ")):
            continue
        keep.append(para)
    return keep


def build_controlled_split(
    root: Path,
    tokenizer: Any,
    cfg: Config,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, Any]]:
    pdf_dir = root / "papers" / "edge-cloud-speculative-decoding"
    pdfs = sorted(p for p in pdf_dir.glob("*.pdf") if p.is_file())
    if not pdfs:
        raise RuntimeError(f"No source PDFs under {pdf_dir}")

    by_doc: dict[str, list[list[int]]] = {}
    seen: set[str] = set()
    for pdf in pdfs:
        proc = subprocess.run(
            ["pdftotext", "-layout", str(pdf), "-"],
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        chunks: list[list[int]] = []
        for para in clean_pdf_text(proc.stdout.decode("utf-8", errors="ignore")):
            ids = tokenizer(para, add_special_tokens=False).input_ids
            # Non-overlapping chunks prevent a held-out record from being a
            # shifted duplicate of a training record.
            for start in range(0, len(ids) - cfg.seq_len + 1, cfg.seq_len):
                chunk = ids[start : start + cfg.seq_len]
                digest = hashlib.sha256(np.asarray(chunk, dtype=np.int32).tobytes()).hexdigest()
                if digest not in seen:
                    seen.add(digest)
                    chunks.append(chunk)
        by_doc[pdf.name] = chunks

    members: list[list[int]] = []
    nonmembers: list[list[int]] = []
    auxiliary: list[list[int]] = []
    allocation: dict[str, dict[str, int]] = {}
    for doc_idx, (name, chunks) in enumerate(by_doc.items()):
        rng = random.Random(cfg.seed + 1009 * (doc_idx + 1))
        rng.shuffle(chunks)
        doc_counts = {"member": 0, "nonmember": 0, "auxiliary": 0}
        # Cycling after a document-specific shuffle stratifies all three sets
        # by source document and makes membership independent of document/date.
        for idx, chunk in enumerate(chunks):
            bucket = idx % 3
            if bucket == 0:
                members.append(chunk)
                doc_counts["member"] += 1
            elif bucket == 1:
                nonmembers.append(chunk)
                doc_counts["nonmember"] += 1
            else:
                auxiliary.append(chunk)
                doc_counts["auxiliary"] += 1
        allocation[name] = doc_counts

    rng = random.Random(cfg.seed + 77)
    rng.shuffle(members)
    rng.shuffle(nonmembers)
    rng.shuffle(auxiliary)
    if len(members) < cfg.n_per_class or len(nonmembers) < cfg.n_per_class:
        raise RuntimeError(
            f"Insufficient controlled records: {len(members)} members, "
            f"{len(nonmembers)} nonmembers"
        )
    if len(auxiliary) < cfg.n_aux:
        raise RuntimeError(f"Insufficient auxiliary records: {len(auxiliary)}")

    members_arr = np.asarray(members[: cfg.n_per_class], dtype=np.int64)
    nonmembers_arr = np.asarray(nonmembers[: cfg.n_per_class], dtype=np.int64)
    aux_arr = np.asarray(auxiliary[: cfg.n_aux], dtype=np.int64)
    metadata = {
        "source_pdf_count": len(pdfs),
        "unique_chunk_count": len(seen),
        "available_counts": {
            "member": len(members),
            "nonmember": len(nonmembers),
            "auxiliary": len(auxiliary),
        },
        "per_document_allocation": allocation,
        "raw_text_persisted": False,
    }
    return members_arr, nonmembers_arr, aux_arr, metadata


def balanced_indices(n_per_class: int, n_train: int, seed: int) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    pos = rng.permutation(n_per_class)
    neg = rng.permutation(n_per_class) + n_per_class
    train = np.concatenate([pos[:n_train], neg[:n_train]])
    test = np.concatenate([pos[n_train:], neg[n_train:]])
    rng.shuffle(train)
    rng.shuffle(test)
    return train, test


def fine_tune_target(
    model: torch.nn.Module,
    member_ids: np.ndarray,
    cfg: Config,
    device: torch.device,
) -> list[float]:
    model.train()
    model.config.use_cache = False
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.target_lr, fused=True)
    history: list[float] = []
    rng = np.random.default_rng(cfg.seed + 10)
    for epoch in range(cfg.target_epochs):
        losses: list[float] = []
        order = rng.permutation(len(member_ids))
        for start in range(0, len(order), cfg.target_batch_size):
            batch_np = member_ids[order[start : start + cfg.target_batch_size]]
            batch = torch.as_tensor(batch_np, device=device, dtype=torch.long)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                loss = model(input_ids=batch, labels=batch, use_cache=False).loss
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            losses.append(float(loss.detach()))
        history.append(float(np.mean(losses)))
        print(f"target epoch {epoch + 1}/{cfg.target_epochs}: loss={history[-1]:.5f}", flush=True)
    del optimizer
    model.eval()
    gc.collect()
    torch.cuda.empty_cache()
    return history


@torch.no_grad()
def extract_features(
    model: torch.nn.Module,
    ids: np.ndarray,
    device: torch.device,
    batch_size: int = 8,
) -> dict[str, np.ndarray]:
    model.eval()
    token_logp: list[np.ndarray] = []
    entropy: list[np.ndarray] = []
    top1_match: list[np.ndarray] = []
    hidden_mean: list[np.ndarray] = []
    grad_proxy: list[np.ndarray] = []
    for start in range(0, len(ids), batch_size):
        batch = torch.as_tensor(ids[start : start + batch_size], device=device, dtype=torch.long)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            out = model(input_ids=batch, output_hidden_states=True, use_cache=False)
        logits = out.logits[:, :-1].float()
        labels = batch[:, 1:]
        log_probs = logits.log_softmax(dim=-1)
        probs = log_probs.exp()
        lp = log_probs.gather(-1, labels.unsqueeze(-1)).squeeze(-1)
        ent = -(probs * log_probs).sum(dim=-1)
        match = logits.argmax(dim=-1).eq(labels)
        hidden = out.hidden_states[-1][:, :-1].float()
        # Exact norm of the per-token gradient w.r.t. the LM-head weight is
        # ||h||_2 * ||softmax(z)-onehot(y)||_2. It is available from a white-box
        # draft without materializing a vocabulary-by-hidden gradient tensor.
        prob_sq = probs.square().sum(dim=-1)
        py = probs.gather(-1, labels.unsqueeze(-1)).squeeze(-1)
        simplex_grad_norm = (prob_sq - 2.0 * py + 1.0).clamp_min(0).sqrt()
        gp = hidden.norm(dim=-1) * simplex_grad_norm

        token_logp.append(lp.cpu().numpy())
        entropy.append(ent.cpu().numpy())
        top1_match.append(match.float().cpu().numpy())
        hidden_mean.append(hidden.mean(dim=1).cpu().numpy())
        grad_proxy.append(gp.cpu().numpy())
        del out, logits, log_probs, probs, hidden
    return {
        "token_logp": np.concatenate(token_logp, axis=0),
        "entropy": np.concatenate(entropy, axis=0),
        "top1_match": np.concatenate(top1_match, axis=0),
        "hidden_mean": np.concatenate(hidden_mean, axis=0),
        "grad_proxy": np.concatenate(grad_proxy, axis=0),
    }


def truncated_config(full_config: Any, layers: int) -> Any:
    cfg = copy.deepcopy(full_config)
    cfg.n_layer = layers
    cfg.use_cache = False
    return cfg


@torch.no_grad()
def copy_truncated_target(target: GPT2LMHeadModel, layers: int, device: torch.device) -> GPT2LMHeadModel:
    draft = GPT2LMHeadModel(truncated_config(target.config, layers))
    draft.to(device=device, dtype=torch.bfloat16)
    draft.transformer.wte.weight.copy_(target.transformer.wte.weight)
    draft.transformer.wpe.weight.copy_(target.transformer.wpe.weight)
    draft.transformer.ln_f.load_state_dict(target.transformer.ln_f.state_dict())
    for idx in range(layers):
        draft.transformer.h[idx].load_state_dict(target.transformer.h[idx].state_dict())
    draft.tie_weights()
    draft.eval()
    return draft


def load_base_draft(target_config: Any, layers: int, device: torch.device) -> GPT2LMHeadModel:
    cfg = truncated_config(target_config, layers)
    draft = GPT2LMHeadModel.from_pretrained(
        MODEL_SNAPSHOT,
        config=cfg,
        torch_dtype=torch.bfloat16,
        local_files_only=True,
        low_cpu_mem_usage=True,
    )
    draft.to(device)
    draft.config.use_cache = False
    draft.eval()
    return draft


def distill_on_auxiliary(
    draft: GPT2LMHeadModel,
    target: GPT2LMHeadModel,
    aux_ids: np.ndarray,
    cfg: Config,
    device: torch.device,
) -> list[float]:
    if cfg.distill_steps <= 0:
        return []
    draft.train()
    target.eval()
    optimizer = torch.optim.AdamW(draft.parameters(), lr=cfg.distill_lr, fused=True)
    rng = np.random.default_rng(cfg.seed + 20)
    losses: list[float] = []
    temperature = cfg.distill_temperature
    for step in range(cfg.distill_steps):
        chosen = rng.integers(0, len(aux_ids), size=cfg.distill_batch_size)
        batch = torch.as_tensor(aux_ids[chosen], device=device, dtype=torch.long)
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
            teacher_logits = target(input_ids=batch, use_cache=False).logits[:, :-1].float()
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            student_logits = draft(input_ids=batch, use_cache=False).logits[:, :-1].float()
            labels = batch[:, 1:]
            ce = F.cross_entropy(student_logits.reshape(-1, student_logits.size(-1)), labels.reshape(-1))
            kl = F.kl_div(
                F.log_softmax(student_logits / temperature, dim=-1),
                F.softmax(teacher_logits / temperature, dim=-1),
                reduction="batchmean",
            ) * (temperature**2) / labels.shape[1]
            loss = 0.20 * ce + 0.80 * kl
        loss.backward()
        torch.nn.utils.clip_grad_norm_(draft.parameters(), 1.0)
        optimizer.step()
        losses.append(float(loss.detach()))
        if (step + 1) % max(1, cfg.distill_steps // 4) == 0:
            print(f"distill step {step + 1}/{cfg.distill_steps}: loss={losses[-1]:.5f}", flush=True)
        del teacher_logits, student_logits
    del optimizer
    draft.eval()
    gc.collect()
    torch.cuda.empty_cache()
    return losses


def bottom_k_indices(token_logp: np.ndarray, fraction: float) -> np.ndarray:
    k = max(1, int(math.ceil(token_logp.shape[1] * fraction)))
    return np.argpartition(token_logp, kth=k - 1, axis=1)[:, :k]


def gather_positions(values: np.ndarray, positions: np.ndarray) -> np.ndarray:
    return np.take_along_axis(values, positions, axis=1)


def transcript_tomography(
    target_logp: np.ndarray,
    draft_logp: np.ndarray,
    selected: np.ndarray,
    repeats: int,
    levels: int,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Estimate selected target token probabilities from accept/reject bits.

    For each token, the client starts from its white-box q and raises the
    candidate-token mass geometrically. At level l, the verifier's intended
    response is Bernoulli(min(1, p/q_l)). The estimator only receives the
    binomial counts and the client-known q_l values.
    """
    rng = np.random.default_rng(seed)
    p = np.exp(gather_positions(target_logp, selected)).astype(np.float64)
    q0 = np.exp(gather_positions(draft_logp, selected)).astype(np.float64)
    estimates = np.empty_like(p)
    query_bits = np.zeros(p.shape[0], dtype=np.int64)
    honest_accept = np.empty_like(p)

    for row in range(p.shape[0]):
        for col in range(p.shape[1]):
            pv = float(p[row, col])
            q_base = float(np.clip(q0[row, col], 1e-7, 0.95))
            honest_alpha = min(1.0, pv / q_base)
            honest_count = rng.binomial(repeats, honest_alpha)
            honest_accept[row, col] = (honest_count + 0.5) / (repeats + 1.0)

            chosen_estimate = q_base
            best_distance = float("inf")
            for level in range(levels):
                q_probe = float(min(0.95, q_base * (2.0**level)))
                alpha = min(1.0, pv / q_probe)
                count = int(rng.binomial(repeats, alpha))
                query_bits[row] += repeats
                smoothed = (count + 0.5) / (repeats + 1.0)
                estimate = q_probe * smoothed
                distance = abs(smoothed - 0.5)
                if distance < best_distance:
                    best_distance = distance
                    chosen_estimate = estimate
                # An unsaturated acceptance rate is locally identifiable. Stop
                # escalating once it is neither all-accept nor nearly zero.
                if 0 < count < repeats and smoothed <= 0.80:
                    chosen_estimate = estimate
                    break
                if q_probe >= 0.95:
                    chosen_estimate = estimate
                    break
            estimates[row, col] = float(np.clip(chosen_estimate, 1e-12, 1.0))
    return np.log(estimates), honest_accept, query_bits


def auc_rank(y: np.ndarray, scores: np.ndarray) -> float:
    y = np.asarray(y, dtype=np.int64)
    scores = np.asarray(scores, dtype=np.float64)
    order = np.argsort(scores, kind="mergesort")
    sorted_scores = scores[order]
    ranks = np.empty(len(scores), dtype=np.float64)
    i = 0
    while i < len(scores):
        j = i + 1
        while j < len(scores) and sorted_scores[j] == sorted_scores[i]:
            j += 1
        ranks[order[i:j]] = (i + 1 + j) / 2.0
        i = j
    n_pos = int(y.sum())
    n_neg = len(y) - n_pos
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    return float((ranks[y == 1].sum() - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg))


def tpr_at_fpr(y: np.ndarray, scores: np.ndarray, target_fpr: float) -> float:
    pos = np.asarray(scores)[np.asarray(y) == 1]
    neg = np.sort(np.asarray(scores)[np.asarray(y) == 0])[::-1]
    allowed_fp = int(math.floor(target_fpr * len(neg)))
    if allowed_fp >= len(neg):
        threshold = -np.inf
    else:
        # Use the first excluded negative score, shifted upward by one ULP.
        # This conservative handling of ties guarantees that the empirical
        # FPR never exceeds the requested budget for discrete transcript scores.
        threshold = np.nextafter(neg[allowed_fp], np.inf)
    return float(np.mean(pos >= threshold))


def bootstrap_auc_ci(y: np.ndarray, scores: np.ndarray, repeats: int, seed: int) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    pos = np.flatnonzero(y == 1)
    neg = np.flatnonzero(y == 0)
    values = []
    for _ in range(repeats):
        idx = np.concatenate([rng.choice(pos, len(pos), replace=True), rng.choice(neg, len(neg), replace=True)])
        values.append(auc_rank(y[idx], scores[idx]))
    return float(np.quantile(values, 0.025)), float(np.quantile(values, 0.975))


def fit_logistic(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_test: np.ndarray,
    steps: int = 1200,
    lr: float = 0.08,
    l2: float = 0.02,
) -> np.ndarray:
    x_train = np.asarray(x_train, dtype=np.float64)
    x_test = np.asarray(x_test, dtype=np.float64)
    mean = x_train.mean(axis=0, keepdims=True)
    std = x_train.std(axis=0, keepdims=True) + 1e-6
    train = np.concatenate([(x_train - mean) / std, np.ones((len(x_train), 1))], axis=1)
    test = np.concatenate([(x_test - mean) / std, np.ones((len(x_test), 1))], axis=1)
    w = np.zeros(train.shape[1], dtype=np.float64)
    y_float = y_train.astype(np.float64)
    for _ in range(steps):
        logits = np.clip(train @ w, -30, 30)
        probs = 1.0 / (1.0 + np.exp(-logits))
        grad = train.T @ (probs - y_float) / len(train)
        grad[:-1] += l2 * w[:-1]
        w -= lr * grad
    return 1.0 / (1.0 + np.exp(-np.clip(test @ w, -30, 30)))


def hashed_bow(ids: np.ndarray, width: int = 2048) -> np.ndarray:
    result = np.zeros((len(ids), width), dtype=np.float32)
    for row, seq in enumerate(ids):
        bins = np.mod(seq * 2654435761, width)
        np.add.at(result[row], bins, 1.0)
    result /= np.maximum(result.sum(axis=1, keepdims=True), 1.0)
    return result


def random_project_hidden(hidden: np.ndarray, seed: int, width: int = 64) -> np.ndarray:
    rng = np.random.default_rng(seed)
    projection = rng.normal(0, 1.0 / math.sqrt(width), size=(hidden.shape[1], width)).astype(np.float32)
    return hidden @ projection


def metric_row(y: np.ndarray, scores: np.ndarray, cfg: Config, seed: int) -> dict[str, float]:
    lo, hi = bootstrap_auc_ci(y, scores, cfg.bootstrap_repeats, seed)
    return {
        "auc": auc_rank(y, scores),
        "auc_ci95_low": lo,
        "auc_ci95_high": hi,
        "tpr_at_1pct_fpr": tpr_at_fpr(y, scores, 0.01),
        "tpr_at_5pct_fpr": tpr_at_fpr(y, scores, 0.05),
    }


def add_draft_metrics(
    prefix: str,
    draft: dict[str, np.ndarray],
    target: dict[str, np.ndarray],
    labels: np.ndarray,
    train_idx: np.ndarray,
    test_idx: np.ndarray,
    cfg: Config,
    results: dict[str, dict[str, float]],
    seed_offset: int,
) -> dict[str, np.ndarray]:
    selected = bottom_k_indices(draft["token_logp"], cfg.min_k_fraction)
    draft_mean = draft["token_logp"].mean(axis=1)
    draft_min = gather_positions(draft["token_logp"], selected).mean(axis=1)
    draft_entropy = -draft["entropy"].mean(axis=1)
    draft_grad = -gather_positions(draft["grad_proxy"], selected).mean(axis=1)
    greedy = gather_positions(target["top1_match"], selected).mean(axis=1)
    oracle_qselected = gather_positions(target["token_logp"], selected).mean(axis=1)
    true_target_min = gather_positions(
        target["token_logp"], bottom_k_indices(target["token_logp"], cfg.min_k_fraction)
    ).mean(axis=1)

    tomo_logp, honest_accept, query_bits = transcript_tomography(
        target["token_logp"],
        draft["token_logp"],
        selected,
        cfg.transcript_repeats,
        cfg.transcript_levels,
        cfg.seed + seed_offset,
    )
    tomo = tomo_logp.mean(axis=1)
    honest = honest_accept.mean(axis=1)

    rng = np.random.default_rng(cfg.seed + seed_offset + 1)
    random_selected = np.vstack(
        [rng.choice(target["token_logp"].shape[1], selected.shape[1], replace=False) for _ in range(len(labels))]
    )
    random_tomo_logp, _, random_query_bits = transcript_tomography(
        target["token_logp"],
        draft["token_logp"],
        random_selected,
        cfg.transcript_repeats,
        cfg.transcript_levels,
        cfg.seed + seed_offset + 2,
    )
    random_tomo = random_tomo_logp.mean(axis=1)

    base_scores = {
        f"{prefix}/draft_mean_logp": draft_mean,
        f"{prefix}/draft_min20_logp": draft_min,
        f"{prefix}/draft_negative_entropy": draft_entropy,
        f"{prefix}/draft_negative_grad_proxy": draft_grad,
        f"{prefix}/greedy_verifier_match_selected": greedy,
        f"{prefix}/honest_accept_rate_selected": honest,
        f"{prefix}/acceptance_tomography_random": random_tomo,
        f"{prefix}/acceptance_tomography_qmin": tomo,
        f"{prefix}/oracle_target_qselected": oracle_qselected,
        f"{prefix}/oracle_target_true_min20": true_target_min,
    }
    y_test = labels[test_idx]
    for metric_idx, (name, scores) in enumerate(base_scores.items()):
        results[name] = metric_row(y_test, scores[test_idx], cfg, cfg.seed + seed_offset + metric_idx + 10)

    hidden = random_project_hidden(draft["hidden_mean"], cfg.seed + seed_offset + 3)
    hidden_test_scores = fit_logistic(hidden[train_idx], labels[train_idx], hidden[test_idx])
    results[f"{prefix}/whitebox_hidden_probe"] = metric_row(
        y_test, hidden_test_scores, cfg, cfg.seed + seed_offset + 40
    )

    joint = np.column_stack([draft_mean, draft_min, draft_entropy, draft_grad, greedy, honest, tomo])
    joint_test_scores = fit_logistic(joint[train_idx], labels[train_idx], joint[test_idx])
    results[f"{prefix}/joint_whitebox_transcript"] = metric_row(
        y_test, joint_test_scores, cfg, cfg.seed + seed_offset + 41
    )
    return {
        "selected": selected,
        "query_bits": query_bits,
        "random_query_bits": random_query_bits,
        "joint_test_scores": joint_test_scores,
    }


def render_markdown(
    cfg: Config,
    metadata: dict[str, Any],
    training: dict[str, Any],
    metrics: dict[str, dict[str, float]],
    query_budget: dict[str, Any],
) -> str:
    lines = [
        "# Edge–Cloud SD Membership-Audit Pilot",
        "",
        "## Material Passport",
        "",
        f"- Experiment ID: `sd-membership-pilot-{cfg.seed}`",
        "- Type: controlled code experiment",
        "- Status: COMPLETED",
        "- Verification status: ANALYZED (single-seed pilot; not a deployment-level claim)",
        "- Signal boundary: intended verifier feedback only; no timing or packet metadata",
        "",
        "## Controlled setting",
        "",
        f"- Source PDFs: {metadata['source_pdf_count']}",
        f"- Candidate records: {cfg.n_per_class} randomized members + {cfg.n_per_class} randomized nonmembers",
        f"- Auxiliary distillation records: {cfg.n_aux}, disjoint from both candidate groups",
        f"- Sequence length: {cfg.seq_len} GPT-2 tokens",
        f"- Target: GPT-2 XL, fine-tuned for {cfg.target_epochs} epochs on member records only",
        f"- Draft: first {cfg.draft_layers} layers; base, auxiliary-distilled, and shared-weight variants",
        "- Member/nonmember construction is source-stratified and randomized; raw extracted text is not persisted",
        "",
        "## Training trace",
        "",
        f"- Target loss: {training['target_loss'][0]:.4f} → {training['target_loss'][-1]:.4f}",
        f"- Distillation loss: {training['distill_loss'][0]:.4f} → {training['distill_loss'][-1]:.4f}",
        f"- Peak allocated GPU memory: {training['peak_gpu_memory_gib']:.2f} GiB",
        "",
        "## Held-out membership-audit results",
        "",
        "All learned probes are fit only on the audit-calibration split. Higher AUC is better; 0.5 is random.",
        "",
        "| Signal | AUC (95% bootstrap CI) | TPR@1%FPR | TPR@5%FPR |",
        "|---|---:|---:|---:|",
    ]
    for name, row in sorted(metrics.items()):
        lines.append(
            f"| `{name}` | {row['auc']:.3f} [{row['auc_ci95_low']:.3f}, {row['auc_ci95_high']:.3f}] "
            f"| {row['tpr_at_1pct_fpr']:.3f} | {row['tpr_at_5pct_fpr']:.3f} |"
        )
    lines.extend(
        [
            "",
            "## Transcript query budget",
            "",
            f"- White-box q-min selection: median {query_budget['qmin_median_bits']:.0f} accept/reject bits per record "
            f"(P95 {query_budget['qmin_p95_bits']:.0f})",
            f"- Random token selection: median {query_budget['random_median_bits']:.0f} bits per record "
            f"(P95 {query_budget['random_p95_bits']:.0f})",
            "- The pilot uses fixed-repeat binomial estimates. A sequential probability-ratio design should be "
            "evaluated next to reduce this budget.",
            "",
            "## Interpretation boundary",
            "",
            "This pilot checks causal feasibility in a randomized fine-tuning setting. It does not establish the "
            "pretraining-membership risk of production models. The shared-weight draft is an operational upper "
            "bound; the auxiliary-distilled draft is closer to a draft trained without direct access to candidate records.",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> None:
    args = parse_args()
    cfg = Config(
        seed=args.seed,
        gpu=args.gpu,
        seq_len=args.seq_len,
        n_per_class=args.n_per_class,
        n_aux=args.n_aux,
        audit_train_per_class=args.audit_train_per_class,
        target_epochs=args.target_epochs,
        target_batch_size=args.target_batch_size,
        target_lr=args.target_lr,
        draft_layers=args.draft_layers,
        distill_steps=args.distill_steps,
        distill_batch_size=args.distill_batch_size,
        distill_lr=args.distill_lr,
        distill_temperature=args.distill_temperature,
        min_k_fraction=args.min_k_fraction,
        transcript_repeats=args.transcript_repeats,
        transcript_levels=args.transcript_levels,
        bootstrap_repeats=args.bootstrap_repeats,
    )
    if cfg.audit_train_per_class >= cfg.n_per_class:
        raise ValueError("audit_train_per_class must be smaller than n_per_class")
    if not MODEL_SNAPSHOT.exists():
        raise FileNotFoundError(MODEL_SNAPSHOT)

    root = Path(__file__).resolve().parents[1]
    output_dir = (root / args.output_dir).resolve() if not args.output_dir.is_absolute() else args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    set_seed(cfg.seed)
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available")
    if torch.cuda.device_count() != 1:
        print(f"visible CUDA devices={torch.cuda.device_count()}; experiment will use cuda:0 only", flush=True)
    device = torch.device("cuda:0")
    torch.cuda.set_device(device)
    torch.cuda.reset_peak_memory_stats(device)
    print(f"device={torch.cuda.get_device_name(device)}", flush=True)

    tokenizer = AutoTokenizer.from_pretrained(MODEL_SNAPSHOT, local_files_only=True)
    # Long paragraphs are only tokenized and then sliced into short chunks;
    # they are never passed to the model as long sequences.
    tokenizer.model_max_length = 10**9
    members, nonmembers, auxiliary, data_metadata = build_controlled_split(root, tokenizer, cfg)
    candidates = np.concatenate([members, nonmembers], axis=0)
    labels = np.concatenate(
        [np.ones(len(members), dtype=np.int64), np.zeros(len(nonmembers), dtype=np.int64)]
    )
    train_idx, test_idx = balanced_indices(
        cfg.n_per_class, cfg.audit_train_per_class, cfg.seed + 30
    )

    started = time.time()
    target = AutoModelForCausalLM.from_pretrained(
        MODEL_SNAPSHOT,
        torch_dtype=torch.bfloat16,
        local_files_only=True,
        low_cpu_mem_usage=True,
    ).to(device)
    if not isinstance(target, GPT2LMHeadModel):
        raise TypeError(type(target))
    target_loss = fine_tune_target(target, members, cfg, device)
    target_features = extract_features(target, candidates, device, cfg.target_batch_size)

    base_draft = load_base_draft(target.config, cfg.draft_layers, device)
    base_features = extract_features(base_draft, candidates, device, cfg.target_batch_size)
    distill_loss = distill_on_auxiliary(base_draft, target, auxiliary, cfg, device)
    distilled_features = extract_features(base_draft, candidates, device, cfg.target_batch_size)
    del base_draft
    gc.collect()
    torch.cuda.empty_cache()

    shared_draft = copy_truncated_target(target, cfg.draft_layers, device)
    shared_features = extract_features(shared_draft, candidates, device, cfg.target_batch_size)
    del shared_draft, target
    gc.collect()
    torch.cuda.empty_cache()

    results: dict[str, dict[str, float]] = {}
    # A model-less negative control detects accidental member/nonmember
    # distribution shift in the randomized benchmark construction.
    bow = hashed_bow(candidates)
    bow_test = fit_logistic(bow[train_idx], labels[train_idx], bow[test_idx])
    y_test = labels[test_idx]
    results["control/model_less_hashed_bow"] = metric_row(
        y_test, bow_test, cfg, cfg.seed + 500
    )

    query_arrays: list[np.ndarray] = []
    random_query_arrays: list[np.ndarray] = []
    for prefix, features, offset in [
        ("base_draft", base_features, 600),
        ("aux_distilled_draft", distilled_features, 700),
        ("shared_weight_draft", shared_features, 800),
    ]:
        extras = add_draft_metrics(
            prefix,
            features,
            target_features,
            labels,
            train_idx,
            test_idx,
            cfg,
            results,
            offset,
        )
        query_arrays.append(extras["query_bits"])
        random_query_arrays.append(extras["random_query_bits"])

    qbits = np.concatenate(query_arrays)
    rbits = np.concatenate(random_query_arrays)
    query_budget = {
        "qmin_median_bits": float(np.median(qbits)),
        "qmin_p95_bits": float(np.quantile(qbits, 0.95)),
        "random_median_bits": float(np.median(rbits)),
        "random_p95_bits": float(np.quantile(rbits, 0.95)),
    }
    training = {
        "target_loss": target_loss,
        "distill_loss": distill_loss,
        "duration_seconds": time.time() - started,
        "peak_gpu_memory_gib": torch.cuda.max_memory_allocated(device) / (1024**3),
        "device": torch.cuda.get_device_name(device),
    }
    artifact = {
        "material_passport": {
            "experiment_id": f"sd-membership-pilot-{cfg.seed}",
            "type": "controlled_code_experiment",
            "status": "COMPLETED",
            "verification_status": "ANALYZED_SINGLE_SEED_PILOT",
        },
        "config": asdict(cfg),
        "data": data_metadata,
        "split": {
            "audit_train_size": int(len(train_idx)),
            "audit_test_size": int(len(test_idx)),
            "audit_test_members": int(labels[test_idx].sum()),
            "audit_test_nonmembers": int((1 - labels[test_idx]).sum()),
        },
        "training": training,
        "metrics": results,
        "query_budget": query_budget,
    }
    (output_dir / "results.json").write_text(
        json.dumps(artifact, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    (output_dir / "RESULTS.md").write_text(
        render_markdown(cfg, data_metadata, training, results, query_budget), encoding="utf-8"
    )
    print(json.dumps({"output_dir": str(output_dir), "training": training}, indent=2), flush=True)


if __name__ == "__main__":
    main()
