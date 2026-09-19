"""Neural nonmember likelihoods for joint position/proposal experiment design.

The null prior is fitted to multi-q accept/reject counts, never exact delta.
All probability access is confined to the verifier simulator. Proposals have
q_lambda(y)=q0(y)**(1-lambda), realizable as (1-w)Q0+w*point_mass(y).
Predictable JS-greedy policies compare a learned null to fixed positive-shift
alternatives. This is cached, position-locked replay, not a live API claim.
"""

from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path

import numpy as np
import torch

from ..replay_cache import (load_replay_data)
from ..audit_metrics import (membership_metrics)
from ..conditional_accept_only import (ConditionalCountTCN)
from ..audit_partitions import legacy_partitions as record_partitions
from ..audit_metrics import (conformal_tail_pvalues)
from ..audit_runtime import (ROOT, _paths, _write_json)


LAMBDAS = np.array([0., .25, .5, .75, 1.])
GRID = np.array([-12., -8., -6., -4., -3., -2., -1.5, -1., -.5, 0., .5, 1., 1.5, 2., 3., 4., 8., 0.])
SHIFTS = np.array([.5, 1., 2.])
METHODS = ("fixed_q", "uniform_ladder", "adaptive_q", "adaptive_position", "joint")
BUDGETS = (1, 2, 4, 8)


def latent_logp(logq):
    values = np.minimum(np.asarray(logq)[..., None] + GRID, 0.)
    values[..., -1] = 0.  # Exact p=1 atom; other states are a q-relative grid.
    return values


def candidate_proposal_weight(q0, proposal_q):
    q0, proposal_q = np.asarray(q0), np.asarray(proposal_q)
    if np.any(q0 <= 0) or np.any(q0 > 1) or np.any(proposal_q < q0) or np.any(proposal_q > 1):
        raise ValueError("proposal must lie between q0 and one")
    return np.divide(proposal_q - q0, 1 - q0, out=np.zeros_like(proposal_q, dtype=float), where=q0 < 1)


def null_features(logq):
    return np.stack((logq, np.broadcast_to(np.linspace(0., 1., logq.shape[1]), logq.shape)), axis=-1).astype(np.float32)


def multiq_log_likelihood(logq: torch.Tensor, counts: torch.Tensor, trials: int = 2):
    grid = torch.tensor(GRID, device=logq.device, dtype=logq.dtype)
    support = torch.minimum(logq[..., None] + grid, torch.zeros((), device=logq.device))
    support[..., -1] = 0.
    levels = torch.tensor(LAMBDAS, device=logq.device, dtype=logq.dtype)
    action_logq = logq[..., None] * (1 - levels)
    loga = torch.minimum(support[..., None, :] - action_logq[..., :, None], torch.zeros((), device=logq.device))
    successes, failures = counts[..., None], trials - counts[..., None]
    # Saturated actions forbid rejection; 0*log(0) contributes exactly zero.
    rejection = torch.where(failures == 0, torch.zeros_like(loga), failures * torch.log(-torch.expm1(loga)))
    return (successes * loga + rejection).sum(-2)


def fit_prior(logq, counts, parts, *, seed, epochs=30):
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    raw = null_features(logq)
    mean = raw[parts["train"]].mean((0, 1))
    scale = np.maximum(raw[parts["train"]].std((0, 1)), 1e-6)
    features = torch.from_numpy((raw - mean) / scale)
    model = ConditionalCountTCN(2, 1, channels=24)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-3)
    observations = torch.from_numpy(counts.astype(np.float32))
    q = torch.from_numpy(logq.astype(np.float32))
    best, state, best_epoch, history = float("inf"), None, 0, []
    for epoch in range(1, epochs + 1):
        losses = {}
        for phase in ("train", "validation"):
            indices = rng.permutation(parts[phase]) if phase == "train" else parts[phase]
            model.train(phase == "train")
            total = 0.
            with torch.set_grad_enabled(phase == "train"):
                for start in range(0, len(indices), 32):
                    batch = indices[start:start + 32]
                    x = features[batch]
                    mask = torch.ones(x.shape[:2], dtype=torch.bool)
                    weights = model.mixture_log_weights(x, mask)
                    likelihood = multiq_log_likelihood(q[batch], observations[batch])
                    loss = -torch.logsumexp(weights + likelihood, -1).mean() / 10
                    if not torch.isfinite(loss):
                        raise RuntimeError("invalid observable multi-q likelihood")
                    if phase == "train":
                        optimizer.zero_grad()
                        loss.backward()
                        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)
                        optimizer.step()
                    total += float(loss.detach()) * len(batch)
            losses[phase] = total / len(indices)
        history.append({"epoch": epoch, **losses})
        if losses["validation"] < best - 1e-5:
            best, best_epoch, state = losses["validation"], epoch, copy.deepcopy(model.state_dict())
        if epoch - best_epoch >= 5:
            break
    model.load_state_dict(state)
    model.eval()
    weights = np.empty((*logq.shape, len(GRID)))
    with torch.inference_mode():
        for start in range(0, len(logq), 64):
            x = features[start:start + 64]
            weights[start:start + len(x)] = model.mixture_log_weights(x, torch.ones(x.shape[:2], dtype=torch.bool)).exp().numpy()
    return model, mean, scale, weights, history, best_epoch


def binary_js(a, b):
    def entropy(p):
        clipped = np.clip(p, 1e-15, 1 - 1e-15)
        return -(p * np.log(clipped) + (1 - p) * np.log1p(-clipped))
    return np.maximum(0., entropy((a + b) / 2) - .5 * (entropy(a) + entropy(b)))


class CachedVerifier:
    """Simulator-only target boundary; policies receive a callable, not p."""
    def __init__(self, logp, logq, original_indices, seed, max_visits=16):
        self.alpha = np.exp(np.minimum(0., logp[..., None] - logq[..., None] * (1 - LAMBDAS)))
        self.uniforms = np.stack([np.random.default_rng(np.random.SeedSequence([seed, int(index), 4817])).random((logq.shape[1], len(LAMBDAS), max_visits)) for index in original_indices])

    def query(self, positions, actions, visits):
        rows = np.arange(len(positions))
        return self.uniforms[rows, positions, actions, visits] < self.alpha[rows, positions, actions]


def replay_policy(logq, prior, query, method, budgets=BUDGETS, cap=16):
    if method not in METHODS or tuple(sorted(set(budgets))) != tuple(budgets) or budgets[0] != 1:
        raise ValueError("unknown policy or invalid budget checkpoints")
    n, length = logq.shape
    if cap < max(budgets):
        raise ValueError("per-position cap cannot support the requested budget")
    support = latent_logp(logq)
    shifted = np.minimum(support[..., None, :] + SHIFTS[:, None], 0.).reshape(n, length, -1)
    action_q = logq[..., None] * (1 - LAMBDAS)
    emission0 = np.exp(np.minimum(0., support[:, :, None, :] - action_q[..., None]))
    emission1 = np.exp(np.minimum(0., shifted[:, :, None, :] - action_q[..., None]))
    w0 = prior / prior.sum(-1, keepdims=True)
    w1 = np.tile(w0, (1, 1, len(SHIFTS))) / len(SHIFTS)
    counts = np.zeros((n, length), dtype=np.int64)
    action_counts = np.zeros((n, length, len(LAMBDAS)), dtype=np.int64)
    utility = np.zeros((n, length, len(LAMBDAS)))
    scores = np.zeros(n)
    checkpoints, rows = {}, np.arange(n)

    def refresh(positions):
        a0 = (w0[rows, positions, None] * emission0[rows, positions]).sum(-1)
        a1 = (w1[rows, positions, None] * emission1[rows, positions]).sum(-1)
        utility[rows, positions] = binary_js(a0, a1)

    for step in range(length * max(budgets)):
        if step < length:
            positions, actions = np.full(n, step), np.zeros(n, dtype=int)
        elif method in ("fixed_q", "uniform_ladder", "adaptive_q"):
            positions = np.full(n, step % length)
            if method == "fixed_q":
                actions = np.zeros(n, dtype=int)
            elif method == "uniform_ladder":
                actions = np.full(n, (step // length - 1) % 4 + 1)
            else:
                actions = utility[rows, positions].argmax(-1)
        else:
            if method == "joint":
                next_action = utility.argmax(-1)
            else:
                next_action = (counts - 1) % 4 + 1
            values = np.take_along_axis(utility, next_action[..., None], axis=-1)[..., 0]
            values = np.where(counts < cap, values, -np.inf)
            positions = values.argmax(-1)
            actions = next_action[rows, positions]
        visits = action_counts[rows, positions, actions]
        bits = query(positions, actions, visits)
        probability0 = emission0[rows, positions, actions]
        probability1 = emission1[rows, positions, actions]
        likelihood0 = np.where(bits[:, None], probability0, 1 - probability0)
        likelihood1 = np.where(bits[:, None], probability1, 1 - probability1)
        updated0 = w0[rows, positions] * likelihood0
        updated1 = w1[rows, positions] * likelihood1
        evidence0, evidence1 = updated0.sum(-1), updated1.sum(-1)
        if np.any(evidence0 <= 0) or np.any(evidence1 <= 0):
            raise RuntimeError("latent grid has no support for an observed event")
        scores += np.log(evidence1) - np.log(evidence0)
        w0[rows, positions] = updated0 / evidence0[:, None]
        w1[rows, positions] = updated1 / evidence1[:, None]
        counts[rows, positions] += 1
        action_counts[rows, positions, actions] += 1
        refresh(positions)
        if (step + 1) % length == 0 and (step + 1) // length in budgets:
            checkpoints[(step + 1) // length] = scores.copy()
    return checkpoints, {"per_position_counts": counts, "action_counts": action_counts}


def suffix_cache(benchmark, epoch, width=64):
    data = load_replay_data(*_paths(benchmark, epoch))
    if np.any(data.lengths < width):
        raise ValueError("records shorter than the candidate suffix")
    p = np.stack([data.logp[end - width:end] for end in data.offsets[1:]])
    q = np.stack([data.logq0[end - width:end] for end in data.offsets[1:]])
    return p, q, data.labels, data.record_ids


def evaluate(benchmark, epoch, seed, output):
    logp, logq, labels, ids = suffix_cache(benchmark, epoch)
    parts = record_partitions(labels, ids)
    counts = np.zeros((*logq.shape, len(LAMBDAS)), dtype=np.int64)
    # Only real nonmembers supply training/selection outcomes.
    for index in parts["reference"]:
        rng = np.random.default_rng(np.random.SeedSequence([seed, int(index), 7159]))
        alpha = np.exp(np.minimum(0., logp[index, :, None] - logq[index, :, None] * (1 - LAMBDAS)))
        counts[index] = rng.binomial(2, alpha)
    model, mean, scale, prior, history, best_epoch = fit_prior(logq, counts, parts, seed=seed)
    selected = np.sort(np.r_[parts["calibration"], parts["test"]])
    verifier = CachedVerifier(logp[selected], logq[selected], selected, seed)
    scores, allocation, stopping = {}, {}, {}
    for method in METHODS:
        print(json.dumps({"condition": f"{benchmark}_{epoch}", "seed": seed, "policy": method}), flush=True)
        snapshots, cost = replay_policy(logq[selected], prior[selected], verifier.query, method)
        for budget, values in snapshots.items():
            whole = np.full(len(labels), np.nan)
            whole[selected] = values
            scores[f"{method}_b{budget}"] = whole
        allocation[method] = {"max_position_count": int(cost["per_position_counts"].max()),
                              "mean_action_counts": cost["action_counts"].sum(1).mean(0).tolist(),
                              "decisions_per_record": int(cost["action_counts"][0].sum())}
        if method == "joint":
            paths = np.stack([scores[f"joint_b{budget}"] for budget in BUDGETS], axis=1)
            calibration_max = paths[parts["calibration"]].max(1)
            test = parts["test"]
            pvalues = np.column_stack([conformal_tail_pvalues(paths[test, j], calibration_max) for j in range(len(BUDGETS))])
            for level in (.01, .05):
                hits = pvalues <= level
                first = np.where(hits.any(1), hits.argmax(1), len(BUDGETS) - 1)
                cost_used = np.array(BUDGETS)[first] * logq.shape[1]
                stopping[str(level)] = {"tpr": float(hits.any(1)[labels[test] == 1].mean()),
                                        "actual_fpr": float(hits.any(1)[labels[test] == 0].mean()),
                                        "mean_decisions": float(cost_used.mean()), "max_decisions": 8 * logq.shape[1]}
    metrics = {name: membership_metrics(value, labels, parts["calibration"], parts["test"]) for name, value in scores.items()}
    output.mkdir(parents=True, exist_ok=True)
    torch.save({"state_dict": model.state_dict(), "mean": torch.tensor(mean), "scale": torch.tensor(scale)}, output / "prior.pt")
    np.savez_compressed(output / "scores.npz", labels=labels, record_ids=ids, **parts, **scores)
    report = {"experiment": "JS-guided joint accept-only probing", "benchmark": benchmark, "epoch": epoch, "seed": seed,
              "training_member_count": 0, "synthetic_member_count": 0, "null_training_queries_per_token": 10,
              "null_training_queries": int(len(parts["reference"]) * 64 * 10),
              "candidate_tokens": 64, "best_epoch": best_epoch, "history": history,
              "metrics": metrics, "allocation": allocation, "positive_stopping": stopping,
              "protocol": "position-locked replay with normalized client-controlled candidate proposals; not a deployment claim",
              "alternatives": "fixed probability boosts .5,1,2 in log coordinates; no member-trained policy"}
    _write_json(output / "REPORT.json", report)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--benchmark", choices=("wikitection", "newstection", "arxivtection"), required=True)
    parser.add_argument("--epoch", type=int, choices=(1, 3), required=True)
    parser.add_argument("--seed", type=int, default=20260914)
    parser.add_argument("--threads", type=int, default=2)
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()
    torch.set_num_threads(args.threads)
    output = args.output_dir or ROOT / "experiments/results/sft_runs/directions_validation/active" / f"{args.benchmark}_epoch{args.epoch}" / f"seed{args.seed}"
    evaluate(args.benchmark, args.epoch, args.seed, output)


if __name__ == "__main__":
    main()
