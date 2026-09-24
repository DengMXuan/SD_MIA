"""Nonmember-only prediction, sparse evidence, and document-level calibration.

Natural SD uses a causal GRU over reached binary outcomes. Fixed probes reuse
the current difficulty/count TCN. Starts are never split into separate records.
"""
from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path

import numpy as np
from scipy.special import logsumexp
import torch
from torch import nn

from experiments.shared.core.audit_metrics import membership_metrics, conformal_tail_pvalues, rank_auc
from experiments.shared.core.audit_partitions import deployment_partitions
from experiments.shared.core.audit_runtime import _write_json
from experiments.shared.methods.conditional_accept_only import make_batch, count_nll, assert_partition_contract
from experiments.shared.methods.deployment_accept_only import _decisions
from experiments.shared.core.deployment_archive import sha256_file
from experiments.shared.protocols.protocol_archive import load_archive, atomic_npz


class CausalAcceptanceGRU(nn.Module):
    def __init__(self, input_dim: int, channels: int = 24):
        super().__init__()
        self.gru = nn.GRU(input_dim, channels, batch_first=True)
        self.head = nn.Linear(channels, 2)

    def forward(self, x, mask):
        return torch.log_softmax(self.head(self.gru(x)[0]), dim=-1)


def natural_features(features, counts, lengths):
    previous = np.zeros(len(counts), dtype=np.float32)
    offset = 0
    for length in lengths:
        previous[offset + 1:offset + length] = counts[offset:offset + length - 1]
        offset += length
    return np.column_stack((features, previous)).astype(np.float32)


def trajectory_partitions(parts, owners):
    """All starts inherit the document partition, including calibration."""
    return {name: np.flatnonzero(np.isin(owners, indices)) for name, indices in parts.items()}


def fit_causal(x, counts, lengths, parts, *, seed, device="cpu", epochs=30):
    if epochs < 1:
        raise ValueError("epochs must be positive")
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    offsets = np.r_[0, lengths.cumsum()]
    training = np.concatenate([np.arange(offsets[i], offsets[i + 1]) for i in parts["train"]])
    mean, scale = x[training].mean(0), x[training].std(0)
    scale = np.where(scale < 1e-6, 1., scale)
    standardized = (x - mean) / scale
    model = CausalAcceptanceGRU(x.shape[1]).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-3)
    best, best_epoch, state, history = np.inf, 0, None, []
    for epoch in range(1, epochs + 1):
        row = {"epoch": epoch}
        for phase in ("train", "validation"):
            model.train(phase == "train")
            indices = rng.permutation(parts[phase]) if phase == "train" else parts[phase]
            if not len(indices):
                raise ValueError(f"empty {phase} partition")
            total = 0.
            with torch.set_grad_enabled(phase == "train"):
                for start in range(0, len(indices), 32):
                    batch = indices[start:start + 32]
                    bx, by, mask = make_batch(standardized, counts, offsets, batch, device)
                    loss = count_nll(model(bx, mask), by, mask)
                    if not torch.isfinite(loss):
                        raise ValueError("nonfinite conditional acceptance loss")
                    if phase == "train":
                        optimizer.zero_grad()
                        loss.backward()
                        nn.utils.clip_grad_norm_(model.parameters(), 1.)
                        optimizer.step()
                    total += float(loss.detach()) * len(batch)
            row[phase + "_nll"] = total / len(indices)
        history.append(row)
        if row["validation_nll"] < best - 1e-5:
            best, best_epoch, state = row["validation_nll"], epoch, copy.deepcopy(model.state_dict())
        if epoch - best_epoch >= 5:
            break
    model.load_state_dict(state)
    return model, mean, scale, history, best_epoch


def predict(model, x, counts, lengths, device):
    model.eval()
    offsets = np.r_[0, lengths.cumsum()]
    output = []
    with torch.inference_mode():
        for start in range(0, len(lengths), 32):
            batch = np.arange(start, min(start + 32, len(lengths)))
            bx, _, mask = make_batch(x, counts, offsets, batch, device)
            rows = model(bx, mask).cpu().numpy()
            output.extend(rows[row, :lengths[index]] for row, index in enumerate(batch))
    return np.concatenate(output)


def evidence_components(logpmf, counts, *, sparse):
    """Preregistered signed tilts; these are calibrated scores, not e-values."""
    eta = np.asarray([.5, 1., 2., -.5, -1., -2.])
    values = np.arange(logpmf.shape[1])
    evidence = (counts.astype(float)[:, None] * eta
                - logsumexp(logpmf[:, :, None] + values[None, :, None] * eta, axis=1))
    if sparse:
        return np.stack([np.logaddexp(np.log1p(-rho), np.log(rho) + evidence)
                         for rho in (.05, .1, .25)], axis=1)
    return evidence[:, None, :]


def document_scores(data, logpmf, *, start_index=None):
    """Sum fixed alternative evidence within each document, then mix alternatives.

    This does not assume starts are independent for calibration validity:
    the combined score is calibrated on identically queried nonmember documents.
    """
    n = len(data["record_ids"])
    offsets = np.r_[0, data["lengths"].cumsum()]
    selected = np.arange(len(data["lengths"]))
    if start_index is not None:
        selected = selected[data["start_indices"] == start_index]
    scores = {}
    for sparse in (False, True):
        components = evidence_components(logpmf, data["counts"], sparse=sparse)
        summed = np.zeros((n, *components.shape[1:]), dtype=np.float64)
        for t in selected:
            summed[data["document_indices"][t]] += components[offsets[t]:offsets[t + 1]].sum(0)
        prefix = "sparse" if sparse else "global"
        for name, sl in (("positive", slice(0, 3)), ("negative", slice(3, 6)), ("two_sided", slice(None))):
            values = summed[..., sl].reshape(n, -1)
            scores[f"{prefix}_{name}"] = logsumexp(values, axis=1) - np.log(values.shape[1])
    size, qsum, accepted = np.zeros(n), np.zeros(n), np.zeros(n)
    for t in selected:
        doc = data["document_indices"][t]
        left, right = offsets[t:t + 2]
        size[doc] += right - left
        qsum[doc] += data["features"][left:right, 0].sum()
        accepted[doc] += data["counts"][left:right].sum()
    if (size == 0).any():
        raise ValueError("a score must cover every document")
    scores["q_only"] = qsum / size
    scores["accept_rate"] = accepted / (size * (logpmf.shape[1] - 1))
    scores["reject_rate"] = 1 - scores["accept_rate"]
    return scores


def score_metrics(scores, labels, parts, *, seed):
    result = {}
    test, calibration = parts["test"], parts["calibration"]
    for name, values in scores.items():
        pvalues = conformal_tail_pvalues(values[test], values[calibration])
        member = values[test[labels[test] == 1]]
        nonmember = values[test[labels[test] == 0]]
        rng = np.random.default_rng(seed)
        auc_samples = [rank_auc(rng.choice(member, len(member)), rng.choice(nonmember, len(nonmember)))
                       for _ in range(200)]
        result[name] = {
            **membership_metrics(values, labels, calibration, test),
            "auc_bootstrap_95": np.quantile(auc_samples, [.025, .975]).tolist(),
            "decisions": _decisions(pvalues, labels[test]),
        }
    return result


def evaluate(path: Path, output: Path, *, seed=20260914, epochs=30, device="cpu"):
    data, envelope = load_archive(path)
    contract = envelope["contract"]
    if contract["data_contract"] != "four_role_600":
        raise ValueError("runtime smoke archives cannot establish membership results")
    parts = deployment_partitions(data["labels"], data["record_ids"], data["record_roles"])
    assert_partition_contract(data["labels"], data["record_ids"], parts)
    source = {"archive_sha256": envelope["archive_sha256"], "sidecar_sha256": envelope["sidecar_sha256"],
              "seed": seed, "epochs": epochs, "device": device,
              "scorer_sha256": sha256_file(Path(__file__))}
    output.mkdir(parents=True, exist_ok=True)
    config_path = output / "EVALUATION.json"
    if config_path.exists() and json.loads(config_path.read_text()) != source:
        raise ValueError("evaluation source/settings changed; choose a new output directory")
    _write_json(config_path, source)
    trajectory_parts = trajectory_partitions(parts, data["document_indices"])
    counts, lengths = data["counts"], data["lengths"]
    if contract["protocol"] == "natural":
        x = natural_features(data["features"], counts, lengths)
        fit = fit_causal
        architecture = "causal_gru_binary"
    else:
        from experiments.shared.methods.difficulty_accept_only import fit
        x = data["features"][:, [0, 4, 1, 2, 3]]
        architecture = "difficulty_tcn_count_b2"
    model, mean, scale, history, best_epoch = fit(
        x, counts, lengths, trajectory_parts, seed=seed, device=device, epochs=epochs,
    )
    logpmf = predict(model, (x - mean) / scale, counts, lengths, device)
    all_scores, reports = {}, {}
    scopes = [("combined", None)]
    if len(contract["starts"]) > 1:
        scopes.extend((f"start_{index}", index) for index in range(len(contract["starts"])))
    for scope, start in scopes:
        scores = document_scores(data, logpmf, start_index=start)
        reports[scope] = score_metrics(scores, data["labels"], parts, seed=seed)
        all_scores.update({f"{scope}__{name}": values for name, values in scores.items()})
    checkpoint = output / "detector.pt"
    torch.save({"architecture": architecture, "state_dict": model.state_dict(),
                "input_dim": x.shape[1], "mean": torch.tensor(mean), "scale": torch.tensor(scale),
                "seed": seed, "best_epoch": best_epoch}, checkpoint)
    atomic_npz(output / "scores.npz", {"record_ids": data["record_ids"], "labels": data["labels"],
                                      **parts, **all_scores})
    report = {
        "protocol": contract["protocol"], "adapter": contract["adapter"],
        "starts": contract["starts"], "rounds_per_start": contract["rounds_per_start"],
        "source": source, "architecture": architecture, "history": history,
        "training_member_count": 0, "synthetic_member_count": 0,
        "partition_documents": {key: len(value) for key, value in parts.items()},
        "best_epoch": best_epoch, "checkpoint_sha256": sha256_file(checkpoint),
        "scores_sha256": sha256_file(output / "scores.npz"),
        "metrics": reports, "costs": envelope["costs"],
        "aggregation": "fixed alternative evidence summed over document trajectories before mixing",
        "selection": "no direction/start/score selected with member test labels",
        "uncertainty": "document bootstrap AUC; Wilson decision intervals conditional on fitted detector/calibration",
        "head_real_model_validation": contract["head_real_model_validation"],
        "execution": contract["execution"],
    }
    _write_json(output / "REPORT.json", report)
    print(json.dumps({"report": str(output / "REPORT.json"), "metrics": reports}), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--observations", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=20260914)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    torch.set_num_threads(2)
    evaluate(args.observations, args.output_dir, seed=args.seed, epochs=args.epochs, device=args.device)


if __name__ == "__main__":
    main()
