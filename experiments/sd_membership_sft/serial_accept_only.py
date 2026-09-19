"""Frozen-model natural speculative decoding and nonmember hazard audit.

Unlike cached candidate replay, rejection corrections here change subsequent
prefixes. The runtime uses target probabilities and correction tokens; the
detector archive exposes only reached bits, local draft features and costs.
Each detector is evaluated on exactly the same transcript and verifier rounds.
"""

from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from .audit_metrics import (membership_metrics)
from .conditional_accept_only import (assert_partition_contract, make_batch)
from .audit_partitions import legacy_partitions as record_partitions
from .data import (prompt_prefix_ids)
from .generalization import (load_draft_model, load_finetuned_model)
from .audit_runtime import (_write_json)
from .scoring_common import (prepare_scoring_records, resolve_run_dir, role_provenance)


def correction_distribution(p: torch.Tensor, q: torch.Tensor) -> torch.Tensor:
    residual = torch.clamp(p - q, min=0)
    mass = residual.sum()
    if not mass > 0:
        raise ValueError("a rejected proposal must leave positive correction mass")
    return residual / mass


def _advance(model, tokens, cache, device, keep=1):
    result = model(input_ids=torch.tensor([tokens], device=device), past_key_values=cache,
                   use_cache=True, logits_to_keep=keep)
    return result.logits[0].float(), result.past_key_values


@torch.inference_mode()
def collect_transcript(target, draft, prefix: list[int], *, device, seed: int,
                       rounds: int = 8, gamma: int = 4, eos_id: int | None = None):
    if not prefix or rounds < 1 or gamma < 1:
        raise ValueError("nonempty prefix and positive protocol sizes required")
    target.eval()
    draft.eval()
    generator = torch.Generator(device=device).manual_seed(seed)
    tlogits, tcache = _advance(target, prefix, None, device)
    qlogits, qcache = _advance(draft, prefix, None, device)
    pnext, qnext = tlogits[-1], qlogits[-1]
    features, bits, round_ids, accepted_lengths = [], [], [], []
    length, generated, rounds_used, verified_candidates = len(prefix), 0, 0, 0
    finished = False
    for round_index in range(rounds):
        proposed, distributions, qfeatures = [], [], []
        for position in range(gamma):
            q = torch.softmax(qnext, dim=-1)
            token = int(torch.multinomial(q, 1, generator=generator).item())
            proposed.append(token)
            distributions.append(q)
            qfeatures.append((float(torch.log(q[token]).item()), float(-(q * q.clamp_min(1e-30).log()).sum().item())))
            qlogits, qcache = _advance(draft, [token], qcache, device)
            qnext = qlogits[-1]
            if token == eos_id:
                break
        verified, tcache = _advance(target, proposed, tcache, device, keep=0)
        verified_candidates += len(proposed)
        rounds_used += 1
        accepted = 0
        correction = None
        for position, token in enumerate(proposed):
            p = torch.softmax(pnext if position == 0 else verified[position - 1], dim=-1)
            q = distributions[position]
            alpha = torch.minimum(torch.ones((), device=device), p[token] / q[token])
            accept = bool(torch.rand((), device=device, generator=generator) < alpha)
            features.append((*qfeatures[position], position / gamma, round_index / rounds))
            bits.append(int(accept))
            round_ids.append(round_index)
            if not accept:
                residual = correction_distribution(p, q)
                correction = int(torch.multinomial(residual, 1, generator=generator).item())
                break
            accepted += 1
            if token == eos_id:
                finished = True
                break
        accepted_lengths.append(accepted)
        if finished:
            generated += accepted
            break
        if correction is None:
            bonus = torch.softmax(verified[-1], dim=-1)
            correction = int(torch.multinomial(bonus, 1, generator=generator).item())
        # Crop all unaccepted speculative tokens, on BOTH model caches.
        tcache.crop(length + accepted)
        qcache.crop(length + accepted)
        generated += accepted + 1
        length += accepted + 1
        if correction == eos_id:
            break
        tlogits, tcache = _advance(target, [correction], tcache, device)
        qlogits, qcache = _advance(draft, [correction], qcache, device)
        pnext, qnext = tlogits[-1], qlogits[-1]
    return {
        "features": np.asarray(features, dtype=np.float32), "bits": np.asarray(bits, dtype=np.uint8),
        "round_ids": np.asarray(round_ids, dtype=np.int64),
        "accepted_lengths": accepted_lengths, "rounds": rounds_used,
        "reached_decisions": len(bits), "verified_candidate_positions": verified_candidates,
        "generated_tokens": generated, "prefix_tokens": len(prefix),
    }


def collect(args):
    if args.shard_count < 1 or not 0 <= args.shard_index < args.shard_count:
        raise ValueError("invalid collection shard")
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA is not visible; frozen-model collection needs GPU access")
    torch.set_num_threads(4)
    run_dir = resolve_run_dir(args.run_dir)
    cfg, prepared = prepare_scoring_records(run_dir)
    original_parts = record_partitions(prepared.labels, prepared.record_ids)
    selected = np.sort(np.unique(np.concatenate([original_parts[name] for name in ("train", "validation", "calibration", "test")])))
    # A crash can resume completed record transcripts without repeating inference.
    args.output_dir.mkdir(parents=True, exist_ok=True)
    provenance = {role: role_provenance(cfg, run_dir, role) for role in ("target", "draft_auxiliary_distilled")}
    contract = {"provenance": provenance, "seed": args.seed, "rounds": args.rounds, "gamma": args.gamma,
                "candidate_suffix_excluded": 64, "selected_ids": prepared.record_ids[selected].tolist()}
    manifest_path = args.output_dir / "COLLECTION.json"
    if manifest_path.exists() and json.loads(manifest_path.read_text()) != contract:
        raise ValueError("resume protocol differs from existing collection")
    _write_json(manifest_path, contract)
    draft = load_draft_model(run_dir, cfg.draft_model, "draft_auxiliary_distilled", device, attn_implementation="sdpa")
    target = load_finetuned_model(run_dir, cfg.target_model, device, attn_implementation="sdpa")
    draft.requires_grad_(False)
    target.requires_grad_(False)
    costs, pieces = [], []
    records_dir = args.output_dir / "records"
    records_dir.mkdir(exist_ok=True)
    for serial, original_index in enumerate(selected):
        if serial % args.shard_count != args.shard_index:
            continue
        path = records_dir / f"{int(original_index)}.npz"
        if path.exists():
            with np.load(path, allow_pickle=False) as cached:
                trace = {key: cached[key] for key in cached.files}
        else:
            record = prepared.records[int(original_index)]
            prefix = prompt_prefix_ids(record, prepared.tokenizer) + list(record.response_ids[:-64])
            local_seed = int(np.random.SeedSequence([args.seed, int(original_index)]).generate_state(1)[0])
            trace = collect_transcript(target, draft, prefix, device=device, seed=local_seed,
                                       rounds=args.rounds, gamma=args.gamma, eos_id=prepared.tokenizer.eos_token_id)
            temporary = path.with_suffix(".tmp.npz")
            np.savez_compressed(temporary, **trace)
            temporary.replace(path)
        if serial % 25 == 0:
            print(json.dumps({"record_position": serial + 1, "total": len(selected), "shard": args.shard_index}), flush=True)
    missing = [int(index) for index in selected if not (records_dir / f"{int(index)}.npz").exists()]
    if missing:
        print(json.dumps({"shard_complete": args.shard_index, "remaining_records_other_shards": len(missing)}), flush=True)
        return
    for index in selected:
        with np.load(records_dir / f"{int(index)}.npz", allow_pickle=False) as cached:
            trace = {key: cached[key] for key in cached.files}
        pieces.append(trace)
        costs.append({name: int(trace[name]) for name in ("rounds", "reached_decisions", "verified_candidate_positions", "generated_tokens", "prefix_tokens")})
    lookup = {int(original): i for i, original in enumerate(selected)}
    parts = {name: np.asarray([lookup[int(index)] for index in values]) for name, values in original_parts.items()}
    final_temporary = args.output_dir / f"transcripts.shard{args.shard_index}.tmp.npz"
    np.savez_compressed(final_temporary,
                        features=np.concatenate([row["features"] for row in pieces]),
                        bits=np.concatenate([row["bits"] for row in pieces]),
                        lengths=np.asarray([len(row["bits"]) for row in pieces]),
                        labels=prepared.labels[selected], record_ids=prepared.record_ids[selected],
                        **parts)
    final_temporary.replace(args.output_dir / "transcripts.npz")
    _write_json(args.output_dir / "COSTS.json", {"records": costs, "models_frozen": True,
                 "protocol": "natural sampled speculative decoding with target residual corrections and bonus tokens",
                 "detector_inputs": "reached bits; local draft logq/entropy; proposal position and round; no target scores or correction token IDs"})


class HazardGRU(nn.Module):
    def __init__(self, input_dim=5, hidden=24):
        super().__init__()
        self.gru = nn.GRU(input_dim, hidden, batch_first=True)
        self.head = nn.Linear(hidden, 1)

    def forward(self, x):
        return self.head(self.gru(x)[0]).squeeze(-1)


def hazard_features(features, bits, lengths):
    previous = np.zeros(len(bits), dtype=np.float32)
    offset = 0
    for length in lengths:
        previous[offset + 1:offset + length] = bits[offset:offset + length - 1]
        offset += length
    return np.column_stack((features, previous)).astype(np.float32)


def fit_hazard(features, bits, lengths, parts, *, seed=20260914, epochs=30):
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    offsets = np.r_[0, np.cumsum(lengths)]
    train_tokens = np.concatenate([np.arange(offsets[i], offsets[i + 1]) for i in parts["train"]])
    mean, scale = features[train_tokens].mean(0), features[train_tokens].std(0)
    scale = np.where(scale < 1e-6, 1, scale)
    x = (features - mean) / scale
    model = HazardGRU()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-3)
    best, state, best_epoch, history = float("inf"), None, 0, []
    for epoch in range(1, epochs + 1):
        values = {}
        for phase in ("train", "validation"):
            indices = rng.permutation(parts[phase]) if phase == "train" else parts[phase]
            model.train(phase == "train")
            total = 0.
            with torch.set_grad_enabled(phase == "train"):
                for start in range(0, len(indices), 32):
                    batch = indices[start:start + 32]
                    bx, by, mask = make_batch(x, bits, offsets, batch, "cpu")
                    loss = ((F.binary_cross_entropy_with_logits(model(bx), by.float(), reduction="none") * mask).sum(1) / mask.sum(1)).mean()
                    if phase == "train":
                        optimizer.zero_grad()
                        loss.backward()
                        nn.utils.clip_grad_norm_(model.parameters(), 1.)
                        optimizer.step()
                    total += float(loss.detach()) * len(batch)
            values[phase] = total / len(indices)
        history.append({"epoch": epoch, **values})
        if values["validation"] < best - 1e-5:
            best, best_epoch, state = values["validation"], epoch, copy.deepcopy(model.state_dict())
        if epoch - best_epoch >= 5:
            break
    model.load_state_dict(state)
    model.eval()
    predictions = np.empty(len(bits))
    with torch.inference_mode():
        for start in range(0, len(lengths), 32):
            batch = np.arange(start, min(start + 32, len(lengths)))
            bx, _, _ = make_batch(x, bits, offsets, batch, "cpu")
            logits = model(bx).numpy()
            for row, index in enumerate(batch):
                predictions[offsets[index]:offsets[index + 1]] = logits[row, :lengths[index]]
    return model, mean, scale, predictions, history, best_epoch


def hazard_evidence(logits, bits):
    """Fixed logit-tilt alternatives; cast unsigned archive bits before negation."""
    from scipy.special import logsumexp

    z = np.asarray(logits, dtype=np.float64)[:, None]
    observed = np.asarray(bits, dtype=np.float64)[:, None]
    eta = np.asarray([.5, 1., 2.])
    positive = observed * eta - np.logaddexp(0., z + eta) + np.logaddexp(0., z)
    negative = -observed * eta - np.logaddexp(0., z - eta) + np.logaddexp(0., z)
    return (float(logsumexp(positive.sum(0)) - np.log(len(eta))),
            float(logsumexp(negative.sum(0)) - np.log(len(eta))),
            float(logsumexp(np.r_[positive.sum(0), negative.sum(0)]) - np.log(2 * len(eta))))


def evaluate(args):
    torch.set_num_threads(2)
    with np.load(args.output_dir / "transcripts.npz", allow_pickle=False) as data:
        archive = dict(data)
    if {"logp", "delta", "target", "correction_tokens", "token_ids"}.intersection(archive):
        raise ValueError("detector transcript contains forbidden target/token fields")
    if (archive["features"].shape != (len(archive["bits"]), 4)
            or not np.all(np.isfinite(archive["features"]))
            or not np.all((archive["bits"] == 0) | (archive["bits"] == 1))
            or np.any(archive["lengths"] <= 0)
            or int(archive["lengths"].sum()) != len(archive["bits"])):
        raise ValueError("invalid reached-feedback transcript")
    parts = {name: archive[name] for name in ("train", "validation", "reference", "calibration", "test")}
    labels, ids = archive["labels"], archive["record_ids"]
    assert_partition_contract(labels, ids, parts)
    features = hazard_features(archive["features"], archive["bits"], archive["lengths"])
    model, mean, scale, logits, history, best_epoch = fit_hazard(features, archive["bits"], archive["lengths"], parts, seed=args.seed)
    scores = {name: np.empty(len(labels)) for name in ("accept_rate", "hazard_evidence", "hazard_per_decision", "hazard_rejection", "hazard_two_sided", "reached_decisions")}
    offsets = np.r_[0, np.cumsum(archive["lengths"])]
    for i, (start, end) in enumerate(zip(offsets[:-1], offsets[1:])):
        bits = archive["bits"][start:end]
        positive, negative, two_sided = hazard_evidence(logits[start:end], bits)
        scores["hazard_evidence"][i] = positive
        scores["hazard_per_decision"][i] = scores["hazard_evidence"][i] / (end - start)
        scores["hazard_rejection"][i] = negative
        scores["hazard_two_sided"][i] = two_sided
        scores["accept_rate"][i] = bits.mean()
        scores["reached_decisions"][i] = end - start
    scores["reject_rate"] = 1 - scores["accept_rate"]
    reference_rates = scores["accept_rate"][parts["validation"]]
    scores["rate_two_sided"] = np.abs((scores["accept_rate"] - reference_rates.mean()) / max(float(reference_rates.std()), 1e-6))
    metrics = {name: membership_metrics(value, labels, parts["calibration"], parts["test"]) for name, value in scores.items()}
    torch.save({"state_dict": model.state_dict(), "mean": torch.tensor(mean), "scale": torch.tensor(scale)}, args.output_dir / "hazard.pt")
    np.savez_compressed(args.output_dir / "scores.npz", labels=labels, record_ids=ids, **parts, **scores)
    collection = json.loads((args.output_dir / "COLLECTION.json").read_text())
    _write_json(args.output_dir / "REPORT.json", {"experiment": "natural SD hazard", "seed": args.seed,
                 "training_member_count": 0, "best_epoch": best_epoch, "history": history, "metrics": metrics,
                 "round_cap": collection["rounds"], "gamma": collection["gamma"],
                 "directions": "fixed positive, negative, and two-sided alternatives; no member-based orientation selection",
                 "query_control": "identical transcripts for every detector; actual costs in COSTS.json"})
    print(json.dumps(metrics), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("collect", "evaluate"))
    parser.add_argument("--run-dir", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=20260914)
    parser.add_argument("--rounds", type=int, default=8)
    parser.add_argument("--gamma", type=int, default=4)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--shard-count", type=int, default=1)
    parser.add_argument("--shard-index", type=int, default=0)
    args = parser.parse_args()
    if args.command == "collect":
        if args.run_dir is None:
            parser.error("--run-dir is required for collection")
        collect(args)
    else:
        evaluate(args)


if __name__ == "__main__":
    main()
