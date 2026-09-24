"""Nonmember-only count TCN with multiplicity-aware support and fit reuse."""
import copy
from dataclasses import dataclass
import json
import os
from pathlib import Path
import tempfile

import numpy as np
import torch

from experiments.shared.audit.costs import timed
from experiments.shared.audit.main import select_documents
from experiments.shared.methods.conditional_accept_only import ConditionalCountTCN, make_batch, count_nll
from experiments.resource_curves.observations import arrays_digest
from experiments.resource_curves.partitions import partition_indices
from experiments.resource_curves.storage import atomic_json, code_fingerprint, digest, file_sha, workspace

FEATURE_COLUMNS = [0, 4, 1, 2, 3]


def fitting_data(observations, point):
    data = observations.count_data()
    parts = partition_indices(data, point)
    reference = np.unique(np.r_[parts["train"], parts["validation"]])
    sub, _ = select_documents(data, reference)
    lookup = {int(doc): i for i, doc in enumerate(reference)}
    fit_parts = {role: np.asarray([lookup[int(i)] for i in parts[role]]) for role in ("train", "validation")}
    return sub, fit_parts


def fitting_identity(observations, point):
    """Calibration/test outcomes, IDs and sizes cannot select the detector."""
    sub, parts = fitting_data(observations, point)
    contract = observations.contract
    return {"fitting_arrays": arrays_digest({key: sub[key] for key in ("features", "counts", "lengths", "record_ids")}),
            "partitions": {role: sub["record_ids"][indices].tolist() for role, indices in parts.items()},
            "multiplicity": observations.multiplicity,
            "collection": {key: contract[key] for key in ("adapter", "seed", "sources", "sampling", "runtime_sha256")}}


@dataclass
class Detector:
    model: ConditionalCountTCN
    mean: np.ndarray
    scale: np.ndarray
    metadata: dict
    checkpoint: Path
    reused: bool


def _fit(sub, parts, multiplicity, *, seed, device, epochs):
    x = sub["features"][:, FEATURE_COLUMNS]
    counts, lengths = sub["counts"], sub["lengths"]
    offsets = np.r_[0, lengths.cumsum()]
    tokens = np.concatenate([np.arange(offsets[i], offsets[i + 1]) for i in parts["train"]])
    mean, scale = x[tokens].mean(0), x[tokens].std(0)
    scale = np.where(scale < 1e-6, 1., scale)
    standardized = (x - mean) / scale
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    model = ConditionalCountTCN(x.shape[1], multiplicity).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-3)
    best, best_epoch, state, history = np.inf, 0, None, []
    for epoch in range(1, epochs + 1):
        record = {"epoch": epoch}
        for phase in ("train", "validation"):
            model.train(phase == "train")
            indices = rng.permutation(parts[phase]) if phase == "train" else parts[phase]
            total = 0.
            with torch.set_grad_enabled(phase == "train"):
                for start in range(0, len(indices), 16):
                    batch = indices[start:start + 16]
                    bx, by, mask = make_batch(standardized, counts, offsets, batch, device)
                    loss = count_nll(model(bx, mask), by, mask)
                    if not torch.isfinite(loss):
                        raise RuntimeError("nonfinite detector loss")
                    if phase == "train":
                        optimizer.zero_grad()
                        loss.backward()
                        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)
                        optimizer.step()
                    total += float(loss.detach()) * len(batch)
            record[phase + "_nll"] = total / len(indices)
        history.append(record)
        if record["validation_nll"] < best - 1e-5:
            best, best_epoch, state = record["validation_nll"], epoch, copy.deepcopy(model.state_dict())
        if epoch - best_epoch >= 5:
            break
    model.load_state_dict(state)
    return model, mean, scale, history, best_epoch


def fit_detector(observations, point, cache_root, *, seed=20260914, device="cpu", epochs=30):
    """Cache by exact train/validation observations, not calibration allocation."""
    if type(epochs) is not int or epochs < 1:
        raise ValueError("positive detector epoch count required")
    identity = fitting_identity(observations, point)
    contract = {"identity": identity, "seed": seed, "epochs": epochs, "device": str(device),
                "runtime_sha256": code_fingerprint(), "torch": str(torch.__version__),
                "feature_columns": FEATURE_COLUMNS, "architecture": "conditional_count_tcn_24"}
    key = digest(contract)
    with workspace(Path(cache_root) / key) as output:
        checkpoint = output / "detector.pt"
        manifest = output / "FIT.json"
        if manifest.exists():
            metadata = json.loads(manifest.read_text())
            if metadata["contract"] != contract or metadata["sha256"] != file_sha(checkpoint):
                raise ValueError("detector cache contract or checksum mismatch")
            saved = torch.load(checkpoint, map_location=device, weights_only=True)
            model = ConditionalCountTCN(5, observations.multiplicity).to(device)
            model.load_state_dict(saved["state_dict"])
            return Detector(model, saved["mean"].cpu().numpy(), saved["scale"].cpu().numpy(),
                            metadata, checkpoint, True)
        sub, parts = fitting_data(observations, point)
        with timed(device) as elapsed:
            model, mean, scale, history, best = _fit(sub, parts, observations.multiplicity,
                                                   seed=seed, device=device, epochs=epochs)
        fd, temporary = tempfile.mkstemp(prefix=".detector-", dir=output)
        try:
            with os.fdopen(fd, "wb") as stream:
                torch.save({"state_dict": {name: tensor.cpu() for name, tensor in model.state_dict().items()},
                            "mean": torch.tensor(mean), "scale": torch.tensor(scale)}, stream)
            os.replace(temporary, checkpoint)
        finally:
            Path(temporary).unlink(missing_ok=True)
        metadata = {"schema": "resource_detector_v1", "key": key, "contract": contract,
                    "sha256": file_sha(checkpoint), "seconds": elapsed["seconds"],
                    "best_epoch": best, "history": history}
        atomic_json(manifest, metadata)
    return Detector(model, mean, scale, metadata, checkpoint, False)
