"""Fresh multiplicity-aware fixed probes and an observable-only archive."""
from dataclasses import asdict, dataclass
import hashlib
import json
from pathlib import Path

import numpy as np
import torch

from experiments.shared.audit.costs import timed, peak_memory, reset_peak
from experiments.shared.data.data import _hash_ids
from experiments.shared.protocols.collect_protocol_observations import protocol_prompt_ids
from experiments.shared.protocols.sd_protocol import draft_features, trajectory_seed
from experiments.resource_curves.config import check_multiplicity
from experiments.resource_curves.storage import atomic_json, atomic_npz, checked_contract, code_fingerprint, digest, file_sha, workspace

ARRAY_KEYS = {"features", "bits", "lengths", "record_ids", "record_roles", "labels", "candidate_positions"}


@torch.inference_mode()
def fixed_trace(adapter, prompt, response, *, seed, multiplicity=2):
    """Repeated judgments, not repeated forward passes or copied feedback.

    Independent streams of pairs retain the historical B=2 random sequence.
    Lower budgets are exact prefixes of the same observed bits at higher B.
    Generating an unused uniform at B=1 does not evaluate another judgment.
    """
    check_multiplicity(multiplicity)
    if not prompt or not response:
        raise ValueError("nonempty prompt and response required")
    generators = [torch.Generator(device=adapter.device).manual_seed(
        seed if pair == 0 else trajectory_seed(seed, "query_pair", str(pair)))
        for pair in range(max(1, multiplicity // 2))]
    logp, logq = adapter.rows(list(prompt) + list(response))
    features, bits = [], []
    for index, token in enumerate(response):
        row = len(prompt) + index - 1
        if not torch.isfinite(logq[row, token]):
            continue
        alpha = torch.exp(torch.minimum(logp[row, token] - logq[row, token], logp.new_zeros(())))
        uniforms = torch.cat([torch.rand(2, device=adapter.device, generator=g) for g in generators])
        features.append(draft_features(logq[row], token, index / max(1, len(response) - 1), 0.))
        bits.append((uniforms[:multiplicity] < alpha).cpu().numpy().astype(np.uint8))
    if not bits:
        raise ValueError("document has no supported fixed candidates")
    return {"features": np.asarray(features, np.float32), "bits": np.asarray(bits, np.uint8),
            "candidate_positions": len(response), "supported_candidates": len(bits),
            "acceptance_judgments": len(bits) * multiplicity}


def validate_arrays(data, multiplicity):
    check_multiplicity(multiplicity)
    if set(data) != ARRAY_KEYS:
        raise ValueError("unexpected archive fields; only observable features, bits and metadata allowed")
    lengths, ids = data["lengths"], data["record_ids"]
    if (lengths.ndim != 1 or not len(lengths) or lengths.dtype.kind not in "iu"
            or (lengths <= 0).any() or ids.shape != lengths.shape or ids.dtype.kind not in "US"
            or (ids == "").any() or len(np.unique(ids)) != len(ids)):
        raise ValueError("invalid document lengths or IDs")
    n = int(lengths.sum())
    if (data["features"].shape != (n, 6) or not np.isfinite(data["features"]).all()
            or (data["features"][:, 0] > 1e-5).any()
            or data["bits"].shape != (n, multiplicity) or not np.isin(data["bits"], (0, 1)).all()):
        raise ValueError("invalid draft features or fresh binary outcomes")
    roles, labels, positions = data["record_roles"], data["labels"], data["candidate_positions"]
    if (roles.shape != ids.shape or labels.shape != ids.shape or positions.shape != ids.shape
            or not np.isin(roles, ("audit_auxiliary", "member", "nonmember")).all()
            or not np.array_equal(labels, (roles == "member").astype(int))
            or positions.dtype.kind not in "iu" or (positions < lengths).any()):
        raise ValueError("invalid role/label or candidate-coverage contract")


def arrays_digest(data):
    h = hashlib.sha256()
    for key, array in sorted(data.items()):
        h.update(digest([key, str(array.dtype), list(array.shape)]).encode())
        h.update(np.ascontiguousarray(array).tobytes())
    return h.hexdigest()


@dataclass
class Observations:
    arrays: dict
    contract: dict
    costs: list
    multiplicity: int

    def __post_init__(self):
        validate_arrays(self.arrays, self.multiplicity)
        collected = self.contract["multiplicity"]
        check_multiplicity(collected)
        if collected < self.multiplicity:
            raise ValueError("cannot invent unobserved acceptance outcomes")
        if len(self.costs) != len(self.arrays["record_ids"]):
            raise ValueError("missing per-document collection costs")
        for i, cost in enumerate(self.costs):
            if (cost["record_id"] != str(self.arrays["record_ids"][i])
                    or cost["acceptance_judgments"] != int(self.arrays["lengths"][i]) * collected
                    or not np.isfinite(cost["seconds"]) or cost["seconds"] < 0):
                raise ValueError("collection cost alignment failed")

    def view(self, multiplicity):
        check_multiplicity(multiplicity)
        if multiplicity > self.multiplicity:
            raise ValueError("requested judgments were not recorded; collect fresh observations")
        return Observations({**self.arrays, "bits": self.arrays["bits"][:, :multiplicity]},
                            self.contract, self.costs, multiplicity)

    def count_data(self):
        return {**self.arrays, "counts": self.arrays["bits"].sum(axis=1).astype(np.int64),
                "document_indices": np.arange(len(self.arrays["lengths"]))}

    @property
    def signature(self):
        return digest({"arrays": arrays_digest(self.arrays), "contract": self.contract,
                       "multiplicity": self.multiplicity})


def load_observations(folder, *, expected_contract=None):
    folder = Path(folder)
    meta = json.loads((folder / "OBSERVATIONS.json").read_text())
    if meta.get("schema") != "resource_observations_v1":
        raise ValueError("unsupported observation schema (legacy aggregate counts cannot supply bits)")
    if expected_contract is not None and meta["contract"] != expected_contract:
        raise ValueError("observation sources or parameters changed")
    if meta["sha256"] != file_sha(folder / "observations.npz"):
        raise ValueError("observation checksum mismatch")
    with np.load(folder / "observations.npz", allow_pickle=False) as saved:
        arrays = dict(saved)
    result = Observations(arrays, meta["contract"], meta["costs"], meta["contract"]["multiplicity"])
    if (result.arrays["record_ids"].tolist() != result.contract["record_ids"]
            or result.arrays["record_roles"].tolist() != result.contract["record_roles"]):
        raise ValueError("archive records differ from the collection contract")
    return result


def collect_observations(prepared, adapter, output, *, sources, seed, multiplicity=2):
    """Measured collection with per-document recovery, in a separate workspace.

    ``prepared`` uses the existing records/tokenizer/record_ids/record_roles/
    labels interface; the caller supplies fresh model and data fingerprints.
    Model loading and any caller warmup are excluded from collection timing.
    """
    check_multiplicity(multiplicity)
    if not sources:
        raise ValueError("frozen model and data provenance is required")
    ids = [record.record_id for record in prepared.records]
    if ids != prepared.record_ids.tolist() or len(set(ids)) != len(ids) or not ids:
        raise ValueError("prepared records/IDs are not unique and aligned")
    roles = np.asarray(prepared.record_roles)
    if (roles.shape != (len(ids),) or not np.isin(roles, ("audit_auxiliary", "member", "nonmember")).all()
            or not np.array_equal(prepared.labels, (roles == "member").astype(int))):
        raise ValueError("invalid prepared role/label contract")
    inputs = [(protocol_prompt_ids(record, prepared.tokenizer), list(record.response_ids))
              for record in prepared.records]
    contract = {"schema": "resource_collection_v1", "protocol": "fixed", "adapter": adapter.kind,
                "multiplicity": multiplicity, "seed": seed, "sources": sources,
                "runtime_sha256": code_fingerprint(), "record_ids": ids,
                "record_roles": roles.tolist(), "sampling": "nested_pair_streams_v1",
                "input_hashes": [[_hash_ids(p), _hash_ids(r)] for p, r in inputs],
                "hardware": {"device": str(adapter.device), "torch": str(torch.__version__),
                             "name": torch.cuda.get_device_name(adapter.device)
                             if torch.device(adapter.device).type == "cuda" else "cpu"}}
    stamp = digest(contract)
    with workspace(output) as folder:
        checked_contract(folder, "COLLECTION.json", contract)
        if (folder / "OBSERVATIONS.json").exists():
            return load_observations(folder, expected_contract=contract)
        records_dir = folder / "records"
        records_dir.mkdir(exist_ok=True)
        traces, costs = [], []
        for index, (prompt, response) in enumerate(inputs):
            path = records_dir / f"{index}.npz"
            sidecar = path.with_suffix(".json")
            if path.exists() and sidecar.exists():
                meta = json.loads(sidecar.read_text())
                if meta["contract_digest"] != stamp or meta["sha256"] != file_sha(path):
                    raise ValueError("cached observation changed or belongs to a different contract")
                with np.load(path, allow_pickle=False) as saved:
                    trace = dict(saved)
                cost = meta["cost"]
            else:
                before = asdict(adapter.cost)
                reset_peak(adapter.device)
                with timed(adapter.device) as elapsed:
                    trace = fixed_trace(adapter, prompt, response,
                                        seed=trajectory_seed(seed, ids[index], "fixed"), multiplicity=multiplicity)
                cost = {name: value - before[name] for name, value in asdict(adapter.cost).items()}
                cost.update({key: value for key, value in trace.items() if key not in ("features", "bits")})
                cost.update(record_id=ids[index], seconds=elapsed["seconds"],
                            peak_allocated_gpu_bytes=peak_memory(adapter.device))
                trace = {key: trace[key] for key in ("features", "bits")}
                atomic_npz(path, trace)
                atomic_json(sidecar, {"contract_digest": stamp, "sha256": file_sha(path), "cost": cost})
            if set(trace) != {"features", "bits"}:
                raise ValueError("unexpected per-record observation fields")
            traces.append(trace)
            costs.append(cost)
        arrays = {key: np.concatenate([trace[key] for trace in traces]) for key in ("features", "bits")}
        arrays.update(lengths=np.asarray([len(trace["bits"]) for trace in traces], np.int64),
                      record_ids=np.asarray(ids), record_roles=roles, labels=np.asarray(prepared.labels),
                      candidate_positions=np.asarray([cost["candidate_positions"] for cost in costs], np.int64))
        result = Observations(arrays, contract, costs, multiplicity)
        atomic_npz(folder / "observations.npz", arrays)
        atomic_json(folder / "OBSERVATIONS.json", {"schema": "resource_observations_v1", "contract": contract,
                                                   "sha256": file_sha(folder / "observations.npz"), "costs": costs})
        return result
