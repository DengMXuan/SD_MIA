"""Measured fixed-candidate main method for one draft condition."""
from __future__ import annotations

import gc
import json
from pathlib import Path

import numpy as np
from scipy.special import logsumexp
import torch

from experiments.shared.core.audit_partitions import deployment_partitions
from experiments.shared.core.audit_runtime import _write_json
from experiments.shared.protocols.collect_protocol_observations import collect_records, protocol_prompt_ids
from experiments.shared.methods.conditional_accept_only import ConditionalCountTCN
from experiments.shared.core.deployment_archive import sha256_file
from experiments.shared.protocols.protocol_archive import load_archive
from experiments.shared.models.loading import load_adapter
from experiments.shared.methods.protocol_accept_only import trajectory_partitions, predict
from experiments.shared.protocols.sd_protocol import fixed_trace
from experiments.shared.audit.artifacts import already_complete, digest, save_result
from experiments.shared.audit.costs import PHASES, timed, peak_memory, summarize_cost, COST_CONVENTIONS
from experiments.shared.audit.metrics import metrics, METRIC_CONVENTIONS

MAIN_METHODS = {
    "fixed": ("main_fixed_sparse_positive",),
}


def select_documents(data, documents):
    trajectories = np.flatnonzero(np.isin(data["document_indices"], documents))
    offsets = np.r_[0, data["lengths"].cumsum()]
    tokens = np.concatenate([np.arange(offsets[t], offsets[t + 1]) for t in trajectories])
    lookup = {int(doc): i for i, doc in enumerate(documents)}
    return dict(features=data["features"][tokens], counts=data["counts"][tokens],
                lengths=data["lengths"][trajectories],
                document_indices=np.asarray([lookup[int(d)] for d in data["document_indices"][trajectories]]),
                record_ids=data["record_ids"][documents]), tokens


def sparse_scores(data, logpmf, two_sided):
    eta = np.array([.5, 1., 2., -.5, -1., -2.] if two_sided else [.5, 1., 2.])
    k = np.arange(logpmf.shape[1])
    evidence = data["counts"].astype(float)[:, None] * eta - logsumexp(
        logpmf[:, :, None] + k[None, :, None] * eta, axis=1)
    local = np.stack([np.logaddexp(np.log1p(-rho), np.log(rho) + evidence)
                      for rho in (.05, .1, .25)], axis=1)
    summed = np.zeros((len(data["record_ids"]), 3, len(eta)))
    offsets = np.r_[0, data["lengths"].cumsum()]
    for t, doc in enumerate(data["document_indices"]):
        summed[doc] += local[offsets[t]:offsets[t + 1]].sum(0)
    return logsumexp(summed.reshape(len(summed), -1), axis=1) - np.log(3 * len(eta))


def fit_detector(data, parts, task, output):
    settings, protocol = task["settings"], task["protocol"]
    key = digest({"task": task, "archive": sha256_file(output / "observations.npz")})
    manifest = output / "FIT.json"
    with timed() as fitting:
        x = data["features"][:, [0, 4, 1, 2, 3]]
        if manifest.exists():
            metadata = json.loads(manifest.read_text())
            if metadata["key"] != key or metadata["sha256"] != sha256_file(output / "detector.pt"):
                raise ValueError("detector source/cache mismatch")
            checkpoint = torch.load(output / "detector.pt", map_location="cpu", weights_only=True)
            model = ConditionalCountTCN(x.shape[1], 2)
            model.load_state_dict(checkpoint["state_dict"])
            return model, x, checkpoint["mean"].numpy(), checkpoint["scale"].numpy(), metadata
        from experiments.shared.methods.difficulty_accept_only import fit
        model, mean, scale, history, best_epoch = fit(
            x, data["counts"], data["lengths"], trajectory_partitions(parts, data["document_indices"]),
            seed=settings["audit_seed"], device="cpu", epochs=settings["detector_epochs"],
        )
    temporary = output / ".detector.tmp.pt"
    torch.save({"state_dict": model.state_dict(), "mean": torch.tensor(mean), "scale": torch.tensor(scale)}, temporary)
    temporary.replace((output / "detector.pt").resolve())
    metadata = dict(key=key, sha256=sha256_file(output / "detector.pt"), seconds=fitting["seconds"],
                    history=history, best_epoch=best_epoch, architecture="count_tcn")
    _write_json(manifest, metadata)
    return model, x, mean, scale, metadata


def run_main(task, device, cfg, prepared, sources):
    if task.get("protocol") not in MAIN_METHODS:
        raise ValueError("unsupported audit protocol")
    from experiments.paths import prepare_audit_cache
    output = Path(task["output"])
    output.mkdir(parents=True, exist_ok=True)
    prepare_audit_cache(output)
    settings, protocol = task["settings"], task["protocol"]
    methods = MAIN_METHODS[protocol]
    remaining = [method for method in methods if not already_complete(
        output / method, digest({"task": task, "method": method}), sources)]
    if not remaining:
        return
    request = digest(task)
    archive = output / "observations.npz"
    parts = deployment_partitions(prepared.labels, prepared.record_ids, prepared.record_roles,
                                  seed=settings["audit_seed"])
    if archive.exists() and archive.with_suffix(".npz.json").exists():
        data, envelope = load_archive(archive, check_sources=False)
        if (envelope["contract"].get("matrix_request_key") != request
                or digest(envelope["contract"]["sources"]) != digest(sources)):
            raise ValueError("observation parameters or frozen sources changed")
    else:
        adapter = load_adapter(Path(task["run_dir"]), "plain", device, task["draft_role"])
        record = prepared.records[int(parts["train"][0])]
        prompt = protocol_prompt_ids(record, prepared.tokenizer)
        response = list(record.response_ids)
        # One untimed warmup on a training auxiliary, never the test/calibration set.
        fixed_trace(adapter, prompt, response, seed=settings["audit_seed"])
        contract = dict(
            protocol=protocol, adapter="plain", draft_role=task["draft_role"],
            starts=["fixed"],
            rounds_per_start=0,
            seed=settings["audit_seed"], sources=sources, matrix_request_key=request,
            data_contract="four_role_600", execution="full_context_reconstruction",
            head_real_model_validation="not_applicable", timing_version=1,
            hardware=dict(name=torch.cuda.get_device_name(device) if torch.device(device).type == "cuda" else "cpu",
                          device=str(device), attention="sdpa", torch=str(torch.__version__),
                          dtype=str(next(adapter.target.parameters()).dtype)),
        )
        collect_records(prepared, adapter, output, contract)
        del adapter
        gc.collect()
        if torch.device(device).type == "cuda":
            torch.cuda.empty_cache()
        data, envelope = load_archive(archive, check_sources=False)
    for name in ("record_ids", "record_roles", "labels"):
        if not np.array_equal(data[name], getattr(prepared, name)):
            raise ValueError(f"observation {name} differ from frozen records")
    model, x, mean, scale, fit_meta = fit_detector(data, parts, task, output)
    phase_data, pmfs, inference = {}, {}, {}
    for phase, partition in (("calibration", "calibration"), ("test", "test")):
        with timed() as elapsed:
            sub, tokens = select_documents(data, parts[partition])
            pmfs[phase] = predict(model, (x[tokens] - mean) / scale, sub["counts"], sub["lengths"], "cpu")
        phase_data[phase], inference[phase] = sub, elapsed["seconds"]
    collection = dict.fromkeys(PHASES, 0.)
    document_phase = {prepared.record_ids[i]: name for name, indices in
                      (("preparation", parts["reference"]), ("calibration", parts["calibration"]), ("test", parts["test"]))
                      for i in indices}
    counters = dict.fromkeys(("target_sequences", "target_forward_calls", "draft_sequences",
                              "target_input_tokens", "draft_input_tokens", "generated_tokens"), 0)
    peaks = []
    for cost in envelope["costs"]:
        if "seconds" not in cost:
            raise ValueError("legacy unmeasured observations cannot supply matrix costs")
        collection[document_phase[cost["record_id"]]] += cost["seconds"]
        for dest, source in (("target_sequences", "target_forward_calls"), ("target_forward_calls", "target_forward_calls"),
                             ("draft_sequences", "draft_forward_calls"), ("target_input_tokens", "target_input_tokens"),
                             ("draft_input_tokens", "draft_input_tokens"), ("generated_tokens", "generated_tokens")):
            counters[dest] += cost[source]
        if cost.get("peak_allocated_gpu_bytes") is not None:
            peaks.append(cost["peak_allocated_gpu_bytes"])
    selected = np.r_[parts["calibration"], parts["test"]]
    labels, ids = data["labels"][selected], data["record_ids"][selected]
    cal, test = np.arange(len(parts["calibration"])), np.arange(len(parts["calibration"]), len(selected))
    method_outputs = {}
    common_seconds = sum(collection.values()) + fit_meta["seconds"] + sum(inference.values())
    all_variant_seconds = 0.
    for method in methods:
        two_sided = method.endswith("two_sided")
        with timed() as cal_time:
            cal_scores = sparse_scores(phase_data["calibration"], pmfs["calibration"], two_sided)
            ordered = np.sort(cal_scores)
        with timed() as test_time:
            test_scores = sparse_scores(phase_data["test"], pmfs["test"], two_sided)
            (1 + len(ordered) - np.searchsorted(ordered, test_scores, side="left")) / (len(ordered) + 1)
        all_variant_seconds += cal_time["seconds"] + test_time["seconds"]
        phases = {**collection}
        phases["preparation"] += fit_meta["seconds"]
        phases["calibration"] += inference["calibration"] + cal_time["seconds"]
        phases["test"] += inference["test"] + test_time["seconds"]
        method_outputs[method] = (np.r_[cal_scores, test_scores], phases)
    for method in remaining:
        values, phases = method_outputs[method]
        cost = summarize_cost(phases, len(test), counters, max(peaks) if peaks else None, execution_group=task["id"])
        cost["execution_group_seconds"] = common_seconds + all_variant_seconds
        report = dict(
            method=method, request_key=digest({"task": task, "method": method}), sources=sources,
            condition=task["condition"], settings=settings, draft_role=task["draft_role"],
            metrics=metrics(values, labels, cal, test, seed=settings["audit_seed"]),
            metric_conventions=METRIC_CONVENTIONS, cost=cost, cost_conventions=COST_CONVENTIONS,
            access_channel="draft_features_and_acceptance_only", training_member_count=0,
            fit=fit_meta, detector={"file": "../detector.pt", "sha256": fit_meta["sha256"]},
            collection_phase_seconds=collection, inference_seconds=inference,
            hardware=envelope["contract"]["hardware"],
            observation_archive={"path": str(archive), "sha256": envelope["archive_sha256"]},
        )
        save_result(output / method, record_ids=ids, labels=labels, scores=values, calibration=cal, test=test, report=report)
