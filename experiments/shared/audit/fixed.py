"""Fixed-candidate orchestration for independent drafts and frozen draft heads."""
from __future__ import annotations

import gc
from pathlib import Path

import numpy as np
import torch

from experiments.shared.core.audit_partitions import deployment_partitions
from experiments.shared.core.audit_runtime import _write_json
from experiments.shared.protocols.collect_protocol_observations import collect_records, protocol_prompt_ids
from experiments.shared.protocols.protocol_archive import load_archive
from experiments.shared.models.loading import load_adapter
from experiments.shared.methods.protocol_accept_only import predict
from experiments.shared.protocols.sd_protocol import fixed_trace
from experiments.shared.audit.provenance import already_complete, digest, save_result
from experiments.shared.audit.costs import PHASES, timed, summarize_cost, COST_CONVENTIONS
from experiments.shared.audit.metrics import metrics, METRIC_CONVENTIONS

MAIN_METHODS = {
    "fixed": ("main_fixed_sparse_positive",),
}


# Preserve the established detector fitting and scoring algorithms.
from experiments.shared.audit.main import select_documents, sparse_scores, fit_detector


def run_main(task, device, cfg, prepared, sources):
    """Controlled-SFT/DP entry with its original model and data contracts."""
    if task.get("protocol") not in MAIN_METHODS:
        raise ValueError("unsupported audit protocol")
    from experiments.shared.models.registry import pair_for
    spec = pair_for(task)
    parts = deployment_partitions(prepared.labels, prepared.record_ids, prepared.record_roles,
                                  seed=task["settings"]["audit_seed"])
    return run_prepared_main(
        task, device, prepared, sources, parts=parts, adapter_kind=spec.adapter,
        is_head=spec.is_head, model_pair=spec.name,
        adapter_factory=lambda: load_adapter(Path(task["run_dir"]), spec.adapter, device, task["draft_role"]),
    )


def run_prepared_main(task, device, prepared, sources, *, parts, adapter_factory,
                      adapter_kind="plain", is_head=False, data_contract="four_role_600",
                      report_context=None, model_pair=None):
    """Apply the fixed B=2 method to frozen externally prepared records.

    Callers supply model loading and disjoint partitions. Only independent
    nonmembers fit/validate/calibrate the detector; language models stay frozen.
    """
    if task.get("protocol") not in MAIN_METHODS:
        raise ValueError("unsupported audit protocol")
    leaves = [np.asarray(parts[name]) for name in ("train", "validation", "calibration", "test")]
    if (any(a.ndim != 1 or not len(a) or a.dtype.kind not in "iu" for a in leaves)
            or sorted(np.concatenate(leaves).tolist()) != list(range(len(prepared.records)))):
        raise ValueError("audit partitions must be nonempty, disjoint and exhaustive")
    reference = np.sort(np.concatenate(leaves[:2]))
    if (not np.array_equal(np.sort(parts["reference"]), reference)
            or np.any(prepared.record_roles[np.concatenate(leaves[:3])] != "audit_auxiliary")
            or np.any(prepared.labels[np.concatenate(leaves[:3])] != 0)
            or set(prepared.labels[parts["test"]]) != {0, 1}):
        raise ValueError("only independent nonmembers may fit/validate/calibrate the detector")
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
    if archive.exists() and archive.with_suffix(".npz.json").exists():
        data, envelope = load_archive(archive, check_sources=False)
        if (envelope["contract"].get("matrix_request_key") != request
                or digest(envelope["contract"]["sources"]) != digest(sources)):
            raise ValueError("observation parameters or frozen sources changed")
        if is_head and (envelope["contract"].get("head_validation") or {}).get("status") != "passed":
            raise ValueError("head observations lack the prefix-consistency validation gate")
    else:
        adapter = adapter_factory()
        record = prepared.records[int(parts["train"][0])]
        prompt = protocol_prompt_ids(record, prepared.tokenizer)
        response = list(record.response_ids)
        validation = None
        if is_head:
            from experiments.shared.models.validation import validate_adapter
            validation = validate_adapter(adapter, prompt, response, seed=settings["audit_seed"])
        # One untimed warmup on a training auxiliary, never the test/calibration set.
        fixed_trace(adapter, prompt, response, seed=settings["audit_seed"])
        contract = dict(
            protocol=protocol, adapter=adapter_kind, draft_role=task["draft_role"],
            starts=["fixed"],
            rounds_per_start=0,
            seed=settings["audit_seed"], sources=sources, matrix_request_key=request,
            data_contract=data_contract, execution="full_context_reconstruction",
            head_real_model_validation="not_applicable" if not is_head else "prefix_consistency_checked",
            head_validation=validation, timing_version=1,
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
    if is_head:
        counters.update(hidden_state_bytes=0, supported_candidates=0, candidate_positions=0)
    peaks = []
    for cost in envelope["costs"]:
        if "seconds" not in cost:
            raise ValueError("legacy unmeasured observations cannot supply matrix costs")
        collection[document_phase[cost["record_id"]]] += cost["seconds"]
        for dest, source in (("target_sequences", "target_forward_calls"), ("target_forward_calls", "target_forward_calls"),
                             ("draft_sequences", "draft_forward_calls"), ("target_input_tokens", "target_input_tokens"),
                             ("draft_input_tokens", "draft_input_tokens"), ("generated_tokens", "generated_tokens")):
            counters[dest] += cost[source]
        if is_head:
            for name in ("hidden_state_bytes", "supported_candidates", "candidate_positions"):
                counters[name] += cost[name]
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
            model_pair=model_pair or task["model_pair"], adapter=adapter_kind,
            metrics=metrics(values, labels, cal, test, seed=settings["audit_seed"]),
            metric_conventions=METRIC_CONVENTIONS, cost=cost, cost_conventions=COST_CONVENTIONS,
            access_channel=("draft_features_and_acceptance_only" if not is_head
                            else "edge_head_target_hidden_states_and_acceptance"),
            detector_features="draft_features_and_acceptance_only", training_member_count=0,
            head_validation=envelope["contract"].get("head_validation"),
            fit=fit_meta, detector={"file": "../detector.pt", "sha256": fit_meta["sha256"]},
            collection_phase_seconds=collection, inference_seconds=inference,
            hardware=envelope["contract"]["hardware"],
            observation_archive={"path": str(archive), "sha256": envelope["archive_sha256"]},
        )
        if report_context is not None:
            report["evaluation_context"] = report_context
        save_result(output / method, record_ids=ids, labels=labels, scores=values, calibration=cal, test=test, report=report)
