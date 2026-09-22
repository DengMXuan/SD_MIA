"""Fixed-candidate orchestration for independent drafts and frozen draft heads."""
from __future__ import annotations

import gc
from pathlib import Path

import numpy as np
import torch

from experiments.sd_membership_sft.core.audit_partitions import deployment_partitions
from experiments.sd_membership_sft.core.audit_runtime import _write_json
from experiments.sd_membership_sft.protocols.collect_protocol_observations import collect_records, protocol_prompt_ids
from experiments.sd_membership_sft.protocols.protocol_archive import load_archive
from experiments.cross_model_audit.models import load_adapter
from experiments.sd_membership_sft.methods.protocol_accept_only import predict
from experiments.sd_membership_sft.protocols.sd_protocol import fixed_trace
from experiments.cross_model_audit.artifacts import already_complete, digest, save_result
from experiments.sd_membership_sft.audit.matrix_costs import PHASES, timed, summarize_cost, COST_CONVENTIONS
from experiments.sd_membership_sft.audit.matrix_metrics import metrics, METRIC_CONVENTIONS

MAIN_METHODS = {
    "fixed": ("main_fixed_sparse_positive",),
}


# Preserve the established detector fitting and scoring algorithms.
from experiments.sd_membership_sft.audit.matrix_main import select_documents, sparse_scores, fit_detector


def run_main(task, device, cfg, prepared, sources):
    from experiments.cross_model_audit.storage import prepare_audit_cache
    from experiments.cross_model_audit.model_registry import pair_for
    adapter_kind = pair_for(task).adapter
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
    parts = deployment_partitions(prepared.labels, prepared.record_ids, prepared.record_roles)
    if archive.exists() and archive.with_suffix(".npz.json").exists():
        data, envelope = load_archive(archive, check_sources=False)
        if (envelope["contract"].get("matrix_request_key") != request
                or digest(envelope["contract"]["sources"]) != digest(sources)):
            raise ValueError("observation parameters or frozen sources changed")
        if adapter_kind != "plain" and (envelope["contract"].get("head_validation") or {}).get("status") != "passed":
            raise ValueError("head observations lack the prefix-consistency validation gate")
    else:
        adapter = load_adapter(Path(task["run_dir"]), adapter_kind, device, task["draft_role"])
        record = prepared.records[int(parts["train"][0])]
        prompt = protocol_prompt_ids(record, prepared.tokenizer)
        response = list(record.response_ids)
        validation = None
        if adapter_kind != "plain":
            from experiments.cross_model_audit.head_validation import validate_adapter
            validation = validate_adapter(adapter, prompt, response)
        # One untimed warmup on a training auxiliary, never the test/calibration set.
        fixed_trace(adapter, prompt, response, seed=settings["audit_seed"])
        contract = dict(
            protocol=protocol, adapter=adapter_kind, draft_role=task["draft_role"],
            starts=["fixed"],
            rounds_per_start=0,
            seed=settings["audit_seed"], sources=sources, matrix_request_key=request,
            data_contract="four_role_600", execution="full_context_reconstruction",
            head_real_model_validation="not_applicable" if adapter_kind == "plain" else "prefix_consistency_checked",
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
    if adapter_kind != "plain":
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
        if adapter_kind != "plain":
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
            model_pair=pair_for(task).name, adapter=adapter_kind,
            metrics=metrics(values, labels, cal, test, seed=settings["audit_seed"]),
            metric_conventions=METRIC_CONVENTIONS, cost=cost, cost_conventions=COST_CONVENTIONS,
            access_channel=("draft_features_and_acceptance_only" if adapter_kind == "plain"
                            else "edge_head_target_hidden_states_and_acceptance"),
            detector_features="draft_features_and_acceptance_only", training_member_count=0,
            head_validation=envelope["contract"].get("head_validation"),
            fit=fit_meta, detector={"file": "../detector.pt", "sha256": fit_meta["sha256"]},
            collection_phase_seconds=collection, inference_seconds=inference,
            hardware=envelope["contract"]["hardware"],
            observation_archive={"path": str(archive), "sha256": envelope["archive_sha256"]},
        )
        save_result(output / method, record_ids=ids, labels=labels, scores=values, calibration=cal, test=test, report=report)
