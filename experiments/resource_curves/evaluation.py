"""Existing sparse main-method metrics with explicit resource/timing provenance."""
import json

import numpy as np

from experiments.shared.audit.costs import timed
from experiments.shared.audit.main import select_documents, sparse_scores
from experiments.shared.audit.metrics import metrics, METRIC_CONVENTIONS
from experiments.shared.methods.protocol_accept_only import predict
from experiments.resource_curves.detector import FEATURE_COLUMNS, fitting_identity
from experiments.resource_curves.config import condition_seed
from experiments.resource_curves.partitions import partition_indices
from experiments.resource_curves.storage import atomic_json, atomic_npz, checked_contract, code_fingerprint, file_sha, workspace


def collection_cost(observations, parts):
    selected = np.concatenate([parts[role] for role in ("train", "validation", "calibration", "test")])
    costs = [observations.costs[int(i)] for i in selected]
    matching = observations.multiplicity == observations.contract["multiplicity"]
    phase_seconds = {role: sum(observations.costs[int(i)]["seconds"] for i in parts[role])
                     for role in ("train", "validation", "calibration", "test")}
    return {
        "acceptance_judgments": int(observations.arrays["lengths"][selected].sum()) * observations.multiplicity,
        "test_acceptance_judgments": int(observations.arrays["lengths"][parts["test"]].sum()) * observations.multiplicity,
        "supported_candidates": int(observations.arrays["lengths"][selected].sum()),
        "candidate_positions": int(observations.arrays["candidate_positions"][selected].sum()),
        "collected_multiplicity": observations.contract["multiplicity"],
        "selected_multiplicity": observations.multiplicity,
        "timing_origin": "measured_same_budget_archive" if matching else "higher_budget_view_no_matching_timing",
        "collection_seconds": sum(phase_seconds.values()) if matching else None,
        "collection_phase_seconds": phase_seconds if matching else None,
        "origin_collection_seconds": sum(phase_seconds.values()),
        "origin_counters": {key: sum(cost[key] for cost in costs)
                            for key in ("acceptance_judgments", "target_forward_calls", "draft_forward_calls",
                                        "target_input_tokens", "draft_input_tokens", "hidden_state_bytes")},
    }


def evaluate(observations, point, detector, output, *, two_sided=False, bootstrap=200, metric_seed=None):
    """Save one future curve point. This API never schedules an experiment matrix."""
    if detector.metadata["contract"]["identity"] != fitting_identity(observations, point):
        raise ValueError("detector was fitted on a different training/validation allocation or budget")
    if type(bootstrap) is not int or bootstrap < 0:
        raise ValueError("bootstrap must be a nonnegative integer")
    metric_seed = condition_seed(observations.contract["seed"], metric_seed)
    condition_seed(observations.contract["seed"], detector.metadata["contract"]["seed"])
    data = observations.count_data()
    parts = partition_indices(data, point)
    contract = {"schema": "resource_evaluation_v1", "observations": observations.signature,
                "point": point, "fit_key": detector.metadata["key"], "two_sided": two_sided,
                "bootstrap": bootstrap, "metric_seed": metric_seed, "runtime_sha256": code_fingerprint()}
    with workspace(output) as folder:
        checked_contract(folder, "EVALUATION.json", contract)
        report_path = folder / "REPORT.json"
        if report_path.exists():
            report = json.loads(report_path.read_text())
            if (report["contract"] != contract or report["scores_sha256"] != file_sha(folder / "scores.npz")
                    or report["detector"]["sha256"] != file_sha(detector.checkpoint)):
                raise ValueError("completed result or detector checksum mismatch")
            return report
        device = next(detector.model.parameters()).device
        scores, times = {}, {}
        for role in ("calibration", "test"):
            with timed(device) as elapsed:
                sub, _ = select_documents(data, parts[role])
                x = (sub["features"][:, FEATURE_COLUMNS] - detector.mean) / detector.scale
                pmf = predict(detector.model, x, sub["counts"], sub["lengths"], device)
                scores[role] = sparse_scores(sub, pmf, two_sided)
                if role == "calibration":
                    ordered = np.sort(scores[role])
                else:
                    pvalues = (1 + len(ordered) - np.searchsorted(ordered, scores[role], side="left")) / (len(ordered) + 1)
            times[role + "_scoring_seconds"] = elapsed["seconds"]
        selected = np.r_[parts["calibration"], parts["test"]]
        labels, ids = data["labels"][selected], data["record_ids"][selected]
        values = np.r_[scores["calibration"], scores["test"]]
        cal = np.arange(len(parts["calibration"]))
        test = np.arange(len(cal), len(selected))
        cost = collection_cost(observations, parts)
        cost.update(times, detector_fit_seconds=detector.metadata["seconds"],
                    detector_reused=detector.reused)
        # Fitting happened in fit_detector, not in evaluate. Report its recorded
        # standalone duration separately from this call's scoring duration.
        online = times["test_scoring_seconds"]
        if cost["collection_seconds"] is not None:
            online += cost["collection_phase_seconds"]["test"]
            cost["standalone_total_seconds"] = cost["collection_seconds"] + detector.metadata["seconds"] + sum(times.values())
            cost["test_ms_per_record"] = 1000 * online / len(test)
        else:
            cost["standalone_total_seconds"] = cost["test_ms_per_record"] = None
        atomic_npz(folder / "scores.npz", {"record_ids": ids, "labels": labels, "scores": values,
                                           "calibration": cal, "test": test, "test_pvalues": pvalues})
        report = {"schema": "resource_curve_result_v1", "contract": contract,
                  "method": "main_fixed_sparse_two_sided" if two_sided else "main_fixed_sparse_positive",
                  "multiplicity": observations.multiplicity, "auxiliary_budget": point["budget"],
                  "metrics": metrics(values, labels, cal, test, seed=metric_seed, bootstrap=bootstrap),
                  "metric_conventions": METRIC_CONVENTIONS, "cost": cost,
                  "cost_conventions": {"queries": "fresh acceptance judgments, not forward calls or network requests",
                                       "timing": "synchronized collection and detector phases; no runtime extrapolation",
                                       "reuse": "collection/fit durations come from their original measured execution",
                                       "exclusions": "model loading, warmup, archive I/O and metric/bootstrap reporting"},
                  "training_member_count": 0, "language_models_frozen": True,
                  "sources": observations.contract["sources"], "hardware": observations.contract["hardware"],
                  "detector": {"path": str(detector.checkpoint.resolve()), "sha256": detector.metadata["sha256"],
                               "key": detector.metadata["key"]},
                  "scores_sha256": file_sha(folder / "scores.npz")}
        atomic_json(report_path, report)
    return report
