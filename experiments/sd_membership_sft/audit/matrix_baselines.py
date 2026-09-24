"""Four-role adapter for existing baseline implementations (kept unchanged)."""
from __future__ import annotations

from collections import defaultdict
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
import json
import traceback

import numpy as np
import torch

from experiments.baseline import METHODS
from experiments.baseline.run import AuditRecord, TargetScorer, _score_methods
from experiments.baseline.costs import CostMeter
from experiments.sd_membership_sft.core.audit_partitions import deployment_partitions
from experiments.sd_membership_sft.finetune.generalization import load_finetuned_model
from experiments.sd_membership_sft.finetune.training import set_seed
from experiments.sd_membership_sft.protocols.collect_protocol_observations import protocol_prompt_ids
from experiments.sd_membership_sft.audit.matrix_artifacts import already_complete, digest, save_result
from experiments.sd_membership_sft.audit.matrix_costs import PHASES, COST_CONVENTIONS, timed, reset_peak, peak_memory, summarize_cost
from experiments.sd_membership_sft.audit.matrix_metrics import metrics, METRIC_CONVENTIONS

BASELINE_DEFAULTS = dict(k_percent=20., recall_shots=4, icp_top_k=5, icp_aggregation="min",
                         sead_samples=50, sead_temperature=1., samia_samples=10,
                         prefix_ratio=.5, perturbation_rate=.15, generation_batch_size=8)


class PhaseMeter:
    def __init__(self, device, n_test):
        self.phase = "preparation"
        self.meters = {name: CostMeter(device, n_test) for name in PHASES}

    def forward(self, input_tokens):
        self.meters[self.phase].forward(input_tokens)

    def generation(self, inputs, outputs, eos_ids):
        self.meters[self.phase].generation(inputs, outputs, eos_ids)


class PhaseProgress:
    """Reuse baseline's iteration boundaries to separate calibration/test work."""
    def __init__(self, meter, device, n_calibration, batch_size):
        self.meter, self.device = meter, device
        self.n_calibration, self.batch_size = n_calibration, batch_size
        self.seconds = dict.fromkeys(PHASES, 0.)

    def track(self, values, stage, unit="records"):
        total = len(values)
        for index, value in enumerate(values):
            if stage == "petal calibration":
                phase = "preparation"  # regression fitting, NOT threshold calibration
            else:
                position = int(value) if unit == "batches" else index
                if unit == "batches" and position < self.n_calibration < position + self.batch_size:
                    raise ValueError("baseline generation batch crosses calibration/test boundary")
                phase = "calibration" if position < self.n_calibration else "test"
            self.meter.phase = phase
            with timed(self.device) as elapsed:
                yield value
            self.seconds[phase] += elapsed["seconds"]
            self.meter.phase = "preparation"
            if index % 25 == 0 or index + 1 == total:
                print(json.dumps({"stage": stage, "completed": index + 1, "total": total}), flush=True)


def access_channel(method):
    return "target_generated_text" if method in ("ws", "rs", "bt", "samia") else "target_probabilities_or_target_features"


def run_baselines(task, device, cfg, prepared, sources):
    output = Path(task["output"])
    methods = task["methods"]
    settings = task["settings"]
    args = SimpleNamespace(**settings["baseline"], seed=settings["audit_seed"])
    parts = deployment_partitions(prepared.labels, prepared.record_ids, prepared.record_roles)
    selected = np.r_[parts["calibration"], parts["test"]]
    cal = np.arange(len(parts["calibration"]))
    test = np.arange(len(cal), len(selected))
    labels, ids = prepared.labels[selected], prepared.record_ids[selected]
    def record_for(index):
        record = prepared.records[index]
        return replace(record, append_eos=False, prompt_ids=tuple(protocol_prompt_ids(record, prepared.tokenizer)))
    reference = [record_for(i) for i in parts["reference"]]
    records = [AuditRecord(record_for(i), int(prepared.labels[i])) for i in selected]
    remaining = [name for name in methods if not already_complete(output / name, digest({"task": task, "method": name}), sources)]
    if not remaining:
        return
    reuse_reference = bool(settings.get("reuse_robustness_reference", False))
    reference_cache = {} if reuse_reference else None
    model = load_finetuned_model(Path(task["run_dir"]), cfg.target_model, torch.device(device),
                                 attn_implementation="sdpa")
    model.requires_grad_(False)
    failures = []
    for method in remaining:
        if method not in METHODS:
            raise ValueError(f"unknown baseline {method}")
        # Only methods with actual reference use receive the 400-record pool.
        auxiliary = reference if method in ("recall", "icp_mia", "petal") else []
        scorer = TargetScorer(model, prepared.tokenizer, torch.device(device), args.sead_samples,
                              args.sead_temperature, args.seed)
        handle = None
        try:
            scorer.stats(reference[0])  # untimed warmup, no calibration or test records
            set_seed(args.seed)
            reset_peak(device)
            meter = PhaseMeter(torch.device(device), len(test))
            progress = PhaseProgress(meter, device, len(cal), args.generation_batch_size)
            forward_calls = defaultdict(int)
            def count_forward(_module, _args):
                forward_calls[meter.phase] += 1
            handle = model.register_forward_pre_hook(count_forward)
            scorer.cost_meter = meter
            reused_reference = reference_cache is not None and method in ("ws", "rs", "bt") and "texts" in reference_cache
            with timed(device) as elapsed:
                score_kwargs = {"reference_cache": reference_cache} if reference_cache is not None and method in ("ws", "rs", "bt") else {}
                values = np.asarray(_score_methods(args, progress, scorer, records, auxiliary,
                                                   prepared.tokenizer, (method,), **score_kwargs)[method], dtype=float)
            phases = dict(progress.seconds)
            # Preparation includes tokenization, reference construction and
            # per-method overhead outside record/batch iteration boundaries.
            phases["preparation"] += max(0., elapsed["seconds"] - sum(phases.values()))
            with timed("cpu") as calibration_time:
                ordered = np.sort(values[cal])
            with timed("cpu") as decision_time:
                (1 + len(ordered) - np.searchsorted(ordered, values[test], side="left")) / (len(ordered) + 1)
            phases["calibration"] += calibration_time["seconds"]
            phases["test"] += decision_time["seconds"]
            raw = {name: {key: getattr(m, key) for key in
                         ("forward_sequences", "generated_sequences", "input_tokens", "output_tokens")}
                   for name, m in meter.meters.items()}
            totals = {key: sum(row[key] for row in raw.values()) for key in next(iter(raw.values()))}
            counters = dict(target_sequences=totals["forward_sequences"] + totals["generated_sequences"],
                            target_forward_calls=sum(forward_calls.values()), draft_sequences=0,
                            target_input_tokens=totals["input_tokens"], draft_input_tokens=0,
                            generated_tokens=totals["output_tokens"])
            cost = summarize_cost(phases, len(test), counters, peak_memory(device),
                                  execution_group=task["id"] + "/" + method)
            if reuse_reference:
                cost["reference_reused"] = reused_reference
            cost_conventions = dict(COST_CONVENTIONS)
            if reuse_reference:
                cost_conventions["reuse"] = ("physical incremental cost: the first pending WS/RS/BT method generates "
                                             "the shared greedy reference; later methods reuse it and exclude that work")
            report = {
                "method": method, "request_key": digest({"task": task, "method": method}),
                "sources": sources, "condition": task["condition"], "settings": settings,
                "metrics": metrics(values, labels, cal, test, seed=args.seed),
                "metric_conventions": METRIC_CONVENTIONS, "cost": cost,
                "phase_work": raw, "phase_target_forward_calls": dict(forward_calls),
                "cost_conventions": cost_conventions, "access_channel": access_channel(method),
                "reference_records_available": len(auxiliary),
                "reference_records_used": min(args.recall_shots, len(auxiliary)) if method == "recall" else len(auxiliary),
                "reference_ids": [r.record_id for r in (auxiliary[:args.recall_shots] if method == "recall" else auxiliary)],
                "reference_role": "audit_auxiliary_fit_validation_400",
                "response_contract": "original response tokens; prompt masked; no appended synthetic EOS",
                "baseline_implementation": "repository target-only adaptation; see baseline/README.md",
                "hardware": {"device": str(device), "name": torch.cuda.get_device_name(device) if torch.device(device).type == "cuda" else "cpu",
                             "dtype": str(next(model.parameters()).dtype), "attention": "sdpa", "torch": torch.__version__},
            }
            if reuse_reference:
                report["baseline_execution_mode"] = "shared_robustness_reference"
            save_result(output / method, record_ids=ids, labels=labels, scores=values,
                        calibration=cal, test=test, report=report)
        except Exception as error:
            failures.append((method, str(error)))
            traceback.print_exc()
        finally:
            if handle is not None:
                handle.remove()
            scorer._probe_vector.cache_clear()
            del scorer
    if failures:
        raise RuntimeError(f"failed baseline methods (completed ones preserved): {failures}")
