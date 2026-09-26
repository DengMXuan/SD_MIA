"""Seven existing target-only baselines on current pretrained deployment roles.

Only orchestration is new. Scorers, metric definitions and cost accounting are
shared with the existing seven-baseline SFT comparison.
"""
from __future__ import annotations

from collections import defaultdict
import fcntl
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

from experiments.baseline.engine import AuditRecord, TargetScorer, score_methods
from experiments.pretraining.data import load_evaluation, load_model, sha256
from experiments.pretraining.evaluation import _snapshot, _sources, _validate_output
from experiments.shared.audit.artifacts import (
    already_complete, check_sources_light, digest, read_result, save_result,
)
from experiments.shared.audit.baselines import PhaseMeter, PhaseProgress, access_channel
from experiments.shared.audit.config import BASELINE_DEFAULTS
from experiments.shared.audit.costs import COST_CONVENTIONS, timed, reset_peak, peak_memory, summarize_cost
from experiments.shared.audit.metrics import metrics, METRIC_CONVENTIONS
from experiments.shared.core.audit_runtime import _write_json
from experiments.shared.training.training import set_seed

from standalone.pretraining_baselines.contract import (
    METHODS, frozen_contract, assert_main_partitions, assert_score_roles, separate_output,
)


def evaluate(manifest_path, output_dir, *, seed, device='cuda:0', main_dir=None,
             detector_train=320, detector_validation=80, calibration=200):
    manifest_path, output = Path(manifest_path).resolve(), Path(output_dir).resolve()
    manifest, partitions, parts, ids, labels = frozen_contract(
        manifest_path, seed, detector_train=detector_train,
        detector_validation=detector_validation, calibration=calibration)
    separate_output(output, [Path(__file__).parent, *([main_dir] if main_dir else [])])
    if main_dir is not None:
        assert_main_partitions(main_dir, partitions)
    target = _snapshot(manifest['models']['target'], local_files_only=True)
    _validate_output(output, manifest_path, {'target': target})
    sources = _sources(manifest_path, manifest, {'target': target})
    sources['files'] += [dict(path=str(p.resolve()), sha256=sha256(p))
                         for p in sorted(Path(__file__).parent.glob('*.py')) if not p.name.startswith('test_')]
    # Resolve the target locally; load_evaluation does not open the draft unless
    # verify_draft=True. No draft checkpoint or target fine-tuning is required.
    evaluation = load_evaluation(manifest_path, model_paths={
        'target': target, 'draft': manifest['models']['draft']['repo_id']})
    records = evaluation.auxiliary + evaluation.members + evaluation.nonmembers
    if [r.record_id for r in records] != ids.tolist():
        raise ValueError('loader/frozen record order mismatch')
    selected = np.r_[parts['calibration'], parts['test']]
    cal = np.arange(len(parts['calibration']))
    test = np.arange(len(cal), len(selected))
    audit_records = [AuditRecord(records[i], int(labels[i])) for i in selected]
    reference = [records[i] for i in parts['reference']]
    settings = dict(audit_seed=seed, baseline=dict(BASELINE_DEFAULTS),
                    seed_policy='condition_v1', execution='seven_independent_target_only_v1')
    context = dict(training_regime='pretraining', language_models_frozen=True,
        language_model_finetuning=False, draft_model_loaded=False, models=manifest['models'],
        label_provenance=manifest['label_provenance'],
        membership_verified=manifest['kind'] == 'mimir_pretraining_v1',
        token_contract=manifest['token_contract'], data_manifest=str(manifest_path),
        caveats=manifest.get('caveats', []))
    task = dict(schema='pretraining_baselines7_v1', id=str(output), output=str(output),
        methods=list(METHODS), condition=dict(benchmark=manifest['benchmark'], condition_seed=seed,
                                             models=manifest['models']),
        settings=settings, device=str(device), partitions_sha256=digest(partitions),
        sources_sha256=digest(sources), evaluation_context=context)
    output.mkdir(parents=True, exist_ok=True)
    with (output / '.worker.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        request = output / 'BASELINE_REQUEST.json'
        if request.exists():
            if json.loads(request.read_text()) != task:
                raise ValueError('baseline parameters/sources changed; use a new output directory')
        else:
            if any(p.name != '.worker.lock' for p in output.iterdir()):
                raise ValueError('refusing to adopt unrelated results')
            _write_json(request, task)
        partition_file = output / 'PARTITIONS.json'
        if partition_file.exists() and json.loads(partition_file.read_text()) != partitions:
            raise ValueError('saved partitions changed')
        _write_json(partition_file, partitions)
        check_sources_light(sources)
        pending = [m for m in METHODS if not already_complete(
            output / m, digest(dict(task=task, method=m)), sources)]
        if pending:
            set_seed(seed)
            model = load_model(dict(repo_id=str(target), revision=manifest['models']['target']['revision']),
                               torch.device(device), 'sdpa', len(evaluation.tokenizer))
            model.requires_grad_(False)
            for method in pending:
                print(json.dumps(dict(method=method, state='running', seed=seed)), flush=True)
                _score_method(task, sources, method, model, evaluation.tokenizer, reference,
                              audit_records, ids[selected], labels[selected], cal, test)
            del model
        reports = {}
        for method in METHODS:
            reports[method] = read_result(output / method, digest(dict(task=task, method=method)),
                                          digest(sources), check_sources=False)
            assert_score_roles(output / method, partitions)
        check_sources_light(sources)
        return reports


def _score_method(task, sources, method, model, tokenizer, reference, records, ids, labels, cal, test):
    device = torch.device(task['device'])
    args = SimpleNamespace(**task['settings']['baseline'], seed=task['settings']['audit_seed'])
    auxiliary = reference if method in ('petal', 'recall', 'icp_mia') else []
    scorer = TargetScorer(model, tokenizer, device, args.sead_samples, args.sead_temperature, args.seed)
    handle = None
    try:
        scorer.stats(reference[0])
        set_seed(args.seed)
        reset_peak(device)
        meter = PhaseMeter(device, len(test))
        progress = PhaseProgress(meter, device, len(cal), args.generation_batch_size)
        forward_calls = defaultdict(int)

        def count_forward(_module, _args):
            forward_calls[meter.phase] += 1

        handle = model.register_forward_pre_hook(count_forward)
        scorer.cost_meter = meter
        with timed(device) as elapsed:
            values = np.asarray(score_methods(args, progress, scorer, records, auxiliary,
                tokenizer, (method,), reference_cache={})[method], dtype=float)
        phases = dict(progress.seconds)
        phases['preparation'] += max(0., elapsed['seconds'] - sum(phases.values()))
        with timed('cpu') as cal_time:
            ordered = np.sort(values[cal])
        with timed('cpu') as test_time:
            (1 + len(ordered) - np.searchsorted(ordered, values[test], side='left')) / (len(ordered) + 1)
        phases['calibration'] += cal_time['seconds']
        phases['test'] += test_time['seconds']
        raw = {name: {key: getattr(m, key) for key in
                      ('forward_sequences', 'generated_sequences', 'input_tokens', 'output_tokens')}
               for name, m in meter.meters.items()}
        totals = {key: sum(v[key] for v in raw.values()) for key in next(iter(raw.values()))}
        counters = dict(target_sequences=totals['forward_sequences'],
            target_forward_calls=sum(forward_calls.values()), draft_sequences=0,
            target_input_tokens=totals['input_tokens'], draft_input_tokens=0, generated_tokens=0)
        used = auxiliary[:args.recall_shots] if method == 'recall' else auxiliary
        report = dict(method=method, request_key=digest(dict(task=task, method=method)),
            sources=sources, condition=task['condition'], settings=task['settings'],
            evaluation_context=task['evaluation_context'], partitions_sha256=task['partitions_sha256'],
            metrics=metrics(values, labels, cal, test, seed=args.seed), metric_conventions=METRIC_CONVENTIONS,
            cost=summarize_cost(phases, len(test), counters, peak_memory(device),
                                execution_group=task['id'] + '/' + method),
            cost_conventions=COST_CONVENTIONS, phase_work=raw, phase_target_forward_calls=dict(forward_calls),
            access_channel=access_channel(method), reference_records_available=len(auxiliary),
            reference_records_used=len(used), reference_ids=[r.record_id for r in used],
            reference_role='audit_auxiliary_train_plus_validation',
            response_contract=task['evaluation_context']['token_contract'],
            baseline_implementation='existing repository target-only adaptation; experiments/baseline/README.md',
            hardware=dict(device=str(device), name=torch.cuda.get_device_name(device) if device.type == 'cuda' else 'cpu',
                          dtype=str(next(model.parameters()).dtype), attention='sdpa', torch=torch.__version__))
        save_result(Path(task['output']) / method, record_ids=ids, labels=labels, scores=values,
                    calibration=cal, test=test, report=report)
    finally:
        if handle is not None:
            handle.remove()
        scorer._probe_vector.cache_clear()
