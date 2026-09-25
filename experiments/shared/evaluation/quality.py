"""Resumable epoch-1 target quality and auxiliary-KD acceptance evaluations."""
import fcntl
import gc
import json
from pathlib import Path

import numpy as np
import torch

from experiments.paths import ROOT, EVALUATIONS, AUDITS
from experiments.shared.audit.artifacts import digest, checkpoint_inventory, check_sources_light
from experiments.shared.audit.evaluation import _validate_output
from experiments.shared.core.audit_runtime import _write_json
from experiments.shared.core.deployment_archive import sha256_file, checkpoint_fingerprint
from experiments.shared.models.registry import MODEL_PAIRS, identify_pair, split_manifest
from experiments.shared.models.readiness import ready
from experiments.shared.models.loading import load_adapter, checkpoint_paths
from experiments.shared.protocols.protocol_archive import atomic_npz
from experiments.shared.training import generalization as gen
from experiments.shared.training.training import load_causal_lm, set_seed
from .data import frozen_split, sample_classes, training_tokenizer_source
from .acceptance import record_acceptance, summarize_acceptance
from .provenance import runtime_files

OUTPUT_ROOT = EVALUATIONS / 'model_quality_v2'
KINDS = ('generalization', 'acceptance')


def condition_task(spec, benchmark, seed, kind, *, output_root=OUTPUT_ROOT, run_dir=None, **settings):
    if kind not in KINDS or seed not in (1919, 1949, 1978):
        raise ValueError('unknown evaluation or condition seed')
    key = f'{spec.name}/{benchmark}/epoch1/seed{seed}/{kind}'
    defaults = dict(bootstrap_repeats=1000)
    if kind == 'generalization':
        defaults.update(samples=500, context_tokens=256, gen_tokens=128, batch_size=4, gap_threshold=.03)
    else:
        defaults.update(per_class=256)
    if set(settings) - set(defaults):
        raise ValueError('unknown evaluation settings')
    defaults.update(settings)
    for name, value in defaults.items():
        if name == 'gap_threshold':
            if not np.isfinite(value) or value <= 0:
                raise ValueError('gap threshold must be positive and finite')
        elif type(value) is not int or value < 1:
            raise ValueError(f'{name} must be a positive integer')
    if defaults.get('samples', defaults.get('per_class')) > 2000:
        raise ValueError('sample exceeds frozen class size')
    return dict(schema='model_quality_task_v1', id=key, evaluation=kind, model_pair=spec.name,
                run_dir=str(Path(run_dir or spec.run_root / benchmark / 'epoch1' / f'seed{seed}').resolve()),
                output=str((Path(output_root) / 'tasks' / key).resolve()),
                draft_role=spec.roles[0] if kind == 'acceptance' else None,
                condition=dict(model_pair=spec.name, benchmark=benchmark, epoch=1, condition_seed=seed),
                settings=defaults, seed_policy='condition_v1')


def make_task(run_dir, kind, *, output=None, **settings):
    run_dir = Path(run_dir).resolve()
    artifact = json.loads((run_dir / 'results.json').read_text())
    cfg, spec = artifact['config'], identify_pair(artifact)
    if cfg['target_epochs'] != 1 or cfg['seed'] != cfg['data_seed']:
        raise ValueError('quality evaluation requires epoch 1 and matching training/data seeds')
    task = condition_task(spec, cfg['benchmark'], cfg['seed'], kind, run_dir=run_dir, **settings)
    if output is not None:
        task['output'] = str(Path(output).resolve())
    return task


def _base_snapshot(spec):
    from huggingface_hub import snapshot_download
    return Path(snapshot_download(spec.target, revision=spec.target_revision, local_files_only=True))


def validate_output(run, output):
    run, output = Path(run).resolve(), Path(output).resolve()
    if any((parent / 'ARCHIVED.json').exists() for parent in (output, *output.parents)):
        raise ValueError('archived evaluation batch is read-only; choose a new output batch')
    _validate_output(run, output)
    if output == AUDITS or output.is_relative_to(AUDITS) or AUDITS.is_relative_to(output):
        raise ValueError('quality output must be separate from membership audits')


def validate_task(task):
    spec = MODEL_PAIRS[task['model_pair']]
    condition = task['condition']
    expected = condition_task(spec, condition['benchmark'], condition['condition_seed'],
                              task['evaluation'], run_dir=task['run_dir'], **task['settings'])
    if any(task.get(key) != expected[key] for key in expected if key != 'output'):
        raise ValueError('invalid quality task identity or settings')
    run, output = Path(task['run_dir']), Path(task['output'])
    validate_output(run, output)
    proxy = {**task, 'kind': 'main' if task['evaluation'] == 'acceptance' else 'baseline'}
    valid, reason = ready(proxy)
    if not valid:
        raise ValueError(reason)
    artifact = json.loads((run / 'results.json').read_text())
    if 'privacy' in artifact or (run / 'DP_REQUEST.json').exists():
        raise ValueError('this asset matrix evaluates the existing non-DP checkpoints only')
    manifest = split_manifest(artifact)
    shared = json.loads(manifest.read_text())
    source = training_tokenizer_source(spec)
    attestation = json.loads(manifest.with_suffix('.audit.json').read_text())['tokenizers'][source]
    if (shared['seed'] != condition['condition_seed'] or shared['benchmark'] != condition['benchmark']
            or attestation['shared_split_sha256'] != sha256_file(manifest)
            or attestation['cross_split_ngram_audit']['gate'] != 'PASS'):
        raise ValueError('target tokenizer/frozen split attestation mismatch')
    if task['evaluation'] == 'generalization':
        from experiments.shared.models.registry import validate_weights
        validate_weights(_base_snapshot(spec))
    return spec, artifact


def sources_for(task):
    """Full weight checksums in workers; status only checks saved inventories."""
    spec = MODEL_PAIRS[task['model_pair']]
    run = Path(task['run_dir'])
    artifact = json.loads((run / 'results.json').read_text())
    manifest = split_manifest(artifact)
    runtime = runtime_files()
    files = [run / 'results.json', manifest, manifest.with_suffix('.audit.json'), *runtime]
    paths = [run / 'checkpoints/target']
    if task['evaluation'] == 'acceptance':
        paths.append(checkpoint_paths(run, spec.adapter, task['draft_role'])[1])
    else:
        paths.append(_base_snapshot(spec))
    checkpoints = []
    for path in paths:
        path = path.resolve()
        inventory = checkpoint_inventory(path)
        checksum = checkpoint_fingerprint(path)
        if inventory != checkpoint_inventory(path):
            raise ValueError('checkpoint changed while fingerprinting')
        checkpoints.append(dict(path=str(path), inventory=inventory, sha256=checksum))
    return dict(files=[dict(path=str(p.resolve()), sha256=sha256_file(p)) for p in files],
                checkpoints=checkpoints, runtime_files=[str(p.resolve()) for p in runtime],
                runtime_source_policy='local_import_closure_v1')


def _check_sources(sources):
    check_sources_light(sources)
    if 'runtime_files' in sources and sources['runtime_files'] != [str(p.resolve()) for p in runtime_files()]:
        raise ValueError('evaluation import dependencies changed; use a new output batch')


def read_report(task, sources=None):
    output = Path(task['output'])
    report = json.loads((output / 'REPORT.json').read_text())
    if report.get('schema') != 'model_quality_report_v1' or report['task'] != task:
        raise ValueError('evaluation parameters changed; use a new output batch')
    if sources is not None and report['sources'] != sources:
        raise ValueError('evaluation sources changed; use a new output batch')
    _check_sources(report['sources'])
    for name, checksum in report['outputs'].items():
        if sha256_file(output / name) != checksum:
            raise ValueError(f'evaluation output checksum mismatch: {name}')
    return report


def inspect_task(task):
    try:
        validate_task(task)
        folder = Path(task['output'])
        if (folder / 'REPORT.json').exists():
            read_report(task)
            return dict(status='complete')
        if (folder / 'REQUEST.json').exists():
            saved = json.loads((folder / 'REQUEST.json').read_text())
            if saved['task'] != task:
                raise ValueError('evaluation parameters changed; use a new output batch')
            _check_sources(saved['sources'])
            return dict(status='resumable')
        return dict(status='ready')
    except (OSError, ValueError, KeyError, RuntimeError) as error:
        return dict(status='blocked', reason=str(error))


def _cached(path, produce):
    if path.exists():
        saved = json.loads(path.read_text())
        if saved['sha256'] != digest(saved['rows']):
            raise ValueError(f'cached evaluation rows changed: {path}')
        return saved['rows']
    rows = produce()
    _write_json(path, dict(rows=rows, sha256=digest(rows)))
    return rows


def _generalization(task, cfg, tokenizer, split, cache, output, device):
    settings, seed = task['settings'], task['condition']['condition_seed']
    samples = {}
    for role, records in (('member', split.members), ('nonmember', split.nonmembers)):
        samples[role] = gen.build_eval_samples(records, tokenizer, settings['samples'],
                                              settings['context_tokens'], settings['gen_tokens'], seed)
    _write_json(output / 'SAMPLES.json', samples)
    arrays, scores = {}, {}
    for variant in ('base', 'tuned'):
        # Load only when an incomplete chunk actually needs this model.
        model = None
        scores[variant] = {}
        try:
            for role, selected in samples.items():
                rows = []
                for offset in range(0, len(selected), settings['batch_size']):
                    batch = selected[offset:offset + settings['batch_size']]
                    def produce():
                        nonlocal model
                        if model is None:
                            model = (load_causal_lm(cfg.target_model, device, revision=cfg.target_revision,
                                                   local_files_only=True, attn_implementation='sdpa')
                                     if variant == 'base' else gen.load_finetuned_model(
                                         Path(task['run_dir']), cfg.target_model, device, attn_implementation='sdpa'))
                            model.eval().requires_grad_(False)
                            model.config.use_cache = True
                        hypotheses = gen.generate_continuations(model, batch, tokenizer, device,
                                                               settings['gen_tokens'], settings['batch_size'])
                        values = gen.score_samples(hypotheses, batch)
                        return [dict(record_id=s['record_id'], hypothesis=hypotheses[i],
                                     **{k: float(v[i]) for k, v in values.items()}) for i, s in enumerate(batch)]
                    chunk = _cached(cache / f'{variant}_{role}_{offset}.json', produce)
                    if [r['record_id'] for r in chunk] != [s['record_id'] for s in batch]:
                        raise ValueError('cached generation record identities differ')
                    rows.extend(chunk)
                    print(json.dumps(dict(stage=f'{variant}/{role}', completed=len(rows), total=len(selected))), flush=True)
                scores[variant][role] = {metric: np.array([r[metric] for r in rows])
                                        for metric in gen.GenerationQualityScorer.METRICS}
                for metric, values in scores[variant][role].items():
                    arrays[f'{variant}__{role}__{metric}'] = values
        finally:
            del model
            gc.collect()
            if device.type == 'cuda':
                torch.cuda.empty_cache()
    for role, selected in samples.items():
        arrays[f'{role}__record_ids'] = np.array([s['record_id'] for s in selected])
        arrays[f'{role}__reference_tokens'] = np.array([s['reference_tokens'] for s in selected])
    summary = gen.summarize_model_scores(scores['tuned'], scores['base'], gen.GenerationQualityScorer.METRICS,
                                        settings['bootstrap_repeats'], seed, settings['gap_threshold'])
    protocol = dict(settings, samples_per_class=settings['samples'], benchmark=cfg.benchmark,
                    decoding='greedy', inference_precision='bfloat16', seed=seed,
                    tokenizer=f'{cfg.target_model}@{cfg.target_revision}',
                    short_records='floor(2L/3) context; remaining reference at default 256/128 budgets',
                    uncertainty='independent document bootstrap for member/nonmember; paired bootstrap for base/tuned',
                    gap_gate='advisory absolute mean gap only; not an equivalence test')
    markdown = gen.render_markdown(Path(task['run_dir']), cfg.target_model, summary, protocol)
    return summary, arrays, protocol, markdown


def _acceptance(task, cfg, tokenizer, split, cache, output, device):
    settings, seed = task['settings'], task['condition']['condition_seed']
    selected = sample_classes(split, settings['per_class'], seed)
    _write_json(output / 'SAMPLES.json', {role: [dict(record_id=r.record_id, response_hash=r.response_hash)
                                               for r in records] for role, records in selected.items()})
    spec, adapter, rows = MODEL_PAIRS[task['model_pair']], None, []
    validation_path = output / 'VALIDATION.json'
    validation = None
    try:
        # Gate every attempt, including cache-only resumption. Use a detector
        # training auxiliary, never a selected member/nonmember evaluation row.
        from experiments.shared.models.validation import validate_adapter
        from experiments.shared.data.data import prompt_prefix_ids
        from experiments.shared.core.audit_partitions import deployment_partitions
        prepared = split.audit_auxiliary + split.members + split.nonmembers
        roles = np.asarray(['audit_auxiliary'] * len(split.audit_auxiliary)
                           + ['member'] * len(split.members) + ['nonmember'] * len(split.nonmembers))
        parts = deployment_partitions((roles == 'member').astype(np.int64),
                                      np.asarray([r.record_id for r in prepared]), roles, seed=seed)
        probe = prepared[int(parts['train'][0])]
        adapter = load_adapter(Path(task['run_dir']), spec.adapter, str(device), task['draft_role'])
        validation = validate_adapter(adapter, prompt_prefix_ids(probe, tokenizer), probe.response_ids, seed=seed)
        validation.update(record_id=probe.record_id, record_role='audit_auxiliary_train', seed=seed)
        _write_json(validation_path, validation)
        for role, records in selected.items():
            for index, record in enumerate(records):
                def produce():
                    return [dict(record_id=record.record_id, role=role,
                                 **record_acceptance(adapter, record, tokenizer))]
                cached = _cached(cache / f'{role}_{index}.json', produce)
                if len(cached) != 1 or cached[0]['record_id'] != record.record_id or cached[0]['role'] != role:
                    raise ValueError('cached acceptance record identities differ')
                rows.extend(cached)
                if (index + 1) % 16 == 0 or index + 1 == len(records):
                    print(json.dumps(dict(stage=role, completed=index + 1, total=len(records))), flush=True)
    finally:
        del adapter
        gc.collect()
        if device.type == 'cuda':
            torch.cuda.empty_cache()
    summary = summarize_acceptance(rows, settings['bootstrap_repeats'], seed)
    arrays = {key: np.asarray([r[key] for r in rows]) for key in rows[0]}
    protocol = dict(settings, seed=seed, adapter=spec.adapter, draft_role=task['draft_role'],
                    inference_precision='bfloat16', adapter_validation=validation,
                    definition='sum_v min(p(v), q(v)) = 1 - TV(p, q)',
                    conditioning='teacher-forced original document prefixes',
                    positions='all response positions including appended EOS; prompt excluded',
                    aggregation='mean of document means; equal weight per document',
                    uncertainty='document bootstrap within each role; pooled document bootstrap for overall',
                    deployment_speedup_claim=False)
    lines = ['# KD draft acceptance', '', 'Teacher-forced one-token expected acceptance; no speedup claim.', '',
             '| Role | Documents | Acceptance (95% CI) | Top-1 agreement | Truth vocabulary coverage |',
             '|---|---:|---|---:|---:|']
    for role, values in summary.items():
        a = values['exact_acceptance']
        lines.append(f"| {role} | {a['count']} | {a['mean']:.4f} [{a['ci95_low']:.4f}, {a['ci95_high']:.4f}] "
                     f"| {values['top1_agreement']['mean']:.4f} | {values['truth_vocab_coverage']['mean']:.4f} |")
    return summary, arrays, protocol, '\n'.join(lines) + '\n'


def evaluate_quality(task, *, device='cuda:0'):
    validate_task(task)
    device = torch.device(device)
    if device.type != 'cuda' or not torch.cuda.is_available():
        raise RuntimeError('real model quality evaluation requires CUDA')
    torch.cuda.set_device(device)
    torch.set_num_threads(2)
    set_seed(task['condition']['condition_seed'])
    output = Path(task['output'])
    output.mkdir(parents=True, exist_ok=True)
    with (output / '.quality.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        sources = sources_for(task)
        if (output / 'REPORT.json').exists():
            return read_report(task, sources)
        import transformers
        request = dict(task=task, sources=sources,
                       runtime=dict(torch=str(torch.__version__), transformers=transformers.__version__,
                                    inference_precision='bfloat16',
                                    gpu=torch.cuda.get_device_name(device)))
        marker = output / 'REQUEST.json'
        if marker.exists():
            if json.loads(marker.read_text()) != request:
                raise ValueError('evaluation request or sources changed; use a new output batch')
        elif any(p.name != '.quality.lock' for p in output.iterdir()):
            raise ValueError('refusing to adopt unrelated evaluation files')
        else:
            _write_json(marker, request)
        # Keep intermediate chunks separate from final reports in the standard layout.
        parent = output
        while parent.name != 'tasks' and parent != parent.parent:
            parent = parent.parent
        cache = (parent.parent / 'intermediate' / output.relative_to(parent)
                 if parent.name == 'tasks' else output / 'intermediate')
        validate_output(task['run_dir'], cache)
        cache.mkdir(parents=True, exist_ok=True)
        cache_marker = cache / 'REQUEST.json'
        if cache_marker.exists():
            if json.loads(cache_marker.read_text()) != request:
                raise ValueError('intermediate cache belongs to a different evaluation request')
        elif any(cache.iterdir()):
            raise ValueError('refusing to adopt unbound intermediate evaluation rows')
        else:
            _write_json(cache_marker, request)
        cfg, tokenizer, split = frozen_split(Path(task['run_dir']))
        function = _generalization if task['evaluation'] == 'generalization' else _acceptance
        summary, arrays, protocol, markdown = function(task, cfg, tokenizer, split, cache, output, device)
        _check_sources(sources)
        atomic_npz(output / 'scores.npz', arrays)
        (output / 'REPORT.md').write_text(markdown)
        outputs = ['scores.npz', 'SAMPLES.json', 'REPORT.md']
        if (output / 'VALIDATION.json').exists():
            outputs.append('VALIDATION.json')
        report = dict(schema='model_quality_report_v1', task=task, sources=sources, runtime=request['runtime'],
                      protocol=protocol, split=split.metadata, summary=summary,
                      outputs={name: sha256_file(output / name) for name in outputs})
        _write_json(output / 'REPORT.json', report)
        return report
