"""The fixed WikiTection epsilon-8 DP target and auxiliary-KD quality study."""
import fcntl
import json
from pathlib import Path

import torch

from experiments.paths import EVALUATIONS, ROOT, TRAINING_ROOT
from experiments.shared.audit.artifacts import digest, check_sources_light
from experiments.shared.core.audit_runtime import _write_json
from experiments.shared.core.deployment_archive import sha256_file
from experiments.shared.evaluation import quality as common
from experiments.shared.evaluation.data import frozen_split, training_tokenizer_source
from experiments.shared.evaluation.provenance import import_closure
from experiments.shared.models.readiness import ready
from experiments.shared.models.registry import MODEL_PAIRS, split_manifest, validate_weights
from experiments.shared.protocols.protocol_archive import atomic_npz
from experiments.shared.training.training import set_seed
from .artifacts import stage_key, verify_run

DP_TRACK = 'dp_qwen3_wikitection_epsilon8_v1'
RUN_ROOT = TRAINING_ROOT / 'dp_defense_v2/runs/qwen3/epsilon8/wikitection/epoch1'
OUTPUT_ROOT = EVALUATIONS / 'dp_quality_v1/qwen3/epsilon8/wikitection/epoch1'
SEEDS = (1919, 1949, 1978)


def make_task(seed, kind, *, output_root=OUTPUT_ROOT, **settings):
    if seed not in SEEDS:
        raise ValueError('unknown DP WikiTection condition seed')
    task = common.condition_task(MODEL_PAIRS['qwen3'], 'wikitection', seed, kind,
                                 run_dir=RUN_ROOT / f'seed{seed}', output_root=output_root, **settings)
    task['evaluation_track'] = DP_TRACK
    return task


def validate_dp_task(task, artifact):
    """Cheap status preflight. The worker additionally hashes every DP stage via verify_run."""
    if task.get('evaluation_track') != DP_TRACK:
        raise ValueError('unknown DP quality track')
    condition = task['condition']
    seed = condition['condition_seed']
    if (seed not in SEEDS or task['model_pair'] != 'qwen3' or condition['benchmark'] != 'wikitection'
            or Path(task['run_dir']).resolve() != (RUN_ROOT / f'seed{seed}').resolve()):
        raise ValueError('DP quality task is outside the fixed WikiTection epsilon-8 matrix')
    run = Path(task['run_dir'])
    request = json.loads((run / 'DP_REQUEST.json').read_text())
    privacy = artifact['privacy']
    if (request['model_pair'] != 'qwen3' or request['draft_variants'] != ['kd']
            or request['config'] != artifact['config']
            or request['plans'].keys() != {'target'}
            or request['plans']['target']['epsilon'] != 8.0
            or privacy['request_key'] != digest(request)
            or privacy.get('draft_variants') != ['kd']
            or privacy.get('draft_roles') != ['draft_auxiliary_distilled']
            or privacy['stages'].keys() != {'target', 'draft_auxiliary_distilled'}
            or privacy['pairs'].keys() != {'draft_auxiliary_distilled'}):
        raise ValueError('DP training passport differs from the epsilon-8 KD condition')
    budget = privacy['pairs']['draft_auxiliary_distilled']
    if (budget['epsilon_cap'] != 8.0 or budget['epsilon'] > 8.0
            or budget['delta'] != request['plans']['target']['delta']
            or budget['composition'] != 'target_postprocessing'):
        raise ValueError('DP pair budget differs from the requested epsilon-8 mechanism')
    for role in ('target', 'draft_auxiliary_distilled'):
        stage = privacy['stages'][role]
        marker = json.loads((run / 'checkpoints' / role / 'DP_STAGE.json').read_text())
        teacher = privacy['stages']['target']['checkpoint_sha256'] if role != 'target' else None
        if (marker != {key: value for key, value in stage.items() if key != 'checkpoint_sha256'}
                or marker['key'] != stage_key(request, role, teacher)):
            raise ValueError(f'DP {role} stage marker differs from the training passport')


def validate_task(task):
    condition = task['condition']
    expected = make_task(condition['condition_seed'], task['evaluation'], **task['settings'])
    if any(task.get(key) != value for key, value in expected.items() if key != 'output'):
        raise ValueError('invalid DP quality task identity or settings')
    run = Path(task['run_dir'])
    common.validate_output(run, task['output'])
    proxy = {**task, 'kind': 'main' if task['evaluation'] == 'acceptance' else 'baseline'}
    valid, reason = ready(proxy)
    if not valid:
        raise ValueError(reason)
    artifact = json.loads((run / 'results.json').read_text())
    validate_dp_task(task, artifact)
    manifest = split_manifest(artifact)
    shared = json.loads(manifest.read_text())
    source = training_tokenizer_source(MODEL_PAIRS['qwen3'])
    attestation = json.loads(manifest.with_suffix('.audit.json').read_text())['tokenizers'][source]
    if (shared['seed'] != condition['condition_seed'] or shared['benchmark'] != 'wikitection'
            or attestation['shared_split_sha256'] != sha256_file(manifest)
            or attestation['cross_split_ngram_audit']['gate'] != 'PASS'):
        raise ValueError('DP target tokenizer/frozen split attestation mismatch')
    if task['evaluation'] == 'generalization':
        validate_weights(common._base_snapshot(MODEL_PAIRS['qwen3']))
    return artifact


def runtime_files():
    return sorted(set(import_closure(ROOT, ['experiments.dp_defense.quality_cli'])))


def sources_for(task):
    """Bind DP proof code and request in addition to the common weight fingerprints."""
    sources = common.sources_for(task)
    files = runtime_files()
    existing = {entry['path'] for entry in sources['files']}
    for path in [*files, Path(task['run_dir']) / 'DP_REQUEST.json']:
        path = path.resolve()
        if str(path) not in existing:
            sources['files'].append(dict(path=str(path), sha256=sha256_file(path)))
            existing.add(str(path))
    sources['runtime_files'] = [str(path.resolve()) for path in files]
    sources['runtime_source_policy'] = 'dp_quality_import_closure_v1'
    sources['evaluation_track'] = DP_TRACK
    return sources


def _check_sources(sources):
    check_sources_light(sources)
    if (sources.get('evaluation_track') != DP_TRACK
            or sources.get('runtime_files') != [str(path.resolve()) for path in runtime_files()]):
        raise ValueError('DP evaluation source code changed; use a new output batch')


def read_report(task, sources=None):
    output = Path(task['output'])
    report = json.loads((output / 'REPORT.json').read_text())
    if report.get('schema') != 'dp_model_quality_report_v1' or report['task'] != task:
        raise ValueError('DP evaluation parameters changed; use a new output batch')
    if sources is not None and report['sources'] != sources:
        raise ValueError('DP evaluation sources changed; use a new output batch')
    _check_sources(report['sources'])
    for name, checksum in report['outputs'].items():
        if sha256_file(output / name) != checksum:
            raise ValueError(f'DP evaluation output checksum mismatch: {name}')
    return report


def inspect_task(task):
    try:
        validate_task(task)
        output = Path(task['output'])
        if (output / 'REPORT.json').exists():
            read_report(task)
            return dict(status='complete')
        if (output / 'REQUEST.json').exists():
            saved = json.loads((output / 'REQUEST.json').read_text())
            if saved['task'] != task:
                raise ValueError('DP evaluation parameters changed; use a new output batch')
            _check_sources(saved['sources'])
            return dict(status='resumable')
        return dict(status='ready')
    except (OSError, ValueError, KeyError, RuntimeError) as error:
        return dict(status='blocked', reason=str(error))


def evaluate_quality(task, *, device='cuda:0'):
    artifact = validate_task(task)
    verify_run(Path(task['run_dir']))
    device = torch.device(device)
    if device.type != 'cuda' or not torch.cuda.is_available():
        raise RuntimeError('real DP model quality evaluation requires CUDA')
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
                                    inference_precision='bfloat16', gpu=torch.cuda.get_device_name(device)))
        marker = output / 'REQUEST.json'
        if marker.exists():
            if json.loads(marker.read_text()) != request:
                raise ValueError('DP evaluation request or sources changed; use a new output batch')
        elif any(path.name != '.quality.lock' for path in output.iterdir()):
            raise ValueError('refusing to adopt unrelated DP evaluation files')
        else:
            _write_json(marker, request)
        parent = output
        while parent.name != 'tasks' and parent != parent.parent:
            parent = parent.parent
        cache = (parent.parent / 'intermediate' / output.relative_to(parent)
                 if parent.name == 'tasks' else output / 'intermediate')
        common.validate_output(task['run_dir'], cache)
        cache.mkdir(parents=True, exist_ok=True)
        cache_marker = cache / 'REQUEST.json'
        if cache_marker.exists():
            if json.loads(cache_marker.read_text()) != request:
                raise ValueError('DP intermediate cache belongs to a different request')
        elif any(cache.iterdir()):
            raise ValueError('refusing to adopt unbound DP evaluation rows')
        else:
            _write_json(cache_marker, request)
        cfg, tokenizer, split = frozen_split(Path(task['run_dir']))
        function = common._generalization if task['evaluation'] == 'generalization' else common._acceptance
        summary, arrays, protocol, markdown = function(task, cfg, tokenizer, split, cache, output, device)
        _check_sources(sources)
        atomic_npz(output / 'scores.npz', arrays)
        (output / 'REPORT.md').write_text(markdown)
        outputs = ['scores.npz', 'SAMPLES.json', 'REPORT.md']
        if (output / 'VALIDATION.json').exists():
            outputs.append('VALIDATION.json')
        report = dict(schema='dp_model_quality_report_v1', task=task, sources=sources,
                      runtime=request['runtime'], protocol=protocol, split=split.metadata,
                      privacy={**artifact['privacy']['pairs']['draft_auxiliary_distilled'],
                               'dp_request_key': artifact['privacy']['request_key']},
                      summary=summary, outputs={name: sha256_file(output / name) for name in outputs})
        _write_json(output / 'REPORT.json', report)
        return report
