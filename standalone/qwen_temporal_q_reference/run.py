"""Frozen q-reference transfer to the registered Qwen3 temporal experiment.

Reuse strictly verified full-position B=2 observations. The established
HistGradientBoosting nonmember predictor runs on CPU; assigning a GPU does not
imply new language-model inference. No baseline or membership-label selection.
"""
from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import fcntl
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import random
import shlex
import shutil
import subprocess
import sys
from types import SimpleNamespace

import joblib
import numpy as np
import torch

from experiments.paths import ROOT
from experiments.pretraining.data import load_evaluation
from experiments.shared.audit.artifacts import check_sources_light, digest, checkpoint_inventory
from experiments.shared.audit.metrics import metrics, METRIC_CONVENTIONS
from experiments.shared.core.audit_partitions import deployment_partitions
from experiments.shared.core.audit_runtime import _write_json
from experiments.shared.core.deployment_archive import checkpoint_fingerprint, sha256_file
from experiments.shared.evaluation.provenance import import_closure
from experiments.shared.methods.q_feature_accept_only import (
    position_features, fit_document_reference, q_context, fit_nonmember_counts,
    count_position_features,
)
from experiments.shared.methods.q_reference_scorer import QReferenceScorer
from experiments.shared.protocols.protocol_archive import load_archive, atomic_npz
from standalone.qwen_gemma_temporal.prepare import DATA_ROOT

SEEDS = (1919, 1949, 1978)
METHOD = 'qref_multiscale_negative_q50_c25'
SCORING_HASH = '57b1cef83aa4df7d8bbba34fd533325f29fe946b61e776be29ccff0329c64bb6'
MODELS = {
    'target': {'repo_id': 'Qwen/Qwen3-8B-Base', 'revision': '49e3418fbbbca6ecbdf9608b4d22e5a407081db4'},
    'draft': {'repo_id': 'Qwen/Qwen3-1.7B-Base', 'revision': 'ea980cb0a6c2ae4b936e82123acc929f1cec04c1'},
}
CACHE_ROOT = ROOT / 'artifacts/audits/qwen_gemma_temporal_v1/tasks/qwen3/length_matched'
SELECTION = ROOT / 'artifacts/audits/pythia_q_exploration_v1/FROZEN_SELECTION.json'
ENTRY = ROOT / 'experiments/scripts/effectiveness/effectiveness_main_qwen3_temporal_q_reference_b2.sh'


def read(path):
    return json.loads(Path(path).read_text())


def require(ok, message):
    if not ok:
        raise ValueError(message)


def event(stage, **fields):
    row = dict(time=datetime.now(timezone.utc).isoformat(), stage=stage, **fields)
    print(json.dumps(row, ensure_ascii=False), flush=True)
    return row


def validate_data_sources(folder):
    """Verify frozen data, with one byte-proven harmless path portability edit."""
    request, portability = read(folder / 'REQUEST.json'), []
    for source in request['sources']:
        path = Path(source['path'])
        actual = sha256_file(path)
        if actual == source['sha256']:
            continue
        require(path == ROOT / 'standalone/qwen_temporal_clean/prepare.py' and
                source['sha256'] == 'c36ae572b08732eac2e0fff179d44599cabfecf3b1632e3359c9a35ef159564a' and
                actual == 'ab43cc8d72f16d3eeefcf9f918f0087ff0a433c884e9d66dad8001704cba4ad9',
                'unrecognized temporal source drift: ' + str(path))
        current = b"SOURCE = ROOT.parent / 'SD_MIA-pretraining-data/qwen3_temporal_shared_split_v1'"
        historical = b"SOURCE = Path('/home/mxd/lib/SD_MIA-pretraining-data/qwen3_temporal_shared_split_v1')"
        content = path.read_bytes()
        require(content.count(current) == 1 and
                hashlib.sha256(content.replace(current, historical)).hexdigest() == source['sha256'],
                'historical source differs beyond the exact path portability edit')
        portability.append(dict(path=str(path), old_sha256=source['sha256'], current_sha256=actual,
            reason='d59f969 -> 9ecc44f: only default SOURCE absolute path became ROOT.parent-relative; no cleaning change'))
    for relative, expected in read(folder / 'COMPLETE.json').items():
        require(sha256_file(folder / relative) == expected, 'frozen temporal data changed: ' + relative)
    return request, portability


def validate_alignment(data, contract, rows, partitions, seed):
    """Bind each observation to exact input tokens, order, roles and seed split."""
    by_id = {r['record_id']: r for r in rows}
    ids = data['record_ids'].tolist()
    require(len(by_id) == len(rows) == len(ids) and set(ids) == set(by_id), 'cache sample IDs differ')
    ordered = [by_id[i] for i in ids]
    require(contract['record_ids'] == ids and contract['record_roles'] == data['record_roles'].tolist(),
            'cache contract order/roles differ')
    require(contract['input_hashes'] == [r['token_hash'] for r in ordered], 'cache sample token hashes differ')
    require(contract['resolved_positions'] == [[0]] * len(ids), 'cache is not full-position fixed probing')
    require(np.array_equal(data['document_indices'], np.arange(len(ids))) and
            np.array_equal(data['start_indices'], np.zeros(len(ids))), 'cache trajectory order differs')
    require(np.array_equal(data['lengths'], [len(r['token_ids']) - 1 for r in ordered]) and
            np.array_equal(data['labels'], [r['label'] for r in ordered]) and
            data['record_roles'].tolist() == ['audit_auxiliary' if r['group'] == 'auxiliary' else r['group']
                                            for r in ordered], 'cache lengths/labels/roles differ')
    counts = SimpleNamespace(audit_auxiliary=600, members=2000, nonmembers=2000,
                             detector_train=320, detector_validation=80)
    parts = deployment_partitions(data['labels'], data['record_ids'], data['record_roles'], counts, seed=seed)
    require(partitions['record_ids'] == {k: data['record_ids'][v].tolist() for k, v in parts.items()},
            'cache partitions differ from the registered seed split')
    return parts, ordered


def inputs(seed, checked_models=None):
    folder = DATA_ROOT / f'seed{seed}'
    data_request, portability = validate_data_sources(folder)
    manifest_path = folder / 'qwen3/length_matched/manifest.json'
    manifest = read(manifest_path)
    require(manifest['models'] == MODELS and manifest['selection_seed'] == seed and
            manifest['membership_verified'] is False, 'wrong model/seed/label provenance')
    cache = CACHE_ROOT / f'seed{seed}'
    request = read(cache / 'MAIN_REQUEST.json')
    partitions = read(cache / 'PARTITIONS.json')
    archive = cache / 'observations.npz'
    data, envelope = load_archive(archive, check_sources=False)
    contract = envelope['contract']
    require(request['condition'] == dict(benchmark=manifest['benchmark'], condition_seed=seed, models=MODELS)
            and request['settings']['audit_seed'] == seed and request['protocol'] == 'fixed'
            and request['evaluation_context']['data_manifest'] == str(manifest_path)
            and request['evaluation_context']['token_contract'] == manifest['token_contract']
            and request['evaluation_context']['membership_verified'] is False,
            'cache request does not describe this temporal condition')
    require(contract['matrix_request_key'] == digest(request) and contract['seed'] == seed
            and contract['adapter'] == 'plain' and contract['draft_role'] == 'pretrained'
            and contract['rounds_per_start'] == 0 and contract['execution'] == 'full_context_reconstruction'
            and contract['data_contract'] == 'pretraining_disjoint_nonmember_reference_v1'
            and contract['hardware']['attention'] == 'sdpa'
            and contract['hardware']['dtype'] == 'torch.bfloat16', 'cache collection configuration differs')
    sources = contract['sources']
    require(digest(sources) == request['sources_sha256'] and partitions['seed'] == seed
            and digest(partitions) == request['partitions_sha256']
            and partitions['manifest_sha256'] == sha256_file(manifest_path), 'cache source/split binding differs')
    check_sources_light(sources)
    checked_models = {} if checked_models is None else checked_models
    paths = {}
    for role, spec in MODELS.items():
        found = [s for s in sources['checkpoints'] if Path(s['path']).name == spec['revision']
                 and Path(s['path']).parents[1].name == 'models--' + spec['repo_id'].replace('/', '--')]
        require(len(found) == 1, 'cache model revision/path differs')
        entry = found[0]
        path = Path(entry['path'])
        key = str(path), digest(entry['inventory'])
        if key not in checked_models:
            checked_models[key] = checkpoint_fingerprint(path)
        require(checked_models[key] == entry['sha256'] and checkpoint_inventory(path) == entry['inventory'],
                'cache model weights differ')
        paths[role] = path
    require(len(sources['checkpoints']) == 2, 'unexpected cached models')
    evaluation = load_evaluation(manifest_path, verify_draft=True, model_paths=paths)
    rows_path = manifest_path.parent / manifest['records_file']
    rows = [json.loads(line) for line in rows_path.read_text().splitlines()]
    selection = read(folder / 'SELECTION.json')
    require(selection['seed'] == data_request['seed'] == seed and
            manifest['source_provenance']['request'] == data_request and
            manifest['benchmark'] == 'wiki_temporal/qwen_gemma_temporal_v1/qwen3/length_matched' and
            manifest['counts'] == dict(member=2000, nonmember=2000, auxiliary=600), 'data selection contract differs')
    for group, key in (('member','historical_ids'), ('nonmember','recent_ids'), ('auxiliary','auxiliary_ids')):
        require([r['source_record_id'] for r in rows if r['group'] == group] == selection[key],
                'frozen article selection differs: ' + group)
    require(sorted(len(r['token_ids']) for r in rows if r['group'] == 'member') ==
            sorted(len(r['token_ids']) for r in rows if r['group'] == 'nonmember'), 'length matching changed')
    parts, ordered = validate_alignment(data, contract, rows, partitions, seed)
    pinned = [manifest_path, rows_path, cache / 'MAIN_REQUEST.json', cache / 'PARTITIONS.json',
              archive, archive.with_suffix('.npz.json')]
    descriptor = dict(seed=seed, models=MODELS, benchmark=manifest['benchmark'],
        files={str(p): sha256_file(p) for p in pinned}, partition_id=digest(partitions),
        collection_contract_sha256=digest(contract), vocabulary_size=len(evaluation.tokenizer),
        scored_positions=int(data['lengths'].sum()), counts=manifest['counts'],
        token_contract=manifest['token_contract'], token_lengths=manifest['token_lengths'],
        actual_creation_ranges={g: [min(r['temporal_metadata']['creation_timestamp'] for r in rows if r['group'] == g),
                                    max(r['temporal_metadata']['creation_timestamp'] for r in rows if r['group'] == g)]
                                for g in ('member', 'nonmember', 'auxiliary')},
        seeds=dict(data_selection=seed, auxiliary_partition=seed, calibration_partition=seed,
                   acceptance_collection=seed, nonmember_predictor=seed, bootstrap=seed,
                   language_model_training=None),
        language_models='frozen; no new model inference; cached acceptance seeds use trajectory_seed(seed, record_id, fixed)',
        cache_policy='strict matching full-position B=2 cache; fail closed on missing or changed inputs',
        membership_verified=False, caveats=manifest['caveats'], new_language_model_queries=0)
    descriptor['historical_source_portability'] = portability
    return descriptor, data, parts, ordered, partitions


def implementation():
    frozen = read(SELECTION)
    scoring_path = ROOT / 'experiments/shared/methods/q_feature_accept_only.py'
    require(frozen['selected']['method'] == METHOD and
            frozen['implementation'][str(scoring_path.relative_to(ROOT))] == SCORING_HASH and
            sha256_file(scoring_path) == SCORING_HASH, 'frozen primary implementation changed')
    files = set(import_closure(ROOT, ['experiments.shared.methods.q_reference_scorer',
        'experiments.shared.audit.metrics', 'experiments.shared.protocols.protocol_archive',
        'experiments.pretraining.data', 'experiments.shared.audit.artifacts',
        'experiments.shared.core.audit_partitions']))
    files.update([Path(__file__), ENTRY, ROOT / 'experiments/scripts/common.sh', SELECTION,
                  ROOT / 'standalone/qwen_gemma_temporal/prepare.py'])
    return {str(p.resolve()): sha256_file(p) for p in sorted(files)}


def prepare(output):
    require(not output.exists(), 'output already exists; use a new batch')
    code = implementation()
    checked_models, tasks = {}, []
    output.mkdir(parents=True)
    for seed in SEEDS:
        event('preflight_started', seed=seed)
        spec, _, _, rows, partitions = inputs(seed, checked_models)
        folder = output / f'seed{seed}'
        folder.mkdir()
        _write_json(folder / 'REQUEST.json', spec)
        _write_json(folder / 'PARTITIONS.json', partitions)
        _write_json(folder / 'SAMPLES.json', [dict(record_id=r['record_id'], role=r['group'],
            token_sha256=r['token_hash'], text_sha256=hashlib.sha256(r['text'].encode()).hexdigest(),
            source_record_id=r['source_record_id']) for r in rows])
        tasks.append(dict(seed=seed, request_sha256=sha256_file(folder / 'REQUEST.json'),
                          partitions_sha256=sha256_file(folder / 'PARTITIONS.json'),
                          samples_sha256=sha256_file(folder / 'SAMPLES.json')))
        event('preflight_passed', seed=seed, scored_positions=spec['scored_positions'],
              partition_id=spec['partition_id'], new_language_model_queries=0)
    for path, expected in code.items():
        require(sha256_file(Path(path)) == expected, 'source changed during preflight')
        dest = output / 'source_snapshot' / Path(path).relative_to(ROOT)
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(path, dest)
    plan = dict(schema='qwen_temporal_frozen_q_reference_v1', method=METHOD, tasks=tasks,
        implementation=code, scoring_device='cpu', assigned_gpus='set explicitly at launch',
        models=MODELS, historical_window=['2023-01-01T00:00:00Z', '2023-12-31T23:59:59Z'],
        historical_snapshot='2023-12-31T23:59:59Z',
        recent_window=['2026-04-01T00:00:00Z', '2026-09-17T23:59:59Z'],
        recipe='(z(A)+z(A_N_low40)+z(A_U_low60))/3 + 0.5*min(z(mean_logq),0) + 0.25*z(U_low20)',
        selection='no membership-label tuning; fixed local/context predictor choice by nonmember validation NLL',
        metric_conventions=METRIC_CONVENTIONS, bootstrap=200,
        unavailable_metrics={'verified_membership_metrics': 'No independent membership ground truth; all metrics use temporal proxies.'},
        cross_seed_interpretation='overlapping frozen pages; stability runs, not independent test datasets',
        versions={n: importlib.metadata.version(n) for n in ('numpy','torch','scikit-learn','transformers','joblib')})
    _write_json(output / 'PLAN.json', plan)
    event('prepared', output=str(output), seeds=SEEDS)


def token_subset(x, counts, lengths, indices):
    offsets = np.r_[0, lengths.cumsum()]
    selected = np.concatenate([np.arange(offsets[i], offsets[i+1]) for i in indices])
    return x[selected], counts[selected], lengths[indices]


def worker(output, seed):
    folder = output / f'seed{seed}'
    plan = read(output / 'PLAN.json')
    with (folder / '.worker.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        require(not (folder / '_COMPLETE.json').exists(), 'completed seed will not be overwritten')
        require(plan['method'] == METHOD and plan['implementation'] == implementation(), 'implementation changed')
        task = next(t for t in plan['tasks'] if t['seed'] == seed)
        for name, key in (('REQUEST.json', 'request_sha256'), ('PARTITIONS.json', 'partitions_sha256'),
                          ('SAMPLES.json', 'samples_sha256')):
            require(sha256_file(folder / name) == task[key], 'prepared seed configuration changed')
        visible = os.environ.get('CUDA_VISIBLE_DEVICES', '')
        require(visible.startswith('GPU-') and ',' not in visible and torch.cuda.is_available()
                and torch.cuda.device_count() == 1, 'one explicitly assigned CUDA GPU required')
        running = event('worker_started', seed=seed, pid=os.getpid(), cuda_visible_devices=visible,
                        cuda_logical_device='cuda:0', gpu_name=torch.cuda.get_device_name(0),
                        computation='CPU cached-observation fit; no new GPU inference')
        _write_json(folder / 'RUNNING.json', running)
        spec, data, parts, _, _ = inputs(seed)
        require(spec == read(folder / 'REQUEST.json'), 'worker inputs changed since preflight')
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        event('cache_verified', seed=seed, new_language_model_queries=0)
        event('nonmember_fit_started', seed=seed, train=320, validation=80, calibration=200,
              method=METHOD, scoring_device='cpu')
        x = q_context(data['features'], data['lengths'])
        model, fit = fit_nonmember_counts(
            *token_subset(x, data['counts'], data['lengths'], parts['train']),
            *token_subset(x, data['counts'], data['lengths'], parts['validation']), seed=seed)
        joblib.dump(model, folder / 'nonmember_model.joblib')
        event('nonmember_fit_complete', seed=seed, selected=fit['selected'])
        q = position_features(data['features'], data['counts'], data['lengths'], vocab_size=spec['vocabulary_size'])
        pi = model['model'].predict_proba(x[:, :model['columns']])
        learned = count_position_features(data['features'], data['counts'], data['lengths'], pi)
        qnorm = fit_document_reference({k: v[parts['reference']] for k, v in q.items()})
        cnorm = fit_document_reference({k: v[parts['validation']] for k, v in learned.items()})
        scorer = QReferenceScorer(model['model'], model['columns'], spec['vocabulary_size'], qnorm, cnorm, METHOD)
        scores = scorer.score(data['features'], data['counts'], data['lengths'])
        result = metrics(scores, data['labels'], parts['calibration'], parts['test'], seed=seed, bootstrap=plan['bootstrap'])
        atomic_npz(folder / 'scores.npz', dict(record_ids=data['record_ids'], labels=data['labels'], scores=scores,
            calibration=parts['calibration'], test=parts['test']))
        report = dict(primary=METHOD, seed=seed, metrics=result, metric_conventions=METRIC_CONVENTIONS,
            normalization=dict(q=qnorm, learned=cnorm), fit=fit, evaluation_context=spec,
            model_sha256=sha256_file(folder / 'nonmember_model.joblib'),
            scores_sha256=sha256_file(folder / 'scores.npz'), plan_sha256=sha256_file(output / 'PLAN.json'),
            target_probability_access=False, new_language_model_queries=0,
            unavailable_metrics=plan['unavailable_metrics'], execution=running)
        _write_json(folder / 'REPORT.json', report)
        _write_json(folder / '_COMPLETE.json', {name: sha256_file(folder / name)
            for name in ('REPORT.json', 'nonmember_model.joblib', 'scores.npz')})
        # Confirm the exported inference artifact reproduces its saved scores.
        try:
            replay = QReferenceScorer.from_artifact(folder, vocabulary_size=spec['vocabulary_size'])
            require(np.array_equal(scores, replay.score(data['features'], data['counts'], data['lengths'])),
                    'exported scorer replay differs')
        except Exception:
            (folder / '_COMPLETE.json').unlink()
            raise
        _write_json(folder / 'INFERENCE_VERIFICATION.json', dict(exact_replay=True))
        event('complete', seed=seed, report=str(folder / 'REPORT.json'))


def gpu_snapshot(gpus):
    output = subprocess.check_output(['nvidia-smi', '--query-gpu=index,uuid,name,memory.total,memory.used,utilization.gpu',
                                      '--format=csv,noheader,nounits'], text=True)
    rows = {int(r[0]): dict(index=int(r[0]), uuid=r[1].strip(), name=r[2].strip(), total_mib=int(r[3]),
                           used_mib=int(r[4]), utilization=int(r[5])) for r in csv.reader(output.splitlines())}
    apps = subprocess.check_output(['nvidia-smi', '--query-compute-apps=gpu_uuid,pid,process_name,used_gpu_memory',
                                    '--format=csv,noheader,nounits'], text=True)
    occupied = {r[0].strip() for r in csv.reader(apps.splitlines()) if r}
    require(bool(gpus) and len(set(gpus)) == len(gpus), 'distinct GPUs required')
    chosen = [rows[g] for g in gpus]
    require(all(r['used_mib'] <= 64 and r['utilization'] == 0 and r['total_mib'] >= 20000
                and r['uuid'] not in occupied for r in chosen), 'requested GPUs are not idle/large enough')
    return dict(gpus=chosen, compute_processes=apps.strip())


def launch(output, gpus):
    require(len(gpus) == 3 and len(set(gpus)) == 3, 'exactly three distinct GPUs required')
    require(shutil.which('tmux'), 'persistent tmux launcher is unavailable')
    schedulers = {name: shutil.which(name) for name in ('squeue', 'sinfo', 'qstat', 'bjobs')}
    scheduler_env = {k: v for k, v in os.environ.items() if k.startswith(('SLURM_', 'PBS_', 'LSB_'))}
    require(not any(schedulers.values()) and not scheduler_env,
            'scheduler detected: use its authorized allocation rather than this bare-host launcher')
    with (output / '.launch.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        require(not (output / 'LAUNCH.json').exists(), 'this batch has already been launched')
        plan = read(output / 'PLAN.json')
        require(plan['implementation'] == implementation(), 'prepared implementation changed')
        first = gpu_snapshot(gpus)
        launch_record = dict(time=datetime.now(timezone.utc).isoformat(), precheck=first,
                             schedulers=schedulers, scheduler_env=scheduler_env, jobs=[])
        for seed, gpu in zip(SEEDS, first['gpus']):
            # Earlier seeds belong to this launch and may already initialize CUDA.
            # Recheck only this seed's device immediately before spawning it.
            gpu_snapshot([gpu['index']])
            log = output / f'seed{seed}.log'
            require(not log.exists(), 'existing log will not be overwritten')
            session = 'qref-' + digest(str(output))[:10] + f'-{seed}'
            command = ['env', f'CUDA_VISIBLE_DEVICES={gpu["uuid"]}', f'PYTHONHASHSEED={seed}',
                       'OMP_NUM_THREADS=2', 'MKL_NUM_THREADS=2', 'OPENBLAS_NUM_THREADS=1',
                       'bash', str(ENTRY), 'worker', '--output-root', str(output), '--seed', str(seed)]
            shell = 'exec ' + shlex.join(command) + ' > ' + shlex.quote(str(log)) + ' 2>&1'
            subprocess.run(['tmux', 'new-session', '-d', '-s', session, '-c', str(ROOT), shell], check=True)
            launch_record['jobs'].append(dict(seed=seed, gpu=gpu, session=session, log=str(log),
                                               output=str(output / f'seed{seed}'), command=command))
            _write_json(output / 'LAUNCH.json', launch_record)
        event('launched', jobs=launch_record['jobs'])


def main():
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument('command', choices=('prepare', 'launch', 'worker'))
    parser.add_argument('--output-root', type=Path, required=True)
    parser.add_argument('--seed', type=int, choices=SEEDS)
    parser.add_argument('--gpus', type=int, nargs=3)
    args = parser.parse_args()
    output = args.output_root.resolve()
    require(output.is_relative_to(ROOT / 'artifacts/audits') and len(output.relative_to(ROOT / 'artifacts/audits').parts) == 1,
            'use a new direct artifacts/audits batch directory')
    if args.command == 'prepare':
        prepare(output)
    elif args.command == 'launch':
        require(args.gpus is not None, 'launch requires three explicit --gpus')
        launch(output, args.gpus)
    else:
        require(args.seed is not None, 'worker requires --seed')
        try:
            worker(output, args.seed)
        except Exception as error:
            folder = output / f'seed{args.seed}'
            if folder.is_dir():
                _write_json(folder / 'FAILED.json', event('failed', seed=args.seed, error=str(error)))
            raise


if __name__ == '__main__':
    main()
