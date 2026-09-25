"""Reviewed, version-pinned repair of the 2026-09-25 quality batch.

Never authorizes changed numerical code or imports caches from another batch.
Backups and a durable plan make metadata application resumable after interruption.
"""
import argparse
from contextlib import ExitStack
from concurrent.futures import ThreadPoolExecutor, as_completed
from copy import deepcopy
from datetime import datetime, timezone
import fcntl
import hashlib
import json
from pathlib import Path

from experiments.paths import ROOT, EVALUATIONS
from experiments.shared.audit.artifacts import checkpoint_inventory, digest
from experiments.shared.core.audit_runtime import _write_json
from experiments.shared.core.deployment_archive import is_runtime_bytecode, sha256_file
from experiments.shared.evaluation.provenance import runtime_files
from experiments.shared.evaluation.quality import validate_task, read_report

APPROVED = {
    'experiments/shared/audit/artifacts.py': (
        'f4286d7b82ca7ce779e5de0ef0f4530237e5ab5f0af253656ed4a26e77a46db2',
        'd08ffacadb8afb168a45392bbe1fd166e762739985650b549e10dd82454fafd2'),
    'experiments/shared/core/deployment_archive.py': (
        '824336d405dda41eca8ea4806e2fca02a5519c3a693eb2d4b73ee060e96330f4',
        '70d8df6785758c594189ea1cd54e838a4e3a9d85b20905a18c7a16daa71fc10c'),
}


def verify_checkpoint(source):
    """Prove the old full digest, then derive the new digest in the same read."""
    path = Path(source['path'])
    expected = [r for r in source['inventory'] if not is_runtime_bytecode(Path(r[0]))]
    if checkpoint_inventory(path) != expected:
        raise ValueError(f'checkpoint assets changed: {path}')
    legacy, assets = hashlib.sha256(), hashlib.sha256()
    separate = any(is_runtime_bytecode(Path(r[0])) for r in source['inventory'])
    for name, _, _ in source['inventory']:
        relative = Path(name)
        if relative.is_absolute() or '..' in relative.parts:
            raise ValueError('unsafe checkpoint inventory path')
        hashes = [legacy, assets] if separate and not is_runtime_bytecode(relative) else [legacy]
        encoded = relative.as_posix().encode()
        for h in hashes:
            h.update(len(encoded).to_bytes(4, 'big')); h.update(encoded)
        with (path / relative).open('rb') as stream:
            while chunk := stream.read(8 * 1024 * 1024):
                for h in hashes:
                    h.update(chunk)
    if legacy.hexdigest() != source['sha256']:
        raise ValueError(f'original full checkpoint digest differs: {path}')
    if checkpoint_inventory(path) != expected:
        raise ValueError(f'checkpoint changed during verification: {path}')
    return dict(path=str(path), inventory=expected, sha256=(assets if separate else legacy).hexdigest())


def repaired_sources(sources, verified):
    result = deepcopy(sources)
    approved = {str(ROOT / name): hashes for name, hashes in APPROVED.items()}
    for file in result['files']:
        current = sha256_file(Path(file['path']))
        if file['path'] in approved:
            old, new = approved[file['path']]
            if (file['sha256'], current) != (old, new):
                raise ValueError(f'unreviewed provenance helper version: {file["path"]}')
            file['sha256'] = new
        elif file['sha256'] != current:
            raise ValueError(f'unrelated source changed: {file["path"]}')
    if sources['runtime_files'] != [str(p.resolve()) for p in runtime_files()]:
        raise ValueError('runtime dependency closure changed')
    result['checkpoints'] = [verified[digest(c)] for c in sources['checkpoints']]
    return result


def repair(batch):
    batch = Path(batch).resolve()
    if batch != (EVALUATIONS / 'model_quality_v2').resolve() or (batch / 'ARCHIVED.json').exists():
        raise ValueError('this incident repair only applies to model_quality_v2')
    receipts = batch / 'repairs/bytecode_inventory_v1'
    receipts.mkdir(parents=True, exist_ok=True)
    for name, (_, new) in APPROVED.items():
        if sha256_file(ROOT / name) != new:
            raise ValueError(f'reviewed repair code changed: {name}')
    requests = sorted(batch.glob('tasks/**/REQUEST.json'))
    if len(requests) != 90:
        raise ValueError('expected this incident\'s complete 90-task request matrix')
    with ExitStack() as stack:
        for path in [batch / '.matrix.lock', *[p.parent / '.quality.lock' for p in requests]]:
            lock = stack.enter_context(path.open('a'))
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        plan_path = receipts / 'PLAN.json'
        if plan_path.exists():
            plan = json.loads(plan_path.read_text())
        else:
            updates, verified, tasks, originals, caches = {}, {}, [], {}, []
            # Independent checkpoints are read in parallel; shared target/base
            # references are hashed only once. Durable results permit resumption.
            checkpoint_inputs, full_checks = {}, set()
            for request_path in requests:
                for source in json.loads(request_path.read_text())['sources']['checkpoints']:
                    key = digest(source)
                    checkpoint_inputs[key] = source
                    if (not (request_path.parent / 'REPORT.json').exists()
                            or any(is_runtime_bytecode(Path(r[0])) for r in source['inventory'])):
                        full_checks.add(key)
            verification_modes = {}
            verified_path = receipts / 'VERIFIED_CHECKPOINTS.json'
            if verified_path.exists():
                saved = json.loads(verified_path.read_text())
                for key, source in checkpoint_inputs.items():
                    candidate = saved.get(key)
                    if candidate and candidate['inventory'] == checkpoint_inventory(Path(source['path'])):
                        verified[key] = candidate
                        verification_modes[key] = 'original_full_digest_verified'
            for key, source in checkpoint_inputs.items():
                if key not in verified and key not in full_checks:
                    # A successful report already binds this exact full digest.
                    # With no bytecode in its original manifest, the fix leaves
                    # that digest unchanged. Keep the normal status guarantee;
                    # all report outputs and source files are checked below.
                    if checkpoint_inventory(Path(source['path'])) != source['inventory']:
                        raise ValueError(f'completed checkpoint inventory changed: {source["path"]}')
                    verified[key] = deepcopy(source)
                    verification_modes[key] = 'prior_success_and_unchanged_inventory'
            with ThreadPoolExecutor(max_workers=4) as pool:
                jobs = {pool.submit(verify_checkpoint, source): key
                        for key, source in checkpoint_inputs.items() if key not in verified}
                for future in as_completed(jobs):
                    key = jobs[future]
                    verified[key] = future.result()
                    verification_modes[key] = 'original_full_digest_verified'
                    _write_json(verified_path, {k: v for k, v in verified.items()
                                               if verification_modes[k] == 'original_full_digest_verified'})
                    print(json.dumps(dict(event='checkpoint_verified', completed=len(verified),
                                          total=len(checkpoint_inputs), path=verified[key]['path'])), flush=True)
            def original(path):
                relative = str(path.relative_to(batch))
                backup = receipts / 'before' / relative
                if not backup.exists():
                    backup.parent.mkdir(parents=True, exist_ok=True)
                    backup.write_bytes(path.read_bytes())
                if sha256_file(path) != sha256_file(backup):
                    raise ValueError(f'unplanned change before repair: {path}')
                originals[relative] = sha256_file(backup)
                return json.loads(backup.read_text())
            for request_path in requests:
                request = original(request_path); task = request['task']
                validate_task(task)
                if Path(task['output']).resolve() != request_path.parent:
                    raise ValueError('task output identity mismatch')
                cache = batch / 'intermediate' / task['id']
                cache_request = cache / 'REQUEST.json'
                if original(cache_request) != request:
                    raise ValueError('cache request differs from task request')
                report_path = request_path.parent / 'REPORT.json'
                report = original(report_path) if report_path.exists() else None
                if report:
                    if report['task'] != task or report['sources'] != request['sources']:
                        raise ValueError('report identity or sources differ')
                    for name, checksum in report['outputs'].items():
                        if sha256_file(report_path.parent / name) != checksum:
                            raise ValueError('completed report artifact was modified')
                else:
                    if task['evaluation'] != 'acceptance':
                        raise ValueError('unexpected unfinished generalization task')
                    samples = json.loads((request_path.parent / 'SAMPLES.json').read_text())
                    gate = json.loads((request_path.parent / 'VALIDATION.json').read_text())
                    if gate['status'] != 'passed' or gate['schema'] != 'adapter_causality_precision_v2':
                        raise ValueError('missing original validation gate')
                    for role in ('member', 'nonmember', 'auxiliary'):
                        if len(samples[role]) != task['settings']['per_class']:
                            raise ValueError('incomplete sample identities')
                        for index, sample in enumerate(samples[role]):
                            p = cache / f'{role}_{index}.json'; chunk = json.loads(p.read_text())
                            if (digest(chunk['rows']) != chunk['sha256'] or len(chunk['rows']) != 1
                                    or chunk['rows'][0]['record_id'] != sample['record_id']
                                    or chunk['rows'][0]['role'] != role):
                                raise ValueError(f'acceptance cache mismatch: {p}')
                            caches.append(dict(path=str(p.relative_to(batch)), sha256=sha256_file(p)))
                for source in request['sources']['checkpoints']:
                    key = digest(source)
                    if key not in verified:
                        print(json.dumps(dict(event='verify_full_checkpoint', path=source['path'])), flush=True)
                        verified[key] = verify_checkpoint(source)
                sources = repaired_sources(request['sources'], verified)
                updated_request = {**request, 'sources': sources}
                updates[str(request_path.relative_to(batch))] = updated_request
                updates[str(cache_request.relative_to(batch))] = updated_request
                if report:
                    updates[str(report_path.relative_to(batch))] = {
                        **report, 'sources': sources,
                        'provenance_repair': dict(receipt=str(receipts / 'APPLIED.json'),
                                                 original_report_sha256=originals[str(report_path.relative_to(batch))],
                                                 original_request_sha256=originals[str(request_path.relative_to(batch))])}
                tasks.append(task)
            plan = dict(schema='quality_bytecode_repair_v1', tasks=tasks, updates=updates,
                        original_sha256=originals, approved_code_changes=APPROVED,
                        checkpoints=verified, checkpoint_verification_modes=verification_modes, cached_rows=caches,
                        created_at=datetime.now(timezone.utc).isoformat())
            _write_json(plan_path, plan)
        # On restart, the only permissible states are exact before/after bytes.
        for relative, value in plan['updates'].items():
            path = batch / relative
            if sha256_file(path) != plan['original_sha256'][relative] and json.loads(path.read_text()) != value:
                raise ValueError(f'unexpected modification during repair: {path}')
        for row in plan['cached_rows']:
            if sha256_file(batch / row['path']) != row['sha256']:
                raise ValueError('cache changed since repair preflight')
        for task in plan['tasks']:
            # Recheck every current source before the metadata transition.
            original = json.loads((receipts / 'before' / Path(task['output']).relative_to(batch) / 'REQUEST.json').read_text())
            repaired_sources(original['sources'], plan['checkpoints'])
        for source in plan['checkpoints'].values():
            if checkpoint_inventory(Path(source['path'])) != source['inventory']:
                raise ValueError('checkpoint changed since repair preflight')
        for relative, value in plan['updates'].items():
            _write_json(batch / relative, value)
        complete = [task for task in plan['tasks'] if (Path(task['output']) / 'REPORT.json').exists()]
        for task in complete:
            read_report(task)
        receipt = dict(schema=plan['schema'], applied_at=datetime.now(timezone.utc).isoformat(),
                       plan_sha256=sha256_file(plan_path), updated_metadata=len(plan['updates']),
                       verified_unique_checkpoints=len(plan['checkpoints']),
                       full_digest_checks=sum(v == 'original_full_digest_verified'
                                              for v in plan['checkpoint_verification_modes'].values()),
                       already_completed=len(complete), acceptance_rows_verified=len(plan['cached_rows']),
                       policy=('ignore only __pycache__/*.pyc and *.pyo; full original digests checked for '
                               'failed tasks and changed digest scopes; prior successful reports retain '
                               'their digest after unchanged inventory and artifact checks'),
                       checkpoint_bytes_modified=False, cached_metrics_modified=False)
        _write_json(receipts / 'APPLIED.json', receipt)
        print(json.dumps(receipt), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--batch', type=Path, default=EVALUATIONS / 'model_quality_v2')
    repair(parser.parse_args().batch)
