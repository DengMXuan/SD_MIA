"""One-time, pinned 2026-09-22 audit metadata migration (never run by workers).

Validate old provenance and every trajectory before preparing an atomic-write
journal. A persisted journal makes interrupted application repeatable. This does
not permit arbitrary source changes: both old and reviewed new hashes are pinned.
"""
from __future__ import annotations

import argparse
from functools import lru_cache
import json
from pathlib import Path

import numpy as np

from experiments.paths import ROOT, TRAINING, QWEN_AUDIT, audit_cache
from experiments.shared.audit.artifacts import checkpoint_inventory, digest, runtime_files, read_result
from experiments.shared.core.audit_runtime import _write_json
from experiments.shared.core.deployment_archive import sha256_file
from experiments.shared.protocols.protocol_archive import load_archive

BACKUP = ROOT / 'artifacts/migrations/20260922_layout'


def require(ok, message):
    if not ok:
        raise ValueError(message)


@lru_cache(maxsize=None)
def checksum(path):
    return sha256_file(Path(path))


def read(path):
    return json.loads(path.read_text())


def verify_pins():
    expected = read(BACKUP / 'reviewed_new_source_hashes.json')
    actual = {str(p): checksum(str(p)) for p in runtime_files()}
    require(actual == expected, 'Code differs from reviewed migration snapshot')
    for relative, stat in read(BACKUP / 'model_inventory_before.json').items():
        now = (TRAINING / relative).stat()
        require([now.st_ino, now.st_size, now.st_mtime_ns] == stat,
                f'Model file changed: {relative}')
    for name, stat in read(BACKUP / 'head_inventory_unchanged.json').items():
        now = Path(name).stat()
        require([now.st_ino, now.st_size, now.st_mtime_ns] == stat, f'Head changed: {name}')
    for relative, (size, sha) in read(BACKUP / 'audit_binary_hashes_before.json').items():
        path = QWEN_AUDIT / relative
        require(path.stat().st_size == size and checksum(str(path)) == sha,
                f'Audit binary changed: {relative}')
    return actual


def build_plan():
    new_hashes = verify_pins()
    old_hashes = read(BACKUP / 'old_source_hashes.json')
    tasks, operations, cleanup, collections = {}, {}, [], []
    source_cache = {}

    def change(path, value):
        before = checksum(str(path))
        operations[str(path)] = dict(before=before, value=value)

    for path in sorted((QWEN_AUDIT / 'executions').glob('*/TASK.json')):
        old = read(path)
        new = {**old, 'run_dir': str(Path(old['run_dir']).resolve()),
               'output': str(Path(old['output']).resolve())}
        if old['id'] in tasks:
            require(tasks[old['id']] == (old, new), 'Conflicting task attempts')
        tasks[old['id']] = old, new
        change(path, new)

    def new_sources(old):
        key = digest(old)
        if key in source_cache:
            return source_cache[key]
        for source in old['files']:
            path = source['path']
            expected = old_hashes[path] if path in old_hashes else checksum(path)
            require(source['sha256'] == expected, f'Old source mismatch: {path}')
        files = old['files']
        run_file = Path(files[0]['path']).resolve()
        manifest = Path(read(run_file)['data']['shared_split_manifest'])
        manifest = manifest if manifest.is_absolute() else ROOT / manifest
        paths = [run_file, *runtime_files(), manifest, manifest.with_suffix('.audit.json')]
        checkpoints = []
        for checkpoint in old['checkpoints']:
            path = Path(checkpoint['path']).resolve()
            inventory = checkpoint_inventory(path)
            require(inventory == checkpoint['inventory'], f'Weight inventory mismatch: {path}')
            checkpoints.append({**checkpoint, 'path': str(path), 'inventory': inventory})
        result = dict(files=[dict(path=str(p.resolve()), sha256=checksum(str(p))) for p in paths],
                      checkpoints=checkpoints)
        source_cache[key] = result
        return result

    for old, task in tasks.values():
        output = Path(task['output'])
        collection_path = output / 'COLLECTION.json'
        archive = output / 'observations.npz'
        if collection_path.exists():
            contract = read(collection_path)
            require(contract['matrix_request_key'] == digest(old), f'Old task differs: {output}')
            replacement = {**contract, 'sources': new_sources(contract['sources']),
                           'matrix_request_key': digest(task)}
            change(collection_path, replacement)
            envelope_path = output / 'observations.npz.json'
            data, envelope = (None, None)
            if archive.exists() and envelope_path.exists():
                data, envelope = load_archive(archive, check_sources=False)
                require(envelope['contract'] == contract, f'Archive contract differs: {output}')
                change(envelope_path, {**envelope, 'contract': replacement})
                offsets = np.r_[0, data['lengths'].cumsum()]
                indices = {(int(i), int(j)): k for k, (i, j) in enumerate(
                    zip(data['document_indices'], data['start_indices']))}
            old_stamp, new_stamp = digest(contract), digest(replacement)
            count = 0
            for sidecar in sorted((output / 'trajectories').glob('*.json')):
                meta = read(sidecar)
                binary = sidecar.with_suffix('.npz')
                require(meta['contract_sha256'] == old_stamp, f'Trajectory contract: {sidecar}')
                require(meta['sha256'] == checksum(str(binary)), f'Trajectory checksum: {binary}')
                count += 1
                if data is not None:
                    pair = tuple(map(int, sidecar.stem.split('_')))
                    k = indices[pair]
                    with np.load(binary, allow_pickle=False) as trace:
                        for field in ('features', 'counts'):
                            require(np.array_equal(trace[field], data[field][offsets[k]:offsets[k+1]]),
                                    f'Trajectory/archive content differs: {binary}')
                    require(meta['cost'] == envelope['costs'][k], f'Trajectory/archive costs differ: {sidecar}')
                    for path in (sidecar, binary):
                        cleanup.append(dict(path=str(path), sha256=checksum(str(path)), bytes=path.stat().st_size))
                else:
                    change(sidecar, {**meta, 'contract_sha256': new_stamp})
            collections.append(dict(task=task['id'], saved_trajectories=count,
                                    complete_archive=data is not None))
            print(f'validated {task["id"]}: {count} trajectories', flush=True)
        fit_path = output / 'FIT.json'
        fit = None
        if fit_path.exists():
            fit = read(fit_path)
            require(fit['key'] == digest({'task': old, 'archive': checksum(str(archive))}), 'Old FIT key differs')
            require(fit['sha256'] == checksum(str(output / 'detector.pt')), 'Detector checksum differs')
            fit = {**fit, 'key': digest({'task': task, 'archive': checksum(str(archive))})}
            change(fit_path, fit)
        for method in task['methods']:
            path = output / method / 'REPORT.json'
            if not path.exists():
                continue
            report = read_result(path.parent, digest({'task': old, 'method': method}), check_sources=False)
            updated = {**report, 'sources': new_sources(report['sources']),
                       'request_key': digest({'task': task, 'method': method})}
            if fit is not None:
                require(report['fit'] == read(fit_path), 'Report/FIT disagreement')
                updated['fit'] = fit
            if 'observation_archive' in updated:
                updated['observation_archive'] = {**updated['observation_archive'], 'path': str(archive)}
            require(updated['metrics'] == report['metrics'] and updated['cost'] == report['cost'],
                    'Scientific outputs must remain unchanged')
            change(path, updated)
    known_reports = {path for path in operations if path.endswith('/REPORT.json')}
    require(known_reports == {str(p) for p in QWEN_AUDIT.rglob('REPORT.json')}, 'Unmapped reports')
    return dict(schema='layout_migration_20260922_v1', new_hashes=new_hashes,
                operations=operations, cleanup=cleanup, collections=collections,
                tasks=[new for old, new in tasks.values()])


def apply(plan):
    require(verify_pins() == plan['new_hashes'], 'Migration source pin changed')
    for name, operation in plan['operations'].items():
        path = Path(name)
        if read(path) == operation['value']:
            continue
        require(sha256_file(path) == operation['before'], f'Metadata modified since planning: {path}')
        _write_json(path, operation['value'])
    # Only verified exact copies already present in the complete archive are removed.
    for item in plan['cleanup']:
        path = Path(item['path'])
        if path.exists():
            require(sha256_file(path) == item['sha256'], f'Cleanup candidate changed: {path}')
            path.unlink()
    for task in plan['tasks']:
        if task['kind'] != 'main':
            continue
        output = Path(task['output'])
        destination = audit_cache(output)
        destination.mkdir(parents=True, exist_ok=True)
        for name in ('trajectories', 'observations.npz', 'observations.npz.json', 'detector.pt', 'FIT.json'):
            path, target = output / name, destination / name
            if path.is_symlink():
                require(path.resolve() == target, f'Unexpected cache alias: {path}')
                continue
            if path.exists():
                require(not target.exists(), f'Cache destination exists: {target}')
                path.rename(target)
            # Also finish a rename interrupted before symlink creation.
            if target.exists():
                path.symlink_to(target, target_is_directory=target.is_dir())
    for task in plan['tasks']:
        for method in task['methods']:
            folder = Path(task['output']) / method
            if (folder / 'REPORT.json').exists():
                read_result(folder, digest({'task': task, 'method': method}))
    verify_pins()
    summary = dict(status='complete', reports=sum(p.endswith('/REPORT.json') for p in plan['operations']),
                   metadata_updates=len(plan['operations']), verified_model_files=len(read(BACKUP / "model_inventory_before.json")),
                   verified_audit_binaries=len(read(BACKUP / 'audit_binary_hashes_before.json')),
                   removed_duplicate_files=len(plan['cleanup']),
                   removed_duplicate_bytes=sum(item['bytes'] for item in plan['cleanup']),
                   collections=plan['collections'])
    _write_json(BACKUP / 'VERIFIED.json', summary)
    print(json.dumps(summary, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=['plan', 'apply'])
    args = parser.parse_args()
    journal = BACKUP / 'metadata_plan.json'
    if args.command == 'plan':
        require(not journal.exists(), 'A migration plan already exists; preserve it for recovery')
        _write_json(journal, build_plan())
    else:
        apply(read(journal))


if __name__ == '__main__':
    main()
