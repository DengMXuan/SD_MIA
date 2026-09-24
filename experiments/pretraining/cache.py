"""Validate pretraining cache provenance and freeze size-aware M1 partitions."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from experiments.pretraining.data import TOKEN_CONTRACT, sha256

ROLE = 'draft_pretrained'


def validate_probability_cache(directory, manifest_path, models):
    directory, manifest_path = Path(directory), Path(manifest_path).resolve()
    metadata = json.loads((directory / 'pq_gap_provenance.json').read_text())
    if metadata.get('kind') != 'pretraining_probability_v1' or metadata.get('eos_included') is not False:
        raise RuntimeError('not a no-EOS pretraining probability cache')
    if metadata.get('models') != models or metadata.get('token_contract') != TOKEN_CONTRACT:
        raise RuntimeError('probability model/token contract mismatch')
    if metadata.get('dataset_manifest') != {'path': str(manifest_path), 'sha256': sha256(manifest_path)}:
        raise RuntimeError('probability dataset provenance mismatch')
    for name in ('pq_gap_token_logps.npz', 'pq_gap_scores.npz'):
        if sha256(directory / name) != metadata.get('files', {}).get(name):
            raise RuntimeError('probability cache content hash mismatch')
    return metadata


def validate_cache_provenance(feature_dir, probability_dir, feature, probability, role):
    if role != ROLE or feature.get('role') != role:
        raise RuntimeError('invalid pretrained draft role')
    if not isinstance(probability, dict):
        raise RuntimeError('missing pretraining probability provenance')
    source = feature.get('dataset_manifest', {})
    manifest_path = Path(source.get('path', ''))
    if not manifest_path.is_file() or sha256(manifest_path) != source.get('sha256'):
        raise RuntimeError('feature dataset manifest has changed')
    data_manifest = json.loads(manifest_path.read_text())
    record_path = manifest_path.parent / data_manifest['records_file']
    if sha256(record_path) != data_manifest['records_sha256']:
        raise RuntimeError('frozen records changed since extraction')
    checked = validate_probability_cache(probability_dir, manifest_path, data_manifest['models'])
    if probability != checked or feature.get('models') != checked['models']:
        raise RuntimeError('feature/probability model provenance mismatch')
    if feature.get('token_contract') != TOKEN_CONTRACT or feature.get('eos_included') is not False:
        raise RuntimeError('feature token contract mismatch')
    for key in ('records', 'record_ids_sha256', 'benchmark'):
        if feature.get(key) != probability.get(key):
            raise RuntimeError(f'feature/probability {key} mismatch')
    if feature.get('total_tokens') != probability.get('tokens'):
        raise RuntimeError('feature/probability token counts mismatch')
    if not feature.get('probability_cache_alignment', {}).get('passed'):
        raise RuntimeError('feature q alignment did not pass')
    for name in ('q.npy', 'h.npy', 'labels.npy', 'record_ids.npy', 'lengths.npy', 'offsets.npy',
                 'eos_mask.npy', 'token_ids.npy', 'input_positions.npy', 'prediction_positions.npy'):
        if sha256(feature_dir / name) != feature.get('files', {}).get(name):
            raise RuntimeError(f'feature content hash mismatch: {name}')


def freeze_partitions(labels, record_ids, output, seed=20260824):
    from experiments.sd_membership_sft.archive.m1_fit import Partitions, partition_manifest, make_partitions
    rng = np.random.default_rng(seed)
    indices = {name: [] for name in ('nuisance_location', 'nuisance_scale', 'detector_fit', 'validation', 'calibration', 'test')}
    for label in (1, 0):
        values = rng.permutation(np.flatnonzero(labels == label))
        if len(values) < 40:
            raise ValueError('M1 partitions require at least 40 records per class (small sets are smoke tests only)')
        cut = len(values) // 5
        train = values[:2 * cut]
        for name, group in zip(('validation', 'calibration', 'test'),
                               (values[2*cut:3*cut], values[3*cut:4*cut], values[4*cut:])):
            indices[name].extend(group)
        if label == 1:
            indices['detector_fit'].extend(train)
        else:
            nuisance = train[:cut]
            location = 3 * len(nuisance) // 4
            indices['nuisance_location'].extend(nuisance[:location])
            indices['nuisance_scale'].extend(nuisance[location:])
            indices['detector_fit'].extend(train[cut:])
    indices = {name: np.sort(np.asarray(values, dtype=np.int64)) for name, values in indices.items()}
    owner = np.full(len(labels), '', dtype='U32')
    for name, values in indices.items():
        owner[values] = name
    indices['nuisance_fit'] = np.sort(np.concatenate([indices['nuisance_location'], indices['nuisance_scale']]))
    manifest = partition_manifest(Partitions(indices, owner), labels, record_ids, split_seed=seed, nuisance_seed=seed)
    manifest.update(protocol='pretraining stratified 40/20/20/20; train negatives half nuisance, half detector; nuisance 75/25 location/scale',
                    low_fpr_resolution=1 / (len(indices['calibration']) // 2 + 1))
    make_partitions(labels, record_ids, frozen_manifest=manifest)
    Path(output).write_text(json.dumps(manifest, indent=2), encoding='utf-8')
