"""Verified read-only access to a previously collected Pythia fixed audit.

No current language model is loaded. The stored observation/checkpoint/data
hashes and original score replay anchor comparisons to the historical run;
the new runner fingerprints its own current implementation separately.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch

from experiments.shared.core.deployment_archive import sha256_file
from experiments.shared.methods.conditional_accept_only import ConditionalCountTCN
from experiments.shared.methods.protocol_accept_only import predict
from experiments.shared.methods.preserved_accept_only import OLD
from experiments.shared.protocols.protocol_archive import load_archive

MODELS = {
    'target': {'repo_id': 'EleutherAI/pythia-6.9b', 'revision': 'c0e3eee36dc47af0c49f361c74cfe459c09f7f23'},
    'draft': {'repo_id': 'EleutherAI/pythia-1.4b', 'revision': 'fedc38a16eea3bd36a96b906d78d11d2ce18ed79'},
}


def read_json(path):
    return json.loads(Path(path).read_text())


def require(condition, message):
    if not condition:
        raise ValueError(message)


def identify_inputs(report_path, benchmark, seed):
    report_path = Path(report_path).resolve()
    report = read_json(report_path)
    context = report['evaluation_context']
    require(report['method'] == OLD and report['condition']['benchmark'] == benchmark
            and report['condition']['condition_seed'] == seed and report['settings']['audit_seed'] == seed
            and context['training_regime'] == 'pretraining' and report['training_member_count'] == 0
            and report['condition']['models'] == MODELS, 'not the requested frozen Pythia B2 audit')
    manifest_path = Path(context['data_manifest']).resolve()
    manifest = read_json(manifest_path)
    folder = report_path.parent.parent
    archive = Path(report['observation_archive']['path']).resolve()
    paths = dict(report=report_path, observations=archive, sidecar=Path(str(archive) + '.json'),
                 detector=(report_path.parent / report['detector']['file']).resolve(),
                 scores=report_path.with_name('scores.npz'), partitions=folder / 'PARTITIONS.json',
                 manifest=manifest_path, records=(manifest_path.parent / manifest['records_file']).resolve())
    fingerprints = {k: dict(path=str(p), sha256=sha256_file(p)) for k, p in paths.items()}
    for key, expected in [('observations', report['observation_archive']['sha256']),
                          ('detector', report['detector']['sha256']), ('scores', report['scores_sha256']),
                          ('records', manifest['records_sha256'])]:
        require(fingerprints[key]['sha256'] == expected, f'{key} checksum mismatch')
    partition = read_json(paths['partitions'])
    require(manifest['models'] == MODELS and manifest['selection_seed'] == seed
            and manifest['benchmark'] == benchmark and partition['seed'] == seed
            and partition['manifest_sha256'] == fingerprints['manifest']['sha256'], 'data provenance mismatch')
    require(context['membership_verified'] == (manifest['kind'] == 'mimir_pretraining_v1'),
            'membership label provenance mismatch')
    return paths, fingerprints, report


def load_inputs(paths, report):
    data, envelope = load_archive(paths['observations'], check_sources=False)
    require(envelope['archive_sha256'] == report['observation_archive']['sha256']
            and envelope['contract']['seed'] == report['settings']['audit_seed'], 'observation provenance mismatch')
    ids, lengths = data['record_ids'], data['lengths']
    require(np.array_equal(data['document_indices'], np.arange(len(ids)))
            and np.array_equal(data['start_indices'], np.zeros(len(ids))), 'one ordered fixed trajectory required')
    lookup = {v: i for i, v in enumerate(ids)}
    names = read_json(paths['partitions'])['record_ids']
    parts, used = {}, set()
    for name in ('train', 'validation', 'calibration', 'test'):
        chosen = names[name]
        require(chosen and len(set(chosen)) == len(chosen) and not used.intersection(chosen)
                and set(chosen).issubset(lookup), f'overlapping or invalid {name} split')
        used.update(chosen)
        parts[name] = np.asarray([lookup[v] for v in chosen])
        roles = data['record_roles'][parts[name]]
        require(np.isin(roles, ['member', 'nonmember']).all() if name == 'test'
                else (roles == 'audit_auxiliary').all(), f'invalid {name} roles')
    require(used == set(ids) and set(names['reference']) == set(names['train'] + names['validation'])
            and len(names['reference']) == len(set(names['reference'])), 'reference or data coverage mismatch')
    parts['reference'] = np.asarray([lookup[v] for v in names['reference']])
    with np.load(paths['scores'], allow_pickle=False) as a:
        saved = dict(a)
    require(set(saved['record_ids']).issubset(lookup), 'unknown saved document')
    selected = np.asarray([lookup[v] for v in saved['record_ids']])
    require(len(set(selected)) == len(selected) and np.array_equal(data['labels'][selected], saved['labels'])
            and saved['scores'].shape == selected.shape and np.isfinite(saved['scores']).all(), 'saved score alignment mismatch')
    used = set()
    for name in ('calibration', 'test'):
        ix = saved[name]
        require(ix.ndim == 1 and np.issubdtype(ix.dtype, np.integer) and (ix >= 0).all()
                and (ix < len(selected)).all() and len(set(ix)) == len(ix)
                and not used.intersection(ix.tolist())
                and set(saved['record_ids'][ix]) == set(names[name]), f'saved {name} mismatch')
        used.update(ix.tolist())
    require(used == set(range(len(selected))), 'incomplete saved evaluation split')
    records = [json.loads(line) for line in paths['records'].read_text().splitlines()]
    frozen = {r['record_id']: (r['label'], len(r['token_ids']) - 1) for r in records}
    require(len(records) == len(frozen) and set(frozen) == set(ids)
            and all(frozen[v] == (int(data['labels'][i]), int(lengths[i])) for i, v in enumerate(ids)),
            'frozen record identity, label or scored length mismatch')
    return data, parts, saved, selected


def predict_frozen(path, data):
    checkpoint = torch.load(path, map_location='cpu', weights_only=True)
    model = ConditionalCountTCN(5, 2)
    model.load_state_dict(checkpoint['state_dict'], strict=True)
    model.requires_grad_(False)
    mean, scale = checkpoint['mean'].numpy(), checkpoint['scale'].numpy()
    require(mean.shape == scale.shape == (5,) and np.isfinite(mean).all()
            and np.isfinite(scale).all() and (scale > 0).all(), 'invalid frozen feature normalization')
    x = (data['features'][:, [0, 4, 1, 2, 3]] - mean) / scale
    return predict(model, x, data['counts'], data['lengths'], 'cpu').astype(np.float64)
