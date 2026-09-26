"""Shared frozen roles for seven baselines and the current pretrained main."""
from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from experiments.pretraining.data import TOKEN_CONTRACT, sha256
from experiments.shared.core.audit_partitions import deployment_partitions

METHODS = ('loss', 'min_k_prob', 'min_k_pp', 'sead', 'petal', 'recall', 'icp_mia')
MAIN = 'main_fixed_sparse_positive'
SEEDS = (1919, 1949, 1978)
SOURCES = ('arxiv', 'dm_mathematics', 'github', 'hackernews', 'pile_cc',
           'pubmed_central', 'wikipedia_(en)', 'full_pile')


def frozen_contract(manifest_path, seed, *, detector_train=320, detector_validation=80,
                    calibration=200):
    """Derive exactly evaluate_main's IDs/order without loading a model/tokenizer."""
    path = Path(manifest_path).resolve()
    manifest = json.loads(path.read_text())
    if (manifest.get('selection_seed') != seed or type(seed) is not int
            or not 0 <= seed < 2**32
            or manifest.get('kind') not in ('mimir_pretraining_v1', 'temporal_pretraining_v1')
            or manifest.get('token_contract') != TOKEN_CONTRACT):
        raise ValueError('frozen model/data seed or pretraining token contract mismatch')
    counts = SimpleNamespace(members=manifest['counts']['member'],
        nonmembers=manifest['counts']['nonmember'], audit_auxiliary=manifest['counts']['auxiliary'],
        detector_train=detector_train, detector_validation=detector_validation, calibration=calibration)
    if (any(type(v) is not int or v < 1 for v in vars(counts).values())
            or detector_train + detector_validation + calibration != counts.audit_auxiliary):
        raise ValueError('positive, exhaustive auxiliary partition sizes required')
    records_path = path.parent / manifest['records_file']
    if sha256(records_path) != manifest['records_sha256']:
        raise ValueError('frozen records checksum mismatch')
    groups = {name: [] for name in ('auxiliary', 'member', 'nonmember')}
    for line in records_path.read_text().splitlines():
        row = json.loads(line)
        if row['group'] not in groups or row['label'] != int(row['group'] == 'member'):
            raise ValueError('invalid record group/label')
        groups[row['group']].append(row['record_id'])
    ids = np.asarray(sum(groups.values(), []))
    labels = np.asarray([0] * len(groups['auxiliary']) + [1] * len(groups['member'])
                        + [0] * len(groups['nonmember']))
    roles = np.asarray(['audit_auxiliary'] * len(groups['auxiliary'])
                       + ['member'] * len(groups['member']) + ['nonmember'] * len(groups['nonmember']))
    parts = deployment_partitions(labels, ids, roles, counts, seed=seed)
    partitions = dict(schema='pretraining_main_partitions_v1', seed=seed,
        manifest_sha256=sha256(path), record_ids={k: ids[v].tolist() for k, v in parts.items()})
    return manifest, partitions, parts, ids, labels


def assert_main_partitions(main_dir, partitions):
    path = Path(main_dir) / 'PARTITIONS.json'
    if path.exists() and json.loads(path.read_text()) != partitions:
        raise ValueError(f'main/baseline frozen partitions differ: {path}')


def assert_score_roles(folder, partitions):
    """Check actual saved score IDs, roles and order, not just result metadata."""
    expected = partitions['record_ids']
    with np.load(Path(folder) / 'scores.npz', allow_pickle=False) as archive:
        for role in ('calibration', 'test'):
            if archive['record_ids'][archive[role]].tolist() != expected[role]:
                raise ValueError(f'{role} score IDs/order differ from frozen main partition')


def separate_output(output, protected):
    output = Path(output).resolve()
    for item in protected:
        item = Path(item).resolve()
        if output == item or output.is_relative_to(item) or item.is_relative_to(output):
            raise ValueError(f'output overlaps protected data/code/main results: {item}')
