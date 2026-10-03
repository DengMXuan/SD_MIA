"""Prevent cross-seed/cache mixups and accidental occupied-GPU launches."""
from copy import deepcopy
from types import SimpleNamespace

import numpy as np
import pytest

from standalone.qwen_temporal_q_reference import run


def fixture():
    groups = ['auxiliary'] * 600 + ['member'] * 2000 + ['nonmember'] * 2000
    rows = [dict(record_id=str(i), group=g, label=int(g == 'member'), token_ids=[i, i+1],
                 token_hash=f'hash{i}') for i, g in enumerate(groups)]
    data = dict(record_ids=np.array([r['record_id'] for r in rows]),
                record_roles=np.array(['audit_auxiliary' if g == 'auxiliary' else g for g in groups]),
                labels=np.array([r['label'] for r in rows]), lengths=np.ones(4600, int),
                document_indices=np.arange(4600), start_indices=np.zeros(4600, int))
    contract = dict(record_ids=data['record_ids'].tolist(), record_roles=data['record_roles'].tolist(),
                    input_hashes=[r['token_hash'] for r in rows], resolved_positions=[[0]] * 4600)
    counts = SimpleNamespace(audit_auxiliary=600, members=2000, nonmembers=2000,
                             detector_train=320, detector_validation=80)
    parts = run.deployment_partitions(data['labels'], data['record_ids'], data['record_roles'], counts, seed=1919)
    partitions = dict(record_ids={k: data['record_ids'][v].tolist() for k, v in parts.items()})
    return data, contract, rows, partitions


def test_matching_cache_preserves_disjoint_registered_splits():
    parts, rows = run.validate_alignment(*fixture(), 1919)
    assert {k: len(v) for k, v in parts.items()} == dict(train=320, validation=80, reference=400,
                                                        calibration=200, test=4000)
    assert len(set(np.concatenate([parts[k] for k in ('train','validation','calibration','test')]))) == 4600
    assert len(rows) == 4600


@pytest.mark.parametrize('mismatch', ['tokens', 'order', 'length', 'seed', 'calibration'])
def test_cache_mixups_fail_closed(mismatch):
    data, contract, rows, partitions = deepcopy(fixture())
    seed = 1919
    if mismatch == 'tokens':
        contract['input_hashes'][0] = 'different-document'
    elif mismatch == 'order':
        data['document_indices'][[0,1]] = [1,0]
    elif mismatch == 'length':
        data['lengths'][0] = 2
    elif mismatch == 'seed':
        seed = 1949
    else:
        partitions['record_ids']['calibration'][0] = partitions['record_ids']['train'][0]
    with pytest.raises(ValueError):
        run.validate_alignment(data, contract, rows, partitions, seed)


@pytest.mark.parametrize('used,util,apps', [(512,0,''), (4,5,''), (4,0,'GPU-0,99,python,100')])
def test_busy_gpu_is_rejected(monkeypatch, used, util, apps):
    def query(command, **kwargs):
        if 'index,uuid' in command[1]:
            return '\n'.join(f'{i},GPU-{i},A100,81920,{used if i == 0 else 4},{util if i == 0 else 0}'
                             for i in range(3))
        return apps
    monkeypatch.setattr(run.subprocess, 'check_output', query)
    with pytest.raises(ValueError, match='not idle'):
        run.gpu_snapshot([0,1,2])
