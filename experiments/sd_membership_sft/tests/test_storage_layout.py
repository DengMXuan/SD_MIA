"""Storage aliases must survive atomic writes and JSON resume round trips."""
import json
from pathlib import Path

import numpy as np

from experiments import paths
from experiments.sd_membership_sft.audit.matrix_artifacts import checkpoint_inventory
from experiments.sd_membership_sft.core.audit_runtime import _write_json
from experiments.sd_membership_sft.protocols.protocol_archive import atomic_npz


def test_cache_atomic_writes_preserve_links(tmp_path):
    output = tmp_path / 'run'
    paths.prepare_audit_cache(output)
    for value in (1, 2):
        _write_json(output / 'FIT.json', {'version': value})
        atomic_npz(output / 'observations.npz', {'counts': np.array([value])})
        assert (output / 'FIT.json').is_symlink()
        assert (output / 'observations.npz').is_symlink()
        assert json.loads((output / 'cache/FIT.json').read_text()) == {'version': value}
        with np.load(output / 'cache/observations.npz') as data:
            assert data['counts'].tolist() == [value]
    paths.prepare_audit_cache(output)
    assert (output / 'trajectories').is_symlink()


def test_inventory_contract_survives_json_roundtrip(tmp_path):
    (tmp_path / 'config.json').write_text('{}')
    contract = {'checkpoints': [{'inventory': checkpoint_inventory(tmp_path)}]}
    assert json.loads(json.dumps(contract)) == contract


def test_new_training_checkpoints_live_under_models(tmp_path, monkeypatch):
    monkeypatch.setattr(paths, 'ARTIFACTS', tmp_path)
    monkeypatch.setattr(paths, 'MODELS', tmp_path / 'models')
    run = tmp_path / 'runs/training/controlled_sft_v2/model_pairs/toy'
    paths.prepare_training_storage(run)
    assert (run / 'checkpoints').is_symlink()
    assert (run / 'checkpoints').resolve() == tmp_path / 'models/controlled_sft_v2/model_pairs/toy/checkpoints'
    assert (run / 'heads').is_symlink()
    assert (run / 'adapters').is_symlink()
    (run / 'checkpoints/config.json').write_text('{}')
    paths.prepare_training_storage(run)
    assert (run / 'checkpoints/config.json').read_text() == '{}'
