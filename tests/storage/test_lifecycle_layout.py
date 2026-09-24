"""Lifecycle routing and lossless migration, including old relative aliases."""
import json
import os
from pathlib import Path

import pytest

from experiments import paths
from experiments.maintenance.migrate_lifecycle_layout import apply, proposed_moves, JOURNAL


def test_audit_batches_and_custom_outputs_do_not_share_intermediates(tmp_path, monkeypatch):
    monkeypatch.setattr(paths, 'ARTIFACTS', tmp_path)
    for name in ('qwen_shared_reference_v1', 'cross_model_fixed_v1', 'dp_defense_v1'):
        root = tmp_path / 'audits' / name
        task = root / 'tasks/wiki/epoch1/seed1919/fixed'
        paths.prepare_audit_cache(task)
        assert (task / 'observations.npz').resolve() == root / 'intermediate/wiki/epoch1/seed1919/fixed/observations.npz'
        assert paths.audit_reports(root / 'tasks') == root / 'reports'
        assert paths.audit_executions(root / 'tasks') == root / 'executions'
    custom = tmp_path / 'custom/fixed'
    assert paths.audit_cache(custom) == custom / 'intermediate'
    assert not custom.exists()  # calculation and dry-run are read-only


def test_training_custom_directory_stays_self_contained(tmp_path):
    run = tmp_path / 'custom'
    paths.prepare_training_storage(run)
    assert not run.exists()


def make_legacy(root):
    run = root / 'runs/training/controlled_sft_v2/model_pairs/toy'
    model = root / 'models/controlled_sft_v2/model_pairs/toy/checkpoints'
    split = root / 'data/splits/controlled_sft_v2'
    audit = root / 'runs/audits/qwen_fixed_v1'
    cache = root / 'cache/audits/qwen_fixed_v1/wiki/fixed'
    for folder in (run, model, split, audit / 'fixed_only_summary', audit / 'executions/attempt', cache):
        folder.mkdir(parents=True)
    (run / 'results.json').write_text('{"historical_path":"unchanged"}')
    (model / 'model.safetensors').write_bytes(b'weights')
    (split / 'split.json').write_text('{}')
    (run / 'checkpoints').symlink_to(os.path.relpath(model, run))
    (run.parent.parent / 'shared_splits').symlink_to(os.path.relpath(split, run.parent.parent))
    (cache / 'observations.npz').write_bytes(b'observations')
    (audit / 'wiki/fixed').mkdir(parents=True)
    (audit / 'wiki/fixed/observations.npz').symlink_to(cache / 'observations.npz')
    (audit / 'fixed_only_summary/SUMMARY.json').write_text('{"complete":true}')
    (audit / 'executions/attempt/STATUS.json').write_text('{}')
    return run, model, audit


def test_migration_preserves_old_paths_bytes_and_inodes_and_is_repeatable(tmp_path):
    run, model, audit = make_legacy(tmp_path)
    inode = (model / 'model.safetensors').stat().st_ino
    before = (run / 'results.json').read_bytes()
    result = apply(tmp_path)
    assert result['verified_files'] == 6
    assert (tmp_path / 'training/controlled_sft_v2/models/model_pairs/toy/checkpoints/model.safetensors').stat().st_ino == inode
    assert (run / 'results.json').read_bytes() == before
    assert (run / 'checkpoints/model.safetensors').read_bytes() == b'weights'
    assert (audit / 'wiki/fixed/observations.npz').read_bytes() == b'observations'
    assert (audit / 'fixed_only_summary/SUMMARY.json').read_text() == '{"complete":true}'
    assert (tmp_path / 'audits/qwen_fixed_v1/reports/SUMMARY.json').exists()
    assert (tmp_path / 'audits/qwen_fixed_v1/executions/attempt/STATUS.json').exists()
    assert apply(tmp_path) == result
    assert proposed_moves(tmp_path) == []


def test_migration_refuses_destination_collision_before_moving(tmp_path):
    run, _, _ = make_legacy(tmp_path)
    (tmp_path / 'training/controlled_sft_v2/runs').mkdir(parents=True)
    with pytest.raises(FileExistsError):
        apply(tmp_path)
    assert not run.is_symlink()
    assert (run / 'results.json').exists()


def test_interrupted_migration_reuses_original_inventory(tmp_path, monkeypatch):
    run, _, _ = make_legacy(tmp_path)
    rename = Path.rename
    calls = 0

    def fail_after_first(self, target):
        nonlocal calls
        calls += 1
        if calls == 2:
            raise OSError('simulated interruption')
        return rename(self, target)

    monkeypatch.setattr(Path, 'rename', fail_after_first)
    with pytest.raises(OSError, match='interruption'):
        apply(tmp_path)
    evidence = json.loads((tmp_path / JOURNAL).read_text())
    monkeypatch.setattr(Path, 'rename', rename)
    apply(tmp_path)
    assert json.loads((tmp_path / JOURNAL).read_text()) == evidence
    assert (run / 'checkpoints/model.safetensors').read_bytes() == b'weights'


@pytest.mark.parametrize('migrated', [False, True])
def test_migration_refuses_active_workers_including_on_retry(tmp_path, migrated):
    import fcntl

    _, _, audit = make_legacy(tmp_path)
    worker_lock = audit / 'wiki/fixed/.worker.lock'
    worker_lock.touch()
    if migrated:
        apply(tmp_path)
    with worker_lock.open('r') as handle:
        fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        with pytest.raises(BlockingIOError):
            apply(tmp_path)
