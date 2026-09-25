"""Loading remote checkpoint code must not mutate its scientific identity."""
import py_compile
import pytest
from experiments.shared.audit.artifacts import checkpoint_inventory, check_sources_light
from experiments.shared.core.deployment_archive import checkpoint_fingerprint


def test_import_generated_bytecode_does_not_invalidate_checkpoint(tmp_path):
    (tmp_path / 'model.safetensors').write_bytes(b'weights')
    source = tmp_path / 'eagle3.py'; source.write_text('value = 42\n')
    before = checkpoint_inventory(tmp_path)
    checksum = checkpoint_fingerprint(tmp_path)
    py_compile.compile(str(source), doraise=True)
    check_sources_light(dict(files=[], checkpoints=[dict(path=str(tmp_path), inventory=before)]))
    assert checkpoint_inventory(tmp_path) == before
    assert checkpoint_fingerprint(tmp_path) == checksum


@pytest.mark.parametrize('name', ['eagle3.py', 'config.json', 'model.safetensors',
                                 'standalone.pyc', '__pycache__/real_asset.safetensors'])
def test_actual_checkpoint_files_remain_protected(tmp_path, name):
    p = tmp_path / name; p.parent.mkdir(parents=True, exist_ok=True); p.write_bytes(b'before')
    before = checkpoint_inventory(tmp_path); checksum = checkpoint_fingerprint(tmp_path)
    p.write_bytes(b'changed')
    with pytest.raises(ValueError, match='checkpoint inventory changed'):
        check_sources_light(dict(files=[], checkpoints=[dict(path=str(tmp_path), inventory=before)]))
    assert checkpoint_fingerprint(tmp_path) != checksum


def test_incident_repair_proves_original_bytes_before_rebinding(tmp_path):
    import os
    from experiments.maintenance.repair_quality_bytecode import verify_checkpoint
    source = tmp_path / 'eagle3.py'; source.write_text('value = 42\n')
    weights = tmp_path / 'model.safetensors'; weights.write_bytes(b'abc')
    saved = dict(path=str(tmp_path), inventory=checkpoint_inventory(tmp_path),
                 sha256=checkpoint_fingerprint(tmp_path))
    py_compile.compile(str(source), doraise=True)
    assert verify_checkpoint(saved) == saved
    # Same size and mtime evade a light inventory check, but never the repair.
    stat = weights.stat(); weights.write_bytes(b'xyz')
    os.utime(weights, ns=(stat.st_atime_ns, stat.st_mtime_ns))
    with pytest.raises(ValueError, match='original full checkpoint digest'):
        verify_checkpoint(saved)


def test_repair_removes_preexisting_bytecode_only_after_verifying_legacy_digest(tmp_path):
    import hashlib
    from experiments.maintenance.repair_quality_bytecode import verify_checkpoint
    source = tmp_path / 'eagle3.py'; source.write_text('value = 42\n')
    py_compile.compile(str(source), doraise=True)
    files = sorted(p for p in tmp_path.rglob('*') if p.is_file())
    inventory = [[str(p.relative_to(tmp_path)), p.stat().st_size, p.stat().st_mtime_ns] for p in files]
    digest = hashlib.sha256()
    for p in files:
        relative = p.relative_to(tmp_path).as_posix().encode()
        digest.update(len(relative).to_bytes(4, 'big')); digest.update(relative); digest.update(p.read_bytes())
    repaired = verify_checkpoint(dict(path=str(tmp_path), inventory=inventory, sha256=digest.hexdigest()))
    assert repaired['inventory'] == checkpoint_inventory(tmp_path)
    assert repaired['sha256'] == checkpoint_fingerprint(tmp_path)
    assert repaired['sha256'] != digest.hexdigest()
