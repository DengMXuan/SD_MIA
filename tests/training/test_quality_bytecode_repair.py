"""Exercise the incident repair across successful and unfinished task metadata."""
import hashlib
import json
from pathlib import Path
from experiments.maintenance import repair_quality_bytecode as repair
from experiments.shared.audit.artifacts import checkpoint_inventory, digest, check_sources_light
from experiments.shared.core.deployment_archive import checkpoint_fingerprint, sha256_file


def test_reviewed_batch_repair_keeps_scores_and_is_resumable(tmp_path, monkeypatch):
    def sha(data): return hashlib.sha256(data).hexdigest()
    provenance = tmp_path / 'provenance.py'; provenance.write_bytes(b'new approved helper')
    approved = {'provenance.py': (sha(b'old helper'), sha(provenance.read_bytes()))}
    monkeypatch.setattr(repair, 'ROOT', tmp_path)
    monkeypatch.setattr(repair, 'EVALUATIONS', tmp_path / 'evaluations')
    monkeypatch.setattr(repair, 'APPROVED', approved)
    monkeypatch.setattr(repair, 'runtime_files', lambda: [provenance])
    monkeypatch.setattr(repair, 'validate_task', lambda _: None)
    def read_report(task):
        d = json.loads((Path(task['output']) / 'REPORT.json').read_text())
        check_sources_light(d['sources'])
        for name, checksum in d['outputs'].items():
            assert sha256_file(Path(task['output']) / name) == checksum
        return d
    monkeypatch.setattr(repair, 'read_report', read_report)
    batch = tmp_path / 'evaluations/model_quality_v2'; batch.mkdir(parents=True)
    checkpoint = tmp_path / 'checkpoint'; checkpoint.mkdir()
    (checkpoint / 'weights').write_bytes(b'weights unchanged')
    sources = dict(files=[dict(path=str(provenance), sha256=approved['provenance.py'][0])],
                   runtime_files=[str(provenance)], checkpoints=[dict(path=str(checkpoint),
                       inventory=checkpoint_inventory(checkpoint), sha256=checkpoint_fingerprint(checkpoint))])
    tasks = []
    def write(p, value): p.write_text(json.dumps(value))
    for i in range(90):
        folder = batch / 'tasks' / str(i); folder.mkdir(parents=True)
        cache = batch / 'intermediate' / str(i); cache.mkdir(parents=True)
        task = dict(id=str(i), output=str(folder), evaluation='acceptance', settings=dict(per_class=1))
        request = dict(task=task, sources=sources)
        write(folder/'REQUEST.json', request); write(cache/'REQUEST.json', request)
        if i < 89:
            (folder/'scores.npz').write_bytes(b'previous valid metrics')
            write(folder/'REPORT.json', dict(task=task, sources=sources,
                  outputs={'scores.npz': sha256_file(folder/'scores.npz')}))
        else:
            samples = {role: [dict(record_id=role)] for role in ('member','nonmember','auxiliary')}
            write(folder/'SAMPLES.json', samples)
            write(folder/'VALIDATION.json', dict(status='passed', schema='adapter_causality_precision_v2'))
            for role in samples:
                rows = [dict(record_id=role, role=role, exact_acceptance=.8)]
                write(cache/f'{role}_0.json', dict(rows=rows, sha256=digest(rows)))
        tasks.append(task)
    repair.repair(batch)
    receipts = batch / 'repairs/bytecode_inventory_v1'
    first = json.loads((receipts/'APPLIED.json').read_text())
    assert first['already_completed'] == 89 and first['acceptance_rows_verified'] == 3
    assert first['full_digest_checks'] == 1
    assert (receipts/'before/tasks/0/REPORT.json').exists()
    assert (batch/'tasks/0/scores.npz').read_bytes() == b'previous valid metrics'
    assert json.loads((batch/'tasks/89/REQUEST.json').read_text()) == json.loads(
        (batch/'intermediate/89/REQUEST.json').read_text())
    repair.repair(batch)
    second = json.loads((receipts/'APPLIED.json').read_text())
    assert second['plan_sha256'] == first['plan_sha256']
    # A subsequent attempt cannot silently overwrite an unrelated modification.
    request_path = batch/'tasks/0/REQUEST.json'
    altered = json.loads(request_path.read_text()); altered['task']['settings']['per_class'] = 2
    write(request_path, altered)
    import pytest
    with pytest.raises(ValueError, match='unexpected modification'):
        repair.repair(batch)
