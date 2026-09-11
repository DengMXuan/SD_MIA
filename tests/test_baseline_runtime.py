import argparse
import json
from types import SimpleNamespace

import numpy as np
import pytest

from experiments.baseline import run
from experiments.baseline.runtime import RunProgress
from experiments.sd_membership_sft.data import SFTRecord


def test_completed_method_survives_later_generation_failure(tmp_path, monkeypatch):
    rows = [run.AuditRecord(SFTRecord(str(i), 'wiki', (1, 2), str(i), prompt_ids=(3,)), i)
            for i in (0, 1)]
    cfg = SimpleNamespace(target_model='fake', benchmark='wikitection')
    monkeypatch.setattr(run, 'load_run_config', lambda _: cfg)
    tokenizer = SimpleNamespace(eos_token_id=0, decode=lambda *a, **k: 'text')
    monkeypatch.setattr(run, '_target_tokenizer', lambda *a: tokenizer)
    monkeypatch.setattr(run, 'load_audit_records', lambda *a: ([rows[1]], [rows[0]], [], {}))
    monkeypatch.setattr(run, 'load_finetuned_model', lambda *a, **k: object())
    monkeypatch.setattr(run.torch.cuda, 'is_available', lambda: False)
    monkeypatch.setattr(run.TargetScorer, 'stats', lambda *a, **k:
                        SimpleNamespace(token_logp=np.array([-.1, -.2])))
    monkeypatch.setattr(run, '_response_prefix_text', lambda *a: (_ for _ in ()).throw(RuntimeError('generation failed')))
    args = argparse.Namespace(methods='loss,samia', prefix_ratio=.5, sead_samples=2,
        samia_samples=2, generation_batch_size=1, run_dir=tmp_path, output_dir=tmp_path,
        pool_path=None, record_start=0, record_end=None, gpu=0, seed=1,
        attn_implementation='sdpa', sead_temperature=1., recall_shots=0,
        k_percent=20, icp_top_k=5)
    with pytest.raises(RuntimeError, match='generation failed'):
        with RunProgress(tmp_path) as progress:
            run._run(args, progress)
    artifact = next(tmp_path.glob('executions/*/loss.npz'))
    with np.load(artifact, allow_pickle=False) as data:
        assert data['loss'].tolist() == pytest.approx([-.15, -.15])
        assert data['labels'].tolist() == [1, 0]
        assert json.loads(str(data['protocol_json']))['methods'] == ['loss']
        saved_cost = json.loads(str(data['cost_json']))
        assert saved_cost['totals']['records'] == 2
        assert saved_cost['amortized_ms_per_record'] >= 0
    assert not list(tmp_path.glob('executions/*/samia.npz'))
    status = json.loads(artifact.with_name('status.json').read_text())
    assert status['stage'] == 'failed'
    assert status['completed_methods'] == ['loss']
    assert 'generation failed' in status['traceback']
    from experiments.baseline.export_completed import export_completed
    recovered = tmp_path / 'recovered'
    export_completed(artifact.parent, recovered)
    report = json.loads((recovered / 'baseline_metrics.json').read_text())
    assert set(report['metrics']) == {'loss'}
    assert report['costs']['loss'] == saved_cost
    assert json.loads((recovered / 'baseline_costs.json').read_text())['methods']['loss'] == saved_cost
    with np.load(recovered / 'baseline_scores.npz') as data:
        assert data['record_ids'].tolist() == ['1', '0']
    with pytest.raises(FileExistsError):
        export_completed(artifact.parent, recovered)


def test_progress_reports_stages_and_uses_isolated_directories(tmp_path, capsys):
    for _ in range(2):
        with RunProgress(tmp_path) as progress:
            assert list(progress.track([1, 2], 'scoring')) == [1, 2]
    assert len(list(tmp_path.glob('executions/*/progress.jsonl'))) == 2
    events = [json.loads(line) for line in capsys.readouterr().out.splitlines()]
    assert any(e.get('completed') == 2 and e.get('total') == 2 for e in events)
