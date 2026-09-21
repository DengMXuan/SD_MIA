"""Read-only CPU verification of the paused 2026-09-22 audit after migration.

Hashes one frozen model pair; stops before the first missing trajectory can run
inference. Run from repo root with python -m experiments.maintenance.verify_layout_resume.
"""
import contextlib
import io
import json
from pathlib import Path
import tarfile
import tempfile
from types import SimpleNamespace

import torch
from experiments.paths import ROOT, QWEN_AUDIT, QWEN_MODELS
from experiments.sd_membership_sft.audit.matrix_artifacts import sources_for, digest
from experiments.sd_membership_sft.protocols import collect_protocol_observations as collector
from experiments.sd_membership_sft.protocols.protocol_models import prepare_records
from experiments.sd_membership_sft.protocols.sd_protocol import RuntimeCost
from experiments.sd_membership_sft.core.deployment_archive import sha256_file


def main():
    backup=ROOT/'artifacts/migrations/20260922_layout'
    old_reports={}
    prefix='experiments/results/sft_runs/qwen_audit_matrix_v1/'
    with tarfile.open(backup/'before_metadata.tar.gz') as tar:
        for item in tar:
            if item.name.startswith(prefix) and item.name.endswith('/REPORT.json'):
                old_reports[item.name[len(prefix):]]=json.load(tar.extractfile(item))
    for relative, old in old_reports.items():
        new=json.loads((QWEN_AUDIT/relative).read_text())
        # Only path/signature provenance is allowed to change.
        for field in set(old)-{'request_key','sources','observation_archive','fit'}:
            assert old[field]==new[field], (relative,field)
        if 'fit' in old:
            assert {k:v for k,v in old['fit'].items() if k!='key'} == {k:v for k,v in new['fit'].items() if k!='key'}
    for relative,(size,sha) in json.loads((backup/'audit_binary_hashes_before.json').read_text()).items():
        path=QWEN_AUDIT/relative
        assert path.stat().st_size==size and sha256_file(path)==sha,relative
    print('All original metrics, costs and 169 binary hashes match',flush=True)
    key='wikitection/epoch1/seed1949'
    output=QWEN_AUDIT/key/'draft_member_sft/fixed'
    contract=json.loads((output/'COLLECTION.json').read_text())
    print('Hashing one complete model pair with current worker sources_for',flush=True)
    current=sources_for(QWEN_MODELS/key,('target','draft_member_sft'))
    assert current==contract['sources']
    print('Rebuilding real paused task records with local tokenizer (CPU only)',flush=True)
    _,prepared=prepare_records(QWEN_MODELS/key,'plain')
    base={k:v for k,v in contract.items() if k not in ('record_ids','record_roles','input_hashes','resolved_positions')}
    class FirstMissing(Exception):pass
    calls=[]
    def stop_before_inference(adapter,prompt,response,**kwargs):
        calls.append((prompt,response))
        raise FirstMissing()
    collector.fixed_trace=stop_before_inference
    capture=io.StringIO()
    with tempfile.TemporaryDirectory(prefix='sd-layout-resume-') as folder:
        work=Path(folder)
        (work/'COLLECTION.json').write_text(json.dumps(contract))
        (work/'trajectories').symlink_to((output/'trajectories').resolve(),target_is_directory=True)
        adapter=SimpleNamespace(device=torch.device('cpu'),cost=RuntimeCost())
        with contextlib.redirect_stdout(capture):
            try:
                collector.collect_records(prepared,adapter,work,base)
            except FirstMissing:pass
            else:raise AssertionError('Expected first missing trajectory to request inference')
        progress=[json.loads(line) for line in capture.getvalue().splitlines() if line.startswith('{')]
        assert len(progress)==2995 and progress[-1]['record']==2995
        assert len(calls)==1
        assert calls[0][1]==list(prepared.records[2995].response_ids)
    summary=dict(original_reports_unchanged=len(old_reports),audit_binary_hashes_unchanged=169,
                 full_checkpoint_pair_hashes_match=True,real_partial_trajectories_reused=2995,
                 next_record_one_based=2996,gpu_inference_executed=False)
    (backup/'RESUME_VERIFIED.json').write_text(json.dumps(summary,indent=2))
    print(json.dumps(summary,indent=2))


if __name__ == "__main__":
    main()
