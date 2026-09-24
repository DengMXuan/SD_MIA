import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from safetensors.numpy import save_file

from experiments.cross_model_audit import engine
from experiments.shared.audit import fixed as main_method
from experiments.shared.models import loading as models
from experiments.shared.audit.provenance import digest, read_result, sources_for, runtime_files
from experiments.cross_model_audit.cli import selected_tasks
from experiments.shared.models.registry import MODEL_PAIRS
from experiments.cross_model_audit.storage import prepare_audit_cache
from experiments.paths import QWEN_AUDIT
from experiments.shared.audit.artifacts import runtime_files as legacy_sources
from experiments.shared.core.deployment_archive import sha256_file
from experiments.shared.protocols.protocol_archive import save_archive
from tests.sft.test_qwen_audit_matrix import settings, prepared_records, write_example_result


def tasks_at(tmp_path, pair='gemma4'):
    return engine.make_tasks(tmp_path / 'models', tmp_path / 'audit', ['newstection'], [1], [1919],
                             settings(), model_pair=pair)


def test_all_pairs_have_distinct_tasks_and_no_natural_sd(tmp_path):
    tasks = selected_tasks(list(MODEL_PAIRS), None, tmp_path / 'out',
                          ['wikitection', 'newstection', 'arxivtection'], [1, 3], [1919, 1949, 1978], settings())
    assert len(tasks) == len({t['id'] for t in tasks}) == len({t['output'] for t in tasks}) == 270
    assert sum(len(t['methods']) for t in tasks) == 1170
    assert all(t.get('protocol', 'fixed') == 'fixed' for t in tasks)
    for name, spec in MODEL_PAIRS.items():
        assert {t['draft_role'] for t in tasks if t['model_pair'] == name and t['kind'] == 'main'} == set(spec.roles)


def test_new_output_cannot_overlap_legacy_audit_even_through_alias(tmp_path):
    alias = tmp_path / 'alias'
    alias.symlink_to(QWEN_AUDIT, target_is_directory=True)
    for folder in (QWEN_AUDIT, QWEN_AUDIT / 'child', QWEN_AUDIT.parent, alias):
        with pytest.raises(ValueError, match='separate'):
            selected_tasks(['gemma4'], None, folder, ['newstection'], [1], [1919], settings())
    with pytest.raises(ValueError, match='exactly one'):
        selected_tasks(['gemma4', 'qwen3'], tmp_path, tmp_path, [], [], [], settings())


def test_new_modules_do_not_enter_legacy_runtime_fingerprint():
    assert not any('cross_model_audit' in p.parts for p in legacy_sources())
    assert any('cross_model_audit' in p.parts for p in runtime_files())
    assert Path(engine.__file__).resolve() not in legacy_sources()


def test_summary_separates_models_and_reuses_baseline_once(tmp_path):
    tasks = tasks_at(tmp_path) + tasks_at(tmp_path, 'qwen3_8b_eagle3')
    for task in tasks:
        for method in task['methods']:
            write_example_result(task, method, .7 if task['model_pair'] == 'gemma4' else .9)
    result = engine.summarize(tasks, tmp_path / 'summary')
    assert result['complete'] and result['completed_rows'] == 48
    assert result['unique_successful_execution_groups'] == 26
    assert result['unique_successful_measured_method_seconds'] == 260
    for row in result['seed_summary']:
        assert row['completed_seeds'] == 1
        assert row['auc_mean'] == (.7 if row['model_pair'] == 'gemma4' else .9)
    assert 'auxiliary_head' in {row['draft_role'] for row in result['rows']}


def dump(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))


@pytest.fixture
def head_run(tmp_path):
    task = tasks_at(tmp_path, 'qwen3_8b_eagle3')[2]
    run = Path(task['run_dir'])
    spec = MODEL_PAIRS[task['model_pair']]
    counts = dict(member=2000, nonmember=2000, auxiliary=2000, audit_auxiliary=600)
    manifest = tmp_path / 'split.json'
    split = dict(benchmark='newstection', seed=1919, counts=counts,
                 splits={k:[dict(record_id=f'{k}{i}') for i in range(n)] for k,n in counts.items()})
    dump(manifest, split)
    checksum = sha256_file(manifest)
    dump(manifest.with_suffix('.audit.json'), dict(tokenizers={f'{spec.target}@{spec.target_revision}':
         dict(shared_split_sha256=checksum, cross_split_ngram_audit=dict(gate='PASS'))}))
    cfg = dict(benchmark='newstection', target_epochs=1, seed=1919, data_seed=1919,
               target_model=spec.target, draft_model=spec.draft,
               target_revision=spec.target_revision, draft_revision=spec.draft_revision,
               n_per_class=2000, n_aux=2000, n_audit_aux=600)
    data = dict(shared_split_sha256=checksum, split_seed=1919, counts=counts)
    stages = {name:dict(data=data) for name in ('target', 'aux_head', 'member_head')}
    artifact = dict(config=cfg, protocol_track=dict(pair=spec.name, target_frozen_before_heads=True,
                    shared_raw_split=str(manifest)), stages=stages)
    dump(run / 'results.json', artifact)
    for role, stage in [('target','target'), ('auxiliary_head','aux_head'), ('member_head','member_head')]:
        folder = run / ('checkpoints' if role == 'target' else 'heads') / role
        dump(folder / 'config.json', {})
        save_file({'weight':np.ones((2,2), dtype=np.float32)}, folder / 'model.safetensors')
        dump(folder / '_COMPLETE.json', dict(status='complete', stage=stage,
             variant='aux' if role=='auxiliary_head' else 'member', target_frozen=True,
             initialized_from_revision=spec.draft_revision, seed=1919,
             target_checkpoint=str(run / 'checkpoints/target')))
    return task, run


def test_head_preflight_validates_selected_branch_and_baseline_needs_no_head(head_run):
    task, run = head_run
    assert engine.ready(task) == (True, 'ready')
    (run / 'heads/auxiliary_head/_COMPLETE.json').unlink()
    assert engine.ready(task) == (True, 'ready')  # member needs its own stage only
    (run / 'heads/member_head/_COMPLETE.json').unlink()
    assert not engine.ready(task)[0]
    baseline = {**task, 'kind':'baseline'}
    assert engine.ready(baseline) == (True, 'ready')


@pytest.mark.parametrize('field,value', [('variant','aux'), ('target_checkpoint','/wrong/target'),
                                         ('initialized_from_revision','wrong'), ('seed',1949)])
def test_head_preflight_rejects_wrong_head_identity(head_run, field, value):
    task, run = head_run
    path=run/'heads/member_head/_COMPLETE.json'; marker=json.loads(path.read_text())
    dump(path,{**marker,field:value})
    assert not engine.ready(task)[0]


def test_head_sources_track_selected_weights_and_reject_tampering(head_run, monkeypatch):
    task, run=head_run
    import experiments.shared.audit.provenance as artifacts
    monkeypatch.setattr(artifacts,'runtime_files',lambda:[])
    result=sources_for(run,['target','member_head'],adapter='eagle3')
    assert Path(result['checkpoints'][1]['path']) == (run/'heads/member_head').resolve()
    from experiments.shared.audit.artifacts import check_sources_light
    check_sources_light(result)
    with (run/'heads/member_head/model.safetensors').open('ab') as f:f.write(b'changed')
    with pytest.raises(ValueError,match='inventory'):check_sources_light(result)


@pytest.mark.parametrize('kind,pair', [('eagle3','qwen3_8b_eagle3'),('mtp','qwen35_9b_mtp')])
def test_model_loader_routes_each_head_to_the_matching_target(head_run, monkeypatch, kind, pair):
    _,run=head_run
    from experiments.shared.drafts import heads
    cfg=SimpleNamespace(target_model='target',draft_model='head')
    monkeypatch.setattr(models,'load_run_config',lambda *_:cfg)
    target=torch.nn.Linear(2,2)
    monkeypatch.setattr(models,'load_finetuned_model',lambda *_args,**_kwargs:target)
    calls=[]
    def load(path,device,**kwargs):
        calls.append((Path(path),kwargs))
        return SimpleNamespace()
    monkeypatch.setattr(heads,'load_eagle3_speculator',load)
    monkeypatch.setattr(heads,'load_mtp_speculator',load)
    from dataclasses import replace
    from experiments.shared.models.adapters import DRAFT_FAMILIES
    monkeypatch.setitem(DRAFT_FAMILIES, kind, replace(DRAFT_FAMILIES[kind],
        protocol_factory=lambda t, d, dev: SimpleNamespace(target=t, kind=kind)))
    for role in ('auxiliary_head','member_head'):
        adapter=models.load_adapter(run,kind,'cpu',role)
        assert calls[-1][0]==run/'heads'/role and adapter.target is target
        if kind=='mtp':assert calls[-1][1]['verifier_checkpoint']==(run/'checkpoints/target').resolve()


@pytest.mark.parametrize('pair', ['gemma4','qwen3_8b_eagle3','llama31_8b_eagle3','qwen35_9b_mtp'])
def test_main_scores_and_resumes_head_or_plain_detector(tmp_path, monkeypatch, pair):
    torch.set_num_threads(1)
    task=tasks_at(tmp_path,pair)[2]
    prepared=prepared_records();output=Path(task['output'])
    prepare_audit_cache(output)
    x=np.zeros((4600,6),dtype=np.float32);x[:,0]=-.8;x[:,1]=.5
    data=dict(features=x,counts=(np.arange(4600)%3).astype(np.uint8),lengths=np.ones(4600,dtype=int),
              document_indices=np.arange(4600),start_indices=np.zeros(4600,dtype=int),
              record_ids=prepared.record_ids,record_roles=prepared.record_roles,labels=prepared.labels)
    sources={'files':[],'checkpoints':[]}
    contract=dict(protocol='fixed',starts=['fixed'],rounds_per_start=0,sources=sources,
                  matrix_request_key=digest(task),hardware={'device':'cpu'},
                  head_validation={'status':'passed'})
    costs=[dict(record_id=str(i),seconds=.01,target_forward_calls=1,draft_forward_calls=1,
                target_input_tokens=3,draft_input_tokens=3,generated_tokens=0,peak_allocated_gpu_bytes=None,
                hidden_state_bytes=24,supported_candidates=1,candidate_positions=1) for i in range(4600)]
    save_archive(output/'observations.npz',data,contract,costs)
    monkeypatch.setattr(main_method,'load_adapter',lambda *a,**k:pytest.fail('must reuse observations'))
    main_method.run_main(task,'cpu',None,prepared,sources)
    report=read_result(output/task['methods'][0])
    assert report['training_member_count']==0 and report['metrics']['n_calibration']==200
    if pair!='gemma4':assert report['cost']['hidden_state_bytes']==4600*24
    before=(output/'detector.pt').read_bytes()
    (output/task['methods'][0]/'REPORT.json').unlink()
    main_method.run_main(task,'cpu',None,prepared,sources)
    assert (output/'detector.pt').is_symlink() and (output/'detector.pt').read_bytes()==before


@pytest.mark.parametrize('pair,kind', [('gemma4','plain'),('qwen3_8b_eagle3','eagle3'),('qwen35_9b_mtp','mtp')])
@pytest.mark.parametrize('baseline', [False, True])
def test_worker_routes_record_reconstruction_and_weight_roles(tmp_path, monkeypatch, pair, kind, baseline):
    from experiments.shared.audit import baselines as matrix_baselines
    task=tasks_at(tmp_path,pair)[0 if baseline else 2]
    monkeypatch.setattr(torch.cuda,'is_available',lambda:True)
    monkeypatch.setattr(engine,'ready',lambda _: (True,'ready'))
    calls=[]
    def prepare(run,adapter,*roles):
        calls.append(('records',adapter,roles))
        return 'cfg','records'
    monkeypatch.setattr(models,'prepare_records',prepare)
    def sources(run,roles,**kwargs):
        calls.append(('sources',roles,kwargs['adapter']))
        return 'sources'
    monkeypatch.setattr(engine,'sources_for',sources)
    def run(*args):calls.append(('run',args))
    monkeypatch.setattr(matrix_baselines,'run_baselines',run)
    monkeypatch.setattr(main_method,'run_main',run)
    engine.execute_worker(task)
    assert calls[0]==('records',kind,() if kind=='plain' else (None if baseline else task['draft_role'],))
    assert calls[1]==('sources',['target'] if baseline else ['target',task['draft_role']],kind)
    assert calls[2][1]==(task,'cuda:0','cfg','records','sources')


def test_shared_reference_summary_counts_incremental_work_once(tmp_path, monkeypatch):
    task = tasks_at(tmp_path)[0]
    task['methods'] = ['ws', 'rs', 'bt']
    monkeypatch.setattr(engine, 'ALL_METHODS', tuple(task['methods']))
    for method, seconds in (('ws', 12.), ('rs', 3.), ('bt', 5.)):
        write_example_result(task, method, .7)
        path = Path(task['output']) / method / 'REPORT.json'
        report = json.loads(path.read_text())
        group = report['cost']['execution_group']
        if method == 'ws':
            report['cost'] = dict(execution_group=group, total_seconds=seconds,
                                  cost_basis='standalone_measured')
        else:
            report['cost'] = dict(execution_group=group, execution_group_seconds=seconds,
                                  cost_basis='physical_incremental',
                                  physical_incremental_total_seconds=seconds,
                                  physical_incremental_amortized_ms_per_record=seconds / 4)
        path.write_text(json.dumps(report))

    result = engine.summarize([task], tmp_path / 'summary')
    assert result['complete'] and result['completed_rows'] == 6  # two draft displays
    assert result['unique_successful_execution_groups'] == 3
    assert result['unique_successful_measured_method_seconds'] == 20.
    incremental = [row for row in result['rows'] if row['method'] in ('rs', 'bt')]
    assert all('total_seconds' not in row and 'amortized_ms_per_record' not in row
               for row in incremental)
    assert all('physical_incremental_amortized_ms_per_record' in row for row in incremental)
    assert 'physical incremental' in (tmp_path / 'summary/RESULTS.md').read_text()
