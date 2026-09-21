"""Frozen-detector 2x2x3 factorial: difficulty inputs, sparse evidence, calibration."""
from __future__ import annotations
import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import itertools
import json
import os
from pathlib import Path
import queue
import subprocess
import sys
import numpy as np
import torch
from experiments.sd_membership_sft.methods.conditional_accept_only import (ConditionalCountTCN, observable_inputs)
from experiments.sd_membership_sft.core.audit_partitions import legacy_partitions as record_partitions
from experiments.sd_membership_sft.methods.difficulty_accept_only import (OUTPUT as PRIORITY, RESULTS, load_feature_observations, predict, aggregate, expanded_calibration, group_pvalues)
from experiments.sd_membership_sft.core.audit_metrics import (membership_metrics)
from experiments.sd_membership_sft.core.audit_metrics import (conformal_tail_pvalues)
from experiments.sd_membership_sft.core.audit_runtime import (_write_json)

OUTPUT = RESULTS / 'combination_validation'
CONDITIONS = tuple(itertools.product(('wikitection','newstection','arxivtection'),(1,3),(20260914,20260915,20260916)))
SCORES = ('q_global','q_sparse','difficulty_global','difficulty_sparse')
CALIBRATIONS = ('pooled200','pooled1200','grouped1200')
PRIMARY = 'difficulty_sparse__grouped1200'
BASELINE = 'q_global__pooled200'


def calibrated_variants(scores, calibration, expanded, test, difficulty, reference):
    """No test membership argument: independently calibrate each score."""
    boundary = float(np.median(difficulty[reference]))
    groups = (difficulty >= boundary).astype(np.int64)
    pvalues = {}
    for name, score in scores.items():
        pvalues[name+'__pooled200'] = conformal_tail_pvalues(score[test],score[calibration])
        pvalues[name+'__pooled1200'] = conformal_tail_pvalues(score[test],score[expanded])
        pvalues[name+'__grouped1200'] = group_pvalues(score,expanded,test,groups)
    return pvalues, groups, boundary


def decision_metrics(pvalues, labels, groups):
    result = {}
    for name, value in pvalues.items():
        result[name] = {}
        for level in (.01,.05,.1):
            hits = value <= level
            record = {'tpr':float(hits[labels==1].mean()),'actual_fpr':float(hits[labels==0].mean()),'groups':{}}
            for group in (0,1):
                selected = (labels==0)&(groups==group)
                record['groups'][str(group)] = {'nonmembers':int(selected.sum()),
                    'fpr':float(hits[selected].mean()) if selected.any() else None}
            result[name][str(level)] = record
    return result


def restore(path, input_dim, device):
    saved = torch.load(path,weights_only=True,map_location='cpu')
    if saved['causal'] or saved['mean'].shape != (input_dim,):
        raise ValueError('wrong saved detector/input contract')
    state = saved['state_dict']
    model = ConditionalCountTCN(input_dim,2,state['projection.weight'].shape[0]).to(device)
    model.load_state_dict(state)
    model.requires_grad_(False)
    model.eval()
    return model,saved['mean'].numpy(),saved['scale'].numpy()


def evaluate(benchmark, epoch, seed, device='cpu'):
    output = OUTPUT/f'{benchmark}_epoch{epoch}'/f'seed{seed}'
    if (output/'REPORT.json').exists():
        return
    obs,labels,ids,extra,source = load_feature_observations(benchmark,epoch,seed)
    parts = record_partitions(labels,ids)
    full = expanded_calibration(labels,parts)
    x,counts,_ = observable_inputs(obs,2,False)
    fit_dir = PRIORITY/f'features/{benchmark}_epoch{epoch}/seed{seed}'
    previous = json.loads((fit_dir/'REPORT.json').read_text())
    if previous['training_member_count'] or previous['synthetic_member_count']:
        raise ValueError('saved detector was not nonmember-only')
    with np.load(fit_dir/'scores.npz',allow_pickle=False) as f:old=dict(f)
    for key,expected in {'labels':labels,'record_ids':ids,**parts}.items():
        if not np.array_equal(old[key],expected):raise ValueError('saved split/identity differs')
    scores={};checkpoints={};reproduction={}
    for name,stored,features in (('q','baseline',x),('difficulty','difficulty',np.column_stack((x,extra)))):
        checkpoint=fit_dir/f'{stored}.pt'
        model,mean,scale=restore(checkpoint,features.shape[1],device)
        pmf=predict(model,(features-mean)/scale,counts,obs.lengths,device)
        for sparse in (False,True):
            scores[name+('_sparse' if sparse else '_global')]=aggregate(pmf,counts,obs.lengths,sparse=sparse)
        reproduction[name]=float(np.max(np.abs(scores[name+'_global']-old[stored])))
        if not np.allclose(scores[name+'_global'],old[stored],atol=.03,rtol=2e-5):
            raise ValueError('global score did not reproduce saved detector')
        checkpoints[name]={'path':str(checkpoint),'sha256':hashlib.sha256(checkpoint.read_bytes()).hexdigest()}
        del pmf,model
    offsets=obs.offsets
    difficulty=np.array([obs.logq[s:e,0].mean() for s,e in zip(offsets[:-1],offsets[1:])])
    pvalues,groups,boundary=calibrated_variants(scores,parts['calibration'],full,parts['test'],difficulty,parts['reference'])
    metrics={k:membership_metrics(v,labels,parts['calibration'],parts['test']) for k,v in scores.items()}
    decisions=decision_metrics(pvalues,labels[parts['test']],groups[parts['test']])
    output.mkdir(parents=True,exist_ok=True)
    np.savez_compressed(output/'scores.npz',labels=labels,record_ids=ids,**parts,**scores,
                        expanded_calibration=full,groups=groups,lengths=obs.lengths,difficulty=difficulty)
    np.savez_compressed(output/'pvalues.npz',labels=labels[parts['test']],record_ids=ids[parts['test']],
                        groups=groups[parts['test']],**pvalues)
    report={'benchmark':benchmark,'epoch':epoch,'seed':seed,'configuration_count':12,
            'language_models_frozen':True,'detectors_frozen':True,'new_detector_fits':0,
            'training_member_count':0,'synthetic_member_count':0,'metrics':metrics,'decisions':decisions,
            'group_boundary_mean_logq':boundary,'group_calibration_counts':[int((groups[full]==g).sum()) for g in (0,1)],
            'global_reproduction_max_abs_error':reproduction,'checkpoints':checkpoints,'source':source,
            'feature_q_sha256':hashlib.sha256(obs.logq.tobytes()).hexdigest(),
            'accept_bits_sha256':hashlib.sha256(obs.bits.tobytes()).hexdigest(),
            'query_costs':{'per_test_record':(2*obs.lengths[parts['test']]).tolist(),
                           'reference_train_validation':int(2*obs.lengths[parts['reference']].sum()),
                           'calibration200':int(2*obs.lengths[parts['calibration']].sum()),
                           'calibration1200':int(2*obs.lengths[full].sum())},
            'protocol':'shared feature-consistent q, B=2 fixed-candidate replay; each score independently calibrated',
            'primary_candidate':PRIMARY}
    _write_json(output/'REPORT.json',report)
    print(json.dumps({'completed':str(output)}),flush=True)


def matrix(args):
    OUTPUT.mkdir(parents=True,exist_ok=True)
    (OUTPUT/'logs').mkdir(exist_ok=True)
    gpu_ids=args.gpus.split(',') if args.gpus else []
    devices=queue.Queue()
    for gpu in gpu_ids:devices.put(gpu)
    jobs=[case for case in CONDITIONS if not (OUTPUT/f'{case[0]}_epoch{case[1]}'/f'seed{case[2]}'/'REPORT.json').exists()]
    print(json.dumps({'pending':len(jobs),'configurations_per_run':12}),flush=True)
    def run(case):
        benchmark,epoch,seed=case
        gpu=devices.get() if gpu_ids else None
        env=os.environ.copy()
        if gpu is not None:env['CUDA_VISIBLE_DEVICES']=gpu
        command=[sys.executable,'-m','experiments.sd_membership_sft.combined_accept_only','evaluate',
                 '--benchmark',benchmark,'--epoch',str(epoch),'--seed',str(seed),'--device','cuda:0' if gpu is not None else 'cpu']
        try:
            with (OUTPUT/'logs'/f'{benchmark}_epoch{epoch}_seed{seed}.log').open('w') as log:
                subprocess.run(command,env=env,stdout=log,stderr=subprocess.STDOUT,check=True)
            print(json.dumps({'completed':case}),flush=True)
        finally:
            if gpu is not None:devices.put(gpu)
    with ThreadPoolExecutor(max_workers=min(args.jobs,len(gpu_ids)) if gpu_ids else args.jobs) as pool:
        list(pool.map(run,jobs))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('command',choices=('evaluate','matrix'))
    p.add_argument('--benchmark',choices=('wikitection','newstection','arxivtection'))
    p.add_argument('--epoch',type=int,choices=(1,3))
    p.add_argument('--seed',type=int,default=20260914)
    p.add_argument('--device',default='cpu')
    p.add_argument('--gpus',default='')
    p.add_argument('--jobs',type=int,default=3)
    args=p.parse_args();torch.set_num_threads(2)
    if args.command=='evaluate':
        if args.benchmark is None or args.epoch is None:p.error('benchmark and epoch required')
        evaluate(args.benchmark,args.epoch,args.seed,args.device)
    else:matrix(args)

if __name__=='__main__':main()
