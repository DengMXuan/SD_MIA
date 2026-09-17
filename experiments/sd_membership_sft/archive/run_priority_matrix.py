"""Run registered follow-ups in priority order, with at most one fit per GPU."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import itertools
import json
import os
import queue
import subprocess
import sys
from ..difficulty_accept_only import (OUTPUT, feature_root)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('phase',choices=('sequence','features','posthoc'))
    p.add_argument('--gpus',default='',help='comma-separated physical GPU indices; empty uses CPU')
    p.add_argument('--jobs',type=int,default=3)
    p.add_argument('--ready-only',action='store_true')
    a=p.parse_args()
    devices=queue.Queue()
    gpu_ids=a.gpus.split(',') if a.gpus else []
    for device in gpu_ids:devices.put(device)
    jobs=[];missing=[]
    for benchmark,epoch,seed in itertools.product(('wikitection','newstection','arxivtection'),(1,3),(20260914,20260915,20260916)):
        target=OUTPUT/a.phase/f'{benchmark}_epoch{epoch}'/f'seed{seed}'
        if (target/'REPORT.json').exists():continue
        if a.phase=='features' and not (feature_root(benchmark,epoch)/'q.npy').exists():
            missing.append(f'{benchmark}_{epoch}');continue
        jobs.append((benchmark,epoch,seed))
    if missing and not a.ready_only:raise FileNotFoundError(str(missing))
    print(json.dumps({'phase':a.phase,'pending':len(jobs),'missing':missing}),flush=True)
    def run(job):
        benchmark,epoch,seed=job
        gpu=devices.get() if gpu_ids else None
        path=OUTPUT/'logs'/f'{a.phase}_{benchmark}_epoch{epoch}_seed{seed}.log'
        env=os.environ.copy()
        if gpu is not None:env['CUDA_VISIBLE_DEVICES']=gpu
        command=[sys.executable,'-m','experiments.sd_membership_sft.priority_accept_only',a.phase,
                 '--benchmark',benchmark,'--epoch',str(epoch),'--seed',str(seed),
                 '--device','cuda:0' if gpu is not None else 'cpu']
        try:
            with path.open('w') as log:subprocess.run(command,env=env,stdout=log,stderr=subprocess.STDOUT,check=True)
            print(json.dumps({'complete':job}),flush=True)
        finally:
            if gpu is not None:devices.put(gpu)
    with ThreadPoolExecutor(max_workers=min(a.jobs,len(gpu_ids)) if gpu_ids else a.jobs) as pool:
        list(pool.map(run,jobs))

if __name__=='__main__':main()
