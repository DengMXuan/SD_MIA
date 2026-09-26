#!/usr/bin/env python3
"""Prepare and run independent, versioned Qwen temporal cleaning conditions."""
import argparse
import json
import os
from pathlib import Path
import sys

import prepare as data

sys.path.insert(0,str(data.ROOT))


def worker(args):
    from experiments.pretraining.evaluation import evaluate_main
    data.validate_output(args.data_root/f'seed{args.seed}')
    manifest=args.data_root/f'seed{args.seed}'/args.variant/'manifest.json'
    report=evaluate_main(manifest,args.output_root/args.variant/f'seed{args.seed}',
                         seed=args.seed,device='cuda:0',detector_epochs=args.detector_epochs)
    print(json.dumps(report['metrics']),flush=True)


def main():
    from experiments.shared.core import gpu_pool
    parser=argparse.ArgumentParser(description=__doc__,allow_abbrev=False)
    parser.add_argument('command',choices=('prepare','dry-run','run','summarize','_worker'))
    parser.add_argument('--source-root',type=Path,default=data.SOURCE)
    parser.add_argument('--data-root',type=Path,default=data.OUTPUT)
    parser.add_argument('--output-root',type=Path,default=data.ROOT/'artifacts/audits/qwen_temporal_clean_v2/tasks')
    parser.add_argument('--model-root',type=Path,default=Path('/home/mxd/.cache/huggingface/hub'))
    parser.add_argument('--seeds',type=int,nargs='+',choices=data.SEEDS,default=list(data.SEEDS))
    parser.add_argument('--variants',nargs='+',choices=data.VARIANTS,default=['length_matched'])
    parser.add_argument('--detector-epochs',type=int,default=30)
    parser.add_argument('--seed',type=int,choices=data.SEEDS,help=argparse.SUPPRESS)
    parser.add_argument('--variant',choices=data.VARIANTS,help=argparse.SUPPRESS)
    gpu_pool.add_arguments(parser)
    args=parser.parse_args()
    args.data_root=args.data_root.resolve();args.output_root=args.output_root.resolve()
    data.require(len(set(args.seeds))==len(args.seeds) and len(set(args.variants))==len(args.variants),'duplicate conditions')
    data.require(args.detector_epochs>0,'detector epochs must be positive')
    if args.command=='_worker':
        data.require(args.seed is not None and args.variant is not None,'worker condition missing')
        worker(args);return
    if args.command=='prepare':
        # Always generate both to separate surface cleaning from length matching.
        for seed in args.seeds:
            data.prepare(args.source_root,args.data_root,seed,model_root=args.model_root)
        data.summarize_data(args.data_root,args.seeds)
        return
    tasks=[]
    for variant in args.variants:
        for seed in args.seeds:
            folder=args.data_root/f'seed{seed}'
            request=data.validate_output(folder)
            data.require(variant in request['variants'],'variant not prepared')
            manifest=folder/variant/'manifest.json'
            output=args.output_root/variant/f'seed{seed}'
            data.require(output!=args.data_root and args.data_root not in output.parents
                         and output not in args.data_root.parents,'evaluation output overlaps derived data')
            tasks.append(dict(seed=seed,variant=variant,manifest=str(manifest),output=str(output)))
    if args.command=='dry-run':
        print(json.dumps(dict(scheduling=gpu_pool.configuration(args),tasks=tasks),indent=2));return
    if args.command=='summarize':
        from experiments.shared.audit.artifacts import read_result
        rows=[]
        for t in tasks:
            folder=Path(t['output'])/'main_fixed_sparse_positive'
            row={**t,'state':'missing'}
            if (folder/'REPORT.json').exists():
                report=read_result(folder)
                data.require(report['evaluation_context']['data_manifest']==t['manifest'] and
                    report['settings']['audit_seed']==t['seed'] and
                    report['settings']['detector_epochs']==args.detector_epochs,'result belongs to another condition')
                row.update(state='complete',metrics=report['metrics'])
            rows.append(row)
        print(json.dumps(rows,indent=2));return
    jobs=[]
    for t in tasks:
        command=[sys.executable,'-B','-u',str(Path(__file__).resolve()),'_worker',
                 '--data-root',str(args.data_root),'--output-root',str(args.output_root),
                 '--seed',str(t['seed']),'--variant',t['variant'],'--detector-epochs',str(args.detector_epochs)]
        jobs.append(gpu_pool.Job(f'{t["variant"]}/seed{t["seed"]}',command,t['seed']))
    # Same pinned pretraining evaluator; all outputs/caches/checkpoints are new.
    rows=gpu_pool.run_jobs(jobs,scheduling=gpu_pool.configuration(args),cwd=data.ROOT,
        log_root=args.log_root or args.output_root.parent/'executions',use_cuda=True)
    if any(r['state']!='complete' for r in rows):raise SystemExit(2)


if __name__=='__main__':
    os.environ.setdefault('HF_HUB_OFFLINE','1')
    os.environ.setdefault('TRANSFORMERS_OFFLINE','1')
    main()
