#!/usr/bin/env python3
"""Independent document-level Pythia delta audit (no experiments imports).

Read frozen test IDs, collect exact selected-token log p and log q offline,
and compare document summaries and document-balanced token distributions.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import time

import numpy as np
from scipy.stats import ks_2samp, rankdata

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DATA = Path('/home/mxd/lib/SD_MIA-pretraining-data/mimir/prepared')
DEFAULT_AUDIT = Path('/home/mxd/lib/SD_MIA/artifacts/audits/pythia_mimir_v1')
SOURCES = ('github', 'wikipedia_(en)', 'dm_mathematics', 'arxiv',
           'hackernews', 'pile_cc', 'pubmed_central', 'full_pile')
SCHEMA = 'pythia_exact_delta_v1'
CONTRACT = {'mode': 'raw_text_completion', 'add_special_tokens': False,
    'first_token': 'context_only', 'append_eos': False,
    'scored_tokens': 'text tokens at positions 1..L-1',
    'vocabulary': 'shared tokenizer IDs only; padded LM-head rows excluded and distributions renormalized'}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def read_json(path):
    return json.loads(Path(path).read_text())


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1 << 20), b''):
            h.update(block)
    return h.hexdigest()


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def write_json(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + '\n')
    temporary.replace(path)


def selection(args):
    conditions, items, models, tokenizer_sha = [], [], None, None
    for source in args.datasets:
        for seed in args.seeds:
            manifest_path = args.data_root / source / f'seed{seed}' / 'manifest.json'
            manifest = read_json(manifest_path)
            require(manifest['kind'] == 'mimir_pretraining_v1' and manifest['token_contract'] == CONTRACT,
                    'not a compatible frozen Pythia corpus')
            require(manifest['models']['target']['repo_id'] == 'EleutherAI/pythia-6.9b' and
                    manifest['models']['draft']['repo_id'] == 'EleutherAI/pythia-1.4b', 'wrong model pair')
            records_path = manifest_path.parent / manifest['records_file']
            require(sha256(records_path) == manifest['records_sha256'], 'frozen records changed')
            if models is None:
                models, tokenizer_sha = manifest['models'], manifest['tokenizer_sha256']
            require(models == manifest['models'] and tokenizer_sha == manifest['tokenizer_sha256'],
                    'mixed models/tokenizers')
            task = args.audit_root / 'tasks' / source / f'seed{seed}'
            partition = read_json(task / 'PARTITIONS.json')
            require(partition['seed'] == seed and partition['manifest_sha256'] == sha256(manifest_path),
                    'partition provenance mismatch')
            allowed = set(partition['record_ids']['test'])
            records = [json.loads(line) for line in records_path.read_text().splitlines()]
            require(len({r['record_id'] for r in records}) == len(records), 'duplicate record ID')
            selected = []
            for label in (1, 0):
                candidates = sorted((r for r in records if r['record_id'] in allowed and r['label'] == label),
                                    key=lambda r: r['record_id'])
                require(all(r['group'] == ('member' if label else 'nonmember') for r in candidates),
                        'auxiliary leakage or label mismatch')
                n = len(candidates) if args.per_class == 0 else args.per_class
                require(2 <= n <= len(candidates), 'not enough frozen test documents')
                rng_seed = int(digest([args.sample_seed, source, seed, label])[:16], 16)
                chosen = np.sort(np.random.default_rng(rng_seed).choice(len(candidates), n, replace=False))
                selected.extend(candidates[i] for i in chosen)
            for r in selected:
                ids = r['token_ids']
                require(2 <= len(ids) <= manifest['max_tokens'] and all(type(t) is int and t >= 0 for t in ids),
                        'invalid frozen tokens')
                r = dict(r, condition=f'{source}/seed{seed}', seed=seed, domain=source)
                r['key'] = digest({'tokens': ids, 'models': models, 'dtype': args.dtype,
                                   'device_type': args.device.split(':')[0], 'schema': SCHEMA})
                items.append(r)
            conditions.append(dict(source=source, seed=seed,
                files={str(p.resolve()): sha256(p) for p in
                       (manifest_path, records_path, task / 'PARTITIONS.json', task / 'observations.npz',
                        task / 'main_fixed_sparse_positive' / 'scores.npz')},
                record_ids=[r['record_id'] for r in selected]))
    plan = dict(schema=SCHEMA, models=models, tokenizer_sha256=tokenizer_sha, token_contract=CONTRACT,
                dtype=args.dtype, device_type=args.device.split(':')[0], attention='sdpa',
                sample_seed=args.sample_seed, per_class=args.per_class, conditions=conditions,
                analysis_unit='document; each domain/seed reported separately',
                scope='exploratory mechanism verification on previously inspected frozen test sets',
                original_feedback_dtype='GPU bfloat16; fresh CPU float32 logps are not bitwise replays')
    return plan, items


def cache_path(output, role, item):
    return output / 'logps' / role / (item['key'] + '.npz')


def load_cached(output, role, item):
    path = cache_path(output, role, item)
    with np.load(path, allow_pickle=False) as z:
        require(str(z['key']) == item['key'] and np.array_equal(z['token_ids'], item['token_ids']),
                'logp cache provenance mismatch')
        values = z['logps'].astype(np.float64)
        require(values.shape == (len(item['token_ids']) - 1,) and np.isfinite(values).all()
                and (values <= 1e-5).all(), 'invalid logps')
    return values


def collect(args, plan, items):
    import gc
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from transformers.utils import logging as hf_logging
    hf_logging.disable_progress_bar()
    torch.set_num_threads(args.threads)
    torch.set_num_interop_threads(1)
    torch.manual_seed(args.sample_seed)
    device = torch.device(args.device)
    dtype = getattr(torch, args.dtype)
    runtime = dict(torch=torch.__version__, threads=args.threads, device=str(device), dtype=args.dtype,
                   script_sha256=sha256(__file__), started=time.time())
    write_json(args.output / 'RUNTIME.json', runtime)
    for role in ('target', 'draft'):
        spec = plan['models'][role]
        model_path = args.model_root / ('models--' + spec['repo_id'].replace('/', '--')) / 'snapshots' / spec['revision']
        tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
        state = json.loads(tokenizer.backend_tokenizer.to_str())
        state.pop('truncation', None); state.pop('padding', None)
        require(hashlib.sha256(json.dumps(state, sort_keys=True).encode()).hexdigest() == plan['tokenizer_sha256'],
                'pinned tokenizer changed')
        require(set(tokenizer.get_vocab().values()) == set(range(len(tokenizer))), 'noncontiguous vocabulary')
        require(all(max(r['token_ids']) < len(tokenizer) for r in items), 'out of vocabulary token')
        unique = {r['key']: r for r in items}
        pending = []
        for r in unique.values():
            if cache_path(args.output, role, r).exists():
                load_cached(args.output, role, r)
            else:
                pending.append(r)
        if not pending:
            print(f'{role}: all {len(unique)} documents already cached', flush=True)
            continue
        (args.output / 'logps' / role).mkdir(parents=True, exist_ok=True)
        print(f'{role}: loading {model_path}; {len(pending)} documents pending', flush=True)
        model = AutoModelForCausalLM.from_pretrained(model_path, local_files_only=True,
                dtype=dtype, attn_implementation='sdpa').to(device).eval().requires_grad_(False)
        started = time.monotonic()
        for index, item in enumerate(pending):
            ids = torch.tensor([item['token_ids']], device=device)
            with torch.inference_mode():
                logits = model(input_ids=ids, use_cache=False).logits[0, :-1, :len(tokenizer)]
                chunks = []
                for start in range(0, logits.shape[0], 64):
                    rows = logits[start:start+64].float()
                    # Exactly the same support and FP32 log_softmax as fixed_trace.
                    values = torch.log_softmax(rows, -1).gather(-1, ids[0, start+1:start+1+len(rows), None]).squeeze(-1)
                    chunks.append(values.cpu().numpy())
                logps = np.concatenate(chunks).astype(np.float32)
            require(np.isfinite(logps).all() and (logps <= 1e-5).all(), 'nonfinite/positive log probabilities')
            path = cache_path(args.output, role, item)
            temporary = path.with_suffix('.tmp')
            with temporary.open('wb') as f:
                np.savez_compressed(f, key=np.asarray(item['key']), token_ids=np.asarray(item['token_ids']), logps=logps)
            temporary.replace(path)
            del logits, chunks, rows, values, ids
            if (index + 1) % 8 == 0 or index == 0 or index + 1 == len(pending):
                elapsed = time.monotonic() - started
                print(f'{role}: {index+1}/{len(pending)}, {elapsed:.1f}s, {item["condition"]}', flush=True)
        del model
        gc.collect()
        if device.type == 'cuda':
            torch.cuda.empty_cache()
    runtime['finished'] = time.time()
    write_json(args.output / 'RUNTIME.json', runtime)


def auc(x, y):
    n = len(x)
    return float((rankdata(np.r_[x, y])[:n].sum() - n * (n + 1) / 2) / (n * len(y)))


def compare(x, y, rng, repeats):
    ix = rng.integers(len(x), size=(repeats, len(x)))
    iy = rng.integers(len(y), size=(repeats, len(y)))
    boot_diff = x[ix].mean(1) - y[iy].mean(1)
    boot_auc = np.array([auc(x[i], y[j]) for i, j in zip(ix, iy)])
    pooled_sd = np.sqrt(((len(x)-1)*x.var(ddof=1) + (len(y)-1)*y.var(ddof=1))/(len(x)+len(y)-2))
    ks = ks_2samp(x, y)
    return dict(member_mean=float(x.mean()), nonmember_mean=float(y.mean()),
                difference=float(x.mean() - y.mean()), difference_ci95=np.quantile(boot_diff, [.025,.975]).tolist(),
                auc=auc(x, y), auc_ci95=np.quantile(boot_auc, [.025,.975]).tolist(),
                cohen_d=float((x.mean()-y.mean())/pooled_sd) if pooled_sd > 0 else None,
                ks_statistic=float(ks.statistic), ks_pvalue=float(ks.pvalue))


def holm(pvalues):
    order = np.argsort(pvalues)
    result = np.empty(len(pvalues))
    running = 0.
    for rank, index in enumerate(order):
        running = max(running, (len(order) - rank) * pvalues[index])
        result[index] = min(1., running)
    return result.tolist()


def feedback(args, source, seed):
    task = args.audit_root / 'tasks' / source / f'seed{seed}'
    with np.load(task / 'observations.npz', allow_pickle=False) as z:
        require(np.unique(z['start_indices']).tolist() == [0], 'expected one fixed trace per document')
        ids, labels = z['record_ids'], z['labels']
        require(len(np.unique(ids)) == len(ids), 'duplicate feedback IDs')
        lengths, features, counts = z['lengths'], z['features'], z['counts']
        require(np.all(counts <= 2) and features.shape[1] == 6, 'not B=2 feedback')
        offsets = np.r_[0, np.cumsum(lengths)]
        require(offsets[-1] == len(counts) == len(features), 'broken feedback alignment')
        result = {str(r): dict(label=int(labels[i]), logq=features[offsets[i]:offsets[i+1],0].astype(np.float64),
                              counts=counts[offsets[i]:offsets[i+1]].copy()) for i,r in enumerate(ids)}
    with np.load(task / 'main_fixed_sparse_positive' / 'scores.npz', allow_pickle=False) as z:
        # The original score file includes its own explicit document identities.
        require('record_ids' in z.files and 'scores' in z.files, 'unknown legacy score schema')
        for r, score in zip(z['record_ids'], z['scores']):
            result[str(r)]['legacy_score'] = float(score)
    return result


def analyze(args, plan, items):
    reports, documents, curve_data = [], [], []
    grid = np.linspace(-2, 2, 161)
    for condition in plan['conditions']:
        source, seed = condition['source'], condition['seed']
        rows = [r for r in items if r['domain'] == source and r['seed'] == seed]
        old = feedback(args, source, seed)
        stats, cdfs, labels, alignment, noise, lengths = [], [], [], [], [], []
        for r in rows:
            p, q = load_cached(args.output, 'target', r), load_cached(args.output, 'draft', r)
            d = p-q
            cached = old[r['record_id']]
            require(cached['label'] == r['label'] and len(cached['logq']) == len(d), 'feedback/record alignment mismatch')
            a = np.exp(np.minimum(d, 0))
            metrics = dict(mean_delta=d.mean(), mean_abs_delta=np.abs(d).mean(), delta_std=d.std(),
                negative_delta_mass=np.minimum(d,0).mean(), positive_delta_mass=np.maximum(d,0).mean(),
                negative_fraction=(d<0).mean(), delta_q10=np.quantile(d,.1), delta_median=np.median(d),
                delta_q90=np.quantile(d,.9), exact_alpha=a.mean(), observed_acceptance=cached['counts'].mean()/2,
                mean_logp=p.mean(), mean_logq=q.mean(), legacy_score=cached['legacy_score'], length=len(d))
            metrics = {k:float(v) for k,v in metrics.items()}
            documents.append(dict(source=source, seed=seed, record_id=r['record_id'], label=r['label'], **metrics))
            stats.append(metrics); labels.append(r['label']); lengths.append(len(d))
            alignment.append(float(np.abs(q-cached['logq']).max()))
            noise.append(float(a.mean()-cached['counts'].mean()/2))
            cdfs.append(np.searchsorted(np.sort(d),grid,side='right')/len(d))
        labels, cdfs = np.asarray(labels), np.asarray(cdfs)
        rng = np.random.default_rng(int(digest(['stats',source,seed,args.sample_seed])[:16],16))
        metrics = {k:compare(np.array([s[k] for s,l in zip(stats,labels) if l==1]),
                            np.array([s[k] for s,l in zip(stats,labels) if l==0]), rng, args.bootstrap)
                   for k in stats[0]}
        observed = np.max(np.abs(cdfs[labels==1].mean(0)-cdfs[labels==0].mean(0)))
        null = []
        for _ in range(args.bootstrap):
            perm = rng.permutation(labels)
            null.append(np.max(np.abs(cdfs[perm==1].mean(0)-cdfs[perm==0].mean(0))))
        balanced = dict(sup_cdf_difference_on_grid=float(observed),
                        permutation_pvalue=float((1+np.sum(np.asarray(null)>=observed))/(1+len(null))),
                        grid_range=[-2,2], grid_points=len(grid), unit='whole document label permutations')
        report = dict(source=source, seed=seed, member_documents=int(sum(labels==1)), nonmember_documents=int(sum(labels==0)),
            tokens=int(sum(lengths)), metrics=metrics, document_balanced_delta_cdf=balanced,
            precision_comparison=dict(max_abs_fresh_fp32_minus_original_bf16_logq=float(max(alignment)),
                mean_exact_alpha_minus_original_sampled_acceptance=float(np.mean(noise))))
        reports.append(report); curve_data.append((source,seed,cdfs,labels,stats))
    corrected = holm([r['document_balanced_delta_cdf']['permutation_pvalue'] for r in reports])
    for r,p in zip(reports,corrected):
        r['document_balanced_delta_cdf']['holm_pvalue_across_conditions'] = p
    report = dict(plan_sha256=sha256(args.output/'PLAN.json'), bootstrap=args.bootstrap, results=reports,
        notes=['AUC uses fixed direction higher=member. No direction is selected from test labels.',
               'Document bootstrap intervals are pointwise; secondary metrics are exploratory.',
               'Holm correction is applied only to the primary document-balanced delta-CDF permutation tests.',
               'Seeds/domains may share documents; never pool them as independent observations.',
               'Float32 p/q measure the pinned models without quantization. Archived feedback used GPU BF16.',
               'A document is the frozen MIMIR excerpt (at most 512 tokens), not necessarily a complete source document.',
               'Exact delta is mechanism-analysis information unavailable to the accept-only attacker.',
               'Failure to reject does not establish equality; membership labels are the official MIMIR labels.'])
    write_json(args.output/'REPORT.json',report)
    with (args.output/'documents.csv').open('w') as f:
        writer=csv.DictWriter(f,fieldnames=list(documents[0]));writer.writeheader();writer.writerows(documents)
    text=['# Pythia document-level delta verification','',
          'Delta = log p − log q (natural log); exact teacher-forced probabilities, equal document weight.', '',
          '| Domain / seed | n per class | Mean delta M / N | M−N [95% CI] | AUC(mean delta) [95% CI] | AUC(exact alpha) | AUC(B2) | CDF Holm p |',
          '|---|---:|---|---|---|---:|---:|---:|']
    for r in reports:
        m=r['metrics']; d=m['mean_delta']
        ci=lambda x:f'[{x[0]:.4f}, {x[1]:.4f}]'
        text.append(f'| {r["source"]} / {r["seed"]} | {r["member_documents"]}/{r["nonmember_documents"]} | '
            f'{d["member_mean"]:.4f} / {d["nonmember_mean"]:.4f} | {d["difference"]:.4f} {ci(d["difference_ci95"])} | '
            f'{d["auc"]:.4f} {ci(d["auc_ci95"])} | {m["exact_alpha"]["auc"]:.4f} | '
            f'{m["observed_acceptance"]["auc"]:.4f} | {r["document_balanced_delta_cdf"]["holm_pvalue_across_conditions"]:.4f} |')
    text += ['', '## Interpretation limits', ''] + ['- '+note for note in report['notes']]
    (args.output/'REPORT.md').write_text('\n'.join(text)+'\n')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(len(curve_data),3,figsize=(13,3.4*len(curve_data)),squeeze=False)
    for row,(source,seed,cdfs,labels,stats) in enumerate(curve_data):
        for label,color,name in ((1,'#c65c27','Member'),(0,'#287ab5','Nonmember')):
            indices=np.flatnonzero(labels==label)
            axes[row,0].plot(grid,cdfs[indices].mean(0),color=color,label=name)
            for col,key in ((1,'mean_delta'),(2,'exact_alpha')):
                values=np.sort([stats[i][key] for i in indices])
                axes[row,col].step(values,np.arange(1,len(values)+1)/len(values),where='post',color=color,label=name)
        for col,title in enumerate(('Document-balanced token delta CDF','ECDF of document mean delta','ECDF of document exact acceptance')):
            axes[row,col].set_title(source+f' / {seed}\n'+title)
            axes[row,col].set_ylim(0,1.02);axes[row,col].grid(alpha=.2);axes[row,col].legend()
        axes[row,0].set_xlabel('Token delta (display window; full data used for summaries)')
        axes[row,1].set_xlabel('Mean log p − log q');axes[row,2].set_xlabel('Mean exp(min(delta, 0))')
    fig.tight_layout()
    fig.savefig(args.output/'delta_distributions.png',dpi=180)
    fig.savefig(args.output/'delta_distributions.pdf')
    plt.close(fig)
    write_json(args.output/'_COMPLETE.json', {name:sha256(args.output/name) for name in
        ('PLAN.json','REPORT.json','REPORT.md','documents.csv','delta_distributions.png','delta_distributions.pdf')})
    print((args.output/'REPORT.md').read_text(),flush=True)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--stage',choices=('all','collect','analyze'),default='all')
    p.add_argument('--datasets',nargs='+',choices=SOURCES,default=list(SOURCES[:3]))
    p.add_argument('--seeds',nargs='+',type=int,default=[1919])
    p.add_argument('--per-class',type=int,default=64,help='0 uses the complete frozen test set')
    p.add_argument('--sample-seed',type=int,default=20260927)
    p.add_argument('--bootstrap',type=int,default=2000)
    p.add_argument('--threads',type=int,default=8)
    p.add_argument('--device',default='cpu')
    p.add_argument('--dtype',choices=('float32','bfloat16'),default='float32')
    p.add_argument('--data-root',type=Path,default=DEFAULT_DATA)
    p.add_argument('--audit-root',type=Path,default=DEFAULT_AUDIT)
    p.add_argument('--model-root',type=Path,default=Path('/home/mxd/.cache/huggingface/hub'))
    p.add_argument('--output',type=Path,default=ROOT/'artifacts/audits/pythia_delta_pilot_v1')
    args=p.parse_args()
    require(args.bootstrap>=100 and args.threads>=1 and args.per_class>=0,'invalid computation sizes')
    require(len(set(args.datasets))==len(args.datasets) and len(set(args.seeds))==len(args.seeds),'duplicate conditions')
    args.output=args.output.resolve()
    for inp in (args.data_root,args.audit_root,args.model_root):
        inp=inp.resolve()
        require(args.output!=inp and inp not in args.output.parents and args.output not in inp.parents,'output overlaps input tree')
    args.output.mkdir(parents=True,exist_ok=True)
    import fcntl
    with (args.output/'.lock').open('w') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        plan,items=selection(args)
        if (args.output/'PLAN.json').exists():
            require(read_json(args.output/'PLAN.json')==plan,'output already belongs to a different frozen plan')
        else:
            write_json(args.output/'PLAN.json',plan)
        if args.stage in ('all','collect'):
            collect(args,plan,items)
        if args.stage in ('all','analyze'):
            analyze(args,plan,items)


if __name__=='__main__':
    main()
