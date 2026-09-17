"""Nonmember-only difficulty TCN, sparse scores and independent calibration.

The successful priority follow-ups live here. Historical causal-history,
ensemble and query-allocation experiments live in archive.priority_accept_only.
Language-model weights and all registered numerical settings are unchanged.
"""
from __future__ import annotations
import argparse
import copy
import hashlib
import json
import numpy as np
from scipy.special import logsumexp
import torch
from torch import nn
from .conditional_accept_only import (ConditionalCountTCN, Observations, record_partitions,
    observable_inputs, make_batch, count_nll)
from .replay_cache import load_replay_data
from .audit_metrics import membership_metrics, conformal_tail_pvalues
from .audit_runtime import ROOT, _paths, _write_json, _record_uniforms

RESULTS = ROOT / 'experiments/results/sft_runs'
OUTPUT = RESULTS / 'priority_validation'
TILTS = np.array([.5, 1., 2.])

def predict(model, x, counts, lengths, device, *, weights=False):
    offsets = np.r_[0, lengths.cumsum()]
    size = 18 if weights else 3
    out = np.empty((len(x), size), dtype=np.float32)
    model.eval()
    with torch.inference_mode():
        for start in range(0, len(lengths), 16):
            batch = np.arange(start, min(start+16, len(lengths)))
            bx, by, mask = make_batch(x, counts, offsets, batch, device)
            value = (model.mixture_log_weights(bx, mask) if weights else
                     model(bx, mask))
            value = value.cpu().numpy()
            for row, i in enumerate(batch):
                out[offsets[i]:offsets[i+1]] = value[row, :lengths[i]]
    return out


def fit(x, counts, lengths, parts, *, seed, device, epochs=30):
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    offsets = np.r_[0, lengths.cumsum()]
    tokens = np.concatenate([np.arange(offsets[i], offsets[i+1]) for i in parts['train']])
    mean, scale = x[tokens].mean(0), x[tokens].std(0)
    scale = np.where(scale < 1e-6, 1., scale)
    standardized = (x-mean)/scale
    model = ConditionalCountTCN(x.shape[1], 2).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-3)
    best, best_epoch, state, history = np.inf, 0, None, []
    for epoch in range(1, epochs+1):
        record = {'epoch': epoch}
        for phase in ('train','validation'):
            model.train(phase=='train')
            indices = rng.permutation(parts[phase]) if phase=='train' else parts[phase]
            total = 0.
            with torch.set_grad_enabled(phase=='train'):
                for start in range(0, len(indices), 16):
                    batch = indices[start:start+16]
                    bx, by, mask = make_batch(standardized, counts, offsets, batch, device)
                    prediction = model(bx, mask)
                    loss = count_nll(prediction, by, mask)
                    if not torch.isfinite(loss):
                        raise RuntimeError('nonfinite loss')
                    if phase=='train':
                        optimizer.zero_grad()
                        loss.backward()
                        nn.utils.clip_grad_norm_(model.parameters(), 1.)
                        optimizer.step()
                    total += float(loss.detach())*len(batch)
            record[phase+'_nll'] = total/len(indices)
        history.append(record)
        if epoch % 5 == 0:
            print(json.dumps({'causal': False, 'fit_seed': seed, **record}), flush=True)
        if record['validation_nll'] < best-1e-5:
            best, best_epoch, state = record['validation_nll'], epoch, copy.deepcopy(model.state_dict())
        if epoch-best_epoch >= 5:
            break
    model.load_state_dict(state)
    return model, mean, scale, history, best_epoch


def restored_baseline(benchmark, epoch, seed, device):
    folder = RESULTS / f'conditional_accept_only/{benchmark}_epoch{epoch}/b2_seed{seed}'
    checkpoint = torch.load(folder/'original.pt', weights_only=True, map_location='cpu')
    model = ConditionalCountTCN(checkpoint['input_dim'], checkpoint['k'], checkpoint['channels']).to(device)
    model.load_state_dict(checkpoint['state_dict'])
    with np.load(folder/'scores.npz', allow_pickle=False) as f:
        archive = dict(f)
    return model, checkpoint['mean'].numpy(), checkpoint['scale'].numpy(), archive


def token_evidence(logpmf, counts):
    logpmf = np.asarray(logpmf, dtype=float)
    return (np.asarray(counts, dtype=float)[:,None]*TILTS -
            logsumexp(logpmf[:,:,None] + np.arange(3)[None,:,None]*TILTS, axis=1))


def aggregate(logpmf, counts, lengths, discount=None, sparse=False):
    result = np.empty(len(lengths))
    offsets = np.r_[0, lengths.cumsum()]
    for i,(start,end) in enumerate(zip(offsets[:-1], offsets[1:])):
        evidence = token_evidence(logpmf[start:end], counts[start:end])
        if discount is not None:
            evidence *= discount[start:end,None]
        if sparse:
            values = [np.logaddexp(np.log1p(-rho), np.log(rho)+evidence).sum(0) for rho in (.05,.1,.25)]
            result[i] = logsumexp(np.concatenate(values))-np.log(9)
        else:
            result[i] = logsumexp(evidence.sum(0))-np.log(3)
    return result


def record_nll(logpmf, counts, lengths, indices):
    losses = -logpmf[np.arange(len(counts)), counts]
    offsets = np.r_[0, lengths.cumsum()]
    return float(np.mean([losses[offsets[i]:offsets[i+1]].mean() for i in indices]))


def save_fit(path, model, mean, scale, history, best_epoch):
    torch.save({'state_dict': {k:v.cpu() for k,v in model.state_dict().items()},
                'mean': torch.tensor(mean), 'scale': torch.tensor(scale),
                'causal': False}, path)
    return {'best_epoch': best_epoch, 'history': history}


def feature_root(benchmark, epoch):
    old = RESULTS / f'm1_conditional/{benchmark}_epoch{epoch}/draft_auxiliary_distilled/features_without_eos'
    return old if (old/'q.npy').exists() else OUTPUT / f'features/{benchmark}_epoch{epoch}'


def load_feature_observations(benchmark, epoch, seed):
    data = load_replay_data(*_paths(benchmark, epoch))
    folder = feature_root(benchmark, epoch)
    q = np.load(folder/'q.npy', mmap_mode='r')
    for name, expected in (('lengths',data.lengths),('record_ids',data.record_ids)):
        if not np.array_equal(np.load(folder/f'{name}.npy'), expected):
            raise ValueError('feature record alignment failed')
    if (folder/'feature_manifest.json').exists():
        manifest = json.loads((folder/'feature_manifest.json').read_text())
        if (manifest['benchmark'] != benchmark or manifest['epoch'] != epoch or
                manifest['role'] != 'draft_auxiliary_distilled' or manifest['eos_included']):
            raise ValueError('wrong feature provenance')
        expected_path = RESULTS / f'{benchmark}_qwen3_8b_epoch{epoch}/checkpoints/draft_auxiliary_distilled'
        if Path(manifest['checkpoint_provenance']['checkpoint_path']).resolve() != expected_path.resolve():
            raise ValueError('wrong frozen draft')
        if hashlib.sha256(_paths(benchmark,epoch)[1].read_bytes()).hexdigest() != manifest['probability_cache']['sha256']:
            raise ValueError('source probability cache changed')
    else:
        manifest = json.loads((folder/'SOURCE.json').read_text())
        if not manifest['models_frozen'] or manifest['provenance']['role'] != 'draft_auxiliary_distilled':
            raise ValueError('wrong new feature provenance')
    if q.shape != (len(data.logq0),6) or not np.isfinite(q).all() or np.any(q[:,0]>1e-5):
        raise ValueError('invalid draft-only features')
    bits = np.empty((len(q),1,2),dtype=np.uint8)
    for i,(start,end) in enumerate(zip(data.offsets[:-1],data.offsets[1:])):
        alpha = np.exp(np.minimum(0., data.logp[start:end]-q[start:end,0]))
        bits[start:end,0] = _record_uniforms(seed,i,end-start,2)<alpha[:,None]
    obs = Observations(np.array(q[:,:1]), bits, data.lengths)
    return obs, data.labels, data.record_ids, np.asarray(q[:,1:4]), {'feature_dir':str(folder),
        'q_difference_max':float(np.abs(q[:,0]-data.logq0).max()),
        'protocol':'bits regenerated with feature q; matched q-only baseline refitted'}


def run_features(args, output):
    obs, labels, ids, extra, source = load_feature_observations(args.benchmark,args.epoch,args.seed)
    parts = record_partitions(labels,ids)
    x, counts, _ = observable_inputs(obs,2,False)
    scores, training, diagnostics = {}, {}, {}
    for name, features in (('baseline',x),('difficulty',np.column_stack((x,extra)))):
        model, mean, scale, history, best = fit(features,counts,obs.lengths,parts,seed=args.seed,device=args.device)
        pmf = predict(model,(features-mean)/scale,counts,obs.lengths,args.device)
        scores[name] = aggregate(pmf,counts,obs.lengths)
        training[name] = save_fit(output/f'{name}.pt',model,mean,scale,history,best)
        diagnostics[name+'_validation_nll'] = record_nll(pmf,counts,obs.lengths,parts['validation'])
        del model, pmf
    finish(args,output,scores,labels,ids,parts,training=training,diagnostics=diagnostics,source=source,
           protocol='feature-consistent B=2 replay; entropy, log-rank, top1-top2 margin')


def expanded_calibration(labels, parts, seed=20260917):
    used = np.unique(np.r_[parts['reference'],parts['calibration'],parts['test']])
    unused = np.flatnonzero((labels==0)&~np.isin(np.arange(len(labels)),used))
    result = np.r_[parts['calibration'],np.random.default_rng(seed).permutation(unused)]
    if len(result)!=1200 or len(np.unique(result))!=1200:
        raise ValueError('expected 1200 disjoint calibration nonmembers')
    return result


def group_pvalues(scores, calibration, test, groups):
    pvalues = np.ones(len(test))
    for group in np.unique(groups[test]):
        selected = np.flatnonzero(groups[test]==group)
        reference = calibration[groups[calibration]==group]
        if len(reference):
            pvalues[selected] = conformal_tail_pvalues(scores[test[selected]],scores[reference])
    return pvalues


def calibration_analysis(scores,labels,parts,lengths,logq, *, return_pvalues=False):
    full = expanded_calibration(labels,parts)
    offsets=np.r_[0,lengths.cumsum()]
    difficulty=np.array([logq[s:e].mean() for s,e in zip(offsets[:-1],offsets[1:])])
    groups={'length':(lengths>=np.median(lengths[parts['reference']])).astype(int),
            'difficulty':(difficulty>=np.median(difficulty[parts['reference']])).astype(int)}
    test=parts['test']; y=labels[test]
    variants={f'pooled_{size}':conformal_tail_pvalues(scores[test],scores[full[:size]]) for size in (200,400,800,1200)}
    variants.update({f'mondrian_{name}':group_pvalues(scores,full,test,g) for name,g in groups.items()})
    result={}
    for name,pvalues in variants.items():
        result[name]={}
        for level in (.01,.05,.1):
            hits=pvalues<=level
            row={'tpr':float(hits[y==1].mean()),'actual_fpr':float(hits[y==0].mean())}
            row['groups']={}
            for group_name,g in groups.items():
                row['groups'][group_name]={}
                for value in (0,1):
                    selected=(y==0)&(g[test]==value)
                    row['groups'][group_name][str(value)]={'test_nonmembers':int(selected.sum()),
                        'calibration_nonmembers':int((g[full]==value).sum()),
                        'fpr':float(hits[selected].mean()) if selected.any() else None}
            result[name][str(level)]=row
    if return_pvalues:
        return result, full, variants, groups
    return result, full


def finish(args,output,scores,labels,ids,parts,**details):
    metrics={name:membership_metrics(values,labels,parts['calibration'],parts['test']) for name,values in scores.items()}
    np.savez_compressed(output/'scores.npz',labels=labels,record_ids=ids,**parts,**scores)
    report={'benchmark':args.benchmark,'epoch':args.epoch,'seed':args.seed,'phase':args.phase,
            'training_member_count':0,'synthetic_member_count':0,'language_models_frozen':True,
            'metrics':metrics,**details}
    _write_json(output/'REPORT.json',report)
    print(json.dumps({'complete':str(output),'auc':{k:v['auc'] for k,v in metrics.items()}}),flush=True)



def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--benchmark', choices=('wikitection', 'newstection', 'arxivtection'), required=True)
    parser.add_argument('--epoch', type=int, choices=(1, 3), required=True)
    parser.add_argument('--seed', type=int, default=20260914)
    parser.add_argument('--device', default='cpu')
    parser.add_argument('--threads', type=int, default=2)
    args = parser.parse_args()
    args.phase = 'features'
    torch.set_num_threads(args.threads)
    output = OUTPUT / args.phase / f'{args.benchmark}_epoch{args.epoch}' / f'seed{args.seed}'
    if (output / 'REPORT.json').exists():
        print('already complete', flush=True)
        return
    output.mkdir(parents=True, exist_ok=True)
    run_features(args, output)


if __name__ == '__main__':
    main()

