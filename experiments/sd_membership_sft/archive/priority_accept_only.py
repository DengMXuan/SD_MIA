"""Registered follow-up ablations: causal counts, draft difficulty and uncertainty.

Only observable q/features and accept bits cross the detector boundary.
Language-model checkpoints are never trained. Membership labels are consumed
only by split checks and final metrics, never by fitting or method selection.
"""
from __future__ import annotations
import argparse
import copy
import hashlib
import json
from pathlib import Path
import numpy as np
from scipy.special import logsumexp, expit
import torch
from torch import nn
from torch.nn import functional as F
from ..conditional_accept_only import (ConditionalCountTCN, Observations, record_partitions, observable_inputs, make_batch, count_nll, replay_observations, directional_scores)
from ..replay_cache import (load_replay_data)
from ..audit_metrics import (membership_metrics)
from ..audit_runtime import (_record_uniforms)
from ..audit_metrics import (conformal_tail_pvalues)
from ..audit_runtime import (ROOT, _paths, _write_json)

from ..difficulty_accept_only import (RESULTS, OUTPUT, TILTS, restored_baseline, token_evidence, aggregate, record_nll, feature_root, load_feature_observations, expanded_calibration, group_pvalues, calibration_analysis, finish)



class CausalCountTCN(nn.Module):
    """Static bidirectional q context plus STRICTLY causal feedback history."""
    def __init__(self, input_dim=2, k=2, channels=24):
        super().__init__()
        self.static = ConditionalCountTCN(input_dim, k, channels)
        self.k = k
        self.projection = nn.Linear(k+1, channels)
        self.convs = nn.ModuleList([nn.Conv1d(channels, channels, 3, dilation=d) for d in (1,2,4,8)])
        self.norms = nn.ModuleList([nn.LayerNorm(channels) for _ in self.convs])
        self.head = nn.Linear(channels, self.static.head.out_features)

    def forward(self, features, mask, counts):
        history = F.one_hot(counts.long(), self.k+1).to(features.dtype)
        history = F.pad(history[:, :-1], (0, 0, 1, 0))
        valid = mask.unsqueeze(-1).to(features.dtype)
        hidden = F.gelu(self.projection(history)) * valid
        for dilation, conv, norm in zip((1,2,4,8), self.convs, self.norms):
            update = conv(F.pad(hidden.transpose(1,2), (2*dilation, 0))).transpose(1,2)
            hidden = (hidden + F.gelu(norm(update))) * valid
        weights = F.log_softmax(self.static.mixture_log_weights(features, mask) + self.head(hidden), -1)
        return torch.logsumexp(weights.unsqueeze(-2) + self.static.log_kernel, -1)


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
                     model(bx, mask, by) if isinstance(model, CausalCountTCN) else model(bx, mask))
            value = value.cpu().numpy()
            for row, i in enumerate(batch):
                out[offsets[i]:offsets[i+1]] = value[row, :lengths[i]]
    return out


def fit(x, counts, lengths, parts, *, seed, device, causal=False, epochs=30):
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    offsets = np.r_[0, lengths.cumsum()]
    tokens = np.concatenate([np.arange(offsets[i], offsets[i+1]) for i in parts['train']])
    mean, scale = x[tokens].mean(0), x[tokens].std(0)
    scale = np.where(scale < 1e-6, 1., scale)
    standardized = (x-mean)/scale
    model = (CausalCountTCN(x.shape[1]) if causal else ConditionalCountTCN(x.shape[1], 2)).to(device)
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
                    prediction = model(bx, mask, by) if causal else model(bx, mask)
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
            print(json.dumps({'causal': causal, 'fit_seed': seed, **record}), flush=True)
        if record['validation_nll'] < best-1e-5:
            best, best_epoch, state = record['validation_nll'], epoch, copy.deepcopy(model.state_dict())
        if epoch-best_epoch >= 5:
            break
    model.load_state_dict(state)
    return model, mean, scale, history, best_epoch








def uncertainty_discount(logpmfs):
    probabilities = np.exp(logpmfs)
    mean = probabilities.mean(0)
    entropy_mean = -(mean*np.log(np.maximum(mean,1e-30))).sum(-1)
    mean_entropy = -(probabilities*np.asarray(logpmfs)).sum(-1).mean(0)
    mi = np.maximum(0., entropy_mean-mean_entropy)/np.log(3)
    return np.log(np.maximum(mean,1e-30)), 1/(1+5*mi)




def save_fit(path, model, mean, scale, history, best_epoch):
    torch.save({'state_dict': {k:v.cpu() for k,v in model.state_dict().items()},
                'mean': torch.tensor(mean), 'scale': torch.tensor(scale),
                'causal': isinstance(model, CausalCountTCN)}, path)
    return {'best_epoch': best_epoch, 'history': history}






def run_sequence(args, output):
    obs, labels, ids = replay_observations(args.benchmark,args.epoch,2,args.seed)
    parts = record_partitions(labels,ids)
    x, counts, _ = observable_inputs(obs,2,False)
    model, mean, scale, old = restored_baseline(args.benchmark,args.epoch,args.seed,args.device)
    if not np.array_equal(old['record_ids'], ids):
        raise ValueError('baseline records differ')
    base = predict(model,(x-mean)/scale,counts,obs.lengths,args.device)
    baseline = aggregate(base,counts,obs.lengths)
    if not np.allclose(baseline,old['original_global'],atol=.03,rtol=2e-5):
        raise ValueError('recomputed baseline differs from registered score')
    scores = {'baseline':old['original_global']}
    training = {}
    diagnostics = {'baseline_validation_nll':record_nll(base,counts,obs.lengths,parts['validation'])}
    causal, cm, cs, history, best = fit(x,counts,obs.lengths,parts,seed=args.seed,device=args.device,causal=True)
    cp = predict(causal,(x-cm)/cs,counts,obs.lengths,args.device)
    scores['sequence'] = aggregate(cp,counts,obs.lengths)
    training['sequence'] = save_fit(output/'sequence.pt',causal,cm,cs,history,best)
    diagnostics['sequence_validation_nll'] = record_nll(cp,counts,obs.lengths,parts['validation'])
    del cp, causal
    pmfs = [base]
    for number, offset in enumerate((101,202),1):
        extra, em, es, history, best = fit(x,counts,obs.lengths,parts,seed=args.seed+offset,device=args.device)
        pmfs.append(predict(extra,(x-em)/es,counts,obs.lengths,args.device))
        training[f'ensemble_{number}'] = save_fit(output/f'ensemble_{number}.pt',extra,em,es,history,best)
        del extra
    pooled, discount = uncertainty_discount(np.stack(pmfs))
    scores['ensemble'] = aggregate(pooled,counts,obs.lengths)
    scores['uncertainty'] = aggregate(pooled,counts,obs.lengths,discount=discount)
    diagnostics['ensemble_validation_nll'] = record_nll(pooled,counts,obs.lengths,parts['validation'])
    diagnostics['mean_discount'] = float(discount.mean())
    finish(args,output,scores,labels,ids,parts,training=training,diagnostics=diagnostics,
           protocol='historical B=2 fixed-candidate replay; original_global restored')


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








def allocation_scores(logweights, bits, lengths, *, seed):
    """Normalized per-token latent Bernoulli models; policy uses only pilot bits."""
    grid=np.r_[expit(np.linspace(-7,7,17)),1.]
    boosted=np.vstack([expit(np.linspace(-7,7,17)[:,None]+TILTS),np.ones((1,3))])
    offsets=np.r_[0,lengths.cumsum()]
    output={name:np.empty(len(lengths)) for name in ('allocation_uniform','allocation_entropy')}
    for i,(start,end) in enumerate(zip(offsets[:-1],offsets[1:])):
        weight=np.exp(logweights[start:end].astype(float));weight/=weight.sum(1,keepdims=True)
        pilot=bits[start:end,0].astype(float)
        post=weight*np.where(pilot[:,None]>0,grid,1-grid)
        post/=post.sum(1,keepdims=True)
        chance=(post*grid).sum(1)
        chance=np.clip(chance,1e-12,1-1e-12)
        entropy=-(chance*np.log(chance)+(1-chance)*np.log1p(-chance))
        count=(end-start)//2
        uniform=np.random.default_rng(np.random.SeedSequence([seed,i,817])).permutation(end-start)[:count]
        informed=np.argsort(-entropy,kind='stable')[:count]
        for name,selected in (('allocation_uniform',uniform),('allocation_entropy',informed)):
            k=np.ones(end-start,dtype=int);k[selected]+=1
            successes=pilot.copy();successes[selected]+=bits[start:end,1][selected]
            failures=k-successes
            with np.errstate(divide='ignore',invalid='ignore'):
                null=successes[:,None]*np.log(grid)+np.where(failures[:,None]==0,0.,failures[:,None]*np.log1p(-grid))
                alt=successes[:,None,None]*np.log(boosted)+np.where(failures[:,None,None]==0,0.,failures[:,None,None]*np.log1p(-boosted))
            null=logsumexp(np.log(weight)+null,axis=1)
            alt=logsumexp(np.log(weight)[:,:,None]+alt,axis=1)
            output[name][i]=logsumexp((alt-null[:,None]).sum(0))-np.log(3)
    return output


def run_posthoc(args,output):
    obs,labels,ids=replay_observations(args.benchmark,args.epoch,2,args.seed)
    parts=record_partitions(labels,ids)
    x,counts,_=observable_inputs(obs,2,False)
    model,mean,scale,old=restored_baseline(args.benchmark,args.epoch,args.seed,args.device)
    pmf=predict(model,(x-mean)/scale,counts,obs.lengths,args.device)
    scores={'baseline':old['original_global'],'sparse':aggregate(pmf,counts,obs.lengths,sparse=True)}
    del pmf
    logweights=predict(model,(x-mean)/scale,counts,obs.lengths,args.device,weights=True)
    scores.update(allocation_scores(logweights,obs.bits[:,0],obs.lengths,seed=args.seed))
    calibration, full=calibration_analysis(scores['baseline'],labels,parts,obs.lengths,obs.logq[:,0])
    costs={'baseline_queries':(2*obs.lengths[parts['test']]).tolist(),
           'allocation_queries':(obs.lengths[parts['test']]+obs.lengths[parts['test']]//2).tolist()}
    finish(args,output,scores,labels,ids,parts,calibration_analysis=calibration,
           expanded_calibration_ids=ids[full].tolist(),costs=costs,
           protocol='sparse B=2; allocation comparisons both L+floor(L/2) fixed-q bits')




def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('phase',choices=('sequence','features','posthoc'))
    parser.add_argument('--benchmark',choices=('wikitection','newstection','arxivtection'),required=True)
    parser.add_argument('--epoch',type=int,choices=(1,3),required=True)
    parser.add_argument('--seed',type=int,default=20260914)
    parser.add_argument('--device',default='cpu')
    parser.add_argument('--threads',type=int,default=2)
    args=parser.parse_args();torch.set_num_threads(args.threads)
    output=OUTPUT/args.phase/f'{args.benchmark}_epoch{args.epoch}'/f'seed{args.seed}'
    output.mkdir(parents=True,exist_ok=True)
    if (output/'REPORT.json').exists():
        print('already complete',flush=True);return
    {'sequence':run_sequence,'features':run_features,'posthoc':run_posthoc}[args.phase](args,output)

if __name__=='__main__':
    main()
