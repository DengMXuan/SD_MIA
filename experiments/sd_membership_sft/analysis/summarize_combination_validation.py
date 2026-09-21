"""Factorial ranking, operating-point, interaction and paired uncertainty report."""
from __future__ import annotations
import argparse
import json
import numpy as np
from experiments.sd_membership_sft.methods.combined_accept_only import (OUTPUT, CONDITIONS, SCORES, CALIBRATIONS, PRIMARY, BASELINE)
from experiments.sd_membership_sft.analysis.summarize_direction_validation import (summarize, render)
from experiments.sd_membership_sft.analysis.analyze_conditional_accept_only import (ranking)
from experiments.sd_membership_sft.core.audit_runtime import (_write_json)

RANK_PAIRS=(('q_sparse','q_global'),('difficulty_global','q_global'),
            ('difficulty_sparse','q_global'),('difficulty_sparse','q_sparse'),
            ('difficulty_sparse','difficulty_global'))
DECISION_PAIRS=((PRIMARY,BASELINE),(PRIMARY,'q_sparse__grouped1200'),
                (PRIMARY,'difficulty_global__grouped1200'),(PRIMARY,'difficulty_sparse__pooled200'),
                (PRIMARY,'difficulty_sparse__pooled1200'),
                ('difficulty_sparse__pooled1200','difficulty_sparse__pooled200'))


def interaction_value(values):
    return values['difficulty_sparse']-values['q_sparse']-values['difficulty_global']+values['q_global']


def paired_intervals(archives, operating, reports, repeats):
    cohorts=[];conditions={}
    for i,(archive,report) in enumerate(zip(archives,reports)):
        test=archive['test']
        cohorts.append((tuple(archive['record_ids'][test]),tuple(archive['labels'][test])))
        conditions.setdefault((report['benchmark'],report['epoch']),[]).append(i)
    def macro(values):return np.mean([np.mean([values[i] for i in indices],axis=0) for indices in conditions.values()],axis=0)
    rng=np.random.default_rng(20260919)
    interaction=[];delta={pair:[] for pair in DECISION_PAIRS}
    observed_interaction=[];observed_delta={pair:[] for pair in DECISION_PAIRS}
    for index in range(repeats+1):
        draws={};local_interaction=[];local_delta={pair:[] for pair in DECISION_PAIRS}
        for i,archive in enumerate(archives):
            test=archive['test'];labels=archive['labels'][test]
            m,n=np.flatnonzero(labels==1),np.flatnonzero(labels==0)
            if index:
                if cohorts[i] not in draws:draws[cohorts[i]]=(rng.choice(m,len(m),replace=True),rng.choice(n,len(n),replace=True))
                m,n=draws[cohorts[i]]
            ranks={name:ranking(archive[name],test[m],test[n]) for name in SCORES}
            local_interaction.append(interaction_value(ranks))
            for pair in DECISION_PAIRS:
                diff=(operating[i][pair[0]]<=.01).astype(float)-(operating[i][pair[1]]<=.01)
                local_delta[pair].append([diff[m].mean(),diff[n].mean()])
        if not index:
            observed_interaction=macro(local_interaction)
            observed_delta={pair:macro(values) for pair,values in local_delta.items()}
        else:
            interaction.append(macro(local_interaction))
            for pair,values in local_delta.items():delta[pair].append(macro(values))
    def packed(point,draws,names):
        draws=np.asarray(draws)
        return {key:{'delta':float(point[j]),'ci95':np.quantile(draws[:,j],[.025,.975]).tolist()} for j,key in enumerate(names)}
    return (packed(observed_interaction,interaction,('auc','pauc')),
            {a+' - '+b:packed(observed_delta[(a,b)],values,('tpr','fpr')) for (a,b),values in delta.items()})


def summarize_matrix(repeats=500, allow_incomplete=False):
    paths=sorted(OUTPUT.glob('*/seed*/REPORT.json'))
    reports=[json.loads(p.read_text()) for p in paths]
    keys={(r['benchmark'],r['epoch'],r['seed']) for r in reports}
    complete=keys==set(CONDITIONS) and len(reports)==18
    if not complete and not allow_incomplete:raise ValueError('expected all six conditions x three seeds, with no duplicates')
    if not paths:raise ValueError('no completed combination experiments')
    archives=[];operating=[]
    for path,report in zip(paths,reports):
        with np.load(path.parent/'scores.npz',allow_pickle=False) as f:archive=dict(f)
        with np.load(path.parent/'pvalues.npz',allow_pickle=False) as f:values=dict(f)
        test=archive['test']
        for key in ('labels','record_ids','groups'):
            if not np.array_equal(archive[key][test],values[key]):raise ValueError('operating-point record mismatch')
        if set(values)-{'labels','record_ids','groups'}!={s+'__'+c for s in SCORES for c in CALIBRATIONS}:
            raise ValueError('incomplete factorial configuration set')
        archives.append(archive);operating.append(values)
    ranking_report=summarize(paths,RANK_PAIRS,repeats=repeats)
    conditions={}
    for i,r in enumerate(reports):conditions.setdefault(r['benchmark']+str(r['epoch']),[]).append(i)
    def macro(values):return np.mean([np.mean([values[i] for i in ix],axis=0) for ix in conditions.values()],axis=0)
    decisions={};per_condition={};group_fpr={}
    for name in reports[0]['decisions']:
        decisions[name]={}
        for level in ('0.01','0.05','0.1'):
            vals=[[r['decisions'][name][level][k] for k in ('tpr','actual_fpr')] for r in reports]
            decisions[name][level]=dict(zip(('tpr','actual_fpr'),macro(vals).tolist()))
        per_condition[name]={condition:{key:float(np.mean([reports[i]['decisions'][name]['0.01'][key] for i in ix])) for key in ('tpr','actual_fpr')} for condition,ix in conditions.items()}
        group_fpr[name]={}
        for group in ('0','1'):
            vals=[r['decisions'][name]['0.01']['groups'][group]['fpr'] for r in reports]
            group_fpr[name][group]=float(np.mean([v for v in vals if v is not None]))
    interaction,decision_ci=paired_intervals(archives,operating,reports,repeats)
    result={'complete':complete,'runs':len(reports),'configurations_per_run':12,'evaluated_cells':len(reports)*12,
            'primary_candidate':PRIMARY,'ranking':ranking_report,'decisions':decisions,'per_condition_decisions':per_condition,
            'group_fpr_1pct':group_fpr,'feature_sparse_interaction':interaction,'decision_comparisons_1pct':decision_ci,
            'new_detector_fits':0,'language_models_frozen':True,'bootstrap_repeats':repeats,
            'interval_scope':'paired records, shared across seeds/checkpoints, fitted models and calibration pools fixed; exploratory unadjusted intervals'}
    _write_json(OUTPUT/'COMBINATION_REPORT.json',result)
    lines=['# Feature × sparsity × calibration combination validation','',f"Status: {'COMPLETE' if complete else 'PARTIAL'}; {len(reports)} runs, {len(reports)*12} factorial cells.",'',
        'Language models AND small detectors remain frozen. Only trusted nonmembers were used in the saved fits and all calibration.',
        'All twelve cells share feature-consistent q, B=2 bits, candidate scope, splits and query budget per test record.',
        'Raw-score AUC/pAUC describe four score variants; the three calibration variants change operating thresholds, not these raw rankings.',
        'Intervals reuse paired test-record draws across seeds/checkpoints and condition on fitted models and calibration pools. They are exploratory and unadjusted for multiple comparisons.','']
    lines+=render('Raw ranking (pooled200 operating points shown here)',ranking_report)+['']
    lines+=['## All twelve calibrated configurations','','| Inputs / Score / Calibration | AUC | pAUC@10% | TPR@nominal 1% | Actual FPR | TPR@nominal 5% | Actual FPR |','|---|---:|---:|---:|---:|---:|---:|']
    for name,values in decisions.items():
        score=name.split('__')[0];rank=ranking_report['means'][score];one=values['0.01'];five=values['0.05']
        lines.append(f"| {name} | {rank['auc']:.4f} | {rank['pauc']:.4f} | {one['tpr']:.4f} | {one['actual_fpr']:.4f} | {five['tpr']:.4f} | {five['actual_fpr']:.4f} |")
    lines+=['','## Feature × sparsity interaction','','(difficulty_sparse − q_sparse) − (difficulty_global − q_global). Positive denotes super-additivity on the stated ranking metric, not causal proof.','',
            '| Metric | Interaction [95% CI] |','|---|---:|']
    for name,v in interaction.items():lines.append(f"| {name} | {v['delta']:+.5f} [{v['ci95'][0]:+.5f}, {v['ci95'][1]:+.5f}] |")
    lines+=['','## Primary candidate and component removals at nominal 1%','','| Comparison | Delta TPR [95% CI] | Delta FPR [95% CI] |','|---|---:|---:|']
    for name,values in decision_ci.items():
        cells=[f"{values[k]['delta']:+.4f} [{values[k]['ci95'][0]:+.4f}, {values[k]['ci95'][1]:+.4f}]" for k in ('tpr','fpr')]
        lines.append(f"| {name} | "+' | '.join(cells)+' |')
    lines+=['','## Per-condition TPR / actual FPR at nominal 1%','','| Configuration | '+' | '.join(conditions)+' |','|---|'+'---:|'*len(conditions)]
    for name,values in per_condition.items():lines.append(f"| {name} | "+' | '.join(f"{v['tpr']:.4f} / {v['actual_fpr']:.4f}" for v in values.values())+' |')
    lines+=['','## Conditional FPR at nominal 1%','','| Configuration | Low mean-logq group | High mean-logq group |','|---|---:|---:|']
    for name,v in group_fpr.items():lines.append(f"| {name} | {v['0']:.4f} | {v['1']:.4f} |")
    lines+=['','## Costs and limits','','No new model fitting or target-model forward passes: saved detectors and verifier caches were reused.',
            'Every cell uses 2L decisions for a test record of L candidate tokens. Enlarging calibration from 200 to 1200 requires 1000 additional nonmember transcripts in a live setting; grouping itself adds no queries.',
            'This is fixed-candidate offline replay, not natural serial generation or a deployed API. Group boundaries use the reference-NM median only; group-wise exchangeability is needed for grouped conformal validity. Nominal 1% is not a guarantee that each observed condition/group attains 1%.','']
    (OUTPUT/'COMBINATION_REPORT.md').write_text('\n'.join(lines)+'\n')
    print(OUTPUT/'COMBINATION_REPORT.md')
    return result


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--bootstrap-repeats',type=int,default=500)
    p.add_argument('--allow-incomplete',action='store_true')
    a=p.parse_args();summarize_matrix(a.bootstrap_repeats,a.allow_incomplete)

if __name__=='__main__':main()
