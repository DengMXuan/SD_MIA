"""Summarize all registered priority ablations without member-based selection."""
import argparse
import json
import numpy as np
from .difficulty_accept_only import (OUTPUT, calibration_analysis)
from .replay_cache import (load_replay_data)
from .audit_runtime import (_paths)
from .summarize_direction_validation import (summarize, render)
from .audit_runtime import (_write_json)


def calibration_intervals(paths, repeats):
    """Paired test-record uncertainty; fitted models/calibration pool are fixed."""
    runs=[]; cached_condition=None; data=None
    for path in sorted(paths):
        report=json.loads(path.read_text())
        condition=(report['benchmark'],report['epoch'])
        if condition!=cached_condition:
            data=load_replay_data(*_paths(*condition));cached_condition=condition
        with np.load(path.parent/'scores.npz',allow_pickle=False) as f:archive=dict(f)
        parts={k:archive[k] for k in ('train','validation','reference','calibration','test')}
        _,_,values,_=calibration_analysis(archive['baseline'],archive['labels'],parts,data.lengths,data.logq0,return_pvalues=True)
        test=parts['test'];labels=archive['labels'][test]
        cohort=(tuple(archive['record_ids'][test]),tuple(labels))
        runs.append((cohort,labels,values))
    if not runs:return {}
    methods=('pooled_1200','mondrian_length','mondrian_difficulty')
    rng=np.random.default_rng(20260918)
    bootstrap={method:[] for method in methods};observed={method:[] for method in methods}
    for _,labels,values in runs:
        for method in methods:
            diff=(values[method]<=.01).astype(float)-(values['pooled_200']<=.01)
            observed[method].append([diff[labels==1].mean(),diff[labels==0].mean()])
    for _ in range(repeats):
        draws={};values_by_method={method:[] for method in methods}
        for cohort,labels,values in runs:
            if cohort not in draws:
                draws[cohort]=[rng.choice(np.flatnonzero(labels==label),size=int((labels==label).sum()),replace=True) for label in (1,0)]
            member,nonmember=draws[cohort]
            for method in methods:
                diff=(values[method]<=.01).astype(float)-(values['pooled_200']<=.01)
                values_by_method[method].append([diff[member].mean(),diff[nonmember].mean()])
        for method in methods:bootstrap[method].append(np.mean(values_by_method[method],axis=0))
    return {method:{metric:{'delta':float(np.mean(observed[method],axis=0)[j]),
                           'ci95':np.quantile(np.array(bootstrap[method])[:,j],[.025,.975]).tolist()}
                    for j,metric in enumerate(('tpr','fpr'))} for method in methods}


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--allow-incomplete',action='store_true')
    p.add_argument('--bootstrap-repeats',type=int,default=500)
    a=p.parse_args()
    comparisons={
        'sequence':[('sequence','baseline'),('ensemble','baseline'),('uncertainty','baseline'),('uncertainty','ensemble')],
        'features':[('difficulty','baseline')],
        'posthoc':[('sparse','baseline'),('allocation_entropy','allocation_uniform')],
    }
    phases={};complete=True;raw={};counts={}
    for phase,pairs in comparisons.items():
        paths=sorted((OUTPUT/phase).glob('*/seed*/REPORT.json'))
        counts[phase]=len(paths)
        if len(paths)!=18:
            complete=False
            if not a.allow_incomplete:raise ValueError(f'{phase}: expected 18 reports, found {len(paths)}')
        raw[phase]=[json.loads(path.read_text()) for path in paths]
        if paths:phases[phase]=summarize(paths,pairs,repeats=a.bootstrap_repeats)
    diagnostics={}
    for phase in ('sequence','features'):
        if raw[phase]:
            diagnostics[phase]={k:float(np.mean([r['diagnostics'][k] for r in raw[phase]])) for k in raw[phase][0]['diagnostics']}
    calibration={};by_condition={};group_results={}
    if raw['posthoc']:
        for method in raw['posthoc'][0]['calibration_analysis']:
            calibration[method]={}
            for level in ('.01','.05','.1'):
                key=str(float(level))
                calibration[method][key]={metric:float(np.mean([r['calibration_analysis'][method][key][metric] for r in raw['posthoc']])) for metric in ('tpr','actual_fpr')}
            by_condition[method]={}
            for condition in sorted({r['benchmark']+str(r['epoch']) for r in raw['posthoc']}):
                selected=[r for r in raw['posthoc'] if r['benchmark']+str(r['epoch'])==condition]
                by_condition[method][condition]={metric:float(np.mean([r['calibration_analysis'][method]['0.01'][metric] for r in selected])) for metric in ('tpr','actual_fpr')}
            group_results[method]={}
            for name in ('length','difficulty'):
                for group in ('0','1'):
                    values=[r['calibration_analysis'][method]['0.01']['groups'][name][group]['fpr'] for r in raw['posthoc']]
                    group_results[method][name+'_'+group]=float(np.mean([v for v in values if v is not None]))
    intervals=calibration_intervals((OUTPUT/'posthoc').glob('*/seed*/REPORT.json'),a.bootstrap_repeats)
    report={'complete':complete,'counts':counts,'phases':phases,'nonmember_diagnostics':diagnostics,
            'calibration':calibration,'calibration_per_condition':by_condition,'conditional_fpr_1pct':group_results,
            'calibration_paired_intervals_1pct':intervals,
            'warning':'exploratory fixed-model record bootstrap, no multiple comparison adjustment; feature q protocol differs from historical q; compare within phases'}
    _write_json(OUTPUT/'PRIORITY_REPORT.json',report)
    lines=['# Priority-ordered accept-only optimization validation','',f"Status: {'COMPLETE' if complete else 'PARTIAL'}.",'',
           'All language-model weights frozen. Real nonmembers only train/select/calibrate detectors.',
           'Whole-document B=2, except allocation: both policies spend L+floor(L/2) decisions.',
           'Feature experiment uses its own matched q-only baseline because its feature q differs from historical replay q.',
           'All comparisons are exploratory; paired record intervals condition on fitted models and reuse draws across seeds/checkpoints.',
           'Calibration sizes/partitions and all alternatives were fixed before viewing these scores. Test members never select a variant.','']
    for name,phase in phases.items():lines+=render(name,phase)+['']
    lines+=['## Nonmember validation diagnostics','','```json',json.dumps(diagnostics,indent=2),'```','',
            '## Calibration: frozen original_global score','','| Method | TPR at nominal 1% | Actual FPR | TPR at nominal 5% | Actual FPR |',
            '|---|---:|---:|---:|---:|']
    for method,values in calibration.items():
        lines.append(f"| {method} | {values['0.01']['tpr']:.4f} | {values['0.01']['actual_fpr']:.4f} | {values['0.05']['tpr']:.4f} | {values['0.05']['actual_fpr']:.4f} |")
    lines+=['','Paired changes vs pooled_200 at nominal 1% (fixed fitted models and calibration pool):','',
            '| Method | Delta TPR [95% CI] | Delta FPR [95% CI] |','|---|---:|---:|']
    for method,values in intervals.items():
        cells=[f"{values[key]['delta']:+.4f} [{values[key]['ci95'][0]:+.4f}, {values[key]['ci95'][1]:+.4f}]" for key in ('tpr','fpr')]
        lines.append(f"| {method} | "+' | '.join(cells)+' |')
    lines+=['','Calibration changes thresholds, not the original ranking AUC. Enlarged pool has 1200 nonmembers disjoint from 320 train, 80 validation and 800 test records.','',
            '### Conditional FPR at nominal 1%','','| Method | Short group | Long group | Low mean-logq group | High mean-logq group |',
            '|---|---:|---:|---:|---:|']
    for method,values in group_results.items():lines.append(f"| {method} | "+' | '.join(f'{values[k]:.4f}' for k in ('length_0','length_1','difficulty_0','difficulty_1'))+' |')
    lines+=['','### Per-condition actual FPR at nominal 1%','','| Method | '+' | '.join(next(iter(by_condition.values()),{}))+' |','|---|'+'---:|'*len(next(iter(by_condition.values()),{}))]
    for method,values in by_condition.items():lines.append(f"| {method} | "+' | '.join(f"{v['actual_fpr']:.4f}" for v in values.values())+' |')
    lines+=['','Group boundaries use reference nonmembers only; length ties can make group sizes unequal. Mondrian validity requires exchangeability within each group; no exact FPR claim on this fixed test set.','',
            'Query accounting: every B=2 fit uses the same 400 reference nonmember transcripts; ensemble initializations reuse those observations. Calibration expansion additionally queries 1000 nonmembers. Difficulty extraction uses only the frozen local draft. Allocation pilot/second-query scores use the same latent-mixture model for both policies.','']
    (OUTPUT/'PRIORITY_REPORT.md').write_text('\n'.join(lines)+'\n')
    print(OUTPUT/'PRIORITY_REPORT.md')

if __name__=='__main__':main()
