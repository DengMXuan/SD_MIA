"""Evaluate a frozen q-reference choice; this module never selects a method."""
from __future__ import annotations

import argparse
import hashlib
from pathlib import Path

import joblib
import numpy as np
import torch

from experiments.paths import AUDITS, ROOT
from experiments.pretraining.frozen_audit import read_json, require
from experiments.pretraining.q_feature_exploration import (
    DEFAULT_OUTPUT, load_task, legacy_scores, fit_positions, snapshot_sources,
)
from experiments.shared.audit.artifacts import digest, read_result, checkpoint_inventory
from experiments.shared.audit.metrics import metrics, roc_operating_point
from experiments.shared.core.audit_metrics import rank_auc
from experiments.shared.core.audit_runtime import _write_json
from experiments.shared.core.deployment_archive import sha256_file
from experiments.shared.evaluation.provenance import import_closure
from experiments.shared.methods.preserved_accept_only import OLD, PRIMARY as PREVIOUS
from experiments.shared.methods.q_feature_accept_only import (
    position_features, fit_document_reference, standardize, round1_fusions,
    q_context, round2_fusions, round3_fusions,
)
from experiments.shared.protocols.protocol_archive import atomic_npz

BASELINES = ('loss', 'min_k_prob', 'min_k_pp', 'sead', 'petal', 'recall', 'icp_mia')
_CHECKED_FILES, _CHECKED_MODELS = set(), set()


def checked_baselines(task, saved):
    group, source, seed = task['group'], task['source'], task['seed']
    if group == 'mimir13':
        folder = AUDITS/'pythia_mimir_baselines7_v1/tasks'/source/f'seed{seed}'
    else:
        folder = AUDITS/'paper_positive_controls_v1/tasks'/('mimir02' if group == 'mimir7' else 'wikimia')/source/f'seed{seed}'/'baselines7'
    partitions = read_json(task['inputs']['partitions']['path'])
    require(read_json(folder/'PARTITIONS.json') == partitions, 'baseline partitions changed')
    request = read_json(folder/'BASELINE_REQUEST.json')
    manifest = read_json(task['inputs']['manifest']['path'])
    require(request['partitions_sha256'] == digest(partitions)
            and request['condition']['models'] == manifest['models']
            and request['condition']['benchmark'] == task['benchmark']
            and request['settings']['audit_seed'] == seed
            and request['evaluation_context']['data_manifest'] == task['inputs']['manifest']['path']
            and request['evaluation_context']['token_contract'] == manifest['token_contract']
            and request['methods'] == list(BASELINES), 'baseline request mismatch')
    historical = read_json(AUDITS/'pythia_preserved_v1/baseline7_comparison_seed1919/COMPARISON.json')['historical_launcher']
    scores, provenance = {}, []
    for method in BASELINES:
        path = folder/method
        report = read_result(path, digest({'task':request,'method':method}), request['sources_sha256'], check_sources=False)
        for item in report['sources']['files']:
            key = item['path'], item['sha256']
            if key not in _CHECKED_FILES:
                actual = sha256_file(Path(item['path']))
                require(actual == item['sha256'] or (
                    Path(item['path']) == ROOT/historical['file'] and actual == historical['current_sha256']
                    and item['sha256'] == historical['old_sha256']), 'unknown baseline source drift: '+item['path'])
                _CHECKED_FILES.add(key)
        for item in report['sources']['checkpoints']:
            key = item['path'], digest(item['inventory'])
            if key not in _CHECKED_MODELS:
                require(checkpoint_inventory(Path(item['path'])) == item['inventory'], 'baseline model inventory changed')
                _CHECKED_MODELS.add(key)
        with np.load(path/'scores.npz', allow_pickle=False) as data:
            for key in ('record_ids', 'labels', 'calibration', 'test'):
                require(np.array_equal(data[key], saved[key]), f'baseline alignment: {method}/{key}')
            scores[method] = data['scores']
        provenance.append({'method':method, 'report':str(path/'REPORT.json'),
                           'report_sha256':sha256_file(path/'REPORT.json'), 'scores_sha256':report['scores_sha256']})
    return scores, provenance


def diagnostic_positions(data, model, vocab_size, seed):
    x, lengths, counts = data['features'], data['lengths'], data['counts']/2
    context = q_context(x, lengths)
    probability = model['model'].predict_proba(context[:, :model['columns']])
    mu = probability @ np.array([0., .5, 1.])
    result = {name:[] for name in ('random_40', 'nm_high_accept_40', 'qcenter_high_40')}
    offsets = np.r_[0, lengths.cumsum()]
    for i,(left,right) in enumerate(zip(offsets[:-1], offsets[1:])):
        a, expected, q = counts[left:right], mu[left:right], x[left:right]
        n = max(1,int(np.ceil(len(a)*.4)))
        random_seed = int.from_bytes(hashlib.sha256(f'{seed}:{data["record_ids"][i]}:position_control'.encode()).digest()[:8], 'little')
        chosen = np.random.default_rng(random_seed).choice(len(a), n, replace=False)
        result['random_40'].append(float(a[chosen].mean()))
        chosen = np.argsort(-expected, kind='stable')[:n]
        result['nm_high_accept_40'].append(float(a[chosen].mean()))
        entropy = q[:,1]*np.log(vocab_size)
        centered = (q[:,0]+entropy)/np.sqrt(np.maximum(entropy,.25))
        chosen = np.argsort(-centered, kind='stable')[:n]
        result['qcenter_high_40'].append(float(a[chosen].mean()))
    return {k:np.asarray(v) for k,v in result.items()}


def measure(scores, saved, test, primary, seed, bootstrap):
    results = {name:metrics(v,saved['labels'],saved['calibration'],test,seed=seed,bootstrap=0)
               for name,v in scores.items()}
    y = saved['labels']
    pos,neg = test[y[test]==1],test[y[test]==0]
    rng = np.random.default_rng(seed)
    draws = {name:np.empty((bootstrap,2)) for name in scores}
    labels = np.r_[np.ones(len(pos),int),np.zeros(len(neg),int)]
    for i in range(bootstrap):
        a,b = rng.choice(pos,len(pos),replace=True),rng.choice(neg,len(neg),replace=True)
        for name,v in scores.items():
            draws[name][i] = rank_auc(v[a],v[b]),roc_operating_point(v[np.r_[a,b]],labels,.01)[0]
    pairs = []
    for name in scores:
        if bootstrap:
            results[name]['auc_ci95'] = np.quantile(draws[name][:,0],[.025,.975]).tolist()
        if name != primary:
            row = {'reference':name, 'delta_auc':results[primary]['auc']-results[name]['auc'],
                   'delta_tpr1':results[primary]['roc_tpr_at_1pct_fpr']-results[name]['roc_tpr_at_1pct_fpr']}
            if bootstrap:
                ci = np.quantile(draws[primary]-draws[name],[.025,.975],axis=0)
                row.update(auc_ci95=ci[:,0].tolist(), tpr1_ci95=ci[:,1].tolist())
            pairs.append(row)
    return results,pairs


def run(output, bootstrap):
    selection_path, split_path = output/'FROZEN_SELECTION.json', output/'SPLIT_PLAN.json'
    frozen,plan = read_json(selection_path),read_json(split_path)
    require(sha256_file(split_path) == frozen['split_sha256'], 'split changed after selection')
    for path,expected in frozen['implementation'].items():
        require(sha256_file(ROOT/path) == expected, 'scoring implementation changed after selection: '+path)
    primary = frozen['selected']['method']
    methods = list(dict.fromkeys([primary, OLD, PREVIOUS, 'accept_rate', 'q_only', 'accept_q_fusion',
         frozen['round_winners']['round1']['method'], frozen['round_winners']['round2']['method'],
         'qref_multiscale_only', 'qref_multiscale_linear_q50_c25', 'qref_qhard_negative_q50_c25',
         'accept_lowq_40', 'accept_qcenter_60', 'accept_qhard_10', 'nm_low_accept_40',
         'random_40', 'nm_high_accept_40', 'qcenter_high_40']))
    root = output/'confirmation'
    root.mkdir(parents=True, exist_ok=True)
    implementation = {str(p.relative_to(ROOT)):sha256_file(p) for p in import_closure(
        ROOT,['experiments.pretraining.q_feature_confirmation'])}
    request = {'selection_sha256':sha256_file(selection_path), 'split_sha256':sha256_file(split_path),
               'primary':primary, 'methods':methods, 'baselines':list(BASELINES), 'bootstrap':bootstrap,
               'implementation':implementation, 'confirmation_policy':'no selection based on these results',
               'stability':'seeds share text; separate per-seed estimates, no independent-data claim'}
    if (root/'PLAN.json').exists():
        require(read_json(root/'PLAN.json') == request, 'confirmation configuration changed; use a new batch')
    else:
        _write_json(root/'PLAN.json',request)
        snapshot_sources(root,implementation)
    reports = []
    for task in plan['tasks']:
        folder = root/task['key']
        folder.mkdir(parents=True,exist_ok=True)
        if (folder/'_COMPLETE.json').exists():
            complete = read_json(folder/'_COMPLETE.json')
            for filename,h in complete.items():
                require(sha256_file(folder/filename)==h,'completed confirmation changed')
            reports.append(read_json(folder/'REPORT.json'))
            continue
        paths,data,parts,saved,selected = load_task(task)
        if task['seed'] == 1919:
            previous = output/'round3'/task['key']
            require(sha256_file(previous/'scores.npz') == frozen['score_archives'][task['key']], 'frozen scores changed')
            with np.load(previous/'scores.npz',allow_pickle=False) as a:
                require(np.array_equal(data['record_ids'],a['record_ids']),'frozen IDs changed')
                all_scores = {k:a[k] for k in a.files if k!='record_ids'}
            previous2 = output/'round2'/task['key']
            r2 = read_json(previous2/'REPORT.json')
            require(sha256_file(previous2/'nonmember_model.joblib') == r2['model_sha256'],'frozen model changed')
            model = joblib.load(previous2/'nonmember_model.joblib')
            fit_report = r2['fit']
            normalization = {'q':r2['normalization'],'learned':r2['learned_normalization']}
        else:
            all_scores,pmf = legacy_scores(paths,data,parts,saved,selected)
            qfeatures = position_features(data['features'],data['counts'],data['lengths'],vocab_size=plan['vocab_size'])
            qnorm = fit_document_reference({k:v[parts['reference']] for k,v in qfeatures.items()})
            z = standardize(qfeatures,qnorm)
            model,fit_report,learned,learned_norm = fit_positions(data,parts,pmf,seed=task['seed'])
            learned_z = standardize(learned,learned_norm)
            all_scores.update(qfeatures)
            all_scores.update(learned)
            all_scores.update(round1_fusions(z))
            all_scores.update(round2_fusions(z,learned_z))
            all_scores.update(round3_fusions(z,learned_z)[0])
            normalization = {'q':qnorm,'learned':learned_norm}
        joblib.dump(model,folder/'nonmember_model.joblib')
        all_scores.update(diagnostic_positions(data,model,plan['vocab_size'],task['seed']))
        scores = {name:all_scores[name][selected] for name in methods}
        baseline_scores,baseline_provenance = checked_baselines(task,saved)
        scores.update(baseline_scores)
        test = saved['test'][np.isin(saved['record_ids'][saved['test']],task['confirmation_ids'])]
        require(len(test)==len(task['confirmation_ids']) and not set(task['development_ids']) & set(task['confirmation_ids']),
                'invalid protected confirmation split')
        results,pairs = measure(scores,saved,test,primary,task['seed'],bootstrap)
        # Descriptive full-original-set values are calculated only after freezing;
        # they combine development and confirmation and are not validation results.
        descriptive = {name:metrics(v,saved['labels'],saved['calibration'],saved['test'],seed=task['seed'],bootstrap=0)
                       for name,v in scores.items()}
        atomic_npz(folder/'scores.npz',dict(record_ids=saved['record_ids'],labels=saved['labels'],
                    calibration=saved['calibration'],test=test,original_test=saved['test'],**scores))
        report = {'task_key':task['key'],'group':task['group'],'source':task['source'],'seed':task['seed'],
                  'primary':primary,'confirmation_metrics':results,'paired_comparisons':pairs,
                  'full_set_descriptive_metrics':descriptive,'fit':fit_report,'normalization':normalization,
                  'baseline_provenance':baseline_provenance,'input_fingerprints':task['inputs'],
                  'selection_sha256':request['selection_sha256'],
                  'model_sha256':sha256_file(folder/'nonmember_model.joblib'),
                  'scores_sha256':sha256_file(folder/'scores.npz'),
                  'target_probability_access':False,'new_language_model_queries':0,
                  'original_feedback_budget_per_token':2,'position_selection_saves_queries':False}
        _write_json(folder/'REPORT.json',report)
        _write_json(folder/'_COMPLETE.json',{f:sha256_file(folder/f) for f in ('REPORT.json','scores.npz','nonmember_model.joblib')})
        reports.append(report)
        print(task['key'],'confirmation AUC',round(results[primary]['auc'],4),
              'previous',round(results[PREVIOUS]['auc'],4), 'full descriptive',round(descriptive[primary]['auc'],4),flush=True)
        _write_json(root/'SUMMARY.json',{'primary':primary,'conditions':reports,'completed':len(reports),'planned':len(plan['tasks'])})
    _write_json(root/'SUMMARY.json',{'primary':primary,'conditions':reports,'completed':len(reports),'planned':len(plan['tasks'])})


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command',choices=('run',))
    parser.add_argument('--output-root',type=Path,default=DEFAULT_OUTPUT)
    parser.add_argument('--bootstrap',type=int,default=1000)
    args = parser.parse_args(argv)
    require(args.bootstrap>=0,'invalid bootstrap count')
    torch.set_num_threads(1)
    run(args.output_root.resolve(),args.bootstrap)


if __name__ == '__main__':
    main()
