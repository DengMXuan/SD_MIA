"""Frozen-data, document-grouped development of q-only accept-feedback scorers."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import torch

from experiments.paths import AUDITS, ROOT
from experiments.pretraining.frozen_audit import identify_inputs, load_inputs, predict_frozen, read_json, require
from experiments.pretraining.preserved_evidence import tasks_for, completed_report
from experiments.shared.core.audit_metrics import rank_auc
from experiments.shared.core.audit_runtime import _write_json
from experiments.shared.core.deployment_archive import sha256_file
from experiments.shared.evaluation.provenance import import_closure
from experiments.shared.methods.preserved_accept_only import (
    OLD, PRIMARY as PREVIOUS, base_scores, fit_normalization, fuse_scores,
)
from experiments.shared.methods.q_feature_accept_only import (
    position_features, fit_document_reference, standardize, round1_fusions,
    q_context, fit_nonmember_counts, count_position_features, round2_fusions,
    round3_fusions,
)
from experiments.shared.protocols.protocol_archive import atomic_npz

CONDITIONS = [('mimir13', s) for s in ('github', 'arxiv', 'wikipedia_(en)')] + [
    ('mimir7', 'github'), ('mimir7', 'arxiv'), ('wikimia', 'length64'), ('wikimia', 'length128')]
DEFAULT_OUTPUT = AUDITS / 'pythia_q_exploration_v1'
SPLIT_SALT = 'q-reference-exploration-20261003'


def code_fingerprint():
    return {str(p.relative_to(ROOT)): sha256_file(p) for p in import_closure(
        ROOT, ['experiments.pretraining.q_feature_exploration'])}


def prepare(output):
    """Freeze all masks from content alone, without calculating new test metrics."""
    tasks, records, parents = [], {}, {}
    def find(key):
        parents.setdefault(key, key)
        while parents[key] != key:
            parents[key] = parents[parents[key]]
            key = parents[key]
        return key
    def union(a, b):
        a, b = find(a), find(b)
        if a != b:
            parents[max(a, b)] = min(a, b)
    for group, source in CONDITIONS:
        for seed in (1919, 1949, 1978):
            original_root = AUDITS / ('pythia_mimir_v1' if group == 'mimir13' else 'paper_positive_controls_v1')
            task = tasks_for(group, [source], seed, original_root)[0]
            paths, task['inputs'], _ = identify_inputs(task['report'], task['benchmark'], seed)
            rows = [json.loads(line) for line in paths['records'].read_text().splitlines()]
            task['group'] = group
            task['key'] = f'{group}/{source}/seed{seed}'
            records[task['key']] = rows
            tasks.append(task)
            for row in rows:
                text_hash = hashlib.sha256(row['text'].encode()).hexdigest()
                signatures = ['tokens:'+row['token_hash'], 'text:'+text_hash]
                # Conservatively keep identical long prefixes together across ngram/length conditions.
                if len(row['token_ids']) >= 64:
                    signatures.append('prefix64:'+hashlib.sha256(np.asarray(row['token_ids'][:64], '<i8').tobytes()).hexdigest())
                for value in signatures:
                    union(signatures[0], value)
    group_ids = {}
    for task in tasks:
        rows = records[task['key']]
        parts = read_json(task['inputs']['partitions']['path'])['record_ids']
        ids = {}
        for row in rows:
            component = find('tokens:'+row['token_hash'])
            value = hashlib.sha256((SPLIT_SALT+component).encode()).hexdigest()
            ids[row['record_id']] = value
            group_ids[row['token_hash']] = value
        task['document_groups'] = ids
        task['development_ids'] = [i for i in parts['test'] if int(ids[i][:8], 16)/2**32 < .6]
        task['confirmation_ids'] = [i for i in parts['test'] if i not in set(task['development_ids'])]
        task['split_counts'] = {role: {str(y): sum(row['label']==y and row['record_id'] in set(task[role+'_ids'])
                                                   for row in rows) for y in (0, 1)}
                                for role in ('development', 'confirmation')}
    from experiments.pretraining.data import load_tokenizer, TARGET
    vocab_size = len(load_tokenizer(TARGET))
    plan = {'schema': 'q_reference_development_v1', 'tasks': tasks, 'vocab_size': vocab_size,
            'split_salt': SPLIT_SALT, 'development_fraction': .6,
            'authorization': 'User permits development-label configuration comparison; fitting only auxiliary nonmembers. Exact p forbidden.',
            'scope': 'Confirmation masks protected during these iterations, but all original datasets have historical exposure; exploratory, not pristine independent validation.',
            'grouping': 'union of token hash, full text hash, identical first 64 tokens; group split shared across all conditions and seeds',
            'prepare_implementation': code_fingerprint()}
    output.mkdir(parents=True, exist_ok=True)
    destination = output/'SPLIT_PLAN.json'
    require(not destination.exists(), 'split already frozen; use existing plan or a new output batch')
    _write_json(destination, plan)
    for task in tasks:
        print(task['key'], task['split_counts'], flush=True)


def load_task(task):
    paths, actual, old = identify_inputs(task['report'], task['benchmark'], task['seed'])
    require(actual == task['inputs'], 'frozen task inputs changed')
    data, parts, saved, selected = load_inputs(paths, old)
    return paths, data, parts, saved, selected


def legacy_scores(paths, data, parts, saved, selected):
    pmf = predict_frozen(paths['detector'], data)
    base = base_scores(pmf, data['counts'], data['lengths'], data['features'][:, 0])
    require(np.array_equal(base[OLD][selected], saved['scores']), 'original score replay mismatch')
    normalization = fit_normalization({k: v[parts['reference']] for k, v in base.items()})
    return fuse_scores(base, normalization), pmf


def development_metrics(scores, data, task):
    mask = np.isin(data['record_ids'], task['development_ids'])
    y = data['labels']
    return {name: float(rank_auc(values[mask & (y == 1)], values[mask & (y == 0)]))
            for name, values in scores.items()}


def run_round1(output):
    plan = read_json(output/'SPLIT_PLAN.json')
    root = output/'round1'
    request = {'round': 1, 'split_sha256': sha256_file(output/'SPLIT_PLAN.json'),
               'implementation': code_fingerprint(), 'purpose': 'q features and fixed q-only position selection',
               'evaluation': 'development IDs of seed1919 only'}
    root.mkdir(parents=True, exist_ok=True)
    if (root/'PLAN.json').exists():
        require(read_json(root/'PLAN.json') == request, 'round1 implementation changed; use a new batch')
    else:
        _write_json(root/'PLAN.json', request)
    results = []
    for task in plan['tasks']:
        if task['seed'] != 1919:
            continue
        folder = root/task['key']
        folder.mkdir(parents=True, exist_ok=True)
        paths, data, parts, saved, selected = load_task(task)
        scores, _ = legacy_scores(paths, data, parts, saved, selected)
        features = position_features(data['features'], data['counts'], data['lengths'], vocab_size=plan['vocab_size'])
        norm = fit_document_reference({k: v[parts['reference']] for k, v in features.items()})
        z = standardize(features, norm)
        scores.update(features)
        scores.update(round1_fusions(z))
        dev = development_metrics(scores, data, task)
        atomic_npz(folder/'scores.npz', dict(record_ids=data['record_ids'], **scores))
        report = {'task_key': task['key'], 'development_auc': dev, 'normalization': norm,
                  'scores_sha256': sha256_file(folder/'scores.npz'), 'split_counts': task['split_counts'],
                  'language_model_queries': 0, 'target_probability_access': False,
                  'confirmation_metrics_computed': False}
        _write_json(folder/'REPORT.json', report)
        results.append(report)
        print(task['key'], 'previous', round(dev[PREVIOUS], 4), 'accept', round(dev['accept_mean'], 4),
              'best dev', sorted(dev.items(), key=lambda kv: -kv[1])[:5], flush=True)
    _write_json(root/'SUMMARY.json', {'conditions': results})
    print('Round 1 development complete:', root)


def subset_tokens(x, c, lengths, indices):
    offsets = np.r_[0, lengths.cumsum()]
    selected = np.concatenate([np.arange(offsets[i], offsets[i+1]) for i in indices])
    return x[selected], c[selected], lengths[indices]


def fit_positions(data, parts, pmf, *, seed):
    context = q_context(data['features'], data['lengths'])
    model, report = fit_nonmember_counts(
        *subset_tokens(context, data['counts'], data['lengths'], parts['train']),
        *subset_tokens(context, data['counts'], data['lengths'], parts['validation']), seed=seed)
    probability = model['model'].predict_proba(context[:, :model['columns']])
    offsets = np.r_[0, data['lengths'].cumsum()]
    nll = -pmf[np.arange(len(pmf)), data['counts']]
    report['legacy_tcn_validation_nll'] = float(np.mean([nll[offsets[i]:offsets[i+1]].mean()
                                                       for i in parts['validation']]))
    features = count_position_features(data['features'], data['counts'], data['lengths'], probability)
    # Learned channels are normalized on held-out validation nonmembers only.
    norm = fit_document_reference({k:v[parts['validation']] for k,v in features.items()})
    return model, report, features, norm


def snapshot_sources(folder, fingerprints):
    for source, expected in fingerprints.items():
        require(sha256_file(ROOT/source) == expected, 'source changed during run')
        path = folder/'source_snapshot'/source
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes((ROOT/source).read_bytes())


def run_round2(output):
    import joblib
    plan = read_json(output/'SPLIT_PLAN.json')
    root = output/'round2'
    root.mkdir(parents=True, exist_ok=True)
    request = {'round': 2, 'split_sha256': sha256_file(output/'SPLIT_PLAN.json'),
               'implementation': code_fingerprint(), 'purpose': 'nonmember q-to-count model and position information',
               'evaluation': 'development IDs of seed1919 only'}
    require(not (root/'PLAN.json').exists(), 'round2 already frozen; use a new output batch')
    _write_json(root/'PLAN.json', request)
    snapshot_sources(root, request['implementation'])
    results = []
    for task in plan['tasks']:
        if task['seed'] != 1919:
            continue
        folder = root/task['key']
        folder.mkdir(parents=True, exist_ok=True)
        paths, data, parts, saved, selected = load_task(task)
        scores, pmf = legacy_scores(paths, data, parts, saved, selected)
        qfeatures = position_features(data['features'], data['counts'], data['lengths'], vocab_size=plan['vocab_size'])
        qnorm = fit_document_reference({k:v[parts['reference']] for k,v in qfeatures.items()})
        z = standardize(qfeatures, qnorm)
        scores.update(qfeatures)
        scores.update(round1_fusions(z))
        first = output/'round1'/task['key']
        first_report = read_json(first/'REPORT.json')
        require(sha256_file(first/'scores.npz') == first_report['scores_sha256'], 'round1 scores corrupted')
        with np.load(first/'scores.npz', allow_pickle=False) as archive:
            for name,values in scores.items():
                require(np.array_equal(archive[name], values), 'round1 replay mismatch: '+name)
        model, fit_report, learned, learned_norm = fit_positions(data, parts, pmf, seed=task['seed'])
        scores.update(learned)
        scores.update(round2_fusions(z, standardize(learned, learned_norm)))
        joblib.dump(model, folder/'nonmember_model.joblib')
        dev = development_metrics(scores, data, task)
        atomic_npz(folder/'scores.npz', dict(record_ids=data['record_ids'], **scores))
        report = {'task_key': task['key'], 'development_auc': dev, 'normalization': qnorm,
                  'learned_normalization': learned_norm, 'learned_reference': 'validation_80_nonmembers',
                  'fit': fit_report, 'scores_sha256': sha256_file(folder/'scores.npz'),
                  'model_sha256': sha256_file(folder/'nonmember_model.joblib'),
                  'confirmation_metrics_computed': False, 'target_probability_access': False}
        _write_json(folder/'REPORT.json', report)
        results.append(report)
        print(task['key'], 'NLL', fit_report, 'best dev',
              sorted(dev.items(), key=lambda kv:-kv[1])[:5], flush=True)
    _write_json(root/'SUMMARY.json', {'conditions': results})
    print('Round 2 development complete:', root)


def choose_global(reports, *, max_condition_regression=.005):
    methods = list(reports[0]['development_auc'])
    baseline = np.array([r['development_auc'][PREVIOUS] for r in reports])
    rows = []
    for method in methods:
        values = np.array([r['development_auc'][method] for r in reports])
        rows.append({'method': method, 'mean_auc': float(values.mean()),
                     'worst_delta_vs_previous': float((values-baseline).min()),
                     'auc_by_condition': dict(zip((r['task_key'] for r in reports), values.tolist())),
                     'eligible': bool((values-baseline).min() >= -max_condition_regression)})
    ranked = sorted(rows, key=lambda r: (-r['mean_auc'], r['method']))
    selected = next(r for r in ranked if r['eligible'])
    return selected, ranked


def run_round3(output):
    plan = read_json(output/'SPLIT_PLAN.json')
    root = output/'round3'
    root.mkdir(parents=True, exist_ok=True)
    request = {'round': 3, 'split_sha256': sha256_file(output/'SPLIT_PLAN.json'),
               'implementation': code_fingerprint(), 'purpose': 'global position aggregation and nonlinear q supplementation',
               'evaluation': 'development IDs of seed1919 only',
               'selection': 'largest seven-condition macro development AUC; no condition more than 0.005 below previous; lexical tie break'}
    require(not (root/'PLAN.json').exists(), 'round3 already frozen; use a new output batch')
    _write_json(root/'PLAN.json', request)
    snapshot_sources(root, request['implementation'])
    results, recipes = [], {}
    for task in plan['tasks']:
        if task['seed'] != 1919:
            continue
        folder = root/task['key']
        folder.mkdir(parents=True, exist_ok=True)
        _, data, _, _, _ = load_task(task)
        previous = output/'round2'/task['key']
        report2 = read_json(previous/'REPORT.json')
        require(sha256_file(previous/'scores.npz') == report2['scores_sha256'], 'round2 score checksum mismatch')
        with np.load(previous/'scores.npz', allow_pickle=False) as a:
            require(np.array_equal(a['record_ids'], data['record_ids']), 'round2 record mismatch')
            scores = {k:a[k] for k in a.files if k!='record_ids'}
        scores3, recipes = round3_fusions(standardize(scores, report2['normalization']),
                                         standardize(scores, report2['learned_normalization']))
        scores.update(scores3)
        dev = development_metrics(scores, data, task)
        for k,v in report2['development_auc'].items():
            require(dev[k] == v, 'round2 metric replay mismatch')
        atomic_npz(folder/'scores.npz', dict(record_ids=data['record_ids'], **scores))
        report = {'task_key': task['key'], 'development_auc': dev,
                  'scores_sha256': sha256_file(folder/'scores.npz'),
                  'round2_report_sha256': sha256_file(previous/'REPORT.json'),
                  'confirmation_metrics_computed': False, 'target_probability_access': False}
        _write_json(folder/'REPORT.json', report)
        results.append(report)
    chosen, ranking = choose_global(results)
    round_winners = {}
    for number in (1,2):
        reports = read_json(output/f'round{number}/SUMMARY.json')['conditions']
        winner, _ = choose_global(reports)
        round_winners[f'round{number}'] = winner
    frozen = {'selected': chosen, 'selection_rule': request['selection'],
              'recipe': recipes.get(chosen['method']), 'round_winners': round_winners,
              'all_round3_recipes': recipes, 'ranking': ranking,
              'confirmation_seen': False, 'split_sha256': request['split_sha256'],
              'implementation': request['implementation'],
              'score_archives': {r['task_key']:r['scores_sha256'] for r in results}}
    _write_json(root/'SUMMARY.json', {'conditions': results})
    _write_json(output/'FROZEN_SELECTION.json', frozen)
    print('FROZEN PRIMARY', json.dumps(chosen, ensure_ascii=False), flush=True)
    print('RECIPE', recipes.get(chosen['method']), flush=True)
    print('PREVIOUS ROUNDS', json.dumps(round_winners, ensure_ascii=False), flush=True)
    print('TOP ELIGIBLE', [(r['method'],round(r['mean_auc'],4),round(r['worst_delta_vs_previous'],4))
                           for r in ranking if r['eligible']][:10], flush=True)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=('prepare', 'round1', 'round2', 'round3'))
    parser.add_argument('--output-root', type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args(argv)
    torch.set_num_threads(1)
    if args.command == 'prepare':
        prepare(args.output_root.resolve())
    elif args.command == 'round1':
        run_round1(args.output_root.resolve())
    elif args.command == 'round2':
        run_round2(args.output_root.resolve())
    else:
        run_round3(args.output_root.resolve())


if __name__ == '__main__':
    main()
