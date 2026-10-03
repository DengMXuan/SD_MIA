"""Single-seed CPU comparison of fixed evidence-preserving Pythia scorers."""
from __future__ import annotations

import argparse
import csv
import fcntl
import platform
from pathlib import Path
import time

import numpy as np
import scipy
import torch

from experiments.paths import ROOT, AUDITS
from experiments.pretraining.frozen_audit import identify_inputs, load_inputs, predict_frozen, read_json, require
from experiments.shared.audit.metrics import metrics
from experiments.shared.core.audit_metrics import rank_auc
from experiments.shared.core.audit_runtime import _write_json
from experiments.shared.core.deployment_archive import sha256_file
from experiments.shared.evaluation.provenance import import_closure
from experiments.shared.methods.preserved_accept_only import (
    OLD, PRIMARY, CORRECTION, FUSIONS, METHODS, base_scores, fit_normalization, fuse_scores,
)
from experiments.shared.protocols.protocol_archive import atomic_npz

SCHEMA = 'pythia_preserved_evidence_v1'
SCOPE = 'Exploratory single-seed frozen-cache experiment; no test-label selection or independent confirmation.'
SOURCES = {
    'mimir13': ('arxiv', 'dm_mathematics', 'github', 'hackernews', 'pile_cc',
                'pubmed_central', 'wikipedia_(en)', 'full_pile'),
    'mimir7': ('github', 'arxiv'),
    'wikimia': ('length64', 'length128'),
}
COMPLETE_FILES = ('REQUEST.json', 'REPORT.json', 'scores.npz', 'NORMALIZATION.json')


def tasks_for(benchmark, sources, seed, input_root):
    tasks = []
    for source in sources:
        if benchmark == 'mimir13':
            suffix = 'none' if source == 'full_pile' else 'ngram_13_0.8'
            identity = f'mimir/{source}/{suffix}'
            folder = input_root / 'tasks' / source / f'seed{seed}'
        else:
            identity = (f'mimir/{source}/ngram_7_0.2' if benchmark == 'mimir7'
                        else f'wikimia/official_{source}/temporal_proxy')
            folder = input_root / 'tasks' / ('mimir02' if benchmark == 'mimir7' else 'wikimia') / source / f'seed{seed}' / 'main'
        tasks.append(dict(source=source, benchmark=identity, seed=seed, report=str(folder / OLD / 'REPORT.json')))
    return tasks


def disjoint_output(output, inputs):
    output = Path(output).resolve()
    for value in inputs.values():
        parent = Path(value['path']).resolve().parent
        require(output != parent and output not in parent.parents and parent not in output.parents,
                'output must be separate from all input artifact directories')


def measure(variants, labels, calibration, test, seed, bootstrap, *, compare_previous=False):
    result = {name: metrics(values, labels, calibration, test, seed=seed, bootstrap=0)
              for name, values in variants.items()}
    positive, negative = test[labels[test] == 1], test[labels[test] == 0]
    draws = {name: [] for name in variants}
    tail_draws = {name: [] for name in variants} if compare_previous else {}
    rng = np.random.default_rng(seed)
    for _ in range(bootstrap):
        member = rng.choice(positive, len(positive), replace=True)
        nonmember = rng.choice(negative, len(negative), replace=True)
        for name, values in variants.items():
            draws[name].append(rank_auc(values[member], values[nonmember]))
            if compare_previous:
                from experiments.shared.audit.metrics import roc_operating_point
                sample = np.r_[values[member], values[nonmember]]
                y = np.r_[np.ones(len(member), int), np.zeros(len(nonmember), int)]
                tail_draws[name].append(roc_operating_point(sample, y, .01)[0])
    references = [(OLD, 'legacy'), ('accept_rate', 'accept_rate')]
    if compare_previous:
        references.append((PRIMARY, 'previous'))
    for name, row in result.items():
        for reference, suffix in references:
            row[f'delta_auc_vs_{suffix}'] = row['auc'] - result[reference]['auc']
            if bootstrap:
                row[f'paired_delta_auc_vs_{suffix}_ci95'] = np.quantile(
                    np.asarray(draws[name]) - draws[reference], [.025, .975]).tolist()
            if compare_previous:
                row[f'delta_roc_tpr1_vs_{suffix}'] = row['roc_tpr_at_1pct_fpr'] - result[reference]['roc_tpr_at_1pct_fpr']
                if bootstrap:
                    row[f'paired_delta_roc_tpr1_vs_{suffix}_ci95'] = np.quantile(
                        np.asarray(tail_draws[name]) - tail_draws[reference], [.025, .975]).tolist()
        if bootstrap:
            row['auc_ci95'] = np.quantile(draws[name], [.025, .975]).tolist()
    return result


def completed_report(folder, request):
    hashes = read_json(folder / '_COMPLETE.json')
    require(set(hashes) == set(COMPLETE_FILES), 'invalid completion marker')
    for name, expected in hashes.items():
        require(sha256_file(folder / name) == expected, f'completed {name} checksum mismatch')
    require(read_json(folder / 'REQUEST.json') == request, 'completed request mismatch')
    report = read_json(folder / 'REPORT.json')
    require(report['request_sha256'] == hashes['REQUEST.json']
            and report['scores_sha256'] == hashes['scores.npz']
            and report['normalization_sha256'] == hashes['NORMALIZATION.json'], 'completed report provenance mismatch')
    return report


def execute_condition(task, settings, output_root):
    paths, fingerprints, old_report = identify_inputs(task['report'], task['benchmark'], task['seed'])
    require(fingerprints == task['inputs'], 'input changed after plan freeze')
    disjoint_output(output_root, fingerprints)
    output = output_root / 'conditions' / task['source']
    output.mkdir(parents=True, exist_ok=True)
    request = dict(task=task, settings=settings)
    request_path = output / 'REQUEST.json'
    if request_path.exists():
        require(read_json(request_path) == request, 'request changed; use a new output root')
    else:
        require(not any(output.iterdir()), 'refusing to adopt a nonempty condition directory')
        _write_json(request_path, request)
    if (output / '_COMPLETE.json').exists():
        return completed_report(output, request)
    started = time.perf_counter()
    data, parts, saved, selected = load_inputs(paths, old_report)
    pmf = predict_frozen(paths['detector'], data)
    base = base_scores(pmf, data['counts'], data['lengths'], data['features'][:, 0])
    error = float(np.max(np.abs(base[OLD][selected] - saved['scores'])))
    require(np.allclose(base[OLD][selected], saved['scores'], atol=1e-8, rtol=1e-8),
            f'legacy replay mismatch ({error}); refusing changed-detector comparison')
    normalization = fit_normalization({k: v[parts['reference']] for k, v in base.items()})
    all_scores = fuse_scores(base, normalization)
    tail_normalization = None
    if settings.get('suite') == 'tail':
        from experiments.shared.methods import tail_accept_only as tail
        features = tail.components(base, data['counts'], data['lengths'])
        tail_normalization = tail.fit_reference({k: v[parts['reference']] for k, v in features.items()})
        all_scores = {**{k: all_scores[k] for k in tail.CONTROL_METHODS},
                      **tail.score(features, tail_normalization)}
    variants = {k: v[selected] for k, v in all_scores.items()}
    variants[OLD] = saved['scores'].copy()
    results = measure(variants, saved['labels'], saved['calibration'], saved['test'],
                      task['seed'], settings['bootstrap'], compare_previous=settings.get('suite') == 'tail')
    for name in ('auc', 'roc_tpr_at_1pct_fpr', 'roc_tpr_at_10pct_fpr',
                 'calibrated_tpr_at_1pct', 'calibrated_actual_fpr_at_1pct',
                 'calibrated_tpr_at_10pct', 'calibrated_actual_fpr_at_10pct'):
        require(abs(results[OLD][name] - old_report['metrics'][name]) < 1e-12, f'legacy metric mismatch: {name}')
    atomic_npz(output / 'scores.npz', {**{k: saved[k] for k in ('record_ids', 'labels', 'calibration', 'test')}, **variants})
    _write_json(output / 'NORMALIZATION.json', dict(
        source_partition='reference=train+validation', reference_ids=data['record_ids'][parts['reference']],
        channels=normalization, weights=FUSIONS, tail_channels=tail_normalization,
        tail_spec=settings.get('tail_spec'), calibration_used_for_fitting=False))
    report = dict(schema=SCHEMA, scope=SCOPE, source=task['source'], benchmark=task['benchmark'], seed=task['seed'],
                  primary_method=settings['primary'], metrics=results, replay_max_abs_error=error,
                  seconds=time.perf_counter() - started, language_model_queries=0, detector_training_steps=0,
                  evaluation_context=old_report['evaluation_context'],
                  request_sha256=sha256_file(request_path), scores_sha256=sha256_file(output / 'scores.npz'),
                  normalization_sha256=sha256_file(output / 'NORMALIZATION.json'))
    _write_json(output / 'REPORT.json', report)
    _write_json(output / '_COMPLETE.json', {name: sha256_file(output / name) for name in COMPLETE_FILES})
    print(f"{task['source']}: legacy={results[OLD]['auc']:.4f}, accept={results['accept_rate']['auc']:.4f}, "
          f"primary={results[settings['primary']]['auc']:.4f}; replay error={error:.3g}", flush=True)
    return report


def summarize(output):
    plan = read_json(output / 'PLAN.json')
    require(plan['schema'] == SCHEMA, 'unexpected experiment schema')
    rows, missing = [], []
    for task in plan['tasks']:
        folder = output / 'conditions' / task['source']
        if not (folder / '_COMPLETE.json').exists():
            missing.append(task['source'])
            continue
        report = completed_report(folder, dict(task=task, settings=plan['settings']))
        rows.extend(dict(source=task['source'], seed=task['seed'], method=name, **values)
                    for name, values in report['metrics'].items())
    primary = plan['settings']['primary']
    result = dict(schema=SCHEMA, scope=SCOPE, primary_method=primary, rows=rows, missing=missing,
                  completed=len(plan['tasks']) - len(missing), planned=len(plan['tasks']))
    _write_json(output / 'SUMMARY.json', result)
    if rows:
        with (output / 'SUMMARY.csv').open('w', newline='') as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    lines = ['# Pythia evidence-preserving scores', '', SCOPE, '',
             f"Primary fixed before scoring: `{primary}`. Seed: {plan['seed']}.",
             f"Complete: {result['completed']}/{result['planned']}. Missing: {missing}.", '',
             'Normalization uses auxiliary reference documents only; calibration remains independent.',
             'Frozen B=2 feedback and frozen TCN; zero new language-model queries or detector training steps.',
             'Full-Pile is a separate mixture; WikiMIA labels are time proxies. No automatic best-method selection.', '',
             '| Source | Method | AUC | Δ vs old | Δ vs accept | ROC TPR@1% | ROC TPR@10% | Cal TPR@1% | Cal FPR@1% |',
             '|---|---|---:|---:|---:|---:|---:|---:|---:|']
    for row in rows:
        lines.append(f"| {row['source']} | {row['method']} | {row['auc']:.4f} | "
                     f"{row['delta_auc_vs_legacy']:+.4f} | {row['delta_auc_vs_accept_rate']:+.4f} | "
                     f"{100*row['roc_tpr_at_1pct_fpr']:.2f}% | {100*row['roc_tpr_at_10pct_fpr']:.2f}% | "
                     f"{100*row['calibrated_tpr_at_1pct']:.2f}% | {100*row['calibrated_actual_fpr_at_1pct']:.2f}% |")
    lines += ['', 'AUC and paired ΔAUC intervals are in SUMMARY.json/CSV. Document bootstrap fixes the detector and calibration set.',
              'These are previously inspected test sets; intervals are not corrected for multiple comparisons.', '']
    (output / 'SUMMARY.md').write_text('\n'.join(lines))
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument('command', choices=('dry-run', 'run', 'summarize'), nargs='?', default='dry-run')
    parser.add_argument('--benchmark', choices=tuple(SOURCES), required=True)
    parser.add_argument('--sources', nargs='+')
    parser.add_argument('--seed', type=int, choices=(1919, 1949, 1978), default=1919)
    parser.add_argument('--input-root', type=Path)
    parser.add_argument('--output-root', type=Path)
    parser.add_argument('--bootstrap', type=int, default=1000)
    parser.add_argument('--threads', type=int, default=1)
    parser.add_argument('--suite', choices=('preserved', 'tail'), default='preserved')
    args = parser.parse_args(argv)
    require(args.bootstrap >= 0 and args.threads > 0, 'invalid bootstrap or thread count')
    sources = args.sources or list(SOURCES[args.benchmark])
    require(len(set(sources)) == len(sources) and set(sources).issubset(SOURCES[args.benchmark]), 'invalid or duplicate sources')
    input_root = (args.input_root or AUDITS / ('pythia_mimir_v1' if args.benchmark == 'mimir13'
                                              else 'paper_positive_controls_v1')).resolve()
    batch = 'pythia_tail_v1' if args.suite == 'tail' else 'pythia_preserved_v1'
    output = (args.output_root or AUDITS / batch / args.benchmark / f'seed{args.seed}').resolve()
    if args.command == 'summarize':
        require(output.is_dir(), 'no output directory')
        with (output / '.lock').open('a') as lock:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            result = summarize(output)
        return 2 if result['missing'] else 0
    tasks = tasks_for(args.benchmark, sources, args.seed, input_root)
    # Preflight the entire requested benchmark before creating an output directory.
    for task in tasks:
        _, task['inputs'], _ = identify_inputs(task['report'], task['benchmark'], task['seed'])
        disjoint_output(output, task['inputs'])
    settings = dict(primary=PRIMARY, methods=list(METHODS), correction=CORRECTION, weights=FUSIONS,
                    normalization='reference-only mean and sample SD; constant channels disabled',
                    budget=2, device='cpu', bootstrap=args.bootstrap, threads=args.threads,
                    implementation={str(p.relative_to(ROOT)): sha256_file(p) for p in import_closure(
                        ROOT, ['experiments.pretraining.preserved_evidence'])},
                    versions=dict(python=platform.python_version(), numpy=np.__version__, scipy=scipy.__version__, torch=torch.__version__))
    if args.suite == 'tail':
        from experiments.shared.methods import tail_accept_only as tail
        settings.update(suite='tail', primary=tail.PRIMARY, methods=list(tail.METHODS), tail_spec=tail.SPEC)
    plan = dict(schema=SCHEMA, scope=SCOPE, benchmark=args.benchmark, seed=args.seed,
                output=str(output), settings=settings, tasks=tasks)
    if args.command == 'dry-run':
        print(f"Validated {len(tasks)} frozen conditions × {len(settings['methods'])} scorers; seed={args.seed}; "
              f"B=2; primary={settings['primary']}; no output created. Destination: {output}")
        return 0
    torch.set_num_threads(args.threads)
    output.mkdir(parents=True, exist_ok=True)
    with (output / '.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        plan_path = output / 'PLAN.json'
        if plan_path.exists():
            require(read_json(plan_path) == plan, 'plan changed; use a new output root')
        else:
            require(set(p.name for p in output.iterdir()) <= {'.lock'}, 'output directory is already in use')
            _write_json(plan_path, plan)
        for task in tasks:
            execute_condition(task, settings, output)
        result = summarize(output)
    print(f"Completed {result['completed']}/{result['planned']}; {len(result['rows'])} rows. {output / 'SUMMARY.md'}")
    return 2 if result['missing'] else 0


if __name__ == '__main__':
    raise SystemExit(main())
