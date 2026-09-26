"""Checked, per-seed and across-seed comparison against saved current-main runs."""
from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np

from experiments.shared.audit.artifacts import digest, read_result
from experiments.shared.audit.config import BASELINE_DEFAULTS
from experiments.shared.core.audit_runtime import _write_json
from standalone.pretraining_baselines.contract import (
    MAIN, METHODS, frozen_contract, assert_main_partitions, assert_score_roles,
)

FIELDS = ('auc', 'pauc_10_normalized', 'roc_tpr_at_1pct_fpr', 'roc_tpr_at_10pct_fpr',
          'calibrated_tpr_at_1pct', 'calibrated_actual_fpr_at_1pct',
          'calibrated_tpr_at_10pct', 'calibrated_actual_fpr_at_10pct')


def checked_report(task, method, manifest, partitions):
    main = method == MAIN
    root = Path(task['main_output'] if main else task['output'])
    assert_main_partitions(root, partitions)
    if not (root / 'PARTITIONS.json').exists():
        raise ValueError('result has no frozen partition manifest')
    request = json.loads((root / ('MAIN_REQUEST.json' if main else 'BASELINE_REQUEST.json')).read_text())
    if (request['partitions_sha256'] != digest(partitions)
            or request['methods'] != ([MAIN] if main else list(METHODS))
            or request['settings']['audit_seed'] != task['seed']
            or request['condition']['benchmark'] != manifest['benchmark']
            or request['condition']['models'] != manifest['models']
            or request['evaluation_context']['data_manifest'] != task['manifest']
            or request['evaluation_context']['token_contract'] != manifest['token_contract']):
        raise ValueError('result request belongs to another data/model/seed/partition condition')
    if not main and request['settings']['baseline'] != BASELINE_DEFAULTS:
        raise ValueError('baseline hyperparameters changed')
    # Historical main is read as a frozen artifact: verify its request, data,
    # weights inventory, scores, detector and observation checksums. Its old
    # source files need not equal today's implementation. Never rerun it here.
    report = read_result(root / method, digest(dict(task=request, method=method)),
                         request['sources_sha256'], check_sources=not main)
    if main:
        from experiments.shared.audit.artifacts import checkpoint_inventory
        for checkpoint in report['sources']['checkpoints']:
            if checkpoint_inventory(Path(checkpoint['path'])) != checkpoint['inventory']:
                raise ValueError('historical main checkpoint inventory changed')
    if report['settings'] != request['settings'] or report['condition'] != request['condition']:
        raise ValueError('report metadata disagrees with request')
    assert_score_roles(root / method, partitions)
    with np.load(root / method / 'scores.npz', allow_pickle=False) as archive:
        expected_labels = [1] * manifest['counts']['member'] + [0] * manifest['counts']['nonmember']
        if archive['labels'][archive['test']].tolist() != expected_labels:
            raise ValueError('saved test labels differ from frozen labels')
    expected_counts = dict(n_test_member=manifest['counts']['member'],
        n_test_nonmember=manifest['counts']['nonmember'], n_calibration=len(partitions['record_ids']['calibration']))
    if any(report['metrics'].get(k) != v for k, v in expected_counts.items()):
        raise ValueError('reported evaluation counts differ from frozen partitions')
    return report


def summarize(tasks, destination):
    rows = []
    for task in tasks:
        manifest, partitions, *_ = frozen_contract(task['manifest'], task['seed'])
        if partitions['manifest_sha256'] != task['manifest_sha256'] or digest(partitions) != task['partitions_sha256']:
            raise ValueError('data changed after planning')
        for method in (MAIN, *METHODS):
            root = Path(task['main_output'] if method == MAIN else task['output'])
            row = dict(experiment=task['experiment'], source=task['source'], seed=task['seed'],
                method=method, state='missing', report=str(root / method / 'REPORT.json'))
            if Path(row['report']).exists():
                try:
                    report = checked_report(task, method, manifest, partitions)
                    row.update(state='complete', metrics=report['metrics'], cost=report['cost'],
                        membership_verified=report['evaluation_context']['membership_verified'])
                except (OSError, ValueError, KeyError, TypeError) as error:
                    row.update(state='invalid', error=str(error))
            rows.append(row)
    aggregates = []
    for source in dict.fromkeys(t['source'] for t in tasks):
        for method in (MAIN, *METHODS):
            group = [r for r in rows if r['source'] == source and r['method'] == method]
            complete = [r for r in group if r['state'] == 'complete']
            entry = dict(source=source, method=method, expected_seeds=[r['seed'] for r in group],
                         completed_seeds=[r['seed'] for r in complete], complete=len(complete) == len(group))
            # Do not silently present a partial seed subset as the full result.
            if entry['complete']:
                entry['metrics'] = {field: dict(
                    mean=float(np.mean([r['metrics'][field] for r in complete])),
                    std=float(np.std([r['metrics'][field] for r in complete], ddof=1)) if len(complete) > 1 else None)
                    for field in FIELDS}
            aggregates.append(entry)
    baselines = [r for r in rows if r['method'] != MAIN]
    destination = Path(destination)
    result = dict(schema='pretraining_baselines7_comparison_v1',
        baseline_complete=all(r['state'] == 'complete' for r in baselines),
        comparison_complete=all(r['state'] == 'complete' for r in rows),
        completed_baseline_rows=sum(r['state'] == 'complete' for r in baselines),
        expected_baseline_rows=len(baselines), reports=str(destination), rows=rows, aggregates=aggregates)
    destination.mkdir(parents=True, exist_ok=True)
    _write_json(destination / 'COMPARISON.json', result)
    columns = ('experiment', 'source', 'seed', 'method', 'state', *FIELDS,
               'auc_ci_low', 'auc_ci_high', 'amortized_ms_per_record', 'report', 'error')
    with (destination / 'COMPARISON.csv').open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=columns, extrasaction='ignore')
        writer.writeheader()
        writer.writerows({**r, **r.get('metrics', {}), **r.get('cost', {})} for r in rows)
    lines = ['# Pretrained main vs seven baselines', '',
        'Same frozen test IDs, labels, raw-token contract and independent calibration IDs.',
        'Qwen temporal classes are historical/recent proxies, not verified membership.',
        'Main results are read-only historical artifacts; missing/invalid results are explicit.',
        'Values below are mean ± sample SD across all requested seeds; see CSV for each seed.', '',
        '| Dataset / variant | Method | Seeds complete | AUC | ROC TPR@1% FPR | Calibrated TPR@1% | Calibrated actual FPR |',
        '|---|---|---:|---:|---:|---:|---:|']
    for entry in aggregates:
        def cell(field):
            if not entry['complete']:
                return '—'
            v = entry['metrics'][field]
            return f'{v["mean"]:.4f}' + (f' ± {v["std"]:.4f}' if v['std'] is not None else '')
        lines.append('| ' + ' | '.join([entry['source'], entry['method'],
            f'{len(entry["completed_seeds"])}/{len(entry["expected_seeds"])}',
            *(cell(f) for f in ('auc', 'roc_tpr_at_1pct_fpr', 'calibrated_tpr_at_1pct',
                               'calibrated_actual_fpr_at_1pct'))]) + ' |')
    errors = [r for r in rows if r['state'] == 'invalid']
    if errors:
        lines += ['', 'Invalid artifacts:', *[f'- {r["source"]}/seed{r["seed"]}/{r["method"]}: {r["error"]}' for r in errors]]
    (destination / 'COMPARISON.md').write_text('\n'.join(lines) + '\n')
    return result
