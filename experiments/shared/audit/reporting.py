"""Checked per-method results, seed aggregation and physical costs for all matrices."""
import csv
import json
from pathlib import Path

import numpy as np

from experiments.paths import audit_executions
from experiments.shared.core.audit_runtime import _write_json
from .artifacts import digest


def summarize(tasks, output_root, *, methods, baseline_methods, describe, title,
              read_result, execution_root=None):
    """Describe supplies display metadata and draft roles, never metric/cost logic.

    One target-only result may be displayed for several draft roles; physical
    execution groups are counted once. Incremental costs never fill standalone
    columns. Invalid sources remain errors rather than silently trusted results.
    """
    output_root = Path(output_root)
    results, errors = {}, []
    for task in tasks:
        for method in task['methods']:
            folder = Path(task['output']) / method
            if not (folder / 'REPORT.json').exists():
                continue
            try:
                results[task['id'], method] = read_result(folder, digest({'task': task, 'method': method}))
            except (OSError, ValueError, KeyError) as error:
                errors.append(dict(task=task['id'], method=method, error=str(error)))
    rows, groups, physical_groups = [], {}, {}
    conditions = {}
    for task in tasks:
        conditions.setdefault(json.dumps(task['condition'], sort_keys=True), []).append(task)
    for matching in conditions.values():
        baseline = next((task for task in matching if task['kind'] == 'baseline'), None)
        representative = baseline or matching[0]
        condition = representative['condition']
        id_hashes = {r['record_ids_sha256'] for task in matching for method in task['methods']
                     if (r := results.get((task['id'], method))) is not None}
        if len(id_hashes) > 1:
            raise ValueError(f'methods scored different records in {condition}')
        metadata, roles = describe(representative)
        for role in roles:
            for method in methods:
                task = baseline if method in baseline_methods else next(
                    t for t in matching if t.get('draft_role') == role and method in t['methods'])
                if task is None:
                    raise ValueError(f'no baseline task for requested method {method}')
                report = results.get((task['id'], method))
                row = {**condition, **metadata, 'draft_role': role, 'method': method,
                       'status': 'complete' if report else 'missing', 'source_task': task['id'],
                       'reused_target_only': method in baseline_methods}
                if report:
                    row.update(report['metrics'])
                    row.update(report['cost'])
                    row['access_channel'] = report['access_channel']
                    row['report'] = str(Path(task['output']) / method / 'REPORT.json')
                    cost = report['cost']
                    seconds = cost.get('execution_group_seconds')
                    if seconds is None:
                        seconds = cost['total_seconds']
                    group = cost['execution_group']
                    physical_groups[group] = max(physical_groups.get(group, 0.), seconds)
                rows.append(row)
                identity = {k: row[k] for k in ('model_pair', 'benchmark', 'epoch', 'draft_role', 'method') if k in row}
                groups.setdefault(tuple(identity.items()), []).append(row)
    aggregates = []
    for identity, values in groups.items():
        completed = [row for row in values if row['status'] == 'complete']
        entry = dict(identity, expected_seeds=len(values), completed_seeds=len(completed))
        fields = set.intersection(*(set(r) for r in completed)) if completed else set()
        for field in sorted(fields - {'epoch', 'condition_seed', 'reused_target_only'}):
            if all(isinstance(r[field], (int, float)) and not isinstance(r[field], bool) for r in completed):
                numbers = [r[field] for r in completed]
                entry[field + '_mean'] = float(np.mean(numbers))
                entry[field + '_std'] = float(np.std(numbers, ddof=1)) if len(numbers) > 1 else None
        aggregates.append(entry)
    output_root.mkdir(parents=True, exist_ok=True)
    for name, table in (('RESULTS', rows), ('SEED_SUMMARY', aggregates)):
        fields = list(dict.fromkeys(key for row in table for key in row))
        temporary = output_root / f'.{name}.tmp.csv'
        with temporary.open('w', newline='') as stream:
            writer = csv.DictWriter(stream, fieldnames=fields)
            writer.writeheader()
            writer.writerows(table)
        temporary.replace(output_root / f'{name}.csv')
    selected = {task['id'] for task in tasks}
    attempts = [json.loads(path.read_text()) for path in
                Path(execution_root or audit_executions(output_root)).glob('*/STATUS.json')]
    completed_count = sum(r['status'] == 'complete' for r in rows)
    shared_costs = any(row.get('cost_basis') == 'physical_incremental' for row in rows)
    note = 'worker time includes loading/retries and is not matrix elapsed wall time'
    if baseline_methods:
        note = 'baseline display duplication is not independent evidence; ' + note
    if shared_costs:
        note += '; shared-reference rows expose physical incremental fields and leave standalone method cost columns empty'
    payload = dict(complete=completed_count == len(rows) and not errors, expected_rows=len(rows),
                   completed_rows=completed_count, errors=errors, rows=rows, seed_summary=aggregates,
                   unique_successful_execution_groups=len(physical_groups),
                   unique_successful_measured_method_seconds=sum(physical_groups.values()),
                   attempted_worker_wall_seconds_sum=sum(r.get('worker_wall_seconds', 0.)
                                                         for r in attempts if r.get('task') in selected),
                   note=note)
    _write_json(output_root / 'SUMMARY.json', payload)
    include_pair = any('model_pair' in row for row in rows)
    columns = (['Model pair'] if include_pair else []) + [
        'Dataset', 'Epoch', 'Seed', 'Draft', 'Method', 'AUC', 'pAUC10 norm', 'ROC TPR10',
        'ROC TPR1', 'Cal TPR1', 'Cal FPR1', 'ms/record', 'Status']
    lines = [f'# {title}', '', f'Completed rows: {completed_count}/{len(rows)}. Complete: {payload["complete"]}.', '',
             'ROC and independently calibrated TPR are separate. pAUC below is area/0.10; raw area is retained in CSV/JSON.', '',
             'Shared-reference rows without standalone cost show — in ms/record; their physical incremental costs use physical_incremental_* fields.' if shared_costs else '', '',
             '| ' + ' | '.join(columns) + ' |', '|' + '|'.join(['---'] * len(columns)) + '|']
    for row in rows:
        numbers = [f'{row[key]:.4f}' if row.get(key) is not None else '—' for key in
                   ('auc', 'pauc_10_normalized', 'roc_tpr_at_10pct_fpr', 'roc_tpr_at_1pct_fpr',
                    'calibrated_tpr_at_1pct', 'calibrated_actual_fpr_at_1pct', 'amortized_ms_per_record')]
        values = ([row['model_pair']] if include_pair else []) + [row['benchmark'], str(row['epoch']),
                  str(row['condition_seed']), row['draft_role'], row['method'], *numbers, row['status']]
        lines.append('| ' + ' | '.join(values) + ' |')
    temporary = output_root / '.RESULTS.tmp.md'
    temporary.write_text('\n'.join(lines) + '\n')
    temporary.replace(output_root / 'RESULTS.md')
    return payload
