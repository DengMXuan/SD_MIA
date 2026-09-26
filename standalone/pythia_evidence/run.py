#!/usr/bin/env python3
"""Independent, CPU-only Pythia evidence ablation on frozen B=2 archives.

No imports from experiments; no language-model queries or detector fitting.
Run `python run.py --help` and read the adjacent README before collecting results.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
import csv
import fcntl
import hashlib
import json
import os
from pathlib import Path
import platform
import time

os.environ.setdefault('OMP_NUM_THREADS', '1')
os.environ.setdefault('MKL_NUM_THREADS', '1')
import numpy as np
import scipy
from scipy.special import logsumexp
from scipy.stats import rankdata
import torch
from torch import nn
from torch.nn import functional as F

ROOT = Path(__file__).resolve().parents[2]
SOURCES = ('github', 'wikipedia_(en)', 'dm_mathematics', 'arxiv', 'hackernews', 'pile_cc', 'pubmed_central', 'full_pile')
DEFAULT_SOURCES = SOURCES[:3]
SEEDS = (1919, 1949, 1978)
FEATURES = ['logq', 'q_entropy_norm', 'q_rank_norm', 'q_top1_margin', 'position_fraction', 'start_fraction']
FEATURE_ORDER = [0, 4, 1, 2, 3]
OLD = 'main_fixed_sparse_positive'
STRONG = [0.5, 1.0, 2.0]
WEAK = [0.05, 0.1, 0.2]
SPARSE = [0.05, 0.1, 0.25]
METHODS = {
    OLD: dict(kind='tilt', eta=STRONG, rho=SPARSE),
    'accept_rate': dict(kind='residual', correction=0.0),
    'partial_residual_050': dict(kind='residual', correction=0.5),
    'full_residual': dict(kind='residual', correction=1.0),
    'weak_sparse_positive': dict(kind='tilt', eta=WEAK, rho=SPARSE),
    'strong_dense_positive': dict(kind='tilt', eta=STRONG, rho=[1.0]),
    'weak_dense_positive': dict(kind='tilt', eta=WEAK, rho=[1.0]),
}
SCOPE = 'exploratory frozen-cache ablation; previously inspected test sets; no automatic method selection'
SCHEMA = 'pythia_evidence_minimal_v1'


def read_json(path):
    return json.loads(Path(path).read_text())


def sha256(path):
    result = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for block in iter(lambda: handle.read(1 << 20), b''):
            result.update(block)
    return result.hexdigest()


def write_json(path, value):
    path = Path(path)
    temporary = path.with_name('.' + path.name + '.tmp')
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + '\n')
    temporary.replace(path)


def require(condition, message):
    if not condition:
        raise ValueError(message)


def disjoint_output(output, input_root):
    output, input_root = Path(output).resolve(), Path(input_root).resolve()
    require(output != input_root and input_root not in output.parents and output not in input_root.parents,
            'output root must be separate from the input artifact tree')


def completed_report(folder):
    checksums = read_json(folder / '_COMPLETE.json')
    require(set(checksums) == {'REQUEST.json', 'REPORT.json', 'scores.npz'}, 'invalid completion marker')
    for name, expected in checksums.items():
        require(sha256(folder / name) == expected, 'completed output checksum mismatch')
    report = read_json(folder / 'REPORT.json')
    require(report['request_sha256'] == checksums['REQUEST.json']
            and report['scores_sha256'] == checksums['scores.npz'], 'report provenance mismatch')
    return report


class FrozenCountTCN(nn.Module):
    """Forward-only snapshot of the saved 4-layer difficulty TCN architecture.

    State keys, masking, GELU, LayerNorm and dilations match the legacy model.
    The binomial kernel is loaded from the checkpoint, never re-estimated.
    Every run must reproduce the archived legacy scores before reporting results.
    """
    def __init__(self, state):
        super().__init__()
        channels, inputs = state['projection.weight'].shape
        kernel = state['log_kernel']
        require(inputs == 5 and kernel.shape == (3, 18), 'expected the frozen B=2 difficulty TCN')
        self.projection = nn.Linear(inputs, channels)
        self.convs = nn.ModuleList([nn.Conv1d(channels, channels, 3, padding=d, dilation=d) for d in (1, 2, 4, 8)])
        self.norms = nn.ModuleList([nn.LayerNorm(channels) for _ in self.convs])
        self.head = nn.Linear(channels, kernel.shape[1])
        self.register_buffer('log_kernel', torch.empty_like(kernel))
        self.load_state_dict(state, strict=True)
        self.requires_grad_(False).eval()

    def forward(self, features, mask):
        valid = mask.unsqueeze(-1).to(features.dtype)
        hidden = F.gelu(self.projection(features * valid)) * valid
        for conv, norm in zip(self.convs, self.norms):
            update = conv(hidden.transpose(1, 2)).transpose(1, 2)
            hidden = (hidden + F.gelu(norm(update))) * valid
        weights = F.log_softmax(self.head(hidden), dim=-1)
        return torch.logsumexp(weights.unsqueeze(-2) + self.log_kernel, dim=-1)


def predict(checkpoint_path, features, lengths):
    checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=True)
    model = FrozenCountTCN(checkpoint['state_dict'])
    mean, scale = checkpoint['mean'].numpy(), checkpoint['scale'].numpy()
    require(mean.shape == scale.shape == (5,) and np.isfinite(mean).all()
            and np.isfinite(scale).all() and (scale > 0).all(), 'invalid frozen feature normalization')
    x = (features[:, FEATURE_ORDER] - mean) / scale
    offsets = np.r_[0, np.cumsum(lengths)]
    output = np.empty((len(x), 3), dtype=np.float64)
    with torch.inference_mode():
        for first in range(0, len(lengths), 32):
            indices = range(first, min(first + 32, len(lengths)))
            width = int(max(lengths[i] for i in indices))
            batch = np.zeros((len(indices), width, 5), dtype=np.float32)
            mask = np.zeros((len(indices), width), dtype=bool)
            for row, i in enumerate(indices):
                batch[row, :lengths[i]] = x[offsets[i]:offsets[i + 1]]
                mask[row, :lengths[i]] = True
            values = model(torch.from_numpy(batch), torch.from_numpy(mask)).numpy()
            for row, i in enumerate(indices):
                output[offsets[i]:offsets[i + 1]] = values[row, :lengths[i]]
    return output


def score_variants(logpmf, counts, lengths):
    """Scoring takes no membership labels, partitions or calibration outcomes."""
    logpmf = np.asarray(logpmf, dtype=np.float64)
    counts, lengths = np.asarray(counts), np.asarray(lengths)
    require(lengths.ndim == 1 and len(lengths) > 0 and np.issubdtype(lengths.dtype, np.integer)
            and (lengths > 0).all(), 'invalid document lengths')
    require(logpmf.shape == (int(lengths.sum()), 3) and counts.shape == (len(logpmf),)
            and np.issubdtype(counts.dtype, np.integer) and np.isin(counts, [0, 1, 2]).all()
            and np.isfinite(logpmf).all() and np.allclose(logsumexp(logpmf, axis=1), 0, atol=2e-6),
            'invalid B=2 count PMFs or observations')
    offsets = np.r_[0, np.cumsum(lengths)]
    def sums(values):
        return np.add.reduceat(values, offsets[:-1], axis=0)
    accepted = sums(counts.astype(np.float64)) / (2 * lengths)
    expected = sums(np.exp(logpmf) @ np.arange(3)) / (2 * lengths)
    result = {}
    for name, spec in METHODS.items():
        if spec['kind'] == 'residual':
            result[name] = accepted - spec['correction'] * expected
            continue
        eta = np.asarray(spec['eta'])
        evidence = counts[:, None] * eta - logsumexp(
            logpmf[:, :, None] + np.arange(3)[None, :, None] * eta, axis=1)
        components = []
        for rho in spec['rho']:
            local = evidence if rho == 1 else np.logaddexp(np.log1p(-rho), np.log(rho) + evidence)
            components.append(sums(local))
        values = np.concatenate(components, axis=1)
        result[name] = logsumexp(values, axis=1) - np.log(values.shape[1])
    return result


def auc(values, labels):
    labels = np.asarray(labels)
    n1, n0 = int((labels == 1).sum()), int((labels == 0).sum())
    require(n1 > 0 and n0 > 0 and n1 + n0 == len(labels), 'AUC needs both binary classes')
    return float((rankdata(values)[labels == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def measure(scores, labels, calibration, test, seed, bootstrap):
    """Tie-preserving ROC and inclusive conformal thresholds; paired AUC bootstrap."""
    y = labels[test]
    member, nonmember = np.flatnonzero(y == 1), np.flatnonzero(y == 0)
    require(len(member) > 0 and len(nonmember) > 0 and (labels[calibration] == 0).all(), 'invalid evaluation roles')
    result = {}
    for name, values in scores.items():
        require(np.isfinite(values).all(), 'nonfinite document score')
        v = values[test]
        order = np.argsort(-v, kind='stable')
        ends = np.r_[np.flatnonzero(v[order][:-1] != v[order][1:]), len(v) - 1]
        tpr = np.r_[0, np.cumsum(y[order])[ends]] / len(member)
        fpr = np.r_[0, np.cumsum(1 - y[order])[ends]] / len(nonmember)
        pvalues = (1 + len(calibration) - np.searchsorted(np.sort(values[calibration]), v, side='left')) / (1 + len(calibration))
        row = dict(auc=auc(v, y), n_member=len(member), n_nonmember=len(nonmember), n_calibration=len(calibration))
        for rate, suffix in ((.01, '1'), (.1, '10')):
            point = np.flatnonzero(fpr <= rate + 1e-12)[-1]
            row[f'roc_tpr_at_{suffix}pct_fpr'] = float(tpr[point])
            row[f'roc_actual_fpr_at_{suffix}pct'] = float(fpr[point])
            row[f'calibrated_tpr_at_{suffix}pct'] = float((pvalues[member] <= rate).mean())
            row[f'calibrated_actual_fpr_at_{suffix}pct'] = float((pvalues[nonmember] <= rate).mean())
        result[name] = row
    for row in result.values():
        row['delta_auc_vs_legacy'] = row['auc'] - result[OLD]['auc']
    if bootstrap:
        rng = np.random.default_rng(seed)
        draws = {name: [] for name in scores}
        for _ in range(bootstrap):
            selected = np.r_[rng.choice(member, len(member)), rng.choice(nonmember, len(nonmember))]
            for name, values in scores.items():
                draws[name].append(auc(values[test][selected], y[selected]))
        for name, row in result.items():
            row['auc_ci95'] = np.quantile(draws[name], [.025, .975]).tolist()
            row['paired_delta_auc_ci95'] = np.quantile(np.asarray(draws[name]) - draws[OLD], [.025, .975]).tolist()
    return result


def identify_inputs(input_root, source, seed):
    folder = input_root / 'tasks' / source / f'seed{seed}'
    report_path = folder / OLD / 'REPORT.json'
    report = read_json(report_path)
    context = report['evaluation_context']
    expected_benchmark = f'mimir/{source}/' + ('none' if source == 'full_pile' else 'ngram_13_0.8')
    require(report['method'] == OLD and report['condition']['benchmark'] == expected_benchmark
            and report['condition']['condition_seed'] == seed and report['settings']['audit_seed'] == seed
            and context['training_regime'] == 'pretraining' and context['membership_verified'] is True
            and report['training_member_count'] == 0, 'input is not the requested frozen Pythia null-only audit')
    models = report['condition']['models']
    require(models['target']['repo_id'] == 'EleutherAI/pythia-6.9b'
            and models['draft']['repo_id'] == 'EleutherAI/pythia-1.4b', 'unexpected model pair')
    manifest = Path(context['data_manifest'])
    frozen = read_json(manifest)
    paths = dict(report=report_path, observations=Path(report['observation_archive']['path']),
                 sidecar=Path(str(report['observation_archive']['path']) + '.json'),
                 detector=report_path.parent / report['detector']['file'], scores=report_path.parent / 'scores.npz',
                 partitions=folder / 'PARTITIONS.json', manifest=manifest, records=manifest.parent / frozen['records_file'])
    fingerprints = {k: dict(path=str(p.resolve()), sha256=sha256(p)) for k, p in paths.items()}
    for key, expected in [('observations', report['observation_archive']['sha256']),
                          ('detector', report['detector']['sha256']), ('scores', report['scores_sha256']),
                          ('records', frozen['records_sha256'])]:
        require(fingerprints[key]['sha256'] == expected, f'{key} checksum differs from the frozen report/manifest')
    partition = read_json(paths['partitions'])
    require(partition['seed'] == seed and partition['manifest_sha256'] == fingerprints['manifest']['sha256']
            and frozen['models'] == models and frozen['selection_seed'] == seed, 'frozen input provenance mismatch')
    return paths, fingerprints, report


def load_inputs(paths, report):
    with np.load(paths['observations'], allow_pickle=False) as archive:
        data = dict(archive)
    with np.load(paths['scores'], allow_pickle=False) as archive:
        saved = dict(archive)
    sidecar = read_json(paths['sidecar'])
    require(sidecar['schema'] == 'sd_mia_protocol_observations_v1' and sidecar['feature_names'] == FEATURES
            and sidecar['archive_sha256'] == report['observation_archive']['sha256']
            and sidecar['contract']['protocol'] == 'fixed' and sidecar['contract']['starts'] == ['fixed']
            and sidecar['contract']['seed'] == report['settings']['audit_seed'], 'unexpected observation contract')
    fields = {'features', 'counts', 'lengths', 'document_indices', 'start_indices', 'record_ids', 'record_roles', 'labels'}
    require(set(data) == fields, 'unexpected observation fields; target probabilities are forbidden')
    ids, lengths, labels = data['record_ids'], data['lengths'], data['labels']
    require(ids.ndim == 1 and len(np.unique(ids)) == len(ids) and lengths.shape == ids.shape
            and labels.shape == ids.shape and np.isin(labels, [0, 1]).all()
            and np.array_equal(data['document_indices'], np.arange(len(ids)))
            and np.array_equal(data['start_indices'], np.zeros(len(ids))), 'invalid record/trajectory alignment')
    require(data['features'].shape == (int(lengths.sum()), 6) and np.isfinite(data['features']).all(), 'invalid features')
    require(data['record_roles'].shape == ids.shape and np.array_equal(labels, (data['record_roles'] == 'member').astype(int)),
            'labels and record roles differ')
    lookup = {v: i for i, v in enumerate(ids)}
    partitions = read_json(paths['partitions'])['record_ids']
    used = set()
    for name in ('train', 'validation', 'calibration', 'test'):
        chosen = partitions[name]
        require(chosen and len(set(chosen)) == len(chosen) and not used.intersection(chosen)
                and set(chosen).issubset(lookup), f'overlapping/invalid {name} partition')
        used.update(chosen)
        ix = [lookup[v] for v in chosen]
        roles = data['record_roles'][ix]
        require(np.isin(roles, ['member', 'nonmember']).all() if name == 'test'
                else (roles == 'audit_auxiliary').all(), f'unexpected {name} roles')
    require(used == set(ids) and set(partitions['reference']) == set(partitions['train'] + partitions['validation']),
            'partitions do not cover the frozen data')
    selected = np.asarray([lookup[v] for v in saved['record_ids']])
    require(len(set(selected)) == len(selected) and np.array_equal(labels[selected], saved['labels'])
            and np.isfinite(saved['scores']).all(), 'saved score identity/labels mismatch')
    used = set()
    for name in ('calibration', 'test'):
        ix = saved[name]
        require(ix.ndim == 1 and np.issubdtype(ix.dtype, np.integer) and (ix >= 0).all() and (ix < len(selected)).all()
                and len(set(ix)) == len(ix) and not used.intersection(ix.tolist()), f'invalid saved {name} indices')
        require(set(saved['record_ids'][ix]) == set(partitions[name]), f'saved {name} split differs')
        used.update(ix.tolist())
    require(used == set(range(len(selected))), 'saved evaluation split is incomplete')
    records = {}
    with paths['records'].open() as handle:
        for line in handle:
            row = json.loads(line)
            require(row['record_id'] not in records, 'duplicate frozen input ID')
            records[row['record_id']] = (row['label'], len(row['token_ids']) - 1)
    require(set(records) == set(ids) and all(records[v] == (int(labels[i]), int(lengths[i])) for i, v in enumerate(ids)),
            'observations differ from frozen input labels/token lengths')
    return data, saved, selected


def execute_condition(input_root, output_root, source, seed, settings):
    paths, fingerprints, old_report = identify_inputs(input_root, source, seed)
    for path in paths.values():
        disjoint_output(output_root, path.resolve().parent)
    output = output_root / 'conditions' / source / f'seed{seed}'
    request = dict(source=source, seed=seed, settings=settings, inputs=fingerprints)
    output.mkdir(parents=True, exist_ok=True)
    request_path = output / 'REQUEST.json'
    if request_path.exists():
        require(read_json(request_path) == request, 'inputs or implementation changed; use a new output root')
    else:
        require(not any(output.iterdir()), 'refusing to adopt a nonempty output directory')
        write_json(request_path, request)
    marker = output / '_COMPLETE.json'
    if marker.exists():
        completed_report(output)
        print(json.dumps(dict(source=source, seed=seed, reused=True)), flush=True)
        return
    started = time.perf_counter()
    data, saved, selected = load_inputs(paths, old_report)
    logpmf = predict(paths['detector'], data['features'], data['lengths'])
    variants = {k: v[selected] for k, v in score_variants(logpmf, data['counts'], data['lengths']).items()}
    error = float(np.max(np.abs(variants[OLD] - saved['scores'])))
    require(np.allclose(variants[OLD], saved['scores'], atol=1e-5, rtol=1e-6),
            f'legacy replay mismatch ({error}); refusing to compare a changed detector')
    # Keep the literal archived control scores after independently reproducing them.
    variants[OLD] = saved['scores'].copy()
    metrics = measure(variants, saved['labels'], saved['calibration'], saved['test'], seed, settings['bootstrap'])
    require(abs(metrics[OLD]['auc'] - old_report['metrics']['auc']) < 1e-12, 'legacy AUC mismatch')
    for suffix in ('1', '10'):
        for name in (f'roc_tpr_at_{suffix}pct_fpr', f'calibrated_tpr_at_{suffix}pct', f'calibrated_actual_fpr_at_{suffix}pct'):
            require(abs(metrics[OLD][name] - old_report['metrics'][name]) < 1e-12, f'legacy metric mismatch: {name}')
    temporary = output / '.scores.tmp.npz'
    np.savez_compressed(temporary, record_ids=saved['record_ids'], labels=saved['labels'],
                        calibration=saved['calibration'], test=saved['test'], **variants)
    temporary.replace(output / 'scores.npz')
    report = dict(scope=SCOPE, source=source, seed=seed, methods=METHODS, metrics=metrics,
                  replay_max_abs_error=error, seconds=time.perf_counter() - started,
                  language_model_queries=0, detector_training_steps=0, request_sha256=sha256(request_path),
                  old_report=str(paths['report'].resolve()), scores_sha256=sha256(output / 'scores.npz'))
    write_json(output / 'REPORT.json', report)
    write_json(marker, {name: sha256(output / name) for name in ('REQUEST.json', 'REPORT.json', 'scores.npz')})
    print(json.dumps(dict(source=source, seed=seed, replay_error=error,
                         auc={k: round(v['auc'], 6) for k, v in metrics.items()})), flush=True)


def summarize(output_root):
    plan = read_json(output_root / 'PLAN.json')
    require(plan.get('schema') == SCHEMA, 'not a standalone evidence experiment')
    rows, missing = [], []
    for task in plan['tasks']:
        folder = output_root / 'conditions' / task['source'] / f"seed{task['seed']}"
        if not (folder / '_COMPLETE.json').exists():
            missing.append(task)
            continue
        report = completed_report(folder)
        require(report['source'] == task['source'] and report['seed'] == task['seed'], 'condition identity mismatch')
        require(read_json(folder / 'REQUEST.json')['settings'] == plan['settings'], 'mixed scorer settings')
        for method, metrics in report['metrics'].items():
            rows.append(dict(source=task['source'], seed=task['seed'], method=method, **metrics))
    grouped = defaultdict(list)
    for row in rows:
        grouped[row['source'], row['method']].append(row)
    aggregates = []
    for (source, method), values in grouped.items():
        fields = ('auc', 'delta_auc_vs_legacy', 'roc_tpr_at_1pct_fpr', 'roc_tpr_at_10pct_fpr',
                  'calibrated_tpr_at_1pct', 'calibrated_actual_fpr_at_1pct',
                  'calibrated_tpr_at_10pct', 'calibrated_actual_fpr_at_10pct')
        aggregates.append(dict(source=source, method=method, seeds=[v['seed'] for v in values],
            **{key: dict(mean=float(np.mean([v[key] for v in values])),
                         std=float(np.std([v[key] for v in values], ddof=1)) if len(values) > 1 else None) for key in fields}))
    summary = dict(scope=SCOPE, missing=missing, rows=rows, aggregates=aggregates,
                   uncertainty='paired stratified document bootstrap within seed; seed SD is descriptive, not an independent-replication CI')
    write_json(output_root / 'SUMMARY.json', summary)
    if rows:
        with (output_root / 'SUMMARY.csv').open('w', newline='') as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    lines = ['# Pythia evidence ablation', '', SCOPE, '',
             f"Completed {len(plan['tasks']) - len(missing)}/{len(plan['tasks'])} archives; {len(rows)} scorer rows.", '',
             'AUC: seed mean ± sample SD. TPR columns use test ROC, not deployment-calibrated thresholds.',
             'Independent calibration TPR/FPR and paired bootstrap intervals are in SUMMARY.csv / SUMMARY.json.', '',
             '| Source | Method | AUC | ΔAUC vs legacy | TPR@1% FPR | TPR@10% FPR |',
             '|---|---|---:|---:|---:|---:|']
    for row in aggregates:
        a = row['auc']
        spread = f" ± {a['std']:.4f}" if a['std'] is not None else ''
        lines.append(f"| {row['source']} | {row['method']} | {a['mean']:.4f}{spread} | "
                     f"{row['delta_auc_vs_legacy']['mean']:+.4f} | {100*row['roc_tpr_at_1pct_fpr']['mean']:.2f}% | "
                     f"{100*row['roc_tpr_at_10pct_fpr']['mean']:.2f}% |")
    lines += ['', 'No best method, direction or hyperparameter is selected automatically.',
              'Sources are reported separately; full_pile is not a domain macro-average.',
              'Seeds can overlap in document identity. Previously inspected test sets do not provide independent confirmation.', '']
    (output_root / 'SUMMARY.md').write_text('\n'.join(lines))
    return dict(completed=len(plan['tasks']) - len(missing), planned=len(plan['tasks']), scorer_rows=len(rows),
                summary=str(output_root / 'SUMMARY.md'))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument('command', choices=('dry-run', 'run', 'summarize'), nargs='?', default='dry-run')
    parser.add_argument('--input-root', type=Path, default=ROOT / 'artifacts/audits/pythia_mimir_v1')
    parser.add_argument('--output-root', type=Path, default=ROOT / 'artifacts/audits/pythia_evidence_minimal_v1')
    parser.add_argument('--sources', nargs='+', choices=(*SOURCES, 'all'), default=list(DEFAULT_SOURCES))
    parser.add_argument('--seeds', nargs='+', type=int, choices=SEEDS, default=list(SEEDS))
    parser.add_argument('--threads', type=int, default=1)
    parser.add_argument('--bootstrap', type=int, default=200)
    args = parser.parse_args(argv)
    require(args.threads > 0 and args.bootstrap >= 0, 'threads must be positive; bootstrap must be nonnegative')
    require(len(set(args.sources)) == len(args.sources) and len(set(args.seeds)) == len(args.seeds), 'duplicate matrix selections')
    require('all' not in args.sources or args.sources == ['all'], 'use --sources all by itself')
    sources = list(SOURCES) if args.sources == ['all'] else args.sources
    input_root, output_root = args.input_root.resolve(), args.output_root.resolve()
    disjoint_output(output_root, input_root)
    if args.command == 'summarize':
        require((output_root / 'PLAN.json').is_file(), 'no plan exists at the output root')
        require(read_json(output_root / 'PLAN.json').get('schema') == SCHEMA, 'not a standalone evidence experiment')
        with (output_root / '.lock').open('a') as lock:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            print(json.dumps(summarize(output_root), indent=2))
        return
    settings = dict(methods=METHODS, budget=2, feature_order=FEATURE_ORDER, scored_tokens='all archived positions',
                    device='cpu', threads=args.threads, bootstrap=args.bootstrap,
                    implementation_sha256=sha256(__file__),
                    versions=dict(python=platform.python_version(), numpy=np.__version__, scipy=scipy.__version__, torch=torch.__version__))
    tasks = [dict(source=source, seed=seed) for source in sources for seed in args.seeds]
    plan = dict(schema=SCHEMA, scope=SCOPE, input_root=str(input_root), output_root=str(output_root), settings=settings, tasks=tasks)
    if args.command == 'dry-run':
        missing = [str(input_root / 'tasks' / t['source'] / f"seed{t['seed']}" / OLD / 'REPORT.json') for t in tasks
                   if not (input_root / 'tasks' / t['source'] / f"seed{t['seed']}" / OLD / 'REPORT.json').is_file()]
        print(json.dumps(dict(**plan, archives=len(tasks), scorer_rows=len(tasks) * len(METHODS),
                              language_model_queries=0, detector_training_steps=0, missing_reports=missing), indent=2))
        return
    require(all((input_root / 'tasks' / t['source'] / f"seed{t['seed']}" / OLD / 'REPORT.json').is_file() for t in tasks),
            'input reports are missing; inspect dry-run before starting')
    torch.set_num_threads(args.threads)
    output_root.mkdir(parents=True, exist_ok=True)
    with (output_root / '.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        path = output_root / 'PLAN.json'
        if path.exists():
            require(read_json(path) == plan, 'plan changed; use a new output root')
        else:
            require(set(p.name for p in output_root.iterdir()) <= {'.lock'}, 'output directory is already in use')
            write_json(path, plan)
        for task in tasks:
            execute_condition(input_root, output_root, task['source'], task['seed'], settings)
        print(json.dumps(summarize(output_root), indent=2))


if __name__ == '__main__':
    main()
