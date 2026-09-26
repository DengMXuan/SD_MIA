"""Qwen3 epoch-1 KD ablations, dispatched as one experimental point per GPU."""
import argparse
from contextlib import contextmanager
import csv
import fcntl
import gc
import json
from pathlib import Path
import statistics
import sys

from experiments.paths import ROOT, audit_cache, audit_executions, audit_reports
from experiments.resource_curves.config import AuxiliaryBudget, QUERY_MULTIPLICITIES, calibration_curve, fitting_curve
from experiments.resource_curves.partitions import build_study, save_study
from experiments.resource_curves.qwen_data import (
    BENCHMARKS, SEEDS, ROLE, PREFLIGHT_ROOT, domain_study, materialize, prepare_domain,
    prepare_in_domain, read_condition, read_extension,
)
from experiments.resource_curves.storage import (
    RUN_ROOT, atomic_json, checked_contract, code_fingerprint, digest, file_sha, workspace,
)
from experiments.shared.core import gpu_pool

EXPERIMENTS = ('auxiliary', 'domain', 'query')


def configurations(experiment):
    if experiment == 'auxiliary':
        points = {}
        for curve, budgets in (('fitting', fitting_curve()), ('calibration', calibration_curve())):
            for budget in budgets:
                key = (budget.fitting, budget.calibration)
                points.setdefault(key, dict(budget=budget.to_dict(), multiplicity=2, curves=[]))['curves'].append(curve)
        return list(points.values())
    if experiment in ('domain', 'query'):
        return [dict(budget=AuxiliaryBudget().to_dict(), multiplicity=b, curves=[experiment])
                for b in (QUERY_MULTIPLICITIES if experiment == 'query' else (2,))]
    raise ValueError('unknown experiment')


def condition(task):
    return {key: task[key] for key in ('experiment', 'benchmark', 'seed', 'multiplicity', 'budget', 'curves')} | dict(
        model_pair='qwen3', target_epoch=1, draft_role=ROLE,
        auxiliary_source='newstection' if task['experiment'] == 'domain' else task['benchmark'])


def validate_task(task):
    if (task['experiment'] not in EXPERIMENTS or task['benchmark'] not in BENCHMARKS
            or task['seed'] not in SEEDS or task['detector_epochs'] < 1 or task['bootstrap'] < 0
            or {k: task[k] for k in ('budget', 'multiplicity', 'curves')}
            not in configurations(task['experiment'])
            or (task['experiment'] == 'domain' and task['benchmark'] == 'newstection')):
        raise ValueError('task is outside the fixed Qwen3 epoch-1 KD ablation matrix')
    from experiments.shared.audit.evaluation import _validate_output
    from experiments.resource_curves.storage import CACHE_ROOT, DATA_ROOT
    for path in (task['output'], task['cache'], task['detectors']):
        output = Path(path).resolve()
        _validate_output(Path(task['run_dir']).resolve(), output)
        if output.is_relative_to(ROOT / 'artifacts') and not any(
                output.is_relative_to(root) for root in (RUN_ROOT, CACHE_ROOT, DATA_ROOT)):
            raise ValueError('choose the resource_curves_v1 roots or a new directory outside artifacts')


def plan(args):
    """Read-only input preflight; no tokenizer/GPU initialization or writes."""
    tasks = []
    for benchmark in args.benchmarks:
        for seed in args.seeds:
            run = args.model_root / benchmark / f'epoch1/seed{seed}'
            inputs = read_condition(run, benchmark, seed)
            extension_path = args.extension_root / benchmark / f'seed{seed}/extension/EXTENSION.json'
            if args.experiment == 'auxiliary':
                read_extension(extension_path, inputs)
            donor_run = args.model_root / 'newstection' / f'epoch1/seed{seed}'
            if args.experiment == 'domain':
                donor = read_condition(donor_run, 'newstection', seed)
                domain_study(inputs['shared'], donor['shared'])
            for config in configurations(args.experiment):
                b = config['budget']
                name = f"fit{b['fitting']}_cal{b['calibration']}_b{config['multiplicity']}"
                relative = Path(args.experiment) / benchmark / f'epoch1/seed{seed}' / name
                output = args.output_root / relative
                task = dict(experiment=args.experiment, benchmark=benchmark, seed=seed, **config,
                            id=relative.as_posix(), run_dir=str(run),
                            output=str(output), cache=str(audit_cache(output)),
                            detectors=str(audit_cache(output.parent) / 'detectors'),
                            detector_epochs=args.detector_epochs, bootstrap=args.bootstrap)
                if args.experiment == 'auxiliary':
                    task['extension_path'] = str(extension_path)
                if args.experiment == 'domain':
                    task['donor_run'] = str(donor_run)
                validate_task(task)
                tasks.append(task)
    return tasks


def prepare_task(task):
    """Materialize frozen inputs and save an independently checked study."""
    from experiments.shared.models.loading import local_tokenizer

    validate_task(task)
    inputs = read_condition(task['run_dir'], task['benchmark'], task['seed'])
    cfg = inputs['passport']['config']
    tokenizer = local_tokenizer(Path(task['run_dir']) / 'checkpoints' / ROLE,
                                cfg['draft_model'], cfg['draft_revision'])
    split = materialize(inputs, tokenizer)
    data = dict(files=inputs['files'], target_split_digest=digest(inputs['shared']))
    if task['experiment'] == 'domain':
        donor = read_condition(task['donor_run'], 'newstection', task['seed'])
        donor_split = materialize(donor, tokenizer)
        prepared, study = prepare_domain(inputs['shared'], donor['shared'], split, donor_split, tokenizer)
        data.update(donor_files=donor['files'], donor_split_digest=digest(donor['shared']),
                    cross_domain_audit=study['cross_domain_audit'])
    else:
        shared = inputs['shared']
        extension = read_extension(task['extension_path'], inputs) if task['experiment'] == 'auxiliary' else dict(
            seed=task['seed'], shared_split_digest=digest(shared), records=[])
        # Construct the WHOLE curve before selecting a point. In particular,
        # validation extensions must not shift when fitting grows 800 -> 1200.
        curve = 'fitting' if task['budget']['fitting'] > 400 else 'calibration'
        budgets = (fitting_curve() if curve == 'fitting' else calibration_curve()) \
            if task['experiment'] == 'auxiliary' else (AuxiliaryBudget(),)
        study = build_study(shared, extension, budgets, name=curve if task['experiment'] == 'auxiliary' else 'query')
        study['points'] = [p for p in study['points'] if p['budget'] == task['budget']]
        if len(study['points']) != 1:
            raise ValueError('expected exactly one study point')
        prepared = prepare_in_domain(inputs, split, tokenizer, study, extension)
        if task['experiment'] == 'auxiliary':
            data.update(extension_digest=digest(extension), extension_file=dict(
                path=task['extension_path'], sha256=file_sha(task['extension_path'])))
    save_study(Path(task['cache']) / 'study', study)
    return prepared, study['points'][0], data


@contextmanager
def worker_lock(folder):
    """Own one condition across collection, fitting and reporting."""
    with workspace(folder):
        pass
    with (Path(folder) / '.worker.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        yield


def execute_task(task, *, prepare_only=False, device='cuda:0'):
    validate_task(task)
    with worker_lock(task['output']):
        checked_contract(task['output'], 'REQUEST.json', task)
        prepared, point, data_sources = prepare_task(task)
        if prepare_only:
            return
        import torch
        from experiments.resource_curves.detector import fit_detector, fitting_identity
        from experiments.resource_curves.evaluation import evaluate
        from experiments.resource_curves.observations import (
            collect_observations, collection_contract, fixed_trace, load_observations,
        )
        from experiments.shared.audit.provenance import sources_for
        from experiments.shared.models.loading import load_adapter
        from experiments.shared.protocols.collect_protocol_observations import protocol_prompt_ids

        # Do not put the selected calibration size/study into fitting provenance:
        # all calibration-only points must reuse exactly the same detector.
        sources = dict(frozen=sources_for(Path(task['run_dir']), ['target', ROLE]),
                       data=data_sources, experiment=task['experiment'],
                       warmup='one untimed training auxiliary at the selected B')
        observations_dir = Path(task['cache']) / 'observations'
        if (observations_dir / 'OBSERVATIONS.json').exists():
            _, expected = collection_contract(prepared, adapter_kind='plain', device=device,
                sources=sources, seed=task['seed'], multiplicity=task['multiplicity'])
            observations = load_observations(observations_dir, expected_contract=expected)
        else:
            adapter = load_adapter(Path(task['run_dir']), 'plain', device, ROLE)
            try:
                record = next(r for r in prepared.records if r.record_id == point['partitions']['train'][0])
                fixed_trace(adapter, protocol_prompt_ids(record, prepared.tokenizer), list(record.response_ids),
                            seed=task['seed'], multiplicity=task['multiplicity'])
                observations = collect_observations(prepared, adapter, observations_dir, sources=sources,
                    seed=task['seed'], multiplicity=task['multiplicity'])
            finally:
                del adapter
                gc.collect()
                if torch.device(device).type == 'cuda':
                    torch.cuda.empty_cache()
        # Several GPUs may reach the shared calibration-curve fit together.
        # Serialize only identical fits; keep the library's nonblocking cache
        # lock and never treat a competing fit as a failed experiment.
        detector_root = Path(task['detectors'])
        detector_root.mkdir(parents=True, exist_ok=True)
        fit_key = digest(fitting_identity(observations, point))
        with (detector_root / f'.{fit_key}.lock').open('a') as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            detector = fit_detector(observations, point, detector_root,
                                    seed=task['seed'], device='cpu', epochs=task['detector_epochs'])
        report = evaluate(observations, point, detector, task['output'],
                          bootstrap=task['bootstrap'], metric_seed=task['seed'])
        report.update(condition=condition(task), observation_archive=dict(
            path=str(observations_dir.resolve()),
            metadata_sha256=file_sha(observations_dir / 'OBSERVATIONS.json')))
        atomic_json(Path(task['output']) / 'REPORT.json', report)
        print(json.dumps(dict(id=task['id'], metrics=report['metrics'])), flush=True)


def read_report(task, runtime):
    """Read-only checks for summaries; execution additionally hashes weights."""
    from experiments.shared.audit.artifacts import check_sources_light

    folder = Path(task['output'])
    report = json.loads((folder / 'REPORT.json').read_text())
    contract = report['contract']
    if (json.loads((folder / 'REQUEST.json').read_text()) != task
            or json.loads((folder / 'EVALUATION.json').read_text()) != contract
            or report.get('condition') != condition(task)
            or report['schema'] != 'resource_curve_result_v1'
            or report['method'] != 'main_fixed_sparse_positive'
            or report['multiplicity'] != task['multiplicity']
            or report['auxiliary_budget'] != task['budget']
            or contract['point']['seed'] != task['seed'] or contract['metric_seed'] != task['seed']
            or contract['bootstrap'] != task['bootstrap'] or contract['runtime_sha256'] != runtime
            or report['scores_sha256'] != file_sha(folder / 'scores.npz')):
        raise ValueError('report settings, sources or scores changed')
    detector = Path(report['detector']['path'])
    fit = json.loads(detector.with_name('FIT.json').read_text())
    if (file_sha(detector) != report['detector']['sha256'] or fit['sha256'] != report['detector']['sha256']
            or fit['key'] != contract['fit_key'] or fit['contract']['epochs'] != task['detector_epochs']
            or fit['contract']['seed'] != task['seed'] or fit['contract']['runtime_sha256'] != runtime):
        raise ValueError('detector checksum/settings mismatch')
    archive = report['observation_archive']
    meta_path = Path(archive['path']) / 'OBSERVATIONS.json'
    meta = json.loads(meta_path.read_text())
    if (file_sha(meta_path) != archive['metadata_sha256']
            or file_sha(meta_path.parent / 'observations.npz') != meta['sha256']
            or meta['contract']['sources'] != report['sources']
            or meta['contract']['multiplicity'] != task['multiplicity']
            or meta['contract']['seed'] != task['seed'] or meta['contract']['runtime_sha256'] != runtime):
        raise ValueError('observation archive changed')
    check_sources_light(report['sources']['frozen'])
    data = report['sources']['data']
    files = data['files'] + data.get('donor_files', [])
    if 'extension_file' in data:
        files.append(data['extension_file'])
    for source in files:
        if file_sha(source['path']) != source['sha256']:
            raise ValueError(f"data source changed: {source['path']}")
    return report


def summarize(tasks, *, write_to=None):
    rows, metric_rows = [], []
    runtime = code_fingerprint()
    for task in tasks:
        row = dict(id=task['id'], condition=condition(task), report=str(Path(task['output']) / 'REPORT.json'),
                   state='missing')
        if Path(row['report']).exists():
            try:
                report = read_report(task, runtime)
                row.update(state='complete', metrics=report['metrics'], cost=report['cost'])
                for curve in task['curves']:
                    metric_rows.append(dict(experiment=task['experiment'], curve=curve, benchmark=task['benchmark'],
                        seed=task['seed'], B=task['multiplicity'], fitting=task['budget']['fitting'],
                        calibration=task['budget']['calibration'], **report['metrics'],
                        test_ms_per_record=report['cost']['test_ms_per_record'], report=row['report']))
            except (ValueError, OSError, KeyError, TypeError) as error:
                row.update(state='invalid', error=str(error))
        rows.append(row)
    grouping = ('experiment', 'curve', 'benchmark', 'B', 'fitting', 'calibration')
    groups = {}
    for task in tasks:
        for curve in task['curves']:
            key = (task['experiment'], curve, task['benchmark'], task['multiplicity'],
                   task['budget']['fitting'], task['budget']['calibration'])
            groups.setdefault(key, dict(expected=[], observed=[]))['expected'].append(task['seed'])
    for row in metric_rows:
        groups[tuple(row[k] for k in grouping)]['observed'].append(row)
    aggregates = []
    for key, group in groups.items():
        observed = group['observed']
        aggregate = dict(zip(grouping, key), n_seeds=len(observed), expected_seeds=len(group['expected']))
        # Average metrics across seeds, not their within-seed confidence bounds.
        if observed:
            fields = [k for k, v in observed[0].items() if isinstance(v, (int, float))
                      and k not in (*grouping, 'seed') and not k.startswith('n_')
                      and not k.endswith(('_ci_low', '_ci_high'))]
            for field in fields:
                values = [r[field] for r in observed]
                aggregate[field + '_mean'] = statistics.mean(values)
                aggregate[field + '_std'] = statistics.stdev(values) if len(values) > 1 else None
        aggregates.append(aggregate)
    result = dict(conditions=len(rows), complete=sum(r['state'] == 'complete' for r in rows),
                  expected_curve_rows=sum(len(t['curves']) for t in tasks), metric_rows=metric_rows,
                  aggregates=aggregates, rows=rows)
    if write_to is not None:
        output = Path(write_to)
        output.mkdir(parents=True, exist_ok=True)
        atomic_json(output / 'SUMMARY.json', result)
        for name, values in (('per_seed.csv', metric_rows), ('mean_std.csv', aggregates)):
            # Always replace the CSV, including when no results remain valid.
            keys = list(dict.fromkeys(k for row in values for k in row)) or list(grouping)
            with (output / name).open('w', newline='') as stream:
                writer = csv.DictWriter(stream, fieldnames=keys)
                writer.writeheader()
                writer.writerows(values)
    return result


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    if argv and argv[0] == '_worker':
        parser = argparse.ArgumentParser(allow_abbrev=False)
        parser.add_argument('_worker')
        parser.add_argument('task')
        parser.add_argument('--prepare-only', action='store_true')
        args = parser.parse_args(argv)
        execute_task(json.loads(args.task), prepare_only=args.prepare_only)
        return
    from experiments.shared.models.registry import MODEL_PAIRS
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument('experiment', choices=EXPERIMENTS)
    parser.add_argument('command', nargs='?', default='dry-run', choices=('dry-run', 'prepare', 'run', 'status', 'summarize'))
    parser.add_argument('--benchmarks', nargs='+', choices=BENCHMARKS)
    parser.add_argument('--seeds', type=int, nargs='+', choices=SEEDS, default=list(SEEDS))
    parser.add_argument('--model-root', type=Path, default=MODEL_PAIRS['qwen3'].run_root)
    parser.add_argument('--extension-root', type=Path, default=PREFLIGHT_ROOT)
    parser.add_argument('--output-root', type=Path, default=RUN_ROOT)
    parser.add_argument('--detector-epochs', type=int, default=30)
    parser.add_argument('--bootstrap', type=int, default=200)
    gpu_pool.add_arguments(parser)
    args = parser.parse_args(argv)
    args.benchmarks = args.benchmarks or (['wikitection', 'arxivtection'] if args.experiment == 'domain' else list(BENCHMARKS))
    if (len(set(args.seeds)) != len(args.seeds) or len(set(args.benchmarks)) != len(args.benchmarks)
            or args.detector_epochs < 1 or args.bootstrap < 0
            or (args.experiment == 'domain' and 'newstection' in args.benchmarks)):
        parser.error('choose unique selections, valid counts and only Wiki/Arxiv for the domain experiment')
    for key in ('model_root', 'extension_root', 'output_root'):
        setattr(args, key, getattr(args, key).resolve())
    try:
        scheduling = gpu_pool.configuration(args)
        tasks = plan(args)
    except (OSError, ValueError, KeyError) as error:
        parser.error(str(error))
    if args.command == 'dry-run':
        print(json.dumps(dict(experiment=args.experiment, conditions=len(tasks),
            curve_rows=sum(len(t['curves']) for t in tasks), model_pair='qwen3', target_epoch=1,
            draft_role=ROLE, scheduling=scheduling,
            seed_mapping=[dict(condition=s, selection=s, auxiliary_split=s, collection=s,
                               detector=s, metrics=s, pythonhashseed=s) for s in args.seeds],
            collection='separate measured archive per B, all auxiliary/test roles use that B',
            tasks=tasks), indent=2))
        return
    summary_dir = audit_reports(args.output_root) / args.experiment
    if args.command in ('status', 'summarize'):
        result = summarize(tasks, write_to=summary_dir if args.command == 'summarize' else None)
        print(json.dumps(result, indent=2))
        if result['complete'] != len(tasks):
            raise SystemExit(2)
        return
    jobs = []
    for task in tasks:
        command = [sys.executable, '-B', '-u', '-m', __spec__.name, '_worker', json.dumps(task)]
        if args.command == 'prepare':
            command.append('--prepare-only')
        jobs.append(gpu_pool.Job(task['id'], command, task['seed']))
    try:
        rows = gpu_pool.run_jobs(jobs, scheduling=scheduling, cwd=ROOT,
            log_root=args.log_root or audit_executions(args.output_root) / args.experiment / args.command,
            use_cuda=args.command == 'run')
    except KeyboardInterrupt:
        raise SystemExit(130)
    failures = [r for r in rows if r['state'] != 'complete']
    if args.command == 'run':
        result = summarize(tasks, write_to=summary_dir)
        print(json.dumps(dict(complete=result['complete'], conditions=len(tasks), summary=str(summary_dir))))
        if result['complete'] != len(tasks):
            raise SystemExit(2)
    if failures:
        raise SystemExit(2)


if __name__ == '__main__':
    main()
