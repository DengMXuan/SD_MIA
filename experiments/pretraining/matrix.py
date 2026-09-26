"""Resumable current-main GPU worker matrices; dry-run never runs a model."""
import argparse
import json
from pathlib import Path
import sys

from experiments.paths import ROOT, AUDITS, audit_executions
from experiments.shared.core import gpu_pool

SEEDS = (1919, 1949, 1978)
SOURCES = ('arxiv', 'dm_mathematics', 'github', 'hackernews', 'pile_cc',
           'pubmed_central', 'wikipedia_(en)', 'full_pile')
DATA_ROOT = ROOT.parent / 'SD_MIA-pretraining-data'


def _checked_manifest(path, seed, kind, models, counts):
    from experiments.pretraining.data import TOKEN_CONTRACT, sha256
    manifest = json.loads(path.read_text())
    if (manifest.get('kind') != kind or manifest.get('selection_seed') != seed
            or manifest.get('models') != models or manifest.get('counts') != counts
            or manifest.get('token_contract') != TOKEN_CONTRACT or manifest.get('max_tokens') != 512):
        raise ValueError(f'manifest model/seed/count/token contract mismatch: {path}')
    if sha256(path.parent / manifest['records_file']) != manifest['records_sha256']:
        raise ValueError(f'manifest records checksum mismatch: {path}')
    return manifest


def plan(args):
    from experiments.pretraining.data import TARGET, DRAFT, sha256
    from experiments.pretraining.datasets import QWEN_MODELS
    from experiments.shared.models.registry import MODEL_PAIRS
    tasks = []
    for source in (args.sources if args.experiment == 'mimir' else ('wikitection',)):
        for seed in args.seeds:
            if args.experiment == 'mimir':
                manifest = args.data_root / 'mimir/prepared' / source / f'seed{seed}/manifest.json'
                count = 2000 if source == 'full_pile' else 400
                metadata = _checked_manifest(manifest, seed, 'mimir_pretraining_v1',
                    dict(target=TARGET, draft=DRAFT), dict(member=count, nonmember=count, auxiliary=600))
                split = 'none' if source == 'full_pile' else 'ngram_13_0.8'
                if metadata['benchmark'] != f'mimir/{source}/{split}':
                    raise ValueError('MIMIR source/split mismatch')
                task = dict(source=source, seed=seed, manifest=str(manifest),
                            counts=metadata['counts'], data_state='prepared')
            else:
                from experiments.pretraining.temporal_reuse import inspect_inputs
                history = args.data_root / f'qwen3_wikitext_temporal_512_seed{seed}/manifest.json'
                _checked_manifest(history, seed, 'temporal_pretraining_v1', QWEN_MODELS,
                                  dict(member=2000, nonmember=2000, auxiliary=600))
                reference = (args.reference_root or MODEL_PAIRS['qwen3'].run_root) / f'wikitection/epoch1/seed{seed}'
                passport = json.loads((reference / 'results.json').read_text())
                cfg = passport['config']
                if (cfg['seed'] != seed or cfg['data_seed'] != seed or cfg['target_epochs'] != 1
                        or cfg['benchmark'] != 'wikitection'
                        or cfg['target_model'] != QWEN_MODELS['target']['repo_id']
                        or cfg['target_revision'] != QWEN_MODELS['target']['revision']
                        or cfg['draft_model'] != QWEN_MODELS['draft']['repo_id']
                        or cfg['draft_revision'] != QWEN_MODELS['draft']['revision']):
                    raise ValueError('Qwen reference model/data/epoch/seed mismatch')
                shared = (ROOT / passport['data']['shared_split_manifest']).resolve()
                pool = (ROOT / passport['data']['pool_path']).resolve()
                if sha256(shared) != passport['data']['shared_split_sha256']:
                    raise ValueError('reference shared split checksum mismatch')
                _, _, contract = inspect_inputs(history, shared, pool, seed=seed)
                manifest = args.data_root / f'qwen3_temporal_shared_split_v1/seed{seed}/manifest.json'
                if manifest.parent.exists():
                    metadata = _checked_manifest(manifest, seed, 'temporal_pretraining_v1', QWEN_MODELS, contract['counts'])
                    if metadata.get('source_provenance', {}).get('reuse_contract') != contract:
                        raise ValueError('prepared temporal sources changed; use a new data root')
                task = dict(source=source, seed=seed, manifest=str(manifest),
                    history=str(history), shared=str(shared), pool=str(pool), counts=contract['counts'],
                    data_state='prepared' if manifest.exists() else 'prepare_on_run',
                    member_selection='existing seed-matched 2000 historical members',
                    negative_selection='exact SFT nonmember=2000 and audit_auxiliary=600 IDs/order')
            task['output'] = str(args.output_root / source / f'seed{seed}')
            # The pool isolates the assigned GPU; every worker uses ordinal 0.
            task['gpu'], task['detector_epochs'] = 0, args.detector_epochs
            tasks.append(task)
    return tasks


def worker(task, *, prepare_only=False):
    if 'history' in task:
        from experiments.pretraining.temporal_reuse import prepare_from_shared_split
        prepare_from_shared_split(task['history'], task['shared'], task['pool'],
                                  Path(task['manifest']).parent, seed=task['seed'])
    if prepare_only:
        return
    from experiments.pretraining.evaluation import evaluate_main
    report = evaluate_main(task['manifest'], task['output'], seed=task['seed'],
        device=f"cuda:{task['gpu']}", detector_epochs=task['detector_epochs'])
    print(json.dumps(dict(source=task['source'], seed=task['seed'], metrics=report['metrics'])), flush=True)


def summarize(tasks):
    from experiments.shared.audit.artifacts import read_result
    rows = []
    for task in tasks:
        path = Path(task['output']) / 'main_fixed_sparse_positive/REPORT.json'
        row = dict(source=task['source'], seed=task['seed'], report=str(path), state='missing')
        if path.exists():
            try:
                report = read_result(path.parent)
                if (report['settings']['audit_seed'] != task['seed']
                        or report['settings']['detector_epochs'] != task['detector_epochs']
                        or report['evaluation_context']['data_manifest'] != str(Path(task['manifest']).resolve())):
                    raise ValueError('report does not match the selected condition')
                row.update(state='complete', metrics=report['metrics'])
            except (ValueError, OSError, KeyError) as error:
                row.update(state='invalid', error=str(error))
        rows.append(row)
    return dict(conditions=len(rows), complete=sum(row['state'] == 'complete' for row in rows), rows=rows)


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    if argv and argv[0] == '_worker':
        parser = argparse.ArgumentParser()
        parser.add_argument('_worker')
        parser.add_argument('task')
        parser.add_argument('--prepare-only', action='store_true')
        args = parser.parse_args(argv)
        worker(json.loads(args.task), prepare_only=args.prepare_only)
        return
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument('experiment', choices=('mimir', 'temporal'))
    parser.add_argument('command', nargs='?', default='dry-run', choices=('dry-run', 'prepare', 'run', 'summarize'))
    parser.add_argument('--seeds', type=int, nargs='+', choices=SEEDS, default=list(SEEDS))
    parser.add_argument('--sources', nargs='+', choices=SOURCES, help='MIMIR domains only')
    parser.add_argument('--data-root', type=Path, default=DATA_ROOT)
    parser.add_argument('--reference-root', type=Path, help='Qwen training root for temporal split reuse')
    parser.add_argument('--output-root', type=Path)
    gpu_pool.add_arguments(parser)
    parser.add_argument('--detector-epochs', type=int, default=30)
    args = parser.parse_args(argv)
    if args.experiment == 'temporal' and args.sources is not None:
        parser.error('--sources only applies to MIMIR')
    if args.experiment == 'mimir' and args.reference_root is not None:
        parser.error('--reference-root only applies to temporal split reuse')
    args.sources = args.sources or list(SOURCES)
    if (args.detector_epochs < 1 or len(set(args.seeds)) != len(args.seeds)
            or len(set(args.sources)) != len(args.sources)):
        parser.error('choose a nonnegative GPU, positive detector epochs and unique selections')
    args.data_root = args.data_root.resolve()
    if args.reference_root is not None:
        args.reference_root = args.reference_root.resolve()
    default = 'pythia_mimir_v1' if args.experiment == 'mimir' else 'qwen3_temporal_shared_split_v1'
    args.output_root = (args.output_root or AUDITS / default / 'tasks').resolve()
    try:
        scheduling = gpu_pool.configuration(args)
        tasks = plan(args)
    except (OSError, ValueError, KeyError) as error:
        parser.error(str(error))
    if args.command == 'dry-run':
        print(json.dumps(dict(experiment=args.experiment, conditions=len(tasks),
                             scheduling=scheduling, tasks=tasks), indent=2))
        return
    if args.command == 'summarize':
        result = summarize(tasks)
        print(json.dumps(result, indent=2))
        if result['complete'] != result['conditions']:
            raise SystemExit(2)
        return
    jobs = []
    for task in tasks:
        command = [sys.executable, '-B', '-u', '-m', __spec__.name, '_worker', json.dumps(task)]
        if args.command == 'prepare':
            command.append('--prepare-only')
        jobs.append(gpu_pool.Job(f"{task['source']}/seed{task['seed']}", command, task['seed']))
    try:
        rows = gpu_pool.run_jobs(jobs, scheduling=scheduling, cwd=ROOT,
            log_root=args.log_root or audit_executions(args.output_root) / args.command,
            use_cuda=args.command == 'run')
    except KeyboardInterrupt:
        raise SystemExit(130)
    except ValueError as error:
        parser.error(str(error))
    failures = [row for row in rows if row['state'] != 'complete']
    print(json.dumps(dict(failures=failures)), flush=True)
    if failures:
        raise SystemExit(2)


if __name__ == '__main__':
    main()
