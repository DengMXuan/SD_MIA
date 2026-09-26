"""Frozen, seed-matched Qwen inputs for resource and domain ablations."""
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from experiments.paths import ROOT
from experiments.resource_curves.auxiliary import _read_pool, _TokenIndex, extension_records
from experiments.resource_curves.config import AuxiliaryBudget, condition_seed
from experiments.resource_curves.partitions import base_partitions, build_study, prepare_study
from experiments.resource_curves.storage import digest, file_sha

BENCHMARKS = ('wikitection', 'newstection', 'arxivtection')
SEEDS = (1919, 1949, 1978)
ROLE = 'draft_auxiliary_distilled'
PREFLIGHT_ROOT = ROOT.parent / 'SD_MIA-pretraining-data/resource_curves'


def read_condition(run_dir, benchmark, seed):
    """No model/tokenizer loading, file creation, or full weight hashing."""
    from experiments.shared.models.readiness import ready
    from experiments.shared.models.registry import split_manifest

    if benchmark not in BENCHMARKS or seed not in SEEDS:
        raise ValueError('unsupported benchmark or condition seed')
    run_dir = Path(run_dir).resolve()
    task = dict(run_dir=str(run_dir), model_pair='qwen3', kind='main', draft_role=ROLE,
                condition=dict(benchmark=benchmark, epoch=1, condition_seed=seed))
    valid, reason = ready(task)
    if not valid:
        raise ValueError(f'{run_dir}: {reason}')
    passport = json.loads((run_dir / 'results.json').read_text())
    if 'privacy' in passport or (run_dir / 'DP_REQUEST.json').exists():
        raise ValueError('resource ablations require the ordinary, non-DP checkpoints')
    shared_path = split_manifest(passport).resolve()
    shared = json.loads(shared_path.read_text())
    condition_seed(shared.get('seed'), seed)
    expected = dict(member=2000, nonmember=2000, auxiliary=2000, audit_auxiliary=600)
    ids = [r['record_id'] for rows in shared['splits'].values() for r in rows]
    if (shared.get('schema_version') != 3 or shared.get('benchmark') != benchmark
            or shared.get('counts') != expected or len(ids) != len(set(ids))
            or any(len(shared['splits'][role]) != count for role, count in expected.items())):
        raise ValueError('frozen shared split count/identity mismatch')
    pool = (ROOT / passport['data']['pool_path']).resolve()
    _, pool_sha = _read_pool(benchmark, pool)
    if pool_sha != shared['pool_sha256'] or pool_sha != passport['data']['pool_sha256']:
        raise ValueError('pool differs from the training passport/shared split')
    files = [run_dir / 'results.json', shared_path, shared_path.with_suffix('.audit.json'),
             pool, pool.with_suffix('.manifest.json')]
    return dict(run_dir=run_dir, passport=passport, shared=shared, shared_path=shared_path,
                pool=pool, files=[dict(path=str(p), sha256=file_sha(p)) for p in files])


def read_extension(path, inputs):
    """Use the existing all-tokenizer preflight, rechecking its frozen sources."""
    extension = json.loads(Path(path).read_text())
    shared = inputs['shared']
    expected = dict(schema='resource_auxiliary_extension_v1', benchmark=shared['benchmark'],
                    seed=shared['seed'], shared_split_digest=digest(shared),
                    shared_split_sha256=file_sha(inputs['shared_path']),
                    pool_sha256=shared['pool_sha256'], token_band=shared['token_band'],
                    tokenizer_sources=sorted(shared['tokenizer_sources']),
                    excluded_roles=sorted(shared['splits']),
                    near_duplicate_ngram=13, near_duplicate_threshold=.5)
    if (any(extension.get(k) != v for k, v in expected.items())
            or Path(extension['pool_path']).resolve() != inputs['pool']
            or Path(extension['shared_split_path']).resolve() != inputs['shared_path']
            or len(extension['records']) != 1000):
        raise ValueError('extension preflight does not match this frozen condition')
    # Check disjointness/size even in dry-run, without loading tokenizers.
    build_study(shared, extension, (AuxiliaryBudget(400, 1200),), name='check')
    return extension


def materialize(inputs, tokenizer):
    from experiments.shared.core.scoring_common import _verify_controlled_split_against_run
    from experiments.shared.data.splits import build_controlled_split_from_shared_manifest

    cfg = inputs['passport']['config']
    source = f"{cfg['draft_model']}@{cfg['draft_revision']}"
    split = build_controlled_split_from_shared_manifest(
        cfg['benchmark'], inputs['pool'], tokenizer, inputs['shared_path'], source)
    _verify_controlled_split_against_run(split, inputs['run_dir'], inputs['passport'])
    return split


def prepared_records(records, roles, tokenizer, seed):
    roles = np.asarray(roles)
    return SimpleNamespace(records=list(records), tokenizer=tokenizer, condition_seed=seed,
                           record_ids=np.asarray([r.record_id for r in records]),
                           record_roles=roles, labels=(roles == 'member').astype(np.int64))


def prepare_in_domain(inputs, split, tokenizer, study, extension):
    base = prepared_records(split.audit_auxiliary + split.members + split.nonmembers,
        ['audit_auxiliary'] * len(split.audit_auxiliary) + ['member'] * len(split.members)
        + ['nonmember'] * len(split.nonmembers), tokenizer, inputs['shared']['seed'])
    cfg = inputs['passport']['config']
    extra = extension_records(extension, tokenizer, f"{cfg['draft_model']}@{cfg['draft_revision']}") \
        if extension['records'] else []
    return prepare_study(base, extra, extension, study)


def domain_study(target, donor):
    """News fitting/validation/calibration; original Wiki/Arxiv test and models."""
    seed = condition_seed(target['seed'], donor.get('seed'))
    if target['benchmark'] not in ('wikitection', 'arxivtection') or donor['benchmark'] != 'newstection':
        raise ValueError('only News -> Wiki/Arxiv domain shifts are supported')
    parts = base_partitions(donor)
    parts['test'] = base_partitions(target)['test']
    target_ids = {r['record_id'] for rows in target['splits'].values() for r in rows}
    donor_aux = {r['record_id'] for r in donor['splits']['audit_auxiliary']}
    if len(donor_aux) != 600 or donor_aux.intersection(target_ids):
        raise ValueError('News audit auxiliaries overlap a target model/audit assignment')
    return dict(schema='resource_domain_study_v1', name='news_to_' + target['benchmark'], seed=seed,
                target_split_digest=digest(target), donor_split_digest=digest(donor),
                auxiliary_source='newstection', target_benchmark=target['benchmark'],
                points=[dict(seed=seed, budget=AuxiliaryBudget().to_dict(), partitions=parts)])


def check_domain_overlap(target, donor, target_split, donor_split):
    """Reject raw/token/13-gram duplicates against every target assignment.

    No replacement sampling: keep the exact seed-matched News audit pool.
    Within-domain disjointness is checked by the original materializer.
    """
    anchors = (target_split.members + target_split.nonmembers + target_split.draft_auxiliary
               + target_split.audit_auxiliary)
    index = _TokenIndex()
    for record in anchors:
        index.add(record.response_ids)
    raw = {r['text_sha256'] for rows in target['splits'].values() for r in rows}
    donor_rows = {r['record_id']: r for r in donor['splits']['audit_auxiliary']}
    for record in donor_split.audit_auxiliary:
        if donor_rows[record.record_id]['text_sha256'] in raw or not index.accepts(record.response_ids):
            raise ValueError(f'News auxiliary duplicates a target assignment: {record.record_id}')
    return dict(gate='PASS', target_anchors=len(anchors), news_auxiliaries=len(donor_rows),
                ngram=13, threshold=.5, tokenizer='frozen Qwen3 tokenizer')


def prepare_domain(target, donor, target_split, donor_split, tokenizer):
    study = domain_study(target, donor)
    study['cross_domain_audit'] = check_domain_overlap(target, donor, target_split, donor_split)
    prepared = prepared_records(donor_split.audit_auxiliary + target_split.members + target_split.nonmembers,
        ['audit_auxiliary'] * len(donor_split.audit_auxiliary) + ['member'] * len(target_split.members)
        + ['nonmember'] * len(target_split.nonmembers), tokenizer, study['seed'])
    from experiments.resource_curves.partitions import partition_indices
    partition_indices(vars(prepared), study['points'][0])
    return prepared, study
