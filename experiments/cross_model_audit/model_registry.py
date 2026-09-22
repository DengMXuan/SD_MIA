"""Frozen model identities, draft roles and passport validation for audit workers."""
from dataclasses import dataclass
import json
from pathlib import Path

from safetensors import safe_open

from experiments.paths import ROOT, TRAINING
from experiments.sd_membership_sft.core.deployment_archive import sha256_file
from experiments.sd_membership_sft.drafts.common import PAIR_MODELS

PLAIN_ROLES = ('draft_auxiliary_distilled', 'draft_member_sft')
HEAD_ROLES = ('auxiliary_head', 'member_head')


@dataclass(frozen=True)
class ModelPair:
    name: str
    adapter: str
    target: str
    draft: str
    target_revision: str
    draft_revision: str

    @property
    def roles(self):
        return PLAIN_ROLES if self.adapter == 'plain' else HEAD_ROLES

    @property
    def run_root(self):
        track = 'model_pairs' if self.adapter == 'plain' else 'speculator_matrix'
        return TRAINING / track / self.name


def _registry():
    path = ROOT / 'experiments/sd_membership_sft/scripts/model_pair_revisions.env'
    revisions = dict(line.split('=', 1) for line in path.read_text().splitlines()
                     if line.strip() and not line.startswith('#'))
    pairs = {}
    for name, prefix, target, draft in (
        ('qwen3', 'QWEN', 'Qwen/Qwen3-8B-Base', 'Qwen/Qwen3-1.7B-Base'),
        ('gemma4', 'GEMMA', 'google/gemma-4-12B', 'google/gemma-4-E2B'),
    ):
        pairs[name] = ModelPair(name, 'plain', target, draft,
                               revisions[prefix + '_TARGET_REVISION'], revisions[prefix + '_DRAFT_REVISION'])
    for name, spec in PAIR_MODELS.items():
        pairs[name] = ModelPair(name, spec['kind'], spec['target'], spec['speculator'],
                               spec['target_revision'], spec['speculator_revision'])
    return pairs


MODEL_PAIRS = _registry()


def identify_pair(artifact):
    """Resolve a saved condition using model names and pinned base revisions."""
    cfg = artifact['config']
    matches = [spec for spec in MODEL_PAIRS.values()
               if (cfg.get('target_model'), cfg.get('draft_model'),
                   cfg.get('target_revision'), cfg.get('draft_revision')) ==
               (spec.target, spec.draft, spec.target_revision, spec.draft_revision)]
    if len(matches) != 1:
        raise ValueError('checkpoint passport does not identify a registered model pair')
    spec = matches[0]
    if spec.adapter != 'plain' and artifact.get('protocol_track', {}).get('pair') != spec.name:
        raise ValueError('head pair differs from model identities')
    return spec


def pair_for(task):
    return MODEL_PAIRS[task.get('model_pair', 'qwen3')]


def split_manifest(artifact):
    name = artifact.get('protocol_track', {}).get('shared_raw_split') or artifact['data']['shared_split_manifest']
    path = Path(name)
    return path if path.is_absolute() else ROOT / path


def validate_weights(folder):
    if not (folder / 'config.json').is_file():
        raise FileNotFoundError(f'checkpoint config missing: {folder}')
    shards = list(folder.glob('*.safetensors'))
    if not shards:
        raise FileNotFoundError(f'checkpoint weights missing: {folder}')
    index = folder / 'model.safetensors.index.json'
    if index.exists():
        expected = set(json.loads(index.read_text())['weight_map'].values())
        if not expected.issubset({p.name for p in shards}):
            raise ValueError(f'checkpoint shards incomplete: {folder}')
    for shard in shards:
        with safe_open(shard, framework='np') as weights:
            if not list(weights.keys()):
                raise ValueError(f'empty checkpoint: {shard}')


def validate_identity(task, artifact):
    spec, cfg, condition = pair_for(task), artifact['config'], task['condition']
    expected = dict(benchmark=condition['benchmark'], target_epochs=condition['epoch'],
                    seed=condition['condition_seed'], data_seed=condition['condition_seed'],
                    target_model=spec.target, draft_model=spec.draft,
                    target_revision=spec.target_revision, draft_revision=spec.draft_revision,
                    n_per_class=2000, n_aux=2000, n_audit_aux=600)
    if any(cfg.get(key) != value for key, value in expected.items()):
        raise ValueError('training condition/model/revision identity mismatch')
    if task['kind'] == 'main' and task['draft_role'] not in spec.roles:
        raise ValueError('draft role does not belong to model pair')


def validate_head_passport(task, artifact):
    """Baseline checks target only; each main branch checks its own head stage."""
    validate_identity(task, artifact)
    spec, cfg, track = pair_for(task), artifact['config'], artifact['protocol_track']
    if track['pair'] != spec.name or not track['target_frozen_before_heads']:
        raise ValueError('head pair/frozen-target contract mismatch')
    manifest = split_manifest(artifact)
    checksum = sha256_file(manifest)
    shared = json.loads(manifest.read_text())
    if shared['benchmark'] != cfg['benchmark'] or shared['seed'] != cfg['data_seed']:
        raise ValueError('shared split condition mismatch')
    expected_counts = dict(member=2000, nonmember=2000, auxiliary=2000, audit_auxiliary=600)
    if shared['counts'] != expected_counts:
        raise ValueError('shared split counts mismatch')
    ids = [entry['record_id'] for role, count in expected_counts.items()
           for entry in shared['splits'][role]]
    if (len(ids) != 6600 or len(set(ids)) != 6600
            or any(len(shared['splits'][role]) != count for role, count in expected_counts.items())):
        raise ValueError('shared split overlaps or has missing records')
    audit = json.loads(manifest.with_suffix('.audit.json').read_text())
    attestation = audit['tokenizers'][f'{spec.target}@{spec.target_revision}']
    if attestation['shared_split_sha256'] != checksum or attestation['cross_split_ngram_audit']['gate'] != 'PASS':
        raise ValueError('shared split tokenizer audit is stale')
    run = Path(task['run_dir'])
    stages = [('target', run / 'checkpoints/target')]
    if task['kind'] == 'main':
        role = task['draft_role']
        stages.append(('aux_head' if role == 'auxiliary_head' else 'member_head', run / 'heads' / role))
    for stage, folder in stages:
        saved = artifact['stages'][stage]
        if (saved['data']['shared_split_sha256'] != checksum
                or saved['data']['split_seed'] != cfg['data_seed']
                or saved['data']['counts'] != expected_counts):
            raise ValueError(f'{stage} used a different shared split')
        marker = json.loads((folder / '_COMPLETE.json').read_text())
        if marker.get('status') != 'complete' or marker.get('stage') != stage:
            raise ValueError(f'invalid completion marker: {folder}')
        if stage != 'target':
            variant = 'aux' if stage == 'aux_head' else 'member'
            if (marker.get('variant') != variant or not marker.get('target_frozen')
                    or marker.get('seed') != cfg['seed']
                    or marker.get('initialized_from_revision') != spec.draft_revision
                    or Path(marker['target_checkpoint']).resolve() != (run / 'checkpoints/target').resolve()):
                raise ValueError(f'head variant/target identity mismatch: {folder}')
        validate_weights(folder)
    return True, 'ready'
