"""Load the target tokenizer and the exact four-role assignment used in SFT."""
import json
from pathlib import Path

import numpy as np

from experiments.paths import ROOT
from experiments.shared.core.deployment_archive import sha256_file
from experiments.shared.data.splits import build_controlled_split_from_shared_manifest, pool_path
from experiments.shared.models.loading import local_tokenizer
from experiments.shared.models.registry import identify_pair, split_manifest
from experiments.shared.training.generalization import load_run_config


def training_tokenizer_source(spec):
    # Plain SFT used its shared draft tokenizer. Head SFT used the target's.
    return (f'{spec.target}@{spec.target_revision}' if spec.is_head
            else f'{spec.draft}@{spec.draft_revision}')


def frozen_split(run_dir):
    run_dir = Path(run_dir)
    artifact = json.loads((run_dir / 'results.json').read_text())
    spec, cfg = identify_pair(artifact), load_run_config(run_dir)
    if cfg.seed != cfg.data_seed:
        raise ValueError('training and data seeds must match')
    manifest = split_manifest(artifact)
    shared = json.loads(manifest.read_text())
    checksum = sha256_file(manifest)
    if shared['seed'] != cfg.seed or shared['benchmark'] != cfg.benchmark:
        raise ValueError('frozen split belongs to a different condition')
    source = training_tokenizer_source(spec)
    attestation = json.loads(manifest.with_suffix('.audit.json').read_text())['tokenizers'][source]
    saved = artifact['stages']['target']['data'] if spec.is_head else artifact['data']
    if (saved['shared_split_sha256'] != checksum
            or attestation['shared_split_sha256'] != checksum
            or attestation['cross_split_ngram_audit']['gate'] != 'PASS'):
        raise ValueError('frozen split or target tokenizer attestation differs from training')
    tokenizer = local_tokenizer(run_dir / 'checkpoints/target', spec.target, spec.target_revision)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    pool = cfg.pool_path or pool_path(cfg.benchmark)
    pool = pool if pool.is_absolute() else ROOT / pool
    split = build_controlled_split_from_shared_manifest(cfg.benchmark, pool, tokenizer, manifest, source)
    if split.metadata['cross_split_ngram_audit']['gate'] != 'PASS':
        raise ValueError('target tokenizer split audit failed')
    if not spec.is_head:
        # Do not silently relabel a draft-tokenizer attestation as target-owned.
        # Check the actual target-tokenized records against all frozen training
        # token hashes, including prompts, before using that shared assignment.
        for role, rows in (('members', split.members), ('nonmembers', split.nonmembers),
                           ('auxiliary', split.draft_auxiliary), ('audit_auxiliary', split.audit_auxiliary)):
            saved_rows = artifact['records'][role]
            keys = ('record_id', 'response_hash', 'prompt_hash')
            if [[getattr(r, k) for k in keys] for r in rows] != [[r[k] for k in keys] for r in saved_rows]:
                raise ValueError(f'target tokenization differs from training for {role}')
    split.metadata['evaluation_tokenizer_source'] = f'{spec.target}@{spec.target_revision}'
    split.metadata['tokenizer_validation'] = ('target attestation' if spec.is_head
                                              else 'target prompt/response hashes equal all saved training records')
    return cfg, tokenizer, split


def sample_classes(split, per_class, seed):
    """Same seed/order pairs raw document IDs across model pairs and studies."""
    classes = {'member': split.members, 'nonmember': split.nonmembers,
               'auxiliary': split.draft_auxiliary}
    if per_class < 1 or any(len(rows) < per_class for rows in classes.values()):
        raise ValueError('not enough distinct records for the requested sample')
    # Restart from the condition seed in each role, matching generalization.
    return {role: [rows[int(i)] for i in np.random.default_rng(seed).permutation(len(rows))[:per_class]]
            for role, rows in classes.items()}
