"""Bind seed-matched historical members to an existing WikiTection audit split."""
from datetime import datetime
import hashlib
import json
from pathlib import Path
import shutil
import tempfile

import numpy as np

from experiments.pretraining.data import load_evaluation, sha256
from experiments.pretraining.datasets import _pool
from experiments.shared.data.data import _hash_ids
from experiments.shared.data.pools import NearDuplicateIndex


def inspect_inputs(historical_manifest, shared_manifest, recent_pool, *, seed):
    """Read-only identities/count checks; no tokenizer, weights or output writes."""
    history_path, shared_path, pool = map(lambda p: Path(p).resolve(),
                                         (historical_manifest, shared_manifest, recent_pool))
    history = json.loads(history_path.read_text())
    shared = json.loads(shared_path.read_text())
    if (type(seed) is not int or history.get('selection_seed') != seed
            or shared.get('seed') != seed):
        raise ValueError('historical and shared split seeds must match the condition')
    if (history.get('kind') != 'temporal_pretraining_v1'
            or history.get('membership_verified') is not False
            or shared.get('schema_version') != 3 or shared.get('benchmark') != 'wikitection'):
        raise ValueError('expected temporal history and a four-role WikiTection split')
    if set(shared['splits']) != {'member', 'nonmember', 'auxiliary', 'audit_auxiliary'}:
        raise ValueError('shared split must have all four data roles')
    n, aux = history['counts']['member'], history['counts']['auxiliary']
    if (n < 1 or aux < 1 or len(shared['splits']['nonmember']) != n
            or len(shared['splits']['audit_auxiliary']) != aux):
        raise ValueError('frozen nonmember/audit auxiliary counts do not match temporal counts')
    ids = [r['record_id'] for values in shared['splits'].values() for r in values]
    if len(set(ids)) != len(ids):
        raise ValueError('shared split roles overlap')
    records = history_path.parent / history['records_file']
    if sha256(records) != history['records_sha256']:
        raise ValueError('historical records checksum mismatch')
    _, meta = _pool(pool)
    if meta['jsonl_sha256'] != shared['pool_sha256']:
        raise ValueError('shared split and recent pool differ')
    if shared['token_band']['max_tokens'] != history['max_tokens']:
        raise ValueError('shared and historical token truncation limits differ')
    sources = [history_path, records, shared_path, pool, pool.with_suffix('.manifest.json')]
    contract = dict(schema='temporal_reuse_shared_split_v1', seed=seed,
                    counts=dict(member=n, nonmember=n, auxiliary=aux),
                    files=[dict(path=str(p), sha256=sha256(p)) for p in sources],
                    member_selection='reuse exact historical member rows from the matching seed',
                    nonmember_selection='shared.splits.nonmember in frozen order',
                    auxiliary_selection='shared.splits.audit_auxiliary in frozen order')
    return history, shared, contract


def prepare_from_shared_split(historical_manifest, shared_manifest, recent_pool, output_dir, *, seed):
    """Keep historical member IDs and both negative roles, including their order.

    Existing frozen historical selection already applies the requested public
    seed. No further random draw is made here. Reject overlaps rather than
    replace any anchored sample. The source temporal manifest is never edited.
    """
    history, shared, contract = inspect_inputs(historical_manifest, shared_manifest, recent_pool, seed=seed)
    output = Path(output_dir).resolve()
    if output.exists():
        manifest = output / 'manifest.json'
        saved = json.loads(manifest.read_text())
        if saved.get('source_provenance', {}).get('reuse_contract') != contract:
            raise ValueError('temporal reuse sources changed; use a new data directory')
        fields = ('kind', 'selection_seed', 'models', 'tokenizer_sha256',
                  'token_contract', 'max_tokens', 'membership_verified', 'model_release')
        if (any(saved.get(key) != history.get(key) for key in fields)
                or saved.get('counts') != contract['counts']
                or saved.get('benchmark') != 'wiki_temporal/wikitext103_shared_split'):
            raise ValueError('saved temporal model/seed/data contract changed')
        load_evaluation(manifest, verify_draft=True)
        return manifest
    evaluation = load_evaluation(Path(historical_manifest), verify_draft=True)
    tokenizer = evaluation.tokenizer
    history_path = Path(historical_manifest).resolve()
    old_rows = [json.loads(line) for line in (history_path.parent / history['records_file']).read_text().splitlines()]
    members = [row for row in old_rows if row['group'] == 'member']
    documents, _ = _pool(recent_pool)
    by_id = {row['record_id']: row for row in documents}
    if len(by_id) != len(documents):
        raise ValueError('duplicate pool document IDs')
    rows = list(members)
    token_hashes = {row['token_hash'] for row in members}
    member_near = NearDuplicateIndex()
    for row in members:
        member_near.add(tokenizer.decode(row['token_ids'], skip_special_tokens=False))
    release = datetime.fromisoformat(history['model_release'][:10]).date()
    for group, role in (('nonmember', 'nonmember'), ('auxiliary', 'audit_auxiliary')):
        for entry in shared['splits'][role]:
            document = by_id[entry['record_id']]
            text = document['text']
            if hashlib.sha256(text.encode()).hexdigest() != entry['text_sha256']:
                raise ValueError('frozen shared document text changed')
            if datetime.fromisoformat(document['creation_timestamp'][:10]).date() <= release:
                raise ValueError('fixed nonmembers must postdate the model release')
            ids = list(tokenizer(text, add_special_tokens=False, truncation=True,
                                 max_length=history['max_tokens']).input_ids)
            if not shared['token_band']['min_tokens'] <= len(ids) <= history['max_tokens']:
                raise ValueError('fixed nonmember violates its frozen token band')
            token_hash = _hash_ids(ids)
            if token_hash in token_hashes:
                raise ValueError('exact token overlap in fixed temporal roles')
            if member_near.is_duplicate(tokenizer.decode(ids, skip_special_tokens=False)):
                raise ValueError('fixed recent document nearly duplicates a historical member')
            token_hashes.add(token_hash)
            rows.append(dict(record_id=f'temporal:{document["record_id"]}',
                source=document['source'], source_record_id=document['record_id'],
                group=group, label=0, text=text, token_ids=ids, token_hash=token_hash,
                temporal_metadata={k: document[k] for k in ('title', 'creation_timestamp') if k in document}))
    if len({row['record_id'] for row in rows}) != len(rows):
        raise ValueError('temporal document IDs overlap')
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix='.temporal-reuse-', dir=output.parent))
    try:
        records = temporary / 'records.jsonl'
        records.write_text(''.join(json.dumps(row, ensure_ascii=False) + '\n' for row in rows))
        lengths = {group: [len(row['token_ids']) for row in rows if row['group'] == group]
                   for group in ('member', 'nonmember', 'auxiliary')}
        manifest = {**history, 'benchmark': 'wiki_temporal/wikitext103_shared_split',
            'counts': contract['counts'], 'records_sha256': sha256(records), 'records_file': 'records.jsonl',
            'min_tokens': min(map(min, lengths.values())),
            'historical_min_tokens': history['min_tokens'],
            'source_provenance': {**history['source_provenance'], 'reuse_contract': contract},
            'filtering': dict(member=history['filtering']['member'],
                recent=dict(selection='frozen shared IDs; no resampling or filtering',
                            shared_token_band=shared['token_band'])),
            'token_lengths': {g: dict(min=min(v), max=max(v), mean=float(np.mean(v))) for g, v in lengths.items()},
            'caveats': [*history.get('caveats', []),
                'recent IDs and order reuse the controlled SFT nonmember/audit auxiliary split',
                'recent lengths retain the shared token band; historical members are unchanged',
                'recent/recent near-deduplication inherits the frozen shared split; cross-label near overlaps rejected']}
        (temporary / 'manifest.json').write_text(json.dumps(manifest, indent=2))
        temporary.rename(output)
    finally:
        if temporary.exists():
            shutil.rmtree(temporary)
    return output / 'manifest.json'
