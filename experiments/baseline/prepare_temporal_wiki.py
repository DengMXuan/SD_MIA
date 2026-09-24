"""Prepare a temporal Wiki audit using the existing tokenizer selection rules.

The historical positive labels are temporal proxies, never verified membership.
No model weights are loaded and the existing nonmember split is preserved.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import tempfile
from pathlib import Path

from experiments.shared.data.data import _hash_ids
from experiments.shared.training.generalization import load_run_config
from experiments.shared.data.pools import NearDuplicateIndex, _write_pool
from experiments.shared.data.splits import build_split, _cross_split_ngram_audit, pool_path
from experiments.baseline.data import _resolve, _target_tokenizer, load_audit_records


def read_pool(path):
    payload = path.read_bytes()
    manifest = json.loads(path.with_suffix('.manifest.json').read_text())
    if hashlib.sha256(payload).hexdigest() != manifest['jsonl_sha256']:
        raise ValueError(f'pool hash mismatch: {path}')
    return [json.loads(line) for line in payload.decode().splitlines() if line], manifest


def prepare(historical_pools, run_dir, output_dir, count=2000):
    run_dir = _resolve(run_dir)
    cfg = load_run_config(run_dir)
    if cfg.benchmark != 'wikitection':
        raise ValueError('nonmember run must be WikiTection')
    tokenizer = _target_tokenizer(run_dir, cfg)
    _, nonmembers, auxiliary, split_metadata = load_audit_records(run_dir, cfg, tokenizer, None)
    if len(nonmembers) != count:
        raise ValueError(f'expected {count} saved nonmembers, found {len(nonmembers)}')
    documents, sources, seen = [], [], set()
    near = NearDuplicateIndex()
    for path in historical_pools:
        rows, manifest = read_pool(path)
        if manifest.get('label_semantics') != 'presumed_member_temporal_proxy':
            raise ValueError(f'not a historical temporal pool: {path}')
        sources.append(dict(path=str(path), sha256=manifest['jsonl_sha256']))
        for row in rows:
            if (not row['creation_timestamp'].startswith('2023-') or
                not row['snapshot_timestamp'].startswith('2023-') or
                int(row['snapshot_revision']) <= 0):
                raise ValueError('historical sample has missing/non-2023 provenance')
            if row['text_sha256'] in seen or (len(historical_pools) > 1 and near.is_duplicate(row['text'])):
                continue
            seen.add(row['text_sha256'])
            near.add(row['text'])
            documents.append(row)
    # Request all selected records in the auxiliary output solely to reuse
    # exactly the original banding, truncation and token/template dedup gates.
    # These temporary split positions do not define the exported labels.
    with tempfile.TemporaryDirectory(prefix='sd-mia-wiki-selection-') as temporary:
        path = Path(temporary) / 'pool.jsonl'
        _write_pool(path, documents, dict(benchmark='wikitection'))
        _, _, positives, selection = build_split('wikitection', path, tokenizer, 0, count, cfg.data_seed)
    negative_records = [row.record for row in nonmembers]
    audit = _cross_split_ngram_audit([positives, negative_records, auxiliary])
    negative_pool = _resolve(Path(cfg.pool_path or pool_path('wikitection')))
    negative_documents, negative_manifest = read_pool(negative_pool)
    by_hash = {}
    for document in documents + negative_documents:
        ids = tokenizer(document['text'], add_special_tokens=False, truncation=True, max_length=512).input_ids
        by_hash.setdefault(_hash_ids(list(ids)), document)
    output_dir.mkdir(parents=True, exist_ok=True)
    output = output_dir / 'audit.jsonl'
    if output.exists():
        raise FileExistsError(output)
    rows = []
    for label, records in [(1, positives), (0, negative_records)]:
        for record in records:
            document = by_hash[record.response_hash]
            rows.append({**document, 'audit_record_id': record.record_id,
                         'label': label, 'membership_verified': False,
                         'membership_status': 'presumed_member_temporal_proxy' if label else 'post_release_nonmember_proxy',
                         'response_ids': list(record.response_ids), 'response_hash': record.response_hash,
                         'prompt_ids': list(record.prompt_ids or ()), 'prompt_text': record.prompt_text})
    # This is an audit with externally defined labels, not an SFT pool to split.
    _write_pool(output, rows, dict(
        benchmark='wiki_temporal_proxy', n_presumed_member=count, n_nonmember=count,
        membership_verified=False, historical_sources=sources,
        nonmember_run=str(run_dir), nonmember_pool_sha256=negative_manifest['jsonl_sha256'],
        nonmember_split_metadata=split_metadata,
        token_band=dict(min_tokens=128, max_tokens=512),
        selection={k: selection[k] for k in ('band_dropped_documents', 'skipped_token_duplicates', 'skipped_template_overlap')},
        cross_split_ngram_audit=audit,
        rendering='historical main revision; current templates and metadata may affect rendering/selection',
    ))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--historical-pools', type=Path, nargs='+', required=True)
    parser.add_argument('--nonmember-run', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    args = parser.parse_args()
    prepare(args.historical_pools, args.nonmember_run, args.output_dir)


if __name__ == '__main__':
    main()
