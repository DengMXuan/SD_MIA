#!/usr/bin/env python3
"""Freeze a Qwen temporal proxy with MediaWiki extraction on both sides.

The historical pool is collected by the *same* WikiTection collector that
produced the recent pool.  Historical pages use a pinned pre-release revision;
the recent test and auxiliary IDs remain those of the frozen SFT split.
Existing WikiText manifests and audit results are never modified.
"""
from __future__ import annotations

import argparse
import ast
from datetime import datetime
import hashlib
import json
import os
from pathlib import Path
import shutil
import tempfile
import time
from urllib.parse import parse_qs, urlsplit

import numpy as np

from experiments.paths import ROOT
from experiments.pretraining.data import TOKEN_CONTRACT, load_tokenizer, sha256, tokenizer_hash
from experiments.pretraining.datasets import QWEN_MODELS, QWEN_RELEASE, _pool
from experiments.shared.data.data import _hash_ids
from experiments.shared.data import pools as wiki_pools
from experiments.shared.data.pools import NearDuplicateIndex, WIKI_API
from standalone.qwen_temporal_mediawiki.collector import checkpoint_status, collect_historical


VERSION = 'qwen_temporal_mediawiki_v1'
BASE = ROOT / 'artifacts/data' / VERSION
HISTORICAL_POOL = BASE / 'historical_pool/pool.jsonl'
RECENT_POOL = ROOT / 'artifacts/data/pools/wikitection/pool.jsonl'
SPLITS = ROOT / 'artifacts/training/controlled_sft_v2/splits/wikitection'
SEEDS = (1919, 1949, 1978)
DEFAULT_START = '2023-01-01T00:00:00Z'
DEFAULT_END = '2023-12-31T23:59:59Z'
DEFAULT_RECORDS = 3000
DEFAULT_CANDIDATE_LIMIT = 50000
DEFAULT_SURVIVOR_LIMIT = 10000
DEFAULT_CONTACT = 'https://github.com/DengMXuan/SD_MIA'
DEFAULT_REQUEST_INTERVAL = 1.0


def require(ok: bool, message: str) -> None:
    if not ok:
        raise ValueError(message)


def _date(value: str):
    return datetime.fromisoformat(value.replace('Z', '+00:00'))


def _source_files(historical_pool: Path, recent_pool: Path, shared: Path) -> list[dict]:
    files = (historical_pool, historical_pool.with_suffix('.manifest.json'),
             historical_pool.parent / 'COLLECTION.json',
             recent_pool, recent_pool.with_suffix('.manifest.json'), shared,
             Path(__file__), Path(collect_historical.__code__.co_filename),
             Path(wiki_pools.__file__))
    return [dict(path=str(p.resolve()), sha256=sha256(p)) for p in files]


def inspect_sources(historical_pool: Path, recent_pool: Path, shared: Path, seed: int,
                    *, n_per_class: int = 2000, n_aux: int = 600):
    """Check provenance and the frozen recent IDs before loading a tokenizer."""
    historical_pool, recent_pool, shared = map(
        lambda p: Path(p).resolve(), (historical_pool, recent_pool, shared))
    history, hmeta = _pool(historical_pool)
    recent, rmeta = _pool(recent_pool)
    receipt = json.loads((historical_pool.parent / 'COLLECTION.json').read_text())
    split = json.loads(shared.read_text())
    release = _date(QWEN_RELEASE + 'T00:00:00Z')
    require(hmeta.get('benchmark') == rmeta.get('benchmark') == 'wikitection'
            and hmeta.get('source_api') == rmeta.get('source_api') == WIKI_API,
            'both pools must come from the MediaWiki WikiTection collector')
    require(receipt.get('collector') == 'standalone.qwen_temporal_mediawiki.collector.collect_historical'
            and receipt.get('full_text') is True
            and receipt.get('pool_sha256') == sha256(historical_pool)
            and receipt.get('manifest_sha256') == sha256(historical_pool.with_suffix('.manifest.json'))
            and receipt.get('collector_code_sha256') ==
                sha256(Path(collect_historical.__code__.co_filename))
            and receipt.get('wiki_page_code_sha256') == sha256(Path(wiki_pools.__file__))
            and hmeta.get('checkpoint_collector_sha256') == receipt.get('collector_code_sha256')
            and hmeta.get('wiki_page_extractor_sha256') == receipt.get('wiki_page_code_sha256'),
            'historical collection lacks a matching full-article receipt')
    require(hmeta.get('label_semantics') == 'presumed_member_temporal_proxy'
            and hmeta.get('snapshot_at') and _date(hmeta['snapshot_at']) < release,
            'historical pool must have a pinned pre-release snapshot')
    require(hmeta.get('selection', {}).get('minimum_clean_characters') ==
            rmeta.get('selection', {}).get('minimum_clean_characters') and
            hmeta.get('selection', {}).get('maximum_clean_characters') ==
            rmeta.get('selection', {}).get('maximum_clean_characters'),
            'historical and recent character filters differ')
    require(split.get('schema_version') == 3 and split.get('benchmark') == 'wikitection'
            and split.get('seed') == seed and split.get('pool_sha256') == rmeta['jsonl_sha256'],
            'shared split does not match the recent pool and seed')
    require(set(split['splits']) == {'member', 'nonmember', 'auxiliary', 'audit_auxiliary'}
            and len(split['splits']['nonmember']) == n_per_class
            and len(split['splits']['audit_auxiliary']) == n_aux,
            'frozen recent role counts differ from the requested experiment')
    require(split['token_band'] == {'min_tokens': 128, 'max_tokens': 512},
            'expected the same 128–512 token band as the frozen recent split')
    by_id = {row['record_id']: row for row in recent}
    require(len(by_id) == len(recent), 'duplicate recent pool ID')
    fixed = []
    for group, role in (('nonmember', 'nonmember'), ('auxiliary', 'audit_auxiliary')):
        for entry in split['splits'][role]:
            row = by_id.get(entry['record_id'])
            require(row is not None and hashlib.sha256(row['text'].encode()).hexdigest() ==
                    entry['text_sha256'], 'frozen recent document changed')
            require(_date(row['creation_timestamp']) > release,
                    'recent document was created before model release')
            fixed.append((group, row))
    require(len({row['record_id'] for _, row in fixed}) == len(fixed),
            'frozen recent roles overlap')
    for row in history:
        require(row.get('source') == 'en.wikipedia.org' and
                int(row.get('snapshot_revision', 0)) > 0 and
                _date(row['creation_timestamp']) < release and
                _date(row['snapshot_timestamp']) < release and
                hashlib.sha256(row['text'].encode()).hexdigest() == row['text_sha256'],
                'historical page lacks a valid pre-release revision or text hash')
    return history, fixed, split, _source_files(historical_pool, recent_pool, shared)


def _token_ids(tokenizer, text: str, limit: int) -> list[int]:
    return list(tokenizer(text, add_special_tokens=False, truncation=True,
                          max_length=limit).input_ids)


def _record(group: str, row: dict, ids: list[int]) -> dict:
    return dict(record_id=f'temporal:{row["record_id"]}',
                source=row['source'], source_record_id=row['record_id'],
                group=group, label=int(group == 'member'), text=row['text'],
                token_ids=ids, token_hash=_hash_ids(ids),
                temporal_metadata={key: row[key] for key in
                                   ('title', 'creation_timestamp', 'snapshot_timestamp', 'snapshot_revision')
                                   if key in row})


def prepare_seed(historical_pool: Path, recent_pool: Path, shared: Path, output: Path,
                 *, seed: int, n_per_class: int = 2000, n_aux: int = 600,
                 tokenizer=None, models=None) -> Path:
    """Use rendered MediaWiki text for both roles, then freeze one seed."""
    output = Path(output).resolve()
    historical_pool, recent_pool, shared = map(
        lambda p: Path(p).resolve(), (historical_pool, recent_pool, shared))
    for source in (historical_pool, recent_pool, shared):
        require(output != source and output not in source.parents and
                source not in output.parents, 'output overlaps a source')
    history, fixed, split, files = inspect_sources(
        historical_pool, recent_pool, shared, seed, n_per_class=n_per_class, n_aux=n_aux)
    models = models or QWEN_MODELS
    supplied_tokenizer = tokenizer is not None
    tokenizer = tokenizer or load_tokenizer(models['target'])
    fingerprint = tokenizer_hash(tokenizer)
    if not supplied_tokenizer:
        # For the real experiment, both fixed pretrained models must tokenize identically.
        require(fingerprint == tokenizer_hash(load_tokenizer(models['draft'])),
                'target and draft tokenizers differ')
    if output.exists():
        manifest = json.loads((output / 'manifest.json').read_text())
        require(manifest.get('source_provenance', {}).get('files') == files and
                manifest.get('selection_seed') == seed and
                manifest.get('tokenizer_sha256') == fingerprint and
                sha256(output / manifest['records_file']) == manifest['records_sha256'],
                'prepared output changed; choose a new directory')
        return output / 'manifest.json'

    minimum, maximum = split['token_band']['min_tokens'], split['token_band']['max_tokens']
    seen_ids, seen_tokens, near = set(), set(), NearDuplicateIndex()
    rows = []
    for group, row in fixed:
        ids = _token_ids(tokenizer, row['text'], maximum)
        require(minimum <= len(ids) <= maximum, 'fixed recent row violates its token band')
        item = _record(group, row, ids)
        require(item['record_id'] not in seen_ids and item['token_hash'] not in seen_tokens,
                'duplicate frozen recent row')
        rows.append(item)
        seen_ids.add(item['record_id']); seen_tokens.add(item['token_hash'])
        near.add(tokenizer.decode(ids, skip_special_tokens=False))

    selected, dropped_short, dropped_duplicate = [], 0, 0
    for index in np.random.default_rng(seed).permutation(len(history)):
        source = history[int(index)]
        ids = _token_ids(tokenizer, source['text'], maximum)
        if len(ids) < minimum:
            dropped_short += 1
            continue
        item = _record('member', source, ids)
        visible = tokenizer.decode(ids, skip_special_tokens=False)
        if (item['record_id'] in seen_ids or item['token_hash'] in seen_tokens or
                near.is_duplicate(visible)):
            dropped_duplicate += 1
            continue
        selected.append(item)
        seen_ids.add(item['record_id']); seen_tokens.add(item['token_hash'])
        near.add(visible)
        if len(selected) == n_per_class:
            break
    require(len(selected) == n_per_class,
            f'insufficient historical articles: {len(selected)}/{n_per_class}; '
            'collect more pages into a new versioned pool')
    rows = selected + rows
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=f'.seed{seed}-', dir=output.parent))
    try:
        records = temporary / 'records.jsonl'
        records.write_text(''.join(json.dumps(row, ensure_ascii=False) + '\n' for row in rows))
        lengths = {group: [len(row['token_ids']) for row in rows if row['group'] == group]
                   for group in ('member', 'nonmember', 'auxiliary')}
        manifest = dict(kind='temporal_pretraining_v1',
                        benchmark=f'wiki_temporal/{VERSION}', models=models,
                        token_contract=TOKEN_CONTRACT, tokenizer_sha256=fingerprint,
                        counts=dict(member=n_per_class, nonmember=n_per_class, auxiliary=n_aux),
                        selection_seed=seed, min_tokens=min(map(min, lengths.values())),
                        max_tokens=maximum, records_file=records.name,
                        records_sha256=sha256(records), model_release=QWEN_RELEASE,
                        membership_verified=False,
                        label_provenance='historical presumed-member / post-release nonmember temporal proxies',
                        draft_exposure='public pretrained Qwen draft; pretraining membership unknown',
                        source_provenance=dict(extraction='MediaWiki action=parse rendered full article on both sides',
                                               historical_snapshot='pinned pre-release revision', files=files),
                        filtering=dict(member=dict(short=dropped_short, duplicate=dropped_duplicate,
                                                   candidates=len(history), selected=len(selected)),
                                       recent='exact frozen SFT nonmember/audit_auxiliary IDs and order'),
                        token_lengths={group: dict(min=min(values), max=max(values),
                                                   mean=float(np.mean(values)))
                                       for group, values in lengths.items()},
                        caveats=['page creation dates do not verify model training membership',
                                 'historical revision rendering may expand current templates',
                                 'matched extraction and token band do not guarantee matching topics or styles'])
        (temporary / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
        temporary.rename(output)
    finally:
        if temporary.exists():
            shutil.rmtree(temporary)
    return output / 'manifest.json'


def collect(args) -> None:
    """Checkpoint extraction using the recent-page renderer and filters."""
    pool = args.historical_pool.resolve()
    require(args.parallel == 1,
            'anonymous Wikimedia Action API collection requires --parallel 1')
    require(args.records > 0 and args.candidate_limit > 0 and
            args.survivor_limit >= args.records and args.request_interval > 0 and
            bool(args.contact),
            'records, candidate/survivor limits, request interval and contact must be valid')
    require(_date(args.window_start) <= _date(args.window_end) <= _date(args.snapshot_at)
            < _date(QWEN_RELEASE + 'T00:00:00Z'), 'historical window must precede model release')
    require(pool != RECENT_POOL.resolve() and pool not in RECENT_POOL.resolve().parents,
            'historical collection cannot overwrite the recent pool')
    receipt = pool.parent / 'COLLECTION.json'
    request = dict(schema=VERSION,
                   collector='standalone.qwen_temporal_mediawiki.collector.collect_historical',
                   full_text=True, window_start=args.window_start, window_end=args.window_end,
                   snapshot_at=args.snapshot_at, min_chars=700, max_chars=9000,
                   records=args.records, candidate_limit=args.candidate_limit,
                   survivor_limit=args.survivor_limit,
                   source_code_sha256=sha256(Path(__file__)),
                   collector_code_sha256=sha256(Path(collect_historical.__code__.co_filename)),
                   wiki_page_code_sha256=sha256(Path(wiki_pools.__file__)))
    if receipt.exists():
        require(pool.exists() and pool.with_suffix('.manifest.json').exists() and
                json.loads(receipt.read_text()) == {
                    **request, 'pool_sha256': sha256(pool),
                    'manifest_sha256': sha256(pool.with_suffix('.manifest.json'))},
                'historical collection differs; use a fresh output directory')
        print(f'reuse {pool}')
        return
    pool.parent.mkdir(parents=True, exist_ok=True)
    original_request_json = wiki_pools._request_json

    def request_json_with_slow_pause(url, **kwargs):
        started = time.monotonic()
        try:
            try:
                return original_request_json(url, **kwargs)
            except RuntimeError as error:
                query = parse_qs(urlsplit(url).query)
                prefix = 'Wikipedia API error: '
                if (query.get('action') == ['parse'] and 'oldid' in query
                        and str(error).startswith(prefix)):
                    try:
                        detail = ast.literal_eval(str(error)[len(prefix):])
                    except (SyntaxError, ValueError):
                        detail = None
                    if (isinstance(detail, dict) and detail.get('code') in
                            {'permissiondenied', 'nosuchrevid', 'missingtitle'}):
                        print(f'skip unavailable historical revision oldid={query["oldid"][0]} '
                              f'code={detail["code"]}',
                              flush=True)
                        return {'parse': {}}
                raise
        finally:
            # Wikimedia asks Action API clients to pause after slow requests.
            if url.startswith(WIKI_API) and time.monotonic() - started > 1:
                time.sleep(5)

    wiki_pools._request_json = request_json_with_slow_pause
    try:
        collect_historical(args)
    finally:
        wiki_pools._request_json = original_request_json
    meta = json.loads(pool.with_suffix('.manifest.json').read_text())
    require(meta.get('records') == args.records and
            meta.get('snapshot_at') == args.snapshot_at and
            meta.get('creation_interval_inclusive') ==
                {'start': args.window_start, 'end': args.window_end} and
            meta.get('checkpoint_collector_sha256') == request['collector_code_sha256'] and
            meta.get('wiki_page_extractor_sha256') == request['wiki_page_code_sha256'] and
            sha256(pool) == meta['jsonl_sha256'],
            'completed pool does not match this collection plan')
    staged = receipt.with_suffix('.tmp')
    with staged.open('w') as stream:
        stream.write(json.dumps({**request, 'pool_sha256': sha256(pool),
                                 'manifest_sha256': sha256(pool.with_suffix('.manifest.json'))}, indent=2) + '\n')
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(staged, receipt)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=('collect', 'prepare', 'status'))
    parser.add_argument('--historical-pool', type=Path, default=HISTORICAL_POOL)
    parser.add_argument('--recent-pool', type=Path, default=RECENT_POOL)
    parser.add_argument('--split-root', type=Path, default=SPLITS)
    parser.add_argument('--output-root', type=Path, default=BASE / 'shared_split')
    parser.add_argument('--seeds', nargs='+', type=int, choices=SEEDS, default=SEEDS)
    parser.add_argument('--window-start', default=DEFAULT_START)
    parser.add_argument('--window-end', default=DEFAULT_END)
    parser.add_argument('--snapshot-at', default=DEFAULT_END)
    parser.add_argument('--records', type=int, default=DEFAULT_RECORDS)
    parser.add_argument('--candidate-limit', type=int, default=DEFAULT_CANDIDATE_LIMIT)
    parser.add_argument('--survivor-limit', type=int, default=DEFAULT_SURVIVOR_LIMIT)
    parser.add_argument('--parallel', type=int, default=1)
    parser.add_argument('--request-interval', type=float, default=DEFAULT_REQUEST_INTERVAL)
    parser.add_argument('--contact', default=DEFAULT_CONTACT,
                        help='public project contact for MediaWiki User-Agent')
    args = parser.parse_args()
    require(len(set(args.seeds)) == len(args.seeds), 'duplicate seed')
    if args.command == 'collect':
        collect(args)
        return
    if args.command == 'status':
        print(json.dumps(checkpoint_status(args.historical_pool), ensure_ascii=False))
    for seed in args.seeds:
        output = args.output_root / f'seed{seed}'
        if args.command == 'status':
            print(f'seed{seed}: {"prepared" if (output / "manifest.json").exists() else "missing"}')
            continue
        print(prepare_seed(args.historical_pool, args.recent_pool,
                           args.split_root / f'seed{seed}.json', output, seed=seed))


if __name__ == '__main__':
    main()
