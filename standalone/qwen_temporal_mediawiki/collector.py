"""Checkpointed historical Wiki collection using the WikiTection page renderer.

Only scheduling and persistence live here. Page revision lookup, rendered-text
extraction, text filters, and near-duplicate detection reuse the recent pool's
existing implementations in experiments.shared.data.pools.
"""
from __future__ import annotations

import fcntl
import hashlib
import json
import os
from pathlib import Path
from urllib.parse import urlencode

from experiments.shared.data import pools
from experiments.shared.data.pool_storage import recover_pool_transaction


SCHEMA = 'historical_wiki_checkpoint_v1'


def _append(path: Path, value: dict) -> None:
    payload = (json.dumps(value, ensure_ascii=False) + '\n').encode('utf-8')
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('ab') as stream:
        stream.write(payload)
        stream.flush()
        os.fsync(stream.fileno())


def _read(path: Path) -> list[dict]:
    if not path.exists():
        return []
    with path.open('rb+') as stream:
        payload = stream.read()
        end = payload.rfind(b'\n') + 1
        if end != len(payload):
            # An interrupted append can leave an incomplete final JSON line.
            stream.truncate(end)
            stream.flush()
            os.fsync(stream.fileno())
            payload = payload[:end]
    return [json.loads(line) for line in payload.splitlines()]


def checkpoint_status(pool: Path) -> dict:
    """Read complete journal lines without touching an active collector."""
    folder = Path(pool).parent / 'checkpoint'
    counts = dict(creation_events=0, prefiltered_pages=0, survivors=0,
                  tried=0, usable=0)
    for name in ('events', 'prefilter', 'articles'):
        path = folder / f'{name}.jsonl'
        if not path.exists():
            continue
        payload = path.read_bytes()
        complete = payload[:payload.rfind(b'\n') + 1]
        for line in complete.splitlines():
            row = json.loads(line)
            if name == 'events':
                counts['creation_events'] += len(row['events'])
            elif name == 'prefilter':
                counts['prefiltered_pages'] += len(row['pageids'])
                counts['survivors'] += len(row['survivors'])
            else:
                counts['tried'] += 1
                counts['usable'] += row.get('record') is not None
    counts['pool_complete'] = Path(pool).exists() and Path(pool).with_suffix('.manifest.json').exists()
    cache = Path(pool).parent / 'api_cache'
    counts['cached_api_responses'] = sum(1 for path in cache.glob('*.json')) if cache.exists() else 0
    return counts


def _immutable_plan(args) -> dict:
    return dict(schema=SCHEMA, source_api=pools.WIKI_API,
                window_start=args.window_start, window_end=args.window_end,
                snapshot_at=args.snapshot_at, min_chars=700, max_chars=9000,
                full_text=True)


def _check_plan(folder: Path, args) -> None:
    path = folder / 'PLAN.json'
    plan = _immutable_plan(args)
    if path.exists():
        if json.loads(path.read_text()) != plan:
            raise ValueError('checkpoint collection window or extraction rules changed; use a new pool directory')
    else:
        staged = path.with_suffix('.tmp')
        with staged.open('wb') as stream:
            stream.write((json.dumps(plan, ensure_ascii=False) + '\n').encode('utf-8'))
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(staged, path)


def _events(folder: Path, args) -> list[dict]:
    journal = folder / 'events.jsonl'
    batches = _read(journal)
    events = []
    for batch in batches:
        if batch.get('start') != len(events) or not isinstance(batch.get('events'), list):
            raise ValueError('creation event checkpoint is inconsistent')
        events.extend(batch['events'])
    if len(events) >= args.candidate_limit or (folder / 'EVENTS_EXHAUSTED').exists():
        return events[:args.candidate_limit]

    pending = []
    source = pools._wiki_creation_events(args.window_start, args.window_end)
    exhausted = True
    for index, event in enumerate(source):
        if index < len(events):
            continue
        if index >= args.candidate_limit:
            exhausted = False
            break
        pending.append(event)
        if len(pending) == 500:
            _append(journal, dict(start=len(events), events=pending))
            events.extend(pending)
            pending = []
            print(f'creation_events={len(events)}/{args.candidate_limit}', flush=True)
    if pending:
        _append(journal, dict(start=len(events), events=pending))
        events.extend(pending)
        print(f'creation_events={len(events)}/{args.candidate_limit}', flush=True)
    if exhausted:
        (folder / 'EVENTS_EXHAUSTED').write_text('creation log exhausted\n')
    return events


def _prefilter(folder: Path, page_ids: list[int], args) -> list[dict]:
    journal = folder / 'prefilter.jsonl'
    rows = _read(journal)
    cursor, survivors = 0, []
    for row in rows:
        ids = row.get('pageids')
        if (not isinstance(ids, list) or not ids
                or ids != page_ids[cursor:cursor + len(ids)]):
            raise ValueError('prefilter checkpoint no longer matches creation events')
        cursor += len(ids)
        survivors.extend(row['survivors'])
    print(f'resume prefiltered_pages={cursor}/{len(page_ids)} survivors={len(survivors)}', flush=True)
    while cursor < len(page_ids) and len(survivors) < args.survivor_limit:
        batch = page_ids[cursor:cursor + 50]
        url = f"{pools.WIKI_API}?{urlencode({
            'format': 'json', 'formatversion': 2, 'action': 'query',
            'pageids': '|'.join(map(str, batch)), 'prop': 'info|pageprops',
            'inprop': 'url',
        })}"
        payload = pools._request_json(url, sleep=0.5, attempts=8)
        kept = []
        for page in payload.get('query', {}).get('pages', []):
            if (page.get('missing') or page.get('redirect')
                    or 'disambiguation' in page.get('pageprops', {})):
                continue
            if int(page.get('length', 0)) < 1200:
                continue
            kept.append(page)
        _append(journal, dict(pageids=batch, survivors=kept))
        cursor += len(batch)
        survivors.extend(kept)
        if cursor % 500 == 0 or cursor == len(page_ids) or len(survivors) >= args.survivor_limit:
            print(f'prefiltered_pages={cursor}/{len(page_ids)} survivors={len(survivors)}', flush=True)
    return survivors[:args.survivor_limit]


def _articles(folder: Path, survivors: list[dict], creations: dict[int, dict], args) -> list[dict]:
    journal = folder / 'articles.jsonl'
    attempts = _read(journal)
    seen_text, near, accepted = set(), pools.NearDuplicateIndex(), []
    for index, attempt in enumerate(attempts):
        if (attempt.get('index') != index or index >= len(survivors)
                or attempt.get('page_id') != int(survivors[index]['pageid'])):
            raise ValueError('article checkpoint no longer matches prefiltered pages')
        record = attempt.get('record')
        if record is not None:
            digest = hashlib.sha256(record['text'].encode()).hexdigest()
            if digest != record['text_sha256'] or digest in seen_text or near.is_duplicate(record['text']):
                raise ValueError('article checkpoint contains a changed or duplicate record')
            accepted.append(record)
            seen_text.add(digest)
            near.add(record['text'])
    print(f'resume tried={len(attempts)} usable={len(accepted)}', flush=True)
    for index in range(len(attempts), len(survivors)):
        if len(accepted) >= args.records:
            break
        page = survivors[index]
        page_id = int(page['pageid'])
        record = pools._wiki_page_record(page, creations[page_id], 700, 9000,
                                         args.snapshot_at)
        if record is not None:
            digest = record['_text_digest']
            if digest in seen_text or near.is_duplicate(record['text']):
                record = None
            else:
                record.pop('_text_digest')
                accepted.append(record)
                seen_text.add(digest)
                near.add(record['text'])
        _append(journal, dict(index=index, page_id=page_id, record=record))
        if (index + 1) % 4 == 0 or len(accepted) >= args.records:
            print(f'tried={index + 1} usable={len(accepted)}', flush=True)
    return accepted[:args.records]


def collect_historical(args) -> None:
    """Resume a journaled collection and publish the final verified pool pair."""
    pool = Path(args.historical_pool).resolve()
    folder = pool.parent / 'checkpoint'
    folder.mkdir(parents=True, exist_ok=True)
    with (folder / '.lock').open('a+b') as lock:
        try:
            fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise RuntimeError('another historical collection is active in this directory') from error
        _check_plan(folder, args)
        recover_pool_transaction(pool)
        if pool.exists() and pool.with_suffix('.manifest.json').exists():
            return
        pools._WIKI_INTERVAL = args.request_interval
        pools.USER_AGENT = f'SD-MIA-research/0.1 ({args.contact})'
        pools._WIKI_CACHE = pool.parent / 'api_cache'
        pools._WIKI_CACHE.mkdir(parents=True, exist_ok=True)

        events = _events(folder, args)
        creations = {int(event['pageid']): event for event in events if event.get('pageid')}
        page_ids = list(creations)
        survivors = _prefilter(folder, page_ids, args)
        accepted = _articles(folder, survivors, creations, args)
        if len(accepted) < args.records:
            raise RuntimeError(
                f'only {len(accepted)}/{args.records} usable historical pages from '
                f'{len(events)} creation events and {len(survivors)} prefiltered pages; '
                'checkpoints are saved; increase --candidate-limit and/or --survivor-limit'
            )
        pools._write_pool(pool, accepted, {
            'benchmark': 'wikitection',
            'checkpoint_collector_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            'wiki_page_extractor_sha256': hashlib.sha256(Path(pools.__file__).read_bytes()).hexdigest(),
            'dataset': 'English Wikipedia main-namespace pages first created in the window',
            'source_api': pools.WIKI_API,
            'creation_interval_inclusive': {'start': args.window_start, 'end': args.window_end},
            'timestamp_semantics': 'page first-creation time (MediaWiki create log)',
            'license': 'CC BY-SA 4.0; per-page attribution URLs in JSONL',
            'provenance': 'Historical Wikipedia temporal membership proxy; pre-cutoff date does not verify training inclusion',
            'snapshot_at': args.snapshot_at,
            'selection_tokenizer': None,
            'label_semantics': 'presumed_member_temporal_proxy',
            'historical_render_caveat':
                'Pinned main-page revision; MediaWiki may expand current transcluded templates. '
                'Page availability, redirect and disambiguation prefilter use current metadata.',
            'selection': {
                'creation_events_considered': len(events),
                'prefiltered_pages_considered': len(survivors),
                'minimum_clean_characters': 700,
                'maximum_clean_characters': 9000,
                'namespace': 0,
                'disambiguation_pages_excluded': True,
                'exact_clean_text_deduplicated': True,
            },
        })
