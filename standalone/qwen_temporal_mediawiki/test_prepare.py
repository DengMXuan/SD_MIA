"""Check source parity and frozen-role preservation without network or GPU."""
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from experiments.pretraining.data import sha256
from experiments.shared.data import pools
from experiments.shared.data.pools import WIKI_API, _write_pool
from standalone.qwen_temporal_mediawiki import prepare as study
from standalone.qwen_temporal_mediawiki import collector


class Tokenizer:
    backend_tokenizer = SimpleNamespace(to_str=lambda: '{"fixture": true}')

    def __call__(self, text, **kwargs):
        ids = [int(hashlib.sha256(word.encode()).hexdigest()[:4], 16)
               for word in text.split()]
        return SimpleNamespace(input_ids=ids[:kwargs['max_length']])

    def decode(self, ids, **kwargs):
        return ' '.join(map(str, ids))


def page(prefix, number, year):
    text = ' '.join(f'{prefix}{number}word{i}' for i in range(160))
    digest = hashlib.sha256(text.encode()).hexdigest()
    return dict(record_id=f'wiki:{prefix}{number}', source='en.wikipedia.org',
                title=f'{prefix}{number}', text=text, text_sha256=digest,
                creation_timestamp=f'{year}-02-01T00:00:00Z',
                snapshot_timestamp=f'{year}-12-01T00:00:00Z', snapshot_revision=number + 1)


def fixture(tmp_path):
    old = tmp_path / 'old/pool.jsonl'
    recent = tmp_path / 'new/pool.jsonl'
    history = [page('old', i, 2023) for i in range(4)]
    current = [page('new', i, 2026) for i in range(4)]
    common = dict(benchmark='wikitection', source_api=WIKI_API,
                  selection=dict(minimum_clean_characters=700, maximum_clean_characters=9000))
    _write_pool(old, history, {**common, 'label_semantics': 'presumed_member_temporal_proxy',
                               'snapshot_at': '2023-12-31T23:59:59Z',
                               'checkpoint_collector_sha256': sha256(Path(collector.__file__)),
                               'wiki_page_extractor_sha256': sha256(Path(pools.__file__))})
    _write_pool(recent, current, common)
    (old.parent / 'COLLECTION.json').write_text(json.dumps(dict(
        collector='standalone.qwen_temporal_mediawiki.collector.collect_historical', full_text=True,
        collector_code_sha256=sha256(Path(collector.__file__)),
        wiki_page_code_sha256=sha256(Path(pools.__file__)),
        pool_sha256=sha256(old), manifest_sha256=sha256(old.with_suffix('.manifest.json')))))
    entries = [dict(record_id=row['record_id'], text_sha256=row['text_sha256'])
               for row in current]
    shared = tmp_path / 'shared.json'
    shared.write_text(json.dumps(dict(schema_version=3, benchmark='wikitection', seed=1919,
        pool_sha256=sha256(recent), token_band=dict(min_tokens=128, max_tokens=512),
        splits=dict(member=[], nonmember=entries[:2], auxiliary=[], audit_auxiliary=entries[2:]))))
    return old, recent, shared, current


def test_same_collector_and_frozen_recent_roles(tmp_path):
    old, recent, shared, current = fixture(tmp_path)
    output = tmp_path / 'prepared/seed1919'
    models = {'target': {'repo_id': 'fixture-target'}, 'draft': {'repo_id': 'fixture-draft'}}
    manifest = study.prepare_seed(old, recent, shared, output, seed=1919,
                                  n_per_class=2, n_aux=2, tokenizer=Tokenizer(), models=models)
    saved = json.loads(manifest.read_text())
    rows = [json.loads(line) for line in (output / 'records.jsonl').read_text().splitlines()]
    assert saved['source_provenance']['extraction'].startswith('MediaWiki action=parse')
    assert [row['source_record_id'] for row in rows[2:]] == [row['record_id'] for row in current]
    assert [row['group'] for row in rows] == ['member'] * 2 + ['nonmember'] * 2 + ['auxiliary'] * 2
    assert all(row['source_record_id'].startswith('wiki:old') for row in rows[:2])
    assert saved['membership_verified'] is False
    assert study.prepare_seed(old, recent, shared, output, seed=1919,
                              n_per_class=2, n_aux=2, tokenizer=Tokenizer(), models=models) == manifest


@pytest.mark.parametrize('fault', ('wrong_collector', 'unfrozen_revision', 'recent_source_changed'))
def test_source_mismatch_rejected_before_output(tmp_path, fault):
    old, recent, shared, _ = fixture(tmp_path)
    if fault == 'wrong_collector':
        receipt = old.parent / 'COLLECTION.json'
        value = json.loads(receipt.read_text())
        value['full_text'] = False
        receipt.write_text(json.dumps(value))
    elif fault == 'unfrozen_revision':
        rows = [json.loads(line) for line in old.read_text().splitlines()]
        rows[0]['snapshot_timestamp'] = '2026-02-01T00:00:00Z'
        meta = json.loads(old.with_suffix('.manifest.json').read_text())
        _write_pool(old, rows, meta)
        receipt = old.parent / 'COLLECTION.json'
        value = json.loads(receipt.read_text())
        value['pool_sha256'] = sha256(old)
        value['manifest_sha256'] = sha256(old.with_suffix('.manifest.json'))
        receipt.write_text(json.dumps(value))
    else:
        value = json.loads(shared.read_text())
        value['pool_sha256'] = 'wrong'
        shared.write_text(json.dumps(value))
    output = tmp_path / 'prepared/seed1919'
    with pytest.raises(ValueError):
        study.inspect_sources(old, recent, shared, 1919, n_per_class=2, n_aux=2)
    assert not output.exists()


def test_collection_preserves_historical_snapshot_and_contact(tmp_path, monkeypatch):
    captured = []

    def fake_builder(args):
        captured.append(args)
        _write_pool(args.historical_pool, [page('old', 0, 2023)], dict(
            benchmark='wikitection', source_api=WIKI_API,
            label_semantics='presumed_member_temporal_proxy', snapshot_at=args.snapshot_at,
            creation_interval_inclusive={'start': args.window_start, 'end': args.window_end},
            checkpoint_collector_sha256=sha256(Path(fake_builder.__code__.co_filename)),
            wiki_page_extractor_sha256=sha256(Path(pools.__file__)),
            selection=dict(minimum_clean_characters=700, maximum_clean_characters=9000)))

    monkeypatch.setattr(study, 'collect_historical', fake_builder)
    args = SimpleNamespace(historical_pool=tmp_path / 'historical/pool.jsonl',
        window_start=study.DEFAULT_START, window_end=study.DEFAULT_END,
        snapshot_at=study.DEFAULT_END, records=1, candidate_limit=10, survivor_limit=10,
        parallel=1, request_interval=study.DEFAULT_REQUEST_INTERVAL,
        contact=study.DEFAULT_CONTACT)
    study.collect(args)
    assert len(captured) == 1
    assert captured[0].request_interval == 1.0
    assert captured[0].contact == study.DEFAULT_CONTACT
    assert captured[0].snapshot_at == study.DEFAULT_END
    assert captured[0].survivor_limit == 10
    study.collect(args)
    assert len(captured) == 1
    args.parallel = 3
    with pytest.raises(ValueError, match='--parallel 1'):
        study.collect(args)


def test_slow_wiki_response_gets_extra_pause_and_restores_requester(tmp_path, monkeypatch):
    pauses = []
    ticks = iter((0.0, 1.2))

    def fake_request(url, **kwargs):
        return {}

    def fake_builder(args):
        pools._request_json(WIKI_API + '?action=query')
        _write_pool(args.historical_pool, [page('old', 0, 2023)], dict(
            benchmark='wikitection', source_api=WIKI_API,
            label_semantics='presumed_member_temporal_proxy', snapshot_at=args.snapshot_at,
            creation_interval_inclusive={'start': args.window_start, 'end': args.window_end},
            checkpoint_collector_sha256=sha256(Path(fake_builder.__code__.co_filename)),
            wiki_page_extractor_sha256=sha256(Path(pools.__file__)),
            selection=dict(minimum_clean_characters=700, maximum_clean_characters=9000)))

    monkeypatch.setattr(pools, '_request_json', fake_request)
    monkeypatch.setattr(study, 'collect_historical', fake_builder)
    monkeypatch.setattr(study.time, 'monotonic', lambda: next(ticks))
    monkeypatch.setattr(study.time, 'sleep', pauses.append)
    args = SimpleNamespace(historical_pool=tmp_path / 'historical/pool.jsonl',
        window_start=study.DEFAULT_START, window_end=study.DEFAULT_END,
        snapshot_at=study.DEFAULT_END, records=1, candidate_limit=10, survivor_limit=10,
        parallel=1, request_interval=1.0, contact=study.DEFAULT_CONTACT)
    study.collect(args)
    assert pauses == [5]
    assert pools._request_json is fake_request


@pytest.mark.parametrize('code', ('permissiondenied', 'nosuchrevid', 'missingtitle'))
def test_unavailable_historical_revision_is_skipped_instead_of_aborting(tmp_path, monkeypatch, code):
    calls = []

    def fake_request(url, **kwargs):
        calls.append(url)
        if 'action=parse' in url:
            raise RuntimeError("Wikipedia API error: {'code': '" + code + "', "
                               "'info': \"You don't have permission to view deleted text "
                               "or changes between deleted revisions.\"}")
        return {'query': {'pages': [{'revisions': [{
            'revid': 12345, 'timestamp': '2023-12-01T00:00:00Z', 'size': 4000,
        }]}]}}

    def fake_builder(args):
        page_info = {'pageid': 42, 'title': 'Deleted revision'}
        creation = {'timestamp': '2023-01-02T00:00:00Z'}
        assert pools._wiki_page_record(page_info, creation, 700, 9000,
                                       snapshot_at=args.snapshot_at) is None
        _write_pool(args.historical_pool, [page('old', 0, 2023)], dict(
            benchmark='wikitection', source_api=WIKI_API,
            label_semantics='presumed_member_temporal_proxy', snapshot_at=args.snapshot_at,
            creation_interval_inclusive={'start': args.window_start, 'end': args.window_end},
            checkpoint_collector_sha256=sha256(Path(fake_builder.__code__.co_filename)),
            wiki_page_extractor_sha256=sha256(Path(pools.__file__)),
            selection=dict(minimum_clean_characters=700, maximum_clean_characters=9000)))

    monkeypatch.setattr(pools, '_request_json', fake_request)
    monkeypatch.setattr(study, 'collect_historical', fake_builder)
    monkeypatch.setattr(study.time, 'sleep', lambda _: None)
    args = SimpleNamespace(historical_pool=tmp_path / 'historical/pool.jsonl',
        window_start=study.DEFAULT_START, window_end=study.DEFAULT_END,
        snapshot_at=study.DEFAULT_END, records=1, candidate_limit=10, survivor_limit=10,
        parallel=1, request_interval=1.0, contact=study.DEFAULT_CONTACT)
    study.collect(args)
    assert any('action=parse' in url for url in calls)
    assert (args.historical_pool.parent / 'COLLECTION.json').exists()


def test_historical_revision_uses_the_same_full_article_renderer(monkeypatch):
    calls = []

    def request(url, **kwargs):
        calls.append(url)
        return {'query': {'pages': [{'revisions': [{
            'revid': 12345, 'timestamp': '2023-12-01T00:00:00Z', 'size': 4000,
        }]}]}}

    def render(page_id, revision_id):
        calls.append((page_id, revision_id))
        return ' '.join(f'the article discusses topic{i} and its history'
                        for i in range(35))

    monkeypatch.setattr(pools, '_request_json', request)
    monkeypatch.setattr(pools, '_fetch_wiki_fulltext', render)
    monkeypatch.setattr(pools.time, 'sleep', lambda _: None)
    page_info = {'pageid': 42, 'title': 'Article', 'lastrevid': 99999}
    creation = {'timestamp': '2023-01-02T00:00:00Z'}
    row = pools._wiki_page_record(page_info, creation, 700, 9000,
                                  snapshot_at=study.DEFAULT_END)
    assert row['snapshot_revision'] == 12345
    assert row['snapshot_timestamp'] == '2023-12-01T00:00:00Z'
    assert calls[-1] == (42, 12345)
    assert 'rvstart=2023-12-31T23%3A59%3A59Z' in calls[0]


def test_checkpoint_writes_each_article_and_resumes_after_failure(tmp_path, monkeypatch):
    events = [dict(pageid=i, timestamp='2023-02-01T00:00:00Z') for i in range(1, 7)]
    rendered = []
    fail_once = {3}

    def prefilter_request(url, **kwargs):
        ids = [int(value) for value in url.split('pageids=')[1].split('&')[0].split('%7C')]
        return {'query': {'pages': [dict(pageid=i, title=f'Article {i}', length=4000)
                                    for i in ids]}}

    def render(page_info, creation, min_chars, max_chars, snapshot_at):
        number = page_info['pageid']
        rendered.append(number)
        if number in fail_once:
            fail_once.remove(number)
            raise RuntimeError('temporary page failure')
        row = page('old', number, 2023)
        row['page_id'] = number
        row['_text_digest'] = row['text_sha256']
        return row

    monkeypatch.setattr(pools, '_wiki_creation_events', lambda *_: iter(events))
    monkeypatch.setattr(pools, '_request_json', prefilter_request)
    monkeypatch.setattr(pools, '_wiki_page_record', render)
    args = SimpleNamespace(historical_pool=tmp_path / 'historical/pool.jsonl',
        window_start=study.DEFAULT_START, window_end=study.DEFAULT_END,
        snapshot_at=study.DEFAULT_END, records=4, candidate_limit=6,
        survivor_limit=6, request_interval=1.0, contact=study.DEFAULT_CONTACT)
    with pytest.raises(RuntimeError, match='temporary page failure'):
        collector.collect_historical(args)
    journal = args.historical_pool.parent / 'checkpoint/articles.jsonl'
    saved = [json.loads(line) for line in journal.read_text().splitlines()]
    assert [item['page_id'] for item in saved] == [1, 2]
    assert all(item['record'] is not None for item in saved)
    assert collector.checkpoint_status(args.historical_pool)['usable'] == 2
    with journal.open('ab') as stream:
        stream.write(b'{"index":')  # simulate a crash during the next append
    collector.collect_historical(args)
    assert rendered == [1, 2, 3, 3, 4]
    assert len(args.historical_pool.read_text().splitlines()) == 4


def test_survivor_limit_stops_prefilter_early(tmp_path, monkeypatch):
    events = [dict(pageid=i, timestamp='2023-02-01T00:00:00Z') for i in range(1, 121)]
    prefilter_calls = []

    def prefilter_request(url, **kwargs):
        ids = [int(value) for value in url.split('pageids=')[1].split('&')[0].split('%7C')]
        prefilter_calls.append(ids)
        return {'query': {'pages': [dict(pageid=i, title=f'Article {i}', length=4000)
                                    for i in ids]}}

    def render(page_info, creation, min_chars, max_chars, snapshot_at):
        row = page('old', page_info['pageid'], 2023)
        row['_text_digest'] = row['text_sha256']
        return row

    monkeypatch.setattr(pools, '_wiki_creation_events', lambda *_: iter(events))
    monkeypatch.setattr(pools, '_request_json', prefilter_request)
    monkeypatch.setattr(pools, '_wiki_page_record', render)
    args = SimpleNamespace(historical_pool=tmp_path / 'historical/pool.jsonl',
        window_start=study.DEFAULT_START, window_end=study.DEFAULT_END,
        snapshot_at=study.DEFAULT_END, records=1, candidate_limit=120,
        survivor_limit=60, request_interval=1.0, contact=study.DEFAULT_CONTACT)
    collector.collect_historical(args)
    assert len(prefilter_calls) == 2  # 100 of 120 pages; stopped after 60 survivors
    meta = json.loads(args.historical_pool.with_suffix('.manifest.json').read_text())
    assert meta['selection']['creation_events_considered'] == 120
    assert meta['selection']['prefiltered_pages_considered'] == 60


def test_shortfall_keeps_progress_when_limits_are_raised(tmp_path, monkeypatch):
    events = [dict(pageid=i, timestamp='2023-02-01T00:00:00Z') for i in range(1, 11)]

    def prefilter_request(url, **kwargs):
        ids = [int(value) for value in url.split('pageids=')[1].split('&')[0].split('%7C')]
        return {'query': {'pages': [dict(pageid=i, title=f'Article {i}', length=4000)
                                    for i in ids]}}

    rendered = []

    def render(page_info, creation, min_chars, max_chars, snapshot_at):
        number = page_info['pageid']
        rendered.append(number)
        if number in (3, 4, 5, 6):
            return None
        row = page('old', number, 2023)
        row['_text_digest'] = row['text_sha256']
        return row

    monkeypatch.setattr(pools, '_wiki_creation_events', lambda *_: iter(events))
    monkeypatch.setattr(pools, '_request_json', prefilter_request)
    monkeypatch.setattr(pools, '_wiki_page_record', render)
    args = SimpleNamespace(historical_pool=tmp_path / 'historical/pool.jsonl',
        window_start=study.DEFAULT_START, window_end=study.DEFAULT_END,
        snapshot_at=study.DEFAULT_END, records=4, candidate_limit=6,
        survivor_limit=6, request_interval=1.0, contact=study.DEFAULT_CONTACT)
    with pytest.raises(RuntimeError, match='only 2/4 usable'):
        collector.collect_historical(args)
    assert not args.historical_pool.exists()
    args.candidate_limit = 10
    args.survivor_limit = 10
    collector.collect_historical(args)
    assert rendered == list(range(1, 9))
    assert len(args.historical_pool.read_text().splitlines()) == 4
