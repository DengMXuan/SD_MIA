from experiments.shared.data import pools


def test_historical_page_uses_pinned_revision_and_shared_cleaner(monkeypatch):
    requests = []
    def request(url, **kwargs):
        requests.append(url)
        if 'prop=revisions' in url:
            return {'query': {'pages': [{'revisions': [{'revid': 99, 'timestamp': '2023-12-20T00:00:00Z', 'size': 2000}]}]}}
        return {'parse': {'text': '<p>' + ('This is the historical article about a town and its people. ' * 40) + '</p>'}}
    monkeypatch.setattr(pools, '_request_json', request)
    monkeypatch.setattr(pools.time, 'sleep', lambda _: None)
    page = dict(pageid=1, title='Example', lastrevid=999, touched='2026-01-01T00:00:00Z')
    result = pools._wiki_page_record(page, {'timestamp': '2023-01-01T00:00:00Z'}, 700, 9000,
                                     '2023-12-31T23:59:59Z')
    assert result['snapshot_revision'] == 99
    assert result['snapshot_timestamp'] == '2023-12-20T00:00:00Z'
    assert 'oldid=99' in requests[1] and 'pageid=' not in requests[1]
    assert '<p>' not in result['text']
    assert page['lastrevid'] == 999


def test_historical_page_without_old_revision_is_excluded(monkeypatch):
    monkeypatch.setattr(pools, '_request_json', lambda *a, **k: {'query': {'pages': [{}]}})
    assert pools._wiki_page_record({'pageid': 1}, {}, 700, 9000, '2023-12-31T23:59:59Z') is None


def test_current_batch_truncates_long_extract_instead_of_dropping_it(monkeypatch):
    text = ("The town and its people are part of the history of this region. " * 300)
    requested = []

    def request(url, **kwargs):
        requested.append(url)
        return {
            "query": {
                "pages": [
                    {
                        "pageid": 1,
                        "title": "Example",
                        "extract": text,
                        "lastrevid": 99,
                        "touched": "2026-04-02T00:00:00Z",
                        "fullurl": "https://en.wikipedia.org/wiki/Example",
                    }
                ]
            }
        }

    monkeypatch.setattr(
        pools,
        "_request_json",
        request,
    )

    records = pools._wiki_batch_records(
        [{"pageid": 1, "title": "Example"}],
        {1: {"timestamp": "2026-04-01T00:00:00Z"}},
        700,
        9000,
    )

    assert len(records) == 1
    assert len(records[0]["text"]) == 9000
    assert "exlimit=20" in requested[0]
    assert "exintro=1" in requested[0]


def test_successful_api_response_is_reused_after_restart(tmp_path, monkeypatch):
    import io
    import json
    calls = []
    monkeypatch.setattr(pools, '_WIKI_CACHE', tmp_path)
    def request(url, **kwargs):
        calls.append(url)
        return io.BytesIO(json.dumps({'query': {'pages': []}}).encode())
    monkeypatch.setattr(pools, '_request', request)
    url = pools.WIKI_API + '?action=query&format=json'
    assert pools._request_json(url) == pools._request_json(url)
    assert len(calls) == 1
    assert len(list(tmp_path.glob('*.json'))) == 1


def test_api_error_is_not_cached_as_a_missing_page(tmp_path, monkeypatch):
    import io
    import pytest
    monkeypatch.setattr(pools, '_WIKI_CACHE', tmp_path)
    monkeypatch.setattr(pools, '_request', lambda *a, **k: io.BytesIO(b'{"error":{"code":"maxlag"}}'))
    with pytest.raises(RuntimeError, match='maxlag'):
        pools._request_json(pools.WIKI_API + '?action=query')
    assert not list(tmp_path.iterdir())
