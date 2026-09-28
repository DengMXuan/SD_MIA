"""Collect independent post-2023 Wikipedia event pages for WikiMIA calibration.

This command is networked and opt-in. It never reads or modifies the official
WikiMIA test rows. A checkpoint records every inspected page for safe resume.
"""
from __future__ import annotations

from collections import deque
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import time
from urllib.parse import urlencode
from urllib.request import Request, urlopen


API = 'https://en.wikipedia.org/w/api.php'
DEFAULT_CATEGORIES = (
    'Category:2026 disasters', 'Category:2026 in politics',
    'Category:2026 in sports', 'Category:2026 in science',
    'Category:2026 in music',
    'Category:2024 disasters', 'Category:2025 disasters',
    'Category:2024 in politics', 'Category:2025 in politics',
    'Category:2024 in sports', 'Category:2025 in sports',
    'Category:2024 in science', 'Category:2025 in science',
    'Category:2024 in music', 'Category:2025 in music',
)


def _date(value: str) -> datetime:
    return datetime.fromisoformat(value.replace('Z', '+00:00')).astimezone(timezone.utc)


class Client:
    def __init__(self, contact: str, interval: float):
        if not contact.strip() or interval < 1.0:
            raise ValueError('provide a contact URL/email and request interval >= 1 second')
        self.user_agent = f'SD-MIA-WikiMIA-aux/0.1 ({contact})'
        self.interval = interval
        self.last = 0.0

    def query(self, **params) -> dict:
        wait = self.interval - (time.monotonic() - self.last)
        if wait > 0:
            time.sleep(wait)
        url = API + '?' + urlencode({'format': 'json', 'formatversion': 2, **params})
        request = Request(url, headers={'User-Agent': self.user_agent,
                                         'Accept': 'application/json'})
        try:
            with urlopen(request, timeout=45) as response:
                payload = json.load(response)
        finally:
            self.last = time.monotonic()
        if 'error' in payload:
            raise RuntimeError(f'Wikipedia API error: {payload["error"]}')
        return payload


def _members(client: Client, category: str):
    continuation = {}
    while True:
        data = client.query(action='query', list='categorymembers', cmtitle=category,
                            cmtype='page|subcat', cmlimit='500', **continuation)
        yield from data.get('query', {}).get('categorymembers', [])
        continuation = data.get('continue', {})
        if not continuation:
            break


def _page(client: Client, pageid: int, title: str, category: str,
          *, created_after: datetime, minimum_words: int) -> dict | None:
    data = client.query(action='query', pageids=str(pageid),
                        prop='revisions|extracts', rvprop='timestamp',
                        rvdir='newer', rvlimit='1', explaintext='1',
                        redirects='0')
    pages = data.get('query', {}).get('pages', [])
    if len(pages) != 1 or pages[0].get('missing'):
        return None
    page = pages[0]
    revisions = page.get('revisions', [])
    if not revisions or _date(revisions[0]['timestamp']) < created_after:
        return None
    text = page.get('extract', '').strip()
    if len(text.split()) < minimum_words:
        return None
    return dict(pageid=pageid, title=title, text=text,
                creation_timestamp=revisions[0]['timestamp'],
                category=category, source='en.wikipedia.org',
                extraction='MediaWiki categorymembers + plain-text extracts')


def _checkpoint(path: Path) -> list[dict]:
    if not path.exists():
        return []
    payload = path.read_bytes()
    last_newline = payload.rfind(b'\n') + 1
    if last_newline != len(payload):
        path.write_bytes(payload[:last_newline])
        payload = payload[:last_newline]
    return [json.loads(line) for line in payload.splitlines()]


def _append(path: Path, row: dict) -> None:
    with path.open('a', encoding='utf-8') as stream:
        stream.write(json.dumps(row, ensure_ascii=False) + '\n')
        stream.flush()
        os.fsync(stream.fileno())


def collect(output: Path, *, records: int = 1000,
            categories: tuple[str, ...] = DEFAULT_CATEGORIES,
            created_after: str = '2023-01-01T00:00:00Z',
            minimum_words: int = 128, max_depth: int = 3,
            max_pages: int = 20000, interval: float = 1.0,
            contact: str = 'https://github.com/DengMXuan/SD_MIA') -> Path:
    """Resume collection in an isolated output folder; publish JSONL when full."""
    output = Path(output).resolve()
    if records < 600 or minimum_words < 128 or max_depth < 0 or max_pages < records:
        raise ValueError('collect >=600 pages, >=128 words, with a sufficient page budget')
    if not categories or not all(c.startswith('Category:') for c in categories):
        raise ValueError('event category titles must start with Category:')
    output.parent.mkdir(parents=True, exist_ok=True)
    checkpoint = output.with_suffix('.checkpoint.jsonl')
    plan_file = output.with_suffix('.plan.json')
    plan = dict(categories=list(categories), created_after=created_after,
                minimum_words=minimum_words, max_depth=max_depth,
                max_pages=max_pages, records=records, source_api=API)
    if plan_file.exists():
        previous = json.loads(plan_file.read_text())
        if previous != plan:
            # Reducing the requested count, adding categories, or reordering
            # them retains the inspected-page checkpoint and accepted rows.
            if (output.exists() or records > previous.get('records', 0)
                    or not set(previous.get('categories', [])).issubset(plan['categories'])
                    or {k: v for k, v in previous.items() if k not in ('records', 'categories')} !=
                       {k: v for k, v in plan.items() if k not in ('records', 'categories')}):
                raise ValueError('collection settings changed; use another output path')
            plan_file.write_text(json.dumps(plan, indent=2) + '\n')
    else:
        plan_file.write_text(json.dumps(plan, indent=2) + '\n')
    attempts = _checkpoint(checkpoint)
    seen_ids = {row['pageid'] for row in attempts}
    accepted = [row['record'] for row in attempts if row.get('record') is not None]
    if output.exists():
        existing = [json.loads(line) for line in output.read_text().splitlines()]
        if existing != accepted[:records] or len(existing) != records:
            raise ValueError('published auxiliary data disagrees with checkpoint')
        return output
    client = Client(contact, interval)
    created_after_date = _date(created_after)
    queue = deque((category, 0) for category in categories)
    visited_categories = set()
    while queue and len(accepted) < records and len(seen_ids) < max_pages:
        category, depth = queue.popleft()
        if category in visited_categories:
            continue
        visited_categories.add(category)
        for member in _members(client, category):
            if member.get('ns') == 14 and depth < max_depth:
                queue.append((member['title'], depth + 1))
                continue
            if member.get('ns') != 0:
                continue
            pageid = int(member['pageid'])
            if pageid in seen_ids:
                continue
            record = _page(client, pageid, member['title'], category,
                           created_after=created_after_date, minimum_words=minimum_words)
            _append(checkpoint, dict(pageid=pageid, record=record))
            seen_ids.add(pageid)
            if record is not None:
                accepted.append(record)
                if len(accepted) % 50 == 0:
                    print(f'accepted={len(accepted)} inspected={len(seen_ids)}', flush=True)
            if len(accepted) >= records or len(seen_ids) >= max_pages:
                break
    if len(accepted) < records:
        raise RuntimeError(f'only {len(accepted)}/{records} eligible event pages; '
                           f'{len(seen_ids)} inspected. Checkpoint preserved at {checkpoint}')
    temporary = output.with_suffix('.tmp')
    temporary.write_text(''.join(json.dumps(row, ensure_ascii=False) + '\n'
                                 for row in accepted[:records]))
    temporary.replace(output)
    return output
