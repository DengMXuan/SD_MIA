"""Construct frozen NART-style benchmark pools: WikiTection, NewsTection, ArXivTection.

The construction follows NART (Tan et al., NDSS 2026): every pool document must
postdate the pretraining cutoff of every candidate target model, so that a
fine-tuned "member" record cannot already sit in the pretraining corpus. The
default window 2026-05-01..2026-08-29 postdates Qwen3-8B (2025-07), Granite 4.0
(released 2025-10), Qwen3.5 (released 2026-02) and Qwen3.6 (released 2026-04,
assumed cutoff <= 2026-03).

Each subcommand writes ``pool.jsonl`` plus a SHA-256-anchored
``pool.manifest.json`` under ``experiments/data/nart_benchmarks/<name>/``.
Raw text is persisted because the pool must be re-tokenized per target model;
all sources are public corpora with per-record provenance URLs.

Per-model token banding, hash deduplication against token IDs, and the
member/nonmember/auxiliary three-class split happen at load time in
``nart_data.build_nart_split``, not here; this module only freezes documents.
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import html as html_module
import io
import json
import random
import re
import threading
import time
import urllib.parse
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from html.parser import HTMLParser
from pathlib import Path
from typing import Any, Iterable
from urllib.error import HTTPError

from warcio.archiveiterator import ArchiveIterator


USER_AGENT = "SD-MIA-research/0.1 (controlled academic benchmark)"
DATA_ROOT = Path("experiments/data/nart_benchmarks")
DEFAULT_WINDOW = ("2026-05-01T00:00:00Z", "2026-08-29T23:59:59Z")
NEWS_MONTHS = ("2026-05", "2026-06", "2026-07", "2026-08")
WIKI_API = "https://en.wikipedia.org/w/api.php"
CCNEWS_PATHS = "https://data.commoncrawl.org/crawl-data/CC-NEWS/{year_month_path}/warc.paths.gz"
CCNEWS_BASE = "https://data.commoncrawl.org/"
ARXIV_API = "https://export.arxiv.org/api/query"
ARXIV_HTML = "https://arxiv.org/html/{arxiv_id}"
AR5IV_HTML = "https://ar5iv.labs.arxiv.org/html/{arxiv_id}"

NART_PROVENANCE = (
    "NART-style post-cutoff benchmark (Tan et al., NDSS 2026); window postdates "
    "the Qwen3-8B 2025-07 cutoff and all repo target models, assuming the "
    "Qwen3.6 cutoff <= 2026-03 from its 2026-04-15 release"
)


def _request(
    url: str,
    headers: dict[str, str] | None = None,
    attempts: int = 6,
    timeout: int = 120,
    sleep: float = 1.0,
) -> Any:
    request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT, **(headers or {})})
    for attempt in range(attempts):
        try:
            return urllib.request.urlopen(request, timeout=timeout)
        except HTTPError as error:
            if attempt + 1 == attempts:
                raise
            retry_after = error.headers.get("Retry-After")
            delay = float(retry_after) if retry_after else min(120.0, sleep * 2**attempt)
            time.sleep(delay)
        except Exception:
            if attempt + 1 == attempts:
                raise
            time.sleep(min(120.0, sleep * 2**attempt))
    raise AssertionError("unreachable")


def _request_json(url: str, **kwargs: Any) -> dict[str, Any]:
    with _request(url, **kwargs) as response:
        return json.load(response)


class ArticleTextExtractor(HTMLParser):
    """Collect readable block text from an article HTML page.

    arxiv.org/html and ar5iv wrap the paper body in ``<article>``; when such an
    element exists, only text inside it is kept, which drops page chrome such
    as feedback banners and navigation.
    """

    _SKIP = {
        "script", "style", "noscript", "svg", "head", "nav", "footer",
        "header", "aside", "form", "iframe", "button", "select", "label",
        "figure", "table",
    }
    _BLOCK = {
        "p", "h1", "h2", "h3", "h4", "h5", "li", "blockquote",
        "figcaption", "dd", "dt", "pre", "div", "section", "article", "br",
    }

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.title = ""
        self._skip_depth = 0
        self._block_parts: list[str] = []
        self._blocks: list[str] = []
        self._article_blocks: list[str] = []
        self._article_depth = 0
        self._in_title = False

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag in ("script", "style", "noscript", "svg"):
            self._skip_depth += 1
            return
        if self._skip_depth:
            return
        if tag == "title":
            self._in_title = True
        elif tag == "head":
            self._skip_depth += 1
        elif tag == "article":
            self._article_depth += 1
        elif tag in self._BLOCK:
            self._flush_block()
        elif tag == "br":
            self._flush_block()

    def handle_endtag(self, tag: str) -> None:
        if tag in ("script", "style", "noscript", "svg", "head"):
            self._skip_depth = max(0, self._skip_depth - 1)
            return
        if self._skip_depth:
            return
        if tag == "title":
            self._in_title = False
        elif tag == "article":
            self._flush_block()
            self._article_depth = max(0, self._article_depth - 1)
        elif tag in self._BLOCK:
            self._flush_block()

    def handle_data(self, data: str) -> None:
        if self._skip_depth:
            return
        if self._in_title:
            self.title += data
        else:
            self._block_parts.append(data)

    def _flush_block(self) -> None:
        text = re.sub(r"\s+", " ", " ".join(self._block_parts)).strip()
        self._block_parts = []
        if text:
            self._blocks.append(text)
            if self._article_depth:
                self._article_blocks.append(text)

    def get_text(self) -> str:
        self._flush_block()
        blocks = self._article_blocks if self._article_blocks else self._blocks
        if len(blocks) > 12:
            # large pages: drop menu/link-list lines, keep real paragraphs
            blocks = [
                block
                for block in blocks
                if len(block.split()) >= 8
                or (len(block.split()) >= 5 and block[-1] in ".!?")
            ] or blocks
        return "\n".join(blocks)

    def get_title(self) -> str:
        return re.sub(r"\s+", " ", self.title).strip()


def _html_to_article(raw_html: bytes) -> tuple[str, str]:
    charset = "utf-8"
    match = re.search(rb'charset=["\']?([\w-]+)', raw_html[:2048])
    if match:
        charset = match.group(1).decode("ascii", errors="replace")
    text_html = raw_html.decode(charset, errors="replace")
    extractor = ArticleTextExtractor()
    try:
        extractor.feed(text_html)
        extractor.close()
    except Exception:
        pass
    return extractor.get_title(), extractor.get_text()


def _printable_ratio(text: str) -> float:
    if not text:
        return 0.0
    printable = sum(
        character.isprintable() and ord(character) < 0x2500 for character in text
    )
    return printable / len(text)


_COMMON_ENGLISH_WORDS = {
    "the", "and", "of", "to", "in", "a", "is", "that", "for", "on", "with",
    "as", "by", "at", "from", "said", "has", "are", "was", "an", "be", "after",
    "it", "he", "she", "they", "their", "have", "not", "this", "will",
}


def _english_score(text: str) -> float:
    words = re.findall(r"[a-z']+", text.lower())
    if not words:
        return 0.0
    common = sum(1 for word in words if word in _COMMON_ENGLISH_WORDS)
    return common / len(words)


def _usable_text(
    text: str, min_chars: int, max_chars: int, min_long_paragraphs: int = 0
) -> bool:
    if not (min_chars <= len(text) <= max_chars):
        return False
    if _printable_ratio(text) < 0.85:
        return False
    ascii_ratio = sum(1 for character in text if ord(character) < 128) / len(text)
    if ascii_ratio < 0.98 or _english_score(text) < 0.08:
        return False
    if min_long_paragraphs:
        # real articles carry many >=20-word paragraphs; headline-list and
        # directory pages do not
        long_paragraphs = sum(
            1 for line in text.split("\n") if len(line.split()) >= 20
        )
        if long_paragraphs < min_long_paragraphs:
            return False
    words = text.split()
    return len(words) >= 60


_SKIP_URL = re.compile(
    r"/(tag|tags|category|author|authors|search|video|videos|live|slideshow|"
    r"photo|gallery|login|signin|signup|account|subscribe|newsletter|"
    r"podcast|episodes|topic|topics|section|archive|page/\d+)/?"
    r"($|[?#])",
    re.IGNORECASE,
)


def _sha256_hex(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _write_pool(
    output: Path,
    records: list[dict[str, Any]],
    manifest: dict[str, Any],
) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
    payload = output.read_bytes()
    manifest = {
        **manifest,
        "records": len(records),
        "jsonl_sha256": _sha256_hex(payload),
        "retrieved_unix_time": time.time(),
    }
    manifest_path = output.with_suffix(".manifest.json")
    manifest_path.write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(
        json.dumps(
            {"output": str(output), "records": len(records), "sha256": manifest["jsonl_sha256"]},
            indent=2,
        ),
        flush=True,
    )


# ---------------------------------------------------------------------------
# WikiTection: English Wikipedia pages first created inside the window.
# ---------------------------------------------------------------------------


def _wiki_creation_events(start: str, end: str) -> Iterable[dict[str, Any]]:
    continuation: str | None = None
    while True:
        parameters: dict[str, Any] = {
            "action": "query",
            "list": "logevents",
            "letype": "create",
            "lenamespace": 0,
            "lestart": end,
            "leend": start,
            "ledir": "older",
            "lelimit": "max",
            "leprop": "ids|title|timestamp",
        }
        if continuation is not None:
            parameters["lecontinue"] = continuation
        payload = _request_json(f"{WIKI_API}?{urllib.parse.urlencode({'format': 'json', 'formatversion': 2, **parameters})}", sleep=0.6)
        batch = payload.get("query", {}).get("logevents", [])
        yield from batch
        continuation = payload.get("continue", {}).get("lecontinue")
        if not batch or continuation is None:
            return
        time.sleep(0.5)


def _fetch_wiki_fulltext(page_id: int) -> str | None:
    """Rendered full text via action=parse; TextExtracts caps exchars at 1200."""
    payload = _request_json(
        f"{WIKI_API}?{urllib.parse.urlencode({
            'format': 'json', 'formatversion': 2,
            'action': 'parse',
            'pageid': page_id,
            'prop': 'text',
            'redirects': 0,
        })}",
        sleep=0.6,
    )
    parse = payload.get("parse", {})
    raw_html = parse.get("text", "")
    if not raw_html:
        return None
    _, text = _html_to_article(raw_html.encode("utf-8"))
    return re.sub(r" +\n", "\n", text).strip()


def _wiki_page_record(
    page: dict[str, Any],
    creation: dict[str, Any],
    min_chars: int,
    max_chars: int,
) -> dict[str, Any] | None:
    text = _fetch_wiki_fulltext(int(page["pageid"]))
    time.sleep(0.25)
    if text is None or not _usable_text(text, min_chars, max_chars):
        return None
    digest = _sha256_hex(text.encode("utf-8"))
    return {
        "record_id": f"wikitection:{digest[:16]}",
        "source": "en.wikipedia.org",
        "title": page["title"],
        "creation_timestamp": creation["timestamp"],
        "snapshot_revision": int(page.get("lastrevid", 0)),
        "snapshot_timestamp": page.get("touched", ""),
        "canonical_url": page.get("fullurl", ""),
        "text_sha256": digest,
        "text": text,
        "_text_digest": digest,
    }


def build_wikitection(args: argparse.Namespace) -> None:
    events: list[dict[str, Any]] = []
    for event in _wiki_creation_events(args.window_start, args.window_end):
        events.append(event)
        if len(events) >= args.candidate_limit:
            break
    print(f"creation events: {len(events)}", flush=True)

    records: list[dict[str, Any]] = []
    seen_text: set[str] = set()
    dup_index = NearDuplicateIndex()
    event_by_page = {int(event["pageid"]): event for event in events if event.get("pageid")}
    page_ids = list(event_by_page)

    # Phase 1: cheap info prefilter (50 pageids per request) drops redirects,
    # disambiguation pages, and two-line stubs before the expensive full-text
    # renders. Requests run through a small parallel pool with per-request
    # pacing so a throttling API degrades throughput instead of stalling.
    survivors: list[dict[str, Any]] = []
    survivor_lock = threading.Lock()
    info_chunks = [
        page_ids[start : start + 50] for start in range(0, len(page_ids), 50)
    ]
    info_workers = max(1, min(args.parallel, 16))
    scanned = 0

    def _prefilter(chunk: list[int]) -> int:
        payload = _request_json(
            f"{WIKI_API}?{urllib.parse.urlencode({
                'format': 'json', 'formatversion': 2,
                'action': 'query',
                'pageids': '|'.join(str(page_id) for page_id in chunk),
                'prop': 'info|pageprops',
                'inprop': 'url',
            })}",
            sleep=0.5,
            attempts=8,
        )
        kept = 0
        for page in payload.get("query", {}).get("pages", []):
            if page.get("missing") or page.get("redirect") or "disambiguation" in page.get("pageprops", {}):
                continue
            if int(page.get("length", 0)) < 1200:
                continue
            with survivor_lock:
                survivors.append(page)
            kept += 1
        return kept

    with ThreadPoolExecutor(max_workers=info_workers) as executor:
        futures = [executor.submit(_prefilter, chunk) for chunk in info_chunks]
        for index, future in enumerate(as_completed(futures), start=1):
            scanned += len(info_chunks[index - 1]) if index - 1 < len(info_chunks) else 0
            if index % 200 == 0 or index == len(futures):
                print(
                    f"prefiltered_chunks={index}/{len(futures)} survivors={len(survivors)}",
                    flush=True,
                )
    print(f"prefilter done: {len(survivors)} survivors from {len(page_ids)} pages", flush=True)

    # Phase 2: full-text renders across a large worker pool, submitted in
    # chunks so the queue stays bounded and we can stop as soon as the target
    # record count is reached.
    workers = max(1, args.parallel)
    with ThreadPoolExecutor(max_workers=workers) as executor:
        for chunk_start in range(0, len(survivors), workers * 4):
            if len(records) >= args.records:
                break
            chunk = survivors[chunk_start : chunk_start + workers * 4]
            futures = [
                executor.submit(
                    _wiki_page_record,
                    page,
                    event_by_page[int(page["pageid"])],
                    args.min_chars,
                    args.max_chars,
                )
                for page in chunk
            ]
            for future in as_completed(futures):
                record = future.result()
                if record is None:
                    continue
                digest = record["_text_digest"]
                if digest in seen_text or dup_index.is_duplicate(record["text"]):
                    continue
                seen_text.add(digest)
                dup_index.add(record["text"])
                records.append(record)
            print(
                f"tried={min(chunk_start + len(chunk), len(survivors))} usable={len(records)}",
                flush=True,
            )
    for record in records:
        del record["_text_digest"]
    if len(records) < args.records:
        raise RuntimeError(
            f"Only {len(records)} usable pages from {len(events)} creation events"
        )
    records = records[: args.records]
    _write_pool(
        DATA_ROOT / "wikitection" / "pool.jsonl",
        records,
        {
            "benchmark": "wikitection",
            "dataset": "English Wikipedia main-namespace pages first created in the window",
            "source_api": WIKI_API,
            "creation_interval_inclusive": {"start": args.window_start, "end": args.window_end},
            "timestamp_semantics": "page first-creation time (MediaWiki create log)",
            "license": "CC BY-SA 4.0; per-page attribution URLs in JSONL",
            "provenance": NART_PROVENANCE,
            "selection": {
                "creation_events_considered": len(events),
                "minimum_clean_characters": args.min_chars,
                "maximum_clean_characters": args.max_chars,
                "namespace": 0,
                "disambiguation_pages_excluded": True,
                "exact_clean_text_deduplicated": True,
            },
        },
    )


# ---------------------------------------------------------------------------
# NewsTection: CC-NEWS warc segments captured inside the window.
# ---------------------------------------------------------------------------


def build_newstection(args: argparse.Namespace) -> None:
    segment_paths: list[str] = []
    for month in args.months:
        year, month_number = month.split("-")
        year_month_path = f"{year}/{month_number}"
        with _request(CCNEWS_PATHS.format(year_month_path=year_month_path), sleep=1.0, timeout=60) as response:
            listing = gzip.decompress(response.read()).decode("utf-8")
        paths = [line.strip() for line in listing.splitlines() if line.strip()]
        print(f"{month}: {len(paths)} segments", flush=True)
        segment_paths.extend(paths)
    random.Random(args.seed).shuffle(segment_paths)
    segment_paths = segment_paths[: args.max_segments]
    print(f"segments queued: {len(segment_paths)}", flush=True)

    records: list[dict[str, Any]] = []
    seen_text: set[str] = set()
    seen_url: set[str] = set()
    dup_index = NearDuplicateIndex()
    for segment_index, segment in enumerate(segment_paths):
        if len(records) >= args.records:
            break
        try:
            fetched = _collect_news_segment(
                segment, records, seen_text, seen_url, dup_index, args,
                needed=args.records - len(records),
            )
        except Exception as error:  # noqa: BLE001 - a broken segment must not stop the pool
            print(f"segment failed, skipping: {segment} ({error})", flush=True)
            continue
        print(
            f"segment {segment_index + 1}/{len(segment_paths)}: usable={fetched} "
            f"total={len(records)}",
            flush=True,
        )
    if len(records) < args.records:
        raise RuntimeError(
            f"Only {len(records)} news records from {len(segment_paths)} segments"
        )
    records = records[: args.records]
    _write_pool(
        DATA_ROOT / "newstection" / "pool.jsonl",
        records,
        {
            "benchmark": "newstection",
            "dataset": "CC-NEWS article pages captured in the window",
            "source": "Common Crawl CC-NEWS warc segments",
            "months": list(args.months),
            "creation_interval_inclusive": {"start": args.window_start, "end": args.window_end},
            "timestamp_semantics": "CC-NEWS warc capture time, not original publication time",
            "license": "CC-NEWS is distributed by Common Crawl for research; "
            "source URLs retained for provenance",
            "provenance": NART_PROVENANCE,
            "selection": {
                "segments_queued": len(segment_paths),
                "minimum_clean_characters": args.min_chars,
                "maximum_clean_characters": args.max_chars,
                "minimum_printable_ratio": 0.85,
                "minimum_word_count": 60,
                "navigation_or_index_urls_excluded": True,
                "exact_clean_text_deduplicated": True,
            },
        },
    )


def _collect_news_segment(
    segment: str,
    records: list[dict[str, Any]],
    seen_text: set[str],
    seen_url: set[str],
    dup_index: NearDuplicateIndex,
    args: argparse.Namespace,
    needed: int,
) -> int:
    usable = 0
    with _request(CCNEWS_BASE + segment, sleep=1.0, timeout=300) as response:
        stream = gzip.GzipFile(fileobj=response)
        for record in ArchiveIterator(stream):
            if usable >= needed:
                break
            if record.rec_type != "response":
                continue
            http_headers = record.http_headers
            if http_headers is None or http_headers.get_statuscode() != "200":
                continue
            content_type = (http_headers.get_header("Content-Type") or "").lower()
            if "text/html" not in content_type and "application/xhtml" not in content_type:
                continue
            url = record.rec_headers.get_header("WARC-Target-URI") or ""
            path = urllib.parse.urlparse(url).path
            if not path or path == "/" or _SKIP_URL.search(path):
                continue
            host = urllib.parse.urlparse(url).netloc.lower().removeprefix("www.")
            if not host:
                continue
            canonical = f"{host}{path}"
            if canonical in seen_url:
                continue
            body = record.content_stream().read()
            encoding = (http_headers.get_header("Content-Encoding") or "").lower()
            if "gzip" in encoding:
                try:
                    body = gzip.decompress(body)
                except OSError:
                    continue
            title, text = _html_to_article(body)
            text = text[: args.max_chars]
            if not _usable_text(
                text, args.min_chars, args.max_chars, min_long_paragraphs=5
            ):
                continue
            digest = _sha256_hex(text.encode("utf-8"))
            if digest in seen_text:
                continue
            if dup_index.is_duplicate(text):
                continue
            seen_text.add(digest)
            seen_url.add(canonical)
            dup_index.add(text)
            capture_time = record.rec_headers.get_header("WARC-Date") or ""
            records.append(
                {
                    "record_id": f"newstection:{digest[:16]}",
                    "source": host,
                    "title": title or host,
                    "creation_timestamp": capture_time,
                    "snapshot_revision": 0,
                    "snapshot_timestamp": capture_time,
                    "canonical_url": url,
                    "text_sha256": digest,
                    "text": text,
                    "segment": segment,
                }
            )
            usable += 1
    return usable


# ---------------------------------------------------------------------------
# ArXivTection: papers submitted in the window, full text >= 2048 tokens.
# ---------------------------------------------------------------------------


def _arxiv_atom_entries(
    window_start: str, window_end: str, wanted: int, per_page: int, sleep: float
) -> list[dict[str, str]]:
    import xml.etree.ElementTree as ElementTree

    atom_ns = "{http://www.w3.org/2005/Atom}"
    arxiv_ns = "{http://arxiv.org/schemas/atom}"
    query_start = window_start.replace("-", "").replace(":", "").replace("T", "")[:12]
    query_end = window_end.replace("-", "").replace(":", "").replace("T", "")[:12]
    entries: list[dict[str, str]] = []
    start = 0
    consecutive_failures = 0
    while len(entries) < wanted:
        url = (
            f"{ARXIV_API}?{urllib.parse.urlencode({
                'search_query': f'submittedDate:[{query_start} TO {query_end}]',
                'start': start,
                'max_results': per_page,
                'sortBy': 'submittedDate',
                'sortOrder': 'ascending',
            })}"
        )
        try:
            with _request(url, sleep=sleep, timeout=90) as response:
                payload = response.read()
        except Exception:
            # the arXiv API intermittently 500s on deep pagination; back off
            # hard and degrade to partial candidate lists instead of dying
            consecutive_failures += 1
            if consecutive_failures >= 5:
                break
            time.sleep(min(300.0, 30.0 * consecutive_failures))
            continue
        consecutive_failures = 0
        try:
            root = ElementTree.fromstring(payload)
        except ElementTree.ParseError:
            time.sleep(max(sleep, 10.0))
            continue
        batch = root.findall(f"{atom_ns}entry")
        if not batch:
            break
        for entry in batch:
            arxiv_id = (entry.findtext(f"{atom_ns}id") or "").rsplit("/", 1)[-1]
            if not arxiv_id:
                continue
            entries.append(
                {
                    "arxiv_id": arxiv_id,
                    "published": (entry.findtext(f"{atom_ns}published") or "")[:10],
                    "title": re.sub(
                        r"\s+", " ", entry.findtext(f"{atom_ns}title") or ""
                    ).strip(),
                    "category": (
                        (entry.find(f"{arxiv_ns}primary_category") is not None
                         and entry.find(f"{arxiv_ns}primary_category").get("term", ""))
                        or ""
                    ),
                }
            )
        start += per_page
        time.sleep(sleep)
    return entries[:wanted]


def _build_arxiv_record(
    entry: dict[str, str], args: argparse.Namespace
) -> dict[str, Any] | None:
    raw = _fetch_arxiv_fulltext(entry["arxiv_id"])
    if raw is None:
        return None
    title, text = _html_to_article(raw)
    text = text[: args.max_chars]
    if not _usable_text(text, args.min_chars, args.max_chars):
        return None
    digest = _sha256_hex(text.encode("utf-8"))
    return {
        "record_id": f"arxivtection:{digest[:16]}",
        "source": "arxiv.org",
        "title": title or entry["title"],
        "creation_timestamp": entry["published"],
        "snapshot_revision": entry["arxiv_id"].split("v")[-1] or "1",
        "snapshot_timestamp": entry["published"],
        "canonical_url": f"https://arxiv.org/abs/{entry['arxiv_id']}",
        "text_sha256": digest,
        "text": text,
        "arxiv_id": entry["arxiv_id"],
        "primary_category": entry["category"],
        "_text_digest": digest,
    }


def build_arxivtection(args: argparse.Namespace) -> None:
    entries = _arxiv_atom_entries(
        args.window_start, args.window_end, args.candidate_limit, 200, 3.0
    )
    print(f"arxiv candidates: {len(entries)}", flush=True)
    random.Random(args.seed).shuffle(entries)

    records: list[dict[str, Any]] = []
    seen_text: set[str] = set()
    dup_index = NearDuplicateIndex()
    workers = max(1, args.parallel)
    with ThreadPoolExecutor(max_workers=workers) as executor:
        for chunk_start in range(0, len(entries), workers * 4):
            if len(records) >= args.records:
                break
            chunk = entries[chunk_start : chunk_start + workers * 4]
            futures = [
                executor.submit(_build_arxiv_record, entry, args) for entry in chunk
            ]
            for future in as_completed(futures):
                record = future.result()
                if record is None:
                    continue
                digest = record["_text_digest"]
                if digest in seen_text or dup_index.is_duplicate(record["text"]):
                    continue
                seen_text.add(digest)
                dup_index.add(record["text"])
                records.append(record)
            print(
                f"tried={min(chunk_start + len(chunk), len(entries))} usable={len(records)}",
                flush=True,
            )
    for record in records:
        del record["_text_digest"]
    if len(records) < args.records:
        raise RuntimeError(
            f"Only {len(records)} arxiv records from {len(entries)} candidates"
        )
    records = records[: args.records]
    _write_pool(
        DATA_ROOT / "arxivtection" / "pool.jsonl",
        records,
        {
            "benchmark": "arxivtection",
            "dataset": "arXiv papers submitted in the window, HTML full text",
            "source_apis": [ARXIV_HTML, AR5IV_HTML],
            "creation_interval_inclusive": {"start": args.window_start, "end": args.window_end},
            "timestamp_semantics": "arXiv submission date (v1)",
            "license": "arXiv per-paper licenses; abstracts and metadata under "
            "arXiv terms; source pages retained for provenance",
            "provenance": NART_PROVENANCE,
            "selection": {
                "candidates_considered": len(entries),
                "minimum_clean_characters": args.min_chars,
                "maximum_clean_characters": args.max_chars,
                "html_sources_tried": ["arxiv.org/html", "ar5iv.labs.arxiv.org"],
                "exact_clean_text_deduplicated": True,
            },
        },
    )


def _fetch_arxiv_fulltext(arxiv_id: str) -> bytes | None:
    for template in (ARXIV_HTML, AR5IV_HTML):
        url = template.format(arxiv_id=arxiv_id)
        try:
            with _request(url, sleep=2.0, timeout=90) as response:
                if response.status == 200:
                    return response.read()
        except HTTPError as error:
            if error.code == 404:
                continue
            time.sleep(5.0)
        except Exception:
            time.sleep(5.0)
    return None


# ---------------------------------------------------------------------------
# Pool-level near-duplicate removal (syndication/boilerplate defense)
# ---------------------------------------------------------------------------


def _word_shingles(text: str, n: int = 13) -> list[int]:
    words = re.sub(r"[^a-z0-9 ]+", " ", text.lower()).split()
    return [
        hash(" ".join(words[index : index + n]))
        for index in range(max(0, len(words) - n + 1))
    ]


class NearDuplicateIndex:
    """Streaming near-duplicate detector over word-level 13-gram shingles.

    Exact-hash deduplication misses syndicated articles and pages whose
    boilerplate dominates; these near-duplicates otherwise end up in different
    split classes and trip the 13-gram cross-split gate. Owner lists are capped
    so ubiquitous boilerplate shingles cannot blow up memory or runtime; true
    duplicates are still caught through their rarer shared shingles.
    """

    _OWNER_CAP = 1000

    def __init__(self, threshold: float = 0.5) -> None:
        self.threshold = threshold
        self._owners: dict[int, list[int]] = {}
        self._kept = 0

    def is_duplicate(self, text: str) -> bool:
        shingles = _word_shingles(text)
        if not shingles:
            return False
        if not self._kept:
            return False
        hits: dict[int, int] = {}
        for shingle in shingles:
            for owner in self._owners.get(shingle, ()):
                hits[owner] = hits.get(owner, 0) + 1
        return bool(hits) and max(hits.values()) >= self.threshold * len(shingles)

    def add(self, text: str) -> None:
        owner_id = self._kept
        for shingle in set(_word_shingles(text)):
            bucket = self._owners.setdefault(shingle, [])
            if len(bucket) < self._OWNER_CAP:
                bucket.append(owner_id)
        self._kept += 1


def near_duplicate_filter(records: list[dict[str, Any]], threshold: float = 0.5) -> list[dict[str, Any]]:
    """Drop records sharing >= threshold of their 13-gram shingles with a kept one."""
    index = NearDuplicateIndex(threshold)
    keep: list[dict[str, Any]] = []
    for record in records:
        text = str(record.get("text", ""))
        if index.is_duplicate(text):
            continue
        index.add(text)
        keep.append(record)
    return keep


def dedupe_pool(args: argparse.Namespace) -> None:
    path = DATA_ROOT / args.benchmark / "pool.jsonl"
    manifest_path = path.with_suffix(".manifest.json")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    raw_bytes = path.read_bytes()
    if hashlib.sha256(raw_bytes).hexdigest() != manifest["jsonl_sha256"]:
        raise RuntimeError("Pool hash does not match its manifest before dedupe")
    records = [json.loads(line) for line in raw_bytes.decode("utf-8").splitlines() if line]
    kept = near_duplicate_filter(records, threshold=args.threshold)
    print(f"{args.benchmark}: {len(records)} -> {len(kept)} after near-duplicate removal", flush=True)
    manifest = {
        **manifest,
        "selection": {
            **manifest.get("selection", {}),
            "near_duplicates_removed": len(records) - len(kept),
            "near_duplicate_shingle_n": 13,
            "near_duplicate_threshold": args.threshold,
        },
    }
    _write_pool(path, kept, manifest)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    window_defaults = {
        "window_start": DEFAULT_WINDOW[0],
        "window_end": DEFAULT_WINDOW[1],
    }
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--window-start", default=window_defaults["window_start"])
    common.add_argument("--window-end", default=window_defaults["window_end"])
    common.add_argument("--seed", type=int, default=20260830)
    common.add_argument("--min-chars", type=int, default=700)
    common.add_argument("--max-chars", type=int, default=9000)

    wiki = subparsers.add_parser(
        "wiki", parents=[common], help="freeze the WikiTection pool"
    )
    wiki.add_argument("--records", type=int, default=6600)
    wiki.add_argument("--candidate-limit", type=int, default=200_000)
    wiki.add_argument("--parallel", type=int, default=48)
    wiki.set_defaults(handler=build_wikitection)

    news = subparsers.add_parser(
        "news", parents=[common], help="freeze the NewsTection pool"
    )
    news.add_argument("--months", nargs="+", default=list(NEWS_MONTHS))
    news.add_argument("--records", type=int, default=6600)
    news.add_argument("--max-segments", type=int, default=40)
    news.add_argument("--parallel", type=int, default=4)
    news.set_defaults(handler=build_newstection)

    arxiv = subparsers.add_parser(
        "arxiv", parents=[common], help="freeze the ArXivTection pool"
    )
    arxiv.add_argument("--records", type=int, default=6600)
    arxiv.add_argument("--candidate-limit", type=int, default=14_000)
    arxiv.add_argument("--parallel", type=int, default=8)
    arxiv.set_defaults(handler=build_arxivtection)

    dedupe = subparsers.add_parser(
        "dedupe", parents=[common], help="remove near-duplicate records from a frozen pool"
    )
    dedupe.add_argument("--benchmark", required=True)
    dedupe.add_argument("--threshold", type=float, default=0.5)
    dedupe.set_defaults(handler=dedupe_pool)

    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.handler(args)


if __name__ == "__main__":
    main()
