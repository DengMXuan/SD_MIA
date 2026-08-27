from __future__ import annotations

import argparse
import hashlib
import html
import json
import re
import time
import urllib.parse
import urllib.request
from urllib.error import HTTPError
from pathlib import Path
from typing import Any


API_URL = "https://en.wikipedia.org/w/api.php"
USER_AGENT = "SD-MIA-research/0.1 (controlled academic benchmark)"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Snapshot English Wikipedia pages created in a fixed interval."
    )
    parser.add_argument("--start", default="2025-09-30T23:59:59Z")
    parser.add_argument("--end", default="2025-09-01T00:00:00Z")
    parser.add_argument("--records", type=int, default=600)
    parser.add_argument("--min-chars", type=int, default=900)
    parser.add_argument("--candidate-limit", type=int, default=5000)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("experiments/data/wikipedia_newpages_2025_09.jsonl"),
    )
    return parser.parse_args()


def _request(parameters: dict[str, Any], attempts: int = 7) -> dict[str, Any]:
    query = urllib.parse.urlencode({"format": "json", "formatversion": 2, **parameters})
    request = urllib.request.Request(
        f"{API_URL}?{query}", headers={"User-Agent": USER_AGENT}
    )
    for attempt in range(attempts):
        try:
            with urllib.request.urlopen(request, timeout=45) as response:
                return json.load(response)
        except HTTPError as error:
            if attempt + 1 == attempts:
                raise
            retry_after = error.headers.get("Retry-After")
            delay = float(retry_after) if retry_after else min(60.0, 3.0 * 2**attempt)
            time.sleep(delay)
        except Exception:
            if attempt + 1 == attempts:
                raise
            time.sleep(min(60.0, 2.0 * 2**attempt))
    raise AssertionError("unreachable")


def _creation_events(start: str, end: str, limit: int) -> list[dict[str, Any]]:
    events: list[dict[str, Any]] = []
    continuation: str | None = None
    while len(events) < limit:
        parameters: dict[str, Any] = {
            "action": "query",
            "list": "logevents",
            "letype": "create",
            "lenamespace": 0,
            "lestart": start,
            "leend": end,
            "ledir": "older",
            "lelimit": "max",
            "leprop": "ids|title|timestamp",
        }
        if continuation is not None:
            parameters["lecontinue"] = continuation
        payload = _request(parameters)
        batch = payload.get("query", {}).get("logevents", [])
        events.extend(batch)
        continuation = payload.get("continue", {}).get("lecontinue")
        if not batch or continuation is None:
            break
        time.sleep(0.5)
    return events[:limit]


def _normalize_text(text: str) -> str:
    text = html.unescape(text).replace("\u00a0", " ")
    text = re.sub(r"\s+", " ", text).strip()
    return text


def _usable(text: str, min_chars: int) -> bool:
    if len(text) < min_chars:
        return False
    letters = sum(character.isalpha() for character in text)
    return letters / max(1, len(text)) >= 0.65


def _fetch_pages(events: list[dict[str, Any]], min_chars: int, wanted: int) -> list[dict[str, Any]]:
    event_by_page = {int(event["pageid"]): event for event in events if event.get("pageid")}
    page_ids = list(event_by_page)
    records: list[dict[str, Any]] = []
    seen_text: set[str] = set()
    for start in range(0, len(page_ids), 20):
        if len(records) >= wanted:
            break
        batch = page_ids[start : start + 20]
        payload = _request(
            {
                "action": "query",
                "pageids": "|".join(str(page_id) for page_id in batch),
                "prop": "extracts|revisions|pageprops|info",
                "explaintext": 1,
                "exintro": 1,
                "exlimit": "max",
                "rvprop": "ids|timestamp",
                "inprop": "url",
            }
        )
        for page in payload.get("query", {}).get("pages", []):
            if page.get("missing") or "disambiguation" in page.get("pageprops", {}):
                continue
            text = _normalize_text(page.get("extract", ""))
            if len(text) < min_chars and len(text) >= 200:
                full_payload = _request(
                    {
                        "action": "query",
                        "pageids": int(page["pageid"]),
                        "prop": "extracts",
                        "explaintext": 1,
                        "exchars": 6000,
                    }
                )
                full_pages = full_payload.get("query", {}).get("pages", [])
                if full_pages:
                    text = _normalize_text(full_pages[0].get("extract", ""))
                time.sleep(0.35)
            text = text[:6000]
            if not _usable(text, min_chars):
                continue
            text_hash = hashlib.sha256(text.encode("utf-8")).hexdigest()
            if text_hash in seen_text:
                continue
            seen_text.add(text_hash)
            page_id = int(page["pageid"])
            creation = event_by_page[page_id]
            revision = (page.get("revisions") or [{}])[0]
            records.append(
                {
                    "page_id": page_id,
                    "title": page["title"],
                    "creation_timestamp": creation["timestamp"],
                    "creation_revision": int(creation.get("revid", 0)),
                    "snapshot_revision": int(revision.get("revid", 0)),
                    "snapshot_timestamp": revision.get("timestamp", ""),
                    "canonical_url": page.get("canonicalurl", ""),
                    "text_sha256": text_hash,
                    "text": text,
                }
            )
            if len(records) >= wanted:
                break
        if (start // 20 + 1) % 25 == 0:
            print(
                f"scanned={min(start + 20, len(page_ids))} usable={len(records)}",
                flush=True,
            )
        time.sleep(0.5)
    return records


def main() -> None:
    args = parse_args()
    if args.records < 1 or args.candidate_limit < args.records:
        raise ValueError("candidate-limit must be at least records")
    events = _creation_events(args.start, args.end, args.candidate_limit)
    records = _fetch_pages(events, args.min_chars, args.records)
    if len(records) < args.records:
        raise RuntimeError(
            f"Only {len(records)} usable pages from {len(events)} creation events"
        )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
    payload = args.output.read_bytes()
    manifest = {
        "dataset": "English Wikipedia newly-created main-namespace pages",
        "source_api": API_URL,
        "creation_interval_inclusive": {"start": args.start, "end": args.end},
        "license": "CC BY-SA 4.0; individual page attribution URLs are in JSONL",
        "selection": {
            "creation_events_considered": len(events),
            "records": len(records),
            "minimum_clean_extract_characters": args.min_chars,
            "source_text": (
                "plaintext intro, with per-page full-extract fallback when intro is "
                "200-899 characters; truncated to 6000 characters"
            ),
            "namespace": 0,
            "disambiguation_pages_excluded": True,
            "exact_clean_text_deduplicated": True,
        },
        "jsonl_sha256": hashlib.sha256(payload).hexdigest(),
        "retrieved_unix_time": time.time(),
    }
    manifest_path = args.output.with_suffix(".manifest.json")
    manifest_path.write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(json.dumps({"output": str(args.output), **manifest}, indent=2))


if __name__ == "__main__":
    main()
