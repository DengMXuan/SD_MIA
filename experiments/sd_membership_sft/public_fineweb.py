from __future__ import annotations

import argparse
import hashlib
import json
import re
import time
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Any


ROWS_URL = "https://datasets-server.huggingface.co/rows"
DATASET = "HuggingFaceFW/fineweb"
DATASET_REVISION = "9bb295ddab0e05d785b879661af7260fed5140fc"
USER_AGENT = "SD-MIA-research/0.1 (controlled academic benchmark)"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Snapshot a bounded FineWeb dump sample.")
    parser.add_argument("--config", default="CC-MAIN-2025-26")
    parser.add_argument("--records", type=int, default=600)
    parser.add_argument("--min-chars", type=int, default=900)
    parser.add_argument("--min-token-count", type=int, default=256)
    parser.add_argument("--min-language-score", type=float, default=0.90)
    parser.add_argument("--scan-limit", type=int, default=5000)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("experiments/data/fineweb_cc-main-2025-26.jsonl"),
    )
    return parser.parse_args()


def _request(parameters: dict[str, Any], attempts: int = 6) -> dict[str, Any]:
    url = f"{ROWS_URL}?{urllib.parse.urlencode(parameters)}"
    request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    for attempt in range(attempts):
        try:
            with urllib.request.urlopen(request, timeout=120) as response:
                return json.load(response)
        except Exception:
            if attempt + 1 == attempts:
                raise
            time.sleep(min(60.0, 2.0 * 2**attempt))
    raise AssertionError("unreachable")


def _clean(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()[:6000]


def main() -> None:
    args = parse_args()
    if args.records < 1 or args.scan_limit < args.records:
        raise ValueError("scan-limit must be at least records")
    records: list[dict[str, Any]] = []
    seen_text: set[str] = set()
    dates: list[str] = []
    for offset in range(0, args.scan_limit, 100):
        payload = _request(
            {
                "dataset": DATASET,
                "config": args.config,
                "split": "train",
                "offset": offset,
                "length": 100,
            }
        )
        for item in payload.get("rows", []):
            row = item["row"]
            if row.get("dump") != args.config:
                continue
            if int(row.get("token_count", 0)) < args.min_token_count:
                continue
            if float(row.get("language_score", 0.0)) < args.min_language_score:
                continue
            text = _clean(row.get("text", ""))
            if len(text) < args.min_chars:
                continue
            digest = hashlib.sha256(text.encode("utf-8")).hexdigest()
            if digest in seen_text:
                continue
            seen_text.add(digest)
            date = str(row.get("date", ""))
            dates.append(date)
            identifier = str(row["id"])
            records.append(
                {
                    "page_id": identifier,
                    "title": urllib.parse.urlparse(row["url"]).netloc or "web document",
                    "creation_timestamp": date,
                    "creation_revision": int(item["row_idx"]),
                    "snapshot_revision": int(item["row_idx"]),
                    "snapshot_timestamp": date,
                    "canonical_url": row["url"],
                    "text_sha256": digest,
                    "text": text,
                    "fineweb_id": identifier,
                    "dump": row["dump"],
                    "file_path": row["file_path"],
                    "language_score": float(row["language_score"]),
                    "source_token_count": int(row["token_count"]),
                }
            )
            if len(records) >= args.records:
                break
        print(f"scanned={min(offset + 100, args.scan_limit)} usable={len(records)}", flush=True)
        if len(records) >= args.records:
            break
    if len(records) < args.records:
        raise RuntimeError(f"Only {len(records)} usable rows within scan limit")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
    payload = args.output.read_bytes()
    manifest = {
        "dataset": DATASET,
        "dataset_revision": DATASET_REVISION,
        "config": args.config,
        "split_unit": "FineWeb document",
        "source_api": ROWS_URL,
        "creation_interval_inclusive": {"start": min(dates), "end": max(dates)},
        "timestamp_semantics": "Common Crawl capture time, not original publication time",
        "license": "ODC-By 1.0 (FineWeb); source URLs retained for provenance",
        "selection": {
            "rows_scanned": min(offset + 100, args.scan_limit),
            "records": len(records),
            "minimum_clean_characters": args.min_chars,
            "minimum_source_token_count": args.min_token_count,
            "minimum_language_score": args.min_language_score,
            "text_truncated_to_characters": 6000,
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
