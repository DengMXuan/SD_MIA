"""Freeze recent Wikipedia event auxiliaries from WikiTection plus collected events."""
from __future__ import annotations

from collections import Counter
from datetime import datetime, timezone
import json
from pathlib import Path
import re
import tempfile
import unicodedata

import pyarrow.parquet as pq

from experiments.pretraining.data import TARGET, load_tokenizer, sha256
from experiments.pretraining.benchmarks.prepare import (
    AUX_COUNT, DATA_ROOT, _aux_candidates, wiki_parquet,
)

from experiments.paths import ROOT
WIKITECTION_POOL = ROOT / 'artifacts/data/pools/wikitection/pool.jsonl'
COLLECTED_CHECKPOINT = DATA_ROOT / 'wikimia_aux/events_post2023.checkpoint.jsonl'
REUSED_AUXILIARY = DATA_ROOT / 'wikimia_aux/from_wikitection_plus_events_2024plus.jsonl'
EVENT = re.compile(r'\b(?:election|attack|protest|war|disaster|incident|earthquake|festival|'
                   r'championship|tournament|massacre|bombing|flood|hurricane|cyclone|fire|'
                   r'crash|strike|referendum|coup|summit|awards|olympics|games)\b', re.I)
RECENT_YEAR = re.compile(r'\b20(?:24|25|26)\b')
OLD_YEAR = re.compile(r'\b(?:19\d{2}|20(?:0\d|1\d|2[0-3]))\b')
SCHEMA = 'wikimia_reused_event_auxiliary_v2'


def _title_key(title: str) -> str:
    return unicodedata.normalize('NFKC', title).casefold().strip()


def _eligible_title(row: dict, source: str) -> bool:
    title = row.get('title')
    text = row.get('text')
    if not isinstance(title, str) or not isinstance(text, str) or len(text.split()) < 128:
        return False
    try:
        created = datetime.fromisoformat(row['creation_timestamp'].replace('Z', '+00:00'))
    except (KeyError, TypeError, AttributeError, ValueError):
        return False
    if created.tzinfo is None or created.astimezone(timezone.utc) < datetime(2024, 1, 1, tzinfo=timezone.utc):
        return False
    if OLD_YEAR.search(title):
        return False
    if RECENT_YEAR.search(title):
        return True
    if not EVENT.search(title):
        return False
    return source == 'collected_events' or bool(RECENT_YEAR.search(' '.join(text.split()[:128])))


def _rows(path: Path, source: str):
    for line in path.read_text().splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        if source == 'collected_events':
            row = row.get('record')
            if row is None:
                continue
        if _eligible_title(row, source):
            yield {**row, 'auxiliary_source_pool': source}


def build(output: Path = REUSED_AUXILIARY, *, wikitection: Path = WIKITECTION_POOL,
          checkpoint: Path = COLLECTED_CHECKPOINT) -> Path:
    output, wikitection, checkpoint = map(lambda p: Path(p).resolve(),
                                          (output, wikitection, checkpoint))
    if not wikitection.is_file() or not checkpoint.is_file():
        raise ValueError('WikiTection pool and collected-event checkpoint are both required')
    sources = {name: dict(path=str(path), sha256=sha256(path)) for name, path in
               (('wikitection', wikitection), ('collected_events', checkpoint))}
    metadata = output.with_suffix('.manifest.json')
    if output.exists() or metadata.exists():
        if not output.is_file() or not metadata.is_file():
            raise ValueError('incomplete auxiliary publication; use another output path')
        frozen = json.loads(metadata.read_text())
        if frozen.get('schema') != SCHEMA or frozen.get('source_files') != sources or frozen.get('sha256') != sha256(output):
            raise ValueError('published auxiliary source changed; use another output path')
        return output
    seen_titles: set[str] = set()
    accepted = []
    for name, source in (('wikitection', wikitection), ('collected_events', checkpoint)):
        for row in _rows(source, name):
            key = _title_key(row['title'])
            if key in seen_titles:
                continue
            seen_titles.add(key)
            accepted.append(row)
    if len(accepted) < AUX_COUNT:
        raise ValueError(f'only {len(accepted)} distinct recent event pages; need {AUX_COUNT}')
    output.parent.mkdir(parents=True, exist_ok=True)
    # Validate the exact downstream deduplication before publishing anything.
    with tempfile.TemporaryDirectory(prefix='wikimia-aux-check-', dir=output.parent) as directory:
        candidate = Path(directory) / 'candidate.jsonl'
        candidate.write_text(''.join(json.dumps(row, ensure_ascii=False) + '\n' for row in accepted))
        tokenizer = load_tokenizer(TARGET)
        eligible = {}
        for length in (64, 128):
            test_texts = pq.read_table(wiki_parquet(length), columns=['input'])['input'].to_pylist()
            eligible[str(length)] = len(_aux_candidates(candidate, length, test_texts, tokenizer))
        if any(count < AUX_COUNT for count in eligible.values()):
            raise ValueError(f'not enough independent auxiliaries after WikiMIA overlap filtering: {eligible}')
        candidate.replace(output)
    publication = dict(schema=SCHEMA, source_files=sources, sha256=sha256(output),
        rows=len(accepted), rows_by_source=dict(Counter(row['auxiliary_source_pool'] for row in accepted)),
        eligible_after_filter=eligible,
        title_filter='recent 2024–2026 event or year title; no older year in title',
        deduplication='NFKC casefold title, then 13-word overlap and exact token filtering at preparation',
        extraction='WikiTection rendered MediaWiki article text plus MediaWiki plain-text event extracts',
        caveat='two related but not identical Wikipedia extraction pipelines')
    metadata.write_text(json.dumps(publication, indent=2) + '\n')
    return output
