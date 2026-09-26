"""Prepare externally labelled raw-text audits; never train a language model.

MIMIR retains official labels. Temporal data uses historical text as presumed
members and post-release text as nonmembers, with disjoint detector auxiliaries.
"""
from __future__ import annotations

import json
import re
import shutil
import tempfile
from datetime import datetime
from pathlib import Path

import numpy as np

from experiments.pretraining.data import TOKEN_CONTRACT, load_tokenizer, tokenizer_hash, sha256
from experiments.pretraining.prepare import freeze, download_mimir, inspect_mimir
from experiments.shared.data.data import _hash_ids
from experiments.shared.data.pools import NearDuplicateIndex
from experiments.shared.models.registry import MODEL_PAIRS

_qwen = MODEL_PAIRS['qwen3']
QWEN_MODELS = {
    'target': dict(repo_id=_qwen.target, revision=_qwen.target_revision),
    'draft': dict(repo_id=_qwen.draft, revision=_qwen.draft_revision),
}
QWEN_RELEASE = '2025-04-29'


def prepare_mimir(member_file, nonmember_file, output_dir, *, source, split, seed,
                  n_per_class, n_aux=600, max_tokens=512, models=None,
                  source_provenance=None):
    """Freeze official cache files for the current method, using one public seed.

    n_aux must cover detector fitting, validation and independent calibration.
    Nothing is borrowed from test nonmembers when the supplied cache is too small.
    """
    if type(seed) is not int or not 0 <= seed < 2**32:
        raise ValueError('seed must be an integer in [0, 2**32)')
    if any(type(n) is not int or n < 1 for n in (n_per_class, n_aux)):
        raise ValueError('test and auxiliary counts must be positive integers')
    output_dir = Path(output_dir).resolve()
    if output_dir.exists():
        raise FileExistsError(output_dir)
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix='.mimir-', dir=output_dir.parent))
    try:
        freeze(member_file, nonmember_file, temporary, source=source, split=split,
               seed=seed, n_per_class=n_per_class, n_aux=n_aux, max_tokens=max_tokens,
               models=models, source_provenance=source_provenance)
        temporary.rename(output_dir)
    finally:
        if temporary.exists():
            shutil.rmtree(temporary)
    return output_dir / 'manifest.json'


def wikitext_articles(parquet_files):
    """Reassemble articles across ordered WikiText-103 raw parquet shards.

    Never treat lines/paragraphs from one article as separate audit documents.
    Retain at most the first 12,000 characters; downstream tokenization records
    its own explicit truncation. Import pyarrow only for this optional reader.
    """
    import pyarrow.parquet as pq
    title, pieces, size, index = None, [], 0, 0
    for path in map(Path, parquet_files):
        for batch in pq.ParquetFile(path).iter_batches(columns=['text'], batch_size=4096):
            for text in batch.column(0).to_pylist():
                match = re.fullmatch(r'\s*= ([^=\n]+) =\s*', text)
                if match:
                    if title is not None and pieces:
                        yield dict(record_id=f'wikitext103:train:article{index}',
                                   source='wikitext-103-raw-v1/train', title=title,
                                   text='\n'.join(pieces))
                        index += 1
                    title, pieces, size = match[1].strip(), [], 0
                elif title is not None and text.strip() and size < 12000:
                    # Section headings are metadata rather than continuation text.
                    if re.fullmatch(r'(?:=\s*){2,}.*', text.strip()):
                        continue
                    piece = text.strip()[:12000-size]
                    pieces.append(piece)
                    size += len(piece)
    if title is not None and pieces:
        yield dict(record_id=f'wikitext103:train:article{index}',
                   source='wikitext-103-raw-v1/train', title=title, text='\n'.join(pieces))


def _pool(path):
    path = Path(path).resolve()
    manifest_path = path.with_suffix('.manifest.json')
    manifest = json.loads(manifest_path.read_text())
    if sha256(path) != manifest['jsonl_sha256']:
        raise ValueError(f'pool hash mismatch: {path}')
    return [json.loads(line) for line in path.read_text().splitlines() if line], manifest


def prepare_temporal(historical_files, recent_pool, output_dir, *, seed,
                     historical_format='wikitext103', historical_provenance=None,
                     models=None, model_release=QWEN_RELEASE,
                     n_per_class=2000, n_aux=600, min_tokens=128, max_tokens=512):
    """Freeze historical/post-release text without touching an SFT run or split.

    Historical formats: complete WikiText raw train shards, or historical Wiki
    pools with pinned revision timestamps. WikiText publication provenance must
    be provided explicitly. All labels are temporal proxies, never verified MIA
    ground truth. The recent pool supplies both test negatives and auxiliaries.
    """
    if (type(seed) is not int or not 0 <= seed < 2**32
            or any(type(n) is not int or n < 1 for n in (n_per_class, n_aux, min_tokens, max_tokens))
            or not 2 <= min_tokens <= max_tokens <= 2048):
        raise ValueError('invalid seed, record counts or token limits')
    output_dir = Path(output_dir).resolve()
    if output_dir.exists():
        raise FileExistsError(output_dir)
    # Sort logical shard filenames BEFORE resolving Hugging Face blob symlinks.
    # Blob hashes do not encode shard order, which matters at article boundaries.
    files = [p.resolve() for p in sorted(map(Path, historical_files))]
    if not files or len(set(files)) != len(files):
        raise ValueError('distinct historical source files required')
    release = datetime.fromisoformat(model_release[:10]).date()
    historical_provenance = dict(historical_provenance or {})
    sources = [dict(path=str(p), sha256=sha256(p)) for p in files]
    if historical_format == 'wikitext103':
        published = historical_provenance.get('published_before')
        if (not published or datetime.fromisoformat(published[:10]).date() >= release
                or not historical_provenance.get('reference')):
            raise ValueError('WikiText requires documented publication before the model release')
        old = list(wikitext_articles(files))
    elif historical_format == 'historical_wiki':
        old = []
        for path in files:
            rows, meta = _pool(path)
            sources.append(dict(path=str(path.with_suffix('.manifest.json')), sha256=sha256(path.with_suffix('.manifest.json'))))
            if meta.get('label_semantics') != 'presumed_member_temporal_proxy':
                raise ValueError('expected a historical-revision Wikipedia pool')
            for row in rows:
                if (int(row.get('snapshot_revision', 0)) <= 0
                        or datetime.fromisoformat(row['creation_timestamp'][:10]).date() >= release
                        or datetime.fromisoformat(row['snapshot_timestamp'][:10]).date() >= release):
                    raise ValueError('historical creation/revision must predate the model release')
            old.extend(rows)
    else:
        raise ValueError('unsupported historical format')
    recent_pool = Path(recent_pool).resolve()
    recent, _ = _pool(recent_pool)
    for row in recent:
        if datetime.fromisoformat(row['creation_timestamp'][:10]).date() <= release:
            raise ValueError('recent Wikipedia creation must postdate the model release')
    sources += [dict(path=str(p), sha256=sha256(p))
                for p in (recent_pool, recent_pool.with_suffix('.manifest.json'))]
    models = models or QWEN_MODELS
    tokenizer = load_tokenizer(models['target'])
    if tokenizer_hash(tokenizer) != tokenizer_hash(load_tokenizer(models['draft'])):
        raise ValueError('target and draft tokenizers differ')
    rng = np.random.default_rng(seed)
    seen, near, selected, filtering = set(), NearDuplicateIndex(), {}, {}
    for group, candidates, count in [('member', old, n_per_class),
                                      ('recent', recent, n_per_class+n_aux)]:
        chosen, dropped_short, dropped_duplicate = [], 0, 0
        for i in rng.permutation(len(candidates)):
            row = candidates[int(i)]
            text = row['text']
            ids = list(tokenizer(text, add_special_tokens=False, truncation=True, max_length=max_tokens).input_ids)
            if len(ids) < min_tokens:
                dropped_short += 1
                continue
            token_hash = _hash_ids(ids)
            # Dedup on precisely the text visible to the models, across ALL roles.
            visible = tokenizer.decode(ids, skip_special_tokens=False)
            if token_hash in seen or near.is_duplicate(visible):
                dropped_duplicate += 1
                continue
            seen.add(token_hash)
            near.add(visible)
            chosen.append(dict(record_id=f'temporal:{row["record_id"]}', source=row['source'],
                               source_record_id=row['record_id'], label=int(group == 'member'),
                               text=text, token_ids=ids, token_hash=token_hash,
                               temporal_metadata={k: row[k] for k in ('title', 'creation_timestamp',
                                   'snapshot_timestamp', 'snapshot_revision') if k in row}))
            if len(chosen) == count:
                break
        if len(chosen) != count:
            raise ValueError(f'insufficient {group} documents after filtering: need {count}, got {len(chosen)}')
        selected[group] = chosen
        filtering[group] = dict(short=dropped_short, duplicate=dropped_duplicate,
                                candidates=len(candidates), selected=len(chosen))
    rows = [{**r, 'group': g} for g, records in (
        ('member', selected['member']), ('nonmember', selected['recent'][:n_per_class]),
        ('auxiliary', selected['recent'][n_per_class:])) for r in records]
    if len({r['record_id'] for r in rows}) != len(rows):
        raise ValueError('source document IDs overlap')
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix='.temporal-', dir=output_dir.parent))
    try:
        records_path = temporary / 'records.jsonl'
        records_path.write_text(''.join(json.dumps(r, ensure_ascii=False)+'\n' for r in rows))
        manifest = dict(kind='temporal_pretraining_v1', benchmark=f'wiki_temporal/{historical_format}',
            models=models, token_contract=TOKEN_CONTRACT, tokenizer_sha256=tokenizer_hash(tokenizer),
            counts=dict(member=n_per_class, nonmember=n_per_class, auxiliary=n_aux),
            selection_seed=seed, min_tokens=min_tokens, max_tokens=max_tokens,
            records_file='records.jsonl', records_sha256=sha256(records_path),
            label_provenance='historical presumed-member / post-release nonmember temporal proxies',
            membership_verified=False, model_release=model_release,
            source_provenance=dict(historical_format=historical_format, historical=historical_provenance, files=sources),
            draft_exposure='public pretrained Qwen draft; pretraining membership unknown',
            caveats=['dates do not prove membership or text novelty',
                     'historical and recent corpus curation may differ',
                     'WikiText raw retains legacy punctuation/spacing artifacts' if historical_format == 'wikitext103'
                     else 'historical article revisions must be interpreted as temporal proxies'],
            filtering=filtering,
            token_lengths={g: dict(min=min(len(r['token_ids']) for r in rows if r['group']==g),
                                   mean=float(np.mean([len(r['token_ids']) for r in rows if r['group']==g])),
                                   max=max(len(r['token_ids']) for r in rows if r['group']==g))
                           for g in ('member', 'nonmember', 'auxiliary')})
        (temporary / 'manifest.json').write_text(json.dumps(manifest, indent=2))
        temporary.rename(output_dir)
    finally:
        if temporary.exists():
            shutil.rmtree(temporary)
    return output_dir / 'manifest.json'
