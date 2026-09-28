"""Freeze official MIMIR and WikiMIA records without changing existing audits."""
from __future__ import annotations

from collections import defaultdict
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re

import numpy as np
import pyarrow.parquet as pq

from experiments.pretraining.data import (
    DRAFT, TARGET, TOKEN_CONTRACT, load_tokenizer, sha256, tokenizer_hash,
)
from experiments.pretraining.prepare import download_mimir, _read_groups
from experiments.shared.data.data import _hash_ids


from experiments.paths import ROOT
DATA_ROOT = ROOT / 'artifacts/data/paper_positive_controls_v1'
OLD_MIMIR_ROOT = Path('/home/mxd/lib/SD_MIA-pretraining-data/mimir')
MIMIR_REVISION = '02500d3b7cece0cb7628e939ba9fc93fdb6362ae'
WIKIMIA_REVISION = 'a89ab76d88f704e9bc5870ac39cc9d458a2a70ac'
SEEDS = (1919, 1949, 1978)
MIMIR_SOURCES = ('github', 'arxiv')
WIKI_LENGTHS = (64, 128)
AUX_COUNT = 600
MAX_TOKENS = 512


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _safe_output(output: Path, inputs: list[Path]) -> None:
    output = output.resolve()
    protected = [ROOT / 'experiments', ROOT / 'standalone', ROOT / 'tests', ROOT / '.git',
                 ROOT / 'artifacts/audits/pythia_mimir_v1', *inputs]
    for item in protected:
        item = item.resolve()
        require(output != item and not output.is_relative_to(item)
                and not item.is_relative_to(output), f'output overlaps protected input: {item}')


def _existing(path: Path, *, benchmark: str, seed: int, files: dict[str, Path]) -> Path:
    manifest_path = path / 'manifest.json'
    require(manifest_path.exists(), f'incomplete frozen output: {path}')
    manifest = json.loads(manifest_path.read_text())
    require(manifest['benchmark'] == benchmark and manifest['selection_seed'] == seed
            and manifest['models'] == dict(target=TARGET, draft=DRAFT)
            and manifest['token_contract'] == TOKEN_CONTRACT
            and manifest['counts']['auxiliary'] == AUX_COUNT
            and sha256(path / manifest['records_file']) == manifest['records_sha256'],
            f'frozen experiment changed: {path}')
    require(set(manifest['source_files']) == set(files), f'frozen source list changed: {path}')
    for name, source in files.items():
        require(manifest['source_files'][name]['path'] == str(source.resolve())
                and manifest['source_files'][name]['sha256'] == sha256(source),
                f'frozen source changed: {source}')
    return manifest_path


def prepare_mimir(source: str, seed: int, data_root: Path = DATA_ROOT,
                  *, official_root: Path | None = None, allow_download: bool = False) -> Path:
    require(source in MIMIR_SOURCES and seed in SEEDS, 'unsupported MIMIR source or seed')
    data_root = Path(data_root).resolve()
    official_root = Path(official_root or data_root / 'official').resolve()
    files = {
        role: official_root / 'cache_100_200_1000_512' / role /
              f'{source}_ngram_7_0.2.jsonl' for role in ('train', 'test')
    }
    auxiliary_file = (OLD_MIMIR_ROOT / 'official/cache_100_200_1000_512/test' /
                      f'{source}_ngram_13_0.8.jsonl').resolve()
    old_manifest_path = OLD_MIMIR_ROOT / 'prepared' / source / 'seed1919/manifest.json'
    require(auxiliary_file.is_file() and old_manifest_path.is_file(),
            f'pinned 13_gram_0.8 nonmember auxiliary source missing: {source}')
    old_manifest = json.loads(old_manifest_path.read_text())
    require(old_manifest['source_provenance']['revision'] == MIMIR_REVISION
            and old_manifest['source_files']['nonmember']['path'] == str(auxiliary_file)
            and old_manifest['source_files']['nonmember']['sha256'] == sha256(auxiliary_file),
            f'old auxiliary source is not the pinned MIMIR revision: {source}')
    output = data_root / 'mimir_7_0.2' / source / f'seed{seed}'
    _safe_output(output, [*files.values(), auxiliary_file])
    if not all(path.exists() for path in files.values()):
        require(allow_download, 'official 7_gram_0.2 files missing; rerun prepare with --allow-download')
        downloaded = download_mimir(source=source, split='ngram_7_0.2', cache_size=1000,
                                    local_dir=official_root, local_files_only=False)
        require(Path(downloaded['member_file']).resolve() == files['train']
                and Path(downloaded['nonmember_file']).resolve() == files['test'],
                'official download paths differ from pinned layout')
    benchmark = f'mimir/{source}/ngram_7_0.2'
    if output.exists():
        return _existing(output, benchmark=benchmark, seed=seed,
                         files={'member': files['train'], 'nonmember': files['test'],
                                'auxiliary_nonmember': auxiliary_file})
    return _freeze_mimir_with_external_aux(files['train'], files['test'], auxiliary_file,
                                           output, source=source, seed=seed)


def _freeze_mimir_with_external_aux(member_file: Path, nonmember_file: Path,
                                    auxiliary_file: Path, output: Path, *, source: str, seed: int) -> Path:
    """Keep the scarce 7-gram negatives for testing; train on disjoint official negatives."""
    tokenizer = load_tokenizer(TARGET)
    require(tokenizer_hash(tokenizer) == tokenizer_hash(load_tokenizer(DRAFT)),
            'Pythia target and draft tokenizer mismatch')
    seven, seven_stats = _read_groups(member_file, nonmember_file, tokenizer,
                                     source=source, max_tokens=MAX_TOKENS)
    auxiliary_groups, auxiliary_stats = _read_groups(member_file, auxiliary_file, tokenizer,
                                                     source=source, max_tokens=MAX_TOKENS)
    require({row['token_hash'] for row in seven[1]} ==
            {row['token_hash'] for row in auxiliary_groups[1]},
            'member parsing changed between official sources')
    excluded_hashes = {row['token_hash'] for rows in seven.values() for row in rows}
    inverted, sizes = _test_index([row['text'] for row in seven[0]])
    auxiliary_candidates = [row for row in auxiliary_groups[0]
                            if row['token_hash'] not in excluded_hashes
                            and not _near_test(row['text'], inverted, sizes)]
    test_size = min(400, len(seven[0]))
    require(test_size >= 2 and len(seven[1]) >= test_size and len(auxiliary_candidates) >= AUX_COUNT,
            f'insufficient disjoint official samples: test={test_size}, aux={len(auxiliary_candidates)}')
    selected = {
        'member': [seven[1][int(i)] for i in np.random.default_rng(seed + 1).permutation(len(seven[1]))[:test_size]],
        'nonmember': [seven[0][int(i)] for i in np.random.default_rng(seed).permutation(len(seven[0]))[:test_size]],
        'auxiliary': [auxiliary_candidates[int(i)] for i in
                      np.random.default_rng(seed + 2).permutation(len(auxiliary_candidates))[:AUX_COUNT]],
    }
    rows = [{**row, 'group': group, 'official_split':
             'ngram_13_0.8' if group == 'auxiliary' else 'ngram_7_0.2'}
            for group, values in selected.items() for row in values]
    require(len({row['token_hash'] for row in rows}) == len(rows), 'cross-role token duplicate')
    output.mkdir(parents=True)
    record_file = output / 'records.jsonl'
    record_file.write_text(''.join(json.dumps(row, ensure_ascii=False) + '\n' for row in rows))
    manifest = dict(kind='mimir_pretraining_v1', benchmark=f'mimir/{source}/ngram_7_0.2',
        models=dict(target=TARGET, draft=DRAFT), token_contract=TOKEN_CONTRACT,
        tokenizer_sha256=tokenizer_hash(tokenizer),
        source_provenance=dict(repo_id='iamgroot42/mimir', revision=MIMIR_REVISION,
            test_split='ngram_7_0.2', auxiliary_split='ngram_13_0.8', source=source),
        source_files={name: dict(path=str(path.resolve()), sha256=sha256(path)) for name, path in
                      (('member', member_file), ('nonmember', nonmember_file),
                       ('auxiliary_nonmember', auxiliary_file))},
        label_provenance='MIMIR official train=member; official test=nonmember in both splits',
        draft_exposure='Pythia 1.4B also pretrained on The Pile; not a member-blind draft',
        counts=dict(member=test_size, nonmember=test_size, auxiliary=AUX_COUNT),
        max_tokens=MAX_TOKENS, selection_seed=seed,
        filtering=dict(test_split=seven_stats, auxiliary_source=auxiliary_stats,
                       auxiliary_eligible_after_disjoint_filter=len(auxiliary_candidates)),
        selection_rng='numpy.default_rng(seed + label) for test; seed + 2 for auxiliary',
        token_lengths={group: dict(min=min(lengths), max=max(lengths), mean=float(np.mean(lengths)))
                       for group, values in selected.items()
                       if (lengths := [len(row['token_ids']) for row in values])},
        caveats=['7_gram_0.2 official nonmember cache has fewer than 1000 rows',
                 'test uses only 7_gram_0.2 records, balanced up to 400 per class',
                 '600 disjoint auxiliaries are official same-domain 13_gram_0.8 nonmembers',
                 'auxiliary overlap with all 7_gram_0.2 nonmembers was excluded',
                 'auxiliary distribution differs from the low-overlap test distribution'],
        records_file=record_file.name, records_sha256=sha256(record_file))
    manifest_file = output / 'manifest.json'
    manifest_file.write_text(json.dumps(manifest, indent=2) + '\n')
    return manifest_file


def wiki_parquet(length: int, snapshot: Path | None = None) -> Path:
    require(length in WIKI_LENGTHS, 'WikiMIA supports length 64 and 128 here')
    if snapshot is None:
        root = Path('/home/mxd/.cache/huggingface/hub/datasets--swj0419--WikiMIA')
        snapshot = root / 'snapshots' / WIKIMIA_REVISION
    matches = list((Path(snapshot) / 'data').glob(f'WikiMIA_length{length}-*.parquet'))
    require(len(matches) == 1, f'one pinned WikiMIA length{length} parquet is required: {snapshot}')
    return matches[0].resolve()


def _word_grams(text: str) -> set[tuple[str, ...]]:
    words = re.findall(r'\w+|[^\w\s]', text.casefold())
    return {tuple(words[i:i + 13]) for i in range(max(0, len(words) - 12))}


def _test_index(texts: list[str]) -> tuple[dict[tuple[str, ...], set[int]], list[int]]:
    inverted: dict[tuple[str, ...], set[int]] = defaultdict(set)
    sizes = []
    for index, text in enumerate(texts):
        grams = _word_grams(text)
        sizes.append(len(grams))
        for gram in grams:
            inverted[gram].add(index)
    return inverted, sizes


def _near_test(text: str, inverted: dict, sizes: list[int]) -> bool:
    grams = _word_grams(text)
    counts: dict[int, int] = defaultdict(int)
    for gram in grams:
        for index in inverted.get(gram, ()):
            counts[index] += 1
    return any(count / min(len(grams), sizes[index]) >= 0.5
               for index, count in counts.items() if min(len(grams), sizes[index]))


def _aux_candidates(aux_file: Path, length: int, test_texts: list[str], tokenizer) -> list[dict]:
    inverted, sizes = _test_index(test_texts)
    auxiliary_index: dict[tuple[str, ...], set[int]] = defaultdict(set)
    auxiliary_sizes: list[int] = []
    seen_source, seen_tokens = set(), set()
    candidates = []
    for line_number, line in enumerate(aux_file.read_text().splitlines(), 1):
        if not line.strip():
            continue
        row = json.loads(line)
        text = row.get('text')
        identity = row.get('pageid', row.get('page_id', row.get('record_id')))
        require(isinstance(text, str) and isinstance(identity, (str, int)),
                f'{aux_file}:{line_number}: expected text and pageid/record_id')
        # Category index pages summarize a whole year rather than one event.
        if isinstance(row.get('title'), str) and re.fullmatch(
                r'20\d{2} in (politics|sports|science|music)', row['title'], re.I):
            continue
        try:
            created = datetime.fromisoformat(row['creation_timestamp'].replace('Z', '+00:00'))
            require(created.tzinfo is not None and
                    created.astimezone(timezone.utc) >= datetime(2024, 1, 1, tzinfo=timezone.utc),
                    f'{aux_file}:{line_number}: auxiliary page predates 2024')
        except (KeyError, TypeError, AttributeError, ValueError) as error:
            raise ValueError(f'{aux_file}:{line_number}: invalid creation timestamp') from error
        words = text.split()
        if len(words) < length:
            continue
        visible = ' '.join(words[:length])
        ids = list(tokenizer(visible, add_special_tokens=False, truncation=True,
                             max_length=MAX_TOKENS).input_ids)
        token_hash = _hash_ids(ids)
        if (len(ids) < 2 or str(identity) in seen_source or token_hash in seen_tokens
                or _near_test(visible, inverted, sizes)
                or _near_test(visible, auxiliary_index, auxiliary_sizes)):
            continue
        seen_source.add(str(identity))
        seen_tokens.add(token_hash)
        candidates.append(dict(source_id=str(identity), text=visible, token_ids=ids,
                               token_hash=token_hash,
                               creation_timestamp=row['creation_timestamp']))
        grams = _word_grams(visible)
        auxiliary_sizes.append(len(grams))
        for gram in grams:
            auxiliary_index[gram].add(len(auxiliary_sizes) - 1)
    require(len(candidates) >= AUX_COUNT,
            f'only {len(candidates)} eligible disjoint WikiMIA auxiliary pages; need {AUX_COUNT}')
    return candidates


def prepare_wikimia(length: int, seed: int, aux_file: Path, data_root: Path = DATA_ROOT,
                    *, snapshot: Path | None = None) -> Path:
    require(length in WIKI_LENGTHS and seed in SEEDS, 'unsupported WikiMIA length or seed')
    official = wiki_parquet(length, snapshot)
    aux_file = Path(aux_file).resolve()
    require(aux_file.is_file(), f'missing independently collected auxiliary JSONL: {aux_file}')
    files = {'official': official, 'auxiliary': aux_file}
    auxiliary_provenance = aux_file.with_suffix('.manifest.json')
    if auxiliary_provenance.is_file():
        files['auxiliary_provenance'] = auxiliary_provenance
    output = Path(data_root).resolve() / f'wikimia_2024plus_{length}' / f'seed{seed}'
    _safe_output(output, list(files.values()))
    benchmark = f'wikimia/official_length{length}/temporal_proxy'
    if output.exists():
        return _existing(output, benchmark=benchmark, seed=seed,
                         files=files)
    table = pq.read_table(official, columns=['input', 'label'])
    texts, labels = table['input'].to_pylist(), table['label'].to_pylist()
    require(set(labels) == {0, 1} and all(isinstance(t, str) for t in texts),
            'official WikiMIA has unexpected fields or labels')
    tokenizer = load_tokenizer(TARGET)
    require(tokenizer_hash(tokenizer) == tokenizer_hash(load_tokenizer(DRAFT)),
            'Pythia target and draft tokenizer mismatch')
    rows, seen_tokens = [], set()
    for index, (text, label) in enumerate(zip(texts, labels)):
        ids = list(tokenizer(text, add_special_tokens=False, truncation=True,
                             max_length=MAX_TOKENS).input_ids)
        require(len(ids) >= 2, f'official WikiMIA row {index} too short')
        token_hash = _hash_ids(ids)
        require(token_hash not in seen_tokens, 'duplicate official WikiMIA token sequence')
        seen_tokens.add(token_hash)
        rows.append(dict(record_id=f'wikimia:{length}:official:{index}',
                         source='wikimia_official', source_row=index,
                         group='member' if label == 1 else 'nonmember', label=label,
                         text=text, token_ids=ids, token_hash=token_hash,
                         text_sha256=hashlib.sha256(text.encode()).hexdigest()))
    candidates = _aux_candidates(aux_file, length, texts, tokenizer)
    selected = []
    for index in np.random.default_rng(seed).permutation(len(candidates)):
        candidate = candidates[int(index)]
        if candidate['token_hash'] in seen_tokens:
            continue
        seen_tokens.add(candidate['token_hash'])
        selected.append(candidate)
        if len(selected) == AUX_COUNT:
            break
    require(len(selected) == AUX_COUNT, 'not enough auxiliary pages after token deduplication')
    for candidate in selected:
        rows.append(dict(record_id=f'wikimia:{length}:aux:{candidate["source_id"]}',
                         source='wikipedia_event_auxiliary', group='auxiliary', label=0,
                         text=candidate['text'], token_ids=candidate['token_ids'],
                         token_hash=candidate['token_hash'],
                         creation_timestamp=candidate['creation_timestamp']))
    output.mkdir(parents=True)
    record_file = output / 'records.jsonl'
    record_file.write_text(''.join(json.dumps(row, ensure_ascii=False) + '\n' for row in rows))
    counts = {name: sum(row['group'] == name for row in rows)
              for name in ('member', 'nonmember', 'auxiliary')}
    manifest = dict(kind='temporal_pretraining_v1', benchmark=benchmark,
        models=dict(target=TARGET, draft=DRAFT), token_contract=TOKEN_CONTRACT,
        tokenizer_sha256=tokenizer_hash(tokenizer),
        source_provenance=dict(official_dataset='swj0419/WikiMIA', revision=WIKIMIA_REVISION,
                               length_words=length, auxiliary='independent post-2023 Wikipedia event pages',
                               auxiliary_metadata=(json.loads(auxiliary_provenance.read_text())
                                                   if auxiliary_provenance.is_file() else None)),
        source_files={name: dict(path=str(path), sha256=sha256(path)) for name, path in files.items()},
        label_provenance='Official WikiMIA old/new Wikipedia event time proxy; not verified Pile membership',
        draft_exposure='Pythia 1.4B pretrained on The Pile; WikiMIA positives are not verified members',
        counts=counts, max_tokens=MAX_TOKENS, selection_seed=seed,
        caveats=['Official WikiMIA test rows preserved, including class imbalance',
                 'positive labels are temporal proxies, not verified pretraining membership',
                 'auxiliary pages include reused WikiTection and collected event pages',
                 'auxiliary extraction pipelines differ; 13-word near-overlap filtered'],
        records_file=record_file.name, records_sha256=sha256(record_file))
    (output / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
    return output / 'manifest.json'
