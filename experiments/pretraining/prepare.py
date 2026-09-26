"""Download/import official MIMIR caches and freeze a labelled pretraining audit."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from huggingface_hub import hf_hub_download
from huggingface_hub.errors import GatedRepoError

from experiments.pretraining.data import TARGET, DRAFT, MIMIR_REVISION, TOKEN_CONTRACT, load_tokenizer, tokenizer_hash, sha256
from experiments.shared.data.data import _hash_ids

SOURCES = ('arxiv', 'dm_mathematics', 'github', 'hackernews', 'pile_cc', 'pubmed_central', 'wikipedia_(en)', 'full_pile')


def download_mimir(*, source, split='ngram_13_0.8', cache_size=1000,
                   local_dir=None, local_files_only=False):
    """Get one official train/test pair, never neighbor caches or model weights.

    Return keyword arguments for prepare_mimir (add output_dir, seed and counts).
    Uses the Hub SDK's existing login; no token is stored in the manifest.
    """
    if source not in SOURCES or split not in ('ngram_7_0.2', 'ngram_13_0.2', 'ngram_13_0.8', 'none'):
        raise ValueError('unknown official MIMIR source/split')
    if ((source == 'full_pile' and (split != 'none' or cache_size != 10000))
            or (source != 'full_pile' and (split == 'none' or cache_size != 1000))):
        raise ValueError('use domain/1000/ngram caches or full_pile/10000/none')
    filename = source + ('' if split == 'none' else '_' + split) + '.jsonl'
    files = [f'cache_100_200_{cache_size}_512/{part}/{filename}' for part in ('train', 'test')]
    try:
        member, nonmember = [Path(hf_hub_download('iamgroot42/mimir', name,
            repo_type='dataset', revision=MIMIR_REVISION, local_dir=local_dir,
            local_files_only=local_files_only)) for name in files]
    except GatedRepoError as error:
        raise RuntimeError('MIMIR access is gated: accept access at '
            'https://huggingface.co/datasets/iamgroot42/mimir using the account '
            'logged into the local Hugging Face SDK, and allow gated datasets in its token permissions.') from error
    return dict(member_file=member, nonmember_file=nonmember, source=source, split=split,
                source_provenance=dict(repo_id='iamgroot42/mimir', revision=MIMIR_REVISION, files=files))


def _matching_tokenizer(models):
    tokenizer = load_tokenizer(models['target'])
    draft_tokenizer = load_tokenizer(models['draft'])
    if tokenizer_hash(tokenizer) != tokenizer_hash(draft_tokenizer):
        raise ValueError('target and draft tokenizers differ')
    return tokenizer


def _read_groups(member_file, nonmember_file, tokenizer, *, source, max_tokens):
    """One parser/filter contract for preflight and freezing; no random sampling."""
    groups, owners, stats = {}, {}, {}
    for label, path in ((1, Path(member_file)), (0, Path(nonmember_file))):
        values, duplicates, short = [], 0, 0
        for index, line in enumerate(path.read_text(encoding='utf-8').splitlines()):
            if not line.strip():
                continue
            datum = json.loads(line)
            text = datum if isinstance(datum, str) else datum.get('text') if isinstance(datum, dict) else None
            if not isinstance(text, str):
                raise ValueError(f'{path}:{index + 1}: expected JSON string or object with text')
            ids = tokenizer(text, add_special_tokens=False, truncation=True, max_length=max_tokens).input_ids
            if len(ids) < 2:
                short += 1
                continue
            token_hash = _hash_ids(list(ids))
            if token_hash in owners:
                if owners[token_hash] != label:
                    raise ValueError('cross-label token duplicate in MIMIR caches; choose a deduplicated source/split')
                duplicates += 1
                continue
            owners[token_hash] = label
            values.append(dict(record_id=f'mimir:{source}:{label}:{index}:{token_hash[:16]}',
                source=source, source_row=index, label=label, text=text, token_ids=list(ids),
                token_hash=token_hash, text_sha256=hashlib.sha256(text.encode()).hexdigest()))
        groups[label] = values
        lengths = [len(row['token_ids']) for row in values]
        stats[str(label)] = dict(usable=len(values), duplicates_removed=duplicates, short_removed=short,
            token_lengths=dict(min=min(lengths), max=max(lengths), mean=float(np.mean(lengths))) if lengths else None)
    return groups, stats


def inspect_mimir(member_file, nonmember_file, *, source, n_aux=600, max_tokens=512, models=None):
    """Tokenizer-only readiness check, with the exact freeze filters and capacity.

    Cross-label token duplicates fail rather than silently changing official labels.
    Within-label exact duplicates and texts shorter than two tokens are removed.
    """
    if type(n_aux) is not int or n_aux < 0 or type(max_tokens) is not int or not 2 <= max_tokens <= 2048:
        raise ValueError('invalid auxiliary count or max_tokens')
    models = models or {'target': TARGET, 'draft': DRAFT}
    tokenizer = _matching_tokenizer(models)
    groups, stats = _read_groups(member_file, nonmember_file, tokenizer, source=source, max_tokens=max_tokens)
    capacity = max(0, min(len(groups[1]), len(groups[0]) - n_aux))
    return dict(filtering=stats, max_balanced_test_per_class=capacity, auxiliary=n_aux,
                max_tokens=max_tokens, tokenizer_sha256=tokenizer_hash(tokenizer),
                source_sha256=dict(member=sha256(Path(member_file)), nonmember=sha256(Path(nonmember_file))))


def freeze(member_file, nonmember_file, output_dir, *, source, split, n_per_class=800,
           n_aux=100, max_tokens=512, seed=20260824, models=None, source_provenance=None):
    if (type(seed) is not int or not 0 <= seed < 2**32
            or type(n_per_class) is not int or n_per_class < 1
            or type(n_aux) is not int or n_aux < 0
            or type(max_tokens) is not int or not 2 <= max_tokens <= 2048):
        raise ValueError('invalid seed, counts or max_tokens (must be 2..2048)')
    output_dir = Path(output_dir)
    if (output_dir / 'manifest.json').exists() or (output_dir / 'records.jsonl').exists():
        raise FileExistsError(f'frozen evaluation already exists: {output_dir}')
    models = models or {'target': TARGET, 'draft': DRAFT}
    tokenizer = _matching_tokenizer(models)
    groups, stats = _read_groups(member_file, nonmember_file, tokenizer, source=source, max_tokens=max_tokens)
    for label, values in groups.items():
        order = np.random.default_rng(seed + label).permutation(len(values))
        groups[label] = [values[i] for i in order]
    if len(groups[1]) < n_per_class or len(groups[0]) < n_per_class + n_aux:
        raise ValueError(f'not enough disjoint records: need member={n_per_class}, nonmember={n_per_class + n_aux}; got {stats}')
    rows = []
    for name, values in (('member', groups[1][:n_per_class]), ('nonmember', groups[0][:n_per_class]),
                         ('auxiliary', groups[0][n_per_class:n_per_class+n_aux])):
        rows.extend({**row, 'group': name} for row in values)
    output_dir.mkdir(parents=True, exist_ok=True)
    record_path = output_dir / 'records.jsonl'
    record_path.write_text(''.join(json.dumps(row, ensure_ascii=False) + '\n' for row in rows), encoding='utf-8')
    manifest = dict(kind='mimir_pretraining_v1', benchmark=f'mimir/{source}/{split}', models=models,
        token_contract=TOKEN_CONTRACT, tokenizer_sha256=tokenizer_hash(tokenizer),
        source_provenance=source_provenance or {'kind': 'user_supplied_official_cache'},
        source_files={name: {'path': str(Path(path).resolve()), 'sha256': sha256(Path(path))}
                      for name, path in [('member', member_file), ('nonmember', nonmember_file)]},
        label_provenance='MIMIR official train=member, test=nonmember; no SFT assignment',
        draft_exposure='Pythia 1.4B also pretrained on The Pile; not a member-blind draft',
        counts=dict(member=n_per_class, nonmember=n_per_class, auxiliary=n_aux),
        max_tokens=max_tokens, selection_seed=seed, filtering=stats,
        selection_rng='numpy.default_rng(condition_seed + official_label); disjoint nonmember slices',
        token_lengths={name: dict(min=min(lengths), max=max(lengths), mean=float(np.mean(lengths)))
                       for name in ('member', 'nonmember', 'auxiliary')
                       if (lengths := [len(row['token_ids']) for row in rows if row['group'] == name])},
        caveats=['official cache labels retained; auxiliary nonmembers held out of official test',
                 'Pythia draft also trained on The Pile',
                 'token lengths vary; max_tokens is a truncation limit, not fixed length',
                 'exact token deduplication only; official ngram filtering is source/split dependent'],
        records_file='records.jsonl', records_sha256=sha256(record_path))
    (output_dir / 'manifest.json').write_text(json.dumps(manifest, indent=2), encoding='utf-8')
    return output_dir / 'manifest.json'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', choices=SOURCES, default='wikipedia_(en)')
    parser.add_argument('--split', choices=('ngram_7_0.2', 'ngram_13_0.2', 'ngram_13_0.8', 'none'), default='ngram_13_0.8')
    parser.add_argument('--cache-size', type=int, choices=(1000, 10000), default=1000)
    parser.add_argument('--member-file', type=Path)
    parser.add_argument('--nonmember-file', type=Path)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--n-per-class', type=int, default=800)
    parser.add_argument('--n-aux', type=int, default=100)
    parser.add_argument('--max-tokens', type=int, default=512)
    parser.add_argument('--seed', type=int, default=20260824)
    args = parser.parse_args()
    if bool(args.member_file) != bool(args.nonmember_file):
        parser.error('provide both --member-file and --nonmember-file, or neither')
    provenance = None
    if args.member_file is None:
        downloaded = download_mimir(source=args.source, split=args.split, cache_size=args.cache_size)
        args.member_file, args.nonmember_file = downloaded['member_file'], downloaded['nonmember_file']
        provenance = downloaded['source_provenance']
    result = freeze(args.member_file, args.nonmember_file, args.output_dir, source=args.source,
        split=args.split, n_per_class=args.n_per_class, n_aux=args.n_aux, max_tokens=args.max_tokens,
        seed=args.seed, source_provenance=provenance)
    print(json.dumps({'manifest': str(result)}), flush=True)


if __name__ == '__main__':
    main()
