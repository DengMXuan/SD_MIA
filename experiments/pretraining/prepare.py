"""Download/import official MIMIR caches and freeze a labelled pretraining audit."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from huggingface_hub import hf_hub_download

from experiments.pretraining.data import TARGET, DRAFT, MIMIR_REVISION, TOKEN_CONTRACT, load_tokenizer, tokenizer_hash, sha256
from experiments.shared.data.data import _hash_ids

SOURCES = ('arxiv', 'dm_mathematics', 'github', 'hackernews', 'pile_cc', 'pubmed_central', 'wikipedia_(en)', 'full_pile')


def freeze(member_file, nonmember_file, output_dir, *, source, split, n_per_class=800,
           n_aux=100, max_tokens=512, seed=20260824, models=None, source_provenance=None):
    if n_per_class < 1 or n_aux < 0 or not 2 <= max_tokens <= 2048:
        raise ValueError('invalid counts or max_tokens (must be 2..2048)')
    output_dir = Path(output_dir)
    if (output_dir / 'manifest.json').exists() or (output_dir / 'records.jsonl').exists():
        raise FileExistsError(f'frozen evaluation already exists: {output_dir}')
    models = models or {'target': TARGET, 'draft': DRAFT}
    tokenizer = load_tokenizer(models['target'])
    draft_tokenizer = load_tokenizer(models['draft'])
    if tokenizer_hash(tokenizer) != tokenizer_hash(draft_tokenizer):
        raise ValueError('target and draft tokenizers differ')
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
        order = np.random.default_rng(seed + label).permutation(len(values))
        groups[label] = [values[i] for i in order]
        stats[str(label)] = dict(usable=len(values), duplicates_removed=duplicates, short_removed=short)
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
        filename = args.source + ('' if args.split == 'none' else '_' + args.split) + '.jsonl'
        prefix = f'cache_100_200_{args.cache_size}_512'
        files = [f'{prefix}/{part}/{filename}' for part in ('train', 'test')]
        args.member_file, args.nonmember_file = [Path(hf_hub_download('iamgroot42/mimir', filename,
            repo_type='dataset', revision=MIMIR_REVISION)) for filename in files]
        provenance = dict(repo_id='iamgroot42/mimir', revision=MIMIR_REVISION, files=files)
    result = freeze(args.member_file, args.nonmember_file, args.output_dir, source=args.source,
        split=args.split, n_per_class=args.n_per_class, n_aux=args.n_aux, max_tokens=args.max_tokens,
        seed=args.seed, source_provenance=provenance)
    print(json.dumps({'manifest': str(result)}), flush=True)


if __name__ == '__main__':
    main()
