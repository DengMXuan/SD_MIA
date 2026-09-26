#!/usr/bin/env python3
"""Versioned, standalone cleaning of the frozen Qwen temporal experiment.

No model inference, score-based selection, or edits to the source data.
"""
from __future__ import annotations

from collections import Counter, defaultdict
import hashlib
import html
import json
from pathlib import Path
import re
import shutil
import tempfile
import unicodedata

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
SOURCE = Path('/home/mxd/lib/SD_MIA-pretraining-data/qwen3_temporal_shared_split_v1')
OUTPUT = ROOT/'artifacts/data/qwen_temporal_clean_v2'
SEEDS = (1919, 1949, 1978)
VARIANTS = ('clean_only', 'length_matched')
VERSION = 'qwen_temporal_clean_v2'
GROUPS = ('member', 'nonmember', 'auxiliary')
MARKER = re.compile(r'@[.,-]@')
SPACED = re.compile(r'\s+[,.;:!?%]|[\(\[]\s+|\s+[\)\]]')
EDITORIAL = re.compile(r'\[\s*(?:\d+|edit|citation needed|note\s+\d+)\s*\]', re.I)
INVISIBLE = re.compile('[\ufeff\u200b\u00ad]')
BOILERPLATE = re.compile(
    r'^(?:You can help expand this article with text translated|'
    r'View a machine-translated version of|'
    r'Machine translation, like DeepL or Google Translate|'
    r'Consider adding a topic to this template:|'
    r'Do not translate text that appears unreliable or low-quality|'
    r'You must provide copyright attribution in the edit summary|'
    r'You may also add the template \{\{Translated|'
    r'For more guidance, see Wikipedia\s*:\s*Translation|'
    r'This article.{0,160}\bis a stub\b|'
    r'Learn how and when to remove this (?:template )?message|'
    r'For other uses, see .+\(disambiguation\))', re.I)
CONTRACT = {'mode': 'raw_text_completion', 'add_special_tokens': False,
    'first_token': 'context_only', 'append_eos': False,
    'scored_tokens': 'text tokens at positions 1..L-1',
    'vocabulary': 'shared tokenizer IDs only; padded LM-head rows excluded and distributions renormalized'}


def require(ok, message):
    if not ok:
        raise ValueError(message)


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda: f.read(1 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


def hash_ids(ids):
    return hashlib.sha256(np.asarray(ids, dtype='<u4').tobytes()).hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text())


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False)+'\n')


def clean_text(text):
    """The same lexical rules for all roles; keep ordinary @, emails and math.

    Restore WikiText punctuation, not all @ characters. Do not rewrite facts,
    infer prose boundaries from grammar, or discard short selected documents.
    """
    text = unicodedata.normalize('NFC', html.unescape(text)).replace('\r\n', '\n').replace('\r', '\n')
    text = INVISIBLE.sub('', text).replace('\xa0', ' ')
    # Known WikiText placeholders are glued to both adjacent text fragments.
    text = re.sub(r'[ \t]*@([.,-])@[ \t]*', r'\1', text)
    text = EDITORIAL.sub('', text)
    lines = []
    for line in text.splitlines():
        line = line.strip()
        if re.fullmatch(r'(?:References|External links|Bibliography|Further reading|Notes and references)', line, re.I):
            break
        if not line or re.match(r'^(?:\^|↑)\s', line) or re.fullmatch(r'={2,}.*?={2,}', line):
            continue
        if BOILERPLATE.match(line):
            continue
        line = re.sub(r'[ \t]+', ' ', line)
        line = re.sub(r'\s+([,.;:!?%])', r'\1', line)
        line = re.sub(r'([\(\[])\s+', r'\1', line)
        line = re.sub(r'\s+([\)\]])', r'\1', line)
        line = re.sub(r'(\w)\s+([\'’](?:s|re|ve|ll|d|m|t))\b(?![\'’])', r'\1\2', line)
        line = re.sub(r'(\w)\s+(n[\'’]t)\b', r'\1\2', line, flags=re.I)
        # Include empty quote pairs so repeated quotes cannot change pairing on
        # a second pass. Leave unbalanced quotes and ambiguous apostrophes alone.
        if line.count('"') % 2 == 0:
            line = re.sub(r'"([^"\n]*)"', lambda m: '"'+m[1].strip()+'"', line)
        lines.append(line.strip())
    return '\n'.join(lines)


def match_lengths(capacities, targets, seed):
    """Match the exact negative length histogram without replacing any ID.

    Randomize ties reproducibly, then pair sorted capacities/targets. If this
    monotone allocation fails, no feasible allocation exists; never pad text.
    """
    cap, target = np.asarray(capacities), np.asarray(targets)
    require(len(cap) == len(target) and len(cap) > 0, 'class counts must match')
    require(np.all(cap >= 2) and np.all(target >= 2), 'empty/one-token document')
    rng = np.random.default_rng(seed)
    order = rng.permutation(len(cap))
    order = order[np.argsort(cap[order], kind='stable')]
    desired = np.sort(target)
    require(np.all(cap[order] >= desired), 'historical texts cannot supply the requested recent length distribution')
    result = np.empty(len(cap), dtype=int)
    result[order] = desired
    return result.tolist()


def read_source(source, seed):
    path = Path(source)/f'seed{seed}/manifest.json'
    manifest = read_json(path)
    require(manifest['kind'] == 'temporal_pretraining_v1' and manifest['selection_seed'] == seed
            and manifest['membership_verified'] is False and manifest['token_contract'] == CONTRACT,
            'source temporal/seed/label/token contract mismatch')
    require(manifest['models']['target']['repo_id'] == 'Qwen/Qwen3-8B-Base' and
            manifest['models']['draft']['repo_id'] == 'Qwen/Qwen3-1.7B-Base', 'wrong model pair')
    records_path = path.parent/manifest['records_file']
    require(sha256(records_path) == manifest['records_sha256'], 'source records checksum mismatch')
    rows = [json.loads(line) for line in records_path.read_text().splitlines()]
    require(dict(Counter(r['group'] for r in rows)) == manifest['counts'], 'source count mismatch')
    require(len({r['record_id'] for r in rows}) == len(rows), 'duplicate source IDs')
    require(all(r['group'] in GROUPS and r['label'] == int(r['group']=='member') and
                hash_ids(r['token_ids']) == r['token_hash'] for r in rows), 'invalid source labels/tokens')
    return path, manifest, rows


def load_tokenizer(manifest, model_root):
    from transformers import AutoTokenizer
    tokenizers = []
    for spec in manifest['models'].values():
        path = Path(model_root)/('models--'+spec['repo_id'].replace('/', '--'))/'snapshots'/spec['revision']
        tok = AutoTokenizer.from_pretrained(path, local_files_only=True)
        state = json.loads(tok.backend_tokenizer.to_str())
        state.pop('truncation', None); state.pop('padding', None)
        require(hashlib.sha256(json.dumps(state, sort_keys=True).encode()).hexdigest() == manifest['tokenizer_sha256'],
                'pinned tokenizer mismatch')
        tokenizers.append(tok)
    return tokenizers[0]


def surface(text, ids):
    words = re.findall(r'\b\w+\b', text)
    return dict(tokens=len(ids), characters_per_token=len(text)/len(ids),
        placeholder_count=len(MARKER.findall(text)),
        spaced_punctuation_count=len(SPACED.findall(text)),
        editorial_marker_count=len(EDITORIAL.findall(text)),
        translation_or_ui_lines=sum(bool(BOILERPLATE.match(line.strip())) for line in text.splitlines()),
        invisible_count=len(INVISIBLE.findall(text)),
        newline_per_100_words=100*text.count('\n')/max(1,len(words)),
        digit_fraction=sum(c.isdigit() for c in text)/max(1,len(text)),
        punctuation_fraction=sum(unicodedata.category(c).startswith('P') for c in text)/max(1,len(text)))


def distribution_audit(stages, tokenizer):
    from scipy.stats import ks_2samp
    groups, comparisons = {}, {}
    for name, rows in stages.items():
        stats = defaultdict(list)
        texts = tokenizer.batch_decode([r['token_ids'] for r in rows], skip_special_tokens=False)
        for row,text in zip(rows,texts):
            stats[row['group']].append(surface(text,row['token_ids']))
        groups[name] = {}
        for group, values in stats.items():
            groups[name][group] = dict(n=len(values), metrics={})
            for metric in values[0]:
                x = np.asarray([v[metric] for v in values])
                groups[name][group]['metrics'][metric] = dict(mean=float(x.mean()), minimum=float(x.min()),
                    maximum=float(x.max()), p10=float(np.quantile(x,.1)), median=float(np.median(x)),
                    p90=float(np.quantile(x,.9)), documents_with_nonzero_percent=float(100*np.mean(x!=0)))
        comparisons[name] = {}
        for metric in stats['member'][0]:
            x = np.asarray([v[metric] for v in stats['member']])
            y = np.asarray([v[metric] for v in stats['nonmember']])
            denom = np.sqrt((x.var(ddof=1)+y.var(ddof=1))/2)
            comparisons[name][metric] = dict(ks_statistic=float(ks_2samp(x,y,method='asymp').statistic),
                standardized_mean_difference=float((x.mean()-y.mean())/denom) if denom else 0.)
    return dict(groups=groups, member_vs_nonmember=comparisons,
                scope='descriptive diagnostics of visible prefixes; no claim of identical topics or membership ground truth')


def deduplicate_audit(rows, tokenizer):
    """Exact token gate plus symmetric cross-role 13-word overlap check."""
    tokens, owners, sizes, previous, near = {}, defaultdict(list), [], [], []
    texts = tokenizer.batch_decode([r['token_ids'] for r in rows], skip_special_tokens=False)
    for row,text in zip(rows,texts):
        key = hash_ids(row['token_ids'])
        require(key not in tokens, f'cleaning created exact duplicate: {row["record_id"]} / {tokens.get(key)}')
        tokens[key] = row['record_id']
        words = re.sub('[^a-z0-9 ]+', ' ', text.lower()).split()
        grams = {hashlib.blake2b(' '.join(words[i:i+13]).encode(),digest_size=8).digest()
                 for i in range(max(0,len(words)-12))}
        hits = Counter(j for g in grams for j in owners[g])
        for j,count in hits.items():
            if previous[j]['group'] != row['group'] and count >= .5*min(len(grams), sizes[j]):
                near.append(dict(left=previous[j]['record_id'], right=row['record_id'], shared_13grams=count,
                                 left_group=previous[j]['group'],right_group=row['group'],
                                 cross_label=previous[j]['label']!=row['label'],
                                 overlap_shorter=count/min(len(grams), sizes[j])))
        for g in grams:
            owners[g].append(len(previous))
        previous.append(row); sizes.append(len(grams))
    return dict(exact_token_duplicates=0, cross_role_near_duplicates=len(near), cross_role_near_pairs=near,
                near_definition='at least 50% of unique 13-word shingles of the shorter set; all cross-role pairs')


def plan(source, seed, variants):
    path, manifest, rows = read_source(source,seed)
    return dict(schema=VERSION, seed=seed, variants=list(variants),
        source_manifest=str(path.resolve()), source_manifest_sha256=sha256(path),
        source_records_sha256=manifest['records_sha256'], cleaning_code_sha256=sha256(__file__),
        preserve='document IDs, source text provenance, group order, labels, model pins and condition seed',
        clean='identical WikiText punctuation restoration and surface normalization for all groups before tokenization',
        length_matched='historical prefix lengths match cleaned test nonmember histogram exactly; capacity-sort with seeded tie order',
        boundary='surface/length matching does not establish semantic exchangeability or verified training membership')


def validate_output(folder, expected=None):
    folder = Path(folder)
    complete = read_json(folder/'_COMPLETE.json')
    for name, expected_hash in complete.items():
        require(sha256(folder/name)==expected_hash, f'output checksum mismatch: {name}')
    saved = read_json(folder/'REQUEST.json')
    if expected is not None:
        require(saved==expected, 'prepared sources/options/cleaner changed; use a new output directory')
    for variant in saved['variants']:
        m=read_json(folder/variant/'manifest.json')
        require(sha256(folder/variant/m['records_file'])==m['records_sha256'], 'derived records changed')
    return saved


def prepare(source, output, seed, *, model_root, variants=VARIANTS):
    source, output = Path(source).resolve(), Path(output).resolve()
    require(source!=output and source not in output.parents and output not in source.parents, 'output overlaps source data')
    request = plan(source,seed,variants)
    destination = output/f'seed{seed}'
    if destination.exists():
        validate_output(destination,request)
        print(f'seed{seed}: reuse validated {destination}',flush=True)
        return
    path, original, rows = read_source(source,seed)
    tokenizer = load_tokenizer(original,model_root)
    cleaned_text = [clean_text(r['text']) for r in rows]
    require(all(clean_text(t)==t for t in cleaned_text), 'cleaner must be idempotent')
    require(all(not MARKER.search(t) for t in cleaned_text), 'unremoved WikiText placeholder')
    ids = tokenizer(cleaned_text, add_special_tokens=False, truncation=True,
                    max_length=original['max_tokens']).input_ids
    require(all(len(v)>=2 for v in ids), 'cleaning removed a selected document; no silent replacement allowed')
    cleaned = []
    for row,text,tokens in zip(rows,cleaned_text,ids):
        cleaned.append({**row,'text':text,'token_ids':tokens,'token_hash':hash_ids(tokens),
            'cleaning_metadata':dict(version=VERSION, source_text_sha256=hashlib.sha256(row['text'].encode()).hexdigest(),
                original_token_hash=row['token_hash'], original_token_count=len(row['token_ids']),
                clean_only_token_count=len(tokens), text_changed=text!=row['text'])})
    stages = {'original':rows,'clean_only':cleaned}
    if 'length_matched' in variants:
        members = [r for r in cleaned if r['group']=='member']
        recent = [r for r in cleaned if r['group']=='nonmember']
        lengths = match_lengths([len(r['token_ids']) for r in members], [len(r['token_ids']) for r in recent],seed)
        caps = {r['record_id']:n for r,n in zip(members,lengths)}
        matched = []
        for r in cleaned:
            cap=caps.get(r['record_id'],len(r['token_ids']))
            tokens=r['token_ids'][:cap]
            matched.append({**r,'token_ids':tokens,'token_hash':hash_ids(tokens),
                            'cleaning_metadata':{**r['cleaning_metadata'],'visible_token_limit':cap}})
        stages['length_matched']=matched
    audit=distribution_audit(stages,tokenizer)
    overlap={variant:deduplicate_audit(stages[variant],tokenizer) for variant in variants}
    for variant, result in overlap.items():
        cross_label=[pair for pair in result['cross_role_near_pairs'] if pair['cross_label']]
        require(not cross_label, f'cleaned member/nonmember near overlap: {cross_label[:3]}')
        result['gate']='no exact duplicates in any role; no cross-label near overlaps'
        result['same_label_overlap_policy']='report nonmember/auxiliary passage overlap; preserve frozen article IDs'
    require(all(r['record_id']==old['record_id'] and r['group']==old['group'] and r['label']==old['label']
                for variant in variants for r,old in zip(stages[variant],rows)), 'identity/role order changed')
    if 'length_matched' in variants:
        require(audit['member_vs_nonmember']['length_matched']['tokens']['ks_statistic']==0, 'length matching failed')
    # Recheck sources immediately before publishing.
    require(plan(source,seed,variants)==request, 'source changed during preparation')
    output.mkdir(parents=True,exist_ok=True)
    temporary=Path(tempfile.mkdtemp(prefix=f'.seed{seed}-',dir=output))
    try:
        write_json(temporary/'REQUEST.json',request)
        write_json(temporary/'AUDIT.json',dict(seed=seed, **audit, deduplication=overlap))
        for variant in variants:
            folder=temporary/variant; folder.mkdir()
            selected=stages[variant]
            records_file=folder/'records.jsonl'
            records_file.write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in selected))
            lengths={g:[len(r['token_ids']) for r in selected if r['group']==g] for g in GROUPS}
            manifest={**original,'benchmark':f'wiki_temporal/{VERSION}/{variant}',
                'records_sha256':sha256(records_file),'min_tokens':min(map(min,lengths.values())),
                'source_provenance':dict(parent_manifest=str(path.resolve()),parent_manifest_sha256=sha256(path),
                                         original=original['source_provenance'],cleaning=request),
                'token_lengths':{g:dict(min=min(v),max=max(v),mean=float(np.mean(v))) for g,v in lengths.items()},
                'filtering':dict(selection='same frozen IDs/roles/order; no dropped or replacement documents',
                                 text_normalization=VERSION,length_policy=variant,**overlap[variant]),
                'caveats':['historical presumed members and post-release nonmembers remain temporal proxies',
                           'surface normalization changes token sequences; document IDs/roles remain anchored',
                           'cleaning and length matching do not guarantee identical topics or extraction style',
                           'nonmember/auxiliary near passage overlaps are reported in filtering; frozen article IDs are retained',
                           'recent IDs reuse the SFT split; cleaned token sequences are a new derived experiment'],
                'cleaning_variant':variant}
            manifest.pop('historical_min_tokens',None)
            write_json(folder/'manifest.json',manifest)
        files=[p for p in temporary.rglob('*') if p.is_file()]
        write_json(temporary/'_COMPLETE.json',{str(p.relative_to(temporary)):sha256(p) for p in files})
        temporary.rename(destination)
    finally:
        if temporary.exists():shutil.rmtree(temporary)
    print(f'seed{seed}: prepared {len(rows)} documents, '+', '.join(variants),flush=True)


def summarize_data(output,seeds):
    lines=['# Qwen temporal surface/length audit','',
        '| Seed | Stage | Historical @ artifacts | Recent @ artifacts | Historical mean tokens | Recent mean tokens | Length KS |',
        '|---|---|---:|---:|---:|---:|---:|']
    reports=[]
    for seed in seeds:
        folder=Path(output)/f'seed{seed}'
        validate_output(folder)
        a=read_json(folder/'AUDIT.json'); reports.append(a)
        for stage,groups in a['groups'].items():
            m,n=groups['member']['metrics'],groups['nonmember']['metrics']
            lines.append(f'| {seed} | {stage} | {m["placeholder_count"]["documents_with_nonzero_percent"]:.2f}% | '
                f'{n["placeholder_count"]["documents_with_nonzero_percent"]:.2f}% | {m["tokens"]["mean"]:.3f} | '
                f'{n["tokens"]["mean"]:.3f} | {a["member_vs_nonmember"][stage]["tokens"]["ks_statistic"]:.4f} |')
    text='\n'.join(lines)+'\n\nKS here is descriptive. Equal token lengths do not prove equal semantic distributions.\n'
    Path(output,'SUMMARY.md').write_text(text)
    write_json(Path(output,'SUMMARY.json'),reports)
    print(text,flush=True)
