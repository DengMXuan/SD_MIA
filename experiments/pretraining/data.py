"""Frozen, externally labelled text records and pinned pretrained models."""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, SuppressTokensLogitsProcessor

from experiments.shared.data.data import SFTRecord, _hash_ids

TARGET = {'repo_id': 'EleutherAI/pythia-6.9b', 'revision': '21bfa02e806e253fe453702c29c81d9f83617255'}
DRAFT = {'repo_id': 'EleutherAI/pythia-1.4b', 'revision': '9cc5c8c8148a4e0115d9e29c6b4f21124cfe748a'}
MIMIR_REVISION = '02500d3b7cece0cb7628e939ba9fc93fdb6362ae'
TOKEN_CONTRACT = {
    'mode': 'raw_text_completion', 'add_special_tokens': False,
    'first_token': 'context_only', 'append_eos': False,
    'scored_tokens': 'text tokens at positions 1..L-1',
    'vocabulary': 'shared tokenizer IDs only; padded LM-head rows excluded and distributions renormalized',
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def load_tokenizer(spec):
    tokenizer = AutoTokenizer.from_pretrained(spec['repo_id'], revision=spec['revision'])
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    if set(tokenizer.get_vocab().values()) != set(range(len(tokenizer))):
        raise ValueError('pretraining requires a contiguous shared tokenizer ID space')
    return tokenizer


def tokenizer_hash(tokenizer):
    # Include segmentation/normalization, not just token-ID correspondence.
    state = json.loads(tokenizer.backend_tokenizer.to_str())
    # Fast tokenizers mutate these per-call options when truncation is used.
    # They are evaluation settings, not tokenizer vocabulary/segmentation.
    state.pop('truncation', None)
    state.pop('padding', None)
    return hashlib.sha256(json.dumps(state, sort_keys=True).encode()).hexdigest()


class ValidTokenModel(torch.nn.Module):
    """Use the same real token support despite different padded Pythia heads."""
    def __init__(self, model, vocab_size):
        super().__init__()
        self.model = model
        self.config = model.config
        self.vocab_size = vocab_size
        if vocab_size > model.get_output_embeddings().weight.shape[0]:
            raise ValueError('model output head is smaller than tokenizer vocabulary')

    def forward(self, *args, **kwargs):
        output = self.model(*args, **kwargs)
        output.logits = output.logits[..., :self.vocab_size]
        return output

    def get_input_embeddings(self):
        return self.model.get_input_embeddings()

    def generate(self, **kwargs):
        processors = list(kwargs.pop('logits_processor', []))
        processors.append(SuppressTokensLogitsProcessor(list(range(self.vocab_size, self.config.vocab_size))))
        return self.model.generate(**kwargs, logits_processor=processors)


def load_model(spec, device, attn_implementation='sdpa', vocab_size=None):
    if vocab_size is None:
        vocab_size = len(load_tokenizer(spec))
    model = AutoModelForCausalLM.from_pretrained(
        spec['repo_id'], revision=spec['revision'],
        torch_dtype=torch.bfloat16 if device.type == 'cuda' else torch.float32,
        attn_implementation=attn_implementation,
    ).to(device).eval()
    return ValidTokenModel(model, vocab_size).eval()


@dataclass
class Evaluation:
    manifest_path: Path
    manifest: dict
    tokenizer: object
    members: list[SFTRecord]
    nonmembers: list[SFTRecord]
    auxiliary: list[SFTRecord]

    @property
    def config(self):
        return SimpleNamespace(target_model=self.manifest['models']['target']['repo_id'],
            draft_model=self.manifest['models']['draft']['repo_id'], benchmark=self.manifest['benchmark'])


def load_evaluation(manifest_path: Path, verify_draft=False):
    manifest_path = manifest_path.resolve()
    manifest = json.loads(manifest_path.read_text())
    if manifest.get('kind') != 'mimir_pretraining_v1' or manifest.get('token_contract') != TOKEN_CONTRACT:
        raise ValueError('unsupported pretraining manifest/token contract')
    tokenizer = load_tokenizer(manifest['models']['target'])
    if tokenizer_hash(tokenizer) != manifest['tokenizer_sha256']:
        raise ValueError('tokenizer changed since MIMIR preparation')
    if verify_draft:
        draft_tokenizer = load_tokenizer(manifest['models']['draft'])
        if tokenizer_hash(draft_tokenizer) != manifest['tokenizer_sha256']:
            raise ValueError('target and draft tokenizers are not identical')
    path = manifest_path.parent / manifest['records_file']
    if sha256(path) != manifest['records_sha256']:
        raise ValueError('frozen MIMIR records hash mismatch')
    groups = {'member': [], 'nonmember': [], 'auxiliary': []}
    seen_ids, seen_tokens = set(), set()
    for line in path.read_text().splitlines():
        row = json.loads(line)
        group = row['group']
        ids = list(row['token_ids'])
        if len(ids) < 2 or len(ids) > manifest['max_tokens'] or any(i < 0 or i >= len(tokenizer) for i in ids):
            raise ValueError('invalid token sequence')
        token_hash = _hash_ids(ids)
        if token_hash != row['token_hash'] or token_hash in seen_tokens or row['record_id'] in seen_ids:
            raise ValueError('duplicate/corrupt MIMIR record identity')
        if row['label'] != (1 if group == 'member' else 0):
            raise ValueError('MIMIR label disagrees with its frozen group')
        seen_ids.add(row['record_id']); seen_tokens.add(token_hash)
        groups[group].append(SFTRecord(record_id=row['record_id'], source=row['source'],
            response_ids=tuple(ids[1:]), response_hash=_hash_ids(ids[1:]),
            prompt_ids=(ids[0],), prompt_text='', append_eos=False))
    if {key: len(value) for key, value in groups.items()} != manifest['counts']:
        raise ValueError('frozen MIMIR group counts disagree with manifest')
    return Evaluation(manifest_path, manifest, tokenizer, groups['member'], groups['nonmember'], groups['auxiliary'])
