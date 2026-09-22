"""Select additional trusted nonmembers without changing frozen manifests."""
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
import random

from experiments.sd_membership_sft.datasets.data import SFTRecord, _hash_ids
from experiments.sd_membership_sft.datasets.splits import (
    SFT_PROMPT, _document_identity, _selection_gram_hashes,
)
from .storage import checked_contract, digest, file_sha, workspace


def _read_pool(benchmark, path):
    """Verify an immutable pool without locking or repairing shared artifacts."""
    path = Path(path)
    if path.with_name(f".{path.name}.transaction.json").exists():
        raise ValueError("pool has an unfinished transaction; resource selection will not repair it")
    manifest = json.loads(path.with_suffix(".manifest.json").read_text())
    payload = path.read_bytes()
    sha = hashlib.sha256(payload).hexdigest()
    if sha != manifest.get("jsonl_sha256") or manifest.get("benchmark") != benchmark:
        raise ValueError("frozen pool manifest/hash mismatch")
    documents = [json.loads(line) for line in payload.decode().splitlines() if line]
    if manifest.get("records", len(documents)) != len(documents):
        raise ValueError("pool record count mismatch")
    return documents, sha


class _TokenIndex:
    """Exact response and 13-gram checks against every anchored record."""

    def __init__(self):
        self.hashes = set()
        self.sizes = []
        self.owners = defaultdict(list)

    def accepts(self, tokens):
        if _hash_ids(tokens) in self.hashes:
            return False
        grams = _selection_gram_hashes(tokens)
        hits = Counter(owner for gram in grams for owner in self.owners.get(gram, ()))
        return not any(count >= .5 * min(len(grams), self.sizes[owner])
                       for owner, count in hits.items())

    def add(self, tokens):
        grams = _selection_gram_hashes(tokens)
        owner = len(self.sizes)
        self.hashes.add(_hash_ids(tokens))
        self.sizes.append(len(grams))
        for gram in grams:
            self.owners[gram].append(owner)


def _tokenize(document, tokenizer, band):
    return list(tokenizer(str(document["text"]), add_special_tokens=False,
                          truncation=True, max_length=band["max_tokens"]).input_ids)


def select_extension(pool_path, shared_manifest_path, tokenizers, *, count, seed=20260922):
    """Return a separate manifest; no files are changed and no models are loaded.

    Supply the exact tokenizer-source mapping audited by the frozen split, so
    future model comparisons use the same eligible extension IDs. Original
    assigned records are immutable anchors, even if another tokenizer would
    prefer to drop one while deduplicating.
    """
    if type(count) is not int or count < 0:
        raise ValueError("extension count must be a nonnegative integer")
    shared_path = Path(shared_manifest_path)
    shared = json.loads(shared_path.read_text())
    if shared.get("schema_version") != 3 or "audit_auxiliary" not in shared["splits"]:
        raise ValueError("a frozen four-role shared split is required")
    if not tokenizers or set(tokenizers) != set(shared["tokenizer_sources"]):
        raise ValueError("supply all tokenizer sources audited by the frozen split")
    documents, pool_sha = _read_pool(shared["benchmark"], pool_path)
    if pool_sha != shared["pool_sha256"]:
        raise ValueError("extension pool differs from the frozen training pool")
    by_id = {}
    for document in documents:
        record_id, _ = _document_identity(document)
        if record_id in by_id:
            raise ValueError("duplicate pool record ID")
        by_id[record_id] = document
    assigned = [entry for rows in shared["splits"].values() for entry in rows]
    excluded = {entry["record_id"] for entry in assigned}
    if len(excluded) != len(assigned):
        raise ValueError("frozen split roles overlap")
    band = shared["token_band"]
    indices = {name: _TokenIndex() for name in tokenizers}
    seen_text = set()
    for entry in assigned:
        document = by_id[entry["record_id"]]
        if _document_identity(document)[1] != entry["text_sha256"]:
            raise ValueError("frozen split text identity changed")
        seen_text.add(entry["text_sha256"])
        for name, tokenizer in tokenizers.items():
            tokens = _tokenize(document, tokenizer, band)
            if not band["min_tokens"] <= len(tokens) <= band["max_tokens"]:
                raise ValueError("frozen assigned record fails tokenizer band")
            indices[name].add(tokens)
    candidates = [document for document in documents if document["record_id"] not in excluded]
    random.Random(seed).shuffle(candidates)
    selected, rejected = [], Counter()
    for document in candidates:
        if len(selected) == count:
            break
        record_id, text_sha = _document_identity(document)
        if text_sha in seen_text:
            rejected["raw_duplicate"] += 1
            continue
        responses = {name: _tokenize(document, tokenizer, band) for name, tokenizer in tokenizers.items()}
        if any(not band["min_tokens"] <= len(tokens) <= band["max_tokens"] for tokens in responses.values()):
            rejected["token_band"] += 1
            continue
        if any(not indices[name].accepts(tokens) for name, tokens in responses.items()):
            rejected["token_or_13gram_duplicate"] += 1
            continue
        seen_text.add(text_sha)
        for name, tokens in responses.items():
            indices[name].add(tokens)
        selected.append({"record_id": record_id, "text_sha256": text_sha,
                         "response_hashes": {name: _hash_ids(tokens) for name, tokens in responses.items()}})
    if len(selected) != count:
        raise ValueError(f"insufficient eligible unassigned nonmembers: need {count}, found {len(selected)}; "
                         f"raw unused={len(candidates)}, rejected={dict(rejected)}")
    return {"schema": "resource_auxiliary_extension_v1", "benchmark": shared["benchmark"],
            "shared_split_path": str(shared_path.resolve()), "shared_split_sha256": file_sha(shared_path),
            "shared_split_digest": digest(shared), "pool_path": str(Path(pool_path).resolve()),
            "pool_sha256": pool_sha, "seed": seed, "token_band": band,
            "tokenizer_sources": sorted(tokenizers), "raw_unassigned_count": len(candidates),
            "excluded_roles": sorted(shared["splits"]), "rejected": dict(rejected),
            "near_duplicate_ngram": 13, "near_duplicate_threshold": .5, "records": selected}


def save_extension(folder, manifest):
    with workspace(folder) as output:
        checked_contract(output, "EXTENSION.json", manifest)
    return output / "EXTENSION.json"


def extension_records(manifest, tokenizer, tokenizer_source):
    """Materialize audited extension IDs as the existing SFTRecord interface."""
    if tokenizer_source not in manifest["tokenizer_sources"]:
        raise ValueError("tokenizer was not audited for this extension")
    if file_sha(manifest["shared_split_path"]) != manifest["shared_split_sha256"]:
        raise ValueError("frozen split changed")
    documents, pool_sha = _read_pool(manifest["benchmark"], manifest["pool_path"])
    if pool_sha != manifest["pool_sha256"]:
        raise ValueError("frozen pool changed")
    by_id = {document["record_id"]: document for document in documents}
    records = []
    for entry in manifest["records"]:
        document = by_id[entry["record_id"]]
        response = _tokenize(document, tokenizer, manifest["token_band"])
        if (_document_identity(document)[1] != entry["text_sha256"]
                or _hash_ids(response) != entry["response_hashes"][tokenizer_source]):
            raise ValueError("extension raw text or tokenizer output changed")
        prompt = SFT_PROMPT.format(topic=str(document.get("title", "a public document")))
        prompt_ids = list(tokenizer(prompt, add_special_tokens=False).input_ids)
        records.append(SFTRecord(record_id=entry["record_id"], source=str(document.get("source", "pool")),
                                 response_ids=tuple(response), response_hash=_hash_ids(response),
                                 prompt_ids=tuple(prompt_ids), prompt_hash=_hash_ids(prompt_ids), prompt_text=prompt))
    return records
