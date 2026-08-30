from __future__ import annotations

import hashlib
import json
import random
from pathlib import Path
from typing import Any

from .data import SFTRecord


def _hash_ids(ids: list[int]) -> str:
    payload = b"".join(int(token_id).to_bytes(4, "little", signed=False) for token_id in ids)
    return hashlib.sha256(payload).hexdigest()


def _token_ngrams(record: SFTRecord, n: int) -> set[tuple[int, ...]]:
    prompt = list(record.prompt_ids or ())
    tokens = prompt + list(record.response_ids)
    return {tuple(tokens[index : index + n]) for index in range(len(tokens) - n + 1)}


def _cross_split_ngram_audit(
    splits: list[list[SFTRecord]], n: int = 13
) -> dict[str, Any]:
    owners: dict[tuple[int, ...], list[tuple[int, int]]] = {}
    sizes: dict[tuple[int, int], int] = {}
    for split_index, records in enumerate(splits):
        for record_index, record in enumerate(records):
            grams = _token_ngrams(record, n)
            sizes[(split_index, record_index)] = len(grams)
            for gram in grams:
                owners.setdefault(gram, []).append((split_index, record_index))

    pair_counts: dict[tuple[tuple[int, int], tuple[int, int]], int] = {}
    shared_ngrams = 0
    for gram_owners in owners.values():
        split_ids = {owner[0] for owner in gram_owners}
        if len(split_ids) < 2:
            continue
        shared_ngrams += 1
        for left_index, left in enumerate(gram_owners):
            for right in gram_owners[left_index + 1 :]:
                if left[0] == right[0]:
                    continue
                key = (left, right) if left < right else (right, left)
                pair_counts[key] = pair_counts.get(key, 0) + 1

    maximum = 0.0
    maximum_pair: tuple[tuple[int, int], tuple[int, int]] | None = None
    for pair, count in pair_counts.items():
        denominator = max(1, min(sizes[pair[0]], sizes[pair[1]]))
        fraction = count / denominator
        if fraction > maximum:
            maximum = fraction
            maximum_pair = pair
    if maximum > 0.80:
        raise RuntimeError(
            "Cross-split 13-gram overlap exceeds the preregistered 80% threshold"
        )
    return {
        "n": n,
        "unique_ngrams": len(owners),
        "cross_split_shared_ngrams": shared_ngrams,
        "maximum_pair_overlap_fraction": maximum,
        "maximum_pair_indices": maximum_pair,
        "threshold": 0.80,
        "gate": "PASS",
    }


def build_public_snapshot_split(
    path: Path,
    tokenizer: Any,
    response_tokens: int,
    n_per_class: int,
    n_aux: int,
    seed: int,
) -> tuple[list[SFTRecord], list[SFTRecord], list[SFTRecord], dict[str, Any]]:
    manifest_path = path.with_suffix(".manifest.json")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    raw_bytes = path.read_bytes()
    actual_sha = hashlib.sha256(raw_bytes).hexdigest()
    if actual_sha != manifest["jsonl_sha256"]:
        raise RuntimeError("Public snapshot hash does not match its manifest")
    documents = [
        json.loads(line) for line in raw_bytes.decode("utf-8").splitlines() if line
    ]
    required = 2 * n_per_class + n_aux
    if len(documents) < required:
        raise RuntimeError(f"Need {required} public documents, found {len(documents)}")

    random.Random(seed).shuffle(documents)
    selected = documents[:required]
    records: list[SFTRecord] = []
    response_hashes: set[str] = set()
    for document_index, document in enumerate(selected):
        prompt_ids: list[int] | None = None
        if "response_ids" in document:
            response_ids = [int(value) for value in document["response_ids"]]
            prompt_ids = [int(value) for value in document.get("prompt_ids", [])]
            page_id = str(document.get("record_id", document_index))
        else:
            ids = list(
                tokenizer(
                    document["text"],
                    add_special_tokens=False,
                    truncation=True,
                    max_length=response_tokens,
                ).input_ids
            )
            if len(ids) < response_tokens:
                raise RuntimeError(
                    f"Document {document['page_id']} has only {len(ids)} tokens after snapshot filtering"
                )
            response_ids = ids[:response_tokens]
            page_id = str(document["page_id"])
        if len(response_ids) < response_tokens:
            raise RuntimeError(
                f"Document {page_id} has only {len(response_ids)} response tokens"
            )
        response_ids = response_ids[:response_tokens]
        response_hash = _hash_ids(response_ids)
        if response_hash in response_hashes:
            raise RuntimeError("Token-level duplicate appeared in the selected public split")
        response_hashes.add(response_hash)
        records.append(
            SFTRecord(
                record_id=(
                    f"public:{page_id}:{document.get('snapshot_revision', 0)}:"
                    f"{response_hash[:16]}"
                ),
                source=document.get("source", document.get("canonical_url", page_id)),
                response_ids=tuple(response_ids),
                response_hash=response_hash,
                prompt_ids=tuple(prompt_ids) if prompt_ids else None,
                prompt_hash=_hash_ids(prompt_ids) if prompt_ids else "",
                prompt_text=document.get("prompt_text"),
                topic=document.get("title"),
                source_char_count=int(
                    document.get("source_char_count", len(document.get("text", "")))
                ),
                source_timestamp=document.get("creation_timestamp", ""),
                source_revision=int(document.get("snapshot_revision", 0)),
            )
        )

    members = records[:n_per_class]
    nonmembers = records[n_per_class : 2 * n_per_class]
    auxiliary = records[2 * n_per_class :]
    ngram_audit = _cross_split_ngram_audit([members, nonmembers, auxiliary])
    metadata = {
        "dataset": manifest["dataset"],
        "license": manifest["license"],
        "creation_interval_inclusive": manifest["creation_interval_inclusive"],
        "snapshot_path": str(path),
        "snapshot_sha256": actual_sha,
        "snapshot_records": len(documents),
        "split_seed": seed,
        "split_unit": manifest.get("split_unit", "public document"),
        "timestamp_semantics": manifest.get("timestamp_semantics", "unspecified"),
        "counts": {
            "member": len(members),
            "nonmember": len(nonmembers),
            "auxiliary": len(auxiliary),
        },
        "response_tokens": response_tokens,
        "target_sft_uses": "member only",
        "exact_token_deduplication": True,
        "raw_activations_persisted": False,
        "prompt_mode": (
            "fixed_token_continuation" if records[0].prompt_ids is not None else "instruction"
        ),
        "prompt_tokens": (
            len(records[0].prompt_ids) if records[0].prompt_ids is not None else None
        ),
        "cross_split_ngram_audit": ngram_audit,
    }
    return members, nonmembers, auxiliary, metadata


# Backward-compatible name for the original Wikipedia-specific pilot design.
build_public_wikipedia_split = build_public_snapshot_split
