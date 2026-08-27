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
    for document in selected:
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
        response_hash = _hash_ids(response_ids)
        if response_hash in response_hashes:
            raise RuntimeError("Token-level duplicate appeared in the selected public split")
        response_hashes.add(response_hash)
        records.append(
            SFTRecord(
                record_id=(
                    f"public:{document['page_id']}:{document['snapshot_revision']}:"
                    f"{response_hash[:16]}"
                ),
                source=document["canonical_url"],
                response_ids=tuple(response_ids),
                response_hash=response_hash,
                topic=document["title"],
                source_char_count=len(document["text"]),
                source_timestamp=document["creation_timestamp"],
                source_revision=int(document["snapshot_revision"]),
            )
        )

    members = records[:n_per_class]
    nonmembers = records[n_per_class : 2 * n_per_class]
    auxiliary = records[2 * n_per_class :]
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
    }
    return members, nonmembers, auxiliary, metadata


# Backward-compatible name for the original Wikipedia-specific pilot design.
build_public_wikipedia_split = build_public_snapshot_split
