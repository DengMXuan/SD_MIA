"""Load a frozen NART benchmark pool into the controlled three-class split.

The pool (see ``nart_benchmarks``) holds raw post-cutoff documents with
provenance. Loading re-tokenizes per target model, applies the NART token band
(128..512 for WikiTection/NewsTection, 1024..2048 for ArXivTection), deduplicates
on token IDs, and splits member/nonmember/auxiliary exactly as
``build_public_snapshot_split`` does, including the 13-gram cross-split gate.

Records use the NART Figure-3 prompt with the document as the continuation;
``make_sft_example`` masks the prompt so only document tokens contribute loss.
"""

from __future__ import annotations

import hashlib
import json
import random
from pathlib import Path
from typing import Any

from .data import SFTRecord, _hash_ids


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


NART_PROMPT = (
    "You are a helpful assistant. Below is a given topic and related contexts. "
    "Please continue writing or analyze the contexts.\nTopic: {topic}\nContext: "
)

DATA_ROOT = Path("experiments/data/nart_benchmarks")

BENCHMARK_TOKEN_BANDS: dict[str, tuple[int, int]] = {
    "wikitection": (128, 512),
    "newstection": (128, 512),
    "arxivtection": (1024, 2048),
}


def pool_path(benchmark: str) -> Path:
    return DATA_ROOT / benchmark / "pool.jsonl"


def build_nart_split(
    benchmark: str,
    path: Path,
    tokenizer: Any,
    n_per_class: int,
    n_aux: int,
    seed: int,
    min_tokens: int | None = None,
    max_tokens: int | None = None,
) -> tuple[list[SFTRecord], list[SFTRecord], list[SFTRecord], dict[str, Any]]:
    manifest_path = path.with_suffix(".manifest.json")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    raw_bytes = path.read_bytes()
    actual_sha = hashlib.sha256(raw_bytes).hexdigest()
    if actual_sha != manifest["jsonl_sha256"]:
        raise RuntimeError("NART pool hash does not match its manifest")
    if manifest.get("benchmark") != benchmark:
        raise RuntimeError(
            f"Pool manifest declares benchmark {manifest.get('benchmark')!r}, expected {benchmark!r}"
        )

    if min_tokens is None or max_tokens is None:
        default_min, default_max = BENCHMARK_TOKEN_BANDS[benchmark]
        min_tokens = default_min if min_tokens is None else min_tokens
        max_tokens = default_max if max_tokens is None else max_tokens

    documents = [json.loads(line) for line in raw_bytes.decode("utf-8").splitlines() if line]
    required = 2 * n_per_class + n_aux
    if len(documents) < required:
        raise RuntimeError(f"Need {required} NART documents, found {len(documents)}")

    random.Random(seed).shuffle(documents)
    required = 2 * n_per_class + n_aux
    records: list[SFTRecord] = []
    response_hashes: set[str] = set()
    dropped_band = 0
    skipped_token_duplicates = 0
    skipped_template_overlap = 0
    selected_gram_owners: dict[int, list[int]] = {}

    def _grams(record: SFTRecord) -> list[int]:
        tokens = list(record.prompt_ids or ()) + list(record.response_ids)
        return [
            hash(tuple(tokens[index : index + 13]))
            for index in range(max(0, len(tokens) - 12))
        ]

    for document in documents:
        if len(records) >= required:
            break
        text = str(document["text"])
        response_ids = list(
            tokenizer(
                text,
                add_special_tokens=False,
                truncation=True,
                max_length=max_tokens,
            ).input_ids
        )
        if len(response_ids) < min_tokens:
            dropped_band += 1
            continue
        prompt_text = NART_PROMPT.format(topic=str(document.get("title", "a public document")))
        prompt_ids = list(tokenizer(prompt_text, add_special_tokens=False).input_ids)
        response_hash = _hash_ids(response_ids)
        if response_hash in response_hashes:
            # syndicated copies often share the identical opening; after the
            # max-length truncation they are the same membership text
            skipped_token_duplicates += 1
            continue
        candidate = SFTRecord(
            record_id=f"nart:{benchmark}:{response_hash[:16]}",
            source=str(document.get("source", document.get("canonical_url", "nart"))),
            response_ids=tuple(response_ids),
            response_hash=response_hash,
            prompt_ids=tuple(prompt_ids),
            prompt_hash=_hash_ids(prompt_ids),
            prompt_text=prompt_text,
            topic=str(document.get("title", "")),
            source_char_count=int(document.get("source_char_count", len(text))),
            source_timestamp=str(document.get("creation_timestamp", "")),
            source_revision=int(document.get("snapshot_revision", 0)),
        )
        # Same-site template text (headers, menus, subscribe banners) can make
        # two pages share most of their truncated token prefix. Enforce the
        # 13-gram gate margin at selection time so the split passes by
        # construction; the final audit re-checks with the 0.80 threshold.
        grams = _grams(candidate)
        hits: dict[int, int] = {}
        for gram in grams:
            for owner in selected_gram_owners.get(gram, ()):
                hits[owner] = hits.get(owner, 0) + 1
        if records and hits and max(hits.values()) >= 0.5 * len(grams):
            skipped_template_overlap += 1
            continue
        for gram in set(grams):
            bucket = selected_gram_owners.setdefault(gram, [])
            if len(bucket) < 1000:
                bucket.append(len(records))
        response_hashes.add(response_hash)
        records.append(candidate)

    if len(records) < required:
        raise RuntimeError(
            f"Only {len(records)} documents survive the [{min_tokens}, {max_tokens}] token band "
            f"under this tokenizer; need {required}"
        )

    members = records[:n_per_class]
    nonmembers = records[n_per_class : 2 * n_per_class]
    auxiliary = records[2 * n_per_class :]
    ngram_audit = _cross_split_ngram_audit([members, nonmembers, auxiliary])
    metadata = {
        "benchmark": benchmark,
        "pool_path": str(path),
        "pool_sha256": actual_sha,
        "pool_records": len(documents),
        "creation_interval_inclusive": manifest.get("creation_interval_inclusive"),
        "timestamp_semantics": manifest.get("timestamp_semantics"),
        "license": manifest.get("license"),
        "provenance": manifest.get("provenance"),
        "token_band": {"min_tokens": min_tokens, "max_tokens": max_tokens},
        "band_dropped_documents": dropped_band,
        "skipped_token_duplicates": skipped_token_duplicates,
        "skipped_template_overlap": skipped_template_overlap,
        "split_seed": seed,
        "split_unit": f"{benchmark} document",
        "counts": {
            "member": len(members),
            "nonmember": len(nonmembers),
            "auxiliary": len(auxiliary),
        },
        "prompt_tokens": len(records[0].prompt_ids) if records else 0,
        "mean_response_tokens": (
            sum(len(r.response_ids) for r in records) / len(records) if records else 0.0
        ),
        "target_sft_uses": "member only",
        "exact_token_deduplication": True,
        "raw_text_persisted_in_pool": True,
        "prompt_mode": "nart_fixed_prompt_continuation",
        "cross_split_ngram_audit": ngram_audit,
    }
    return members, nonmembers, auxiliary, metadata
