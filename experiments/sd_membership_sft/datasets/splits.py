"""Load a frozen benchmark pool into the controlled three-class split.

The pool (see ``pools``) holds raw post-cutoff documents with provenance.
Loading re-tokenizes per target model, applies the per-pool token band
(128..512 for WikiTection/NewsTection, 1024..2048 for ArXivTection), deduplicates
on token IDs, and splits member/nonmember/auxiliary exactly as
``build_public_snapshot_split`` does, including the 13-gram cross-split gate.

Records use the fixed instruction prompt with the document as the continuation;
``make_sft_example`` masks the prompt so only document tokens contribute loss.
"""

from __future__ import annotations

import hashlib
import json
import os
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from experiments.sd_membership_sft.datasets.data import SFTRecord, _hash_ids
from experiments.sd_membership_sft.core.data_contract import DEFAULT_DATA_CONTRACT
from experiments.sd_membership_sft.datasets.pool_storage import read_verified_pool


def _token_ngrams(record: SFTRecord, n: int) -> set[tuple[int, ...]]:
    # The fixed instruction prompt is protocol scaffolding shared by every
    # record, not document content. Including it creates all-to-all matches and
    # can hide the overlap fraction that the gate is intended to measure.
    tokens = list(record.response_ids)
    return {tuple(tokens[index : index + n]) for index in range(len(tokens) - n + 1)}


def _cross_split_ngram_audit(
    splits: list[list[SFTRecord]], n: int = 13
) -> dict[str, Any]:
    owners: dict[tuple[int, ...], list[tuple[int, int]]] = {}
    sizes: dict[tuple[int, int], int] = {}
    record_grams: dict[tuple[int, int], set[tuple[int, ...]]] = {}
    for split_index, records in enumerate(splits):
        for record_index, record in enumerate(records):
            grams = _token_ngrams(record, n)
            owner = (split_index, record_index)
            sizes[owner] = len(grams)
            record_grams[owner] = grams
            for gram in grams:
                owners.setdefault(gram, []).append(owner)

    # Enumerating every owner pair for ubiquitous boilerplate is quadratic.
    # High-overlap document pairs necessarily share at least one rarer gram
    # unless >80% of a document consists solely of ubiquitous text. Track that
    # latter case as a conservative upper bound and enumerate only rare-gram
    # candidate pairs exactly.
    owner_cap = 64
    candidate_pairs: set[tuple[tuple[int, int], tuple[int, int]]] = set()
    ubiquitous_cross_grams = {owner: 0 for owner in sizes}
    shared_ngrams = 0
    for gram_owners in owners.values():
        split_ids = {owner[0] for owner in gram_owners}
        if len(split_ids) < 2:
            continue
        shared_ngrams += 1
        if len(gram_owners) > owner_cap:
            for owner in gram_owners:
                ubiquitous_cross_grams[owner] += 1
            continue
        for left_index, left in enumerate(gram_owners):
            for right in gram_owners[left_index + 1 :]:
                if left[0] == right[0]:
                    continue
                key = (left, right) if left < right else (right, left)
                candidate_pairs.add(key)

    maximum = 0.0
    maximum_pair: tuple[tuple[int, int], tuple[int, int]] | None = None
    for pair in candidate_pairs:
        count = len(record_grams[pair[0]].intersection(record_grams[pair[1]]))
        denominator = max(1, min(sizes[pair[0]], sizes[pair[1]]))
        fraction = count / denominator
        if fraction > maximum:
            maximum = fraction
            maximum_pair = pair
    unexamined_upper_bound = max(
        (
            ubiquitous_cross_grams[owner] / max(1, size)
            for owner, size in sizes.items()
        ),
        default=0.0,
    )
    conservative_maximum = max(maximum, unexamined_upper_bound)
    if conservative_maximum > 0.80:
        raise RuntimeError(
            "Cross-split 13-gram overlap exceeds or cannot be certified below "
            "the preregistered 80% threshold"
        )
    return {
        "n": n,
        "unique_ngrams": len(owners),
        "cross_split_shared_ngrams": shared_ngrams,
        "maximum_pair_overlap_fraction": conservative_maximum,
        "maximum_enumerated_pair_overlap_fraction": maximum,
        "maximum_pair_indices": (
            maximum_pair if maximum >= unexamined_upper_bound else None
        ),
        "unexamined_pair_overlap_upper_bound": unexamined_upper_bound,
        "ubiquitous_ngram_owner_cap": owner_cap,
        "threshold": 0.80,
        "gate": "PASS",
    }


SFT_PROMPT = (
    "You are a helpful assistant. Below is a given topic and related contexts. "
    "Please continue writing or analyze the contexts.\nTopic: {topic}\nContext: "
)

DATA_ROOT = Path("artifacts/data/pools")

BENCHMARK_TOKEN_BANDS: dict[str, tuple[int, int]] = {
    "wikitection": (128, 512),
    "newstection": (128, 512),
    "arxivtection": (1024, 2048),
}

SHARED_SPLIT_SCHEMA_VERSION = 2
CONTROLLED_SPLIT_SCHEMA_VERSION = 3
SHARED_SELECTION_OVERLAP_THRESHOLD = 0.50
_SELECTION_OWNER_CAP = 1000


@dataclass(frozen=True)
class ControlledSplit:
    """Four disjoint data roles used by the deployed accept-only audit.

    ``draft_auxiliary`` is the 2,000-record corpus used to tune the draft
    model. ``audit_auxiliary`` is a separate trusted-nonmember corpus used only
    to fit, select, and calibrate the small membership detector.
    """

    members: list[SFTRecord]
    nonmembers: list[SFTRecord]
    draft_auxiliary: list[SFTRecord]
    audit_auxiliary: list[SFTRecord]
    metadata: dict[str, Any]


def pool_path(benchmark: str) -> Path:
    return DATA_ROOT / benchmark / "pool.jsonl"


def _verified_pool(benchmark: str, path: Path) -> tuple[list[dict[str, Any]], dict[str, Any], str]:
    """Load a pool only after its frozen manifest and content hash agree."""
    documents, manifest, raw_bytes = read_verified_pool(path, benchmark)
    return documents, manifest, hashlib.sha256(raw_bytes).hexdigest()


def _document_identity(document: dict[str, Any]) -> tuple[str, str]:
    text = str(document["text"])
    text_sha = hashlib.sha256(text.encode("utf-8")).hexdigest()
    declared_sha = str(document.get("text_sha256", text_sha))
    if declared_sha != text_sha:
        raise RuntimeError(
            f"Document {document.get('record_id')!r} has an invalid text_sha256"
        )
    record_id = str(document.get("record_id", ""))
    if not record_id:
        raise RuntimeError("Every shared-split document must have a record_id")
    return record_id, text_sha


def _selection_gram_hashes(response_ids: list[int], n: int = 13) -> set[int]:
    tokens = response_ids
    return {
        hash(tuple(tokens[index : index + n]))
        for index in range(max(0, len(tokens) - n + 1))
    }


def _filter_tokenizer_near_duplicates(
    documents: list[dict[str, Any]],
    tokenizer: Any,
    *,
    min_tokens: int,
    max_tokens: int,
) -> tuple[list[dict[str, Any]], dict[str, int | float]]:
    """Greedily retain a deterministic near-duplicate-free document sequence.

    The caller supplies documents in seeded order.  Filtering is deliberately
    performed on the exact truncated response representation later used by the
    final 13-gram audit; raw full-document shingles cannot detect pages
    whose first ``max_tokens`` tokens are near-identical but whose tails differ.
    """
    kept: list[dict[str, Any]] = []
    response_hashes: set[str] = set()
    gram_sizes: list[int] = []
    gram_owners: dict[int, int | list[int]] = {}
    rejected_below_band = 0
    rejected_exact_token_duplicate = 0
    rejected_near_duplicate = 0

    for document in documents:
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
            rejected_below_band += 1
            continue
        response_hash = _hash_ids(response_ids)
        if response_hash in response_hashes:
            rejected_exact_token_duplicate += 1
            continue
        grams = _selection_gram_hashes(response_ids)
        hits: dict[int, int] = {}
        for gram in grams:
            owners = gram_owners.get(gram)
            if owners is None:
                continue
            if isinstance(owners, int):
                hits[owners] = hits.get(owners, 0) + 1
            else:
                for owner in owners:
                    hits[owner] = hits.get(owner, 0) + 1
        if any(
            count
            >= SHARED_SELECTION_OVERLAP_THRESHOLD
            * min(len(grams), gram_sizes[owner])
            for owner, count in hits.items()
        ):
            rejected_near_duplicate += 1
            continue

        owner = len(kept)
        for gram in grams:
            owners = gram_owners.get(gram)
            if owners is None:
                gram_owners[gram] = owner
            elif isinstance(owners, int):
                if owners != owner:
                    gram_owners[gram] = [owners, owner]
            elif len(owners) < _SELECTION_OWNER_CAP:
                owners.append(owner)
        response_hashes.add(response_hash)
        gram_sizes.append(len(grams))
        kept.append(document)

    return kept, {
        "input_documents": len(documents),
        "kept_documents": len(kept),
        "rejected_below_band": rejected_below_band,
        "rejected_exact_token_duplicate": rejected_exact_token_duplicate,
        "rejected_near_duplicate": rejected_near_duplicate,
        "near_duplicate_ngram": 13,
        "near_duplicate_threshold": SHARED_SELECTION_OVERLAP_THRESHOLD,
    }


def prepare_shared_split_manifest(
    benchmark: str,
    path: Path,
    tokenizers: dict[str, Any],
    n_per_class: int,
    n_aux: int,
    seed: int,
    output_path: Path,
    min_tokens: int | None = None,
    max_tokens: int | None = None,
    eligibility_cache: dict[tuple[Any, ...], dict[str, Any]] | None = None,
    n_audit_aux: int = 0,
) -> dict[str, Any]:
    """Freeze one raw document assignment that is valid for every tokenizer.

    Candidate documents are filtered in seeded order for each tokenizer before
    the first ``required`` shared IDs are frozen.  Rejected candidates are
    therefore deterministically backfilled from the same frozen pool; no model
    receives a tokenizer-specific substitute after the manifest is written.
    """
    if not tokenizers:
        raise ValueError("At least one tokenizer is required")
    if min_tokens is None or max_tokens is None:
        default_min, default_max = BENCHMARK_TOKEN_BANDS[benchmark]
        min_tokens = default_min if min_tokens is None else min_tokens
        max_tokens = default_max if max_tokens is None else max_tokens
    cache_key = (
        str(path.resolve()),
        benchmark,
        min_tokens,
        max_tokens,
        tuple(sorted(tokenizers)),
    )
    cached = eligibility_cache.get(cache_key) if eligibility_cache is not None else None
    if cached is None:
        documents, pool_manifest, pool_sha = _verified_pool(benchmark, path)
        eligible: list[dict[str, Any]] = []
        seen_ids: set[str] = set()
        seen_text: set[str] = set()
        skipped_duplicate_raw_text = 0
        dropped_by_tokenizer = {name: 0 for name in tokenizers}
        for document in documents:
            record_id, text_sha = _document_identity(document)
            if record_id in seen_ids:
                raise RuntimeError(f"Duplicate pool record_id: {record_id}")
            seen_ids.add(record_id)
            if text_sha in seen_text:
                skipped_duplicate_raw_text += 1
                continue
            text = str(document["text"])
            valid = True
            for name, tokenizer in tokenizers.items():
                response_ids = list(
                    tokenizer(
                        text,
                        add_special_tokens=False,
                        truncation=True,
                        max_length=max_tokens,
                    ).input_ids
                )
                if len(response_ids) < min_tokens:
                    dropped_by_tokenizer[name] += 1
                    valid = False
            if not valid:
                continue
            seen_text.add(text_sha)
            eligible.append(document)
        cached = {
            "documents": documents,
            "pool_manifest": pool_manifest,
            "pool_sha": pool_sha,
            "eligible": eligible,
            "dropped_by_tokenizer": dropped_by_tokenizer,
            "skipped_duplicate_raw_text": skipped_duplicate_raw_text,
        }
        if eligibility_cache is not None:
            eligibility_cache[cache_key] = cached
    documents = cached["documents"]
    pool_manifest = cached["pool_manifest"]
    pool_sha = cached["pool_sha"]
    eligible = list(cached["eligible"])
    dropped_by_tokenizer = cached["dropped_by_tokenizer"]
    skipped_duplicate_raw_text = cached["skipped_duplicate_raw_text"]

    if n_audit_aux < 0:
        raise ValueError("n_audit_aux must be nonnegative")
    required = 2 * n_per_class + n_aux + n_audit_aux
    if len(eligible) < required:
        raise RuntimeError(
            f"Only {len(eligible)} documents satisfy every tokenizer's "
            f"[{min_tokens}, {max_tokens}] token band; need {required}"
        )
    random.Random(seed).shuffle(eligible)
    candidates = eligible
    tokenizer_selection: dict[str, dict[str, int | float]] = {}
    for source in sorted(tokenizers):
        candidates, stats = _filter_tokenizer_near_duplicates(
            candidates,
            tokenizers[source],
            min_tokens=min_tokens,
            max_tokens=max_tokens,
        )
        tokenizer_selection[source] = stats
        if len(candidates) < required:
            raise RuntimeError(
                f"Only {len(candidates)} documents survive shared exact-token and "
                f"13-gram filtering through tokenizer {source!r}; need {required}"
            )
    selected = candidates[:required]

    def entries(rows: list[dict[str, Any]]) -> list[dict[str, str]]:
        output: list[dict[str, str]] = []
        for row in rows:
            record_id, text_sha = _document_identity(row)
            output.append({"record_id": record_id, "text_sha256": text_sha})
        return output

    split_rows = {
        "member": entries(selected[:n_per_class]),
        "nonmember": entries(selected[n_per_class : 2 * n_per_class]),
        "auxiliary": entries(
            selected[2 * n_per_class : 2 * n_per_class + n_aux]
        ),
    }
    counts = {
        "member": n_per_class,
        "nonmember": n_per_class,
        "auxiliary": n_aux,
    }
    if n_audit_aux:
        counts["audit_auxiliary"] = n_audit_aux
        split_rows["audit_auxiliary"] = entries(selected[-n_audit_aux:])

    artifact = {
        "schema_version": (
            CONTROLLED_SPLIT_SCHEMA_VERSION
            if n_audit_aux
            else SHARED_SPLIT_SCHEMA_VERSION
        ),
        "benchmark": benchmark,
        "pool_path": str(path),
        "pool_sha256": pool_sha,
        "pool_records": len(documents),
        "seed": seed,
        "token_band": {"min_tokens": min_tokens, "max_tokens": max_tokens},
        "tokenizer_sources": sorted(tokenizers),
        "common_eligible_documents": len(eligible),
        "dropped_by_tokenizer": dropped_by_tokenizer,
        "selection": {
            "input_documents": len(eligible),
            "survivors_after_all_tokenizers": len(candidates),
            "selected_documents": len(selected),
            "skipped_duplicate_raw_text": skipped_duplicate_raw_text,
            "rejected_exact_token_duplicate": sum(
                int(stats["rejected_exact_token_duplicate"])
                for stats in tokenizer_selection.values()
            ),
            "rejected_near_duplicate": sum(
                int(stats["rejected_near_duplicate"])
                for stats in tokenizer_selection.values()
            ),
            "near_duplicate_ngram": 13,
            "near_duplicate_threshold": SHARED_SELECTION_OVERLAP_THRESHOLD,
            "per_tokenizer": tokenizer_selection,
        },
        "counts": counts,
        "splits": split_rows,
        "pool_provenance": {
            "creation_interval_inclusive": pool_manifest.get(
                "creation_interval_inclusive"
            ),
            "timestamp_semantics": pool_manifest.get("timestamp_semantics"),
        },
    }
    if output_path.exists():
        current = json.loads(output_path.read_text(encoding="utf-8"))
        if current == artifact:
            return current
        audit_path = output_path.with_suffix(".audit.json")
        if audit_path.exists():
            raise RuntimeError(
                "Refusing to replace a different audited shared split manifest: "
                f"{output_path}"
            )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_path.with_name(f".{output_path.name}.tmp.{os.getpid()}")
    temporary.write_text(
        json.dumps(artifact, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, output_path)
    return artifact


def build_split_from_shared_manifest(
    benchmark: str,
    path: Path,
    tokenizer: Any,
    shared_manifest_path: Path,
    tokenizer_source: str,
    *,
    include_audit_auxiliary: bool = False,
) -> tuple[list[SFTRecord], list[SFTRecord], list[SFTRecord], dict[str, Any]] | ControlledSplit:
    """Tokenize an immutable shared raw split and fail closed on every audit."""
    documents, pool_manifest, pool_sha = _verified_pool(benchmark, path)
    manifest_bytes = shared_manifest_path.read_bytes()
    shared = json.loads(manifest_bytes)
    schema_version = shared.get("schema_version")
    if schema_version not in {
        SHARED_SPLIT_SCHEMA_VERSION,
        CONTROLLED_SPLIT_SCHEMA_VERSION,
    }:
        raise RuntimeError(f"Unsupported shared split schema: {shared_manifest_path}")
    if shared.get("benchmark") != benchmark or shared.get("pool_sha256") != pool_sha:
        raise RuntimeError("Shared split does not match the requested frozen pool")
    if tokenizer_source not in shared.get("tokenizer_sources", []):
        raise RuntimeError(
            f"Tokenizer {tokenizer_source!r} was not audited by the shared split"
        )

    by_id: dict[str, dict[str, Any]] = {}
    for document in documents:
        record_id, _text_sha = _document_identity(document)
        if record_id in by_id:
            raise RuntimeError(f"Duplicate pool record_id: {record_id}")
        by_id[record_id] = document

    token_band = shared["token_band"]
    min_tokens = int(token_band["min_tokens"])
    max_tokens = int(token_band["max_tokens"])
    response_owners: dict[str, str] = {}
    raw_text_owners: dict[str, str] = {}
    selected_record_ids: set[str] = set()

    def convert(entry: dict[str, str]) -> SFTRecord:
        record_id = str(entry["record_id"])
        if record_id in selected_record_ids:
            raise RuntimeError(f"Shared split repeats document ID: {record_id}")
        selected_record_ids.add(record_id)
        if record_id not in by_id:
            raise RuntimeError(f"Shared split document is absent from pool: {record_id}")
        document = by_id[record_id]
        actual_id, text_sha = _document_identity(document)
        if actual_id != record_id or text_sha != entry["text_sha256"]:
            raise RuntimeError(f"Shared split identity mismatch for {record_id}")
        previous_raw = raw_text_owners.setdefault(text_sha, record_id)
        if previous_raw != record_id:
            raise RuntimeError(
                f"Shared split contains duplicate raw text: {previous_raw}, {record_id}"
            )
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
            raise RuntimeError(
                f"Shared document {record_id} falls below this tokenizer's "
                f"{min_tokens}-token minimum; the manifest will not substitute it"
            )
        response_hash = _hash_ids(response_ids)
        previous_response = response_owners.setdefault(response_hash, record_id)
        if previous_response != record_id:
            raise RuntimeError(
                "Tokenizer produces duplicate response text for shared documents "
                f"{previous_response} and {record_id}"
            )
        prompt_text = SFT_PROMPT.format(
            topic=str(document.get("title", "a public document"))
        )
        prompt_ids = list(tokenizer(prompt_text, add_special_tokens=False).input_ids)
        return SFTRecord(
            record_id=record_id,
            source=str(document.get("source", document.get("canonical_url", "pool"))),
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

    has_audit_auxiliary = "audit_auxiliary" in shared.get("splits", {})
    split_names = (
        "member",
        "nonmember",
        "auxiliary",
        *(("audit_auxiliary",) if has_audit_auxiliary else ()),
    )
    converted = [
        [convert(entry) for entry in shared["splits"][name]] for name in split_names
    ]
    expected_counts = shared["counts"]
    for name, records in zip(split_names, converted):
        if len(records) != int(expected_counts[name]):
            raise RuntimeError(f"Shared split count mismatch for {name}")
    ngram_audit = _cross_split_ngram_audit(converted)
    members, nonmembers, auxiliary = converted[:3]
    metadata = {
        "shared_split_schema_version": schema_version,
        "benchmark": benchmark,
        "pool_path": str(path),
        "pool_sha256": pool_sha,
        "pool_records": len(documents),
        "creation_interval_inclusive": pool_manifest.get(
            "creation_interval_inclusive"
        ),
        "timestamp_semantics": pool_manifest.get("timestamp_semantics"),
        "license": pool_manifest.get("license"),
        "provenance": pool_manifest.get("provenance"),
        "token_band": token_band,
        "split_seed": int(shared["seed"]),
        "split_unit": f"{benchmark} raw document ID",
        "counts": {name: len(rows) for name, rows in zip(split_names, converted)},
        "shared_split_manifest": str(shared_manifest_path),
        "shared_split_sha256": hashlib.sha256(manifest_bytes).hexdigest(),
        "tokenizer_source": tokenizer_source,
        "exact_raw_text_deduplication": True,
        "exact_token_deduplication": True,
        "raw_document_assignment_shared": True,
        "target_sft_uses": "member only",
        "prompt_mode": "fixed_prompt_continuation",
        "cross_split_ngram_audit": ngram_audit,
    }
    if include_audit_auxiliary:
        if not has_audit_auxiliary:
            raise RuntimeError(
                "Shared split has no independent audit_auxiliary role"
            )
        return ControlledSplit(
            members=members,
            nonmembers=nonmembers,
            draft_auxiliary=auxiliary,
            audit_auxiliary=converted[3],
            metadata=metadata,
        )
    return members, nonmembers, auxiliary, metadata


def build_controlled_split_from_shared_manifest(
    benchmark: str,
    path: Path,
    tokenizer: Any,
    shared_manifest_path: Path,
    tokenizer_source: str,
) -> ControlledSplit:
    """Load a four-role shared manifest with an independent audit corpus."""
    result = build_split_from_shared_manifest(
        benchmark,
        path,
        tokenizer,
        shared_manifest_path,
        tokenizer_source,
        include_audit_auxiliary=True,
    )
    assert isinstance(result, ControlledSplit)
    return result


def build_split(
    benchmark: str,
    path: Path,
    tokenizer: Any,
    n_per_class: int,
    n_aux: int,
    seed: int,
    min_tokens: int | None = None,
    max_tokens: int | None = None,
) -> tuple[list[SFTRecord], list[SFTRecord], list[SFTRecord], dict[str, Any]]:
    documents, manifest, actual_sha = _verified_pool(benchmark, path)

    if min_tokens is None or max_tokens is None:
        default_min, default_max = BENCHMARK_TOKEN_BANDS[benchmark]
        min_tokens = default_min if min_tokens is None else min_tokens
        max_tokens = default_max if max_tokens is None else max_tokens

    required = 2 * n_per_class + n_aux
    if len(documents) < required:
        raise RuntimeError(f"Need {required} pool documents, found {len(documents)}")

    random.Random(seed).shuffle(documents)
    required = 2 * n_per_class + n_aux
    records: list[SFTRecord] = []
    response_hashes: set[str] = set()
    dropped_band = 0
    skipped_token_duplicates = 0
    skipped_template_overlap = 0
    selected_gram_owners: dict[int, list[int]] = {}

    def _grams(record: SFTRecord) -> list[int]:
        tokens = list(record.response_ids)
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
        prompt_text = SFT_PROMPT.format(topic=str(document.get("title", "a public document")))
        prompt_ids = list(tokenizer(prompt_text, add_special_tokens=False).input_ids)
        response_hash = _hash_ids(response_ids)
        if response_hash in response_hashes:
            # syndicated copies often share the identical opening; after the
            # max-length truncation they are the same membership text
            skipped_token_duplicates += 1
            continue
        candidate = SFTRecord(
            record_id=f"sft:{benchmark}:{response_hash[:16]}",
            source=str(document.get("source", document.get("canonical_url", "pool"))),
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
        "prompt_mode": "fixed_prompt_continuation",
        "cross_split_ngram_audit": ngram_audit,
    }
    return members, nonmembers, auxiliary, metadata


def build_controlled_split(
    benchmark: str,
    path: Path,
    tokenizer: Any,
    n_per_class: int = DEFAULT_DATA_CONTRACT.members,
    n_draft_aux: int = DEFAULT_DATA_CONTRACT.draft_auxiliary,
    n_audit_aux: int = DEFAULT_DATA_CONTRACT.audit_auxiliary,
    seed: int = 20260824,
    min_tokens: int | None = None,
    max_tokens: int | None = None,
) -> ControlledSplit:
    """Build the four mutually exclusive roles for the accept-only study.

    Selection is performed once over the entire frozen pool after the token
    band, exact-token, and 13-gram gates. The final auxiliary block is then
    divided into draft-model data and trusted nonmember audit data, so no audit
    record is borrowed from the 2,000 evaluation nonmembers.
    """
    if n_draft_aux <= 0 or n_audit_aux <= 0:
        raise ValueError("both auxiliary roles must contain records")
    members, nonmembers, combined_auxiliary, metadata = build_split(
        benchmark,
        path,
        tokenizer,
        n_per_class=n_per_class,
        n_aux=n_draft_aux + n_audit_aux,
        seed=seed,
        min_tokens=min_tokens,
        max_tokens=max_tokens,
    )
    draft_auxiliary = combined_auxiliary[:n_draft_aux]
    audit_auxiliary = combined_auxiliary[n_draft_aux:]
    ngram_audit = _cross_split_ngram_audit(
        [members, nonmembers, draft_auxiliary, audit_auxiliary]
    )
    metadata = {
        **metadata,
        "split_contract": "controlled_four_role_v1",
        "counts": {
            "member": len(members),
            "nonmember": len(nonmembers),
            "draft_auxiliary": len(draft_auxiliary),
            "audit_auxiliary": len(audit_auxiliary),
        },
        "cross_split_ngram_audit": ngram_audit,
    }
    return ControlledSplit(
        members=members,
        nonmembers=nonmembers,
        draft_auxiliary=draft_auxiliary,
        audit_auxiliary=audit_auxiliary,
        metadata=metadata,
    )
