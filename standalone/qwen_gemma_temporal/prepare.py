"""Prepare one shared Wiki selection for two frozen, independent draft pairs.

Only this standalone module writes the new batch. It reads the completed
MediaWiki pool and frozen recent SFT IDs; it never changes either source.
"""
from __future__ import annotations

from collections import Counter
import hashlib
import json
from pathlib import Path
import shutil
import tempfile

import numpy as np
from transformers import AutoTokenizer

from experiments.paths import ROOT
from experiments.pretraining.data import TOKEN_CONTRACT, sha256, tokenizer_hash
from experiments.shared.models.registry import MODEL_PAIRS
from experiments.shared.data.data import _hash_ids
from experiments.shared.data.pools import NearDuplicateIndex
from standalone.qwen_temporal_clean.prepare import clean_text, match_lengths
from standalone.qwen_temporal_mediawiki.prepare import (
    HISTORICAL_POOL, RECENT_POOL, SPLITS, inspect_sources,
)

VERSION = "qwen_gemma_temporal_v1"
PAIRS = ("qwen3", "gemma4")
SEEDS = (1919, 1949, 1978)
VARIANTS = ("clean_only", "length_matched")
DATA_ROOT = ROOT / "artifacts/data" / VERSION
MAX_TOKENS = 512
N_MEMBER = 2000
N_AUXILIARY = 600


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def write_json(path: Path, value: dict) -> None:
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n")


def _source_entry(path: Path) -> dict:
    path = Path(path).resolve()
    return {"path": str(path), "sha256": sha256(path)}


def _request(source_files: list[dict], seed: int) -> dict:
    extra = [Path(__file__), Path(clean_text.__code__.co_filename),
             ROOT / "experiments/shared/models/model_pairs.json"]
    return dict(schema=VERSION, seed=seed, pairs=list(PAIRS), variants=list(VARIANTS),
                selection="seeded ties among highest shared cleaned-token capacities; "
                          "same raw IDs for both model pairs",
                counts=dict(member=N_MEMBER, nonmember=N_MEMBER, auxiliary=N_AUXILIARY),
                max_tokens=MAX_TOKENS, sources=source_files + [_source_entry(p) for p in extra])


def _tokenizer(pair: str):
    spec = MODEL_PAIRS[pair]
    require(spec.adapter == "plain", f"{pair} requires an independent plain draft")
    target = AutoTokenizer.from_pretrained(spec.target, revision=spec.target_revision,
                                           local_files_only=True)
    draft = AutoTokenizer.from_pretrained(spec.draft, revision=spec.draft_revision,
                                          local_files_only=True)
    digest = tokenizer_hash(target)
    require(digest == tokenizer_hash(draft), f"{pair} target/draft tokenizer mismatch")
    require(set(target.get_vocab().values()) == set(range(len(target))),
            f"{pair} tokenizer IDs are not contiguous")
    return target, digest


def _tokenize(tokenizer, texts: list[str]) -> list[list[int]]:
    # Batches bound tokenizer memory while keeping the same truncation rule.
    result = []
    for start in range(0, len(texts), 128):
        result.extend(tokenizer(texts[start:start + 128], add_special_tokens=False,
                                truncation=True, max_length=MAX_TOKENS).input_ids)
    require(all(len(ids) >= 2 for ids in result), "cleaning produced a one-token document")
    return result


def _selection(history: list[dict], fixed: list[tuple[str, dict]], seed: int,
               tokenizers: dict, history_ids: dict, fixed_ids: dict) -> tuple[list[int], dict]:
    lengths = {pair: np.asarray([len(ids) for ids in history_ids[pair]]) for pair in PAIRS}
    score = np.minimum.reduce([lengths[pair] for pair in PAIRS])
    rng = np.random.default_rng(seed)
    ties = rng.permutation(len(history))
    ranked = ties[np.argsort(-score[ties], kind="stable")]
    seen_ids = {row["record_id"] for _, row in fixed}
    seen_text = {hashlib.sha256(clean_text(row["text"]).encode()).hexdigest()
                 for _, row in fixed}
    seen_tokens = {pair: {_hash_ids(ids) for ids in fixed_ids[pair]} for pair in PAIRS}
    near = {pair: NearDuplicateIndex() for pair in PAIRS}
    for pair in PAIRS:
        for ids in fixed_ids[pair]:
            near[pair].add(tokenizers[pair].decode(ids, skip_special_tokens=False))
    selected, drops = [], Counter()
    for index in ranked:
        index = int(index)
        row = history[index]
        visible = clean_text(row["text"])
        digest = hashlib.sha256(visible.encode()).hexdigest()
        if row["record_id"] in seen_ids or digest in seen_text:
            drops["exact"] += 1
            continue
        token_digests = {pair: _hash_ids(history_ids[pair][index]) for pair in PAIRS}
        if any(token_digests[pair] in seen_tokens[pair] for pair in PAIRS):
            drops["token"] += 1
            continue
        decoded = {pair: tokenizers[pair].decode(history_ids[pair][index],
                                                skip_special_tokens=False) for pair in PAIRS}
        if any(near[pair].is_duplicate(decoded[pair]) for pair in PAIRS):
            drops["near"] += 1
            continue
        selected.append(index)
        seen_ids.add(row["record_id"])
        seen_text.add(digest)
        for pair in PAIRS:
            seen_tokens[pair].add(token_digests[pair])
            near[pair].add(decoded[pair])
        if len(selected) == N_MEMBER:
            break
    require(len(selected) == N_MEMBER,
            f"only {len(selected)}/{N_MEMBER} nonduplicate historical articles")
    for pair in PAIRS:
        capacity = np.sort(lengths[pair][selected])
        desired = np.sort([len(ids) for ids, (group, _) in zip(fixed_ids[pair], fixed)
                           if group == "nonmember"])
        require(np.all(capacity >= desired),
                f"{pair} cannot match cleaned recent lengths with shared historical selection")
    return selected, dict(drops=drops, minimum_shared_capacity=int(score[selected].min()),
                          source_candidates=len(history))


def _record(group: str, row: dict, text: str, ids: list[int]) -> dict:
    return dict(record_id=f'temporal:{row["record_id"]}', source=row["source"],
                source_record_id=row["record_id"], group=group, label=int(group == "member"),
                text=text, token_ids=ids, token_hash=_hash_ids(ids),
                temporal_metadata={key: row[key] for key in
                                   ("title", "creation_timestamp", "snapshot_timestamp",
                                    "snapshot_revision") if key in row})


def _manifest(pair: str, variant: str, seed: int, records: list[dict],
              tokenizer_digest: str, request: dict, selection: dict) -> dict:
    spec = MODEL_PAIRS[pair]
    lengths = {group: [len(r["token_ids"]) for r in records if r["group"] == group]
               for group in ("member", "nonmember", "auxiliary")}
    return dict(kind="temporal_pretraining_v1", benchmark=f"wiki_temporal/{VERSION}/{pair}/{variant}",
                models=dict(target=dict(repo_id=spec.target, revision=spec.target_revision),
                            draft=dict(repo_id=spec.draft, revision=spec.draft_revision)),
                token_contract=TOKEN_CONTRACT, tokenizer_sha256=tokenizer_digest,
                counts=dict(member=N_MEMBER, nonmember=N_MEMBER, auxiliary=N_AUXILIARY),
                selection_seed=seed, min_tokens=min(map(min, lengths.values())),
                max_tokens=MAX_TOKENS, records_file="records.jsonl", membership_verified=False,
                label_provenance="historical presumed-member / post-weight nonmember temporal proxies",
                draft_exposure="frozen public pretrained draft; its historical membership is unknown",
                source_provenance=dict(extraction="same MediaWiki rendered-page extractor for both groups",
                                       historical_snapshot="2023-12-31", recent_role="frozen WikiTection nonmember/audit_auxiliary IDs",
                                       request=request, selection=selection),
                filtering=dict(text_normalization="qwen_temporal_clean_v2.clean_text for all roles",
                               historical_selection="joint cleaned-token capacity, seeded ties, cross-model dedup",
                               length_policy=variant),
                token_lengths={group: dict(min=min(values), max=max(values),
                                           mean=float(np.mean(values)))
                               for group, values in lengths.items()},
                caveats=["2023 dates do not prove actual pretraining membership",
                         "a 2026 page may copy older text", "length matching does not equalize topics"])


def validate_seed(folder: Path, expected_request: dict | None = None) -> dict:
    folder = Path(folder)
    request = json.loads((folder / "REQUEST.json").read_text())
    if expected_request is not None:
        require(request == expected_request, "temporal source or preparation code changed; use a new data root")
    for source in request["sources"]:
        require(sha256(Path(source["path"])) == source["sha256"],
                f'temporal source changed: {source["path"]}')
    complete = json.loads((folder / "COMPLETE.json").read_text())
    for relative, digest in complete.items():
        require(sha256(folder / relative) == digest, f"temporal output changed: {relative}")
    selection = json.loads((folder / "SELECTION.json").read_text())
    require(selection["seed"] == request["seed"] and
            len(selection["historical_ids"]) == N_MEMBER and
            len(selection["recent_ids"]) == N_MEMBER and
            len(selection["auxiliary_ids"]) == N_AUXILIARY,
            "temporal selection counts or seed changed")
    expected_ids = {
        "member": selection["historical_ids"],
        "nonmember": selection["recent_ids"],
        "auxiliary": selection["auxiliary_ids"],
    }
    require(len(set(sum(expected_ids.values(), []))) == 2 * N_MEMBER + N_AUXILIARY,
            "temporal selection IDs overlap")
    for pair in PAIRS:
        spec = MODEL_PAIRS[pair]
        rows_by_variant = {}
        for variant in VARIANTS:
            manifest = json.loads((folder / pair / variant / "manifest.json").read_text())
            require(manifest["kind"] == "temporal_pretraining_v1" and
                    manifest["benchmark"] == f"wiki_temporal/{VERSION}/{pair}/{variant}" and
                    manifest["selection_seed"] == request["seed"] and
                    manifest["token_contract"] == TOKEN_CONTRACT and
                    manifest["counts"] == request["counts"] and
                    manifest["models"] == {
                        "target": {"repo_id": spec.target, "revision": spec.target_revision},
                        "draft": {"repo_id": spec.draft, "revision": spec.draft_revision},
                    } and manifest["membership_verified"] is False,
                    "temporal manifest contract changed")
            require(sha256(folder / pair / variant / manifest["records_file"]) ==
                    manifest["records_sha256"], "temporal record checksum mismatch")
            rows = [json.loads(line) for line in
                    (folder / pair / variant / manifest["records_file"]).read_text().splitlines()]
            require(len(rows) == 2 * N_MEMBER + N_AUXILIARY,
                    "temporal record count changed")
            for group in expected_ids:
                selected_rows = [row for row in rows if row["group"] == group]
                require([row["source_record_id"] for row in selected_rows] == expected_ids[group],
                        f"{pair}/{variant} source ID or order changed for {group}")
            require(all(row["record_id"] == f'temporal:{row["source_record_id"]}' and
                        row["label"] == int(row["group"] == "member") and
                        2 <= len(row["token_ids"]) <= MAX_TOKENS and
                        row["token_hash"] == _hash_ids(row["token_ids"]) for row in rows),
                    f"{pair}/{variant} record contract changed")
            require(len({row["token_hash"] for row in rows}) == len(rows),
                    f"{pair}/{variant} has duplicate token sequences")
            rows_by_variant[variant] = rows
        clean_rows, matched_rows = (rows_by_variant[name] for name in VARIANTS)
        require(all(clean["token_ids"][:len(matched["token_ids"])] == matched["token_ids"]
                    for clean, matched in zip(clean_rows, matched_rows)),
                f"{pair} length matching did not preserve cleaned prefixes")
        require(sorted(len(row["token_ids"]) for row in matched_rows if row["group"] == "member") ==
                sorted(len(row["token_ids"]) for row in matched_rows if row["group"] == "nonmember"),
                f"{pair} member/nonmember token-length histograms differ")
    return request


def prepare_seed(seed: int, data_root: Path = DATA_ROOT, *, historical_pool: Path = HISTORICAL_POOL,
                 recent_pool: Path = RECENT_POOL, split_root: Path = SPLITS) -> Path:
    require(seed in SEEDS, "unsupported temporal seed")
    data_root = Path(data_root).resolve()
    history, fixed, _, sources = inspect_sources(
        historical_pool, recent_pool, Path(split_root) / f"seed{seed}.json", seed,
        n_per_class=N_MEMBER, n_aux=N_AUXILIARY)
    request = _request(sources, seed)
    folder = data_root / f"seed{seed}"
    if folder.exists():
        validate_seed(folder, request)
        print(f"seed{seed}: reused {folder}", flush=True)
        return folder
    tokenizers, fingerprints, history_ids, fixed_ids = {}, {}, {}, {}
    history_texts = [clean_text(row["text"]) for row in history]
    fixed_texts = [clean_text(row["text"]) for _, row in fixed]
    require(all(clean_text(text) == text for text in history_texts + fixed_texts),
            "text cleaner is not idempotent")
    for pair in PAIRS:
        tokenizers[pair], fingerprints[pair] = _tokenizer(pair)
        history_ids[pair] = _tokenize(tokenizers[pair], history_texts)
        fixed_ids[pair] = _tokenize(tokenizers[pair], fixed_texts)
    selected, selection = _selection(history, fixed, seed, tokenizers, history_ids, fixed_ids)
    data_root.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=f".seed{seed}-", dir=data_root))
    try:
        write_json(temporary / "REQUEST.json", request)
        write_json(temporary / "SELECTION.json", dict(seed=seed, historical_ids=[history[i]["record_id"] for i in selected],
                                                      recent_ids=[row["record_id"] for group, row in fixed if group == "nonmember"],
                                                      auxiliary_ids=[row["record_id"] for group, row in fixed if group == "auxiliary"],
                                                      diagnostics=selection))
        for pair in PAIRS:
            members = [_record("member", history[i], history_texts[i], history_ids[pair][i])
                       for i in selected]
            recent = [_record(group, row, text, ids)
                      for (group, row), text, ids in zip(fixed, fixed_texts, fixed_ids[pair])]
            records = members + recent
            visible = match_lengths([len(r["token_ids"]) for r in members],
                                    [len(r["token_ids"]) for r in recent if r["group"] == "nonmember"], seed)
            matched = []
            for row, limit in zip(records, visible + [len(r["token_ids"]) for r in recent]):
                ids = row["token_ids"][:limit]
                matched.append({**row, "token_ids": ids, "token_hash": _hash_ids(ids)})
            for variant, rows in (("clean_only", records), ("length_matched", matched)):
                location = temporary / pair / variant
                location.mkdir(parents=True)
                records_file = location / "records.jsonl"
                records_file.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows))
                manifest = _manifest(pair, variant, seed, rows, fingerprints[pair], request, selection)
                manifest["records_sha256"] = sha256(records_file)
                write_json(location / "manifest.json", manifest)
        files = [path for path in temporary.rglob("*") if path.is_file()]
        write_json(temporary / "COMPLETE.json",
                   {str(path.relative_to(temporary)): sha256(path) for path in files})
        temporary.rename(folder)
    finally:
        if temporary.exists():
            shutil.rmtree(temporary)
    print(f"seed{seed}: prepared common historical IDs for {', '.join(PAIRS)}", flush=True)
    return folder
