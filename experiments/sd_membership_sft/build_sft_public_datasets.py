from __future__ import annotations

import argparse
import hashlib
import json
import random
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Iterator

import pyarrow.parquet as pq
from huggingface_hub import hf_hub_download, snapshot_download
from transformers import AutoTokenizer


@dataclass(frozen=True)
class DatasetSpec:
    name: str
    repo_id: str
    revision: str
    config: str
    files: tuple[str, ...]
    license: str
    text_field: str
    id_field: str
    title_field: str | None = None


DATASETS = {
    "wikitext103": DatasetSpec(
        name="WikiText-103 raw",
        repo_id="Salesforce/wikitext",
        revision="b08601e04326c79dfdd32d625aee71d232d685c3",
        config="wikitext-103-raw-v1",
        files=(
            "wikitext-103-raw-v1/train-00000-of-00002.parquet",
            "wikitext-103-raw-v1/train-00001-of-00002.parquet",
        ),
        license="CC-BY-SA-3.0 AND GFDL",
        text_field="text",
        id_field="article_index",
        title_field="title",
    ),
    "xsum": DatasetSpec(
        name="XSum",
        repo_id="EdinburghNLP/xsum",
        revision="7d4d486c2f8ef850b1a11aead99b894ff3dd7da9",
        config="default",
        files=("data/train-00000-of-00001.parquet",),
        license="UNKNOWN (dataset card)",
        text_field="document",
        id_field="id",
    ),
    "cnn_dailymail": DatasetSpec(
        name="CNN/DailyMail 3.0.0",
        repo_id="abisee/cnn_dailymail",
        revision="96df5e686bee6baa90b8bee7c28b81fa3fa6223d",
        config="3.0.0",
        files=(
            "3.0.0/train-00000-of-00003.parquet",
            "3.0.0/train-00001-of-00003.parquet",
            "3.0.0/train-00002-of-00003.parquet",
        ),
        license="Apache-2.0 (dataset card)",
        text_field="article",
        id_field="id",
    ),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build revision-pinned continuation-SFT membership pools."
    )
    parser.add_argument(
        "--dataset", choices=(*DATASETS, "all"), default="all"
    )
    parser.add_argument(
        "--output-dir", type=Path, default=Path("experiments/data/sft_public")
    )
    parser.add_argument("--tokenizer", default="Qwen/Qwen3-1.7B-Base")
    parser.add_argument(
        "--tokenizer-revision",
        default="ea980cb0a6c2ae4b936e82123acc929f1cec04c1",
    )
    parser.add_argument("--records", type=int, default=1800)
    parser.add_argument("--prompt-tokens", type=int, default=64)
    parser.add_argument("--response-tokens", type=int, default=128)
    parser.add_argument("--seed", type=int, default=20260828)
    return parser.parse_args()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _hash_ids(ids: list[int]) -> str:
    payload = b"".join(
        int(token_id).to_bytes(4, "little", signed=False) for token_id in ids
    )
    return hashlib.sha256(payload).hexdigest()


def _reservoir(
    rows: Iterable[dict[str, Any]], capacity: int, seed: int
) -> list[dict[str, Any]]:
    rng = random.Random(seed)
    selected: list[dict[str, Any]] = []
    for index, row in enumerate(rows):
        if len(selected) < capacity:
            selected.append(row)
            continue
        replacement = rng.randrange(index + 1)
        if replacement < capacity:
            selected[replacement] = row
    rng.shuffle(selected)
    return selected


def _iter_parquet_rows(
    paths: list[Path], columns: list[str]
) -> Iterator[dict[str, Any]]:
    for path in paths:
        parquet = pq.ParquetFile(path)
        for batch in parquet.iter_batches(batch_size=1024, columns=columns):
            values = batch.to_pydict()
            for row_index in range(batch.num_rows):
                yield {name: values[name][row_index] for name in columns}


_WIKI_TITLE = re.compile(r"^\s*=\s*([^=].*?)\s*=\s*$")


def _iter_wikitext_articles(paths: list[Path]) -> Iterator[dict[str, Any]]:
    title: str | None = None
    paragraphs: list[str] = []
    article_index = 0
    for row in _iter_parquet_rows(paths, ["text"]):
        text = str(row["text"] or "").strip()
        match = _WIKI_TITLE.match(text)
        if match:
            if title and paragraphs:
                yield {
                    "article_index": article_index,
                    "title": title,
                    "text": "\n\n".join(paragraphs),
                }
                article_index += 1
            title = match.group(1).strip()
            paragraphs = []
        elif text and title:
            paragraphs.append(text)
    if title and paragraphs:
        yield {
            "article_index": article_index,
            "title": title,
            "text": "\n\n".join(paragraphs),
        }


def _source_rows(spec: DatasetSpec, paths: list[Path]) -> Iterator[dict[str, Any]]:
    if spec.repo_id == "Salesforce/wikitext":
        yield from _iter_wikitext_articles(paths)
        return
    columns = [spec.text_field, spec.id_field]
    if spec.title_field:
        columns.append(spec.title_field)
    yield from _iter_parquet_rows(paths, columns)


def _make_record(
    spec: DatasetSpec,
    row: dict[str, Any],
    tokenizer: Any,
    prompt_tokens: int,
    response_tokens: int,
) -> dict[str, Any] | None:
    raw_text = str(row.get(spec.text_field, "") or "").strip()
    ids = list(tokenizer(raw_text, add_special_tokens=False).input_ids)
    needed = prompt_tokens + response_tokens
    if len(ids) < needed:
        return None
    prompt = ids[:prompt_tokens]
    response = ids[prompt_tokens:needed]
    source_id = str(row.get(spec.id_field, ""))
    record_hash = _hash_ids(prompt + response)
    return {
        "record_id": f"{spec.repo_id}:{source_id}:{record_hash[:16]}",
        "source": f"hf://datasets/{spec.repo_id}@{spec.revision}/{source_id}",
        "prompt_ids": prompt,
        "response_ids": response,
        "source_char_count": len(raw_text),
        "snapshot_revision": 0,
        "creation_timestamp": "",
        "content_token_sha256": record_hash,
    }


def build_dataset(
    spec: DatasetSpec,
    output_dir: Path,
    tokenizer: Any,
    tokenizer_id: str,
    tokenizer_revision: str,
    records: int,
    prompt_tokens: int,
    response_tokens: int,
    seed: int,
) -> tuple[Path, Path]:
    downloaded = [
        Path(
            hf_hub_download(
                repo_id=spec.repo_id,
                repo_type="dataset",
                filename=filename,
                revision=spec.revision,
            )
        )
        for filename in spec.files
    ]
    candidates = _reservoir(
        _source_rows(spec, downloaded), max(records * 4, records), seed
    )
    output_records: list[dict[str, Any]] = []
    seen: set[str] = set()
    for row in candidates:
        record = _make_record(
            spec, row, tokenizer, prompt_tokens, response_tokens
        )
        if record is None or record["content_token_sha256"] in seen:
            continue
        seen.add(record["content_token_sha256"])
        output_records.append(record)
        if len(output_records) == records:
            break
    if len(output_records) < records:
        raise RuntimeError(
            f"{spec.name}: requested {records} records, built {len(output_records)}"
        )

    output_dir.mkdir(parents=True, exist_ok=True)
    stem = spec.repo_id.split("/")[-1].replace("_", "-")
    jsonl_path = output_dir / f"{stem}-{spec.config}.jsonl"
    with jsonl_path.open("w", encoding="utf-8") as handle:
        for record in output_records:
            handle.write(json.dumps(record, ensure_ascii=False, separators=(",", ":")))
            handle.write("\n")
    manifest_path = jsonl_path.with_suffix(".manifest.json")
    manifest = {
        "schema": "sft_membership_continuation_pool_v1",
        "dataset": spec.name,
        "repo_id": spec.repo_id,
        "revision": spec.revision,
        "config": spec.config,
        "split": "train",
        "license": spec.license,
        "creation_interval_inclusive": None,
        "timestamp_semantics": "not supplied by the source dataset",
        "split_unit": "source document; at most one record per document",
        "task": "fixed-token causal continuation SFT",
        "tokenizer": tokenizer_id,
        "tokenizer_revision": tokenizer_revision,
        "prompt_tokens": prompt_tokens,
        "response_tokens": response_tokens,
        "records": len(output_records),
        "construction_seed": seed,
        "jsonl_sha256": _sha256(jsonl_path),
        "source_files": [
            {
                "filename": filename,
                "sha256": _sha256(path),
                "bytes": path.stat().st_size,
            }
            for filename, path in zip(spec.files, downloaded, strict=True)
        ],
        "raw_text_persisted": False,
        "persisted_content_fields": ["prompt_ids", "response_ids"],
    }
    manifest_path.write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    return jsonl_path, manifest_path


def main() -> None:
    args = parse_args()
    if args.records < 1 or args.prompt_tokens < 1 or args.response_tokens < 1:
        raise ValueError("records and token lengths must be positive")
    snapshot = snapshot_download(
        repo_id=args.tokenizer,
        revision=args.tokenizer_revision,
        local_files_only=True,
    )
    tokenizer = AutoTokenizer.from_pretrained(snapshot, local_files_only=True)
    selected = DATASETS.values() if args.dataset == "all" else [DATASETS[args.dataset]]
    for offset, spec in enumerate(selected):
        paths = build_dataset(
            spec,
            args.output_dir,
            tokenizer,
            args.tokenizer,
            args.tokenizer_revision,
            args.records,
            args.prompt_tokens,
            args.response_tokens,
            args.seed + offset,
        )
        print(json.dumps({"dataset": spec.name, "outputs": [str(path) for path in paths]}))


if __name__ == "__main__":
    main()
