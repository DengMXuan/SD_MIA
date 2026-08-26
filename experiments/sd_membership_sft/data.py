from __future__ import annotations

import hashlib
import random
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from pypdf import PdfReader


@dataclass(frozen=True)
class SFTRecord:
    """One controlled SFT response; raw response text is kept in memory only."""

    record_id: str
    source: str
    response_ids: tuple[int, ...]
    response_hash: str

    @property
    def prompt(self) -> str:
        title = Path(self.source).stem.replace("_", " ").replace("-", " ")
        return (
            f"请以严谨的技术文档风格，介绍论文《{title}》中的一个连续技术段落。"
            "只输出正文，不要添加标题、引文标记或评论。"
        )


def clean_pdf_text(raw: str) -> list[str]:
    raw = raw.replace("\f", "\n\n")
    raw = re.sub(r"-\n(?=[a-z])", "", raw)
    raw = re.sub(r"(?<!\n)\n(?!\n)", " ", raw)
    paragraphs = re.split(r"\n\s*\n+", raw)
    result: list[str] = []
    for paragraph in paragraphs:
        paragraph = re.sub(r"\s+", " ", paragraph).strip()
        if not (260 <= len(paragraph) <= 5000):
            continue
        printable = sum(
            ch.isalpha() or ch.isspace() or ch in ".,;:()[]-'" for ch in paragraph
        )
        if printable / max(1, len(paragraph)) < 0.78:
            continue
        if paragraph.lower().startswith(("references ", "acknowledg", "appendix ")):
            continue
        result.append(paragraph)
    return result


def read_pdf_paragraphs(path: Path) -> list[str]:
    reader = PdfReader(str(path))
    text = "\n\n".join(page.extract_text() or "" for page in reader.pages)
    return clean_pdf_text(text)


def _hash_ids(ids: list[int]) -> str:
    payload = b"".join(int(token_id).to_bytes(4, "little", signed=False) for token_id in ids)
    return hashlib.sha256(payload).hexdigest()


def _record(source: str, index: int, ids: list[int]) -> SFTRecord:
    digest = _hash_ids(ids)
    return SFTRecord(
        record_id=f"{source}:{index}:{digest[:16]}",
        source=source,
        response_ids=tuple(ids),
        response_hash=digest,
    )


def build_controlled_split(
    root: Path,
    tokenizer: Any,
    response_tokens: int,
    n_per_class: int,
    n_aux: int,
    seed: int,
) -> tuple[list[SFTRecord], list[SFTRecord], list[SFTRecord], dict[str, Any]]:
    pdf_dir = root / "papers" / "edge-cloud-speculative-decoding"
    pdfs = sorted(path for path in pdf_dir.glob("*.pdf") if path.is_file())
    if not pdfs:
        raise RuntimeError(f"No source PDFs under {pdf_dir}")

    by_doc: dict[str, list[SFTRecord]] = {}
    seen: set[str] = set()
    for pdf in pdfs:
        records: list[SFTRecord] = []
        for paragraph in read_pdf_paragraphs(pdf):
            ids = tokenizer(paragraph, add_special_tokens=False).input_ids
            for start in range(0, len(ids) - response_tokens + 1, response_tokens):
                chunk = list(ids[start : start + response_tokens])
                digest = _hash_ids(chunk)
                if digest in seen:
                    continue
                seen.add(digest)
                records.append(_record(pdf.name, len(records), chunk))
        by_doc[pdf.name] = records

    members: list[SFTRecord] = []
    nonmembers: list[SFTRecord] = []
    auxiliary: list[SFTRecord] = []
    allocation: dict[str, dict[str, int]] = {}
    for doc_idx, (name, records) in enumerate(by_doc.items()):
        shuffled = list(records)
        random.Random(seed + 1009 * (doc_idx + 1)).shuffle(shuffled)
        counts = {"member": 0, "nonmember": 0, "auxiliary": 0}
        for index, record in enumerate(shuffled):
            bucket = index % 3
            if bucket == 0:
                members.append(record)
                counts["member"] += 1
            elif bucket == 1:
                nonmembers.append(record)
                counts["nonmember"] += 1
            else:
                auxiliary.append(record)
                counts["auxiliary"] += 1
        allocation[name] = counts

    rng = random.Random(seed + 77)
    rng.shuffle(members)
    rng.shuffle(nonmembers)
    rng.shuffle(auxiliary)
    if len(members) < n_per_class or len(nonmembers) < n_per_class:
        raise RuntimeError(
            f"Insufficient controlled records: {len(members)} members, "
            f"{len(nonmembers)} nonmembers"
        )
    if len(auxiliary) < n_aux:
        raise RuntimeError(f"Insufficient auxiliary records: {len(auxiliary)}")

    metadata = {
        "source_pdf_count": len(pdfs),
        "unique_chunk_count": len(seen),
        "available_counts": {
            "member": len(members),
            "nonmember": len(nonmembers),
            "auxiliary": len(auxiliary),
        },
        "per_document_allocation": allocation,
        "response_tokens": response_tokens,
        "raw_text_persisted": False,
        "sft_format": "user_prompt_to_assistant_response; prompt labels masked",
    }
    return (
        members[:n_per_class],
        nonmembers[:n_per_class],
        auxiliary[:n_aux],
        metadata,
    )


def prompt_prefix_ids(record: SFTRecord, tokenizer: Any) -> list[int]:
    messages = [{"role": "user", "content": record.prompt}]
    try:
        prefix_ids = tokenizer.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            enable_thinking=False,
        )
    except (TypeError, ValueError, AttributeError):
        try:
            prefix_ids = tokenizer.apply_chat_template(
                messages, tokenize=True, add_generation_prompt=True
            )
        except (ValueError, AttributeError):
            prefix_ids = tokenizer(
                f"### User:\n{record.prompt}\n### Assistant:\n",
                add_special_tokens=True,
            ).input_ids
    return list(prefix_ids)


def make_sft_example(record: SFTRecord, tokenizer: Any) -> dict[str, list[int]]:
    prefix_ids = prompt_prefix_ids(record, tokenizer)

    response_ids = list(record.response_ids)
    if tokenizer.eos_token_id is not None:
        response_ids.append(int(tokenizer.eos_token_id))
    input_ids = list(prefix_ids) + response_ids
    labels = [-100] * len(prefix_ids) + response_ids
    return {"input_ids": input_ids, "labels": labels}


def collate_sft(features: list[dict[str, list[int]]], pad_token_id: int) -> dict[str, Any]:
    max_len = max(len(feature["input_ids"]) for feature in features)
    import torch

    input_ids = torch.full((len(features), max_len), pad_token_id, dtype=torch.long)
    labels = torch.full((len(features), max_len), -100, dtype=torch.long)
    attention_mask = torch.zeros((len(features), max_len), dtype=torch.long)
    for row, feature in enumerate(features):
        length = len(feature["input_ids"])
        input_ids[row, :length] = torch.tensor(feature["input_ids"], dtype=torch.long)
        labels[row, :length] = torch.tensor(feature["labels"], dtype=torch.long)
        attention_mask[row, :length] = 1
    return {
        "input_ids": input_ids,
        "labels": labels,
        "attention_mask": attention_mask,
        "response_mask": labels.ne(-100),
    }


def records_metadata(records: list[SFTRecord]) -> list[dict[str, Any]]:
    return [
        {
            "record_id": record.record_id,
            "source": record.source,
            "response_hash": record.response_hash,
            "response_token_count": len(record.response_ids),
        }
        for record in records
    ]
