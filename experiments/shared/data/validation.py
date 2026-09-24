"""Frozen passport and tokenizer validation shared by all audit callers."""
from __future__ import annotations
import json
from pathlib import Path
from transformers import AutoTokenizer
from .data import SFTRecord

def verify_split_against_run(
    members: list[SFTRecord],
    nonmembers: list[SFTRecord],
    run_dir: Path,
) -> None:
    """Fail loud if the rebuilt split disagrees with the run's data passport.

    Identity is the response hash (the token-sequence digest); record ids
    carry a scheme prefix that was renamed (``nart:`` -> ``sft:``) after
    these runs were trained, so the prefix is not stable across commits.
    """
    artifact = json.loads((run_dir / "results.json").read_text(encoding="utf-8"))
    stored = artifact["records"]
    for class_name, rebuilt in (
        ("members", members),
        ("nonmembers", nonmembers),
    ):
        stored_hashes = [record["response_hash"] for record in stored[class_name]]
        rebuilt_hashes = [record.response_hash for record in rebuilt]
        if stored_hashes != rebuilt_hashes:
            raise RuntimeError(
                f"Rebuilt {class_name} split does not match the run passport in {run_dir}"
            )



def verify_shared_tokenizer(target_id: str, draft_id: str) -> None:
    """The teacher-forced audit assumes one tokenization for p and q."""
    target_tokenizer = AutoTokenizer.from_pretrained(target_id)
    draft_tokenizer = AutoTokenizer.from_pretrained(draft_id)
    probe = "edge-cloud speculative decoding membership audit probe 0123"
    if (
        target_tokenizer(probe).input_ids != draft_tokenizer(probe).input_ids
        or target_tokenizer.vocab_size != draft_tokenizer.vocab_size
    ):
        raise RuntimeError(
            f"Target {target_id} and draft {draft_id} tokenizers disagree; "
            "the shared-tokenization audit protocol does not hold"
        )
