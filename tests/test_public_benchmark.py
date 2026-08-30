from __future__ import annotations

import pytest

from experiments.sd_membership_sft.public_benchmark import (
    _alignment_recovery,
    _check_tokenizers,
)


class _TokenizerStub:
    def __init__(
        self,
        vocab: dict[str, int],
        *,
        bos: int = 1,
        eos: int = 2,
        pad: int = 0,
        unk: int = 3,
    ) -> None:
        self._vocab = vocab
        self.bos_token_id = bos
        self.eos_token_id = eos
        self.pad_token_id = pad
        self.unk_token_id = unk

    def get_vocab(self) -> dict[str, int]:
        return dict(self._vocab)

    def __len__(self) -> int:
        return len(self._vocab)


def test_tokenizer_check_requires_special_token_alignment() -> None:
    draft = _TokenizerStub({"a": 0, "b": 1}, eos=1)
    target = _TokenizerStub({"a": 0, "b": 1}, eos=0)

    with pytest.raises(ValueError, match="special-token IDs"):
        _check_tokenizers(draft, target)


def test_alignment_recovery_requires_acceptance_and_rmse_improvements() -> None:
    diagnostics = {
        "base_pair_unadapted": {
            "exact_acceptance_overall": 0.70,
            "top1_agreement_overall": 0.68,
            "candidate_logp_rmse_overall": 1.1,
        },
        "target_only_mismatch": {
            "exact_acceptance_overall": 0.58,
            "top1_agreement_overall": 0.57,
            "candidate_logp_rmse_overall": 2.0,
        },
        "restored": {
            "exact_acceptance_overall": 0.64,
            "top1_agreement_overall": 0.61,
            "candidate_logp_rmse_overall": 1.6,
        },
        "acceptance_only": {
            "exact_acceptance_overall": 0.62,
            "top1_agreement_overall": 0.60,
            "candidate_logp_rmse_overall": 2.1,
        },
    }

    result = _alignment_recovery(
        diagnostics, ["restored", "acceptance_only"]
    )

    assert result["restored"]["alignment_gate"] == "RESTORED"
    assert result["restored"]["base_acceptance_loss_restored_fraction"] == pytest.approx(0.5)
    assert result["acceptance_only"]["alignment_gate"] == "NOT_RESTORED"
