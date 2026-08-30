from __future__ import annotations

import torch

from experiments.sd_membership_sft.dspark_protocol_benchmark import (
    _draft_to_target_ids,
)


def test_dspark_d2t_values_are_offsets() -> None:
    draft_ids = torch.tensor([0, 1, 2, 3])
    offsets = torch.tensor([0, 4, 8, 12])
    torch.testing.assert_close(
        _draft_to_target_ids(draft_ids, offsets),
        torch.tensor([0, 5, 10, 15]),
    )


def test_dspark_full_vocab_mapping_is_identity() -> None:
    draft_ids = torch.tensor([3, 7])
    torch.testing.assert_close(_draft_to_target_ids(draft_ids, None), draft_ids)
