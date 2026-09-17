import argparse

import pytest
import torch
import torch.nn.functional as F

from experiments.sd_membership_sft.drafts.plain import (
    _checkpoint_complete,
    _offload_model,
    load_config,
)
from experiments.sd_membership_sft.training import (
    _backward_chunked_distillation_loss,
)


class RecordingModule(torch.nn.Linear):
    def __init__(self):
        super().__init__(2, 2)
        self.to_calls = []

    def to(self, *args, **kwargs):
        self.to_calls.append(args[0] if args else kwargs.get("device"))
        return super().to(*args, **kwargs)


def test_offload_model_moves_model_to_cpu_and_clears_cuda_cache(monkeypatch):
    model = RecordingModule()
    empty_cache_calls = []
    monkeypatch.setattr(torch.cuda, "empty_cache", lambda: empty_cache_calls.append(True))

    _offload_model(model, torch.device("cuda:0"))

    assert model.to_calls == ["cpu"]
    assert empty_cache_calls == [True]


def test_checkpoint_complete_rejects_partial_save(tmp_path):
    checkpoint = tmp_path / "target"
    checkpoint.mkdir()
    (checkpoint / "config.json").write_text("{}")
    assert not _checkpoint_complete(checkpoint)

    (checkpoint / "model-00001-of-00002.safetensors").touch()
    assert not _checkpoint_complete(checkpoint)

    (checkpoint / "model.safetensors.index.json").write_text("{}")
    assert _checkpoint_complete(checkpoint)


def test_model_pair_condition_rejects_different_data_seed():
    args = argparse.Namespace(
        config=None,
        skip_trained_drafts=None,
        no_save_adapters=None,
        resume=None,
        seed=1919,
        data_seed=1949,
    )
    with pytest.raises(ValueError, match="--seed and --data-seed"):
        load_config(args)


def test_chunked_distillation_matches_full_objective_and_gradient():
    torch.manual_seed(7)
    teacher = torch.randn(2, 4, 11)
    labels = torch.tensor([[-100, 2, 3, 4], [-100, -100, 6, 7]])
    temperature = 2.0

    expected_logits = torch.randn(2, 4, 11, requires_grad=True)
    valid = labels.ne(-100)
    expected_selected = expected_logits[valid]
    teacher_selected = teacher[valid]
    expected_loss = 0.20 * F.cross_entropy(
        expected_selected, labels[valid]
    ) + 0.80 * F.kl_div(
        F.log_softmax(expected_selected / temperature, dim=-1),
        F.softmax(teacher_selected / temperature, dim=-1),
        reduction="batchmean",
    ) * (temperature**2)
    expected_loss.backward()

    actual_logits = expected_logits.detach().clone().requires_grad_(True)
    actual_loss = _backward_chunked_distillation_loss(
        actual_logits, teacher, labels, temperature, chunk_size=2
    )

    assert actual_loss == pytest.approx(float(expected_loss.detach()), rel=1e-6)
    torch.testing.assert_close(actual_logits.grad, expected_logits.grad)
