import argparse

import pytest
import torch
import torch.nn.functional as F

from experiments.sd_membership_sft import training as training_module
from experiments.sd_membership_sft.drafts.plain import (
    _checkpoint_complete,
    _offload_model,
    load_config,
)
from experiments.sd_membership_sft.training import (
    _backward_chunked_distillation_loss,
    _update_bnb_8bit_parameter_in_chunks,
    distill_on_auxiliary,
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


def test_chunked_distillation_scales_only_the_gradient():
    torch.manual_seed(11)
    teacher = torch.randn(1, 3, 7)
    labels = torch.tensor([[-100, 2, 3]])
    baseline_logits = torch.randn(1, 3, 7, requires_grad=True)
    baseline_loss = _backward_chunked_distillation_loss(
        baseline_logits, teacher, labels, temperature=2.0, chunk_size=1
    )

    scaled_logits = baseline_logits.detach().clone().requires_grad_(True)
    scaled_loss = _backward_chunked_distillation_loss(
        scaled_logits,
        teacher,
        labels,
        temperature=2.0,
        chunk_size=1,
        loss_scale=0.25,
    )

    assert scaled_loss == pytest.approx(baseline_loss, rel=1e-6)
    torch.testing.assert_close(scaled_logits.grad, baseline_logits.grad * 0.25)


def test_distillation_steps_are_optimizer_updates_with_gradient_accumulation(
    monkeypatch,
):
    class DummyModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = torch.nn.Parameter(torch.tensor(0.0))

        def forward(self, input_ids, attention_mask, use_cache):
            del attention_mask, use_cache
            batch_size, sequence_length = input_ids.shape
            logits = self.weight.expand(batch_size, sequence_length, 5)
            return type("Output", (), {"logits": logits})()

    class RecordingOptimizer:
        def __init__(self):
            self.zero_grad_calls = 0
            self.step_calls = 0

        def zero_grad(self, *, set_to_none):
            assert set_to_none
            self.zero_grad_calls += 1

        def step(self):
            self.step_calls += 1

    optimizer = RecordingOptimizer()
    loss_scales = []
    monkeypatch.setattr(
        training_module,
        "_make_optimizer",
        lambda model, lr, optimizer_name: optimizer,
    )
    monkeypatch.setattr(
        training_module,
        "make_sft_example",
        lambda record, tokenizer: record,
    )
    monkeypatch.setattr(
        training_module,
        "collate_sft",
        lambda examples, pad_token_id: {
            "input_ids": torch.ones(len(examples), 3, dtype=torch.long),
            "attention_mask": torch.ones(len(examples), 3, dtype=torch.long),
            "labels": torch.tensor([[1, 2, 3]] * len(examples)),
        },
    )
    monkeypatch.setattr(
        training_module,
        "_backward_chunked_distillation_loss",
        lambda *args, loss_scale: loss_scales.append(loss_scale) or 4.0,
    )

    losses = distill_on_auxiliary(
        DummyModel(),
        DummyModel(),
        records=[object(), object()],
        tokenizer=type("Tokenizer", (), {"pad_token_id": 0})(),
        device=torch.device("cpu"),
        steps=3,
        batch_size=1,
        grad_accum=4,
        lr=2e-5,
        temperature=2.0,
        seed=1919,
    )

    assert optimizer.zero_grad_calls == 3
    assert optimizer.step_calls == 3
    assert loss_scales == [0.25] * 12
    assert losses == [4.0, 4.0, 4.0]


def test_bnb_8bit_large_parameter_update_is_split_on_block_boundaries():
    parameter = torch.nn.Parameter(torch.zeros(700))
    parameter.grad = torch.ones_like(parameter)

    class FakeOptimizer:
        optimizer_name = "adam"

        def __init__(self):
            state1 = torch.zeros(700, dtype=torch.uint8)
            state2 = torch.zeros(700, dtype=torch.uint8)
            state1.is_paged = True
            state2.is_paged = True
            self.state = {
                parameter: {
                    "step": 0,
                    "state1": state1,
                    "state2": state2,
                    "qmap1": torch.zeros(256),
                    "qmap2": torch.zeros(256),
                    "absmax1": torch.zeros(3),
                    "absmax2": torch.zeros(3),
                }
            }

        @staticmethod
        def get_config(group_index, parameter_index, group):
            return {
                "betas": (0.9, 0.999),
                "eps": 1e-8,
                "lr": 2e-5,
                "alpha": 0.0,
                "weight_decay": 0.01,
                "skip_zeros": False,
            }

    class FakeFunctional:
        calls = []

        @classmethod
        def optimizer_update_8bit_blockwise(
            cls,
            optimizer_name,
            gradient,
            current_parameter,
            state1,
            state2,
            beta1,
            beta2,
            beta3,
            alpha,
            eps,
            step,
            lr,
            qmap1,
            qmap2,
            absmax1,
            absmax2,
            weight_decay,
            gnorm_scale,
            skip_zeros,
        ):
            cls.calls.append(
                (
                    gradient.numel(),
                    state1.numel(),
                    absmax1.numel(),
                    step,
                    getattr(state1, "is_paged", False),
                    getattr(state2, "is_paged", False),
                )
            )
            current_parameter.add_(gradient)

    optimizer = FakeOptimizer()
    _update_bnb_8bit_parameter_in_chunks(
        optimizer,
        {},
        parameter,
        0,
        0,
        functional=FakeFunctional,
        chunk_elements=512,
    )

    assert FakeFunctional.calls == [
        (512, 512, 2, 1, True, True),
        (188, 188, 1, 1, True, True),
    ]
    assert optimizer.state[parameter]["step"] == 1
    torch.testing.assert_close(parameter, torch.ones_like(parameter))
