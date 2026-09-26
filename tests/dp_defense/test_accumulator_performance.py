"""Bound copies without changing the per-document DP mechanism (CPU fixtures)."""
from types import SimpleNamespace

import pytest
import torch
from torch.utils._python_dispatch import TorchDispatchMode

from experiments.dp_defense import training
from experiments.dp_defense.training import DocumentGradientSum
from tests.dp_defense.test_dp_defense import FixedNoise, single_thread


class CopyOperations(TorchDispatchMode):
    def __init__(self):
        super().__init__()
        self.elements = 0
        self.adds = 0

    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        if func == torch.ops.aten._to_copy.default:
            self.elements += args[0].numel()
        if func == torch.ops.aten.add_.Tensor:
            self.adds += 1
        return func(*args, **(kwargs or {}))


def test_resident_mixed_precision_accumulation_has_no_gradient_sized_cast_copy(monkeypatch):
    monkeypatch.setattr(training, 'CHUNK_ELEMENTS', 32)
    parameter = torch.nn.Parameter(torch.zeros(1024, dtype=torch.bfloat16))
    parameter.grad = torch.full_like(parameter, .25)
    accumulator = DocumentGradientSum([parameter], 1.)
    copies = CopyOperations()
    with copies:
        accumulator.add_document()
    # The FP64 global norm is retained; adding BF16 into FP32 needs no materialized cast.
    assert copies.elements == 0
    assert copies.adds == 1
    torch.testing.assert_close(accumulator.sums[0], torch.full((1024,), 1 / 32))
    assert parameter.grad is None


@pytest.mark.parametrize('dtype', [torch.float32, torch.bfloat16])
def test_chunked_clipping_noise_and_buffer_reuse_match_reference(monkeypatch, dtype):
    monkeypatch.setattr(training, 'CHUNK_ELEMENTS', 3)
    parameters = [torch.nn.Parameter(torch.zeros(n, dtype=dtype)) for n in (7, 2)]
    accumulator = DocumentGradientSum(parameters, 1.)
    expected = [torch.zeros(p.numel()) for p in parameters]
    documents = ([torch.arange(7.), torch.tensor([3., 4.])],
                 [torch.full((7,), -2.), None],
                 [torch.full((7,), float('inf')), torch.ones(2)])
    for document in documents:
        for p, g in zip(parameters, document):
            p.grad = None if g is None else g.to(dtype)
        norm = torch.linalg.vector_norm(torch.cat([p.grad.double() for p in parameters if p.grad is not None]))
        factor = min(1., 1. / max(float(norm), 1e-30)) if torch.isfinite(norm) else 0.
        if factor:
            for total, p in zip(expected, parameters):
                if p.grad is not None:
                    total.add_(p.grad.float(), alpha=factor)
        accumulator.add_document()
    noise = FixedNoise(value=.5)
    accumulator.set_noisy_gradients(2., 4, noise)
    for p, total in zip(parameters, expected):
        torch.testing.assert_close(p.grad, ((total + 1.) / 4).to(dtype), rtol=0, atol=0)
    assert all(torch.count_nonzero(s) == 0 for s in accumulator.sums)
    # A following empty Poisson batch still releases noise, with no old sum retained.
    accumulator.set_noisy_gradients(2., 4, noise)
    for p in parameters:
        torch.testing.assert_close(p.grad, torch.full_like(p, .25))


def test_resident_noise_does_not_copy_fp32_accumulator():
    parameter = torch.nn.Parameter(torch.zeros(1024))
    accumulator = DocumentGradientSum([parameter], 1.)
    copies = CopyOperations()
    with copies:
        accumulator.set_noisy_gradients(1., 2, FixedNoise(value=1.))
    assert copies.elements == 0
    torch.testing.assert_close(parameter.grad, torch.full_like(parameter, .5))


def test_cuda_accumulator_requires_cuda_model_before_optimizer_creation(monkeypatch):
    model = torch.nn.Linear(1, 1)
    monkeypatch.setattr(training, '_make_optimizer', lambda *a: pytest.fail('optimizer initialized'))
    with pytest.raises(ValueError, match='CUDA'):
        training.dp_sft_train(model, [], SimpleNamespace(pad_token_id=0), 'cpu',
                              SimpleNamespace(population=1), accumulator_device='cuda')


@pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA comparison is opt-in on a free GPU')
def test_cuda_resident_and_cpu_offload_match():
    parameters = [[torch.nn.Parameter(torch.zeros(7, device='cuda', dtype=torch.bfloat16))]
                  for _ in range(2)]
    accumulators = [DocumentGradientSum(parameters[0], 1., device='cpu'),
                    DocumentGradientSum(parameters[1], 1., device='cuda')]
    for scale in (1., -2.):
        for ps, accumulator in zip(parameters, accumulators):
            ps[0].grad = torch.arange(7, device='cuda', dtype=torch.bfloat16) * scale
            accumulator.add_document()
    for accumulator in accumulators:
        accumulator.set_noisy_gradients(2., 4, FixedNoise(value=.5))
    torch.testing.assert_close(parameters[0][0].grad, parameters[1][0].grad, rtol=0, atol=0)
