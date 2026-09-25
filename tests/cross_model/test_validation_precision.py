"""Regression: sequence-length BF16 drift is distinct from future-token leakage."""
from types import SimpleNamespace
import pytest
import torch
from experiments.shared.models.validation import validate_adapter
from experiments.shared.protocols.sd_protocol import FrozenAdapter


class LengthSensitiveLM(torch.nn.Module):
    def __init__(self, *, drift=False, leak=False, misalign=False, fail_fp32=False):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.ones(4, dtype=torch.bfloat16))
        self.register_buffer('precise_buffer', torch.tensor([1.001], dtype=torch.float32))
        self.register_buffer('integer_buffer', torch.tensor([2]))
        self.drift, self.leak, self.misalign, self.fail_fp32 = drift, leak, misalign, fail_fp32
    def forward(self, input_ids, **kwargs):
        if self.fail_fp32 and self.weight.dtype == torch.float32:
            raise RuntimeError('unsupported FP32 kernel')
        logits = torch.zeros((*input_ids.shape, 4), dtype=self.weight.dtype)
        if self.drift and self.weight.dtype == torch.bfloat16 or self.misalign:
            logits[..., 0] = input_ids.shape[1] / 2
        if self.leak:
            logits[..., 0] += input_ids.sum()
        return SimpleNamespace(logits=logits)


def pair(**kwargs):
    return FrozenAdapter(LengthSensitiveLM(), LengthSensitiveLM(**kwargs), 'plain', 'cpu')


def test_bf16_shape_drift_rechecks_in_fp32_and_restores_models():
    adapter = pair(drift=True)
    before = {name: value.clone() for name, value in adapter.draft.state_dict().items()}
    report = validate_adapter(adapter, [0, 1], [2, 3, 1, 2], seed=1949)
    assert report['status'] == 'passed' and report['fp32_recheck']['status'] == 'passed'
    assert any(row['prefix']['total_variation'] > 0 for row in report['checks'])
    assert all(row['causality_max_abs_log_error'] == 0 for row in report['checks'])
    for name, value in adapter.draft.state_dict().items():
        assert value.dtype == before[name].dtype and torch.equal(value, before[name])
    assert adapter.draft.weight.dtype == torch.bfloat16


def test_real_future_dependence_fails_even_if_length_unchanged():
    with pytest.raises(ValueError, match='future-token'):
        validate_adapter(pair(leak=True), [0, 1], [2, 3, 1, 2])


def test_fp32_does_not_excuse_genuine_prefix_misalignment():
    adapter = pair(misalign=True)
    with pytest.raises(ValueError, match='FP32.*prefix'):
        validate_adapter(adapter, [0, 1], [2, 3, 1, 2])
    assert adapter.draft.weight.dtype == torch.bfloat16


def test_failed_high_precision_kernel_blocks_and_restores_bf16():
    adapter = pair(drift=True, fail_fp32=True)
    with pytest.raises(ValueError, match='FP32'):
        validate_adapter(adapter, [0, 1], [2, 3, 1, 2])
    assert adapter.target.weight.dtype == adapter.draft.weight.dtype == torch.bfloat16
    assert adapter.draft.precise_buffer.dtype == torch.float32


def test_causal_stable_pair_does_not_need_fp32():
    report = validate_adapter(pair(), [0, 1], [2, 3, 1, 2])
    assert report['fp32_recheck']['status'] == 'not_needed'


def test_gemma_reference_attention_policy_is_scoped_and_restored():
    from experiments.shared.models.precision import inference_attention
    model = SimpleNamespace(config=SimpleNamespace(text_config=SimpleNamespace(model_type='gemma4_text')))
    before = torch.backends.cuda.mem_efficient_sdp_enabled()
    with pytest.raises(RuntimeError, match='forward failure'):
        with inference_attention(model):
            assert torch.backends.cuda.math_sdp_enabled()
            assert not torch.backends.cuda.mem_efficient_sdp_enabled()
            raise RuntimeError('forward failure')
    assert torch.backends.cuda.mem_efficient_sdp_enabled() == before
