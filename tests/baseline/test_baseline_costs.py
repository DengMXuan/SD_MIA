import pytest
import torch

from experiments.baseline.costs import CostMeter


def test_generation_cost_counts_expansion_and_actual_eos_not_padding():
    meter = CostMeter(torch.device('cpu'), records=2)
    meter.forward(5)
    inputs = [[10, 11], [12]]
    outputs = [[[5, 2, 0, 0], [6, 7, 2, 0]], [[2, 0, 0], [8, 9, 10, 11]]]
    meter.generation(inputs, outputs, {2})
    with meter.measure():
        pass
    result = meter.result()
    assert result['totals']['input_tokens'] == 11
    assert result['totals']['generated_tokens'] == 10
    assert result['target_sequences_per_record'] == 2.5
    assert result['tokens_per_record'] == 10.5
    # Splitting a batch cannot change logical cost.
    split = CostMeter(torch.device('cpu'), records=2)
    split.forward(5)
    for prompt, sequences in zip(inputs, outputs):
        split.generation([prompt], [sequences], {2})
    assert (split.input_tokens, split.output_tokens, split.generated_sequences) == (11, 10, 4)


def test_timing_synchronizes_cuda_and_amortizes_over_both_classes(monkeypatch):
    calls = []
    monkeypatch.setattr(torch.cuda, 'synchronize', lambda device: calls.append(str(device)))
    clock = iter((10., 10.04))
    monkeypatch.setattr('experiments.baseline.costs.time.perf_counter', lambda: next(clock))
    meter = CostMeter(torch.device('cuda:0'), records=2)
    with meter.measure():
        pass
    assert calls == ['cuda:0', 'cuda:0']
    assert meter.result()['amortized_ms_per_record'] == pytest.approx(20.)


def test_failed_method_has_no_complete_cost():
    meter = CostMeter(torch.device('cpu'), records=2)
    with pytest.raises(ValueError):
        with meter.measure():
            raise ValueError('failed')
    with pytest.raises(RuntimeError, match='unfinished'):
        meter.result()
