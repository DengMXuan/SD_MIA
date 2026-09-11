"""Three comparable per-record costs for independently executed methods."""
from __future__ import annotations

import time
from contextlib import contextmanager

import torch


class CostMeter:
    def __init__(self, device, records):
        if records <= 0:
            raise ValueError('cost measurement requires nonempty audit records')
        self.device = device
        self.records = records
        self.forward_sequences = 0
        self.generated_sequences = 0
        self.input_tokens = 0
        self.output_tokens = 0
        self.seconds = None

    def forward(self, input_tokens):
        self.forward_sequences += 1
        self.input_tokens += input_tokens

    def generation(self, inputs, outputs, eos_ids):
        # outputs are grouped by input, expanded over num_return_sequences.
        # Replicate prompt cost per returned sequence, independent of batch size.
        for prompt, sequences in zip(inputs, outputs, strict=True):
            for sequence in sequences:
                self.generated_sequences += 1
                self.input_tokens += len(prompt)
                self.output_tokens += next((i + 1 for i, token in enumerate(sequence) if token in eos_ids), len(sequence))

    def _sync(self):
        if self.device.type == 'cuda':
            torch.cuda.synchronize(self.device)

    @contextmanager
    def measure(self):
        self._sync()
        start = time.perf_counter()
        yield
        self._sync()
        self.seconds = time.perf_counter() - start

    def result(self):
        if self.seconds is None:
            raise RuntimeError('cannot report an unfinished method cost')
        return {
            'amortized_ms_per_record': self.seconds * 1000 / self.records,
            'target_sequences_per_record': (self.forward_sequences + self.generated_sequences) / self.records,
            'tokens_per_record': (self.input_tokens + self.output_tokens) / self.records,
            # Raw totals make sharded/amortized accounting auditable. They are
            # supporting data, not additional headline comparison metrics.
            'totals': dict(records=self.records, seconds=self.seconds,
                forward_sequences=self.forward_sequences, generated_sequences=self.generated_sequences,
                input_tokens=self.input_tokens, generated_tokens=self.output_tokens),
        }


def cost_protocol(model, device, warmup_records, batch_size):
    try:
        dtype = str(next(model.parameters()).dtype)
    except (AttributeError, StopIteration):
        dtype = 'unknown'
    return dict(version=1, execution='independent methods; one resident model',
        time_scope='method preparation/calibration + scoring + postprocessing; CUDA synchronized',
        exclusions='model/data loading, warmup, result serialization; progress bookkeeping is included',
        amortization='all member and nonmember audit records; auxiliary costs included in numerator',
        sequences='one teacher-forced sequence or one returned generation sequence, including repeats; not API calls or decoder steps',
        tokens='nonpadding input + actual generated tokens through first EOS inclusive; prompt repeated per returned sequence',
        warmup_records_per_method=warmup_records, teacher_forced_batch_size=1, generation_batch_size=batch_size,
        device=str(device), device_name=torch.cuda.get_device_name(device) if device.type == 'cuda' else 'cpu',
        model_dtype=dtype, torch_version=torch.__version__)


def cost_table(costs):
    if not costs:
        return []
    lines = ['', '## Cost and efficiency', '',
        'Independent execution; setup/calibration amortized over the audit. Lower is better.', '',
        '| Method | ms / record | Target sequences / record | Tokens / record |',
        '|---|---:|---:|---:|']
    for name, row in costs.items():
        lines.append(f"| `{name}` | {row['amortized_ms_per_record']:.3f} | {row['target_sequences_per_record']:.3f} | {row['tokens_per_record']:.1f} |")
    lines.extend(['', 'Time is batch-amortized cost, not batch-size-1 latency. Tokens are a workload proxy, not FLOPs or API billing.'])
    return lines


def write_cost_report(directory, protocol, costs):
    import json
    from pathlib import Path
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    payload = {'measurement': protocol.get('cost_measurement'), 'methods': costs}
    for name, text in (
        ('baseline_costs.json', json.dumps(payload, indent=2) + '\n'),
        ('BASELINE_COSTS.md', '# Baseline cost comparison\n' + '\n'.join(cost_table(costs)) + '\n'),
    ):
        temporary = directory / (name + '.tmp')
        temporary.write_text(text, encoding='utf-8')
        temporary.replace(directory / name)
