"""Synthetic accumulator benchmark; no private data, model loading or training.

CPU is the default. Use --device cuda:N only when that GPU is free. CUDA runs
compare CPU offload with resident FP32 accumulation on the same synthetic grads.
Results cover gradient aggregation/noise only, not end-to-end model throughput.
"""
from __future__ import annotations

import argparse
import gc
import json
import time

import torch

from experiments.dp_defense.training import DocumentGradientSum, PrivateRandomness


def benchmark(args, storage, parameters, gradients, device):
    accumulator = DocumentGradientSum(parameters, 1., device=storage)
    randomness = PrivateRandomness(device)

    def synchronize():
        if device.type == 'cuda':
            torch.cuda.synchronize(device)

    def step():
        for _ in range(args.documents):
            for parameter, gradient in zip(parameters, gradients):
                parameter.grad = gradient
            accumulator.add_document()
        accumulator.set_noisy_gradients(1., args.documents, randomness)
        for parameter in parameters:
            parameter.grad = None

    step()  # Untimed warmup; resets every sum before measured steps.
    synchronize()
    if device.type == 'cuda':
        torch.cuda.reset_peak_memory_stats(device)
    started = time.perf_counter()
    for _ in range(args.steps):
        step()
    synchronize()
    seconds = time.perf_counter() - started
    result = dict(accumulator_device=storage, seconds=seconds,
                  seconds_per_step=seconds / args.steps,
                  accumulator_bytes=sum(p.numel() for p in parameters) * 4)
    if device.type == 'cuda':
        result['peak_cuda_allocated_bytes'] = torch.cuda.max_memory_allocated(device)
    del accumulator
    gc.collect()
    if device.type == 'cuda':
        torch.cuda.empty_cache()
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--device', default='cpu', help='cpu, or cuda:N on an idle GPU')
    parser.add_argument('--parameters', type=int, default=4)
    parser.add_argument('--elements-per-parameter', type=int, default=1 << 20)
    parser.add_argument('--documents', type=int, default=4)
    parser.add_argument('--steps', type=int, default=3)
    parser.add_argument('--threads', type=int, default=1)
    args = parser.parse_args()
    if min(args.parameters, args.elements_per_parameter, args.documents, args.steps, args.threads) < 1:
        parser.error('sizes, steps and threads must be positive')
    device = torch.device(args.device)
    if device.type not in ('cpu', 'cuda'):
        parser.error('choose cpu or cuda:N')
    if device.type == 'cuda':
        if not torch.cuda.is_available():
            parser.error('CUDA is unavailable')
        torch.cuda.set_device(device)
    torch.set_num_threads(args.threads)
    parameters = [torch.nn.Parameter(torch.zeros(args.elements_per_parameter, dtype=torch.bfloat16,
                                                device=device)) for _ in range(args.parameters)]
    gradients = [torch.full_like(p, .01) for p in parameters]
    results = [benchmark(args, storage, parameters, gradients, device)
               for storage in (('cpu', 'cuda') if device.type == 'cuda' else ('cpu',))]
    report = dict(scope='synthetic gradient aggregation and noise; excludes model forward/backward and optimizer',
                  device=str(device), dtype='bfloat16 gradients / float32 sums', settings=vars(args), results=results)
    if device.type == 'cuda':
        report['resident_speedup'] = results[0]['seconds'] / results[1]['seconds']
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
