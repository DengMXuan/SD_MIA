"""Synchronized phase timing and explicit standalone/physical cost accounting."""
from __future__ import annotations

from contextlib import contextmanager
import time
import torch

PHASES = ("preparation", "calibration", "test")


def sync(device):
    if torch.device(device).type == "cuda":
        torch.cuda.synchronize(device)


@contextmanager
def timed(device="cpu"):
    sync(device)
    start = time.perf_counter()
    value = {}
    yield value
    sync(device)
    value["seconds"] = time.perf_counter() - start


def peak_memory(device):
    return int(torch.cuda.max_memory_allocated(device)) if torch.device(device).type == "cuda" else None


def reset_peak(device):
    if torch.device(device).type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)


def summarize_cost(phases, n_test, counters, peak_bytes, *, execution_group):
    if n_test <= 0 or set(phases) != set(PHASES) or any(v < 0 for v in phases.values()):
        raise ValueError("invalid phase cost")
    total = sum(phases.values())
    result = {f"{name}_seconds": float(value) for name, value in phases.items()}
    result.update(total_seconds=total, amortized_ms_per_record=1000 * total / n_test,
                  amortized_records_per_second=n_test / total if total else None,
                  scoring_records_per_second=n_test / phases["test"] if phases["test"] else None,
                  peak_allocated_gpu_bytes=peak_bytes, execution_group=execution_group)
    result.update(counters)
    for name in ("target_sequences", "draft_sequences", "target_input_tokens", "draft_input_tokens", "generated_tokens"):
        value = counters.get(name)
        result[name + "_per_record"] = value / n_test if value is not None else None
    return result


COST_CONVENTIONS = {
    "scope": "preparation/fit + independent calibration + test scoring; amortized over test documents",
    "exclusions": "model/data loading, untimed warmup, archive I/O, ROC/bootstrap reporting",
    "synchronization": "CUDA synchronized at measurement boundaries; wall-clock elapsed time, not kernel-only time",
    "queries": "logical forward sequences and returned generation sequences; actual target forward calls stored separately",
    "tokens": "input/generated token workload, not FLOPs; target and draft work recorded separately",
    "reuse": "each display row retains standalone cost; execution_group identifies shared physical execution",
    "memory": "peak torch allocated bytes including resident models; not total device memory or NVML usage",
}
