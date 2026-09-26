"""Full-parameter DP-SGD with CPU-offloaded or GPU-resident document sums.

Uses the usual ideal-Gaussian DP analysis with finite-precision PyTorch PRNGs;
this is a research implementation, not a hardened cryptographic DP runtime.
No private loss, sampled IDs/counts, or random seeds are emitted.
"""
from __future__ import annotations

import gc
import secrets

import numpy as np
import torch

from experiments.shared.data.data import collate_sft, make_sft_example
from experiments.shared.training.training import _enable_checkpointing, _make_optimizer, _autocast
from experiments.dp_defense.accounting import PrivacyPlan, epsilon_for
from experiments.dp_defense.conditions import accumulator_settings

CHUNK_ELEMENTS = 1 << 20


class PrivateRandomness:
    """Separate unpublished stage-specific sampling/noise streams, never cfg.seed."""
    def __init__(self, device):
        self.sampling = np.random.default_rng(secrets.randbits(128))
        self.noise = torch.Generator(device=device).manual_seed(secrets.randbits(63))

    def select(self, size, probability):
        return np.flatnonzero(self.sampling.random(size) < probability)

    def normal(self, size, device):
        # Four independent normals / 2, following Opacus' noise hardening idea.
        # This does not turn the underlying PRNG into a CSPRNG.
        value = torch.zeros(size, dtype=torch.float32, device=device)
        for _ in range(4):
            value.add_(torch.randn(size, generator=self.noise, device=device, dtype=torch.float32))
        return value.mul_(.5)


class DocumentGradientSum:
    """FP32 sums: CPU saves VRAM; CUDA avoids gradient offload per document."""
    def __init__(self, parameters, max_norm, *, device="cpu"):
        self.parameters = list(parameters)
        self.max_norm = max_norm
        self.device = torch.device(device)
        if self.device.type not in ("cpu", "cuda"):
            raise ValueError("accumulator device must be CPU or CUDA")
        if self.device.type == "cuda":
            if not self.parameters or any(p.device.type != "cuda" for p in self.parameters):
                raise ValueError("CUDA accumulation requires CUDA parameters")
            if self.device.index is None:
                self.device = self.parameters[0].device
            if any(p.device != self.device for p in self.parameters):
                raise ValueError("CUDA accumulator must share the parameters' device")
        self.sums = [torch.zeros(p.numel(), dtype=torch.float32, device=self.device) for p in self.parameters]

    @torch.no_grad()
    def add_document(self):
        norms = []
        for parameter in self.parameters:
            if parameter.grad is not None:
                if parameter.grad.is_sparse:
                    raise ValueError("sparse gradients are unsupported")
                norms.append(torch.linalg.vector_norm(parameter.grad.detach(), dtype=torch.float64))
        norm = float(torch.linalg.vector_norm(torch.stack(norms))) if norms else 0.
        # Map a nonfinite document gradient to zero, without a data-dependent
        # training abort, log entry, or schedule change.
        factor = min(1., self.max_norm / max(norm, 1e-30)) if np.isfinite(norm) else 0.
        if factor:
            for parameter, total in zip(self.parameters, self.sums):
                if parameter.grad is None:
                    continue
                grad = parameter.grad.detach().reshape(-1)
                if grad.device == total.device:
                    # No temporary buffer is needed, so a whole-parameter add
                    # also avoids thousands of tiny GPU kernel launches.
                    total.add_(grad, alpha=factor)
                    continue
                for start in range(0, len(total), CHUNK_ELEMENTS):
                    end = start + CHUNK_ELEMENTS
                    # Mixed-dtype add accumulates in FP32 without materializing
                    # a cast. CPU offload transfers gradients in their original
                    # dtype; resident sums need neither transfer nor cast copy.
                    total[start:end].add_(grad[start:end].to(total.device), alpha=factor)
        for parameter in self.parameters:
            parameter.grad = None

    @torch.no_grad()
    def set_noisy_gradients(self, noise_multiplier, expected_batch_size, randomness):
        for parameter, total in zip(self.parameters, self.sums):
            # Even unused parameters and empty Poisson batches receive noise.
            parameter.grad = torch.empty_like(parameter)
            gradient = parameter.grad.view(-1)
            for start in range(0, len(total), CHUNK_ELEMENTS):
                end = min(start + CHUNK_ELEMENTS, len(total))
                # Sums are consumed and reset below; same-device noise may be
                # added in place without allocating another FP32 chunk.
                value = total[start:end].to(parameter.device)
                value.add_(randomness.normal(end - start, parameter.device),
                           alpha=noise_multiplier * self.max_norm)
                gradient[start:end].copy_(value.div_(expected_batch_size))
            total.zero_()


def dp_sft_train(model, records, tokenizer, device, plan: PrivacyPlan, *, lr=2e-5,
                 optimizer_name="adamw8bit", progress=None, _randomness=None,
                 document_loss=None, allow_frozen_parameters=False, accumulator_device="cpu"):
    """One privacy event per noisy optimizer step, including empty batches.

    `_randomness` is solely a test seam; production CLI never exposes it.
    The public population in plan, not len(records), determines the mechanism.
    """
    if len({r.record_id for r in records}) != len(records):
        raise ValueError("each raw document must occur once; duplicate document IDs")
    if len(records) > plan.population:
        raise ValueError("records exceed public reference population")
    if not allow_frozen_parameters and any(not p.requires_grad for p in model.parameters()):
        raise ValueError("DP experiment requires full-parameter adaptation")
    trainable = [p for p in model.parameters() if p.requires_grad]
    if not trainable:
        raise ValueError("DP training requires trainable parameters")
    if any(isinstance(m, torch.nn.modules.batchnorm._BatchNorm) for m in model.modules()):
        raise ValueError("data-dependent BatchNorm buffers violate the supported mechanism")
    device = torch.device(device)
    accumulator_settings(accumulator_device)
    if accumulator_device == "cuda":
        if device.type != "cuda" or any(p.device.type != "cuda" for p in trainable):
            raise ValueError("CUDA accumulation requires a CUDA model and training device")
        if device.index is not None and any(p.device != device for p in trainable):
            raise ValueError("CUDA accumulator must share the model's training device")
    randomness = _randomness if _randomness is not None else PrivateRandomness(device)
    if document_loss is None:
        _enable_checkpointing(model)
    optimizer = _make_optimizer(model, lr, optimizer_name)
    accumulator = DocumentGradientSum(trainable, plan.max_grad_norm,
                                      device=device if accumulator_device == "cuda" else "cpu")
    examples = [make_sft_example(record, tokenizer) for record in records]
    model.train()
    for step in range(plan.steps):
        optimizer.zero_grad(set_to_none=True)
        for index in randomness.select(len(records), plan.sample_rate):
            batch = collate_sft([examples[int(index)]], tokenizer.pad_token_id)
            batch = {key: batch[key].to(device) for key in ("input_ids", "attention_mask", "labels")}
            with _autocast(device):
                loss = model(**batch, use_cache=False).loss if document_loss is None else document_loss(model, batch)
            loss.backward()
            accumulator.add_document()
            del loss, batch
        accumulator.set_noisy_gradients(plan.noise_multiplier, plan.expected_batch_size, randomness)
        optimizer.step()
        if progress is not None:
            progress(step + 1, plan.steps)
    optimizer.zero_grad(set_to_none=True)
    model.eval()
    del optimizer, accumulator
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()
    actual = epsilon_for(plan.noise_multiplier, plan.sample_rate, plan.steps, plan.delta)
    if actual > plan.epsilon:
        raise RuntimeError("completed privacy expenditure exceeds the budget")
    return {**plan.as_dict(), "completed_steps": plan.steps, "accounted_epsilon": actual,
            "physical_microbatch_size": 1, **accumulator_settings(accumulator_device),
            "randomness": "independent_unpublished_os_seeded_prng_streams",
            "numerics": "finite_precision_research_implementation",
            "private_training_metrics_released": False}
