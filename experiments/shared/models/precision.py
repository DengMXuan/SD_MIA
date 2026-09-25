"""Temporary diagnostic precision; deployment weights and buffers are restored."""
from contextlib import contextmanager
import torch
from torch.nn.attention import SDPBackend, sdpa_kernel


@contextmanager
def inference_attention(*models):
    """Gemma4's efficient SDPA kernel fails prefix equivalence on this stack.

    Keep BF16, but use the math implementation for both full rows and generation.
    This is evaluation-only; language-model training retains its own settings.
    """
    configs = [getattr(model, 'config', None) for model in models]
    gemma = any(str(getattr(getattr(cfg, 'text_config', cfg), 'model_type', '')).startswith('gemma4')
                for cfg in configs)
    if gemma:
        with sdpa_kernel(SDPBackend.MATH):
            yield
    else:
        yield


@contextmanager
def fp32_reference(adapter):
    models = [getattr(adapter, name, None) for name in ('target', 'draft')]
    if any(not isinstance(model, torch.nn.Module) for model in models):
        raise ValueError('FP32 recheck unavailable for prefix inconsistency (possible future-token leakage)')
    # Preserve per-tensor dtypes, including FP32 norms and nonpersistent RoPE
    # buffers. Do not call bfloat16() on the whole model on exit.
    modules = list(dict.fromkeys(module for model in models for module in model.modules()))
    parameters = list({id(p): p for module in modules for p in module.parameters(recurse=False)}.values())
    dtypes = [(p, p.dtype) for p in parameters if p.is_floating_point()]
    buffers = [(module, dict(module._buffers)) for module in modules]
    tf32 = torch.backends.cuda.matmul.allow_tf32
    try:
        torch.backends.cuda.matmul.allow_tf32 = False
        for p, _ in dtypes:
            p.data = p.data.float()
        for module in modules:
            for name, value in module._buffers.items():
                if value is not None and value.is_floating_point():
                    # Separate storage: forward may mutate a diagnostic buffer.
                    module._buffers[name] = value.float().clone()
        with torch.autocast(device_type=adapter.device.type, enabled=False), sdpa_kernel(SDPBackend.MATH):
            yield
    finally:
        for p, dtype in dtypes:
            if p.dtype != dtype:
                p.data = p.data.to(dtype=dtype)
        for module, original in buffers:
            module._buffers.clear()
            module._buffers.update(original)
        torch.backends.cuda.matmul.allow_tf32 = tf32
