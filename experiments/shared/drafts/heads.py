"""Drafter-head loading for the EAGLE-3 / MTP draft approaches.

Two head families are supported, both hidden-conditioned (class-B speculators
in the DraVer-X sense): they consume verifier hidden states, so the client
only runs them when the protocol exports those states.

The native Qwen3.5 MTP layer (``mtp.*`` checkpoint keys, gated attention)
is used through its speculators conversion (:func:`ensure_mtp_conversion` /
:func:`load_mtp_speculator`); a from-scratch self-trained MTP head was tried
and rejected (acceptance ceiling 0.37 — see PROTOCOL_FT_REPORT.md).

EAGLE-3 heads are loaded through the checkpoint's own ``eagle3.py`` remote
code (the historical, proven path): ``speculator(input_ids, hidden_states=
[B, L, 3*target_H])`` returns draft-vocabulary logits; ``speculator.d2t``
maps draft positions back into the target vocabulary.
"""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from typing import Any

import torch



# ---------------------------------------------------------------------------
# Native MTP speculator (speculators MTPConverter path, Qwen3.5 line)
# ---------------------------------------------------------------------------


def ensure_mtp_conversion(
    model_path: str, converted: Path | str, num_speculative_steps: int = 1
) -> Path:
    """Convert a checkpoint's native ``mtp.*`` layer into a speculators model.

    Depth-1 matches the native module (Qwen3.5 ships a single MTP layer);
    larger step counts replicate it into a chain.
    """
    converted = Path(converted)
    if (converted / "config.json").is_file() and any(converted.glob("*.safetensors")):
        return converted
    if converted.exists():
        raise RuntimeError(f"Converted MTP path exists but is incomplete: {converted}")
    from speculators.convert.mtp import MTPConverter

    MTPConverter().convert(
        input_path=model_path,
        output_path=str(converted),
        base_model=model_path,
        num_speculative_steps=num_speculative_steps,
        validate=True,
    )
    return converted


def load_mtp_speculator(
    converted: Path | str,
    device: torch.device,
    verifier_checkpoint: Path | str | None = None,
) -> Any:
    """Load a converted native-MTP speculator (batch-1 teacher-forced API).

    The library wraps forward in ``conditional_torch_compile``, which breaks
    the autograd graph in eval-then-train flows; unwrap it so head training
    (joint / adapt) and measurement share the eager path.
    """
    from speculators import SpeculatorModel, SpeculatorModelConfig

    config = SpeculatorModelConfig.from_pretrained(
        str(converted), local_files_only=True
    )
    if verifier_checkpoint is not None:
        # The trainable native MTP layer remains the pinned published layer,
        # while verifier-owned frozen embeddings/output weights must match the
        # condition's SFT target checkpoint.
        config.speculators_config.verifier.name_or_path = str(verifier_checkpoint)

    model = SpeculatorModel.from_pretrained(
        str(converted),
        config=config,
        local_files_only=True,
        dtype=torch.bfloat16,
    )
    model = model.to(device)
    model.eval()
    model.config.use_cache = False
    wrapped = getattr(model.forward, "__wrapped__", None)
    if wrapped is not None:
        import types

        model.forward = types.MethodType(wrapped, model)
    return model


# ---------------------------------------------------------------------------
# EAGLE-3 head (checkpoint remote code)
# ---------------------------------------------------------------------------


def load_eagle3_speculator(
    speculator_id: str,
    device: torch.device,
    revision: str | None = None,
) -> Any:
    """Load a RedHatAI eagle3 checkpoint through its bundled ``eagle3.py``.

    Follows the historical (August 2026) loading path: the remote module is
    imported directly so the checkpoint's own implementation wins over any
    speculators-registry version. Accepts either a hub repo id or a local
    directory (a head saved by this track copies ``eagle3.py`` along).
    """
    from speculators import SpeculatorModel, SpeculatorModelConfig

    local = Path(speculator_id)
    if local.is_dir():
        snapshot = str(local)
    else:
        from huggingface_hub import snapshot_download

        snapshot = snapshot_download(
            repo_id=speculator_id,
            revision=revision,
            local_files_only=True,
        )
    implementation = Path(snapshot) / "eagle3.py"
    if not implementation.is_file():
        raise FileNotFoundError(f"eagle3.py not found in {snapshot}")
    spec = importlib.util.spec_from_file_location(
        "sd_mia_eagle3_remote", implementation
    )
    module = importlib.util.module_from_spec(spec)
    import sys

    sys.modules[spec.name] = module  # pydantic resolves annotations via sys.modules
    # The remote module registers its classes via BOTH the explicit
    # @register decorator and pydantic's speculators_model_type auto hook;
    # neutralise the decorator so the second registration does not raise.
    SpeculatorModel.registry.pop("eagle3", None)
    SpeculatorModelConfig.registry.pop("eagle3", None)
    _identity = lambda *args, **kwargs: (lambda cls: cls)  # noqa: E731
    original_register_model = SpeculatorModel.register
    original_register_config = SpeculatorModelConfig.register
    SpeculatorModel.register = staticmethod(_identity)
    SpeculatorModelConfig.register = staticmethod(_identity)
    try:
        spec.loader.exec_module(module)
    finally:
        SpeculatorModel.register = original_register_model
        SpeculatorModelConfig.register = original_register_config
    if hasattr(module.Eagle3SpeculatorConfig, "model_rebuild"):
        module.Eagle3SpeculatorConfig.model_rebuild(_types_namespace={"torch": torch})
    # transformers 5 calls tie_weights(recompute_mapping=...); the remote
    # module predates that signature
    _original_tie_weights = module.Eagle3Speculator.tie_weights

    def _tie_weights_compat(self, *args: Any, **kwargs: Any) -> Any:
        for unsupported in ("recompute_mapping", "missing_keys"):
            kwargs.pop(unsupported, None)
        return _original_tie_weights(self, *args, **kwargs)

    module.Eagle3Speculator.tie_weights = _tie_weights_compat
    config = module.Eagle3SpeculatorConfig.from_pretrained(
        snapshot, local_files_only=True
    )
    model = module.Eagle3Speculator.from_pretrained(
        snapshot,
        config=config,
        local_files_only=True,
        dtype=torch.bfloat16,
    )
    model.to(device)
    model.eval()
    model.config.use_cache = False
    return model


def eagle3_target_layer_ids(target: Any) -> list[int]:
    """The three verifier layers fused by the EAGLE-3 head (August convention)."""
    config = target.config
    if hasattr(target, "get_base_model"):
        config = target.get_base_model().config
    text = getattr(config, "text_config", config)
    layers = int(getattr(text, "num_hidden_layers"))
    return [2, layers // 2, layers - 3]


def eagle3_connector(target_output: Any, target: Any) -> torch.Tensor:
    """Concatenate the three fused verifier hidden states for the head."""
    layers = eagle3_target_layer_ids(target)
    return torch.cat(
        [target_output.hidden_states[index] for index in layers], dim=-1
    )
