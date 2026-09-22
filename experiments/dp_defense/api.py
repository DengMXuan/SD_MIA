"""Callable DP preparation and training for all registered model pairs."""
import json
from pathlib import Path

from experiments.cross_model_audit.model_registry import identify_pair


def _trainer(reference_run):
    artifact = json.loads((reference_run / "results.json").read_text())
    if "privacy" in artifact or (reference_run / "DP_REQUEST.json").exists():
        raise ValueError("use the ordinary reference recipe; DP models start from pinned public bases")
    spec = identify_pair(artifact)
    if spec.adapter == "plain":
        from . import train
        return train
    from . import head_train
    return head_train


def plan_private_training(reference_run, output_dir, *, epsilon, max_grad_norm=1., gpu=0):
    """Return a provenance-bound request without writing files or loading models.

Epsilon caps apply separately to target and member adaptation. The auxiliary
deployment inherits the target budget; the member deployment composes both.
"""
    if type(gpu) is not int or gpu < 0:
        raise ValueError("GPU must be a nonnegative logical device index")
    reference_run, output = Path(reference_run).resolve(), Path(output_dir).resolve()
    return _trainer(reference_run).prepare_request(reference_run, output, epsilon, max_grad_norm, gpu)[-1]


def train_private(reference_run, output_dir, *, epsilon, max_grad_norm=1., gpu=0):
    """Train/resume one condition from public initializations and return its passport.

This explicitly starts training. It never launches a sweep or modifies the
ordinary reference checkpoint. Completed stages are reused after verification.
"""
    if type(gpu) is not int or gpu < 0:
        raise ValueError("GPU must be a nonnegative logical device index")
    reference_run, output = Path(reference_run).resolve(), Path(output_dir).resolve()
    _trainer(reference_run).run(reference_run, output, epsilon, max_grad_norm, gpu)
    from .artifacts import verify_run
    return verify_run(output)
