"""Experiment lifecycle paths. See docs/artifact_layout.md for the storage contract.

Path calculation is read-only. Explicit prepare calls create compatibility slots
for algorithms that still address weights/observations by their original names.
"""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ARTIFACTS = ROOT / "artifacts"
DATA = ARTIFACTS / "data"
TRAINING_ROOT = ARTIFACTS / "training"
AUDITS = ARTIFACTS / "audits"
REPORTS = ARTIFACTS / "reports"
MAINTENANCE = ARTIFACTS / "maintenance"
TRAINING = TRAINING_ROOT / "controlled_sft_v2/runs"
MODELS = TRAINING_ROOT / "controlled_sft_v2/models"
SPLITS = TRAINING_ROOT / "controlled_sft_v2/splits"
QWEN_MODELS = TRAINING / "model_pairs/qwen3"
QWEN_AUDIT = AUDITS / "qwen_fixed_v1/tasks"
QWEN_CURRENT_AUDIT = AUDITS / "qwen_shared_reference_v1/tasks"
QWEN_CACHE = AUDITS / "qwen_fixed_v1/intermediate"
ARCHIVE = ARTIFACTS / "archive"


def prepare_training_storage(run_dir: Path) -> None:
    """Store weights beside the batch's runs, keeping custom runs self-contained."""
    run_dir = Path(run_dir).resolve()
    try:
        batch, slot, *condition = run_dir.relative_to(ARTIFACTS / "training").parts
    except ValueError:
        return
    if slot != "runs":
        return
    run_dir.mkdir(parents=True, exist_ok=True)
    for name in ("checkpoints", "heads", "adapters"):
        target = ARTIFACTS / "training" / batch / "models" / Path(*condition) / name
        target.mkdir(parents=True, exist_ok=True)
        link = run_dir / name
        if not link.exists() and not link.is_symlink():
            link.symlink_to(target, target_is_directory=True)


def _audit_slot(output: Path, name: str) -> Path:
    output = Path(output).resolve()
    try:
        batch, slot, *condition = output.relative_to(ARTIFACTS / "audits").parts
    except ValueError:
        return output / name
    if slot != "tasks":
        return output / name
    return ARTIFACTS / "audits" / batch / name / Path(*condition)


def audit_cache(task_output: Path) -> Path:
    return _audit_slot(task_output, "intermediate")


def audit_reports(output_root: Path) -> Path:
    return _audit_slot(output_root, "reports")


def audit_executions(output_root: Path) -> Path:
    return _audit_slot(output_root, "executions")


def prepare_audit_cache(task_output: Path) -> None:
    """Expose stable filenames while intermediates occupy a separate directory.

    Existing slots are retained; moving existing data is an explicit maintenance
    operation, never a side effect of launching an experiment.
    """
    output = Path(task_output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    destination = audit_cache(output)
    destination.mkdir(parents=True, exist_ok=True)
    for name in ("trajectories", "observations.npz", "observations.npz.json", "detector.pt", "FIT.json"):
        link = output / name
        if link.is_symlink() or link.exists():
            continue
        if name == "trajectories":
            (destination / name).mkdir(exist_ok=True)
        link.symlink_to(destination / name, target_is_directory=name == "trajectories")
