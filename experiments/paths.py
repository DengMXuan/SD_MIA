"""Canonical storage layout; legacy paths remain filesystem aliases after migration."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ARTIFACTS = ROOT / "artifacts"
DATA = ARTIFACTS / "data"
MODELS = ARTIFACTS / "models"
TRAINING = ARTIFACTS / "runs/training/controlled_sft_v2"
SPLITS = DATA / "splits/controlled_sft_v2"
QWEN_MODELS = TRAINING / "model_pairs/qwen3"
QWEN_AUDIT = ARTIFACTS / "runs/audits/qwen_fixed_v1"
QWEN_CACHE = ARTIFACTS / "cache/audits/qwen_fixed_v1"
ARCHIVE = ARTIFACTS / "archive"


def prepare_training_storage(run_dir: Path) -> None:
    """Keep weight directories out of training passports/logs for canonical runs."""
    run_dir = Path(run_dir).resolve()
    try:
        relative = run_dir.relative_to(ARTIFACTS / "runs/training")
    except ValueError:
        return
    run_dir.mkdir(parents=True, exist_ok=True)
    for name in ("checkpoints", "heads", "adapters"):
        target = MODELS / relative / name
        target.mkdir(parents=True, exist_ok=True)
        link = run_dir / name
        if not link.exists() and not link.is_symlink():
            link.symlink_to(target, target_is_directory=True)


def audit_cache(task_output: Path) -> Path:
    """Default matrix caches are separate; custom output roots stay self-contained."""
    output = Path(task_output).resolve()
    try:
        return QWEN_CACHE / output.relative_to(QWEN_AUDIT)
    except ValueError:
        return output / "cache"


def prepare_audit_cache(task_output: Path) -> None:
    """Link fixed cache slots without changing the algorithms' file contract."""
    output = Path(task_output)
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
