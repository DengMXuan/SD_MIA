"""Storage owned exclusively by the opt-in cross-model runner."""
from pathlib import Path
from experiments.paths import ARTIFACTS

OUTPUT_ROOT = ARTIFACTS / 'runs/audits/cross_model_fixed_v1'
CACHE_ROOT = ARTIFACTS / 'cache/audits/cross_model_fixed_v1'


def prepare_audit_cache(output):
    output = Path(output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    try:
        destination = CACHE_ROOT / output.relative_to(OUTPUT_ROOT)
    except ValueError:
        destination = output / 'cache'
    destination.mkdir(parents=True, exist_ok=True)
    for name in ('trajectories', 'observations.npz', 'observations.npz.json', 'detector.pt', 'FIT.json'):
        link = output / name
        if link.is_symlink() or link.exists():
            continue
        if name == 'trajectories':
            (destination / name).mkdir(exist_ok=True)
        link.symlink_to(destination / name, target_is_directory=name == 'trajectories')
