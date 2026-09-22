"""Dedicated, locked outputs with checked identities and atomic completion."""
from contextlib import contextmanager
import fcntl
import hashlib
import json
import os
from pathlib import Path
import tempfile

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
RUN_ROOT = ROOT / "artifacts/runs/audits/resource_curves_v1"
CACHE_ROOT = ROOT / "artifacts/cache/audits/resource_curves_v1"
DATA_ROOT = ROOT / "artifacts/data/splits/resource_curves_v1"


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, allow_nan=False).encode()).hexdigest()


def file_sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def code_fingerprint():
    from experiments.sd_membership_sft.audit.matrix_artifacts import runtime_files

    files = runtime_files() + sorted(p for p in Path(__file__).parent.rglob("*.py")
                                     if "tests" not in p.parts)
    return digest({str(p.relative_to(ROOT)): file_sha(p) for p in files})


def atomic_json(path, value):
    path = Path(path)
    fd, temporary = tempfile.mkstemp(prefix=".resource-", dir=path.parent)
    try:
        with os.fdopen(fd, "w") as stream:
            json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
            stream.write("\n")
        os.replace(temporary, path)
    finally:
        Path(temporary).unlink(missing_ok=True)


def atomic_npz(path, arrays):
    path = Path(path)
    fd, temporary = tempfile.mkstemp(prefix=".resource-", dir=path.parent)
    try:
        with os.fdopen(fd, "wb") as stream:
            np.savez_compressed(stream, **arrays)
        os.replace(temporary, path)
    finally:
        Path(temporary).unlink(missing_ok=True)


@contextmanager
def workspace(folder):
    """Refuse an existing unrelated directory, including through legacy symlinks."""
    folder = Path(folder).resolve()
    if folder.is_relative_to(ROOT / "experiments"):
        raise ValueError("resource outputs must not be stored in experiment source directories")
    artifacts = ROOT / "artifacts"
    if folder.is_relative_to(artifacts) and not any(
        folder.is_relative_to(root) for root in (RUN_ROOT, CACHE_ROOT, DATA_ROOT)
    ):
        raise ValueError("use a dedicated resource_curves_v1 artifact root")
    folder.mkdir(parents=True, exist_ok=True)
    marker = folder / "RESOURCE_CURVES.json"
    expected = {"schema": "resource_curves_workspace_v1"}
    if not marker.exists():
        if any(folder.iterdir()):
            raise ValueError("refusing to adopt a nonempty, unrelated output directory")
        # Exclusive creation also prevents simultaneous first-time adoption.
        with marker.open("x") as stream:
            json.dump(expected, stream)
    if json.loads(marker.read_text()) != expected:
        raise ValueError("invalid resource workspace marker")
    with (folder / ".lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        try:
            yield folder
        finally:
            fcntl.flock(lock, fcntl.LOCK_UN)


def checked_contract(folder, name, contract):
    path = Path(folder) / name
    if path.exists():
        if json.loads(path.read_text()) != contract:
            raise ValueError(f"{name}: resume parameters or sources changed; use another directory")
    else:
        atomic_json(path, contract)
