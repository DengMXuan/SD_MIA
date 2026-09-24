"""Crash-recoverable storage for a pool JSONL file and its hash manifest.

The pool and manifest are two filesystem entries, so one rename cannot publish
both. Writers therefore leave a small transaction journal before replacing
either file. Repository readers take the same lock and complete an interrupted
commit before verifying the pair.
"""

from __future__ import annotations

from contextlib import contextmanager
import fcntl
import hashlib
import json
import os
from pathlib import Path
from typing import Any, Iterator
import uuid


TRANSACTION_SCHEMA_VERSION = 1


def _sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _manifest_path(path: Path) -> Path:
    return path.with_suffix(".manifest.json")


def _journal_path(path: Path) -> Path:
    return path.with_name(f".{path.name}.transaction.json")


def _lock_path(path: Path) -> Path:
    return path.with_name(f".{path.name}.lock")


@contextmanager
def _pool_lock(path: Path) -> Iterator[None]:
    path.parent.mkdir(parents=True, exist_ok=True)
    with _lock_path(path).open("a+b") as handle:
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


def _write_bytes_durable(path: Path, payload: bytes) -> None:
    with path.open("wb") as handle:
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _safe_staged_path(path: Path, name: str) -> Path:
    candidate = path.parent / name
    if candidate.parent != path.parent or not candidate.name.startswith(f".{path.name}."):
        raise RuntimeError("invalid pool transaction staging path")
    return candidate


def _complete_transaction_unlocked(path: Path) -> None:
    journal_path = _journal_path(path)
    if not journal_path.exists():
        return
    journal = json.loads(journal_path.read_text(encoding="utf-8"))
    if (
        journal.get("schema_version") != TRANSACTION_SCHEMA_VERSION
        or journal.get("output_name") != path.name
    ):
        raise RuntimeError(f"Invalid pool transaction journal: {journal_path}")
    staged_data = _safe_staged_path(path, str(journal["staged_data_name"]))
    staged_manifest = _safe_staged_path(path, str(journal["staged_manifest_name"]))
    expected_sha = str(journal["jsonl_sha256"])

    if staged_data.exists():
        if _sha256(staged_data.read_bytes()) != expected_sha:
            raise RuntimeError("Staged pool data does not match its transaction hash")
        os.replace(staged_data, path)
    if not path.exists() or _sha256(path.read_bytes()) != expected_sha:
        raise RuntimeError("Interrupted pool transaction has no valid data payload")

    if staged_manifest.exists():
        staged = json.loads(staged_manifest.read_text(encoding="utf-8"))
        if staged.get("jsonl_sha256") != expected_sha:
            raise RuntimeError("Staged pool manifest does not match transaction data")
        os.replace(staged_manifest, _manifest_path(path))
    manifest_path = _manifest_path(path)
    if not manifest_path.exists():
        raise RuntimeError("Interrupted pool transaction has no valid manifest")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("jsonl_sha256") != expected_sha:
        raise RuntimeError("Published pool manifest does not match transaction data")

    journal_path.unlink()
    _fsync_directory(path.parent)


def recover_pool_transaction(path: Path) -> None:
    """Finish a commit interrupted between the data and manifest renames."""
    path = Path(path)
    with _pool_lock(path):
        _complete_transaction_unlocked(path)


def write_pool_pair(path: Path, payload: bytes, manifest: dict[str, Any]) -> None:
    """Publish a matching data/manifest pair with journal-based recovery."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    expected_sha = _sha256(payload)
    if manifest.get("jsonl_sha256") != expected_sha:
        raise ValueError("manifest jsonl_sha256 does not match pool payload")
    manifest_payload = json.dumps(
        manifest, indent=2, ensure_ascii=False
    ).encode("utf-8")

    with _pool_lock(path):
        _complete_transaction_unlocked(path)
        token = uuid.uuid4().hex
        staged_data = path.with_name(f".{path.name}.{token}.data.tmp")
        staged_manifest = path.with_name(f".{path.name}.{token}.manifest.tmp")
        staged_journal = path.with_name(f".{path.name}.{token}.journal.tmp")
        journal_path = _journal_path(path)
        journal = {
            "schema_version": TRANSACTION_SCHEMA_VERSION,
            "output_name": path.name,
            "staged_data_name": staged_data.name,
            "staged_manifest_name": staged_manifest.name,
            "jsonl_sha256": expected_sha,
        }
        try:
            _write_bytes_durable(staged_data, payload)
            _write_bytes_durable(staged_manifest, manifest_payload)
            _write_bytes_durable(
                staged_journal,
                (json.dumps(journal, indent=2) + "\n").encode("utf-8"),
            )
            os.replace(staged_journal, journal_path)
            _fsync_directory(path.parent)
            _complete_transaction_unlocked(path)
        except BaseException:
            # Keep the journal and staged files: the next repository reader can
            # finish a commit even after a hard process interruption.
            raise


def read_verified_pool(
    path: Path, benchmark: str
) -> tuple[list[dict[str, Any]], dict[str, Any], bytes]:
    """Recover, then verify a pool's hash, benchmark, row count, and JSON."""
    path = Path(path)
    with _pool_lock(path):
        _complete_transaction_unlocked(path)
        manifest_path = _manifest_path(path)
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        payload = path.read_bytes()
        if _sha256(payload) != manifest.get("jsonl_sha256"):
            raise RuntimeError(f"Pool hash does not match its manifest: {path}")
        if manifest.get("benchmark") != benchmark:
            raise RuntimeError(
                f"Pool benchmark mismatch at {path}: declares "
                f"{manifest.get('benchmark')!r}, expected {benchmark!r}"
            )
        records = [
            json.loads(line)
            for line in payload.decode("utf-8").splitlines()
            if line
        ]
        if int(manifest.get("records", len(records))) != len(records):
            raise RuntimeError(f"Pool record count does not match its manifest: {path}")
        return records, manifest, payload
