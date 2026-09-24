"""Move lifecycle directories without copying weights or rewriting provenance.

Run ``plan`` to inspect, then ``apply`` while experiment workers are stopped.
The durable journal supports retrying an interrupted move; ``verify`` checks
file identities, metadata hashes and both old/new paths against that journal.
"""
from __future__ import annotations

import argparse
from contextlib import ExitStack
import fcntl
import hashlib
import json
import os
from pathlib import Path

from experiments.paths import ARTIFACTS

JOURNAL = "maintenance/migrations/20260925_lifecycle/PLAN.json"


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n")
    temporary.replace(path)


def sha256(path):
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def proposed_moves(root):
    moves = []

    def add(source, target):
        source, target = root / source, root / target
        if source.is_symlink() or not source.exists():
            return
        if target.exists() or target.is_symlink():
            raise FileExistsError(f"destination already exists: {target}")
        moves.append({"source": str(source), "target": str(target)})

    for source in sorted((root / "runs/training").glob("*")):
        add(source.relative_to(root), f"training/{source.name}/runs")
    for source in sorted((root / "models").glob("*")):
        add(source.relative_to(root), f"training/{source.name}/models")
    for source in sorted((root / "data/splits").glob("*")):
        stage = "audits" if source.name == "resource_curves_v1" else "training"
        add(source.relative_to(root), f"{stage}/{source.name}/splits")
    for source in sorted((root / "runs/audits").glob("*")):
        if source.is_symlink():
            continue
        # Children move first so summaries and logs are siblings of tasks.
        add(source.relative_to(root) / "fixed_only_summary", f"audits/{source.name}/reports")
        add(source.relative_to(root) / "executions", f"audits/{source.name}/executions")
        add(source.relative_to(root), f"audits/{source.name}/tasks")
    for source in sorted((root / "cache/audits").glob("*")):
        add(source.relative_to(root), f"audits/{source.name}/intermediate")
    add("figures", "reports/figures")
    # Keep previous migration evidence together; the current journal is separate.
    add("migrations", "maintenance/legacy_migrations")
    add("branch_backups", "maintenance/branch_backups")
    return moves


def relocated(path, moves):
    path = Path(path)
    for move in sorted(moves, key=lambda m: len(Path(m["source"]).parts), reverse=True):
        try:
            return Path(move["target"]) / path.relative_to(move["source"])
        except ValueError:
            continue
    return path


def inventory(moves):
    files, links = {}, {}
    for move in moves:
        for path in Path(move["source"]).rglob("*"):
            if path.is_symlink():
                links[str(path)] = {"target": str(path.resolve()), "exists": path.exists()}
            elif path.is_file() and str(path) not in files:
                stat = path.stat()
                row = {"device": stat.st_dev, "inode": stat.st_ino,
                       "size": stat.st_size, "mtime_ns": stat.st_mtime_ns}
                # Multi-TB LM weights are checked by inode/size/mtime; every
                # observation, detector, score, passport and report is hashed.
                if path.suffix not in (".safetensors", ".bin"):
                    row["sha256"] = sha256(path)
                files[str(path)] = row
    return files, links


def verify(plan):
    moves = plan["moves"]
    for old, expected in plan["files"].items():
        new = relocated(old, moves)
        stat = new.stat()
        actual = dict(device=stat.st_dev, inode=stat.st_ino, size=stat.st_size, mtime_ns=stat.st_mtime_ns)
        if "sha256" in expected:
            actual["sha256"] = sha256(new)
        if actual != expected or not os.path.samefile(old, new):
            raise ValueError(f"file changed or old path no longer resolves: {old}")
    for old, row in plan["links"].items():
        link = relocated(old, moves)
        if not link.is_symlink() or link.resolve() != relocated(row["target"], moves):
            raise ValueError(f"link changed: {old}")
        if row["exists"] and (not link.exists() or not Path(old).exists()):
            raise ValueError(f"link became dangling: {old}")
    for move in moves:
        if not Path(move["source"]).is_symlink() or not os.path.samefile(move["source"], move["target"]):
            raise ValueError(f"missing compatibility alias: {move['source']}")
    return {"verified_files": len(plan["files"]), "verified_links": len(plan["links"]),
            "moves": len(moves), "weights": "same device/inode/size/mtime; no weight copy",
            "provenance": "original metadata bytes retained; no source re-signing"}


def apply(root):
    root = Path(root).resolve()
    journal = root / JOURNAL
    journal.parent.mkdir(parents=True, exist_ok=True)
    with ExitStack() as stack:
        lock = stack.enter_context((journal.parent / ".lock").open("a"))
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if journal.exists():
            plan = json.loads(journal.read_text())
            if plan["root"] != str(root):
                raise ValueError("migration journal belongs to another artifact root")
        else:
            plan = None
        moves = plan["moves"] if plan else proposed_moves(root)
        if not moves:
            return {"moves": 0}
        # Recheck locks on retries as well as first application. A partially
        # moved workspace can contain locks under either spelling.
        lock_paths = {p.resolve() for move in moves for key in ("source", "target")
                      for p in Path(move[key]).rglob("*.lock") if p.is_file()}
        for path in sorted(lock_paths):
            handle = stack.enter_context(path.open("r"))
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if plan is None:
            files, links = inventory(moves)
            plan = {"schema": "sd_mia_lifecycle_migration_v1", "root": str(root),
                    "moves": moves, "files": files, "links": links}
            write_json(journal, plan)
        moves = plan["moves"]
        for move in moves:
            source, target = Path(move["source"]), Path(move["target"])
            if not target.exists():
                if source.is_symlink():
                    raise ValueError(f"unexpected source alias: {source}")
                target.parent.mkdir(parents=True, exist_ok=True)
                # rename refuses cross-device moves; never copy or delete data.
                source.rename(target)
        # Relative symlinks depend on their containing directory. Rebase them
        # explicitly after moves rather than leaving broken checkpoint paths.
        for old, row in plan["links"].items():
            link = relocated(old, moves)
            destination = relocated(row["target"], moves)
            if link.is_symlink() and link.resolve() == destination:
                continue
            if link.exists() and not link.is_symlink():
                raise FileExistsError(link)
            temporary = link.with_name(link.name + ".lifecycle-link")
            if temporary.is_symlink():
                temporary.unlink()
            temporary.symlink_to(os.path.relpath(destination, link.parent))
            temporary.replace(link)
        for move in moves:
            source, target = Path(move["source"]), Path(move["target"])
            # A child alias belongs in the relocated parent (e.g. tasks/executions).
            alias = relocated(source.parent, moves) / source.name
            if alias.is_symlink():
                if alias.resolve() != target.resolve():
                    raise ValueError(f"conflicting compatibility alias: {alias}")
                continue
            if alias.exists():
                raise FileExistsError(alias)
            alias.parent.mkdir(parents=True, exist_ok=True)
            alias.symlink_to(os.path.relpath(target, alias.parent), target_is_directory=True)
        result = verify(plan)
        write_json(journal.parent / "VERIFIED.json", result)
        return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("plan", "apply", "verify"))
    parser.add_argument("--artifacts-root", type=Path, default=ARTIFACTS)
    args = parser.parse_args()
    root = args.artifacts_root.resolve()
    if args.command == "plan":
        result = proposed_moves(root)
    elif args.command == "apply":
        result = apply(root)
    else:
        result = verify(json.loads((root / JOURNAL).read_text()))
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
