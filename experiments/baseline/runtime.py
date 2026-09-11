"""Durable per-execution progress and completed-method artifacts."""
from __future__ import annotations

import json
import os
import time
import traceback
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import numpy as np

from .costs import write_cost_report


class RunProgress:
    def __init__(self, output_dir: Path, interval: float = 30.0):
        if interval <= 0:
            raise ValueError("progress interval must be positive")
        run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "-" + uuid.uuid4().hex[:8]
        self.directory = output_dir / "executions" / run_id
        self.directory.mkdir(parents=True)
        self.interval = interval
        self.started = time.monotonic()
        self.completed: list[str] = []
        self.costs: dict = {}
        self.active_method = None
        self.log = (self.directory / "progress.jsonl").open("a", encoding="utf-8")

    def event(self, stage: str, **fields: Any) -> None:
        event = dict(timestamp=datetime.now(timezone.utc).isoformat(),
                     elapsed_seconds=round(time.monotonic() - self.started, 3),
                     stage=stage, completed_methods=list(self.completed), **fields)
        if self.active_method is not None:
            event.setdefault("method", self.active_method)
        line = json.dumps(event, ensure_ascii=False)
        self.log.write(line + "\n")
        self.log.flush()
        print(line, flush=True)
        temporary = self.directory / "status.json.tmp"
        temporary.write_text(line + "\n", encoding="utf-8")
        temporary.replace(self.directory / "status.json")

    def configure(self, protocol, labels, record_ids):
        self.protocol = protocol
        self.labels = np.asarray(labels)
        self.record_ids = np.asarray(record_ids)
        if not len(self.labels):
            raise ValueError("audit selection is empty")

    def track(self, values: Iterable, stage: str, unit: str = "records"):
        total = len(values)
        start = last = time.monotonic()
        self.event(stage, completed=0, total=total, unit=unit)
        for index, value in enumerate(values, 1):
            yield value
            now = time.monotonic()
            if now - last >= self.interval or index == total:
                elapsed = now - start
                self.event(stage, completed=index, total=total, unit=unit,
                           stage_seconds=round(elapsed, 3),
                           rate_per_second=index / max(elapsed, 1e-9),
                           eta_seconds=elapsed / index * (total - index))
                last = now

    def scores(self, methods):
        return {name: _MethodScores(self, name) for name in methods}

    def save_method(self, name, values, cost=None):
        # One atomic NPZ is the complete recovery unit; readers never see a
        # half-written score vector or metadata from another execution.
        temporary = self.directory / f"{name}.npz.tmp"
        destination = self.directory / f"{name}.npz"
        protocol = {**self.protocol, "methods": [name]}
        extra = {"cost_json": np.asarray(json.dumps(cost))} if cost is not None else {}
        with temporary.open("wb") as stream:
            np.savez_compressed(stream, labels=self.labels, record_ids=self.record_ids,
                                protocol_json=np.asarray(json.dumps(protocol)), **extra,
                                **{name: np.asarray(values, dtype=np.float32)})
            stream.flush()
            os.fsync(stream.fileno())
        temporary.replace(destination)
        self.completed.append(name)
        if cost is not None:
            self.costs[name] = cost
            write_cost_report(self.directory, self.protocol, self.costs)
        self.event("method_saved", method=name, artifact=str(destination), cost=cost)

    def __enter__(self):
        self.event("started", pid=os.getpid(), artifact_directory=str(self.directory))
        return self

    def __exit__(self, exc_type, exc, tb):
        try:
            if exc is None:
                self.event("completed")
            else:
                self.event("failed", error_type=exc_type.__name__, error=str(exc),
                           traceback="".join(traceback.format_exception(exc_type, exc, tb)))
        finally:
            self.log.close()
        return False


class _MethodScores(list):
    def __init__(self, progress: RunProgress, name: str):
        super().__init__()
        self.progress = progress
        self.name = name

    def append(self, value):
        if len(self) >= len(self.progress.labels):
            raise ValueError(f"too many scores for {self.name}")
        super().append(value)
        if len(self) == len(self.progress.labels):
            self.progress.save_method(self.name, self)
