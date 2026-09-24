"""One coordinator for GPU locks, durable attempts and interrupted-worker cleanup.

Experiment adapters supply readiness checks and their worker entry point. The
scheduler owns dispatch, persistence and cancellation for every model family.
"""
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
import fcntl
import json
import os
from pathlib import Path
import queue
import subprocess
import sys
import threading
import time
import uuid
from experiments.paths import ROOT, audit_executions
from experiments.shared.core.audit_runtime import _write_json
from .artifacts import digest

def wait_worker(process, stop, heartbeat):
    """Stop the child before releasing its GPU lock, including on interruption."""
    last_update = time.monotonic()
    try:
        while not stop.is_set():
            try:
                return process.wait(timeout=1)
            except subprocess.TimeoutExpired:
                if time.monotonic() - last_update >= 30:
                    heartbeat()
                    last_update = time.monotonic()
        return None
    finally:
        if process.poll() is None:
            process.terminate()
            try:
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()


def inspect_task(task, *, ready, read_result, check_sources=True):
    worker_lock = Path(task["output"]) / ".worker.lock"
    if worker_lock.exists():
        with worker_lock.open("r") as lock:
            try:
                fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError:
                return {"state": "running", "completed_methods": [], "reason": "worker holds task lock"}
    completed, errors = [], []
    for method in task["methods"]:
        folder = Path(task["output"]) / method
        if not (folder / "REPORT.json").exists():
            continue
        try:
            read_result(folder, digest({"task": task, "method": method}), check_sources=check_sources)
            completed.append(method)
        except (OSError, ValueError, KeyError) as error:
            errors.append(f"{method}: {error}")
    if errors:
        return {"state": "stale", "completed_methods": completed, "reason": "; ".join(errors)}
    if len(completed) == len(task["methods"]):
        return {"state": "complete", "completed_methods": completed, "reason": "checked result hashes and sources"}
    valid, reason = ready(task)
    return {"state": "ready" if valid else ("pending" if reason.startswith("pending") else "invalid"),
            "completed_methods": completed, "reason": reason}


def run_tasks(tasks, output_root, gpus, *, inspect_task, check_gpus, worker_module, wait=None):
    wait = wait or wait_worker
    output_root.mkdir(parents=True, exist_ok=True)
    with (output_root / ".matrix.lock").open("a") as matrix_lock:
        fcntl.flock(matrix_lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        states = {task["id"]: inspect_task(task) for task in tasks}
        _write_json(output_root / "STATUS.json", states)
        runnable = [task for task in tasks if states[task["id"]]["state"] == "ready"]
        if not runnable:
            return all(row["state"] == "complete" for row in states.values())
        check_gpus(gpus)
        locks = []
        try:
            for gpu in gpus:
                lock = Path(f"/tmp/sd_mia_audit_gpu_{os.getuid()}_{gpu}.lock").open("a")
                locks.append(lock)
                fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            jobs = queue.Queue()
            for task in runnable:
                jobs.put(task)
            control = audit_executions(output_root)
            control.mkdir(parents=True, exist_ok=True)
            stop = threading.Event()
            def worker(gpu):
                while not stop.is_set():
                    try:
                        task = jobs.get_nowait()
                    except queue.Empty:
                        return
                    check_gpus([gpu])
                    if stop.is_set():
                        return
                    attempt = control / (datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S") + "-" + uuid.uuid4().hex[:12])
                    attempt.mkdir()
                    _write_json(attempt / "TASK.json", task)
                    env = {**os.environ, "CUDA_VISIBLE_DEVICES": str(gpu), "HF_HUB_OFFLINE": "1",
                           "TRANSFORMERS_OFFLINE": "1", "TOKENIZERS_PARALLELISM": "false",
                           "OMP_NUM_THREADS": "2", "PYTHONHASHSEED": str(task["settings"]["audit_seed"])}
                    started = time.perf_counter()
                    _write_json(attempt / "STATUS.json", {"state": "running", "task": task["id"], "gpu": gpu})
                    print(json.dumps({"started": task["id"], "gpu": gpu, "log": str(attempt / "worker.log")}), flush=True)
                    with (attempt / "worker.log").open("w") as log:
                        process = subprocess.Popen(
                            [sys.executable, "-u", "-m", worker_module,
                             "worker", "--task-file", str(attempt / "TASK.json")],
                            cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT)
                        code = wait(process, stop, lambda: print(json.dumps(
                            {"running": task["id"], "gpu": gpu,
                             "elapsed_seconds": round(time.perf_counter() - started)}), flush=True))
                    state = {"state": "interrupted" if code is None else ("complete" if code == 0 else "failed"), "task": task["id"], "gpu": gpu,
                             "exit_code": code, "worker_wall_seconds": time.perf_counter() - started}
                    _write_json(attempt / "STATUS.json", state)
                    print(json.dumps(state), flush=True)
                    jobs.task_done()
            with ThreadPoolExecutor(max_workers=len(gpus)) as executor:
                futures = [executor.submit(worker, gpu) for gpu in gpus]
                try:
                    for future in as_completed(futures):
                        future.result()
                except BaseException:
                    stop.set()
                    raise
        finally:
            for lock in locks:
                lock.close()
        states = {task["id"]: inspect_task(task) for task in tasks}
        _write_json(output_root / "STATUS.json", states)
        return all(row["state"] == "complete" for row in states.values())
