"""Plan, run, inspect and summarize the 36-configuration Qwen audit matrix."""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import csv
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

import numpy as np
from safetensors import SafetensorError

from experiments.baseline import METHODS
from experiments.sd_membership_sft.core.audit_runtime import ROOT, _write_json
from experiments.sd_membership_sft.core.deployment_archive import sha256_file
from experiments.sd_membership_sft.audit.matrix_artifacts import digest, read_result, sources_for
from experiments.sd_membership_sft.audit.matrix_baselines import BASELINE_DEFAULTS
from experiments.sd_membership_sft.audit.matrix_main import MAIN_METHODS

ROLES = ("draft_auxiliary_distilled", "draft_member_sft")
ALL_METHODS = (*MAIN_METHODS["fixed"], *MAIN_METHODS["natural"], *METHODS)
from experiments.paths import QWEN_MODELS as DEFAULT_MODEL_ROOT, QWEN_AUDIT as DEFAULT_OUTPUT_ROOT


def make_tasks(model_root, output_root, benchmarks, epochs, seeds, settings):
    tasks = []
    for benchmark in benchmarks:
        for epoch in epochs:
            for seed in seeds:
                condition = dict(benchmark=benchmark, epoch=epoch, condition_seed=seed)
                key = f"{benchmark}/epoch{epoch}/seed{seed}"
                base = dict(run_dir=str((model_root / key).resolve()), condition=condition, settings=settings)
                tasks.append({**base, "id": key + "/baseline", "kind": "baseline", "methods": list(METHODS),
                              "output": str((output_root / key / "baseline").resolve())})
                for role in ROLES:
                    for protocol in MAIN_METHODS:
                        suffix = f"{role}/{protocol}"
                        tasks.append({**base, "id": key + "/" + suffix, "kind": "main", "draft_role": role,
                                      "protocol": protocol, "methods": list(MAIN_METHODS[protocol]),
                                      "output": str((output_root / key / suffix).resolve())})
    return tasks


def ready(task):
    """Read-only preflight; complete passports, shards and shared split required."""
    run = Path(task["run_dir"])
    if not (run / "results.json").is_file():
        return False, "pending: training passport is not ready"
    try:
        artifact = json.loads((run / "results.json").read_text())
        cfg = artifact["config"]
        condition = task["condition"]
        if artifact["material_passport"]["status"] != "COMPLETED":
            return False, "pending: training has not completed"
        expected = dict(benchmark=condition["benchmark"], target_epochs=condition["epoch"],
                        seed=condition["condition_seed"], data_seed=condition["condition_seed"],
                        target_model="Qwen/Qwen3-8B-Base", draft_model="Qwen/Qwen3-1.7B-Base")
        if any(cfg.get(key) != value for key, value in expected.items()):
            raise ValueError("training condition/model identity mismatch")
        records = artifact["records"]
        for name, count in (("members", 2000), ("nonmembers", 2000), ("auxiliary", 2000), ("audit_auxiliary", 600)):
            if len(records[name]) != count:
                raise ValueError(f"wrong {name} count")
        ids = [row["record_id"] for rows in records.values() for row in rows]
        if len(set(ids)) != len(ids):
            raise ValueError("training/audit record IDs overlap")
        manifest = Path(artifact["data"]["shared_split_manifest"])
        manifest = manifest if manifest.is_absolute() else ROOT / manifest
        checksum = sha256_file(manifest)
        if checksum != artifact["data"]["shared_split_sha256"]:
            raise ValueError("shared split differs from training")
        audit = json.loads(manifest.with_suffix(".audit.json").read_text())
        token_source = f"{cfg['draft_model']}@{cfg['draft_revision']}"
        attestation = audit["tokenizers"][token_source]
        if attestation["shared_split_sha256"] != checksum or attestation["cross_split_ngram_audit"]["gate"] != "PASS":
            raise ValueError("shared split tokenizer audit is stale")
        roles = ("target",) if task["kind"] == "baseline" else ("target", task["draft_role"])
        from safetensors import safe_open
        for role in roles:
            folder = run / "checkpoints" / role
            if not (folder / "config.json").is_file():
                return False, f"pending: {role} config missing"
            shards = list(folder.glob("*.safetensors"))
            if not shards:
                return False, f"pending: {role} weights missing"
            index = folder / "model.safetensors.index.json"
            if index.exists():
                expected_shards = set(json.loads(index.read_text())["weight_map"].values())
                if not expected_shards.issubset({p.name for p in shards}):
                    return False, f"pending: {role} shards incomplete"
            for shard in shards:
                with safe_open(shard, framework="np") as weights:
                    if not list(weights.keys()):
                        raise ValueError(f"empty checkpoint {shard}")
        return True, "ready"
    except (OSError, ValueError, KeyError, TypeError, SafetensorError) as error:
        return False, f"invalid: {error}"


def inspect_task(task, *, check_sources=True):
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


def check_gpus(gpus):
    if not gpus or len(set(gpus)) != len(gpus) or any(g < 0 for g in gpus):
        raise ValueError("choose distinct nonnegative physical GPUs")
    inventory = subprocess.check_output(
        ["nvidia-smi", "--query-gpu=index,uuid", "--format=csv,noheader,nounits"], text=True)
    available = {int(line.split(",")[0]): line.split(",")[1].strip() for line in inventory.splitlines() if line.strip()}
    apps = subprocess.check_output(
        ["nvidia-smi", "--query-compute-apps=gpu_uuid,pid", "--format=csv,noheader,nounits"], text=True)
    busy = {line.split(",")[0].strip() for line in apps.splitlines() if line.strip()}
    for gpu in gpus:
        if gpu not in available or available[gpu] in busy:
            raise RuntimeError(f"GPU {gpu} is absent or has an active compute process")


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


def run_tasks(tasks, output_root, gpus):
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
            control = output_root / "executions"
            control.mkdir(exist_ok=True)
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
                            [sys.executable, "-u", "-m", "experiments.sd_membership_sft.audit.qwen_audit_matrix",
                             "worker", "--task-file", str(attempt / "TASK.json")],
                            cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT)
                        code = wait_worker(process, stop, lambda: print(json.dumps(
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


def summarize(tasks, output_root):
    results, errors = {}, []
    for task in tasks:
        for method in task["methods"]:
            folder = Path(task["output"]) / method
            if not (folder / "REPORT.json").exists():
                continue
            try:
                results[(task["id"], method)] = read_result(folder, digest({"task": task, "method": method}))
            except (OSError, ValueError, KeyError) as error:
                errors.append({"task": task["id"], "method": method, "error": str(error)})
    rows, groups, physical_groups = [], {}, {}
    baseline_tasks = {tuple(task["condition"].values()): task for task in tasks if task["kind"] == "baseline"}
    for key, baseline in baseline_tasks.items():
        condition = baseline["condition"]
        matching = [task for task in tasks if task["condition"] == condition]
        # Every method/branch must have the same ordered calibration and test IDs.
        id_hashes = {r["record_ids_sha256"] for task in matching for method in task["methods"]
                     if (r := results.get((task["id"], method))) is not None}
        if len(id_hashes) > 1:
            raise ValueError(f"methods scored different records in {condition}")
        for role in ROLES:
            for method in ALL_METHODS:
                task = baseline if method in METHODS else next(t for t in matching if t.get("draft_role") == role and method in t["methods"])
                report = results.get((task["id"], method))
                row = {**condition, "draft_role": role, "method": method,
                       "status": "complete" if report else "missing", "source_task": task["id"],
                       "reused_target_only": method in METHODS}
                if report:
                    row.update(report["metrics"])
                    row.update(report["cost"])
                    row["access_channel"] = report["access_channel"]
                    row["report"] = str(Path(task["output"]) / method / "REPORT.json")
                    group = report["cost"]["execution_group"]
                    seconds = report["cost"].get("execution_group_seconds")
                    if seconds is None:
                        seconds = report["cost"]["total_seconds"]
                    physical_groups[group] = max(physical_groups.get(group, 0.), seconds)
                rows.append(row)
                group_key = (condition["benchmark"], condition["epoch"], role, method)
                groups.setdefault(group_key, []).append(row)
    aggregates = []
    for (benchmark, epoch, role, method), values in groups.items():
        completed = [row for row in values if row["status"] == "complete"]
        entry = dict(benchmark=benchmark, epoch=epoch, draft_role=role, method=method,
                     expected_seeds=len(values), completed_seeds=len(completed))
        fields = set.intersection(*(set(r) for r in completed)) if completed else set()
        for field in sorted(fields - {"epoch", "condition_seed", "reused_target_only"}):
            if all(isinstance(r[field], (int, float)) and not isinstance(r[field], bool) for r in completed):
                numbers = [r[field] for r in completed]
                entry[field + "_mean"] = float(np.mean(numbers))
                entry[field + "_std"] = float(np.std(numbers, ddof=1)) if len(numbers) > 1 else None
        aggregates.append(entry)
    output_root.mkdir(parents=True, exist_ok=True)
    for name, table in (("RESULTS", rows), ("SEED_SUMMARY", aggregates)):
        fields = list(dict.fromkeys(key for row in table for key in row))
        temporary = output_root / f".{name}.tmp.csv"
        with temporary.open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=fields)
            writer.writeheader()
            writer.writerows(table)
        temporary.replace(output_root / f"{name}.csv")
    completed_count = sum(r["status"] == "complete" for r in rows)
    attempts = []
    for path in sorted((output_root / "executions").glob("*/STATUS.json")):
        attempts.append(json.loads(path.read_text()))
    shared_costs = any(row.get("cost_basis") == "physical_incremental" for row in rows)
    note = "baseline display duplication is not independent evidence; worker time includes loading/retries and is not matrix elapsed wall time"
    if shared_costs:
        note += "; shared-reference rows expose physical incremental fields and leave standalone method cost columns empty"
    payload = dict(complete=completed_count == len(rows) and not errors, expected_rows=len(rows),
                   completed_rows=completed_count, errors=errors, rows=rows, seed_summary=aggregates,
                   unique_successful_execution_groups=len(physical_groups),
                   unique_successful_measured_method_seconds=sum(physical_groups.values()),
                   attempted_worker_wall_seconds_sum=sum(r.get("worker_wall_seconds", 0.) for r in attempts),
                   note=note)
    _write_json(output_root / "SUMMARY.json", payload)
    lines = ["# Qwen audit matrix", "", f"Completed rows: {completed_count}/{len(rows)}. Complete: {payload['complete']}.", "",
             "ROC and independently calibrated TPR are separate. pAUC below is area/0.10; raw area is retained in CSV/JSON.", "",
             "Shared-reference rows without standalone cost show — in ms/record; their measured incremental costs use physical_incremental_* fields." if shared_costs else "", "",
             "| Dataset | Epoch | Seed | Draft | Method | AUC | pAUC10 norm | ROC TPR10 | ROC TPR1 | Cal TPR1 | Cal FPR1 | ms/record | Status |",
             "|---|---:|---:|---|---|---:|---:|---:|---:|---:|---:|---:|---|"]
    for row in rows:
        numbers = [f"{row[key]:.4f}" if row.get(key) is not None else "—" for key in
                   ("auc", "pauc_10_normalized", "roc_tpr_at_10pct_fpr", "roc_tpr_at_1pct_fpr",
                    "calibrated_tpr_at_1pct", "calibrated_actual_fpr_at_1pct", "amortized_ms_per_record")]
        lines.append("| " + " | ".join([row["benchmark"], str(row["epoch"]), str(row["condition_seed"]),
                                          row["draft_role"], row["method"], *numbers, row["status"]]) + " |")
    temporary = output_root / ".RESULTS.tmp.md"
    temporary.write_text("\n".join(lines) + "\n")
    temporary.replace(output_root / "RESULTS.md")
    return payload


def execute_worker(task):
    import torch
    from experiments.sd_membership_sft.protocols.protocol_models import prepare_records
    from experiments.sd_membership_sft.audit.matrix_baselines import run_baselines
    from experiments.sd_membership_sft.audit.matrix_main import run_main

    torch.set_num_threads(2)
    if not torch.cuda.is_available():
        raise RuntimeError("matrix worker requires a visible GPU")
    output = Path(task["output"])
    output.mkdir(parents=True, exist_ok=True)
    with (output / ".worker.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        valid, reason = ready(task)
        if not valid:
            raise RuntimeError(reason)
        cfg, prepared = prepare_records(Path(task["run_dir"]), "plain")
        roles = ["target"] if task["kind"] == "baseline" else ["target", task["draft_role"]]
        sources = sources_for(Path(task["run_dir"]), roles)
        (run_baselines if task["kind"] == "baseline" else run_main)(task, "cuda:0", cfg, prepared, sources)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("dry-run", "run", "status", "summarize", "worker"))
    parser.add_argument("--model-root", type=Path, default=DEFAULT_MODEL_ROOT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--benchmarks", nargs="+", choices=("wikitection", "newstection", "arxivtection"), default=["wikitection", "newstection", "arxivtection"])
    parser.add_argument("--epochs", nargs="+", type=int, choices=(1, 3), default=[1, 3])
    parser.add_argument("--seeds", nargs="+", type=int, choices=(1919, 1949, 1978), default=[1919, 1949, 1978])
    parser.add_argument("--gpus", nargs="+", type=int, default=[0])
    parser.add_argument("--starts", nargs="+", default=["suffix64"])
    parser.add_argument("--rounds-per-start", type=int, default=32)
    parser.add_argument("--audit-seed", type=int, default=20260914)
    parser.add_argument("--detector-epochs", type=int, default=30)
    parser.add_argument("--reuse-robustness-reference", action="store_true",
                        help="reuse one greedy reference across WS/RS/BT; requires a separate --output-root")
    parser.add_argument("--task-file", type=Path)
    args = parser.parse_args()
    if args.command == "worker":
        if args.task_file is None:
            parser.error("worker requires --task-file")
        execute_worker(json.loads(args.task_file.read_text()))
        return
    if args.rounds_per_start < 1 or args.detector_epochs < 1:
        parser.error("round and detector epoch budgets must be positive")
    if args.reuse_robustness_reference and args.output_root.resolve() == DEFAULT_OUTPUT_ROOT.resolve():
        parser.error("--reuse-robustness-reference requires a separate --output-root")
    for values in (args.benchmarks, args.epochs, args.seeds, args.starts):
        if len(set(values)) != len(values):
            parser.error("duplicate matrix entries are not allowed")
    from experiments.sd_membership_sft.protocols.sd_protocol import resolve_starts
    resolve_starts(2048, args.starts)
    settings = dict(starts=args.starts, rounds_per_start=args.rounds_per_start, audit_seed=args.audit_seed,
                    detector_epochs=args.detector_epochs, baseline=BASELINE_DEFAULTS)
    if args.reuse_robustness_reference:
        settings["reuse_robustness_reference"] = True
    tasks = make_tasks(args.model_root.resolve(), args.output_root.resolve(), args.benchmarks, args.epochs, args.seeds, settings)
    if args.command in ("dry-run", "status"):
        states = {task["id"]: inspect_task(task) for task in tasks}
        print(json.dumps({"audit_configurations": len(tasks) // 5 * 2, "worker_tasks": len(tasks),
                          "expected_method_rows": len(tasks) // 5 * 2 * len(ALL_METHODS),
                          "settings": settings, "states": states}, indent=2))
    elif args.command == "summarize":
        result = summarize(tasks, args.output_root)
        print(json.dumps({key: result[key] for key in ("complete", "expected_rows", "completed_rows", "errors")}))
        if not result["complete"]:
            raise SystemExit(2)
    else:
        completed = run_tasks(tasks, args.output_root, args.gpus)
        result = summarize(tasks, args.output_root)
        if not completed or not result["complete"]:
            raise SystemExit(2)


if __name__ == "__main__":
    main()
