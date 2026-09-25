"""Plan, run, inspect and summarize the 36-configuration Qwen audit matrix."""
from __future__ import annotations

from experiments.shared.audit.config import audit_settings, condition_settings
import argparse
import fcntl
import json
from pathlib import Path
import subprocess

from safetensors import SafetensorError

from experiments.shared.audit import scheduler, reporting
from experiments.shared.audit.scheduler import wait_worker
from experiments.baseline import METHODS
from experiments.shared.core.audit_runtime import ROOT, _write_json
from experiments.shared.core.deployment_archive import sha256_file
from experiments.shared.audit.artifacts import read_result, sources_for
from experiments.shared.audit.baselines import BASELINE_DEFAULTS
from experiments.shared.audit.main import MAIN_METHODS

ROLES = ("draft_auxiliary_distilled", "draft_member_sft")
ALL_METHODS = (*MAIN_METHODS["fixed"], *METHODS)
from experiments.paths import QWEN_MODELS as DEFAULT_MODEL_ROOT, QWEN_AUDIT as LEGACY_OUTPUT_ROOT, QWEN_CURRENT_AUDIT, audit_executions, audit_reports

DEFAULT_OUTPUT_ROOT = QWEN_CURRENT_AUDIT


def make_tasks(model_root, output_root, benchmarks, epochs, seeds, settings):
    tasks = []
    for benchmark in benchmarks:
        for epoch in epochs:
            for seed in seeds:
                condition = dict(benchmark=benchmark, epoch=epoch, condition_seed=seed)
                key = f"{benchmark}/epoch{epoch}/seed{seed}"
                base = dict(run_dir=str((model_root / key).resolve()), condition=condition, settings=condition_settings(settings, seed))
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
    return scheduler.inspect_task(task, ready=ready, read_result=read_result,
                                  check_sources=check_sources)



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





def run_tasks(tasks, output_root, gpus):
    return scheduler.run_tasks(tasks, output_root, gpus, inspect_task=inspect_task,
                               check_gpus=check_gpus, worker_module='experiments.sd_membership_sft.audit.qwen_audit_matrix', wait=wait_worker)



def summarize(tasks, output_root, *, execution_root=None, methods=None):
    return reporting.summarize(tasks, output_root, methods=ALL_METHODS if methods is None else methods,
                               baseline_methods=METHODS, describe=lambda task: ({}, ROLES),
                               title="Qwen audit matrix", read_result=read_result,
                               execution_root=execution_root)



def execute_worker(task):
    if task["kind"] == "main" and task.get("protocol") not in MAIN_METHODS:
        raise ValueError("unsupported audit protocol")
    import torch
    from experiments.shared.models.loading import prepare_records
    from experiments.shared.audit.baselines import run_baselines
    from experiments.shared.audit.main import run_main

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
    parser.add_argument("--starts", nargs="+", default=["suffix64"], help="legacy request identity only")
    parser.add_argument("--rounds-per-start", type=int, default=32, help="legacy request identity only")
    parser.add_argument("--audit-seed", type=int, default=None, help="must match the condition seed; defaults to it")
    parser.add_argument("--detector-epochs", type=int, default=30)
    parser.add_argument("--task-file", type=Path)
    args = parser.parse_args()
    if args.command == "worker":
        if args.task_file is None:
            parser.error("worker requires --task-file")
        execute_worker(json.loads(args.task_file.read_text()))
        return
    if args.rounds_per_start < 1 or args.detector_epochs < 1:
        parser.error("round and detector epoch budgets must be positive")
    if args.output_root.resolve().is_relative_to(LEGACY_OUTPUT_ROOT.resolve()):
        parser.error("the historical audit output root is read-only; choose a separate --output-root")
    for values in (args.benchmarks, args.epochs, args.seeds, args.starts):
        if len(set(values)) != len(values):
            parser.error("duplicate matrix entries are not allowed")
    settings = audit_settings(audit_seed=args.audit_seed, detector_epochs=args.detector_epochs, starts=args.starts, rounds_per_start=args.rounds_per_start)
    tasks = make_tasks(args.model_root.resolve(), args.output_root.resolve(), args.benchmarks, args.epochs, args.seeds, settings)
    if args.command in ("dry-run", "status"):
        states = {task["id"]: inspect_task(task) for task in tasks}
        print(json.dumps({"audit_configurations": len(tasks) // 3 * 2, "worker_tasks": len(tasks),
                          "expected_method_rows": len(tasks) // 3 * 2 * len(ALL_METHODS),
                          "settings": settings, "states": states}, indent=2))
    elif args.command == "summarize":
        result = summarize(tasks, audit_reports(args.output_root), execution_root=audit_executions(args.output_root))
        print(json.dumps({key: result[key] for key in ("complete", "expected_rows", "completed_rows", "errors")}))
        if not result["complete"]:
            raise SystemExit(2)
    else:
        completed = run_tasks(tasks, args.output_root, args.gpus)
        result = summarize(tasks, audit_reports(args.output_root), execution_root=audit_executions(args.output_root))
        if not completed or not result["complete"]:
            raise SystemExit(2)


if __name__ == "__main__":
    main()
