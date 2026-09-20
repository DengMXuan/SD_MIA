"""Collect fixed probes or multi-start, single-proposal natural SD observations.

Usage: python -m experiments.sd_membership_sft.collect_protocol_observations --help
Head adapters are implemented but await real-model validation on new checkpoints.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict
import fcntl
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from .audit_partitions import deployment_partitions
from .audit_runtime import _write_json
from .data import SFTRecord, prompt_prefix_ids, _hash_ids
from .deployment_archive import sha256_file
from .protocol_archive import atomic_npz, save_archive
from .protocol_models import checkpoint_paths, load_adapter, prepare_records, source_contract, local_tokenizer, DRAFT_ROLES
from .matrix_costs import timed, reset_peak, peak_memory
from .sd_protocol import fixed_trace, natural_trace, resolve_starts, trajectory_seed


def _digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def protocol_prompt_ids(record, tokenizer):
    prefix = prompt_prefix_ids(record, tokenizer)
    if prefix and isinstance(prefix[0], str):
        # Transformers 5 can return BatchEncoding by default. The historical
        # helper assumes a token list; explicitly request that contract here.
        prefix = tokenizer.apply_chat_template(
            [{"role": "user", "content": record.prompt}], tokenize=True,
            add_generation_prompt=True, enable_thinking=False, return_dict=False,
        )
    if not prefix or not all(isinstance(value, (int, np.integer)) for value in prefix):
        raise ValueError("prompt tokenizer did not return a nonempty token-ID sequence")
    return list(prefix)


def collect_records(prepared, adapter, output: Path, contract: dict) -> Path:
    """Collect whole records with checksum-validated per-trajectory recovery."""
    starts = contract["starts"]
    inputs, positions, failures = [], [], []
    for record in prepared.records:
        prompt = protocol_prompt_ids(record, prepared.tokenizer)
        response = list(record.response_ids)
        try:
            resolved = resolve_starts(len(response), starts) if contract["protocol"] == "natural" else [0]
        except ValueError as error:
            failures.append({"record_id": record.record_id, "reason": str(error)})
            continue
        positions.append(resolved)
        inputs.append((prompt, response))
    if failures:
        _write_json(output / "PREFLIGHT_FAILURES.json", failures)
        raise ValueError("invalid starts; see PREFLIGHT_FAILURES.json (no records silently excluded)")
    contract = {**contract, "record_ids": prepared.record_ids.tolist(),
                "record_roles": prepared.record_roles.tolist(),
                "input_hashes": [_hash_ids(p + r) for p, r in inputs],
                "resolved_positions": positions}
    manifest = output / "COLLECTION.json"
    if manifest.exists() and json.loads(manifest.read_text()) != contract:
        raise ValueError("collection/resume configuration or source changed")
    _write_json(manifest, contract)
    stamp = _digest(contract)
    records_dir = output / "trajectories"
    records_dir.mkdir(exist_ok=True)
    features, counts, lengths, owners, start_ids, all_costs = [], [], [], [], [], []
    eos = prepared.tokenizer.eos_token_id
    eos_ids = tuple(eos) if isinstance(eos, (list, tuple)) else (() if eos is None else (int(eos),))
    for i, record in enumerate(prepared.records):
        prompt, response = inputs[i]
        for j, start in enumerate(starts):
            path = records_dir / f"{i}_{j}.npz"
            sidecar = path.with_suffix(".json")
            if path.exists() and sidecar.exists():
                meta = json.loads(sidecar.read_text())
                if meta["contract_sha256"] != stamp or meta["sha256"] != sha256_file(path):
                    raise ValueError(f"trajectory source/hash mismatch: {path}")
                with np.load(path, allow_pickle=False) as saved:
                    if set(saved.files) != {"features", "counts"}:
                        raise ValueError("unexpected fields in trajectory")
                    trace = dict(saved)
                cost = meta["cost"]
            else:
                seed = trajectory_seed(contract["seed"], record.record_id, start)
                before = asdict(adapter.cost)
                reset_peak(adapter.device)
                with timed(adapter.device) as elapsed:
                    if contract["protocol"] == "natural":
                        position = positions[i][j]
                        trace = natural_trace(
                            adapter, prompt + response[:position], rounds=contract["rounds_per_start"],
                            seed=seed, start_fraction=position / len(response), eos_ids=eos_ids,
                        )
                    else:
                        trace = fixed_trace(adapter, prompt, response, seed=seed)
                cost = {key: value for key, value in trace.items() if key not in ("features", "counts")}
                cost.update({key: value - before[key] for key, value in asdict(adapter.cost).items()})
                cost.update(record_id=record.record_id, start=start)
                cost.update(seconds=elapsed["seconds"], peak_allocated_gpu_bytes=peak_memory(adapter.device))
                atomic_npz(path, {key: trace[key] for key in ("features", "counts")})
                _write_json(sidecar, {"contract_sha256": stamp, "sha256": sha256_file(path), "cost": cost})
            features.append(trace["features"])
            counts.append(trace["counts"])
            lengths.append(len(trace["counts"]))
            owners.append(i)
            start_ids.append(j)
            all_costs.append(cost)
        print(json.dumps({"record": i + 1, "total": len(prepared.records), "starts": len(starts)}), flush=True)
    archive = output / "observations.npz"
    save_archive(archive, {
        "features": np.concatenate(features), "counts": np.concatenate(counts),
        "lengths": np.asarray(lengths, dtype=np.int64),
        "document_indices": np.asarray(owners, dtype=np.int64),
        "start_indices": np.asarray(start_ids, dtype=np.int64),
        "record_ids": prepared.record_ids, "record_roles": prepared.record_roles,
        "labels": prepared.labels,
    }, contract, all_costs)
    return archive


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("collect", "smoke"))
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--adapter", choices=("plain", "eagle3", "mtp"), default="plain")
    parser.add_argument("--draft-role", choices=DRAFT_ROLES, default=DRAFT_ROLES[0])
    parser.add_argument("--protocol", choices=("natural", "fixed"), default="natural")
    parser.add_argument("--starts", nargs="+", default=["0.5", "0.75"], help="response token fractions or suffix64")
    parser.add_argument("--rounds-per-start", type=int, default=32)
    parser.add_argument("--seed", type=int, default=20260914)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--smoke-text-file", type=Path, help="optional text for a runtime-only smoke test")
    args = parser.parse_args()
    if args.rounds_per_start < 1:
        parser.error("rounds-per-start must be positive")
    if args.command == "collect" and args.smoke_text_file is not None:
        parser.error("smoke text is only permitted for the smoke command")
    run_dir = args.run_dir.resolve()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    with (args.output_dir / ".collection.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if args.command == "collect":
            _, prepared = prepare_records(run_dir, args.adapter)
            deployment_partitions(prepared.labels, prepared.record_ids, prepared.record_roles)
        else:
            from .generalization import load_run_config
            target_path, _ = checkpoint_paths(run_dir, args.adapter, args.draft_role)
            cfg = load_run_config(run_dir)
            tokenizer = local_tokenizer(target_path, cfg.target_model, cfg.target_revision)
            text = (args.smoke_text_file.read_text() if args.smoke_text_file else
                    "A research team measured the temperature of a lake at several depths. "
                    "They recorded observations each morning and compared the readings across seasons. "
                    "The measurements showed that surface water responded quickly to changes in weather, "
                    "while deeper water changed more slowly. The researchers checked their instruments "
                    "against a reference thermometer and published the measurement procedure. " * 4)
            tokens = tokenizer.encode(text, add_special_tokens=False)
            record = SFTRecord("runtime-smoke", "synthetic", tuple(tokens), _hash_ids(tokens),
                               prompt_text="Continue the following document.", append_eos=False)
            prepared = SimpleNamespace(
                records=[record], tokenizer=tokenizer, labels=np.asarray([0]),
                record_ids=np.asarray([record.record_id]), record_roles=np.asarray(["smoke"]),
            )
        print("Fingerprinting frozen sources before collection", flush=True)
        sources = source_contract(run_dir, args.adapter, args.draft_role)
        for filename in ("sd_protocol.py", "protocol_models.py", "collect_protocol_observations.py"):
            path = Path(__file__).with_name(filename)
            sources["files"].append({"path": str(path.resolve()), "sha256": sha256_file(path)})
        contract = {
            "protocol": args.protocol, "adapter": args.adapter, "draft_role": args.draft_role,
            "timing_version": 1,
            "starts": args.starts if args.protocol == "natural" else ["fixed"],
            "rounds_per_start": args.rounds_per_start if args.protocol == "natural" else 0,
            "seed": args.seed, "temperature": 1., "top_k": None, "top_p": None,
            "proposals_per_round": 1, "sources": sources,
            "execution": "full_context_reconstruction",
            "data_contract": "four_role_600" if args.command == "collect" else "runtime_smoke_no_membership_claim",
            "head_real_model_validation": "pending" if args.adapter != "plain" else "not_applicable",
        }
        adapter = load_adapter(run_dir, args.adapter, args.device, args.draft_role)
        path = collect_records(prepared, adapter, args.output_dir, contract)
        print(json.dumps({"archive": str(path), "contract": contract["data_contract"]}), flush=True)


if __name__ == "__main__":
    main()
