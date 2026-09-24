"""DP CLI for registered model pairs, plus the plain-pair training implementation."""
from __future__ import annotations

import argparse
from dataclasses import replace
import gc
from importlib.metadata import version
import json
from pathlib import Path

from experiments.shared.core.audit_runtime import ROOT, _write_json
from experiments.shared.core.deployment_archive import sha256_file
from experiments.shared.training.generalization import load_run_config
from experiments.shared.audit.artifacts import digest
from experiments.dp_defense.accounting import make_plan, pair_budgets
from experiments.dp_defense.artifacts import code_sources, owned_run, read_stage, save_stage, stage_key, ROLES


def prepare_request(reference: Path, output: Path, epsilon: float, clip: float, gpu: int):
    reference, output = reference.resolve(), output.resolve()
    if reference == output or reference in output.parents or output in reference.parents:
        raise ValueError("DP output must be separate from the reference experiment")
    artifact = json.loads((reference / "results.json").read_text())
    cfg = load_run_config(reference)
    from experiments.shared.models.registry import identify_pair
    spec = identify_pair(artifact)
    if (cfg.trainer != "full" or spec.adapter != "plain"
            or not cfg.target_revision or not cfg.draft_revision
            or cfg.seed != cfg.data_seed or artifact["material_passport"]["status"] != "COMPLETED"):
        raise ValueError("reference must be a complete pinned full-parameter plain model-pair condition")
    if (cfg.n_per_class, cfg.n_aux, cfg.n_audit_aux) != (2000, 2000, 600):
        raise ValueError("reference must use the four-role 2000/2000/2000/600 contract")
    if cfg.distill_steps <= 0 or cfg.target_lr <= 0 or cfg.draft_lr <= 0:
        raise ValueError("positive training budgets required")
    manifest = Path(artifact["data"]["shared_split_manifest"])
    manifest = manifest if manifest.is_absolute() else ROOT / manifest
    if sha256_file(manifest) != artifact["data"]["shared_split_sha256"]:
        raise ValueError("reference split changed")
    cfg = replace(cfg, output_dir=output, gpu=gpu, run_auxiliary_draft=True,
                  run_member_draft=True, save_adapters=True)
    plans = {
        role: make_plan(epsilon=epsilon, max_grad_norm=clip, population=cfg.n_per_class,
                       expected_batch_size=batch * accumulation, epochs=cfg.target_epochs)
        for role, batch, accumulation in (
            ("target", cfg.target_batch_size, cfg.target_grad_accum),
            ("draft_member_sft", cfg.draft_batch_size, cfg.draft_grad_accum),
        )
    }
    source_files = [reference / "results.json", manifest, manifest.with_suffix(".audit.json")]
    request = dict(
        schema="sd_mia_dp_request_v1", model_pair=spec.name,
        reference_run=str(reference), config=cfg.as_dict(),
        plans={role: plan.as_dict() for role, plan in plans.items()},
        sources=code_sources() + [{"path": str(p), "sha256": sha256_file(p)} for p in source_files],
        environment={name: version(name) for name in ("torch", "transformers", "opacus", "numpy", "scipy", "bitsandbytes")},
        scope="models_and_protocol_outputs_only; trusted experiment metadata excluded",
    )
    return cfg, artifact, manifest, plans, request


def run(reference, output, epsilon, clip, gpu):
    import torch
    from experiments.shared.data.data import records_metadata
    from experiments.shared.drafts.plain import _load_condition_split
    from experiments.shared.training.training import load_causal_lm, load_tokenizer, distill_on_auxiliary, set_seed
    from experiments.dp_defense.training import dp_sft_train

    cfg, reference_artifact, manifest, plans, request = prepare_request(reference, output, epsilon, clip, gpu)
    with owned_run(output, request) as output:
        if (output / "results.json").exists():
            from experiments.dp_defense.artifacts import verify_run
            verify_run(output)
            print(json.dumps({"complete": str(output), "reused": True}))
            return
        if not torch.cuda.is_available():
            raise RuntimeError("DP full-model training requires a visible CUDA GPU")
        device = torch.device(f"cuda:{gpu}")
        torch.cuda.set_device(device)
        set_seed(cfg.seed)  # Public initialization/dropout only; never sampling/noise.
        tokenizer = load_tokenizer(cfg.draft_model, cfg.draft_revision, local_files_only=True)
        if tokenizer.pad_token_id is None:
            tokenizer.pad_token = tokenizer.eos_token
        target_tokenizer = load_tokenizer(cfg.target_model, cfg.target_revision, local_files_only=True)
        if tokenizer.get_vocab() != target_tokenizer.get_vocab():
            raise ValueError("target/draft token IDs differ")
        del target_tokenizer
        members, nonmembers, auxiliary, audit_auxiliary, metadata = _load_condition_split(
            cfg, argparse.Namespace(split_manifest=manifest), tokenizer, ROOT)
        records = dict(members=members, nonmembers=nonmembers, auxiliary=auxiliary,
                       audit_auxiliary=audit_auxiliary)
        record_metadata = {role: records_metadata(values) for role, values in records.items()}
        if record_metadata != reference_artifact["records"]:
            raise ValueError("DP data reconstruction differs from the reference experiment")
        stages = {}
        for role in ROLES:
            teacher_sha = stages["target"]["checkpoint_sha256"] if role == "draft_auxiliary_distilled" or (request.get("head_pair") and role == "draft_member_sft") else None
            key = stage_key(request, role, teacher_sha)
            cached = read_stage(output, role, key)
            if cached is not None:
                stages[role] = cached
                print(json.dumps({"stage": role, "reused": True}), flush=True)
                continue
            model_id = cfg.target_model if role == "target" else cfg.draft_model
            revision = cfg.target_revision if role == "target" else cfg.draft_revision
            model = load_causal_lm(model_id, device, revision=revision, local_files_only=True,
                                   attn_implementation="sdpa")
            if role == "draft_auxiliary_distilled":
                teacher = load_causal_lm(str(output / "checkpoints" / "target"), device,
                                        local_files_only=True, attn_implementation="sdpa")
                distill_on_auxiliary(model, teacher, auxiliary, tokenizer, device, cfg.distill_steps,
                                     cfg.draft_batch_size, cfg.draft_grad_accum, cfg.draft_lr,
                                     cfg.distill_temperature, cfg.seed, optimizer_name=cfg.optimizer)
                del teacher
                privacy = dict(mechanism="postprocessing", teacher_sha256=teacher_sha,
                               member_data_access=False, additional_member_epsilon=0., additional_member_delta=0.)
            else:
                def progress(step, total):
                    if step == total or step % 10 == 0:
                        print(json.dumps({"stage": role, "optimizer_step": step, "steps": total}), flush=True)
                privacy = dp_sft_train(model, members, tokenizer, device, plans[role],
                                       lr=cfg.target_lr if role == "target" else cfg.draft_lr,
                                       optimizer_name=cfg.optimizer, progress=progress)
            stages[role] = save_stage(output, role, key, model, tokenizer, privacy)
            del model
            gc.collect()
            torch.cuda.empty_cache()
        result = dict(
            material_passport={**reference_artifact["material_passport"], "status": "COMPLETED",
                               "experiment_id": f"dp-{cfg.benchmark}-{cfg.seed}-epoch{cfg.target_epochs}-epsilon{epsilon:g}",
                               "verification_status": "COMPLETED_DP_CONTROLLED_SFT_CONDITION"},
            config=cfg.as_dict(), data=metadata, records=record_metadata,
            training={"private_losses": "not logged or released", "models_full_parameter": True},
            privacy={"schema": "sd_mia_dp_condition_v1", "request_key": digest(request),
                     "stages": stages, "pairs": pair_budgets(stages["target"]["privacy"], stages["draft_member_sft"]["privacy"]),
                     "scope": request["scope"], "reference_run": str(reference.resolve()),
                     "release_warning": "records, split manifests and audit labels are trusted experiment metadata, not DP releases"},
        )
        _write_json(output / "results.json", result)
        print(json.dumps({"complete": str(output), "pairs": result["privacy"]["pairs"]}), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("dry-run", "run", "status"))
    parser.add_argument("--reference-run", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--epsilon", type=float, choices=(1., 4., 8.), required=True)
    parser.add_argument("--max-grad-norm", type=float, default=1.)
    parser.add_argument("--gpu", type=int, default=0)
    args = parser.parse_args()
    if args.gpu < 0:
        parser.error("GPU index must be nonnegative")
    from .api import _trainer
    trainer = _trainer(args.reference_run)
    if args.command == "run":
        trainer.run(args.reference_run, args.output_dir, args.epsilon, args.max_grad_norm, args.gpu)
        return
    *_, plans, request = trainer.prepare_request(args.reference_run, args.output_dir,
                                             args.epsilon, args.max_grad_norm, args.gpu)
    if args.command == "dry-run":
        print(json.dumps({"request": request, "pairs": pair_budgets(*(p.as_dict() for p in plans.values()))}, indent=2))
    else:
        stages = {}
        manifest = args.output_dir / "DP_REQUEST.json"
        if manifest.exists() and json.loads(manifest.read_text()) != request:
            raise ValueError("DP request changed")
        for role in ROLES:
            teacher = stages.get("target", {}).get("checkpoint_sha256") if role == "draft_auxiliary_distilled" or (request.get("head_pair") and role == "draft_member_sft") else None
            stage = read_stage(args.output_dir, role, stage_key(request, role, teacher))
            stages[role] = stage or {}
        print(json.dumps({"stages": {k: "complete" if v else "pending" for k, v in stages.items()}}, indent=2))


if __name__ == "__main__":
    main()
