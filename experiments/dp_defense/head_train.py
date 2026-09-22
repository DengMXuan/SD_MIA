"""DP full targets and frozen-target EAGLE-3/MTP head adaptation."""
from __future__ import annotations

from dataclasses import replace
import gc
from importlib.metadata import version
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from experiments.sd_membership_sft.audit_runtime import _write_json
from experiments.sd_membership_sft.data import collate_sft, make_sft_example
from experiments.sd_membership_sft.deployment_archive import sha256_file, checkpoint_fingerprint
from experiments.sd_membership_sft.drafts.common import PAIR_MODELS, cached_snapshot
from experiments.sd_membership_sft.generalization import load_run_config
from .head_contract import inspect_head_run, validate_checkpoint
from experiments.sd_membership_sft.matrix_artifacts import digest
from experiments.sd_membership_sft.training import load_causal_lm, _autocast, set_seed
from .accounting import make_plan, pair_budgets
from .artifacts import ROLES, code_sources, owned_run, read_stage, save_stage, stage_key, verify_run
from .training import dp_sft_train


def prepare_request(reference, output, epsilon, clip, gpu):
    reference, output = reference.resolve(), output.resolve()
    if reference == output or reference in output.parents or output in reference.parents:
        raise ValueError("DP output must be separate from reference models")
    details = inspect_head_run(reference)
    cfg = replace(load_run_config(reference), trainer="full", optimizer="adamw8bit", output_dir=output, gpu=gpu,
                  pool_path=details["pool"], run_auxiliary_draft=True, run_member_draft=True)
    for role in ("draft_auxiliary_distilled", "draft_member_sft"):
        marker = json.loads((details["paths"][role] / "_COMPLETE.json").read_text())
        if (marker["optimizer_updates"] != 384 or marker["effective_batch_size"] != 16
                or marker["learning_rate"] != 2e-5):
            raise ValueError("reference head budget differs from the registered 384 updates/batch16/lr2e-5")
    plans = dict(
        target=make_plan(epsilon=epsilon, max_grad_norm=clip, population=cfg.n_per_class,
                         expected_batch_size=cfg.target_batch_size * cfg.target_grad_accum, epochs=cfg.target_epochs),
        draft_member_sft=make_plan(epsilon=epsilon, max_grad_norm=clip, population=cfg.n_per_class,
                                   expected_batch_size=16, steps=384),
    )
    source_head = None
    source_fingerprint = None
    if details["kind"] == "mtp":
        marker = json.loads((details["paths"]["draft_auxiliary_distilled"] / "_COMPLETE.json").read_text())
        source_head = Path(marker["source_head"]).resolve()
        source_marker = validate_checkpoint(source_head)
        if (not source_marker.get("native_mtp_export") or source_marker.get("training_updates") != 0
                or source_marker["initialized_from_revision"] != cfg.target_revision):
            raise ValueError("MTP source must be the unadapted pinned native export")
        source_fingerprint = checkpoint_fingerprint(source_head)
    files = [reference / "results.json", details["manifest"], details["audit_path"]]
    request = dict(schema="sd_mia_dp_head_request_v1", head_pair=details["pair"], config=cfg.as_dict(),
                   reference_run=str(reference), plans={k: p.as_dict() for k, p in plans.items()},
                   source_head=str(source_head) if source_head else None, source_head_sha256=source_fingerprint,
                   environment={name: version(name) for name in ("torch", "transformers", "speculators", "opacus")},
                   sources=code_sources() + [dict(path=str(p), sha256=sha256_file(p)) for p in files],
                   scope="model_weights_and_protocol_outputs; trusted metadata excluded")
    return cfg, details, plans, request


def head_loss(kind, target, speculator, batch, device, *, member, temperature=2.):
    if kind not in ("eagle3", "mtp"):
        raise ValueError("unsupported head kind")
    if any(p.requires_grad for p in target.parameters()) or target.training:
        raise ValueError("head adaptation requires a frozen eval-mode target")
    if kind == "eagle3":
        from experiments.sd_membership_sft.drafts.eagle3 import _eagle_kd_loss
        return _eagle_kd_loss(speculator, target, batch, device, temperature)[0]
    with torch.no_grad(), _autocast(device):
        output = target(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"],
                        output_hidden_states=True, use_cache=False)
    # Native MTP API is batch-one; preserve original rowwise objective.
    losses = []
    for row in range(len(batch["input_ids"])):
        length = int(batch["attention_mask"][row].sum())
        ids = batch["input_ids"][row:row + 1, :length]
        hidden = output.hidden_states[-1][row:row + 1, :length]
        labels = batch["labels"][row:row + 1, :length]
        logits, ce, _ = speculator(input_ids=ids, hidden_states=hidden, attention_mask=None,
                                  loss_mask=labels.ne(-100), return_dict=True)
        if member:
            losses.append(ce)
        else:
            student = logits[0].float()
            teacher = output.logits[row:row + 1, 1:1 + student.shape[1]].float()
            valid = labels[:, 2:2 + student.shape[1]].ne(-100)
            losses.append(F.kl_div(F.log_softmax(student[valid] / temperature, -1),
                                  F.softmax(teacher[valid] / temperature, -1),
                                  reduction="batchmean") * temperature**2)
    return torch.stack(losses).mean()


def fit_auxiliary(model, target, records, tokenizer, device, kind, *, seed, updates=384):
    examples = [make_sft_example(r, tokenizer) for r in records]
    rng = np.random.default_rng(seed)
    parameters = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(parameters, lr=2e-5)
    model.train()
    for step in range(updates):
        optimizer.zero_grad(set_to_none=True)
        for _ in range(8):
            indices = rng.integers(0, len(examples), size=2)
            batch = collate_sft([examples[int(i)] for i in indices], tokenizer.pad_token_id)
            batch = {k: v.to(device) for k, v in batch.items()}
            with _autocast(device):
                loss = head_loss(kind, target, model, batch, device, member=False)
            (loss / 8).backward()
        torch.nn.utils.clip_grad_norm_(parameters, 1.)
        optimizer.step()
        if (step + 1) % 32 == 0:
            print(json.dumps(dict(stage="aux_head", step=step + 1, steps=updates)), flush=True)
    optimizer.zero_grad(set_to_none=True)
    model.eval()


def load_initial_head(pair, source_head, target_checkpoint, device):
    from experiments.sd_membership_sft.drafts.heads import load_eagle3_speculator, load_mtp_speculator
    spec = PAIR_MODELS[pair]
    if spec["kind"] == "eagle3":
        return load_eagle3_speculator(spec["speculator"], device, spec["speculator_revision"])
    return load_mtp_speculator(source_head, device, verifier_checkpoint=target_checkpoint)


def run(reference, output, epsilon, clip, gpu):
    from experiments.sd_membership_sft.drafts.common import tokenizer_for, tokenizer_source_for
    from experiments.sd_membership_sft.splits import build_controlled_split_from_shared_manifest
    cfg, details, plans, request = prepare_request(reference, output, epsilon, clip, gpu)
    with owned_run(output, request) as output:
        if (output / "results.json").exists():
            verify_run(output)
            print(json.dumps(dict(complete=str(output), reused=True)))
            return
        if not torch.cuda.is_available():
            raise RuntimeError("DP head training requires a visible CUDA GPU")
        device = torch.device(f"cuda:{gpu}")
        torch.cuda.set_device(device)
        set_seed(cfg.seed)
        pair, kind = details["pair"], details["kind"]
        spec = PAIR_MODELS[pair]
        tokenizer = tokenizer_for(pair)
        split = build_controlled_split_from_shared_manifest(cfg.benchmark, details["pool"], tokenizer,
                    details["manifest"], tokenizer_source_for(pair))
        if any(split.metadata[k] != details["data"][k] for k in
               ("shared_split_sha256", "pool_sha256", "counts", "split_seed", "tokenizer_source")):
            raise ValueError("reference split changed")
        target_checkpoint = output / "checkpoints/target"
        stages, manifests = {}, {}
        for role in ROLES:
            teacher_sha = stages["target"]["checkpoint_sha256"] if role != "target" else None
            key = stage_key(request, role, teacher_sha)
            cached = read_stage(output, role, key)
            stage_name = "target" if role == "target" else ("aux_head" if role == ROLES[1] else "member_head")
            if cached is None:
                if role == "target":
                    model = load_causal_lm(cfg.target_model, device, revision=cfg.target_revision,
                                          local_files_only=True, attn_implementation="sdpa")
                    privacy = dp_sft_train(model, split.members, tokenizer, device, plans[role],
                                          lr=cfg.target_lr, optimizer_name="adamw8bit",
                                          progress=lambda step, total: print(json.dumps(dict(stage="target", step=step, steps=total)), flush=True))
                    marker = dict(stage="target", pair=pair, base_model=cfg.target_model, base_revision=cfg.target_revision,
                                  seed=cfg.seed, data_seed=cfg.data_seed, epochs=cfg.target_epochs, full_parameter_sft=True)
                    implementation = None
                else:
                    target = load_causal_lm(str(target_checkpoint), device, local_files_only=True, attn_implementation="sdpa")
                    target.requires_grad_(False)
                    target.eval()
                    model = load_initial_head(pair, request["source_head"], target_checkpoint, device)
                    if kind == "mtp" and model.config.num_speculative_steps != 1:
                        raise ValueError("DP head adaptation requires depth-one native MTP")
                    member = role == "draft_member_sft"
                    if member:
                        privacy = dp_sft_train(model, split.members, tokenizer, device, plans[role], lr=2e-5,
                            optimizer_name="adamw", allow_frozen_parameters=True,
                            document_loss=lambda model, batch: head_loss(kind, target, model, batch, device, member=True),
                            progress=lambda step, total: print(json.dumps(dict(stage="member_head", step=step, steps=total)), flush=True))
                    else:
                        fit_auxiliary(model, target, split.draft_auxiliary, tokenizer, device, kind, seed=cfg.seed)
                        privacy = dict(mechanism="postprocessing", teacher_sha256=teacher_sha, member_data_access=False,
                                       additional_member_epsilon=0., additional_member_delta=0.)
                    if any(p.grad is not None for p in target.parameters()):
                        raise ValueError("head training unexpectedly populated target gradients")
                    del target
                    marker = dict(stage=stage_name, variant="member" if member else "aux",
                        objective="native-mtp-cross-entropy" if kind == "mtp" and member else "temperature-kl",
                        initialized_from=spec["speculator"], initialized_from_revision=spec["speculator_revision"],
                        target_checkpoint=str(target_checkpoint), target_frozen=True, seed=cfg.seed,
                        optimizer_updates=384, effective_batch_size=16, learning_rate=2e-5,
                        temperature=None if kind == "mtp" and member else 2.)
                    if kind == "mtp":
                        marker.update(source_head=request["source_head"], verifier_owned_weights_from_target=True)
                    implementation = cached_snapshot(spec["speculator"], spec["speculator_revision"]) / "eagle3.py" if kind == "eagle3" else None
                cached = save_stage(output, role, key, model, tokenizer, privacy, marker=marker, implementation=implementation)
                del model
                gc.collect()
                torch.cuda.empty_cache()
            stages[role] = cached
            manifests[stage_name] = dict(stage=stage_name, data=split.metadata)
        artifact = dict(config=cfg.as_dict(), protocol_track=dict(pair=pair, target_frozen_before_heads=True,
                         shared_raw_split=str(details["manifest"])), stages=manifests,
                         privacy=dict(schema="sd_mia_dp_condition_v1", request_key=digest(request), stages=stages,
                                      pairs=pair_budgets(stages["target"]["privacy"], stages["draft_member_sft"]["privacy"]),
                                      scope=request["scope"], reference_run=str(reference.resolve())))
        _write_json(output / "results.json", artifact)
        print(json.dumps(dict(complete=str(output), pairs=artifact["privacy"]["pairs"])), flush=True)
