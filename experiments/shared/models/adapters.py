"""Draft-family extension point for loading frozen deployment pairs.

An adapter supplies checkpoint roles, a draft loader and a protocol factory.
New families implement this interface once; audit scheduling, detector fitting,
DP comparisons and reporting do not branch on individual model names.
"""
from dataclasses import dataclass
from typing import Callable


@dataclass(frozen=True)
class DraftFamily:
    name: str
    roles: tuple[str, ...]
    checkpoint_directory: str
    load_draft: Callable
    protocol_factory: Callable
    shared_tokenizer: bool = False

    @property
    def uses_head(self):
        return self.checkpoint_directory == 'heads'


DRAFT_FAMILIES: dict[str, DraftFamily] = {}


def register_family(family: DraftFamily):
    if not family.name or family.name in DRAFT_FAMILIES:
        raise ValueError(f'draft family already registered or unnamed: {family.name}')
    if (family.checkpoint_directory not in ('checkpoints', 'heads') or not family.roles
            or len(set(family.roles)) != len(family.roles)
            or any(not role or '/' in role or role in ('.', '..') for role in family.roles)):
        raise ValueError('invalid draft-family checkpoint roles')
    DRAFT_FAMILIES[family.name] = family


def family_for(name):
    try:
        return DRAFT_FAMILIES[name]
    except KeyError:
        raise ValueError(f'unsupported draft family: {name}') from None


def _plain(run_dir, cfg, draft_path, target_path, role, device):
    from experiments.shared.training.generalization import load_draft_model
    return load_draft_model(run_dir, cfg.draft_model, role, device, attn_implementation='sdpa')


def _eagle(run_dir, cfg, draft_path, target_path, role, device):
    from experiments.shared.drafts.heads import load_eagle3_speculator
    return load_eagle3_speculator(str(draft_path), device)


def _mtp(run_dir, cfg, draft_path, target_path, role, device):
    from experiments.shared.drafts.heads import load_mtp_speculator
    return load_mtp_speculator(draft_path, device, verifier_checkpoint=target_path)


def _protocol(kind):
    def create(target, draft, device):
        from experiments.shared.protocols.sd_protocol import FrozenAdapter
        return FrozenAdapter(target, draft, kind, device)
    return create


register_family(DraftFamily('plain', ('draft_auxiliary_distilled', 'draft_member_sft'),
                            'checkpoints', _plain, _protocol('plain'), shared_tokenizer=True))
register_family(DraftFamily('eagle3', ('auxiliary_head', 'member_head'),
                            'heads', _eagle, _protocol('eagle3')))
register_family(DraftFamily('mtp', ('auxiliary_head', 'member_head'),
                            'heads', _mtp, _protocol('mtp')))
