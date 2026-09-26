"""Selected DP deployments and one public seed per frozen condition."""
import json

DRAFT_VARIANTS = ('kd', 'member')
STAGE_FOR = dict(kd='draft_auxiliary_distilled', member='draft_member_sft')


def variants(values=None):
    values = DRAFT_VARIANTS if values is None else (values,) if isinstance(values, str) else tuple(values)
    if not values or len(set(values)) != len(values) or any(v not in DRAFT_VARIANTS for v in values):
        raise ValueError('choose distinct draft variants from kd, member')
    return tuple(v for v in DRAFT_VARIANTS if v in values)


def stage_roles(selected=None):
    return ('target', *(STAGE_FOR[v] for v in variants(selected)))


def draft_roles(is_head, selected=None):
    names = dict(kd='auxiliary_head', member='member_head') if is_head else STAGE_FOR
    return [names[v] for v in variants(selected)]


def seed_policy(config, manifest):
    """Validate the actual frozen split, not just a seed-bearing directory name."""
    seed = config['seed']
    shared = json.loads(manifest.read_text())
    if (type(seed) is not int or not 0 <= seed < 2**32 or config['data_seed'] != seed
            or shared['seed'] != seed or shared['benchmark'] != config['benchmark']):
        raise ValueError('DP condition/data/shared-split seeds and benchmark must match')
    return dict(condition_seed=seed, data_seed=seed, target_public_seed=seed,
                draft_public_seed=seed, kd_sampling_seed=seed, audit_seed=seed,
                private_randomness='independent_unpublished_os_seeded_prng_streams')
