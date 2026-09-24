"""Pinned model identities; shared by ordinary training, DP and audit registries."""
import json
from pathlib import Path
from .adapters import family_for

CATALOG_PATH = Path(__file__).with_name('model_pairs.json')
MODEL_CONFIGS = json.loads(CATALOG_PATH.read_text())
HEAD_PAIRS = {
    name: dict(target=spec['target'], target_revision=spec['target_revision'],
               speculator=spec['draft'], speculator_revision=spec['draft_revision'], kind=spec['adapter'])
    for name, spec in MODEL_CONFIGS.items() if family_for(spec['adapter']).uses_head
}
