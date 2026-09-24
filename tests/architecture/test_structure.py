"""Guard dependency direction and the public extension points after relocation."""
import ast
from dataclasses import replace
import importlib
from importlib.util import resolve_name
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from experiments.paths import ROOT
from experiments.shared.models.adapters import DRAFT_FAMILIES, DraftFamily, register_family
from experiments.shared.models.registry import MODEL_PAIRS, ModelPair


def imports(path):
    package = '.'.join(path.relative_to(ROOT).parent.parts)
    for node in ast.walk(ast.parse(path.read_text())):
        if isinstance(node, ast.ImportFrom):
            module = resolve_name('.' * node.level + (node.module or ''), package) if node.level else node.module
            yield node.lineno, module
            for name in node.names:
                yield node.lineno, f'{module}.{name.name}'
        elif isinstance(node, ast.Import):
            for name in node.names:
                yield node.lineno, name.name


def test_shared_implementations_do_not_import_experiment_entry_points():
    forbidden = ('experiments.sd_membership_sft', 'experiments.cross_model_audit', 'experiments.dp_defense')
    failures = [(str(path.relative_to(ROOT)), line, name)
                for path in (ROOT / 'experiments/shared').rglob('*.py')
                for line, name in imports(path) if name and name.startswith(forbidden)]
    assert failures == []


def test_internal_code_uses_canonical_imports_and_no_tests_live_under_experiments():
    aliases = json.loads((ROOT / 'experiments/MODULE_ALIASES.json').read_text())
    failures = [(str(path.relative_to(ROOT)), line, name)
                for path in (ROOT / 'experiments').rglob('*.py') if 'archive' not in path.parts
                for line, name in imports(path) if name in aliases]
    assert failures == []
    assert not list((ROOT / 'experiments').rglob('test_*.py'))


def test_legacy_aliases_have_canonical_identity_without_search_path_modification():
    for old, new in (
        ('experiments.sd_membership_sft.data', 'experiments.shared.data.data'),
        ('experiments.sd_membership_sft.datasets.splits', 'experiments.shared.data.splits'),
        ('experiments.cross_model_audit.models', 'experiments.shared.models.loading'),
        ('experiments.sd_membership_sft.matrix_costs', 'experiments.shared.audit.costs'),
    ):
        assert importlib.import_module(old) is importlib.import_module(new)
    package = importlib.import_module('experiments.sd_membership_sft')
    assert list(package.__path__) == [str(ROOT / 'experiments/sd_membership_sft')]


def test_new_model_pair_uses_existing_scheduler_and_new_draft_family(tmp_path, monkeypatch):
    from experiments.cross_model_audit import engine
    from experiments.shared.models import loading

    family = DraftFamily('toy_extension', ('draft_auxiliary_distilled', 'draft_member_sft'),
                         'checkpoints', lambda *args: 'loaded draft',
                         lambda target, draft, device: SimpleNamespace(target=target, draft=draft),
                         shared_tokenizer=False)
    # monkeypatch removes the registered family after exercising the public seam.
    monkeypatch.setattr('experiments.shared.models.adapters.DRAFT_FAMILIES', dict(DRAFT_FAMILIES))
    register_family(family)
    spec = ModelPair('toy_base', family.name, 'vendor/base', 'vendor/draft', 'base-revision', 'draft-revision')
    monkeypatch.setitem(MODEL_PAIRS, spec.name, spec)
    tasks = engine.make_tasks(tmp_path / 'models', tmp_path / 'audit', ['wikitection'], [1], [1919], {}, spec.name)
    assert len(tasks) == 3
    assert {t.get('draft_role') for t in tasks if t['kind'] == 'main'} == set(spec.roles)
    run = Path(tasks[1]['run_dir'])
    for role in ('target', *spec.roles):
        (run / 'checkpoints' / role).mkdir(parents=True)
    monkeypatch.setattr(loading, 'load_run_config', lambda _: SimpleNamespace(target_model=spec.target))
    monkeypatch.setattr(loading, 'load_finetuned_model', lambda *args, **kwargs: 'loaded target')
    adapter = loading.load_adapter(run, spec.adapter, 'cpu', spec.roles[1])
    assert (adapter.target, adapter.draft) == ('loaded target', 'loaded draft')
    with pytest.raises(ValueError, match='unsupported draft role'):
        loading.checkpoint_paths(run, spec.adapter, 'unknown')


def test_model_catalog_keeps_frozen_shell_recipe_revisions():
    path = ROOT / 'experiments/sd_membership_sft/scripts/model_pair_revisions.env'
    values = dict(line.split('=', 1) for line in path.read_text().splitlines() if line and not line.startswith('#'))
    for pair, prefix in [('qwen3', 'QWEN'), ('gemma4', 'GEMMA')]:
        assert MODEL_PAIRS[pair].target_revision == values[prefix + '_TARGET_REVISION']
        assert MODEL_PAIRS[pair].draft_revision == values[prefix + '_DRAFT_REVISION']


def test_shared_runtime_fingerprint_covers_new_implementations_and_catalog():
    from experiments.shared.audit.artifacts import runtime_files
    paths = set(runtime_files())
    for name in ('shared/audit/scheduler.py', 'shared/audit/reporting.py', 'shared/models/model_pairs.json',
                 'shared/models/adapters.py', 'baseline/engine.py', 'baseline/scorer.py', 'MODULE_ALIASES.json'):
        assert ROOT / 'experiments' / name in paths
