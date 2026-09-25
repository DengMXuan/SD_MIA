"""Quality provenance follows imports and never adopts archived evaluations."""
from pathlib import Path
import pytest
from experiments.paths import ROOT
from experiments.shared.evaluation import quality


def test_quality_dependency_closure_tracks_gate_and_ignores_audit_reporting():
    from experiments.shared.evaluation.provenance import runtime_files
    files = set(runtime_files())
    for name in ('evaluation/quality.py', 'models/validation.py', 'models/precision.py',
                 'protocols/sd_protocol.py', 'training/generalization.py', 'data/splits.py'):
        assert ROOT / 'experiments/shared' / name in files
    assert ROOT / 'experiments/shared/audit/reporting.py' not in files
    assert ROOT / 'experiments/sd_membership_sft/audit/qwen_kd_epoch1.py' not in files


def test_dependency_walk_handles_relative_imports_local_imports_and_new_dependencies(tmp_path):
    from experiments.shared.evaluation.provenance import import_closure
    package = tmp_path / 'experiments'; package.mkdir()
    (package / '__init__.py').write_text('')
    (package / 'entry.py').write_text('from . import helper\ndef lazy():\n from .deeper import thing\n')
    (package / 'helper.py').write_text('import math\n')
    (package / 'deeper.py').write_text('thing = 1\n')
    (package / 'unused.py').write_text('')
    files = import_closure(tmp_path, ['experiments.entry'])
    assert {p.name for p in files} == {'__init__.py','entry.py','helper.py','deeper.py'}
    (package / 'helper.py').write_text('from . import unused\n')
    assert package / 'unused.py' in import_closure(tmp_path, ['experiments.entry'])


def test_archived_batches_cannot_be_written_through_compatibility_symlink(tmp_path):
    archived = tmp_path / 'archived'; archived.mkdir()
    (archived / 'ARCHIVED.json').write_text('{}')
    link = tmp_path / 'old'; link.symlink_to(archived, target_is_directory=True)
    with pytest.raises(ValueError, match='archived'):
        quality.validate_output(tmp_path / 'training', link / 'tasks' / 'condition')


def test_resume_ignores_unrelated_change_but_blocks_computational_change(tmp_path, monkeypatch):
    from experiments.shared.evaluation.provenance import import_closure
    package = tmp_path / 'experiments'; package.mkdir()
    (package / '__init__.py').write_text('')
    (package / 'entry.py').write_text('from . import compute\n')
    compute = package / 'compute.py'; compute.write_text('value = 1\n')
    unrelated = package / 'reporting.py'; unrelated.write_text('value = 1\n')
    files = import_closure(tmp_path, ['experiments.entry'])
    sources = dict(files=[dict(path=str(p), sha256=quality.sha256_file(p)) for p in files],
                   checkpoints=[], runtime_files=[str(p) for p in files])
    monkeypatch.setattr(quality, 'runtime_files', lambda: import_closure(tmp_path, ['experiments.entry']))
    unrelated.write_text('value = 2\n')
    quality._check_sources(sources)
    compute.write_text('value = 2\n')
    with pytest.raises(ValueError, match='source changed'):
        quality._check_sources(sources)
