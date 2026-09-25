"""Conservative local import closure for evaluation reproducibility.

Walk imports inside functions as well as module imports, without importing or
running GPU code. Package initializers and newly introduced dependencies count.
Checkpoint fingerprints separately include checkpoint-owned remote Python code.
"""
import ast
from importlib.util import resolve_name
from pathlib import Path
from experiments.paths import ROOT


def import_closure(root, entries):
    root = Path(root)
    pending, seen, files = list(entries), set(), set()
    while pending:
        name = pending.pop()
        if name in seen or not (name == 'experiments' or name.startswith('experiments.')):
            continue
        seen.add(name)
        path = root.joinpath(*name.split('.')).with_suffix('.py')
        if not path.is_file():
            path = root.joinpath(*name.split('.'), '__init__.py')
        if not path.is_file():
            continue  # Imported attribute, not a module.
        files.add(path)
        parts = name.split('.')
        pending.extend('.'.join(parts[:i]) for i in range(1, len(parts)))
        package = name if path.name == '__init__.py' else name.rpartition('.')[0]
        for node in ast.walk(ast.parse(path.read_text(), filename=str(path))):
            if isinstance(node, ast.Import):
                pending.extend(alias.name for alias in node.names)
            elif isinstance(node, ast.ImportFrom):
                module = node.module or ''
                if node.level:
                    module = resolve_name('.' * node.level + module, package)
                pending.append(module)
                pending.extend(module + '.' + alias.name for alias in node.names if alias.name != '*')
    return sorted(files)


def runtime_files():
    return sorted([*import_closure(ROOT, ['experiments.model_quality.cli']),
                   ROOT / 'experiments/MODULE_ALIASES.json',
                   ROOT / 'experiments/shared/models/model_pairs.json'])
