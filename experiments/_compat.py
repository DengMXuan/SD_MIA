"""Explicit, lazy aliases for historical imports and ``python -m`` entry points.

Only names in MODULE_ALIASES.json are intercepted. Canonical modules retain
normal filesystem discovery; no package search paths or implementations are
copied. Runtime code must import canonical names.
"""
import importlib
from importlib.abc import MetaPathFinder, Loader
from importlib.util import spec_from_loader
import json
from pathlib import Path
import sys


class AliasLoader(Loader):
    def __init__(self, name, target):
        self.name, self.target = name, target

    def create_module(self, spec):
        return None

    def exec_module(self, module):
        sys.modules[self.name] = importlib.import_module(self.target)

    def is_package(self, fullname):
        return importlib.util.find_spec(self.target).submodule_search_locations is not None

    def get_code(self, fullname):
        # runpy uses get_code for historical command-line entry points.
        source = (f'from importlib import import_module\n'
                  f'_implementation = import_module({self.target!r})\n'
                  f'if __name__ == "__main__":\n    _implementation.main()\n')
        return compile(source, f'<legacy entry {fullname}>', 'exec')


class AliasFinder(MetaPathFinder):
    def __init__(self):
        self.aliases = json.loads(Path(__file__).with_name('MODULE_ALIASES.json').read_text())

    def find_spec(self, fullname, path=None, target=None):
        destination = self.aliases.get(fullname)
        if destination is None:
            return None
        spec = importlib.util.find_spec(destination)
        return spec_from_loader(fullname, AliasLoader(fullname, destination), origin=spec.origin,
                                is_package=spec.submodule_search_locations is not None)


def install():
    if not any(isinstance(finder, AliasFinder) for finder in sys.meta_path):
        sys.meta_path.insert(0, AliasFinder())
