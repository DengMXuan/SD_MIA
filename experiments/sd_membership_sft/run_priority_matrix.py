"""Compatibility entry point; implementation moved to archive.run_priority_matrix."""
from importlib import import_module
import sys

_implementation = import_module(".archive.run_priority_matrix", __package__)

if __name__ == "__main__":
    _implementation.main()
else:
    sys.modules[__name__] = _implementation
