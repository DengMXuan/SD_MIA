"""Compatibility entry point; implementation moved to archive.interpretable_scale_gate."""
from importlib import import_module
import sys

_implementation = import_module(".archive.interpretable_scale_gate", __package__)

if __name__ == "__main__":
    _implementation.main()
else:
    sys.modules[__name__] = _implementation
