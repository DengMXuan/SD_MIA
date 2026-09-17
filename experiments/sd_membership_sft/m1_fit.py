"""Compatibility entry point; implementation moved to archive.m1_fit."""
from importlib import import_module
import sys

_implementation = import_module(".archive.m1_fit", __package__)

if __name__ == "__main__":
    _implementation.main()
else:
    sys.modules[__name__] = _implementation
