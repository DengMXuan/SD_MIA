"""Compatibility entry point; implementation moved to archive.full_delta_mia."""
from importlib import import_module
import sys

_implementation = import_module(".archive.full_delta_mia", __package__)

if __name__ == "__main__":
    _implementation.main()
else:
    sys.modules[__name__] = _implementation
