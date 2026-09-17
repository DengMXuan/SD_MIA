"""Compatibility entry point; implementation moved to archive.combine_shadow_active."""
from importlib import import_module
import sys

_implementation = import_module(".archive.combine_shadow_active", __package__)

if __name__ == "__main__":
    _implementation.main()
else:
    sys.modules[__name__] = _implementation
