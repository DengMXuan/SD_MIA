"""Compatibility entry point; implementation moved to archive.aggregate_neural_adaptive_accept_only."""
from importlib import import_module
import sys

_implementation = import_module(".archive.aggregate_neural_adaptive_accept_only", __package__)

if __name__ == "__main__":
    _implementation.main()
else:
    sys.modules[__name__] = _implementation
