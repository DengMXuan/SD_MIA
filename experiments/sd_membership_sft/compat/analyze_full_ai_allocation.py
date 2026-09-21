"""Compatibility entry point; implementation moved to archive.analyze_full_ai_allocation."""
from importlib import import_module
import sys

_implementation = import_module("experiments.sd_membership_sft.archive.analyze_full_ai_allocation", __package__)

if __name__ == "__main__":
    _implementation.main()
else:
    sys.modules[__name__] = _implementation
