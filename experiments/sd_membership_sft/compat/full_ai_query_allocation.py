"""Compatibility entry point; implementation moved to archive.full_ai_query_allocation."""
from importlib import import_module
import sys

_implementation = import_module("experiments.sd_membership_sft.archive.full_ai_query_allocation", __package__)

if __name__ == "__main__":
    _implementation.main()
else:
    sys.modules[__name__] = _implementation
