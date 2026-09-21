"""Compatibility entry point; implementation moved to archive.analyze_dynamic_marginal_query."""
from importlib import import_module
import sys

_implementation = import_module("experiments.sd_membership_sft.archive.analyze_dynamic_marginal_query", __package__)

if __name__ == "__main__":
    _implementation.main()
else:
    sys.modules[__name__] = _implementation
