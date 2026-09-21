"""Compatibility entry point; implementation moved to archive.active_protocol_design."""
from importlib import import_module
import sys

_implementation = import_module("experiments.sd_membership_sft.archive.active_protocol_design", __package__)

if __name__ == "__main__":
    _implementation.main()
else:
    sys.modules[__name__] = _implementation
