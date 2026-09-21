"""Legacy module alias for experiments.sd_membership_sft.datasets.data."""
from importlib import import_module
import sys
_implementation = import_module("experiments.sd_membership_sft.datasets.data")
if __name__ == "__main__":
    _implementation.main()
else:
    sys.modules[__name__] = _implementation
