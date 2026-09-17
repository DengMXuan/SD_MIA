"""Compatibility import; implementation moved to archive.m2_models."""
from importlib import import_module
import sys

sys.modules[__name__] = import_module(".archive.m2_models", __package__)
