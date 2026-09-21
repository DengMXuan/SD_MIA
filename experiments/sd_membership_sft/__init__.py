"""Controlled SD experiments. Canonical modules live in themed subpackages.

The compat search path preserves old imports/CLI names without duplicate logic.
"""
from pathlib import Path
__path__.append(str(Path(__file__).parent / "compat"))
