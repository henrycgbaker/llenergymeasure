"""Single source of truth for the package version string.

This module MUST NOT import anything from the package -- it is imported
by modules at every layer to avoid pulling in the full __init__.py
import chain.
"""

__version__: str = "0.9.0"
