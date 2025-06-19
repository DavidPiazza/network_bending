"""Top-level package for network_bending."""

__all__ = [
    "NODE_CLASS_MAPPINGS",
    "NODE_DISPLAY_NAME_MAPPINGS",
    "WEB_DIRECTORY",
]

__author__ = """David Piazza"""
__email__ = "david.piazza@umontreal.ca"
__version__ = "0.0.1"

from .src.network_bending.nodes import NODE_CLASS_MAPPINGS
from .src.network_bending.nodes import NODE_DISPLAY_NAME_MAPPINGS
from .src.network_bending.nodes import WEB_DIRECTORY


