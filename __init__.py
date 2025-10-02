"""
ComfyUI entrypoint for the network_bending custom node pack.

Exports NODE_CLASS_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS and WEB_DIRECTORY.
Handles adding the local src/ path so the packaged code under src/network_bending
is importable without installation.
"""

import os
import sys

# Ensure the local src directory is importable when running inside ComfyUI
_HERE = os.path.dirname(__file__)
_SRC_DIR = os.path.join(_HERE, "src")
if os.path.isdir(_SRC_DIR) and _SRC_DIR not in sys.path:
    sys.path.insert(0, _SRC_DIR)

# Import node mappings from the packaged module
try:
    from network_bending.nodes import (  # type: ignore
        NODE_CLASS_MAPPINGS as _NODE_CLASS_MAPPINGS,
        NODE_DISPLAY_NAME_MAPPINGS as _NODE_DISPLAY_NAME_MAPPINGS,
    )
except Exception as e:  # pragma: no cover - surface helpful error in UI
    # Provide clearer error if import fails (e.g., missing deps)
    raise RuntimeError(
        f"Failed to import network_bending nodes. Error: {e}. "
        "Ensure dependencies are installed and the 'src' folder exists."
    )


# Re-export for ComfyUI
NODE_CLASS_MAPPINGS = _NODE_CLASS_MAPPINGS
NODE_DISPLAY_NAME_MAPPINGS = _NODE_DISPLAY_NAME_MAPPINGS

# Expose web directory for frontend helpers
WEB_DIRECTORY = "./src/network_bending/js"

__all__ = [
    "NODE_CLASS_MAPPINGS",
    "NODE_DISPLAY_NAME_MAPPINGS",
    "WEB_DIRECTORY",
]

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


