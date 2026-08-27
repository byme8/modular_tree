"""Root pytest configuration for MTree addon tests.

This module sets up the Python path for testing modules that don't require Blender.
We avoid importing through python_classes/__init__.py which imports bpy.
"""

import importlib.util
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
PYTHON_CLASSES = PROJECT_ROOT / "python_classes"

# Add paths for modules that don't import bpy at top level
paths_to_add = [
    str(PYTHON_CLASSES),
    str(PYTHON_CLASSES / "presets"),
    str(PYTHON_CLASSES / "pivot_painter"),
    str(PYTHON_CLASSES / "pivot_painter" / "formats"),
    str(PYTHON_CLASSES / "viewport"),
]

for path in paths_to_add:
    if path not in sys.path:
        sys.path.insert(0, path)


def load_node_base():
    """node.py imports bpy only under TYPE_CHECKING so it loads standalone, but
    its package __init__ does not — which rules out a plain import.
    """
    path = PYTHON_CLASSES / "nodes" / "base_types" / "node.py"
    spec = importlib.util.spec_from_file_location("_mtree_node_base", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module
