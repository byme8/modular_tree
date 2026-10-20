"""Logging for the build pipeline.

Blender swallows exceptions raised inside operators and node callbacks into a
short status string, so a failure deep in the pipeline surfaces as one line with
no context. Every stage logs what it received and produced, which turns "no
branch chains found" into a trace naming the stage that actually went wrong.

Off by default. Enable from the Tree Mesher node's Diagnostics panel, or set
MTREE_LOG=1 (or MTREE_LOG=/path/to/file.log) before starting Blender.
"""

from __future__ import annotations

import logging
import os
import time
from contextlib import contextmanager

LOGGER_NAME = "mtree"
ENV_VAR = "MTREE_LOG"

_logger = logging.getLogger(LOGGER_NAME)
_configured = False


def _configure():
    global _configured
    if _configured:
        return

    _logger.setLevel(logging.DEBUG)
    _logger.propagate = False

    console = logging.StreamHandler()
    console.setFormatter(logging.Formatter("[mtree] %(message)s"))
    _logger.addHandler(console)

    destination = os.environ.get(ENV_VAR, "")
    if destination and destination not in ("1", "true", "TRUE"):
        handler = logging.FileHandler(destination, mode="w", encoding="utf-8")
        handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)-7s %(message)s"))
        _logger.addHandler(handler)

    _configured = True


def set_enabled(enabled: bool):
    if enabled:
        _configure()
    _logger.disabled = not enabled


def is_enabled() -> bool:
    return _configured and not _logger.disabled


def log(message: str, *args):
    if is_enabled():
        _logger.info(message, *args)


def warn(message: str, *args):
    if is_enabled():
        _logger.warning(message, *args)


def exception(message: str, *args):
    """Log a failure with its traceback, which Blender's status bar discards."""
    if is_enabled():
        _logger.exception(message, *args)


@contextmanager
def stage(name: str, **context):
    """Log entry, exit and duration of a pipeline stage.

    A stage that raises logs the traceback before the exception propagates, so
    the cause survives even though the caller reduces it to a status string.
    """
    if not is_enabled():
        yield _NullStage()
        return

    detail = " ".join(f"{key}={value}" for key, value in context.items())
    _logger.info("> %s %s", name, detail)

    record = _Stage(name)
    started = time.perf_counter()
    try:
        yield record
    except Exception:
        _logger.exception("! %s FAILED after %.3fs", name, time.perf_counter() - started)
        raise
    else:
        elapsed = time.perf_counter() - started
        results = " ".join(f"{key}={value}" for key, value in record.results.items())
        _logger.info("< %s %.3fs %s", name, elapsed, results)


class _NullStage:
    def record(self, **results):
        pass


class _Stage:
    def __init__(self, name):
        self.name = name
        self.results = {}

    def record(self, **results):
        self.results.update(results)


def describe_mesh(mesh) -> str:
    """Summarise a mesh the way the rig reads it, not the way Blender shows it."""
    if mesh is None:
        return "mesh=None"

    vertices = len(mesh.vertices)
    parts = [f"verts={vertices}", f"polys={len(mesh.polygons)}"]

    expected = (
        "stem_id",
        "branch_extent",
        "radius",
        "hierarchy_depth",
        "pivot_position",
        "direction",
    )
    missing = [name for name in expected if name not in mesh.attributes]
    parts.append("attrs=" + ("all" if not missing else f"MISSING:{','.join(missing)}"))

    if vertices and "stem_id" in mesh.attributes:
        import numpy as np

        values = np.zeros(vertices)
        mesh.attributes["stem_id"].data.foreach_get("value", values)
        parts.append(f"stems={len(set(values.tolist()))}")

    if mesh.shape_keys is not None:
        parts.append(f"shape_keys={len(mesh.shape_keys.key_blocks)}")

    return " ".join(parts)


def describe_graph(node) -> str:
    tree = node.id_data
    links = " ".join(f"{link.from_node.name}->{link.to_node.name}" for link in tree.links)
    return f"nodes={len(tree.nodes)} links=[{links}]"
