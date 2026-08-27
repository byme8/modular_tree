"""Foliage vertex streams and bone rig.

Mtree does not simulate wind. It bakes the per-vertex data a real-time foliage
shader needs in order to do wind itself, and optionally a bone rig that replaces
that vertex wind entirely.

``ExportResult`` lives here rather than in either exporter because both return
it and neither owns it.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class ExportResult:
    """Result of an export operation."""

    success: bool
    message: str
    files_created: list[str] | None = None


__all__ = ["ExportResult"]
